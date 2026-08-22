from __future__ import annotations

from time import perf_counter

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from scripts.v2.phase1_rag.chunking import (
    build_page_chunks,
)
from scripts.v2.phase1_rag.ingestion import (
    ingest_document,
    read_split,
)


MODEL_NAME = "BAAI/bge-m3"
EXPECTED_DIMENSION = 1024
EXPECTED_MAX_TOKENS = 8192
DEVICE = "cuda"


def gibibytes(number_of_bytes: int) -> float:
    return number_of_bytes / 1024**3


def encode_and_report(
    model: SentenceTransformer,
    chunk: dict,
    label: str,
) -> None:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start_time = perf_counter()

    embedding = model.encode(
        [chunk["text"]],
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    torch.cuda.synchronize()
    elapsed = perf_counter() - start_time

    vector = embedding[0]
    vector_norm = float(np.linalg.norm(vector))

    peak_allocated = gibibytes(
        torch.cuda.max_memory_allocated()
    )
    peak_reserved = gibibytes(
        torch.cuda.max_memory_reserved()
    )

    print(f"\n{label}")
    print(f"Document: {chunk['doc_id']}")
    print(f"PDF page: {chunk['pdf_page']}")
    print(
        "Embedding tokens:",
        chunk["embedding_token_count"],
    )
    print(f"Embedding shape: {embedding.shape}")
    print(f"Finite values: {np.isfinite(vector).all()}")
    print(f"Vector norm: {vector_norm:.6f}")
    print(f"Latency: {elapsed:.3f} seconds")
    print(
        f"Peak allocated VRAM: "
        f"{peak_allocated:.3f} GiB"
    )
    print(
        f"Peak reserved VRAM: "
        f"{peak_reserved:.3f} GiB"
    )

    if embedding.shape != (1, EXPECTED_DIMENSION):
        raise ValueError(
            "Unexpected embedding shape: "
            f"{embedding.shape}"
        )

    if not np.isfinite(vector).all():
        raise ValueError(
            "Embedding contains non-finite values."
        )

    if not np.isclose(vector_norm, 1.0, atol=1e-4):
        raise ValueError(
            "Normalized embedding does not have unit norm."
        )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print(f"Loading model: {MODEL_NAME}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load on CPU first, convert to FP16, then move it to
    # the GPU. This avoids briefly keeping both FP32 and
    # FP16 model copies in the limited 6 GB GPU memory.
    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
    )

    model.half()
    model.to(DEVICE)
    model.eval()

    print(f"Model device: {model.device}")
    print(
        "Model parameter dtype:",
        next(model.parameters()).dtype,
    )
    print(
        "Model maximum sequence length:",
        model.max_seq_length,
    )

    if model.max_seq_length != EXPECTED_MAX_TOKENS:
        raise ValueError(
            "Unexpected model sequence limit: "
            f"{model.max_seq_length}"
        )

    model_memory = gibibytes(
        torch.cuda.memory_allocated()
    )
    print(
        f"Model allocated VRAM: "
        f"{model_memory:.3f} GiB"
    )

    doc_ids = read_split("dev") + read_split("test")
    chunks: list[dict] = []

    for doc_id in doc_ids:
        pages = ingest_document(doc_id)

        chunks.extend(
            build_page_chunks(
                pages=pages,
                tokenizer=model.tokenizer,
            )
        )

    ordered_chunks = sorted(
        chunks,
        key=lambda chunk: chunk[
            "embedding_token_count"
        ],
    )

    median_chunk = ordered_chunks[
        len(ordered_chunks) // 2
    ]
    longest_chunk = ordered_chunks[-1]

    print(f"Validated chunks: {len(chunks)}")

    encode_and_report(
        model=model,
        chunk=median_chunk,
        label="MEDIAN-LENGTH PAGE",
    )

    encode_and_report(
        model=model,
        chunk=longest_chunk,
        label="LONGEST PAGE",
    )

    print("\nEmbedding smoke test passed.")


if __name__ == "__main__":
    main()