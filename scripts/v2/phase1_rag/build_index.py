from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import faiss
import numpy as np
import sentence_transformers
import torch
from sentence_transformers import SentenceTransformer

from scripts.v2.phase1_rag.chunking import (
    build_page_chunks,
)
from scripts.v2.phase1_rag.ingestion import (
    PROJECT_ROOT,
    ingest_document,
    read_split,
)


MODEL_NAME = "BAAI/bge-m3"
MODEL_REVISION = (
    "5617a9f61b028005a4858fdac845db406aefb181"
)
EXPECTED_DIMENSION = 1024
EXPECTED_MAX_TOKENS = 8192
DEFAULT_BATCH_SIZE = 1

INDEX_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "v2"
    / "phase1_rag"
    / "indexes"
)


def get_model_revision(
    model: SentenceTransformer,
) -> str | None:
    """
    Return the Hugging Face commit hash resolved by the
    underlying Transformers model, when available.
    """

    first_module = model._first_module()

    auto_model = getattr(
        first_module,
        "auto_model",
        None,
    )
    config = getattr(
        auto_model,
        "config",
        None,
    )

    return getattr(
        config,
        "_commit_hash",
        None,
    )


def load_embedding_model() -> SentenceTransformer:
    """
    Load the frozen BGE-M3 revision in FP16 on the GPU.
    """

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    model = SentenceTransformer(
        MODEL_NAME,
        revision=MODEL_REVISION,
        device="cpu",
    )

    model.half()
    model.to("cuda")
    model.eval()

    if model.max_seq_length != EXPECTED_MAX_TOKENS:
        raise ValueError(
            "Unexpected BGE-M3 sequence limit: "
            f"{model.max_seq_length}"
        )

    resolved_revision = get_model_revision(model)

    if resolved_revision != MODEL_REVISION:
        raise ValueError(
            "Unexpected BGE-M3 revision: "
            f"{resolved_revision}. Expected "
            f"{MODEL_REVISION}."
        )

    return model


def encode_chunks(
    model: SentenceTransformer,
    chunks: list[dict[str, Any]],
    batch_size: int,
) -> np.ndarray:
    """
    Generate normalized dense BGE-M3 embeddings for
    page-level chunks.
    """

    if not chunks:
        raise ValueError(
            "Cannot embed an empty chunk list."
        )

    if batch_size < 1:
        raise ValueError(
            "batch_size must be positive."
        )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    embeddings = np.ascontiguousarray(
        embeddings,
        dtype=np.float32,
    )
    # Normalize after conversion to the final FAISS
    # storage precision. The operation modifies the
    # matrix in place.
    faiss.normalize_L2(embeddings)

    expected_shape = (
        len(chunks),
        EXPECTED_DIMENSION,
    )

    if embeddings.shape != expected_shape:
        raise ValueError(
            f"Unexpected embedding shape "
            f"{embeddings.shape}; expected "
            f"{expected_shape}."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "Embeddings contain non-finite values."
        )

    norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    if not np.allclose(
        norms,
        1.0,
        atol=1e-5,
    ):
        raise ValueError(
            "One or more embeddings are not normalized."
        )

    return embeddings


def create_faiss_index(
    embeddings: np.ndarray,
) -> faiss.IndexFlatIP:
    """
    Create an exact inner-product FAISS index.

    Since embeddings are normalized, inner product is
    equivalent to cosine similarity.
    """

    if embeddings.ndim != 2:
        raise ValueError(
            "Embeddings must be a two-dimensional array."
        )

    if embeddings.shape[0] == 0:
        raise ValueError(
            "Cannot index zero embeddings."
        )

    if embeddings.shape[1] != EXPECTED_DIMENSION:
        raise ValueError(
            "Unexpected embedding dimension: "
            f"{embeddings.shape[1]}"
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "Embeddings contain non-finite values."
        )

    vectors = np.ascontiguousarray(
        embeddings,
        dtype=np.float32,
    )

    norms = np.linalg.norm(
        vectors,
        axis=1,
    )

    if not np.allclose(
        norms,
        1.0,
        atol=1e-5,
    ):
        raise ValueError(
            "FAISS vectors must be L2-normalized."
        )

    index = faiss.IndexFlatIP(
        EXPECTED_DIMENSION
    )
    index.add(vectors)

    if index.ntotal != len(vectors):
        raise RuntimeError(
            "FAISS did not store every embedding."
        )

    return index


def prepare_output_directory(
    doc_id: str,
    force: bool,
) -> Path:
    """
    Create the report-specific artifact directory.

    Existing non-empty directories are protected unless
    --force is supplied.
    """

    output_directory = INDEX_ROOT / doc_id

    if output_directory.exists():
        existing_files = list(
            output_directory.iterdir()
        )

        if existing_files and not force:
            raise FileExistsError(
                f"Index artifacts already exist for "
                f"{doc_id}: {output_directory}. "
                "Use --force to replace them."
            )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_directory


def save_chunks(
    chunks: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Save chunk text and metadata in the same order as
    their vectors were inserted into FAISS.
    """

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as stream:
        for faiss_position, chunk in enumerate(chunks):
            record = {
                "faiss_position": faiss_position,
                **chunk,
            }

            stream.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
            )
            stream.write("\n")


def write_json(
    output_path: Path,
    payload: dict[str, Any],
) -> None:
    """Write one human-readable UTF-8 JSON file."""

    output_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def build_document_index(
    doc_id: str,
    model: SentenceTransformer,
    *,
    batch_size: int,
    force: bool,
    model_load_seconds: float,
) -> dict[str, Any]:
    """
    Embed and index every usable page of one report.
    """

    print("\n" + "=" * 72)
    print(f"Building index: {doc_id}")
    print("=" * 72)

    output_directory = prepare_output_directory(
        doc_id=doc_id,
        force=force,
    )

    total_start = perf_counter()

    ingestion_start = perf_counter()

    pages = ingest_document(doc_id)

    chunks = build_page_chunks(
        pages=pages,
        tokenizer=model.tokenizer,
    )

    ingestion_seconds = (
        perf_counter() - ingestion_start
    )

    print(f"Total PDF pages: {len(pages)}")
    print(f"Indexable chunks: {len(chunks)}")
    print(
        "Skipped pages:",
        len(pages) - len(chunks),
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    embedding_start = perf_counter()

    embeddings = encode_chunks(
        model=model,
        chunks=chunks,
        batch_size=batch_size,
    )

    torch.cuda.synchronize()

    embedding_seconds = (
        perf_counter() - embedding_start
    )

    peak_allocated_gib = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )
    peak_reserved_gib = (
        torch.cuda.max_memory_reserved()
        / 1024**3
    )

    faiss_start = perf_counter()

    index = create_faiss_index(
        embeddings
    )

    faiss_seconds = (
        perf_counter() - faiss_start
    )

    index_path = (
        output_directory / "index.faiss"
    )
    chunks_path = (
        output_directory / "chunks.jsonl"
    )
    build_meta_path = (
        output_directory / "build_meta.json"
    )

    serialization_start = perf_counter()

    faiss.write_index(
        index,
        str(index_path),
    )

    save_chunks(
        chunks=chunks,
        output_path=chunks_path,
    )

    serialization_seconds = (
        perf_counter() - serialization_start
    )
    total_seconds = (
        perf_counter() - total_start
    )

    token_counts = [
        chunk["embedding_token_count"]
        for chunk in chunks
    ]

    build_meta = {
        "doc_id": doc_id,
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "resolved_model_revision": (
            get_model_revision(model)
        ),
        "model_precision": "float16",
        "embedding_storage_dtype": "float32",
        "embedding_dimension": EXPECTED_DIMENSION,
        "model_max_tokens": EXPECTED_MAX_TOKENS,
        "embedding_normalized": True,
        "normalization_stage": (
            "faiss_L2_after_float32_conversion"
        ),
        "similarity": (
            "cosine_via_normalized_inner_product"
        ),
        "faiss_index_type": "IndexFlatIP",
        "faiss_version": faiss.__version__,
        "sentence_transformers_version": (
            sentence_transformers.__version__
        ),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "batch_size": batch_size,
        "total_pdf_pages": len(pages),
        "indexable_chunks": len(chunks),
        "skipped_pages": (
            len(pages) - len(chunks)
        ),
        "minimum_chunk_tokens": min(
            token_counts
        ),
        "maximum_chunk_tokens": max(
            token_counts
        ),
        "model_load_seconds_shared": (
            model_load_seconds
        ),
        "ingestion_and_chunking_seconds": (
            ingestion_seconds
        ),
        "embedding_seconds": (
            embedding_seconds
        ),
        "faiss_build_seconds": (
            faiss_seconds
        ),
        "serialization_seconds": (
            serialization_seconds
        ),
        "total_seconds_excluding_model_load": (
            total_seconds
        ),
        "peak_allocated_vram_gib": (
            peak_allocated_gib
        ),
        "peak_reserved_vram_gib": (
            peak_reserved_gib
        ),
        "index_file_bytes": (
            index_path.stat().st_size
        ),
        "chunks_file_bytes": (
            chunks_path.stat().st_size
        ),
    }

    write_json(
        output_path=build_meta_path,
        payload=build_meta,
    )

    print(f"Embeddings: {embeddings.shape}")
    print(f"FAISS vectors: {index.ntotal}")
    print(
        f"Embedding time: "
        f"{embedding_seconds:.3f}s"
    )
    print(
        f"FAISS construction time: "
        f"{faiss_seconds:.6f}s"
    )
    print(
        f"Peak allocated VRAM: "
        f"{peak_allocated_gib:.3f} GiB"
    )
    print(
        f"Peak reserved VRAM: "
        f"{peak_reserved_gib:.3f} GiB"
    )
    print(f"Index: {index_path}")
    print(f"Chunks: {chunks_path}")
    print(f"Metadata: {build_meta_path}")

    return build_meta


def read_documents(
    split: str,
    only_doc: str | None,
) -> list[str]:
    """
    Select documents using either one explicit document
    or a dataset split.
    """

    if only_doc:
        return [only_doc]

    if split == "all":
        return (
            read_split("dev")
            + read_split("test")
        )

    return read_split(split)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one frozen BGE-M3 and FAISS index "
            "per complete annual report."
        )
    )

    parser.add_argument(
        "--split",
        choices=[
            "dev",
            "test",
            "all",
        ],
        default="dev",
    )
    parser.add_argument(
        "--only-doc",
        help="Build only one document index.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing index artifacts.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if args.batch_size < 1:
        raise ValueError(
            "--batch-size must be positive."
        )

    doc_ids = read_documents(
        split=args.split,
        only_doc=args.only_doc,
    )

    print(f"Documents: {len(doc_ids)}")
    print(f"Model: {MODEL_NAME}")
    print(f"Model revision: {MODEL_REVISION}")
    print(f"Batch size: {args.batch_size}")

    model_load_start = perf_counter()

    model = load_embedding_model()

    torch.cuda.synchronize()

    model_load_seconds = (
        perf_counter() - model_load_start
    )

    resolved_revision = get_model_revision(
        model
    )

    print(
        f"Model load time: "
        f"{model_load_seconds:.3f}s"
    )
    print(
        "Resolved model revision:",
        resolved_revision,
    )

    results: list[dict[str, Any]] = []

    for doc_id in doc_ids:
        result = build_document_index(
            doc_id=doc_id,
            model=model,
            batch_size=args.batch_size,
            force=args.force,
            model_load_seconds=(
                model_load_seconds
            ),
        )
        results.append(result)

    print("\n" + "=" * 72)
    print("INDEX BUILD SUMMARY")
    print("=" * 72)
    print(
        f"Documents completed: "
        f"{len(results)}"
    )
    print(
        "Total vectors:",
        sum(
            result["indexable_chunks"]
            for result in results
        ),
    )
    print(
        "Total embedding time:",
        round(
            sum(
                result["embedding_seconds"]
                for result in results
            ),
            3,
        ),
        "seconds",
    )


if __name__ == "__main__":
    main()