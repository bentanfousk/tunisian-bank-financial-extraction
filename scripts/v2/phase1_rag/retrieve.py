from __future__ import annotations

import argparse
import json
import re
import unicodedata
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from scripts.v2.phase1_rag.build_index import (
    EXPECTED_DIMENSION,
    INDEX_ROOT,
    MODEL_REVISION,
    load_embedding_model,
)
from scripts.v2.phase1_rag.retrieval_queries import (
    build_retrieval_queries,
)


CANDIDATE_K = 64
HEADING_CHARACTER_LIMIT = 1000

SUPPORTED_QUERY_IDS = {
    "balance_sheet",
    "income_statement",
}

CONSOLIDATED_PATTERNS = (
    # The optional "ide" handles OCR-fragmented titles
    # such as "BILAN CONSOL<br>...ID...É".
    r"\bbilan\s+consol(?:ide)?\b",
    (
        r"\betat\s+de\s+resultat\s+"
        r"consol(?:ide)?\b"
    ),
    (
        r"\betats?\s+financiers?\s+"
        r"consolides?\b"
    ),
    r"\bcomptes?\s+consolides?\b",
)

FIELD_PATTERNS = {
    "balance_sheet": (
        r"\btotal\s+(?:des?\s+)?actifs?\b|\btotal\s+bilan\b",
        r"\bcapitaux\s+propres\b",
        r"\bdepots?\b.{0,80}\bclientele\b",
        r"\bcreances?\b.{0,80}\bclientele\b",
    ),
    "income_statement": (
        r"\bproduit\s+net\s+bancaire\b",
        r"\bresultat\s+d.?exploitation\b",
        r"\bresultat\s+(?:net|de\s+l.?exercice)\b",
    ),
}

STRUCTURE_PATTERNS = {
    "balance_sheet": (
        r"\bac\s*\d+\b",
        r"\bpa\s*\d+\b",
        r"\bcp\s*\d+\b",
    ),
    "income_statement": (
        r"\bpr\s*\d+\b",
        r"\bch\s*\d+\b",
    ),
}


def normalize_for_matching(text: str) -> str:
    """
    Normalize Unicode and remove extraction markup before
    applying deterministic retrieval patterns.
    """

    decomposed = unicodedata.normalize(
        "NFKD",
        text,
    )

    normalized = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )

    normalized = (
        normalized
        .casefold()
        .replace("’", "'")
        .replace("‘", "'")
        .replace("`", "'")
    )

    # Remove HTML tags introduced by table and picture
    # extraction, for example <br> and <sup>.
    normalized = re.sub(
        r"<[^>]*>",
        " ",
        normalized,
    )

    # Markdown table/emphasis characters can touch row
    # codes, as in **_AC 1 -_**, and prevent regex word
    # boundaries from matching.
    normalized = re.sub(
        r"[*_#|]+",
        " ",
        normalized,
    )

    # Produce stable spacing after markup removal.
    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    )

    return normalized.strip()


def _matches_any(
    patterns: tuple[str, ...],
    text: str,
) -> bool:
    return any(
        re.search(pattern, text, flags=re.DOTALL)
        is not None
        for pattern in patterns
    )


def _pattern_coverage(
    patterns: tuple[str, ...],
    text: str,
) -> int:
    return sum(
        re.search(pattern, text, flags=re.DOTALL)
        is not None
        for pattern in patterns
    )


def candidate_features(
    *,
    text: str,
    query_id: str,
    fiscal_year: int,
) -> dict[str, Any]:
    """
    Derive generic, ground-truth-free page features used
    to prefer authoritative individual statement tables.
    """

    if query_id not in SUPPORTED_QUERY_IDS:
        raise ValueError(
            f"Unsupported query ID: {query_id}"
        )

    normalized = normalize_for_matching(text)
    heading = normalized[
        :HEADING_CHARACTER_LIMIT
    ]

    is_consolidated = _matches_any(
        CONSOLIDATED_PATTERNS,
        heading,
    )

    if query_id == "balance_sheet":
        statement_heading_match = (
            re.search(
                r"\bbilan\b",
                heading,
            )
            is not None
        )
    else:
        statement_heading_match = (
            re.search(
                r"\betat\s+de\s+resultat\b",
                heading,
            )
            is not None
        )

    field_coverage = _pattern_coverage(
        FIELD_PATTERNS[query_id],
        normalized,
    )
    structure_coverage = _pattern_coverage(
        STRUCTURE_PATTERNS[query_id],
        normalized,
    )

    year_match = (
        str(fiscal_year) in heading
    )

    return {
        "is_consolidated": is_consolidated,
        "statement_heading_match": (
            statement_heading_match
        ),
        "structure_coverage": structure_coverage,
        "field_coverage": field_coverage,
        "year_match": year_match,
    }


def rerank_candidates(
    *,
    candidates: list[dict[str, Any]],
    query_id: str,
    fiscal_year: int,
    top_k: int,
) -> list[dict[str, Any]]:
    """
    Remove explicit consolidated statements and rerank
    candidates using generic statement-page evidence.

    Dense similarity remains the final tie-breaker.
    No page IDs, annotations or manifests are used.
    """

    if top_k < 1:
        raise ValueError(
            "top_k must be positive."
        )

    enriched: list[dict[str, Any]] = []

    for candidate in candidates:
        features = candidate_features(
            text=candidate["text"],
            query_id=query_id,
            fiscal_year=fiscal_year,
        )

        enriched_candidate = {
            **candidate,
            "dense_rank": candidate["rank"],
            "dense_score": candidate["score"],
            **features,
        }

        if not features["is_consolidated"]:
            enriched.append(
                enriched_candidate
            )

    enriched.sort(
        key=lambda result: (
            -int(
                result[
                    "statement_heading_match"
                ]
            ),
            -result["structure_coverage"],
            -result["field_coverage"],
            -int(result["year_match"]),
            -result["dense_score"],
            result["pdf_page"],
            result["chunk_id"],
        )
    )

    selected = enriched[:top_k]

    for reranked_rank, result in enumerate(
        selected,
        start=1,
    ):
        result["rank"] = reranked_rank

    return selected


def load_index_artifacts(
    doc_id: str,
) -> tuple[
    faiss.Index,
    list[dict[str, Any]],
    dict[str, Any],
]:
    """
    Load and validate one report's persisted FAISS index,
    chunk metadata and construction metadata.
    """

    directory = INDEX_ROOT / doc_id
    index_path = directory / "index.faiss"
    chunks_path = directory / "chunks.jsonl"
    meta_path = directory / "build_meta.json"

    required_paths = [
        index_path,
        chunks_path,
        meta_path,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required index artifact not found: "
                f"{path}"
            )

    index = faiss.read_index(str(index_path))

    chunks = [
        json.loads(line)
        for line in chunks_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    meta = json.loads(
        meta_path.read_text(encoding="utf-8")
    )

    if index.d != EXPECTED_DIMENSION:
        raise ValueError(
            f"{doc_id}: unexpected FAISS dimension "
            f"{index.d}."
        )

    if index.ntotal != len(chunks):
        raise ValueError(
            f"{doc_id}: FAISS contains "
            f"{index.ntotal} vectors but metadata "
            f"contains {len(chunks)} chunks."
        )

    expected_positions = list(range(len(chunks)))
    actual_positions = [
        chunk["faiss_position"]
        for chunk in chunks
    ]

    if actual_positions != expected_positions:
        raise ValueError(
            f"{doc_id}: chunk positions are not "
            "consecutive or correctly ordered."
        )

    if not all(
        chunk["doc_id"] == doc_id
        for chunk in chunks
    ):
        raise ValueError(
            f"{doc_id}: index contains chunks from "
            "another document."
        )

    if meta["model_revision"] != MODEL_REVISION:
        raise ValueError(
            f"{doc_id}: index was built with an "
            "unexpected BGE-M3 revision."
        )

    if not meta["embedding_normalized"]:
        raise ValueError(
            f"{doc_id}: stored embeddings are not "
            "marked as normalized."
        )

    if (
        meta["normalization_stage"]
        != "faiss_L2_after_float32_conversion"
    ):
        raise ValueError(
            f"{doc_id}: unexpected normalization "
            "configuration."
        )

    return index, chunks, meta


def encode_query(
    model: SentenceTransformer,
    query_text: str,
) -> np.ndarray:
    """Encode and L2-normalize one semantic query."""

    if not query_text.strip():
        raise ValueError(
            "Query text cannot be empty."
        )

    query_vector = model.encode(
        [query_text],
        batch_size=1,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    query_vector = np.ascontiguousarray(
        query_vector,
        dtype=np.float32,
    )

    if query_vector.shape != (
        1,
        EXPECTED_DIMENSION,
    ):
        raise ValueError(
            "Unexpected query embedding shape: "
            f"{query_vector.shape}"
        )

    if not np.isfinite(query_vector).all():
        raise ValueError(
            "Query embedding contains non-finite "
            "values."
        )

    faiss.normalize_L2(query_vector)

    norm = np.linalg.norm(query_vector)

    if not np.isclose(norm, 1.0, atol=1e-5):
        raise ValueError(
            "Query embedding is not normalized."
        )

    return query_vector


def search_index(
    *,
    index: faiss.Index,
    chunks: list[dict[str, Any]],
    query_vector: np.ndarray,
    query_id: str,
    query_text: str,
    top_k: int,
) -> list[dict[str, Any]]:
    """Search FAISS and map positions to page chunks."""

    if top_k < 1:
        raise ValueError(
            "top_k must be positive."
        )

    if top_k > index.ntotal:
        raise ValueError(
            f"top_k={top_k} exceeds the index size "
            f"of {index.ntotal}."
        )

    scores, positions = index.search(
        query_vector,
        top_k,
    )

    results: list[dict[str, Any]] = []

    for rank, (score, faiss_position) in enumerate(
        zip(scores[0], positions[0], strict=True),
        start=1,
    ):
        position = int(faiss_position)
        chunk = chunks[position]

        if chunk["faiss_position"] != position:
            raise ValueError(
                "FAISS position does not match "
                "chunk metadata."
            )

        results.append(
            {
                "query_id": query_id,
                "query_text": query_text,
                "rank": rank,
                "score": float(score),
                "faiss_position": position,
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "pdf_page": chunk["pdf_page"],
                "report_page": chunk[
                    "report_page"
                ],
                "text": chunk["text"],
            }
        )

    return results


def retrieve_query(
    *,
    index: faiss.Index,
    chunks: list[dict[str, Any]],
    query_vector: np.ndarray,
    query_id: str,
    query_text: str,
    fiscal_year: int,
    top_k: int,
    candidate_k: int = CANDIDATE_K,
) -> list[dict[str, Any]]:
    """
    Retrieve a broad semantic pool, then return a small
    authority-aware reranked result set.
    """

    if candidate_k < top_k:
        raise ValueError(
            "candidate_k cannot be smaller than top_k."
        )

    raw_candidates = search_index(
        index=index,
        chunks=chunks,
        query_vector=query_vector,
        query_id=query_id,
        query_text=query_text,
        top_k=min(candidate_k, index.ntotal),
    )

    return rerank_candidates(
        candidates=raw_candidates,
        query_id=query_id,
        fiscal_year=fiscal_year,
        top_k=top_k,
    )


def merge_query_results(
    results_by_query: dict[
        str,
        list[dict[str, Any]],
    ],
) -> list[dict[str, Any]]:
    """Merge and deduplicate both query result sets."""

    merged_by_chunk: dict[
        str,
        dict[str, Any],
    ] = {}

    for query_id, results in results_by_query.items():
        for result in results:
            chunk_id = result["chunk_id"]

            if chunk_id not in merged_by_chunk:
                merged_by_chunk[chunk_id] = {
                    "chunk_id": chunk_id,
                    "doc_id": result["doc_id"],
                    "pdf_page": result["pdf_page"],
                    "report_page": result[
                        "report_page"
                    ],
                    "text": result["text"],
                    "best_rank": result["rank"],
                    "best_score": result["score"],
                    "matches": [],
                }

            merged = merged_by_chunk[chunk_id]
            merged["best_rank"] = min(
                merged["best_rank"],
                result["rank"],
            )
            merged["best_score"] = max(
                merged["best_score"],
                result["score"],
            )

            match = {
                "query_id": query_id,
                "rank": result["rank"],
                "score": result["score"],
            }

            for key in (
                "dense_rank",
                "dense_score",
                "statement_heading_match",
                "structure_coverage",
                "field_coverage",
                "year_match",
            ):
                if key in result:
                    match[key] = result[key]

            merged["matches"].append(match)

    merged_results = list(
        merged_by_chunk.values()
    )

    merged_results.sort(
        key=lambda result: (
            result["best_rank"],
            -result["best_score"],
            result["pdf_page"],
            result["chunk_id"],
        )
    )

    for merged_rank, result in enumerate(
        merged_results,
        start=1,
    ):
        result["merged_rank"] = merged_rank

    return merged_results


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run authority-aware semantic retrieval "
            "against one complete-report FAISS index."
        )
    )

    parser.add_argument(
        "--doc-id",
        required=True,
        help="Document index to search.",
    )
    parser.add_argument(
        "--pages-per-query",
        type=int,
        choices=[2, 3, 4],
        default=4,
        help=(
            "Final number of pages returned for each "
            "fixed query."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    index, chunks, meta = load_index_artifacts(
        args.doc_id
    )

    first_chunk = chunks[0]
    bank = first_chunk["bank"]
    fiscal_year = first_chunk["fiscal_year"]

    queries = build_retrieval_queries(
        bank=bank,
        fiscal_year=fiscal_year,
    )

    print(f"Document: {args.doc_id}")
    print(f"Indexed pages: {index.ntotal}")
    print(f"Candidate pool per query: {min(CANDIDATE_K, index.ntotal)}")
    print("Final pages per query:", args.pages_per_query)
    print("Model revision:", meta["model_revision"])

    model = load_embedding_model()

    results_by_query: dict[
        str,
        list[dict[str, Any]],
    ] = {}

    for query in queries:
        query_id = query["query_id"]
        query_text = query["text"]

        query_vector = encode_query(
            model=model,
            query_text=query_text,
        )

        results = retrieve_query(
            index=index,
            chunks=chunks,
            query_vector=query_vector,
            query_id=query_id,
            query_text=query_text,
            fiscal_year=fiscal_year,
            top_k=args.pages_per_query,
        )

        results_by_query[query_id] = results

        print("\n" + "=" * 72)
        print(f"QUERY: {query_id}")
        print("=" * 72)
        print(query_text)

        for result in results:
            print(
                f"Rank {result['rank']}: "
                f"PDF page {result['pdf_page']} "
                f"dense_rank={result['dense_rank']} "
                f"score={result['dense_score']:.6f} "
                f"heading={result['statement_heading_match']} "
                f"structure={result['structure_coverage']} "
                f"fields={result['field_coverage']}"
            )

    merged_results = merge_query_results(
        results_by_query
    )

    print("\n" + "=" * 72)
    print("MERGED AND DEDUPLICATED RESULTS")
    print("=" * 72)

    for result in merged_results:
        matched_queries = ", ".join(
            match["query_id"]
            for match in result["matches"]
        )

        print(
            f"Rank {result['merged_rank']}: "
            f"PDF page {result['pdf_page']} "
            f"best_score={result['best_score']:.6f} "
            f"queries={matched_queries}"
        )

    print(
        "\nMaximum before deduplication:",
        len(queries) * args.pages_per_query,
    )
    print(
        "Pages after deduplication:",
        len(merged_results),
    )


if __name__ == "__main__":
    main()