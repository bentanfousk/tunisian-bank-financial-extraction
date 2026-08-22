from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from scripts.phase2_page_select.page_select import (
    MANIFEST_PATH,
    get_document_entry,
    get_page_refs,
    load_json,
)
from scripts.v2.phase1_rag.build_index import (
    MODEL_NAME,
    MODEL_REVISION,
    PROJECT_ROOT,
    load_embedding_model,
)
from scripts.v2.phase1_rag.retrieve import (
    CANDIDATE_K,
    encode_query,
    load_index_artifacts,
    merge_query_results,
    retrieve_query,
)
from scripts.v2.phase1_rag.retrieval_queries import (
    build_retrieval_queries,
)


ANNOTATIONS_DIR = (
    PROJECT_ROOT / "data" / "annotations"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "v2"
    / "phase2_rag_evaluation_reranked"
)


def load_annotation(
    doc_id: str,
) -> dict[str, Any]:
    path = ANNOTATIONS_DIR / f"{doc_id}.json"

    if not path.exists():
        raise FileNotFoundError(
            f"Annotation not found: {path}"
        )

    annotation = json.loads(
        path.read_text(encoding="utf-8")
    )

    if annotation.get("doc_id") != doc_id:
        raise ValueError(
            f"Unexpected annotation document ID "
            f"in {path}."
        )

    fields = annotation.get("fields")

    if not isinstance(fields, list):
        raise ValueError(
            f"{doc_id}: annotation has no fields list."
        )

    return annotation


def build_report_to_pdf_mapping(
    document: dict[str, Any],
) -> dict[int, int]:
    """
    Build an evaluation-only mapping from visible report
    pages to physical PDF pages.
    """

    scopes = document.get("scopes")

    if not isinstance(scopes, dict):
        raise ValueError(
            "Manifest document has no scopes object."
        )

    mapping: dict[int, int] = {}

    for scope_entry in scopes.values():
        if not isinstance(scope_entry, dict):
            continue

        statement_pages = scope_entry.get(
            "statement_pages"
        )

        if not isinstance(
            statement_pages,
            dict,
        ):
            continue

        for page_reference in (
            statement_pages.values()
        ):
            report_page = page_reference.get(
                "report_page"
            )
            pdf_page = page_reference.get(
                "pdf_page"
            )

            if not isinstance(
                report_page,
                int,
            ):
                raise ValueError(
                    "Invalid report page in manifest."
                )

            if not isinstance(pdf_page, int):
                raise ValueError(
                    "Invalid PDF page in manifest."
                )

            existing = mapping.get(report_page)

            if (
                existing is not None
                and existing != pdf_page
            ):
                raise ValueError(
                    f"Report page {report_page} maps "
                    "to multiple physical PDF pages."
                )

            mapping[report_page] = pdf_page

    return mapping


def get_authoritative_pdf_pages(
    document: dict[str, Any],
    doc_id: str,
) -> set[int]:
    """
    Return the individual balance-sheet and
    income-statement physical pages.
    """

    references = get_page_refs(
        document=document,
        doc_id=doc_id,
        scope="individual",
        core_only=True,
    )

    return {
        reference["pdf_page"]
        for reference in references
    }


def get_consolidated_pdf_pages(
    document: dict[str, Any],
) -> set[int]:
    """
    Return all known consolidated statement pages.

    Documents without verified consolidated statements
    return an empty set.
    """

    scopes = document.get("scopes", {})
    consolidated = scopes.get("consolidated")

    if not isinstance(consolidated, dict):
        return set()

    statement_pages = consolidated.get(
        "statement_pages"
    )

    if not isinstance(statement_pages, dict):
        return set()

    return {
        page_reference["pdf_page"]
        for page_reference
        in statement_pages.values()
        if isinstance(
            page_reference.get("pdf_page"),
            int,
        )
    }


def get_annotation_distractor_pages(
    annotation: dict[str, Any],
    report_to_pdf: dict[int, int],
) -> tuple[set[int], set[int]]:
    """
    Map annotation distractor report pages to physical
    PDF pages where a deterministic mapping exists.

    Returns:
        mapped PDF distractor pages,
        unmapped report distractor pages.
    """

    mapped: set[int] = set()
    unmapped: set[int] = set()

    for field in annotation["fields"]:
        distractors = field.get(
            "distractors",
            [],
        )

        for distractor in distractors:
            report_page = distractor.get("page")

            if not isinstance(report_page, int):
                continue

            pdf_page = report_to_pdf.get(
                report_page
            )

            if pdf_page is None:
                unmapped.add(report_page)
            else:
                mapped.add(pdf_page)

    return mapped, unmapped


def compact_result(
    result: dict[str, Any],
) -> dict[str, Any]:
    """
    Remove full page text from the evaluation JSON while
    preserving retrieval provenance.
    """

    return {
        "merged_rank": result["merged_rank"],
        "chunk_id": result["chunk_id"],
        "pdf_page": result["pdf_page"],
        "report_page": result["report_page"],
        "best_score": result["best_score"],
        "matches": result["matches"],
    }


def evaluate_configuration(
    *,
    configuration_name: str,
    pages_per_query: int,
    merged_results: list[dict[str, Any]],
    chunks_by_id: dict[str, dict[str, Any]],
    annotation: dict[str, Any],
    document: dict[str, Any],
    doc_id: str,
) -> dict[str, Any]:
    retrieved_pages = [
        result["pdf_page"]
        for result in merged_results
    ]
    retrieved_page_set = set(retrieved_pages)

    authoritative_pages = (
        get_authoritative_pdf_pages(
            document=document,
            doc_id=doc_id,
        )
    )

    retrieved_authoritative_pages = (
        retrieved_page_set
        & authoritative_pages
    )

    authoritative_page_recall = (
        len(retrieved_authoritative_pages)
        / len(authoritative_pages)
    )

    report_to_pdf = (
        build_report_to_pdf_mapping(
            document
        )
    )

    field_coverage: list[dict[str, Any]] = []

    for field in annotation["fields"]:
        report_page = field.get("page")
        pdf_page = (
            report_to_pdf.get(report_page)
            if isinstance(report_page, int)
            else None
        )

        covered = (
            pdf_page in retrieved_page_set
            if pdf_page is not None
            else False
        )

        field_coverage.append(
            {
                "field": field["field"],
                "status": field["status"],
                "authoritative_report_page": (
                    report_page
                ),
                "authoritative_pdf_page": (
                    pdf_page
                ),
                "covered": covered,
            }
        )

    covered_fields = sum(
        field["covered"]
        for field in field_coverage
    )

    field_evidence_recall = (
        covered_fields
        / len(field_coverage)
    )

    consolidated_pages = (
        get_consolidated_pdf_pages(
            document
        )
    )

    (
        annotation_distractor_pages,
        unmapped_distractor_report_pages,
    ) = get_annotation_distractor_pages(
        annotation=annotation,
        report_to_pdf=report_to_pdf,
    )

    known_contaminant_pages = (
        consolidated_pages
        | annotation_distractor_pages
    )

    # A page that is itself authoritative is not counted
    # as contaminated merely because it also contains a
    # prior-year comparison column.
    retrieved_contaminant_pages = (
        retrieved_page_set
        & known_contaminant_pages
        - authoritative_pages
    )

    contamination_rate = (
        len(retrieved_contaminant_pages)
        / len(retrieved_page_set)
        if retrieved_page_set
        else 0.0
    )

    non_authoritative_pages = (
        retrieved_page_set
        - authoritative_pages
    )

    total_characters = 0
    total_bge_tokens = 0

    for result in merged_results:
        chunk = chunks_by_id[
            result["chunk_id"]
        ]

        total_characters += chunk[
            "character_count"
        ]
        total_bge_tokens += chunk[
            "embedding_token_count"
        ]

    return {
        "configuration": configuration_name,
        "pages_per_query": pages_per_query,
        "maximum_before_deduplication": (
            pages_per_query * 2
        ),
        "unique_pages_after_deduplication": (
            len(retrieved_page_set)
        ),
        "retrieved_pages_in_rank_order": (
            retrieved_pages
        ),
        "retrieved_pages_in_document_order": (
            sorted(retrieved_page_set)
        ),
        "authoritative_pdf_pages": sorted(
            authoritative_pages
        ),
        "retrieved_authoritative_pdf_pages": (
            sorted(
                retrieved_authoritative_pages
            )
        ),
        "authoritative_page_recall": (
            authoritative_page_recall
        ),
        "covered_fields": covered_fields,
        "total_fields": len(field_coverage),
        "field_evidence_recall": (
            field_evidence_recall
        ),
        "field_coverage": field_coverage,
        "known_consolidated_pdf_pages": (
            sorted(consolidated_pages)
        ),
        "mapped_annotation_distractor_pages": (
            sorted(
                annotation_distractor_pages
            )
        ),
        "unmapped_distractor_report_pages": (
            sorted(
                unmapped_distractor_report_pages
            )
        ),
        "retrieved_contaminant_pdf_pages": (
            sorted(
                retrieved_contaminant_pages
            )
        ),
        "contamination_rate": (
            contamination_rate
        ),
        "non_authoritative_pdf_pages": (
            sorted(non_authoritative_pages)
        ),
        "total_retrieved_characters": (
            total_characters
        ),
        "total_retrieved_bge_tokens": (
            total_bge_tokens
        ),
        "results": [
            compact_result(result)
            for result in merged_results
        ],
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate RAG-4, RAG-6 and RAG-8 on one "
            "development report."
        )
    )

    parser.add_argument(
        "--doc-id",
        required=True,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    # Retrieval happens before ground-truth artifacts
    # are loaded.
    index, chunks, index_meta = (
        load_index_artifacts(args.doc_id)
    )

    first_chunk = chunks[0]

    queries = build_retrieval_queries(
        bank=first_chunk["bank"],
        fiscal_year=first_chunk[
            "fiscal_year"
        ],
    )

    model_load_start = perf_counter()
    model = load_embedding_model()
    torch.cuda.synchronize()
    model_load_seconds = (
        perf_counter() - model_load_start
    )

    results_by_query: dict[
        str,
        list[dict[str, Any]],
    ] = {}
    query_timings: dict[
        str,
        dict[str, float],
    ] = {}

    for query in queries:
        torch.cuda.synchronize()
        embedding_start = perf_counter()

        query_vector = encode_query(
            model=model,
            query_text=query["text"],
        )

        torch.cuda.synchronize()
        embedding_seconds = (
            perf_counter() - embedding_start
        )

        search_start = perf_counter()

        results = retrieve_query(
            index=index,
            chunks=chunks,
            query_vector=query_vector,
            query_id=query["query_id"],
            query_text=query["text"],
            fiscal_year=first_chunk[
                "fiscal_year"
            ],
            top_k=4,
        )

        search_seconds = (
            perf_counter() - search_start
        )

        results_by_query[
            query["query_id"]
        ] = results

        query_timings[query["query_id"]] = {
            "embedding_seconds": (
                embedding_seconds
            ),
            "retrieval_seconds": search_seconds,
        }

    # Evaluation-only artifacts are loaded after the
    # retrieval results already exist.
    manifest = load_json(MANIFEST_PATH)

    document = get_document_entry(
        manifest=manifest,
        doc_id=args.doc_id,
    )

    annotation = load_annotation(
        args.doc_id
    )

    chunks_by_id = {
        chunk["chunk_id"]: chunk
        for chunk in chunks
    }

    configurations: list[
        dict[str, Any]
    ] = []

    for pages_per_query in [2, 3, 4]:
        selected_results = {
            query_id: results[
                :pages_per_query
            ]
            for query_id, results
            in results_by_query.items()
        }

        merged_results = merge_query_results(
            selected_results
        )

        configuration = (
            evaluate_configuration(
                configuration_name=(
                    f"RAG-{pages_per_query * 2}"
                ),
                pages_per_query=(
                    pages_per_query
                ),
                merged_results=merged_results,
                chunks_by_id=chunks_by_id,
                annotation=annotation,
                document=document,
                doc_id=args.doc_id,
            )
        )

        configurations.append(
            configuration
        )

    compact_query_results = {
        query_id: [
            {
                "rank": result["rank"],
                "score": result["score"],
                "pdf_page": result[
                    "pdf_page"
                ],
                "chunk_id": result[
                    "chunk_id"
                ],
            }
            for result in results
        ]
        for query_id, results
        in results_by_query.items()
    }

    output = {
        "retrieval_method": (
            "bge_m3_faiss_authority_rerank_v1"
        ),
        "doc_id": args.doc_id,
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "index_model_revision": (
            index_meta["model_revision"]
        ),
        "queries": queries,
        "model_load_seconds": (
            model_load_seconds
        ),
        "query_timings": query_timings,
        "top_4_results_by_query": (
            compact_query_results
        ),
        "configurations": configurations,
    }

    output_directory = (
        OUTPUT_ROOT / args.doc_id
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_directory
        / "retrieval_evaluation.json"
    )

    output_path.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Document: {args.doc_id}")

    for configuration in configurations:
        print()
        print(
            configuration["configuration"]
        )
        print(
            "Retrieved PDF pages:",
            configuration[
                "retrieved_pages_in_rank_order"
            ],
        )
        print(
            "Authoritative page recall:",
            f"{configuration['authoritative_page_recall']:.1%}",
        )
        print(
            "Field evidence recall:",
            f"{configuration['field_evidence_recall']:.1%}",
        )
        print(
            "Contamination rate:",
            f"{configuration['contamination_rate']:.1%}",
        )
        print(
            "Non-authoritative pages:",
            configuration[
                "non_authoritative_pdf_pages"
            ],
        )

    print()
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()