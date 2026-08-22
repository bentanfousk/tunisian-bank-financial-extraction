from __future__ import annotations

import argparse
import math
from collections import defaultdict
from typing import Any

from transformers import AutoTokenizer

from scripts.v2.phase1_rag.ingestion import (
    ingest_document,
    read_split,
)


MODEL_NAME = "BAAI/bge-m3"
MAX_MODEL_TOKENS = 8192
THRESHOLDS = (512, 1024, 2048, 4096, 8192)


def calculate_percentile(
    values: list[int],
    percentile: float,
) -> float:
    if not values:
        raise ValueError("Cannot calculate an empty percentile.")

    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile

    lower_index = math.floor(position)
    upper_index = math.ceil(position)

    if lower_index == upper_index:
        return float(ordered[lower_index])

    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    fraction = position - lower_index

    return lower_value + (
        upper_value - lower_value
    ) * fraction


def count_tokens(tokenizer: Any, text: str) -> int:
    token_ids = tokenizer.encode(
        text,
        add_special_tokens=True,
        truncation=False,
    )

    return len(token_ids)


def read_documents(
    split: str,
    only_doc: str | None,
) -> list[str]:
    if only_doc:
        return [only_doc]

    if split == "all":
        return read_split("dev") + read_split("test")

    return read_split(split)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit complete-report page lengths using the "
            "BGE-M3 tokenizer without loading model weights."
        )
    )

    parser.add_argument(
        "--split",
        choices=["dev", "test", "all"],
        default="all",
    )
    parser.add_argument(
        "--only-doc",
        help="Audit only one document identifier.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    doc_ids = read_documents(
        split=args.split,
        only_doc=args.only_doc,
    )

    print(f"Loading tokenizer: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    print(f"Tokenizer class: {type(tokenizer).__name__}")
    print(
        "Tokenizer model maximum:",
        tokenizer.model_max_length,
    )

    if tokenizer.model_max_length != MAX_MODEL_TOKENS:
        raise ValueError(
            "Unexpected BGE-M3 tokenizer maximum: "
            f"{tokenizer.model_max_length}"
        )

    all_results: list[dict[str, Any]] = []
    results_by_document: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for doc_id in doc_ids:
        pages = ingest_document(doc_id)

        for page in pages:
            if not page["indexable"]:
                continue

            token_count = count_tokens(
                tokenizer,
                page["text"],
            )

            result = {
                "doc_id": doc_id,
                "pdf_page": page["pdf_page"],
                "character_count": page["character_count"],
                "token_count": token_count,
            }

            all_results.append(result)
            results_by_document[doc_id].append(result)

    print("\n" + "=" * 72)
    print("PER-DOCUMENT TOKEN LENGTHS")
    print("=" * 72)

    for doc_id in doc_ids:
        results = results_by_document[doc_id]
        token_counts = [
            result["token_count"]
            for result in results
        ]

        maximum_result = max(
            results,
            key=lambda result: result["token_count"],
        )

        over_limit = sum(
            count > MAX_MODEL_TOKENS
            for count in token_counts
        )

        print(f"\nDocument: {doc_id}")
        print(f"Indexable pages: {len(results)}")
        print(
            "Median tokens:",
            round(
                calculate_percentile(
                    token_counts,
                    0.50,
                ),
                1,
            ),
        )
        print(
            "P95 tokens:",
            round(
                calculate_percentile(
                    token_counts,
                    0.95,
                ),
                1,
            ),
        )
        print(
            "Maximum tokens:",
            maximum_result["token_count"],
        )
        print(
            "Maximum-token PDF page:",
            maximum_result["pdf_page"],
        )
        print(
            f"Pages over {MAX_MODEL_TOKENS}:",
            over_limit,
        )

    all_token_counts = [
        result["token_count"]
        for result in all_results
    ]

    print("\n" + "=" * 72)
    print("GLOBAL TOKEN-LENGTH SUMMARY")
    print("=" * 72)
    print(f"Documents: {len(doc_ids)}")
    print(f"Indexable pages: {len(all_results)}")
    print(
        "Median tokens:",
        round(
            calculate_percentile(
                all_token_counts,
                0.50,
            ),
            1,
        ),
    )
    print(
        "P95 tokens:",
        round(
            calculate_percentile(
                all_token_counts,
                0.95,
            ),
            1,
        ),
    )
    print(f"Maximum tokens: {max(all_token_counts)}")

    for threshold in THRESHOLDS:
        count = sum(
            token_count > threshold
            for token_count in all_token_counts
        )

        percentage = (
            count / len(all_token_counts) * 100
        )

        print(
            f"Pages over {threshold}: "
            f"{count} ({percentage:.2f}%)"
        )

    print("\n" + "=" * 72)
    print("TEN LONGEST PAGES")
    print("=" * 72)

    longest_pages = sorted(
        all_results,
        key=lambda result: result["token_count"],
        reverse=True,
    )[:10]

    for rank, result in enumerate(
        longest_pages,
        start=1,
    ):
        print(
            f"{rank:2}. "
            f"{result['doc_id']} "
            f"PDF page {result['pdf_page']} "
            f"tokens={result['token_count']} "
            f"characters={result['character_count']}"
        )


if __name__ == "__main__":
    main()