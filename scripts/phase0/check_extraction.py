from __future__ import annotations

import json
from pathlib import Path

import pymupdf
import argparse
from extract_reports import (
    SPLITS_DIR,
    OUTPUT_DIR,
    RAW_DIR,
    build_pdf_index,
    read_doc_ids,
    resolve_pdf_path,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate extracted report JSONL files."
    )

    parser.add_argument(
        "--split",
        choices=["dev", "test", "temporal"],
        default="dev",
        help="Dataset split to validate.",
    )

    return parser.parse_args()

def load_jsonl(jsonl_path: Path) -> list[dict[str, object]]:
    """Load and validate the basic syntax of one JSONL file."""

    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Extracted JSONL not found: {jsonl_path}"
        )

    records: list[dict[str, object]] = []

    with jsonl_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                raise ValueError(
                    f"{jsonl_path.name}: blank line at "
                    f"JSONL line {line_number}."
                )

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{jsonl_path.name}: invalid JSON "
                    f"at line {line_number}."
                ) from error

            if not isinstance(record, dict):
                raise TypeError(
                    f"{jsonl_path.name}: line {line_number} "
                    "is not a JSON object."
                )

            records.append(record)

    if not records:
        raise ValueError(f"{jsonl_path.name}: file contains no records.")

    return records


def check_report(
    doc_id: str,
    pdf_path: Path,
    jsonl_path: Path,
) -> dict[str, object]:
    """Validate one extracted report against its source PDF."""

    records = load_jsonl(jsonl_path)

    extracted_page_numbers: list[int] = []
    empty_page_numbers: list[int] = []

    for line_number, record in enumerate(records, start=1):
        record_doc_id = record.get("doc_id")
        page_number = record.get("page")
        text = record.get("text")

        if record_doc_id != doc_id:
            raise ValueError(
                f"{jsonl_path.name}: line {line_number} has "
                f"doc_id {record_doc_id!r}, expected {doc_id!r}."
            )

        if not isinstance(page_number, int):
            raise TypeError(
                f"{jsonl_path.name}: line {line_number} has "
                "an invalid page number."
            )

        if not isinstance(text, str):
            raise TypeError(
                f"{jsonl_path.name}: line {line_number} has "
                "an invalid text value."
            )

        extracted_page_numbers.append(page_number)

        if not text.strip():
            empty_page_numbers.append(page_number)

    expected_page_numbers = list(range(1, len(records) + 1))

    if extracted_page_numbers != expected_page_numbers:
        raise ValueError(
            f"{doc_id}: page numbers are missing, duplicated, "
            "or out of order."
        )

    empty_page_details: list[dict[str, object]] = []

    with pymupdf.open(pdf_path) as document:
        if len(records) != document.page_count:
            raise ValueError(
                f"{doc_id}: JSONL contains {len(records)} pages, "
                f"but the PDF contains {document.page_count} pages."
            )

        for page_number in empty_page_numbers:
            # PyMuPDF uses zero-based page indexes.
            page = document[page_number - 1]

            plain_text = page.get_text(
                "text",
                sort=True,
            ).strip()

            images = page.get_images(full=True)
            drawings = page.get_drawings()

            empty_page_details.append(
                {
                    "page": page_number,
                    "plain_text_characters": len(plain_text),
                    "images": len(images),
                    "vector_drawings": len(drawings),
                    "text_preview": plain_text[:150].replace(
                        "\n",
                        " ",
                    ),
                }
            )

        pdf_page_count = document.page_count

    return {
        "doc_id": doc_id,
        "pdf_page_count": pdf_page_count,
        "jsonl_record_count": len(records),
        "empty_page_count": len(empty_page_numbers),
        "empty_page_details": empty_page_details,
    }


def main() -> None:
    args = parse_arguments()

    split_path = SPLITS_DIR / f"{args.split}.txt"

    doc_ids = read_doc_ids(split_path)
    pdf_index = build_pdf_index(RAW_DIR)

    failures: list[tuple[str, str]] = []
    passed_count = 0

    print(f"{args.split.capitalize()} reports to check: {len(doc_ids)}")
    print(f"Split file: {split_path}\n")

    for position, doc_id in enumerate(doc_ids, start=1):
        print(f"[{position}/{len(doc_ids)}] Checking {doc_id}")

        try:
            pdf_path = resolve_pdf_path(doc_id, pdf_index)
            jsonl_path = OUTPUT_DIR / f"{doc_id}.jsonl"

            summary = check_report(
                doc_id=doc_id,
                pdf_path=pdf_path,
                jsonl_path=jsonl_path,
            )

            passed_count += 1

            print(f"  PDF pages: {summary['pdf_page_count']}")
            print(
                f"  JSONL records: "
                f"{summary['jsonl_record_count']}"
            )
            print(
                f"  Empty pages: "
                f"{summary['empty_page_count']}"
            )

            empty_page_details = summary["empty_page_details"]

            for details in empty_page_details:
                print(f"\n  Empty page {details['page']}")
                print(
                    "    Plain PyMuPDF characters: "
                    f"{details['plain_text_characters']}"
                )
                print(f"    Images: {details['images']}")
                print(
                    f"    Vector drawings: "
                    f"{details['vector_drawings']}"
                )
                print(
                    f"    Text preview: "
                    f"{details['text_preview']!r}"
                )

            print("\n  Result: PASS\n")

        except Exception as error:
            failures.append((doc_id, str(error)))
            print(f"  Result: FAIL")
            print(f"  Error: {error}\n")

    print("=" * 60)
    print("Validation summary")
    print(f"Passed: {passed_count}")
    print(f"Failed: {len(failures)}")

    if failures:
        print("\nFailures:")

        for doc_id, message in failures:
            print(f"- {doc_id}: {message}")

        raise SystemExit(1)


if __name__ == "__main__":
    main()