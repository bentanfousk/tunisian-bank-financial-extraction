from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import pymupdf4llm


# This file is located at:
# <project_root>/scripts/phase0/extract_reports.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "reports_txt"


def read_doc_ids(split_path: Path) -> list[str]:
    """Read one doc_id per line from a split file."""

    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    doc_ids: list[str] = []

    with split_path.open("r", encoding="utf-8") as split_file:
        for line_number, raw_line in enumerate(split_file, start=1):
            doc_id = raw_line.strip()

            # Allow empty lines and comment lines.
            if not doc_id or doc_id.startswith("#"):
                continue

            if doc_id in doc_ids:
                raise ValueError(
                    f"Duplicate doc_id {doc_id!r} in "
                    f"{split_path} at line {line_number}."
                )

            doc_ids.append(doc_id)

    if not doc_ids:
        raise ValueError(f"No document IDs found in: {split_path}")

    return doc_ids


def build_pdf_index(raw_dir: Path) -> dict[str, Path]:
    """
    Index PDFs by filename stem.

    Example:
        data/raw/biat_2024.pdf
        becomes:
        {"biat_2024": Path(...)}
    """

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw PDF directory not found: {raw_dir}")

    pdf_index: dict[str, Path] = {}

    pdf_paths = sorted(
        path
        for path in raw_dir.iterdir()
        if path.is_file() and path.suffix.casefold() == ".pdf"
    )

    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in: {raw_dir}")

    for pdf_path in pdf_paths:
        normalized_stem = pdf_path.stem.casefold()

        if normalized_stem in pdf_index:
            previous_path = pdf_index[normalized_stem]

            raise ValueError(
                "Multiple PDFs have the same case-insensitive filename stem: "
                f"{previous_path.name} and {pdf_path.name}"
            )

        pdf_index[normalized_stem] = pdf_path

    return pdf_index


def resolve_pdf_path(
    doc_id: str,
    pdf_index: dict[str, Path],
) -> Path:
    """Find the raw PDF whose filename stem matches the doc_id."""

    normalized_doc_id = doc_id.casefold()

    if normalized_doc_id not in pdf_index:
        expected_filename = f"{doc_id}.pdf"

        raise FileNotFoundError(
            f"No PDF found for doc_id {doc_id!r}. "
            f"Expected a matching file such as "
            f"{RAW_DIR / expected_filename}"
        )

    return pdf_index[normalized_doc_id]


def extract_report(
    pdf_path: Path,
    doc_id: str,
    output_dir: Path,
    overwrite: bool = False,
) -> dict[str, object]:
    """Extract one PDF as page-indexed Markdown JSONL."""

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{doc_id}.jsonl"

    if output_path.exists() and not overwrite:
        return {
            "status": "skipped",
            "doc_id": doc_id,
            "pdf_path": str(pdf_path),
            "output_path": str(output_path),
        }

    start_time = perf_counter()

    page_chunks = pymupdf4llm.to_markdown(
        pdf_path,
        page_chunks=True,
    )

    if not page_chunks:
        raise ValueError(f"{doc_id}: PyMuPDF4LLM returned no pages.")

    page_numbers: list[int] = []

    for chunk_index, chunk in enumerate(page_chunks, start=1):
        metadata = chunk.get("metadata")

        if not isinstance(metadata, dict):
            raise TypeError(
                f"{doc_id}: page chunk {chunk_index} has no valid metadata."
            )

        page_number = metadata.get("page_number")

        if not isinstance(page_number, int):
            raise TypeError(
                f"{doc_id}: page chunk {chunk_index} has no valid page number."
            )

        page_numbers.append(page_number)

    expected_page_numbers = list(range(1, len(page_chunks) + 1))

    if page_numbers != expected_page_numbers:
        raise ValueError(
            f"{doc_id}: page numbers are missing, duplicated, "
            "or out of order."
        )

    empty_pages: list[int] = []

    # Write to a temporary file first. If extraction or writing fails,
    # an incomplete final JSONL file will not be left behind.
    temporary_path = output_path.with_suffix(".jsonl.tmp")

    try:
        with temporary_path.open("w", encoding="utf-8") as output_file:
            for chunk in page_chunks:
                metadata = chunk["metadata"]
                page_number = metadata["page_number"]

                text = chunk.get("text", "")

                if not isinstance(text, str):
                    raise TypeError(
                        f"{doc_id}: page {page_number} returned "
                        "non-text content."
                    )

                if not text.strip():
                    empty_pages.append(page_number)

                record = {
                    "doc_id": doc_id,
                    "page": page_number,
                    "text": text,
                }

                output_file.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

        temporary_path.replace(output_path)

    except Exception:
        # Remove an incomplete temporary file if something failed.
        temporary_path.unlink(missing_ok=True)
        raise

    elapsed = perf_counter() - start_time

    return {
        "status": "written",
        "doc_id": doc_id,
        "pdf_path": str(pdf_path),
        "page_count": len(page_chunks),
        "empty_page_count": len(empty_pages),
        "empty_pages": empty_pages,
        "output_path": str(output_path),
        "elapsed_seconds": round(elapsed, 2),
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract all reports listed in data/splits/dev.txt "
            "as page-indexed Markdown JSONL."
        )
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract reports whose JSONL output already exists.",
    )
    parser.add_argument(
        "--split",
        choices=["dev", "test", "temporal"],
        default="dev",
        help="Dataset split to extract.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    split_path = SPLITS_DIR / f"{args.split}.txt"
    doc_ids = read_doc_ids(split_path)
    pdf_index = build_pdf_index(RAW_DIR)

    failures: list[tuple[str, str]] = []
    written_count = 0
    skipped_count = 0

    print(f"Development reports to process: {len(doc_ids)}")
    print(f"Split file: {split_path}")
    print(f"Output directory: {OUTPUT_DIR}\n")

    for position, doc_id in enumerate(doc_ids, start=1):
        print(
            f"[{position}/{len(doc_ids)}] Processing {doc_id}...",
            flush=True,
        )

        try:
            pdf_path = resolve_pdf_path(doc_id, pdf_index)

            summary = extract_report(
                pdf_path=pdf_path,
                doc_id=doc_id,
                output_dir=OUTPUT_DIR,
                overwrite=args.overwrite,
            )

            if summary["status"] == "skipped":
                skipped_count += 1
                print(
                    "  Skipped: output already exists.\n"
                    f"  Output: {summary['output_path']}\n"
                )
                continue

            written_count += 1

            print(f"  PDF: {summary['pdf_path']}")
            print(f"  Pages extracted: {summary['page_count']}")
            print(f"  Empty pages: {summary['empty_page_count']}")
            print(f"  Empty page numbers: {summary['empty_pages']}")
            print(f"  Output: {summary['output_path']}")
            print(
                f"  Elapsed time: "
                f"{summary['elapsed_seconds']} seconds\n"
            )

        except Exception as error:
            failures.append((doc_id, str(error)))
            print(f"  ERROR: {error}\n")

    print("=" * 60)
    print("Extraction summary")
    print(f"Written: {written_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Failed: {len(failures)}")

    if failures:
        print("\nFailures:")

        for doc_id, message in failures:
            print(f"- {doc_id}: {message}")

        raise SystemExit(1)


if __name__ == "__main__":
    main()