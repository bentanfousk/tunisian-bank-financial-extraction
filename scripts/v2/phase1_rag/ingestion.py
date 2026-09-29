from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pymupdf


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORTS_TEXT_DIR = (
    PROJECT_ROOT / "data" / "processed" / "reports_txt"
)
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

DOC_ID_PATTERN = re.compile(
    r"^(?P<bank>.+)_(?P<fiscal_year>\d{4})$"
)


def parse_doc_id(doc_id: str) -> tuple[str, int]:
    """
    Extract bank identifier and fiscal year from a document ID.

    Example:
        UIB_2024 -> ("UIB", 2024)
    """

    match = DOC_ID_PATTERN.fullmatch(doc_id)

    if match is None:
        raise ValueError(
            f"Invalid document identifier {doc_id!r}. "
            "Expected a value such as 'UIB_2024'."
        )

    bank = match.group("bank")
    fiscal_year = int(match.group("fiscal_year"))

    return bank, fiscal_year


def build_pdf_index() -> dict[str, Path]:
    """Index raw PDFs by case-insensitive filename stem."""

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Raw PDF directory not found: {RAW_DIR}"
        )

    pdf_index: dict[str, Path] = {}

    for pdf_path in sorted(RAW_DIR.glob("*.pdf")):
        normalized_stem = pdf_path.stem.casefold()

        if normalized_stem in pdf_index:
            raise ValueError(
                "Duplicate case-insensitive PDF stem: "
                f"{pdf_path.stem}"
            )

        pdf_index[normalized_stem] = pdf_path

    return pdf_index


def resolve_pdf_path(
    doc_id: str,
    pdf_index: dict[str, Path],
) -> Path:
    """Resolve the original PDF associated with a document ID."""

    pdf_path = pdf_index.get(doc_id.casefold())

    if pdf_path is None:
        raise FileNotFoundError(
            f"No raw PDF found for {doc_id!r}."
        )

    return pdf_path


def load_pdf_page_metadata(
    pdf_path: Path,
) -> list[dict[str, Any]]:
    """
    Load deterministic page metadata from the original PDF.

    report_page is populated only when PyMuPDF provides a
    clean numeric embedded page label. The raw label is kept
    separately for auditability.
    """

    page_metadata: list[dict[str, Any]] = []

    with pymupdf.open(pdf_path) as document:
        for page_index in range(document.page_count):
            pdf_page = page_index + 1
            raw_label = document[page_index].get_label()

            if isinstance(raw_label, str):
                raw_label = raw_label.strip() or None
            else:
                raw_label = None

            report_page = (
                int(raw_label)
                if raw_label is not None
                and raw_label.isdecimal()
                else None
            )

            page_metadata.append(
                {
                    "pdf_page": pdf_page,
                    "page_label": raw_label,
                    "report_page": report_page,
                }
            )

    return page_metadata


def load_jsonl_records(
    jsonl_path: Path,
    expected_doc_id: str,
) -> list[dict[str, Any]]:
    """Load and validate existing V1 page-indexed text."""

    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Extracted report not found: {jsonl_path}"
        )

    records: list[dict[str, Any]] = []

    with jsonl_path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(
            stream,
            start=1,
        ):
            if not raw_line.strip():
                raise ValueError(
                    f"{jsonl_path.name}: blank line "
                    f"at line {line_number}."
                )

            try:
                record = json.loads(raw_line)
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

            if record.get("doc_id") != expected_doc_id:
                raise ValueError(
                    f"{jsonl_path.name}: unexpected doc_id "
                    f"at line {line_number}."
                )

            pdf_page = record.get("page")
            text = record.get("text")

            if not isinstance(pdf_page, int) or pdf_page < 1:
                raise ValueError(
                    f"{jsonl_path.name}: invalid physical "
                    f"page at line {line_number}."
                )

            if not isinstance(text, str):
                raise TypeError(
                    f"{jsonl_path.name}: invalid text "
                    f"at line {line_number}."
                )

            records.append(record)

    expected_pages = list(range(1, len(records) + 1))
    actual_pages = [record["page"] for record in records]

    if actual_pages != expected_pages:
        raise ValueError(
            f"{jsonl_path.name}: pages are missing, "
            "duplicated, or out of order."
        )

    return records


def ingest_document(
    doc_id: str,
    *,
    pdf_path: Path | None = None,
    jsonl_path: Path | None = None,
) -> list[dict[str, Any]]:
    """
    Build normalized V2 page records for one complete report.

    This function does not access annotations or the manually
    verified V1 page manifest.
    """

    bank, fiscal_year = parse_doc_id(doc_id)

    if pdf_path is None:
        pdf_index = build_pdf_index()
        pdf_path = resolve_pdf_path(doc_id, pdf_index)
    if jsonl_path is None:
        jsonl_path = REPORTS_TEXT_DIR / f"{doc_id}.jsonl"

    text_records = load_jsonl_records(
        jsonl_path=jsonl_path,
        expected_doc_id=doc_id,
    )

    pdf_metadata = load_pdf_page_metadata(pdf_path)

    if len(text_records) != len(pdf_metadata):
        raise ValueError(
            f"{doc_id}: JSONL contains {len(text_records)} "
            f"pages but PDF contains {len(pdf_metadata)}."
        )

    normalized_pages: list[dict[str, Any]] = []

    for text_record, page_metadata in zip(
        text_records,
        pdf_metadata,
        strict=True,
    ):
        pdf_page = text_record["page"]

        if pdf_page != page_metadata["pdf_page"]:
            raise ValueError(
                f"{doc_id}: page alignment mismatch "
                f"at physical page {pdf_page}."
            )

        text = text_record["text"]
        indexable = bool(text.strip())

        normalized_pages.append(
            {
                "doc_id": doc_id,
                "bank": bank,
                "fiscal_year": fiscal_year,
                "pdf_path": str(pdf_path.relative_to(PROJECT_ROOT)),
                "pdf_page": pdf_page,
                "report_page": page_metadata["report_page"],
                "page_label": page_metadata["page_label"],
                "text": text,
                "character_count": len(text),
                "indexable": indexable,
                "skip_reason": (
                    None if indexable else "empty_text"
                ),
            }
        )

    return normalized_pages


def read_split(split: str) -> list[str]:
    """Read one dataset split."""

    split_path = SPLITS_DIR / f"{split}.txt"

    if not split_path.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_path}"
        )

    doc_ids = [
        line.strip()
        for line in split_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip() and not line.startswith("#")
    ]

    if not doc_ids:
        raise ValueError(
            f"No documents found in {split_path}."
        )

    return doc_ids


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and normalize complete-report pages "
            "for the V2 semantic RAG pipeline."
        )
    )

    parser.add_argument(
        "--split",
        choices=["dev", "test", "temporal"],
        default="dev",
    )
    parser.add_argument(
        "--only-doc",
        help="Process only one document identifier.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    doc_ids = (
        [args.only_doc]
        if args.only_doc
        else read_split(args.split)
    )

    total_pages = 0
    total_indexable = 0
    total_skipped = 0

    for doc_id in doc_ids:
        pages = ingest_document(doc_id)

        indexable = sum(
            page["indexable"] for page in pages
        )
        skipped = len(pages) - indexable
        numeric_labels = sum(
            page["report_page"] is not None
            for page in pages
        )

        total_pages += len(pages)
        total_indexable += indexable
        total_skipped += skipped

        print(f"\nDocument: {doc_id}")
        print(f"Bank: {pages[0]['bank']}")
        print(f"Fiscal year: {pages[0]['fiscal_year']}")
        print(f"Total pages: {len(pages)}")
        print(f"Indexable pages: {indexable}")
        print(f"Skipped empty pages: {skipped}")
        print(f"Numeric PDF labels: {numeric_labels}")

    print("\n" + "=" * 60)
    print("Ingestion summary")
    print(f"Documents: {len(doc_ids)}")
    print(f"Total pages: {total_pages}")
    print(f"Indexable pages: {total_indexable}")
    print(f"Skipped empty pages: {total_skipped}")


if __name__ == "__main__":
    main()
