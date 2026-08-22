from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "manifests"
    / "page_manifest.json"
)

REPORTS_TEXT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "reports_txt"
)

STATEMENT_ORDER = [
    "balance_sheet",
    "off_balance_sheet",
    "income_statement",
    "cash_flow_statement",
]


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""

    if not path.exists():
        raise FileNotFoundError(
            f"JSON file not found: {path}"
        )

    try:
        with path.open("r", encoding="utf-8") as json_file:
            data = json.load(json_file)

    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in {path} at line "
            f"{error.lineno}, column {error.colno}: "
            f"{error.msg}"
        ) from error

    if not isinstance(data, dict):
        raise TypeError(
            f"Expected a JSON object in {path}."
        )

    return data


def load_report_pages(
    jsonl_path: Path,
    expected_doc_id: str,
) -> dict[int, str]:
    """
    Load one report JSONL.

    Important:
        record["page"] is the 1-based physical PDF page.

    Returns:
        {
            pdf_page: text,
            ...
        }
    """

    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Extracted report not found: {jsonl_path}"
        )

    pages: dict[int, str] = {}

    with jsonl_path.open(
        "r",
        encoding="utf-8",
    ) as input_file:

        for line_number, raw_line in enumerate(
            input_file,
            start=1,
        ):
            if not raw_line.strip():
                raise ValueError(
                    f"{jsonl_path.name}: blank JSONL line "
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
                    f"{jsonl_path.name}: line "
                    f"{line_number} is not a JSON object."
                )

            record_doc_id = record.get("doc_id")
            pdf_page = record.get("page")
            text = record.get("text")

            if record_doc_id != expected_doc_id:
                raise ValueError(
                    f"{jsonl_path.name}: line "
                    f"{line_number} has doc_id "
                    f"{record_doc_id!r}, expected "
                    f"{expected_doc_id!r}."
                )

            if (
                not isinstance(pdf_page, int)
                or pdf_page < 1
            ):
                raise ValueError(
                    f"{jsonl_path.name}: line "
                    f"{line_number} has invalid "
                    f"physical PDF page {pdf_page!r}."
                )

            if not isinstance(text, str):
                raise TypeError(
                    f"{jsonl_path.name}: line "
                    f"{line_number} has invalid text."
                )

            if pdf_page in pages:
                raise ValueError(
                    f"{jsonl_path.name}: duplicate "
                    f"physical PDF page {pdf_page}."
                )

            pages[pdf_page] = text

    if not pages:
        raise ValueError(
            f"{jsonl_path.name}: no pages loaded."
        )

    return pages


def get_document_entry(
    manifest: dict[str, Any],
    doc_id: str,
) -> dict[str, Any]:
    """Return one document entry from manifest v2."""

    if manifest.get("manifest_version") != 2:
        raise ValueError(
            "page_manifest.json must use "
            "manifest_version 2."
        )

    documents = manifest.get("documents")

    if not isinstance(documents, dict):
        raise TypeError(
            "Manifest is missing a valid "
            "'documents' object."
        )

    document = documents.get(doc_id)

    if not isinstance(document, dict):
        raise KeyError(
            f"No manifest entry found for "
            f"doc_id {doc_id!r}."
        )

    return document


def get_page_refs(
    document: dict[str, Any],
    doc_id: str,
    scope: str = "individual",
    core_only: bool = False,
) -> list[dict[str, Any]]:
    """
    Return verified statement page references.

    Each reference contains:
        statement_type
        pdf_page
        report_page
    """

    verification = document.get("verification")

    if not isinstance(verification, dict):
        raise ValueError(
            f"{doc_id}: missing verification object."
        )

    if verification.get(scope) != "manually_verified":
        raise ValueError(
            f"{doc_id}: scope {scope!r} is not "
            "manually verified."
        )

    scopes = document.get("scopes")

    if not isinstance(scopes, dict):
        raise ValueError(
            f"{doc_id}: missing scopes object."
        )

    scope_entry = scopes.get(scope)

    if not isinstance(scope_entry, dict):
        raise ValueError(
            f"{doc_id}: scope {scope!r} is unavailable."
        )

    statement_pages = scope_entry.get(
        "statement_pages"
    )

    if not isinstance(statement_pages, dict):
        raise ValueError(
            f"{doc_id}: missing statement_pages "
            f"for scope {scope!r}."
        )

    if core_only:
        selected_types = scope_entry.get(
            "core_statement_types"
        )

        if (
            not isinstance(selected_types, list)
            or not selected_types
        ):
            raise ValueError(
                f"{doc_id}: missing "
                "core_statement_types."
            )

    else:
        selected_types = [
            statement_type
            for statement_type in STATEMENT_ORDER
            if statement_type in statement_pages
        ]

    page_refs: list[dict[str, Any]] = []

    for statement_type in selected_types:
        page_ref = statement_pages.get(
            statement_type
        )

        if not isinstance(page_ref, dict):
            raise ValueError(
                f"{doc_id}: missing page reference "
                f"for {statement_type!r}."
            )

        pdf_page = page_ref.get("pdf_page")
        report_page = page_ref.get("report_page")

        if (
            not isinstance(pdf_page, int)
            or pdf_page < 1
        ):
            raise ValueError(
                f"{doc_id}: invalid pdf_page for "
                f"{statement_type!r}."
            )

        if (
            not isinstance(report_page, int)
            or report_page < 1
        ):
            raise ValueError(
                f"{doc_id}: invalid report_page for "
                f"{statement_type!r}."
            )

        page_refs.append(
            {
                "statement_type": statement_type,
                "pdf_page": pdf_page,
                "report_page": report_page,
            }
        )

    return page_refs


def select_pages(
    doc_id: str,
    core_only: bool = False,
) -> dict[str, Any]:
    """
    Load the manually verified individual statement pages.

    The manifest's pdf_page is used to retrieve text.

    The manifest's report_page is used in the text header
    shown to the model and therefore in model citations.

    This function performs no automatic retrieval,
    keyword search, ranking, or page scoring.
    """

    manifest = load_json(MANIFEST_PATH)

    document = get_document_entry(
        manifest=manifest,
        doc_id=doc_id,
    )

    page_refs = get_page_refs(
        document=document,
        doc_id=doc_id,
        scope="individual",
        core_only=core_only,
    )

    unique_page_refs: list[dict[str, Any]] = []
    seen_page_refs: set[tuple[int, int]] = set()

    for page_ref in page_refs:
        key = (
            page_ref["pdf_page"],
            page_ref["report_page"],
        )

        if key in seen_page_refs:
            continue

        seen_page_refs.add(key)
        unique_page_refs.append(page_ref)

    report_jsonl_path = (
        REPORTS_TEXT_DIR / f"{doc_id}.jsonl"
    )

    report_pages = load_report_pages(
        jsonl_path=report_jsonl_path,
        expected_doc_id=doc_id,
    )

    missing_pdf_pages = [
        page_ref["pdf_page"]
        for page_ref in unique_page_refs
        if page_ref["pdf_page"] not in report_pages
    ]

    if missing_pdf_pages:
        raise ValueError(
            f"{doc_id}: physical PDF/JSONL pages "
            f"{missing_pdf_pages} are missing from "
            f"{report_jsonl_path.name}."
        )

    page_blocks: list[str] = []

    for page_ref in unique_page_refs:
        pdf_page = page_ref["pdf_page"]
        report_page = page_ref["report_page"]

        page_text = report_pages[
            pdf_page
        ].strip()

        page_block = (
            f"===== REPORT_PAGE {report_page} =====\n"
            f"{page_text}"
        )

        page_blocks.append(page_block)

    combined_text = "\n\n".join(page_blocks)

    return {
        "doc_id": doc_id,
        "page_set": (
            "core"
            if core_only
            else "full_statement_block"
        ),
        "report_pages": [
            page_ref["report_page"]
            for page_ref in unique_page_refs
        ],
        "pdf_pages": [
            page_ref["pdf_page"]
            for page_ref in unique_page_refs
        ],
        "statement_types": [
            page_ref["statement_type"]
            for page_ref in page_refs
        ],
        "page_count": len(unique_page_refs),
        "text": combined_text,
        "character_count": len(combined_text),
    }


def resolve_output_path(path: Path) -> Path:
    """Resolve output path relative to project root."""

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load the manually verified individual "
            "financial-statement pages for one report."
        )
    )

    parser.add_argument(
        "doc_id",
        help=(
            "Report identifier, e.g. STB_2024, "
            "UIB_2024, UBCI_2024."
        ),
    )

    parser.add_argument(
        "--core-only",
        action="store_true",
        help=(
            "Load only the statement types containing "
            "the seven benchmark fields."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Optional path in which to save the "
            "selected text."
        ),
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Print all selected text.",
    )

    parser.add_argument(
        "--preview-chars",
        type=int,
        default=1500,
        help=(
            "Preview length when --show is not used. "
            "Default: 1500."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if args.preview_chars < 0:
        raise ValueError(
            "--preview-chars cannot be negative."
        )

    selected = select_pages(
        doc_id=args.doc_id,
        core_only=args.core_only,
    )

    print(f"Document: {selected['doc_id']}")
    print(f"Page set: {selected['page_set']}")
    print(
        f"Statement types: "
        f"{selected['statement_types']}"
    )
    print(
        f"Report pages: "
        f"{selected['report_pages']}"
    )
    print(
        f"Physical PDF/JSONL pages: "
        f"{selected['pdf_pages']}"
    )
    print(
        f"Page count: "
        f"{selected['page_count']}"
    )
    print(
        f"Character count: "
        f"{selected['character_count']}"
    )

    if args.output is not None:
        output_path = resolve_output_path(
            args.output
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            selected["text"],
            encoding="utf-8",
        )

        print(f"Output: {output_path}")

    if args.show:
        print("\n" + selected["text"])

    elif args.preview_chars > 0:
        preview = selected["text"][
            :args.preview_chars
        ]

        print("\nPreview:\n")
        print(preview)

        if (
            len(selected["text"])
            > args.preview_chars
        ):
            print(
                "\n... preview truncated ..."
            )


if __name__ == "__main__":
    main()