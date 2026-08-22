from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pymupdf


FOOTER_START_RATIO = 0.90


def extract_visible_report_page(
    page: pymupdf.Page,
) -> int | None:
    """
    Read an isolated integer from the PDF footer.

    This uses only PDF layout metadata. It does not use
    annotations, the V1 page manifest, or bank-specific
    mappings.
    """

    footer_start = (
        page.rect.height * FOOTER_START_RATIO
    )

    candidates: set[int] = set()

    for block in page.get_text("blocks"):
        y0 = float(block[1])
        text = str(block[4]).strip()

        if y0 < footer_start:
            continue

        match = re.fullmatch(
            r"\s*(\d{1,4})\s*",
            text,
        )

        if match is None:
            continue

        value = int(match.group(1))

        if value >= 1:
            candidates.add(value)

    if len(candidates) == 1:
        return next(iter(candidates))

    return None


def resolve_report_page(
    *,
    page: pymupdf.Page,
    metadata_report_page: Any,
) -> tuple[int | None, str | None]:
    """
    Prefer existing preprocessing metadata. Otherwise,
    recover the visible footer number deterministically.
    """

    if (
        isinstance(metadata_report_page, int)
        and metadata_report_page >= 1
    ):
        return (
            metadata_report_page,
            "preprocessing_metadata",
        )

    raw_label = page.get_label()

    embedded_page = (
        int(raw_label)
        if isinstance(raw_label, str)
        and raw_label.strip().isdecimal()
        else None
    )

    footer_page = extract_visible_report_page(
        page
    )

    if (
        embedded_page is not None
        and footer_page is not None
        and embedded_page != footer_page
    ):
        raise ValueError(
            "Conflicting embedded and visible report "
            f"page numbers: {embedded_page} and "
            f"{footer_page}."
        )

    if embedded_page is not None:
        return embedded_page, "embedded_pdf_label"

    if footer_page is not None:
        return footer_page, "visible_footer_integer"

    return None, None


def build_retrieved_context(
    *,
    pdf_path: Path,
    merged_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Construct deterministic extraction context in frozen
    merged-rank order.

    Physical PDF pages are preserved in metadata but are
    never relabeled as REPORT_PAGE values.
    """

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    context_sections: list[str] = []
    page_records: list[dict[str, Any]] = []
    unresolved_pdf_pages: list[int] = []

    with pymupdf.open(pdf_path) as document:
        for result in merged_results:
            pdf_page = result.get("pdf_page")
            text = result.get("text")

            if (
                not isinstance(pdf_page, int)
                or pdf_page < 1
                or pdf_page > document.page_count
            ):
                raise ValueError(
                    f"Invalid retrieved PDF page: "
                    f"{pdf_page!r}."
                )

            if not isinstance(text, str):
                raise TypeError(
                    "Retrieved page text must be a string."
                )

            pdf_page_object = document[
                pdf_page - 1
            ]

            (
                report_page,
                report_page_source,
            ) = resolve_report_page(
                page=pdf_page_object,
                metadata_report_page=result.get(
                    "report_page"
                ),
            )

            if report_page is None:
                marker = (
                    "[REPORT_PAGE_UNAVAILABLE "
                    f"PDF_PAGE={pdf_page}]"
                )
                unresolved_pdf_pages.append(pdf_page)
            else:
                marker = (
                    f"[REPORT_PAGE {report_page}]"
                )

            context_sections.append(
                f"{marker}\n{text.strip()}"
            )

            page_records.append(
                {
                    "merged_rank": result.get(
                        "merged_rank"
                    ),
                    "chunk_id": result.get(
                        "chunk_id"
                    ),
                    "pdf_page": pdf_page,
                    "report_page": report_page,
                    "report_page_source": (
                        report_page_source
                    ),
                    "marker": marker,
                    "character_count": len(text),
                }
            )

    context = "\n\n".join(context_sections)

    return {
        "text": context,
        "pages": page_records,
        "retrieved_pdf_pages": [
            page["pdf_page"]
            for page in page_records
        ],
        "retrieved_report_pages": [
            page["report_page"]
            for page in page_records
            if page["report_page"] is not None
        ],
        "unresolved_pdf_pages": (
            unresolved_pdf_pages
        ),
        "character_count": len(context),
    }