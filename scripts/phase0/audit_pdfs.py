#!/usr/bin/env python3
"""
Phase 0 PDF audit for Tunisian bank annual reports.

What it does:
- Scans every PDF in a folder
- Extracts text page by page
- Detects individual/consolidated scope terms
- Searches for the locked benchmark fields
- Exports a CSV inventory with matching page numbers
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import unicodedata
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Missing dependency: PyMuPDF")
    print("Install it with: pip install PyMuPDF")
    sys.exit(1)


FIELD_PATTERNS = {
    "total_assets": [
        r"\btotal\s+(?:du\s+)?bilan\b",
        r"\btotal\s+(?:de\s+l['’]\s*)?actif\b",
    ],
    "total_equity": [
        r"\bcapitaux\s+propres\b",
        r"\bfonds\s+propres\b",
        r"\bcapitaux\s+propres\s+avant\s+affectation\b",
    ],
    "net_banking_income": [
        r"\bproduit\s+net\s+bancaire\b",
        r"\bpnb\b",
    ],
    "operating_income": [
        r"\bresultat\s+d['’]?\s*exploitation\b",
        r"\bresultat\s+operationnel\b",
    ],
    "net_income": [
        r"\bresultat\s+net\b",
        r"\bresultat\s+de\s+l['’]?\s*exercice\b",
        r"\bbenefice\s+net\b",
    ],
    "customer_deposits": [
        r"\bdepots?\s+(?:et\s+avoirs?\s+)?(?:de\s+la\s+)?clientele\b",
        r"\bdepots?\s+clientele\b",
    ],
    "net_customer_loans": [
        r"\bcreances?\s+(?:nettes?\s+)?sur\s+la\s+clientele\b",
        r"\bcreances?\s+clientele\b",
        r"\bcredits?\s+(?:a|à)\s+la\s+clientele\b",
    ],
}

INDIVIDUAL_SCOPE_PATTERNS = [
    r"\betats?\s+financiers?\s+individuels?\b",
    r"\bcomptes?\s+individuels?\b",
    r"\bbilan\s+individuel\b",
    r"\betat\s+de\s+la\s+situation\s+financiere\s+individuelle\b",
]

CONSOLIDATED_SCOPE_PATTERNS = [
    r"\betats?\s+financiers?\s+consolides?\b",
    r"\bcomptes?\s+consolides?\b",
    r"\bbilan\s+consolide\b",
    r"\bpart\s+du\s+groupe\b",
]


def normalize(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text


def matching_pages(page_texts: list[str], patterns: list[str]) -> list[int]:
    compiled = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    pages: list[int] = []

    for index, text in enumerate(page_texts, start=1):
        if any(pattern.search(text) for pattern in compiled):
            pages.append(index)

    return pages


def infer_scope(individual_pages: list[int], consolidated_pages: list[int]) -> str:
    if individual_pages and consolidated_pages:
        return "both"
    if individual_pages:
        return "individual_only_detected"
    if consolidated_pages:
        return "consolidated_only_detected"
    return "not_detected"


def audit_pdf(pdf_path: Path) -> dict[str, str | int]:
    try:
        document = fitz.open(pdf_path)
    except Exception as exc:
        return {
            "filename": pdf_path.name,
            "status": f"ERROR: {exc}",
        }

    page_texts: list[str] = []
    text_pages = 0
    total_chars = 0

    for page in document:
        raw_text = page.get_text("text") or ""
        cleaned = normalize(raw_text)
        page_texts.append(cleaned)

        character_count = len(re.sub(r"\s+", "", cleaned))
        total_chars += character_count
        if character_count >= 50:
            text_pages += 1

    total_pages = len(document)
    document.close()

    searchable_ratio = text_pages / total_pages if total_pages else 0
    searchable = searchable_ratio >= 0.50 and total_chars >= 1000

    individual_pages = matching_pages(page_texts, INDIVIDUAL_SCOPE_PATTERNS)
    consolidated_pages = matching_pages(page_texts, CONSOLIDATED_SCOPE_PATTERNS)

    result: dict[str, str | int] = {
        "filename": pdf_path.name,
        "status": "OK",
        "total_pages": total_pages,
        "text_pages": text_pages,
        "searchable_ratio": f"{searchable_ratio:.2f}",
        "total_characters": total_chars,
        "text_extractable": "yes" if searchable else "no",
        "scope": infer_scope(individual_pages, consolidated_pages),
        "individual_scope_pages": ",".join(map(str, individual_pages)),
        "consolidated_scope_pages": ",".join(map(str, consolidated_pages)),
    }

    fields_found = 0
    for field_name, patterns in FIELD_PATTERNS.items():
        pages = matching_pages(page_texts, patterns)
        result[f"{field_name}_found"] = "yes" if pages else "no"
        result[f"{field_name}_pages"] = ",".join(map(str, pages))
        if pages:
            fields_found += 1

    result["fields_found"] = fields_found
    result["eligible_auto"] = (
        "yes"
        if searchable and fields_found >= 4 and bool(individual_pages)
        else "review"
    )

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pdf_folder",
        nargs="?",
        default="pdfs",
        help="Folder containing PDFs. Default: ./pdfs",
    )
    parser.add_argument(
        "--output",
        default="phase0_pdf_inventory.csv",
        help="CSV output path.",
    )
    args = parser.parse_args()

    pdf_folder = Path(args.pdf_folder)
    output_path = Path(args.output)

    if not pdf_folder.exists():
        print(f"Folder not found: {pdf_folder.resolve()}")
        sys.exit(1)

    pdf_files = sorted(pdf_folder.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in: {pdf_folder.resolve()}")
        sys.exit(1)

    rows = [audit_pdf(pdf_path) for pdf_path in pdf_files]

    all_columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in all_columns:
                all_columns.append(key)

    with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=all_columns)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Audited {len(pdf_files)} PDF(s).")
    print(f"Inventory created: {output_path.resolve()}")


if __name__ == "__main__":
    main()
