import tempfile
import unittest
from pathlib import Path

import pymupdf

from scripts.v2.phase3_rag_benchmark.context_builder import (
    build_retrieved_context,
    extract_visible_report_page,
)


class TestPhase3ContextBuilder(unittest.TestCase):

    def test_footer_report_page_is_recovered(self):
        with pymupdf.open() as document:
            page = document.new_page(
                width=595,
                height=842,
            )
            page.insert_text(
                (520, 805),
                "140",
            )

            self.assertEqual(
                extract_visible_report_page(page),
                140,
            )

    def test_body_integer_is_not_a_page_marker(self):
        with pymupdf.open() as document:
            page = document.new_page(
                width=595,
                height=842,
            )
            page.insert_text(
                (300, 400),
                "140",
            )

            self.assertIsNone(
                extract_visible_report_page(page)
            )

    def test_context_uses_report_page_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            pdf_path = Path(directory) / "report.pdf"

            with pymupdf.open() as document:
                page = document.new_page(
                    width=595,
                    height=842,
                )
                page.insert_text(
                    (520, 805),
                    "140",
                )
                document.save(pdf_path)

            result = build_retrieved_context(
                pdf_path=pdf_path,
                merged_results=[
                    {
                        "merged_rank": 1,
                        "chunk_id": "doc::pdf_page::1",
                        "pdf_page": 1,
                        "report_page": None,
                        "text": "TOTAL ACTIF 8 235 520",
                    }
                ],
            )

            self.assertTrue(
                result["text"].startswith(
                    "[REPORT_PAGE 140]"
                )
            )
            self.assertEqual(
                result["pages"][0]["pdf_page"],
                1,
            )
            self.assertEqual(
                result["pages"][0]["report_page"],
                140,
            )
            self.assertEqual(
                result["unresolved_pdf_pages"],
                [],
            )

    def test_physical_page_is_not_mislabeled(self):
        with tempfile.TemporaryDirectory() as directory:
            pdf_path = Path(directory) / "report.pdf"

            with pymupdf.open() as document:
                document.new_page(
                    width=595,
                    height=842,
                )
                document.save(pdf_path)

            result = build_retrieved_context(
                pdf_path=pdf_path,
                merged_results=[
                    {
                        "merged_rank": 1,
                        "chunk_id": "doc::pdf_page::1",
                        "pdf_page": 1,
                        "report_page": None,
                        "text": "Financial text",
                    }
                ],
            )

            self.assertIn(
                "[REPORT_PAGE_UNAVAILABLE PDF_PAGE=1]",
                result["text"],
            )
            self.assertNotIn(
                "[REPORT_PAGE 1]",
                result["text"],
            )


if __name__ == "__main__":
    unittest.main()