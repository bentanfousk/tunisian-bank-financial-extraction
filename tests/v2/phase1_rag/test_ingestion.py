import unittest
from pathlib import Path

from scripts.v2.phase1_rag.ingestion import (
    PROJECT_ROOT,
    ingest_document,
    parse_doc_id,
    read_split,
)


class TestV2Ingestion(unittest.TestCase):

    def test_parse_doc_id(self):
        bank, year = parse_doc_id("UIB_2024")

        self.assertEqual(bank, "UIB")
        self.assertEqual(year, 2024)

    def test_all_development_reports_load(self):
        for doc_id in read_split("dev"):
            with self.subTest(doc_id=doc_id):
                pages = ingest_document(doc_id)

                self.assertGreater(len(pages), 0)

                self.assertEqual(
                    [page["pdf_page"] for page in pages],
                    list(range(1, len(pages) + 1)),
                )

                for page in pages:
                    self.assertEqual(
                        page["doc_id"],
                        doc_id,
                    )
                    self.assertIn("bank", page)
                    self.assertIn("fiscal_year", page)
                    self.assertIn("pdf_page", page)
                    self.assertIn("report_page", page)
                    self.assertIn("text", page)
                    self.assertIn("indexable", page)

    def test_empty_pages_are_preserved(self):
        pages = ingest_document("BT_2024")

        empty_pages = [
            page
            for page in pages
            if not page["indexable"]
        ]

        self.assertEqual(len(pages), 220)
        self.assertEqual(len(empty_pages), 17)

        for page in empty_pages:
            self.assertEqual(
                page["skip_reason"],
                "empty_text",
            )

    def test_no_annotation_or_manifest_dependency(self):
        source_path = (
            PROJECT_ROOT
            / "scripts"
            / "v2"
            / "phase1_rag"
            / "ingestion.py"
        )

        source = source_path.read_text(
            encoding="utf-8"
        ).casefold()

        self.assertNotIn("data/annotations", source)
        self.assertNotIn("page_manifest.json", source)


if __name__ == "__main__":
    unittest.main()



