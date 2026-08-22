import json
import tempfile
import unittest
from pathlib import Path

from scripts.phase6.oracle_run import (
    build_oracle_selection,
    get_oracle_report_pages,
)


class TestOracleRun(unittest.TestCase):

    def test_oracle_pages_are_unique_and_sorted(
        self,
    ):
        annotation = {
            "fields": [
                {
                    "field": "a",
                    "status": "found",
                    "page": 142,
                },
                {
                    "field": "b",
                    "status": "found",
                    "page": 140,
                },
                {
                    "field": "c",
                    "status": "found",
                    "page": 142,
                },
                {
                    "field": "d",
                    "status": "not_found",
                    "page": None,
                },
            ]
        }

        pages = get_oracle_report_pages(
            annotation
        )

        self.assertEqual(
            pages,
            [140, 142],
        )

    def test_oracle_selection_uses_only_pages(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            annotation_path = (
                root / "TEST_2024.json"
            )

            manifest_path = (
                root / "page_manifest.json"
            )

            reports_dir = (
                root / "reports_txt"
            )

            reports_dir.mkdir()

            annotation = {
                "doc_id": "TEST_2024",
                "fields": [
                    {
                        "field": "total_assets",
                        "status": "found",
                        "page": 100,
                        "value": 123456,
                        "evidence": (
                            "SECRET_ANNOTATION_TEXT"
                        ),
                        "distractors": [
                            {
                                "value": 999999,
                                "reason": (
                                    "SECRET_DISTRACTOR"
                                ),
                            }
                        ],
                    },
                    {
                        "field": (
                            "net_banking_income"
                        ),
                        "status": "found",
                        "page": 102,
                        "value": 654321,
                    },
                ],
            }

            manifest = {
                "manifest_version": 2,
                "documents": {
                    "TEST_2024": {
                        "verification": {
                            "individual": (
                                "manually_verified"
                            )
                        },
                        "scopes": {
                            "individual": {
                                "statement_pages": {
                                    "balance_sheet": {
                                        "pdf_page": 10,
                                        "report_page": 100,
                                    },
                                    "off_balance_sheet": {
                                        "pdf_page": 11,
                                        "report_page": 101,
                                    },
                                    "income_statement": {
                                        "pdf_page": 12,
                                        "report_page": 102,
                                    },
                                },
                                "core_statement_types": [
                                    "balance_sheet",
                                    "income_statement",
                                ],
                            }
                        },
                    }
                },
            }

            annotation_path.write_text(
                json.dumps(annotation),
                encoding="utf-8",
            )

            manifest_path.write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )

            records = [
                {
                    "doc_id": "TEST_2024",
                    "page": 10,
                    "text": "BALANCE PAGE",
                },
                {
                    "doc_id": "TEST_2024",
                    "page": 11,
                    "text": (
                        "PAGE THAT MUST NOT APPEAR"
                    ),
                },
                {
                    "doc_id": "TEST_2024",
                    "page": 12,
                    "text": "INCOME PAGE",
                },
            ]

            report_path = (
                reports_dir
                / "TEST_2024.jsonl"
            )

            with report_path.open(
                "w",
                encoding="utf-8",
            ) as file:
                for record in records:
                    file.write(
                        json.dumps(record)
                        + "\n"
                    )

            selection = (
                build_oracle_selection(
                    doc_id="TEST_2024",
                    annotation_path=(
                        annotation_path
                    ),
                    manifest_path=(
                        manifest_path
                    ),
                    reports_text_dir=(
                        reports_dir
                    ),
                )
            )

            self.assertEqual(
                selection["page_set"],
                "oracle_pages",
            )

            self.assertEqual(
                selection["report_pages"],
                [100, 102],
            )

            self.assertEqual(
                selection["pdf_pages"],
                [10, 12],
            )

            self.assertEqual(
                selection["page_count"],
                2,
            )

            self.assertIn(
                "BALANCE PAGE",
                selection["text"],
            )

            self.assertIn(
                "INCOME PAGE",
                selection["text"],
            )

            self.assertNotIn(
                "PAGE THAT MUST NOT APPEAR",
                selection["text"],
            )

            # Annotation metadata must not leak
            # directly into the prompt context.
            self.assertNotIn(
                "SECRET_ANNOTATION_TEXT",
                selection["text"],
            )

            self.assertNotIn(
                "SECRET_DISTRACTOR",
                selection["text"],
            )

    def test_unmapped_oracle_page_fails(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            annotation_path = (
                root / "TEST_2024.json"
            )

            manifest_path = (
                root / "page_manifest.json"
            )

            reports_dir = (
                root / "reports_txt"
            )

            reports_dir.mkdir()

            annotation_path.write_text(
                json.dumps(
                    {
                        "doc_id": "TEST_2024",
                        "fields": [
                            {
                                "field": (
                                    "total_assets"
                                ),
                                "status": "found",
                                "page": 999,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            manifest_path.write_text(
                json.dumps(
                    {
                        "manifest_version": 2,
                        "documents": {
                            "TEST_2024": {
                                "verification": {
                                    "individual": (
                                        "manually_verified"
                                    )
                                },
                                "scopes": {
                                    "individual": {
                                        "statement_pages": {
                                            "balance_sheet": {
                                                "pdf_page": 10,
                                                "report_page": 100,
                                            }
                                        },
                                        "core_statement_types": [
                                            "balance_sheet"
                                        ],
                                    }
                                },
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            (
                reports_dir
                / "TEST_2024.jsonl"
            ).write_text(
                json.dumps(
                    {
                        "doc_id": "TEST_2024",
                        "page": 10,
                        "text": "TEST",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(
                ValueError
            ):
                build_oracle_selection(
                    doc_id="TEST_2024",
                    annotation_path=(
                        annotation_path
                    ),
                    manifest_path=(
                        manifest_path
                    ),
                    reports_text_dir=(
                        reports_dir
                    ),
                )


if __name__ == "__main__":
    unittest.main()