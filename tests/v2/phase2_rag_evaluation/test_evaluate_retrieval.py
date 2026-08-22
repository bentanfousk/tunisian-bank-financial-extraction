import unittest
from pathlib import Path

from scripts.v2.phase1_rag.ingestion import (
    PROJECT_ROOT,
)
from scripts.v2.phase2_rag_evaluation.evaluate_retrieval import (
    build_report_to_pdf_mapping,
)


class TestRetrievalEvaluation(unittest.TestCase):

    def test_report_to_pdf_mapping(self):
        document = {
            "scopes": {
                "individual": {
                    "statement_pages": {
                        "balance_sheet": {
                            "report_page": 140,
                            "pdf_page": 137,
                        },
                        "income_statement": {
                            "report_page": 142,
                            "pdf_page": 139,
                        },
                    }
                },
                "consolidated": None,
            }
        }

        mapping = (
            build_report_to_pdf_mapping(
                document
            )
        )

        self.assertEqual(
            mapping[140],
            137,
        )
        self.assertEqual(
            mapping[142],
            139,
        )

    def test_online_retrieval_has_no_ground_truth_dependency(
        self,
    ):
        online_files = [
            (
                PROJECT_ROOT
                / "scripts"
                / "v2"
                / "phase1_rag"
                / "build_index.py"
            ),
            (
                PROJECT_ROOT
                / "scripts"
                / "v2"
                / "phase1_rag"
                / "retrieve.py"
            ),
            (
                PROJECT_ROOT
                / "scripts"
                / "v2"
                / "phase1_rag"
                / "retrieval_queries.py"
            ),
        ]

        forbidden_terms = [
            "data/annotations",
            "page_manifest.json",
            "phase2_page_select",
        ]

        for path in online_files:
            source = path.read_text(
                encoding="utf-8"
            ).casefold()

            for forbidden_term in forbidden_terms:
                with self.subTest(
                    path=path,
                    term=forbidden_term,
                ):
                    self.assertNotIn(
                        forbidden_term,
                        source,
                    )
    def test_evaluator_uses_reranked_retrieval(
        self,
    ):
        path = (
            PROJECT_ROOT
            / "scripts"
            / "v2"
            / "phase2_rag_evaluation"
            / "evaluate_retrieval.py"
        )

        source = path.read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "retrieve_query(",
            source,
        )
        self.assertNotIn(
            "search_index(",
            source,
        )


if __name__ == "__main__":
    unittest.main()