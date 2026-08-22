import unittest

from scripts.phase6.compare_oracle import (
    classify_transition,
    compare_scores,
)


class TestCompareOracle(unittest.TestCase):

    def test_all_transition_types(self):
        self.assertEqual(
            classify_transition(True, True),
            "correct_both",
        )

        self.assertEqual(
            classify_transition(False, True),
            "context_failure",
        )

        self.assertEqual(
            classify_transition(False, False),
            "generation_failure",
        )

        self.assertEqual(
            classify_transition(True, False),
            "oracle_regression",
        )

    def test_compare_scores(self):
        pipeline = {
            "doc_id": "TEST_2024",
            "fields": [
                {
                    "field": "a",
                    "correct": True,
                },
                {
                    "field": "b",
                    "correct": False,
                },
                {
                    "field": "c",
                    "correct": False,
                },
                {
                    "field": "d",
                    "correct": True,
                },
            ],
            "summary": {
                "valid_json": True,
            },
        }

        oracle = {
            "doc_id": "TEST_2024",
            "fields": [
                {
                    "field": "a",
                    "correct": True,
                },
                {
                    "field": "b",
                    "correct": True,
                },
                {
                    "field": "c",
                    "correct": False,
                },
                {
                    "field": "d",
                    "correct": False,
                },
            ],
            "summary": {
                "valid_json": True,
            },
        }

        result = compare_scores(
            pipeline,
            oracle,
        )

        self.assertEqual(
            result["summary"],
            {
                "comparison_status": "both_valid",
                "field_comparison_available": True,
                "pipeline_valid_json": True,
                "oracle_valid_json": True,
                "total_fields": 4,
                "correct_both": 1,
                "context_failure": 1,
                "generation_failure": 1,
                "oracle_regression": 1,
            },
        )

    def test_pipeline_invalid_oracle_valid(self):
        pipeline = {
            "doc_id": "TEST_2024",
            "fields": [
                {
                    "field": "a",
                    "correct": False,
                },
            ],
            "summary": {
                "valid_json": False,
            },
        }

        oracle = {
            "doc_id": "TEST_2024",
            "fields": [
                {
                    "field": "a",
                    "correct": True,
                },
            ],
            "summary": {
                "valid_json": True,
            },
        }

        result = compare_scores(
            pipeline,
            oracle,
        )

        self.assertEqual(
            result["fields"],
            [],
        )

        self.assertEqual(
            result["summary"][
                "comparison_status"
            ],
            "pipeline_invalid_oracle_valid",
        )

        self.assertFalse(
            result["summary"][
                "field_comparison_available"
            ]
        )

        self.assertIsNone(
            result["summary"][
                "context_failure"
            ]
        )

    def test_both_invalid(self):
        pipeline = {
            "doc_id": "TEST_2024",
            "fields": [],
            "summary": {
                "valid_json": False,
            },
        }

        oracle = {
            "doc_id": "TEST_2024",
            "fields": [],
            "summary": {
                "valid_json": False,
            },
        }

        result = compare_scores(
            pipeline,
            oracle,
        )

        self.assertEqual(
            result["summary"][
                "comparison_status"
            ],
            "both_invalid",
        )

        self.assertFalse(
            result["summary"][
                "field_comparison_available"
            ]
        )

    def test_pipeline_valid_oracle_invalid(self):
        pipeline = {
            "doc_id": "TEST_2024",
            "fields": [],
            "summary": {
                "valid_json": True,
            },
        }

        oracle = {
            "doc_id": "TEST_2024",
            "fields": [],
            "summary": {
                "valid_json": False,
            },
        }

        result = compare_scores(
            pipeline,
            oracle,
        )

        self.assertEqual(
            result["summary"][
                "comparison_status"
            ],
            "pipeline_valid_oracle_invalid",
        )

    def test_document_mismatch_fails(self):
        pipeline = {
            "doc_id": "A_2024",
            "fields": [],
            "summary": {
                "valid_json": True,
            },
        }

        oracle = {
            "doc_id": "B_2024",
            "fields": [],
            "summary": {
                "valid_json": True,
            },
        }

        with self.assertRaises(ValueError):
            compare_scores(
                pipeline,
                oracle,
            )


if __name__ == "__main__":
    unittest.main()