import inspect
import unittest

from scripts.v2.phase3_rag_benchmark.run_benchmark import (
    EXPECTED_TEST_DOCUMENTS,
    MODEL_KEY,
    MODEL_TAG,
    REQUIRED_COMPLETED_ARTIFACTS,
    attach_faiss_positions,
    run_online_pipeline,
    validate_execution_request,
)


class TestPhase3Runner(unittest.TestCase):

    def test_frozen_test_roster(self):
        self.assertEqual(
            EXPECTED_TEST_DOCUMENTS,
            [
                "amen_2024",
                "ATB_2023",
                "attijari_2024",
                "bh_2024",
                "bna_2024",
                "BT_2024",
            ],
        )

    def test_only_frozen_ministral_is_used(self):
        self.assertEqual(
            MODEL_KEY,
            "ministral3_3b",
        )
        self.assertEqual(
            MODEL_TAG,
            "ministral-3:"
            "3b-instruct-2512-q4_K_M",
        )

    def test_test_split_rejects_single_report(
        self,
    ):
        with self.assertRaises(ValueError):
            validate_execution_request(
                split="test",
                only_doc="amen_2024",
                force=False,
            )

    def test_test_split_rejects_force(self):
        with self.assertRaises(ValueError):
            validate_execution_request(
                split="test",
                only_doc=None,
                force=True,
            )

    def test_online_pipeline_has_no_ground_truth_access(
        self,
    ):
        source = inspect.getsource(
            run_online_pipeline
        ).casefold()

        for forbidden in (
            "annotation",
            "page_manifest",
            "phase2_page_select",
            "score_prediction",
        ):
            with self.subTest(
                forbidden=forbidden
            ):
                self.assertNotIn(
                    forbidden,
                    source,
                )

    def test_required_completed_artifact_names(
        self,
    ):
        self.assertEqual(
            REQUIRED_COMPLETED_ARTIFACTS,
            (
                "retrieval.json",
                "retrieved_context.txt",
                "raw_response.txt",
                "validation.json",
                "score.json",
                "retrieval_evaluation.json",
                "run_meta.json",
            ),
        )
    def test_in_memory_chunks_receive_faiss_positions(
        self,
    ):
        chunks = [
            {
                "chunk_id": "doc::pdf_page::1",
            },
            {
                "chunk_id": "doc::pdf_page::2",
            },
        ]

        positioned = attach_faiss_positions(
            chunks
        )

        self.assertEqual(
            [
                chunk["faiss_position"]
                for chunk in positioned
            ],
            [0, 1],
        )

        self.assertEqual(
            positioned[0]["chunk_id"],
            "doc::pdf_page::1",
        )


if __name__ == "__main__":
    unittest.main()