import unittest
from pathlib import Path

from scripts.v2.phase3_rag_benchmark.frozen_config import (
    PROJECT_ROOT,
)


class TestPhase3OnlineLeakage(unittest.TestCase):

    def test_online_foundation_has_no_ground_truth_dependency(
        self,
    ):
        files = [
            (
                PROJECT_ROOT
                / "scripts"
                / "v2"
                / "phase3_rag_benchmark"
                / "frozen_config.py"
            ),
            (
                PROJECT_ROOT
                / "scripts"
                / "v2"
                / "phase3_rag_benchmark"
                / "context_builder.py"
            ),
        ]

        forbidden = [
            "data/annotations",
            "page_manifest.json",
            "phase2_page_select",
        ]

        for path in files:
            source = path.read_text(
                encoding="utf-8"
            ).casefold()

            for term in forbidden:
                with self.subTest(
                    file=path.name,
                    term=term,
                ):
                    self.assertNotIn(
                        term,
                        source,
                    )


if __name__ == "__main__":
    unittest.main()