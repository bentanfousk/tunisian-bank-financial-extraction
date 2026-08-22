import hashlib
import unittest
from pathlib import Path


from scripts.benchmark_versions import (
    PROJECT_ROOT,
    get_benchmark_version,
)
from scripts.phase4.prompt_builder import (
    build_messages,
)
from scripts.phase4.run_extract import (
    build_artifact_hashes,
)


EXPECTED_FIELDS = [
    "total_assets",
    "total_equity",
    "net_banking_income",
    "operating_income",
    "net_income",
    "customer_deposits",
    "net_customer_loans",
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


class TestBenchmarkVersioning(
    unittest.TestCase
):

    def test_prompt_version_selection(self):
        v1 = get_benchmark_version(
            "v1"
        )
        v11 = get_benchmark_version(
            "v1_1"
        )

        self.assertEqual(
            v1.system_prompt_path.name,
            "extract_v1.txt",
        )
        self.assertEqual(
            v11.system_prompt_path.name,
            "extract_v1_1.txt",
        )

        v1_messages = build_messages(
            doc_id="UIB_2024",
            bank="UIB",
            target_year=2024,
            context="TEST",
            prompt_version="v1",
        )

        v11_messages = build_messages(
            doc_id="UIB_2024",
            bank="UIB",
            target_year=2024,
            context="TEST",
            prompt_version="v1_1",
        )

        self.assertEqual(
            v1_messages[0]["content"],
            v1.system_prompt_path.read_text(
                encoding="utf-8"
            ),
        )

        self.assertEqual(
            v11_messages[0]["content"],
            v11.system_prompt_path.read_text(
                encoding="utf-8"
            ),
        )

        self.assertNotEqual(
            v1_messages[0]["content"],
            v11_messages[0]["content"],
        )

    def test_unknown_version_rejected(self):
        with self.assertRaises(
            ValueError
        ):
            get_benchmark_version(
                "banana"
            )

    def test_original_prompt_immutable(self):
        prompt_path = (
            PROJECT_ROOT
            / "prompts"
            / "extract_v1.txt"
        )

        hash_path = (
            PROJECT_ROOT
            / "prompts"
            / "extract_v1.sha256"
        )

        expected = (
            hash_path
            .read_text(
                encoding="ascii"
            )
            .strip()
            .lower()
        )

        actual = sha256_file(
            prompt_path
        )

        self.assertEqual(
            actual,
            expected,
        )

    def test_artifact_namespaces_are_separate(self):
        v1 = get_benchmark_version(
            "v1"
        )
        v11 = get_benchmark_version(
            "v1_1"
        )

        self.assertEqual(
            v1.phase7_output_root,
            PROJECT_ROOT
            / "artifacts"
            / "phase7",
        )

        self.assertEqual(
            v11.phase7_output_root,
            PROJECT_ROOT
            / "artifacts"
            / "phase7_v1_1",
        )

        self.assertNotEqual(
            v1.phase7_output_root,
            v11.phase7_output_root,
        )

        self.assertNotEqual(
            v1.mlflow_experiment,
            v11.mlflow_experiment,
        )

    def test_selected_prompt_hash_is_recorded(self):
        v1 = get_benchmark_version(
            "v1"
        )
        v11 = get_benchmark_version(
            "v1_1"
        )

        hashes_v1 = (
            build_artifact_hashes(
                "v1"
            )
        )
        hashes_v11 = (
            build_artifact_hashes(
                "v1_1"
            )
        )

        self.assertEqual(
            hashes_v1[
                "system_prompt_sha256"
            ],
            sha256_file(
                v1.system_prompt_path
            ),
        )

        self.assertEqual(
            hashes_v11[
                "system_prompt_sha256"
            ],
            sha256_file(
                v11.system_prompt_path
            ),
        )

        self.assertNotEqual(
            hashes_v1[
                "system_prompt_sha256"
            ],
            hashes_v11[
                "system_prompt_sha256"
            ],
        )

    def test_v11_contains_all_fields_in_order(self):
        prompt = (
            get_benchmark_version(
                "v1_1"
            )
            .system_prompt_path
            .read_text(
                encoding="utf-8"
            )
        )

        output_section = prompt.split(
            "Retourne exactement une entrée "
            "pour chacun des 7 champs",
            1,
        )[1]

        positions = [
            output_section.index(
                field
            )
            for field in EXPECTED_FIELDS
        ]

        self.assertEqual(
            positions,
            sorted(positions),
        )

        for field in EXPECTED_FIELDS:
            self.assertIn(
                field,
                prompt,
            )

    def test_v11_has_customer_loan_clarification(self):
        prompt = (
            get_benchmark_version(
                "v1_1"
            )
            .system_prompt_path
            .read_text(
                encoding="utf-8"
            )
        )

        self.assertIn(
            "Créances sur la clientèle",
            prompt,
        )

        self.assertIn(
            "AC 3 - Créances sur la clientèle",
            prompt,
        )

        self.assertIn(
            "ne contient pas les mots "
            "« net », « nette » ou « nettes »",
            prompt,
        )

    def test_online_extraction_has_no_annotation_dependency(
        self,
    ):
        source_paths = [
            PROJECT_ROOT
            / "scripts"
            / "phase4"
            / "prompt_builder.py",

            PROJECT_ROOT
            / "scripts"
            / "phase4"
            / "run_extract.py",

            PROJECT_ROOT
            / "scripts"
            / "phase2_page_select"
            / "page_select.py",
        ]

        for path in source_paths:
            source = path.read_text(
                encoding="utf-8"
            ).lower()

            self.assertNotIn(
                "annotations_dir",
                source,
                msg=str(path),
            )

            self.assertNotIn(
                "data/annotations",
                source.replace("\\", "/"),
                msg=str(path),
            )


if __name__ == "__main__":
    unittest.main()