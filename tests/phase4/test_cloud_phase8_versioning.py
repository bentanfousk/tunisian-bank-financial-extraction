import hashlib
import unittest
from pathlib import Path

from scripts.benchmark_versions import (
    PROJECT_ROOT,
    get_benchmark_version,
)
from scripts.phase7.run_cloud_baseline import (
    build_cloud_artifact_hashes,
    get_cloud_output_root,
)
from scripts.phase8.analyze import (
    DEFAULT_CLOUD_ROOT,
    DEFAULT_LOCAL_ROOT,
    DEFAULT_OUTPUT_DIR,
    resolve_prompt_provenance,
    validate_prompt_provenance,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


class TestCloudBenchmarkVersioning(
    unittest.TestCase
):

    def test_gemini_namespaces_follow_prompt_version(
        self,
    ):
        self.assertEqual(
            get_cloud_output_root("v1"),
            PROJECT_ROOT
            / "artifacts"
            / "phase7"
            / "cloud",
        )
        self.assertEqual(
            get_cloud_output_root("v1_1"),
            PROJECT_ROOT
            / "artifacts"
            / "phase7_v1_1"
            / "cloud",
        )

    def test_gemini_selected_prompt_hash_matches(
        self,
    ):
        for prompt_version in (
            "v1",
            "v1_1",
        ):
            with self.subTest(
                prompt_version=prompt_version
            ):
                version = get_benchmark_version(
                    prompt_version
                )
                hashes = (
                    build_cloud_artifact_hashes(
                        prompt_version
                    )
                )
                self.assertEqual(
                    hashes[
                        "system_prompt_sha256"
                    ],
                    sha256_file(
                        version.system_prompt_path
                    ),
                )


class TestPhase8PromptProvenance(
    unittest.TestCase
):

    def setUp(self):
        self.v1 = get_benchmark_version(
            "v1"
        )
        self.v11 = get_benchmark_version(
            "v1_1"
        )
        self.v1_sha = sha256_file(
            self.v1.system_prompt_path
        )
        self.v11_sha = sha256_file(
            self.v11.system_prompt_path
        )

    @staticmethod
    def provenance(
        prompt_version: str,
        prompt_sha: str,
    ) -> dict[str, str]:
        return {
            "prompt_version": prompt_version,
            "system_prompt_sha256": prompt_sha,
        }

    def test_rejects_v1_local_with_v11_cloud(
        self,
    ):
        with self.assertRaisesRegex(
            ValueError,
            "prompt version mismatch",
        ):
            validate_prompt_provenance(
                [
                    self.provenance(
                        "v1",
                        self.v1_sha,
                    )
                ],
                [
                    self.provenance(
                        "v1_1",
                        self.v11_sha,
                    )
                ],
            )

    def test_accepts_matching_v11_provenance(
        self,
    ):
        validate_prompt_provenance(
            [
                self.provenance(
                    "v1_1",
                    self.v11_sha,
                )
            ],
            [
                self.provenance(
                    "v1_1",
                    self.v11_sha,
                )
            ],
        )

    def test_missing_version_is_inferred_only_by_hash(
        self,
    ):
        version, digest = (
            resolve_prompt_provenance(
                {
                    "artifact_hashes": {
                        "system_prompt_sha256":
                            self.v1_sha,
                    }
                },
                Path("legacy_v1/run_meta.json"),
            )
        )
        self.assertEqual(
            version,
            "v1",
        )
        self.assertEqual(
            digest,
            self.v1_sha,
        )

        with self.assertRaisesRegex(
            ValueError,
            "Cannot infer missing prompt_version",
        ):
            resolve_prompt_provenance(
                {
                    "artifact_hashes": {
                        "system_prompt_sha256":
                            "0" * 64,
                    }
                },
                Path("ambiguous/run_meta.json"),
            )

    def test_original_phase8_defaults_unchanged(
        self,
    ):
        self.assertEqual(
            DEFAULT_LOCAL_ROOT,
            Path("artifacts/phase7/test"),
        )
        self.assertEqual(
            DEFAULT_CLOUD_ROOT,
            Path("artifacts/phase7/cloud/test"),
        )
        self.assertEqual(
            DEFAULT_OUTPUT_DIR,
            Path("artifacts/phase8"),
        )


if __name__ == "__main__":
    unittest.main()
