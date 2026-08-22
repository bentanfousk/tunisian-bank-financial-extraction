from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BenchmarkVersion:
    name: str
    system_prompt_path: Path
    user_template_path: Path
    phase7_output_root: Path
    mlflow_experiment: str

    @property
    def cloud_output_root(self) -> Path:
        return self.phase7_output_root / "cloud"


VERSIONS = {
    "v1": BenchmarkVersion(
        name="v1",
        system_prompt_path=(
            PROJECT_ROOT
            / "prompts"
            / "extract_v1.txt"
        ),
        user_template_path=(
            PROJECT_ROOT
            / "prompts"
            / "extract_user_v1.txt"
        ),
        phase7_output_root=(
            PROJECT_ROOT
            / "artifacts"
            / "phase7"
        ),
        mlflow_experiment="track_a_v1_dev",
    ),
    "v1_1": BenchmarkVersion(
        name="v1_1",
        system_prompt_path=(
            PROJECT_ROOT
            / "prompts"
            / "extract_v1_1.txt"
        ),
        user_template_path=(
            PROJECT_ROOT
            / "prompts"
            / "extract_user_v1.txt"
        ),
        phase7_output_root=(
            PROJECT_ROOT
            / "artifacts"
            / "phase7_v1_1"
        ),
        mlflow_experiment="track_a_v1_1_dev",
    ),
}


def get_benchmark_version(
    name: str,
) -> BenchmarkVersion:
    try:
        version = VERSIONS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown prompt/benchmark version {name!r}. "
            f"Allowed: {sorted(VERSIONS)}"
        ) from exc

    if not version.system_prompt_path.exists():
        raise FileNotFoundError(
            f"System prompt not found: "
            f"{version.system_prompt_path}"
        )

    if not version.user_template_path.exists():
        raise FileNotFoundError(
            f"User template not found: "
            f"{version.user_template_path}"
        )

    return version