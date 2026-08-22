from pathlib import Path
from typing import Any
import argparse
import hashlib
import json
import sys

import ollama
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]

FIELD_SCHEMA_PATH = (
    REPO_ROOT / "schemas" / "field_schema.json"
)



RUN_CONFIG_PATH = (
    REPO_ROOT / "configs" / "run.yaml"
)

MODELS_LOCK_PATH = (
    REPO_ROOT / "configs" / "models.lock.json"
)


# Make the project root importable when this script
# is executed directly.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts.phase2_page_select.page_select import select_pages
from scripts.phase4.prompt_builder import build_messages
from scripts.phase4.generation_policy import (
    generate_with_one_repair,
    REPAIR_PROMPT_SHA256,
)
from scripts.phase4.mlflow_tracking import (
    log_extraction_run,
    log_generation_failure,
)
from scripts.phase4.gpu_monitor import GpuMemoryMonitor

from scripts.benchmark_versions import (
    get_benchmark_version,
)

def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a YAML object in {path}"
        )

    return data


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    with path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a JSON object in {path}"
        )

    return data


def sha256_file(path: Path) -> str:
    """
    Compute the SHA-256 digest of a file exactly as stored
    on disk.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


FIELD_SCHEMA = load_json(FIELD_SCHEMA_PATH)

RUN_CONFIG = load_yaml(RUN_CONFIG_PATH)

MODELS_LOCK = load_json(MODELS_LOCK_PATH)


GENERATION_SETTINGS = RUN_CONFIG.get("generation")

if not isinstance(GENERATION_SETTINGS, dict):
    raise ValueError(
        "configs/run.yaml must contain a "
        "'generation' mapping."
    )


# Fingerprints of the frozen/configuration artifacts.
def build_artifact_hashes(
    prompt_version: str,
) -> dict[str, str]:

    version = get_benchmark_version(
        prompt_version
    )

    return {
        "system_prompt_sha256": sha256_file(
            version.system_prompt_path
        ),
        "user_template_sha256": sha256_file(
            version.user_template_path
        ),
        "field_schema_sha256": sha256_file(
            FIELD_SCHEMA_PATH
        ),
        "run_config_sha256": sha256_file(
            RUN_CONFIG_PATH
        ),
        "repair_prompt_sha256": (
            REPAIR_PROMPT_SHA256
        ),
    }


# Backward-compatible frozen V1 hashes for Phase 6 and any
# existing imports that predate prompt version selection.
ARTIFACT_HASHES = build_artifact_hashes("v1")


def parse_doc_id(doc_id: str) -> tuple[str, int]:
    try:
        bank, year_text = doc_id.rsplit("_", 1)
        target_year = int(year_text)

    except (ValueError, AttributeError):
        raise ValueError(
            f"Invalid doc_id '{doc_id}'. "
            "Expected format like 'UIB_2024'."
        )

    if not bank:
        raise ValueError(
            f"Invalid doc_id '{doc_id}': bank is empty."
        )

    return bank, target_year


def get_locked_model(
    model_id: str,
) -> dict[str, Any]:
    """
    Return the frozen lock entry for model_id.

    A model that is not present in models.lock.json
    is not allowed in the frozen benchmark.
    """

    models = MODELS_LOCK.get("models")

    if not isinstance(models, list):
        raise ValueError(
            "configs/models.lock.json must contain "
            "a 'models' list."
        )

    for model in models:
        if (
            isinstance(model, dict)
            and model.get("name") == model_id
        ):
            return model

    raise ValueError(
        f"Model '{model_id}' is not present in "
        "configs/models.lock.json."
    )


def verify_model_digest(
    model_id: str,
    expected_digest: str,
) -> None:
    """
    Verify that the installed Ollama model is exactly
    the model frozen in models.lock.json.
    """

    response = ollama.list()

    for model in response.models:
        if model.model == model_id:
            actual_digest = model.digest

            if actual_digest != expected_digest:
                raise RuntimeError(
                    f"Model digest mismatch for "
                    f"'{model_id}'. "
                    f"Expected {expected_digest}, "
                    f"found {actual_digest}."
                )

            return

    raise RuntimeError(
        f"Model '{model_id}' is not installed "
        "in Ollama."
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one Track-A extraction for one "
            "document and one frozen Ollama model."
        )
    )

    parser.add_argument(
        "doc_id",
        help=(
            "Document ID, for example UIB_2024."
        ),
    )

    parser.add_argument(
        "model_id",
        help=(
            "Exact frozen Ollama model ID, for example "
            "ministral-3:3b-instruct-2512-q4_K_M."
        ),
    )
    parser.add_argument(
        "--prompt-version",
        default="v1",
        choices=[
            "v1",
            "v1_1",
        ],
        help=(
            "Frozen extraction-prompt version. "
            "Default: v1."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    doc_id = args.doc_id
    model_id = args.model_id

    prompt_version = (
        args.prompt_version
    )

    benchmark_version = (
        get_benchmark_version(
            prompt_version
        )
    )

    artifact_hashes = (
        build_artifact_hashes(
            prompt_version
        )
    )

    bank, target_year = parse_doc_id(doc_id)

    # Resolve the exact frozen model.
    locked_model = get_locked_model(model_id)

    model_digest = locked_model.get("digest")

    if not isinstance(model_digest, str):
        raise ValueError(
            f"Frozen model '{model_id}' does not have "
            "a valid digest in models.lock.json."
        )

    # Make sure the installed Ollama model has not
    # changed since the roster was frozen.
    verify_model_digest(
        model_id=model_id,
        expected_digest=model_digest,
    )

    # Phase 2:
    # frozen full individual statement block.
    selection = select_pages(
        doc_id=doc_id,
        core_only=False,
    )

    # Phase 3:
    # frozen SYSTEM + USER messages.
    messages = build_messages(
        doc_id=doc_id,
        bank=bank,
        target_year=target_year,
        context=selection["text"],
        prompt_version=prompt_version,
    )

    print("Document:", doc_id)
    print("Model:", model_id)
    print("Model digest:", model_digest)
    print(
        "Prompt version:",
        prompt_version,
    )
    print(
        "MLflow experiment:",
        benchmark_version.mlflow_experiment,
    )
    print("Page set:", selection["page_set"])
    print(
        "Report pages:",
        selection["report_pages"],
    )
    print(
        "PDF/JSONL pages:",
        selection["pdf_pages"],
    )
    print(
        "Character count:",
        selection["character_count"],
    )
    print(
        "Schema loaded:",
        FIELD_SCHEMA_PATH.name,
    )
    print(
        "Generation settings:",
        GENERATION_SETTINGS,
    )

    print("\nFrozen artifact hashes:")

    for name, digest in artifact_hashes.items():
        print(f"{name}: {digest}")

    print("\nCalling Ollama...")

    try:
        with GpuMemoryMonitor() as gpu:
            generation_policy = (
                generate_with_one_repair(
                    model_id=model_id,
                    messages=messages,
                    schema=FIELD_SCHEMA,
                    settings=GENERATION_SETTINGS,
                )
            )

        result = generation_policy[
            "generation_result"
        ]

        validation = generation_policy[
            "validation"
        ]

    except Exception as exc:
        failure_run_id = log_generation_failure(
            doc_id=doc_id,
            model_id=model_id,
            model_digest=model_digest,
            bank=bank,
            target_year=target_year,
            selection=selection,
            generation_settings=GENERATION_SETTINGS,
            artifact_hashes=artifact_hashes,
            experiment_name=(
                benchmark_version.mlflow_experiment
            ),
            error=exc,
        )

        print("\n--- GENERATION FAILED ---")
        print(
            "Error type:",
            type(exc).__name__,
        )
        print(
            "Error:",
            str(exc),
        )
        print(
            "MLflow Run ID:",
            failure_run_id,
        )

        # Keep the command failed.
        raise

    # Compute actual model generation throughput.
    generation_duration_s = (
        result["eval_duration_ns"]
        / 1_000_000_000
    )

    if generation_duration_s > 0:
        tokens_per_second = (
            result["output_tokens"]
            / generation_duration_s
        )
    else:
        tokens_per_second = 0.0

    result["tokens_per_second"] = (
        tokens_per_second
    )

    result["gpu_memory_baseline_mb"] = (
        gpu.baseline_mb
    )

    result["gpu_memory_peak_used_mb"] = (
        gpu.peak_mb
    )

    result["gpu_memory_incremental_peak_mb"] = (
        gpu.incremental_peak_mb
    )

    print("\n--- MODEL RESPONSE ---")
    print(result["text"])

    
    print("\n--- FORMAT POLICY ---")

    print(
        "First-pass valid:",
        generation_policy[
            "first_validation"
        ]["valid"],
    )

    print(
        "Retry used:",
        generation_policy[
            "retry_used"
        ],
    )


    print("\n--- VALIDATION ---")
    print(
        "Valid:",
        validation["valid"],
    )

    if not validation["valid"]:
        print(
            "Error type:",
            validation["error_type"],
        )
        print(
            "Error message:",
            validation["error_message"],
        )

    print("\n--- GENERATION METADATA ---")

    print(
        "Input tokens:",
        result["input_tokens"],
    )

    print(
        "Output tokens:",
        result["output_tokens"],
    )

    print(
        "Total duration (ns):",
        result["total_duration_ns"],
    )

    print(
        "Load duration (ns):",
        result["load_duration_ns"],
    )

    print(
        "Prompt eval duration (ns):",
        result["prompt_eval_duration_ns"],
    )

    print(
        "Generation duration (ns):",
        result["eval_duration_ns"],
    )

    print(
        "Done reason:",
        result["done_reason"],
    )

    print(
        "Tokens/sec:",
        round(
            result["tokens_per_second"],
            2,
        ),
    )

    print(
        "GPU baseline (MiB):",
        round(
            result["gpu_memory_baseline_mb"],
            2,
        ),
    )

    print(
        "GPU peak used (MiB):",
        round(
            result["gpu_memory_peak_used_mb"],
            2,
        ),
    )

    print(
        "GPU incremental peak (MiB):",
        round(
            result[
                "gpu_memory_incremental_peak_mb"
            ],
            2,
        ),
    )

    # Record the complete run in MLflow.
    mlflow_run_id = log_extraction_run(
        doc_id=doc_id,
        model_id=model_id,
        model_digest=model_digest,
        bank=bank,
        target_year=target_year,
        selection=selection,
        generation_settings=GENERATION_SETTINGS,
        artifact_hashes=artifact_hashes,
        experiment_name=(
            benchmark_version.mlflow_experiment
        ),
        generation_result=result,
        validation=validation,
        retry_info=generation_policy,
        
    )

    print("\n--- MLFLOW ---")
    print(
        "Run ID:",
        mlflow_run_id,
    )


if __name__ == "__main__":
    main()
