import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml
from mlflow.tracking import MlflowClient

from scripts.phase4.mlflow_tracking import configure_mlflow

from scripts.benchmark_versions import (
    get_benchmark_version,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"
REPORTS_TEXT_DIR = PROJECT_ROOT / "data" / "processed" / "reports_txt"

MODELS_CONFIG_PATH = PROJECT_ROOT / "configs" / "models.yaml"
MODELS_LOCK_PATH = PROJECT_ROOT / "configs" / "models.lock.json"

MANIFEST_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "manifests"
    / "page_manifest.json"
)

OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "phase7"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_json(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )


def read_split(split_name: str) -> list[str]:
    path = SPLITS_DIR / f"{split_name}.txt"

    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}"
        )

    documents = []

    for raw_line in path.read_text(
        encoding="utf-8"
    ).splitlines():
        line = raw_line.split("#", 1)[0].strip()

        if line:
            documents.append(line)

    return documents


def load_local_models() -> list[dict]:
    config = load_yaml(MODELS_CONFIG_PATH)

    return [
        model
        for model in config["local_models"]
        if model.get("enabled", False)
    ]


def load_model_digests() -> dict[str, str]:
    lock = load_json(MODELS_LOCK_PATH)

    return {
        model["name"]: model["digest"]
        for model in lock["models"]
    }


def parse_mlflow_run_id(stdout: str) -> str | None:
    matches = re.findall(
        r"(?:MLflow )?Run ID:\s*([A-Za-z0-9_-]+)",
        stdout,
    )

    if not matches:
        return None

    return matches[-1]


def run_subprocess(command: list[str]):
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def download_artifact(
    client: MlflowClient,
    run_id: str,
    artifact_name: str,
    output_dir: Path,
) -> Path:
    downloaded = Path(
        client.download_artifacts(
            run_id,
            artifact_name,
            str(output_dir),
        )
    )

    expected = output_dir / artifact_name

    if downloaded.resolve() != expected.resolve():
        shutil.copy2(downloaded, expected)

    return expected


def is_completed(run_dir: Path) -> bool:
    meta_path = run_dir / "run_meta.json"
    score_path = run_dir / "score.json"

    if not meta_path.exists():
        return False

    if not score_path.exists():
        return False

    try:
        meta = load_json(meta_path)
    except Exception:
        return False

    return meta.get("status") == "completed"


def validate_input_files(doc_id: str):
    annotation = (
        ANNOTATIONS_DIR / f"{doc_id}.json"
    )

    report_text = (
        REPORTS_TEXT_DIR / f"{doc_id}.jsonl"
    )

    missing = []

    if not annotation.exists():
        missing.append(str(annotation))

    if not report_text.exists():
        missing.append(str(report_text))

    if not MANIFEST_PATH.exists():
        missing.append(str(MANIFEST_PATH))

    if missing:
        raise FileNotFoundError(
            "Missing required benchmark input(s):\n"
            + "\n".join(missing)
        )


def rebuild_results_jsonl(split_output_dir: Path):
    records = []

    for meta_path in sorted(
        split_output_dir.glob("*/run_meta.json")
    ):
        meta = load_json(meta_path)

        if meta.get("status") != "completed":
            continue

        score_path = meta_path.parent / "score.json"

        if not score_path.exists():
            continue

        score = load_json(score_path)

        record = {
            "split": meta["split"],
            "doc_id": meta["doc_id"],
            "model_id": meta["model_id"],
            "model_tag": meta["model_tag"],
            "model_digest": meta["model_digest"],
            "mlflow_run_id": meta["mlflow_run_id"],
            "first_pass_valid": meta.get(
                "first_pass_valid"
            ),
            "retry_used": meta.get(
                "retry_used"
            ),
            **score.get("summary", {}),
            "prompt_version": meta.get(
                "prompt_version",
                "v1",
            ),
            
        }

        records.append(record)

    output_path = split_output_dir / "results.jsonl"

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the controlled Track-A benchmark "
            "using frozen Phase-2 to Phase-5 machinery."
        )
    )

    parser.add_argument(
        "--split",
        required=True,
        choices=[
            "dev",
            "test",
            "temporal",
        ],
    )

    parser.add_argument(
        "--only-doc",
        help=(
            "Run only one document from the selected split."
        ),
    )

    parser.add_argument(
        "--only-model",
        help=(
            "Run only one models.yaml local model ID, "
            "for example ministral3_3b."
        ),
    )

    parser.add_argument(
        "--prompt-version",
        default="v1",
        choices=[
            "v1",
            "v1_1",
        ],
    )

    return parser.parse_args()


def main():
    args = parse_args()
    benchmark_version = (
        get_benchmark_version(
            args.prompt_version
        )
    )

    documents = read_split(args.split)
    models = load_local_models()
    model_digests = load_model_digests()

    if args.only_doc:
        if args.only_doc not in documents:
            raise ValueError(
                f"{args.only_doc} is not in "
                f"{args.split}.txt"
            )

        documents = [args.only_doc]

    if args.only_model:
        models = [
            model
            for model in models
            if model["id"] == args.only_model
        ]

        if not models:
            raise ValueError(
                "Unknown or disabled model ID: "
                f"{args.only_model}"
            )

    for model in models:
        if model["tag"] not in model_digests:
            raise ValueError(
                "Frozen digest missing for model: "
                f"{model['tag']}"
            )

    split_output_dir = (
        benchmark_version
        .phase7_output_root
        / args.split
    )
    split_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    configure_mlflow(
        benchmark_version.mlflow_experiment
    )
    mlflow_client = MlflowClient()

    total_runs = len(documents) * len(models)

    print(
        f"\nSplit: {args.split}"
        f"\nDocuments: {len(documents)}"
        f"\nModels: {len(models)}"
        f"\nRuns: {total_runs}"
        f"\nPython: {sys.executable}"
    )

    run_number = 0

    for doc_id in documents:
        validate_input_files(doc_id)

        annotation_path = (
            ANNOTATIONS_DIR
            / f"{doc_id}.json"
        )

        report_text_path = (
            REPORTS_TEXT_DIR
            / f"{doc_id}.jsonl"
        )

        for model in models:
            run_number += 1

            model_id = model["id"]
            model_tag = model["tag"]

            run_dir = (
                split_output_dir
                / f"{doc_id}__{model_id}"
            )

            print(
                "\n"
                + "=" * 70
                + f"\n[{run_number}/{total_runs}] "
                + f"{doc_id} × {model_id}"
                + "\n"
                + "=" * 70
            )

            if is_completed(run_dir):
                print("Already completed -> skipping.")
                continue

            if run_dir.exists():
                shutil.rmtree(run_dir)

            run_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            # ---------------------------------
            # Phase 4: frozen extraction
            # ---------------------------------

            phase4_command = [
                sys.executable,
                "-m",
                "scripts.phase4.run_extract",
                doc_id,
                model_tag,
                "--prompt-version",
                args.prompt_version,
            ]

            phase4 = run_subprocess(
                phase4_command
            )

            (
                run_dir / "phase4_stdout.txt"
            ).write_text(
                phase4.stdout,
                encoding="utf-8",
            )

            (
                run_dir / "phase4_stderr.txt"
            ).write_text(
                phase4.stderr,
                encoding="utf-8",
            )

            mlflow_run_id = parse_mlflow_run_id(
                phase4.stdout
            )

            if phase4.returncode != 0:
                write_json(
                    run_dir / "run_meta.json",
                    {
                        "status": "phase4_failed",
                        "split": args.split,
                        "doc_id": doc_id,
                        "model_id": model_id,
                        "model_tag": model_tag,
                        "model_digest": model_digests[
                            model_tag
                        ],
                        "mlflow_run_id": mlflow_run_id,
                        "prompt_version": args.prompt_version,
                    },
                )

                raise RuntimeError(
                    "Phase 4 failed for "
                    f"{doc_id} × {model_id}. "
                    "See phase4_stdout.txt and "
                    "phase4_stderr.txt."
                )

            if mlflow_run_id is None:
                raise RuntimeError(
                    "Phase 4 completed but no MLflow "
                    "run ID could be parsed."
                )

            # ---------------------------------
            # Recover Phase-4 artifacts
            # ---------------------------------

            mlflow_run = mlflow_client.get_run(
                mlflow_run_id
            )

            tags = mlflow_run.data.tags

            first_pass_valid = (
                tags.get(
                    "first_pass_valid",
                    tags.get("valid", "false"),
                ).lower()
                == "true"
            )

            retry_used = (
                tags.get(
                    "retry_used",
                    "false",
                ).lower()
                == "true"
            )

            raw_response_path = download_artifact(
                mlflow_client,
                mlflow_run_id,
                "raw_response.txt",
                run_dir,
            )

            validation_path = download_artifact(
                mlflow_client,
                mlflow_run_id,
                "validation.json",
                run_dir,
            )

            # First-pass artifacts exist for every
            # run produced by the retry-enabled
            # Phase-4 generation policy.
            first_raw_response_path = (
                download_artifact(
                    mlflow_client,
                    mlflow_run_id,
                    "first_raw_response.txt",
                    run_dir,
                )
            )

            first_validation_path = (
                download_artifact(
                    mlflow_client,
                    mlflow_run_id,
                    "first_validation.json",
                    run_dir,
                )
            )

            retry_raw_response_path = None
            retry_validation_path = None

            if retry_used:
                retry_raw_response_path = (
                    download_artifact(
                        mlflow_client,
                        mlflow_run_id,
                        "retry_raw_response.txt",
                        run_dir,
                    )
                )

                retry_validation_path = (
                    download_artifact(
                        mlflow_client,
                        mlflow_run_id,
                        "retry_validation.json",
                        run_dir,
                    )
                )

            validation = load_json(
                validation_path
            )

            prediction_path = None

            if validation["valid"]:
                prediction_path = (
                    download_artifact(
                        mlflow_client,
                        mlflow_run_id,
                        "prediction.json",
                        run_dir,
                    )
                )

            # ---------------------------------
            # Phase 5: frozen scoring
            # ---------------------------------

            phase5_command = [
                sys.executable,
                "-m",
                "scripts.phase5.score_prediction",
                "--annotation",
                str(annotation_path),
                "--validation",
                str(validation_path),
                "--manifest",
                str(MANIFEST_PATH),
                "--report-text",
                str(report_text_path),
            ]

            if prediction_path is not None:
                phase5_command.extend(
                    [
                        "--prediction",
                        str(prediction_path),
                    ]
                )

            phase5 = run_subprocess(
                phase5_command
            )

            (
                run_dir / "phase5_stdout.txt"
            ).write_text(
                phase5.stdout,
                encoding="utf-8",
            )

            (
                run_dir / "phase5_stderr.txt"
            ).write_text(
                phase5.stderr,
                encoding="utf-8",
            )

            if phase5.returncode != 0:
                write_json(
                    run_dir / "run_meta.json",
                    {
                        "status": "phase5_failed",
                        "split": args.split,
                        "doc_id": doc_id,
                        "model_id": model_id,
                        "model_tag": model_tag,
                        "model_digest": model_digests[
                            model_tag
                        ],
                        "mlflow_run_id": mlflow_run_id,
                        "prompt_version": args.prompt_version,
                    },
                )

                raise RuntimeError(
                    "Phase 5 failed for "
                    f"{doc_id} × {model_id}. "
                    "See phase5_stdout.txt and "
                    "phase5_stderr.txt."
                )

            try:
                score = json.loads(
                    phase5.stdout
                )
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "Phase 5 exited successfully but "
                    "did not return valid JSON."
                ) from exc

            write_json(
                run_dir / "score.json",
                score,
            )

            # ---------------------------------
            # Capture MLflow provenance
            # ---------------------------------

            artifact_hashes = {
                key: value
                for key, value in tags.items()
                if key.endswith("_sha256")
            }

            run_meta = {
                "status": "completed",
                "split": args.split,
                "prompt_version": (
                    args.prompt_version
                ),
                "doc_id": doc_id,
                "model_id": model_id,
                "model_tag": model_tag,
                "model_digest": model_digests[
                    model_tag
                ],
                "mlflow_run_id": mlflow_run_id,
                "report_pages": tags.get(
                    "report_pages"
                ),
                "pdf_pages": tags.get(
                    "pdf_pages"
                ),
                "page_set": tags.get(
                    "page_set"
                ),
                "generation_settings": dict(
                    mlflow_run.data.params
                ),
                "runtime_metrics": dict(
                    mlflow_run.data.metrics
                ),
                "artifact_hashes": (
                    artifact_hashes
                ),
                "first_pass_valid": (
                    first_pass_valid
                ),
                "retry_used": (
                    retry_used
                ),
                "valid_json": validation[
                    "valid"
                ],
                "raw_response": (
                    raw_response_path.name
                ),
                "first_raw_response": (
                    first_raw_response_path.name
                ),
                "first_validation": (
                    first_validation_path.name
                ),
                "retry_raw_response": (
                    retry_raw_response_path.name
                    if retry_raw_response_path
                    else None
                ),
                "retry_validation": (
                    retry_validation_path.name
                    if retry_validation_path
                    else None
                ),
                "validation": (
                    validation_path.name
                ),
                "prediction": (
                    prediction_path.name
                    if prediction_path
                    else None
                ),
                "score": "score.json",
            }

            write_json(
                run_dir / "run_meta.json",
                run_meta,
            )

            rebuild_results_jsonl(
                split_output_dir
            )

            correct = score.get(
                "summary",
                {},
            )

            print(
                "Completed."
            )

            print(
                "MLflow run:",
                mlflow_run_id,
            )

            print(
                "First-pass valid:",
                first_pass_valid,
            )

            print(
                "Retry used:",
                retry_used,
            )

            print(
                "Valid JSON:",
                validation["valid"],
            )

            print(
                "Score summary:",
                json.dumps(
                    correct,
                    ensure_ascii=False,
                ),
            )

    rebuild_results_jsonl(
        split_output_dir
    )

    print(
        "\nBenchmark command completed."
    )

    print(
        "Results:",
        split_output_dir,
    )


if __name__ == "__main__":
    main()
