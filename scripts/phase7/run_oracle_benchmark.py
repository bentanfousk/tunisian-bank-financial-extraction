from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml
from mlflow.tracking import MlflowClient

from scripts.phase4.mlflow_tracking import configure_mlflow
from scripts.phase6.compare_oracle import compare_scores


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"
REPORTS_TEXT_DIR = (
    PROJECT_ROOT / "data" / "processed" / "reports_txt"
)

MODELS_CONFIG_PATH = PROJECT_ROOT / "configs" / "models.yaml"
MODELS_LOCK_PATH = PROJECT_ROOT / "configs" / "models.lock.json"

MANIFEST_PATH = (
    PROJECT_ROOT / "artifacts" / "manifests" / "page_manifest.json"
)

PIPELINE_ROOT = PROJECT_ROOT / "artifacts" / "phase7"
OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "phase7" / "oracle"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object in {path}")

    return data


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise TypeError(f"Expected YAML object in {path}")

    return data


def write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def read_split(split_name: str) -> list[str]:
    path = SPLITS_DIR / f"{split_name}.txt"

    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")

    docs = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()

        if line:
            docs.append(line)

    return docs


def load_local_models() -> list[dict[str, Any]]:
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

    return matches[-1] if matches else None


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

    if not meta_path.exists():
        return False

    required = [
        run_dir / "score.json",
        run_dir / "comparison.json",
    ]

    if not all(path.exists() for path in required):
        return False

    try:
        meta = load_json(meta_path)
    except Exception:
        return False

    return meta.get("status") == "completed"


def validate_inputs(
    split_name: str,
    doc_id: str,
    model_id: str,
) -> tuple[Path, Path, Path]:
    annotation_path = ANNOTATIONS_DIR / f"{doc_id}.json"
    report_text_path = REPORTS_TEXT_DIR / f"{doc_id}.jsonl"

    pipeline_score_path = (
        PIPELINE_ROOT
        / split_name
        / f"{doc_id}__{model_id}"
        / "score.json"
    )

    missing = [
        path
        for path in [
            annotation_path,
            report_text_path,
            MANIFEST_PATH,
            pipeline_score_path,
        ]
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required oracle input(s):\n"
            + "\n".join(str(path) for path in missing)
        )

    return (
        annotation_path,
        report_text_path,
        pipeline_score_path,
    )


def run_phase5(
    annotation_path: Path,
    validation_path: Path,
    report_text_path: Path,
    prediction_path: Path | None,
    run_dir: Path,
) -> dict[str, Any]:
    command = [
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
        command.extend(
            [
                "--prediction",
                str(prediction_path),
            ]
        )

    result = run_subprocess(command)

    (run_dir / "phase5_stdout.txt").write_text(
        result.stdout,
        encoding="utf-8",
    )

    (run_dir / "phase5_stderr.txt").write_text(
        result.stderr,
        encoding="utf-8",
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Phase 5 failed. See phase5_stdout.txt "
            "and phase5_stderr.txt."
        )

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Phase 5 exited successfully but did not "
            "return valid JSON."
        ) from exc


def rebuild_results_jsonl(split_dir: Path) -> None:
    records = []

    for meta_path in sorted(
        split_dir.glob("*/run_meta.json")
    ):
        meta = load_json(meta_path)

        if meta.get("status") != "completed":
            continue

        score_path = meta_path.parent / "score.json"
        comparison_path = meta_path.parent / "comparison.json"

        if not score_path.exists() or not comparison_path.exists():
            continue

        score = load_json(score_path)
        comparison = load_json(comparison_path)

        records.append(
            {
                "split": meta["split"],
                "doc_id": meta["doc_id"],
                "model_id": meta["model_id"],
                "model_tag": meta["model_tag"],
                "mlflow_run_id": meta["mlflow_run_id"],
                "oracle_fully_correct_fields": score["summary"][
                    "fully_correct_fields"
                ],
                "oracle_total_fields": score["summary"]["total_fields"],
                "oracle_fully_correct_rate": score["summary"][
                    "fully_correct_rate"
                ],
                "oracle_valid_json": score["summary"]["valid_json"],
                **comparison["summary"],
            }
        )

    output_path = split_dir / "results.jsonl"

    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(record, ensure_ascii=False)
                + "\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen Phase-6 oracle diagnostic "
            "across a Phase-7 benchmark split."
        )
    )

    parser.add_argument(
        "--split",
        required=True,
        choices=["dev", "test"],
    )

    parser.add_argument(
        "--only-doc",
        help="Run only one document from the selected split.",
    )

    parser.add_argument(
        "--only-model",
        help=(
            "Run only one models.yaml local model ID, "
            "for example qwen3_4b."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    documents = read_split(args.split)
    models = load_local_models()
    model_digests = load_model_digests()

    if args.only_doc:
        if args.only_doc not in documents:
            raise ValueError(
                f"{args.only_doc} is not in {args.split}.txt"
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
                f"Unknown or disabled model ID: {args.only_model}"
            )

    for model in models:
        if model["tag"] not in model_digests:
            raise ValueError(
                f"Frozen digest missing for model: {model['tag']}"
            )

    split_dir = OUTPUT_ROOT / args.split
    split_dir.mkdir(parents=True, exist_ok=True)

    configure_mlflow()
    client = MlflowClient()

    total_runs = len(documents) * len(models)

    print(
        f"\nSplit: {args.split}"
        f"\nDocuments: {len(documents)}"
        f"\nModels: {len(models)}"
        f"\nOracle runs: {total_runs}"
        f"\nPython: {sys.executable}"
    )

    run_number = 0

    for doc_id in documents:
        for model in models:
            run_number += 1

            model_id = model["id"]
            model_tag = model["tag"]

            run_dir = (
                split_dir
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

            (
                annotation_path,
                report_text_path,
                pipeline_score_path,
            ) = validate_inputs(
                args.split,
                doc_id,
                model_id,
            )

            if run_dir.exists():
                shutil.rmtree(run_dir)

            run_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            # Frozen Phase-6 oracle execution.
            phase6_command = [
                sys.executable,
                "-m",
                "scripts.phase6.oracle_run",
                doc_id,
                model_tag,
            ]

            phase6 = run_subprocess(
                phase6_command
            )

            (run_dir / "phase6_stdout.txt").write_text(
                phase6.stdout,
                encoding="utf-8",
            )

            (run_dir / "phase6_stderr.txt").write_text(
                phase6.stderr,
                encoding="utf-8",
            )

            mlflow_run_id = parse_mlflow_run_id(
                phase6.stdout
            )

            if phase6.returncode != 0:
                write_json(
                    run_dir / "run_meta.json",
                    {
                        "status": "phase6_failed",
                        "split": args.split,
                        "doc_id": doc_id,
                        "model_id": model_id,
                        "model_tag": model_tag,
                        "model_digest": model_digests[model_tag],
                        "mlflow_run_id": mlflow_run_id,
                    },
                )

                raise RuntimeError(
                    "Phase 6 failed for "
                    f"{doc_id} × {model_id}. "
                    "See phase6_stdout.txt and "
                    "phase6_stderr.txt."
                )

            if mlflow_run_id is None:
                raise RuntimeError(
                    "Phase 6 completed but no MLflow "
                    "run ID could be parsed."
                )

            # Recover Phase-6 artifacts from MLflow.
            raw_response_path = download_artifact(
                client,
                mlflow_run_id,
                "raw_response.txt",
                run_dir,
            )

            validation_path = download_artifact(
                client,
                mlflow_run_id,
                "validation.json",
                run_dir,
            )

            validation = load_json(
                validation_path
            )

            prediction_path = None

            if validation["valid"]:
                prediction_path = download_artifact(
                    client,
                    mlflow_run_id,
                    "prediction.json",
                    run_dir,
                )

            # Frozen Phase-5 scoring of the oracle run.
            oracle_score = run_phase5(
                annotation_path=annotation_path,
                validation_path=validation_path,
                report_text_path=report_text_path,
                prediction_path=prediction_path,
                run_dir=run_dir,
            )

            write_json(
                run_dir / "score.json",
                oracle_score,
            )

            # Frozen Phase-6 pipeline-vs-oracle interpretation.
            pipeline_score = load_json(
                pipeline_score_path
            )

            comparison = compare_scores(
                pipeline_score=pipeline_score,
                oracle_score=oracle_score,
            )

            write_json(
                run_dir / "comparison.json",
                comparison,
            )

            # Capture MLflow provenance.
            mlflow_run = client.get_run(
                mlflow_run_id
            )

            tags = mlflow_run.data.tags

            artifact_hashes = {
                key: value
                for key, value in tags.items()
                if key.endswith("_sha256")
                or key == "diagnostic_mode"
            }

            run_meta = {
                "status": "completed",
                "split": args.split,
                "doc_id": doc_id,
                "model_id": model_id,
                "model_tag": model_tag,
                "model_digest": model_digests[model_tag],
                "mlflow_run_id": mlflow_run_id,
                "report_pages": tags.get("report_pages"),
                "pdf_pages": tags.get("pdf_pages"),
                "page_set": tags.get("page_set"),
                "generation_settings": dict(
                    mlflow_run.data.params
                ),
                "runtime_metrics": dict(
                    mlflow_run.data.metrics
                ),
                "artifact_hashes": artifact_hashes,
                "valid_json": validation["valid"],
                "raw_response": raw_response_path.name,
                "validation": validation_path.name,
                "prediction": (
                    prediction_path.name
                    if prediction_path is not None
                    else None
                ),
                "score": "score.json",
                "pipeline_score": str(
                    pipeline_score_path.relative_to(PROJECT_ROOT)
                ),
                "comparison": "comparison.json",
            }

            write_json(
                run_dir / "run_meta.json",
                run_meta,
            )

            rebuild_results_jsonl(
                split_dir
            )

            print("Completed.")
            print("MLflow run:", mlflow_run_id)
            print("Valid JSON:", validation["valid"])
            print(
                "Oracle score:",
                json.dumps(
                    oracle_score.get("summary", {}),
                    ensure_ascii=False,
                ),
            )
            print(
                "Comparison:",
                json.dumps(
                    comparison["summary"],
                    ensure_ascii=False,
                ),
            )

    rebuild_results_jsonl(
        split_dir
    )

    print("\nOracle benchmark command completed.")
    print("Results:", split_dir)


if __name__ == "__main__":
    main()