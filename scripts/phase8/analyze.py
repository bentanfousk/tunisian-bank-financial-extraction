"""Analyze the frozen Phase-7 controlled test benchmark.

Primary scores come only from Phase-5 ``score.json`` files. Invalid final
predictions remain strict failures; raw model responses are never reparsed.

Example:
    python -m scripts.phase8.analyze
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from scripts.benchmark_versions import (
    VERSIONS,
    get_benchmark_version,
)


EXPECTED_LOCAL_MODELS = {
    "qwen3_4b",
    "ministral3_3b",
    "gemma3_4b",
    "mistral_7b",
}
EXPECTED_DOCS = {
    "amen_2024",
    "ATB_2023",
    "attijari_2024",
    "bh_2024",
    "bna_2024",
    "BT_2024",
}
EXPECTED_FIELDS = (
    "total_assets",
    "total_equity",
    "net_banking_income",
    "operating_income",
    "net_income",
    "customer_deposits",
    "net_customer_loans",
)
COMPONENTS = (
    "status_match",
    "value_match",
    "unit_match",
    "scope_match",
    "source_year_match",
    "citation_value_present",
)
RUNTIME_METRICS = (
    "total_duration_s",
    "first_total_duration_s",
    "retry_total_duration_s",
    "generation_duration_s",
    "tokens_per_second",
    "input_tokens",
    "output_tokens",
    "gpu_memory_peak_used_mb",
    "gpu_memory_incremental_peak_mb",
)
DEFAULT_LOCAL_ROOT = Path("artifacts/phase7/test")
DEFAULT_CLOUD_ROOT = Path("artifacts/phase7/cloud/test")
DEFAULT_OUTPUT_DIR = Path("artifacts/phase8")


def read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def sha256_file(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def resolve_prompt_provenance(
    meta: dict[str, Any],
    meta_path: Path,
) -> tuple[str, str]:
    artifact_hashes = meta.get(
        "artifact_hashes"
    )
    if not isinstance(artifact_hashes, dict):
        raise ValueError(
            "Missing artifact_hashes in "
            f"{meta_path}"
        )

    system_prompt_sha256 = (
        artifact_hashes.get(
            "system_prompt_sha256"
        )
    )
    if not isinstance(system_prompt_sha256, str) or not system_prompt_sha256:
        raise ValueError(
            "Missing system_prompt_sha256 in "
            f"{meta_path}"
        )

    prompt_version = meta.get(
        "prompt_version"
    )
    known_hashes = {
        name: sha256_file(
            version.system_prompt_path
        )
        for name, version in VERSIONS.items()
    }

    if prompt_version is None:
        matches = [
            name
            for name, digest in known_hashes.items()
            if digest == system_prompt_sha256
        ]
        if len(matches) != 1:
            raise ValueError(
                "Cannot infer missing prompt_version "
                "unambiguously from "
                f"system_prompt_sha256 in {meta_path}"
            )
        prompt_version = matches[0]
    elif not isinstance(prompt_version, str):
        raise ValueError(
            "Invalid prompt_version in "
            f"{meta_path}"
        )

    version = get_benchmark_version(
        prompt_version
    )
    expected_sha256 = known_hashes[
        version.name
    ]
    if system_prompt_sha256 != expected_sha256:
        raise ValueError(
            "prompt_version/system_prompt_sha256 "
            f"mismatch in {meta_path}"
        )

    return (
        version.name,
        system_prompt_sha256,
    )


def safe_rate(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def rounded(value: Any, digits: int = 6) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    return value


def mean_or_none(values: Iterable[float]) -> float | None:
    data = list(values)
    return statistics.fmean(data) if data else None


def median_or_none(values: Iterable[float]) -> float | None:
    data = list(values)
    return statistics.median(data) if data else None


def load_run(run_dir: Path, source: str) -> dict[str, Any]:
    meta_path = run_dir / "run_meta.json"
    score_path = run_dir / "score.json"
    validation_path = run_dir / "validation.json"
    for required in (meta_path, score_path, validation_path):
        if not required.exists():
            raise ValueError(f"Incomplete run directory; missing {required}")

    meta = read_json(meta_path)
    score = read_json(score_path)
    validation = read_json(validation_path)
    doc_id = meta.get("doc_id")
    model_id = meta.get("model_id")
    valid = validation.get("valid")
    if not isinstance(doc_id, str) or not isinstance(model_id, str):
        raise ValueError(f"Missing doc_id/model_id in {meta_path}")
    if not isinstance(valid, bool):
        raise ValueError(f"Missing boolean valid in {validation_path}")
    if score.get("doc_id") != doc_id:
        raise ValueError(f"Document mismatch between {meta_path} and {score_path}")

    raw_fields = score.get("fields")
    if not isinstance(raw_fields, list):
        raise ValueError(f"Missing fields list in {score_path}")
    fields: dict[str, dict[str, Any]] = {}
    for verdict in raw_fields:
        if not isinstance(verdict, dict) or not isinstance(verdict.get("field"), str):
            raise ValueError(f"Malformed field verdict in {score_path}")
        field = verdict["field"]
        if field in fields:
            raise ValueError(f"Duplicate field {field!r} in {score_path}")
        if not isinstance(verdict.get("correct"), bool):
            raise ValueError(f"Missing boolean correct for {field} in {score_path}")
        fields[field] = verdict
    if set(fields) != set(EXPECTED_FIELDS):
        raise ValueError(
            f"Unexpected fields in {score_path}: "
            f"missing={sorted(set(EXPECTED_FIELDS) - set(fields))}, "
            f"extra={sorted(set(fields) - set(EXPECTED_FIELDS))}"
        )

    first_valid: bool | None = None
    retry_used = False
    retry_valid: bool | None = None
    if source == "local":
        first_path = run_dir / "first_validation.json"
        if not first_path.exists():
            raise ValueError(f"Missing local first-pass validation: {first_path}")
        first_validation = read_json(first_path)
        first_valid = first_validation.get("valid")
        if not isinstance(first_valid, bool):
            raise ValueError(f"Missing boolean valid in {first_path}")
        retry_path = run_dir / "retry_validation.json"
        retry_used = retry_path.exists() or bool(meta.get("retry_used"))
        if retry_used:
            if not retry_path.exists():
                raise ValueError(f"Retry marked as used but missing {retry_path}")
            retry_validation = read_json(retry_path)
            retry_valid = retry_validation.get("valid")
            if not isinstance(retry_valid, bool):
                raise ValueError(f"Missing boolean valid in {retry_path}")
        if first_valid and retry_used:
            raise ValueError(f"First-pass-valid run unexpectedly used retry: {run_dir}")
        if valid != (first_valid or retry_valid is True):
            raise ValueError(f"Inconsistent final validity in {run_dir}")

    summary = score.get("summary", {})
    correct_count = sum(int(fields[name]["correct"]) for name in EXPECTED_FIELDS)
    if summary.get("fully_correct_fields") != correct_count:
        raise ValueError(f"Score summary mismatch in {score_path}")
    if bool(summary.get("valid_json")) != valid:
        raise ValueError(f"Score/validation validity mismatch in {run_dir}")
    if not valid and correct_count != 0:
        raise ValueError(f"Invalid run has nonzero strict score in {score_path}")

    prompt_version, system_prompt_sha256 = (
        resolve_prompt_provenance(
            meta,
            meta_path,
        )
    )

    return {
        "source": source,
        "run_dir": str(run_dir),
        "doc_id": doc_id,
        "model_id": model_id,
        "model_tag": meta.get("model_tag") or meta.get("provider_model") or model_id,
        "prompt_version": prompt_version,
        "system_prompt_sha256": system_prompt_sha256,
        "valid": valid,
        "first_valid": first_valid,
        "retry_used": retry_used,
        "retry_valid": retry_valid,
        "first_error_type": first_validation.get("error_type") if source == "local" else None,
        "retry_error_type": (
            retry_validation.get("error_type")
            if source == "local" and retry_used
            else None
        ),
        "runtime": meta.get("runtime_metrics", {}),
        "fields": fields,
    }


def load_runs(root: Path, source: str) -> list[dict[str, Any]]:
    if not root.is_dir():
        raise ValueError(f"Artifact root does not exist: {root}")
    run_dirs = sorted({path.parent for path in root.glob("*/run_meta.json")})
    if not run_dirs:
        raise ValueError(f"No run directories found under {root}")
    return [load_run(run_dir, source) for run_dir in run_dirs]


def validate_prompt_provenance(
    local_runs: list[dict[str, Any]],
    cloud_runs: list[dict[str, Any]],
) -> None:
    def one_condition(
        label: str,
        runs: list[dict[str, Any]],
    ) -> tuple[str, str]:
        versions = {
            run["prompt_version"]
            for run in runs
        }
        hashes = {
            run["system_prompt_sha256"]
            for run in runs
        }
        if len(versions) != 1:
            raise ValueError(
                f"Inconsistent {label} prompt versions: "
                f"{sorted(versions)}"
            )
        if len(hashes) != 1:
            raise ValueError(
                f"Inconsistent {label} system prompt "
                f"SHA values: {sorted(hashes)}"
            )
        return (
            next(iter(versions)),
            next(iter(hashes)),
        )

    local_version, local_sha = one_condition(
        "local",
        local_runs,
    )
    cloud_version, cloud_sha = one_condition(
        "cloud",
        cloud_runs,
    )
    if local_version != cloud_version:
        raise ValueError(
            "Local/cloud prompt version mismatch: "
            f"{local_version!r} != {cloud_version!r}"
        )
    if local_sha != cloud_sha:
        raise ValueError(
            "Local/cloud system prompt SHA mismatch: "
            f"{local_sha} != {cloud_sha}"
        )


def validate_experiment(local_runs: list[dict[str, Any]], cloud_runs: list[dict[str, Any]]) -> None:
    validate_prompt_provenance(
        local_runs,
        cloud_runs,
    )

    def pairs(runs: list[dict[str, Any]]) -> list[tuple[str, str]]:
        return [(run["doc_id"], run["model_id"]) for run in runs]

    for label, runs in (("local", local_runs), ("cloud", cloud_runs)):
        run_pairs = pairs(runs)
        duplicates = [pair for pair, count in Counter(run_pairs).items() if count > 1]
        if duplicates:
            raise ValueError(f"Duplicate {label} doc/model runs: {duplicates}")

    local_models = {run["model_id"] for run in local_runs}
    local_docs = {run["doc_id"] for run in local_runs}
    cloud_models = {run["model_id"] for run in cloud_runs}
    cloud_docs = {run["doc_id"] for run in cloud_runs}
    if local_models != EXPECTED_LOCAL_MODELS:
        raise ValueError(f"Local models mismatch: found {sorted(local_models)}")
    if local_docs != EXPECTED_DOCS:
        raise ValueError(f"Local documents mismatch: found {sorted(local_docs)}")
    expected_local_pairs = {(doc, model) for doc in EXPECTED_DOCS for model in EXPECTED_LOCAL_MODELS}
    if set(pairs(local_runs)) != expected_local_pairs:
        raise ValueError("Local benchmark is not the complete 6 x 4 experiment")
    if len(cloud_models) != 1:
        raise ValueError(f"Expected exactly one cloud model, found {sorted(cloud_models)}")
    if cloud_docs != EXPECTED_DOCS:
        raise ValueError(f"Cloud documents mismatch: found {sorted(cloud_docs)}")


def component_counts(runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for component in COMPONENTS:
        applicable = [
            verdict[component]
            for run in runs
            if run["valid"]
            for verdict in run["fields"].values()
            if verdict.get(component) is not None
        ]
        correct = sum(value is True for value in applicable)
        output[component] = {
            "correct": correct,
            "applicable": len(applicable),
            "accuracy": safe_rate(correct, len(applicable)),
        }
    return output


def aggregate_model(model_id: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    strict_correct = sum(
        verdict["correct"] for run in runs for verdict in run["fields"].values()
    )
    total_fields = len(runs) * len(EXPECTED_FIELDS)
    valid_runs = sum(run["valid"] for run in runs)
    local = runs[0]["source"] == "local"
    first_valid = sum(run["first_valid"] is True for run in runs) if local else None
    retry_count = sum(run["retry_used"] for run in runs) if local else None
    retry_recovered = sum(run["retry_valid"] is True for run in runs) if local else None

    failure_reasons: Counter[str] = Counter()
    for run in runs:
        for verdict in run["fields"].values():
            if not verdict["correct"]:
                reasons = verdict.get("failure_reasons") or ["unspecified_failure"]
                failure_reasons.update(reasons)

    runtime: dict[str, Any] = {}
    if local:
        for metric in RUNTIME_METRICS:
            values = [
                float(run["runtime"][metric])
                for run in runs
                if isinstance(run["runtime"].get(metric), (int, float))
            ]
            runtime[f"{metric}_mean"] = mean_or_none(values)
            runtime[f"{metric}_median"] = median_or_none(values)
            runtime[f"{metric}_n"] = len(values)
        retry_durations = [
            float(run["runtime"]["retry_total_duration_s"])
            for run in runs
            if run["retry_used"]
            and isinstance(run["runtime"].get("retry_total_duration_s"), (int, float))
        ]
        total_durations = [
            float(run["runtime"]["total_duration_s"])
            for run in runs
            if isinstance(run["runtime"].get("total_duration_s"), (int, float))
        ]
        runtime["retry_duration_total_s"] = sum(retry_durations)
        runtime["retry_overhead_share_of_total_duration"] = safe_rate(
            sum(retry_durations), sum(total_durations)
        )
    else:
        latencies = [
            float(run["runtime"]["latency_s"])
            for run in runs
            if isinstance(run["runtime"].get("latency_s"), (int, float))
        ]
        runtime = {
            "latency_s_mean": mean_or_none(latencies),
            "latency_s_median": median_or_none(latencies),
            "latency_s_n": len(latencies),
        }

    return {
        "model_id": model_id,
        "source": runs[0]["source"],
        "model_tag": runs[0]["model_tag"],
        "runs": len(runs),
        "strict_correct_fields": strict_correct,
        "total_fields": total_fields,
        "strict_field_accuracy": safe_rate(strict_correct, total_fields),
        "final_schema_valid_runs": valid_runs,
        "final_schema_valid_rate": safe_rate(valid_runs, len(runs)),
        "first_pass_valid_runs": first_valid,
        "first_pass_valid_rate": safe_rate(first_valid, len(runs)) if local else None,
        "retry_count": retry_count,
        "retry_recovered": retry_recovered,
        "retry_recovery_rate": safe_rate(retry_recovered, retry_count) if local else None,
        "retry_failed": retry_count - retry_recovered if local else None,
        "components": component_counts(runs),
        "failure_reason_counts": dict(sorted(failure_reasons.items())),
        "runtime": runtime,
    }


def aggregate_fields(all_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    models = sorted({run["model_id"] for run in all_runs})
    for model_id in models:
        model_runs = [run for run in all_runs if run["model_id"] == model_id]
        for field in EXPECTED_FIELDS:
            verdicts = [run["fields"][field] for run in model_runs]
            correct = sum(verdict["correct"] for verdict in verdicts)
            failure_reasons: Counter[str] = Counter()
            for verdict in verdicts:
                if not verdict["correct"]:
                    failure_reasons.update(verdict.get("failure_reasons") or ["unspecified_failure"])
            row: dict[str, Any] = {
                "model_id": model_id,
                "source": model_runs[0]["source"],
                "field": field,
                "correct": correct,
                "total": len(verdicts),
                "accuracy": safe_rate(correct, len(verdicts)),
                "failures": len(verdicts) - correct,
                "failure_reasons": dict(sorted(failure_reasons.items())),
            }
            for component in COMPONENTS:
                values = [verdict.get(component) for verdict in verdicts]
                applicable = [value for value in values if value is not None]
                row[f"{component}_failures"] = sum(value is False for value in applicable)
                row[f"{component}_applicable"] = len(applicable)
            rows.append(row)
    return rows


def aggregate_local_fields(local_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field in EXPECTED_FIELDS:
        verdicts = [run["fields"][field] for run in local_runs]
        correct = sum(verdict["correct"] for verdict in verdicts)
        reasons: Counter[str] = Counter()
        for verdict in verdicts:
            if not verdict["correct"]:
                reasons.update(verdict.get("failure_reasons") or ["unspecified_failure"])
        rows.append({
            "field": field,
            "correct": correct,
            "total": len(verdicts),
            "accuracy": safe_rate(correct, len(verdicts)),
            "failures": len(verdicts) - correct,
            "failure_reasons": dict(sorted(reasons.items())),
        })
    rows.sort(key=lambda row: (-row["accuracy"], row["field"]))
    return rows


def aggregate_local_failures(local_runs: list[dict[str, Any]]) -> dict[str, Any]:
    verdicts = [verdict for run in local_runs for verdict in run["fields"].values()]
    failed = [verdict for verdict in verdicts if not verdict["correct"]]
    invalid = [
        verdict for verdict in failed
        if "invalid_prediction" in (verdict.get("failure_reasons") or [])
    ]
    valid_structured_failures = [verdict for verdict in failed if verdict not in invalid]
    reason_counts: Counter[str] = Counter()
    for verdict in valid_structured_failures:
        reason_counts.update(verdict.get("failure_reasons") or ["unspecified_failure"])
    component_failures = {
        component: sum(verdict.get(component) is False for verdict in valid_structured_failures)
        for component in COMPONENTS
    }
    citation_values = [
        verdict.get("citation_value_present")
        for run in local_runs if run["valid"]
        for verdict in run["fields"].values()
        if verdict.get("citation_value_present") is not None
    ]
    return {
        "total_field_decisions": len(verdicts),
        "strict_correct_fields": len(verdicts) - len(failed),
        "strict_failed_fields": len(failed),
        "invalid_prediction_field_failures": len(invalid),
        "invalid_prediction_share_of_strict_failures": safe_rate(len(invalid), len(failed)),
        "valid_structured_field_failures": len(valid_structured_failures),
        "valid_structured_share_of_strict_failures": safe_rate(
            len(valid_structured_failures), len(failed)
        ),
        "valid_failure_reason_events": dict(sorted(reason_counts.items())),
        "valid_failure_component_events": component_failures,
        "citation_eligible_fields": len(citation_values),
        "citation_failures": sum(value is False for value in citation_values),
        "citation_accuracy": safe_rate(
            sum(value is True for value in citation_values), len(citation_values)
        ),
        "overlap_note": (
            "Reason/component event counts can overlap when one field has multiple mismatches."
        ),
    }


def exact_mcnemar(
    model_a: str,
    model_b: str,
    run_lookup: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    a_correct_b_wrong = 0
    a_wrong_b_correct = 0
    both_correct = 0
    both_wrong = 0
    for doc_id in sorted(EXPECTED_DOCS):
        run_a = run_lookup[(doc_id, model_a)]
        run_b = run_lookup[(doc_id, model_b)]
        for field in EXPECTED_FIELDS:
            a = run_a["fields"][field]["correct"]
            b = run_b["fields"][field]["correct"]
            if a and b:
                both_correct += 1
            elif a:
                a_correct_b_wrong += 1
            elif b:
                a_wrong_b_correct += 1
            else:
                both_wrong += 1
    discordant = a_correct_b_wrong + a_wrong_b_correct
    if discordant == 0:
        p_value = 1.0
    else:
        smaller = min(a_correct_b_wrong, a_wrong_b_correct)
        lower_tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (2**discordant)
        p_value = min(1.0, 2 * lower_tail)
    return {
        "model_a": model_a,
        "model_b": model_b,
        "paired_decisions": len(EXPECTED_DOCS) * len(EXPECTED_FIELDS),
        "both_correct": both_correct,
        "a_correct_b_wrong": a_correct_b_wrong,
        "a_wrong_b_correct": a_wrong_b_correct,
        "both_wrong": both_wrong,
        "discordant_pairs": discordant,
        "exact_two_sided_p_value": p_value,
        "interpretation": "exploratory_small_sample",
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: rounded(row.get(name)) for name in fieldnames})


def percent(value: float | None) -> str:
    return "N/A" if value is None else f"{100 * value:.1f}%"


def number(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def render_report(analysis: dict[str, Any]) -> str:
    ranking = analysis["model_ranking"]
    locals_ranked = [row for row in ranking if row["source"] == "local"]
    cloud = next(row for row in ranking if row["source"] == "cloud")
    lines = [
        "# Phase 8 — Controlled Benchmark Analysis",
        "",
        "This report analyzes frozen Phase-7 predictions. Invalid final responses remain strict 0/7 failures; raw responses were not reinterpreted.",
        "",
        "## Primary model ranking",
        "",
        "| Model | Strict fields | Accuracy | Final valid | First-pass valid | Retry recovered | Citation accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ranking:
        citation = row["components"]["citation_value_present"]["accuracy"]
        retry = (
            f'{row["retry_recovered"]}/{row["retry_count"]}'
            if row["source"] == "local"
            else "N/A"
        )
        first = (
            f'{row["first_pass_valid_runs"]}/{row["runs"]} ({percent(row["first_pass_valid_rate"])})'
            if row["source"] == "local"
            else "N/A"
        )
        lines.append(
            f'| {row["model_id"]} | {row["strict_correct_fields"]}/{row["total_fields"]} '
            f'| {percent(row["strict_field_accuracy"])} '
            f'| {row["final_schema_valid_runs"]}/{row["runs"]} ({percent(row["final_schema_valid_rate"])}) '
            f'| {first} | {retry} | {percent(citation)} |'
        )

    lines.extend([
        "",
        "## Component accuracy on valid structured predictions",
        "",
        "Null/non-applicable component verdicts are excluded from each component denominator.",
        "",
        "| Model | Status | Value | Unit | Scope | Year | Citation |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in ranking:
        c = row["components"]
        cell = lambda name: f'{c[name]["correct"]}/{c[name]["applicable"]} ({percent(c[name]["accuracy"])})'
        lines.append(
            f'| {row["model_id"]} | {cell("status_match")} | {cell("value_match")} '
            f'| {cell("unit_match")} | {cell("scope_match")} '
            f'| {cell("source_year_match")} | {cell("citation_value_present")} |'
        )

    lines.extend([
        "",
        "## Local schema adherence",
        "",
        "| Model | First-pass valid | Repaired | Retry still invalid | Final valid |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in locals_ranked:
        lines.append(
            f'| {row["model_id"]} | {row["first_pass_valid_runs"]}/{row["runs"]} '
            f'| {row["retry_recovered"]}/{row["retry_count"]} '
            f'| {row["retry_failed"]} | {row["final_schema_valid_runs"]}/{row["runs"]} |'
        )
    overall = analysis["local_schema_overall"]
    lines.append(
        f'| **All local** | **{overall["first_pass_valid_runs"]}/{overall["runs"]}** '
        f'| **{overall["retry_recovered"]}/{overall["retry_count"]}** '
        f'| **{overall["retry_failed"]}** | **{overall["final_valid_runs"]}/{overall["runs"]}** |'
    )

    failures = analysis["local_failure_summary"]
    lines.extend([
        "",
        "## Local failure decomposition",
        "",
        f'Local models failed {failures["strict_failed_fields"]} of {failures["total_field_decisions"]} strict field decisions. '
        f'{failures["invalid_prediction_field_failures"]} failures ({percent(failures["invalid_prediction_share_of_strict_failures"])}) came from schema-invalid final runs; '
        f'{failures["valid_structured_field_failures"]} ({percent(failures["valid_structured_share_of_strict_failures"])}) occurred inside valid structured predictions.',
        "",
        "Mismatch-event counts inside valid structured failures (counts can overlap):",
        "",
        "| Event | Count |",
        "|---|---:|",
    ])
    for reason, count in failures["valid_failure_reason_events"].items():
        lines.append(f"| {reason} | {count} |")
    lines.extend([
        f'| citation issue (separate diagnostic) | {failures["citation_failures"]} |',
        "",
        "## Overall local field difficulty",
        "",
        "| Field | Correct across four models | Accuracy | Failures | Invalid-prediction failures |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in analysis["local_field_summary"]:
        lines.append(
            f'| {row["field"]} | {row["correct"]}/{row["total"]} '
            f'| {percent(row["accuracy"])} | {row["failures"]} '
            f'| {row["failure_reasons"].get("invalid_prediction", 0)} |'
        )

    lines.extend([
        "",
        "## Per-field strict accuracy",
        "",
        "| Model | Field | Correct | Accuracy | Invalid-prediction failures | Value failures | Unit failures | Status failures |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in analysis["per_field"]:
        invalid = row["failure_reasons"].get("invalid_prediction", 0)
        lines.append(
            f'| {row["model_id"]} | {row["field"]} | {row["correct"]}/{row["total"]} '
            f'| {percent(row["accuracy"])} | {invalid} '
            f'| {row["value_match_failures"]} | {row["unit_match_failures"]} '
            f'| {row["status_match_failures"]} |'
        )

    lines.extend([
        "",
        "## Local runtime and resource comparison",
        "",
        "| Model | Mean total duration (s) | Median duration (s) | Tokens/s | Peak VRAM (MiB) | Incremental peak (MiB) | Mean input tokens | Mean output tokens | Retry time share |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in locals_ranked:
        r = row["runtime"]
        lines.append(
            f'| {row["model_id"]} | {number(r.get("total_duration_s_mean"))} '
            f'| {number(r.get("total_duration_s_median"))} '
            f'| {number(r.get("tokens_per_second_mean"))} '
            f'| {number(r.get("gpu_memory_peak_used_mb_mean"))} '
            f'| {number(r.get("gpu_memory_incremental_peak_mb_mean"))} '
            f'| {number(r.get("input_tokens_mean"), 1)} '
            f'| {number(r.get("output_tokens_mean"), 1)} '
            f'| {percent(r.get("retry_overhead_share_of_total_duration"))} |'
        )
    lines.extend([
        "",
        "GPU peak-used values include the measured baseline; incremental peak is included to show observed growth during a run. Gemini GPU metrics are unavailable and are not compared.",
        "",
        "## Exploratory paired comparisons",
        "",
        "| Comparison | A correct/B wrong | A wrong/B correct | Discordant | Exact p-value |",
        "|---|---:|---:|---:|---:|",
    ])
    for test in analysis["paired_comparisons"]:
        lines.append(
            f'| {test["model_a"]} vs {test["model_b"]} '
            f'| {test["a_correct_b_wrong"]} | {test["a_wrong_b_correct"]} '
            f'| {test["discordant_pairs"]} | {test["exact_two_sided_p_value"]:.6f} |'
        )

    best = locals_ranked[0]
    second = locals_ranked[1]
    lines.extend([
        "",
        "These exact McNemar tests use 42 paired document-field decisions. Results are exploratory because the sample is small and fields within a report may not be independent.",
        "",
        "## Evidence-based selection",
        "",
        f'**Advance `{best["model_id"]}`** to the later automatic-selector/end-to-end experiment: it has the highest local strict field accuracy ({best["strict_correct_fields"]}/{best["total_fields"]}, {percent(best["strict_field_accuracy"])}).',
        "",
        f'The second-ranked local model is `{second["model_id"]}` at {second["strict_correct_fields"]}/{second["total_fields"]} ({percent(second["strict_field_accuracy"])}). Gemini achieved {cloud["strict_correct_fields"]}/{cloud["total_fields"]} ({percent(cloud["strict_field_accuracy"])}). Runtime/resource results should be treated as the deployment tradeoff, while strict controlled-context accuracy remains the primary advancement criterion.',
        "",
        "Failure-reason counts are mismatch events and may overlap when one field fails more than one component. Citation failures are diagnostic and do not change the frozen strict correctness metric.",
        "",
    ])
    return "\n".join(lines)


def analyze(local_root: Path, cloud_root: Path) -> dict[str, Any]:
    local_runs = load_runs(local_root, "local")
    cloud_runs = load_runs(cloud_root, "cloud")
    validate_experiment(local_runs, cloud_runs)
    all_runs = local_runs + cloud_runs
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in all_runs:
        grouped[run["model_id"]].append(run)
    models = [aggregate_model(model, runs) for model, runs in grouped.items()]
    models.sort(
        key=lambda row: (
            row["strict_field_accuracy"],
            row["final_schema_valid_rate"],
            row["components"]["citation_value_present"]["accuracy"] or -1,
        ),
        reverse=True,
    )
    locals_ranked = [row for row in models if row["source"] == "local"]
    cloud_model = next(row["model_id"] for row in models if row["source"] == "cloud")
    lookup = {(run["doc_id"], run["model_id"]): run for run in all_runs}
    paired = [
        exact_mcnemar(locals_ranked[0]["model_id"], cloud_model, lookup),
        exact_mcnemar(locals_ranked[0]["model_id"], locals_ranked[1]["model_id"], lookup),
    ]
    retry_count = sum(run["retry_used"] for run in local_runs)
    retry_recovered = sum(run["retry_valid"] is True for run in local_runs)
    analysis = {
        "experiment": {
            "name": "phase8_controlled_test_analysis",
            "prompt_version": local_runs[0]["prompt_version"],
            "system_prompt_sha256": local_runs[0]["system_prompt_sha256"],
            "local_runs": len(local_runs),
            "cloud_runs": len(cloud_runs),
            "documents": sorted(EXPECTED_DOCS),
            "fields": list(EXPECTED_FIELDS),
            "primary_invalid_policy": "final invalid prediction is strict 0/7",
            "raw_response_policy": "not parsed or reinterpreted",
        },
        "model_ranking": models,
        "local_schema_overall": {
            "runs": len(local_runs),
            "first_pass_valid_runs": sum(run["first_valid"] is True for run in local_runs),
            "first_pass_valid_rate": safe_rate(
                sum(run["first_valid"] is True for run in local_runs), len(local_runs)
            ),
            "retry_count": retry_count,
            "retry_recovered": retry_recovered,
            "retry_recovery_rate": safe_rate(retry_recovered, retry_count),
            "retry_failed": retry_count - retry_recovered,
            "final_valid_runs": sum(run["valid"] for run in local_runs),
            "final_valid_rate": safe_rate(sum(run["valid"] for run in local_runs), len(local_runs)),
            "first_pass_error_types": dict(sorted(Counter(
                run["first_error_type"] for run in local_runs if not run["first_valid"]
            ).items())),
            "failed_retry_error_types": dict(sorted(Counter(
                run["retry_error_type"]
                for run in local_runs if run["retry_used"] and not run["retry_valid"]
            ).items())),
        },
        "local_failure_summary": aggregate_local_failures(local_runs),
        "local_field_summary": aggregate_local_fields(local_runs),
        "per_field": aggregate_fields(all_runs),
        "paired_comparisons": paired,
    }
    return analysis


def write_outputs(analysis: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    (output_dir / "report.md").write_text(render_report(analysis), encoding="utf-8")

    ranking_rows = []
    component_rows = []
    runtime_rows = []
    schema_rows = []
    for model in analysis["model_ranking"]:
        ranking_rows.append({
            **model,
            "citation_accuracy": model["components"]["citation_value_present"]["accuracy"],
        })
        for component, verdict in model["components"].items():
            component_rows.append({"model_id": model["model_id"], "component": component, **verdict})
        if model["source"] == "local":
            runtime_rows.append({"model_id": model["model_id"], **model["runtime"]})
            schema_rows.append({
                "model_id": model["model_id"],
                "runs": model["runs"],
                "first_pass_valid_runs": model["first_pass_valid_runs"],
                "retry_count": model["retry_count"],
                "retry_recovered": model["retry_recovered"],
                "retry_failed": model["retry_failed"],
                "final_schema_valid_runs": model["final_schema_valid_runs"],
            })

    write_csv(
        output_dir / "model_ranking.csv",
        ranking_rows,
        ["model_id", "source", "strict_correct_fields", "total_fields", "strict_field_accuracy",
         "final_schema_valid_runs", "final_schema_valid_rate", "first_pass_valid_runs",
         "first_pass_valid_rate", "retry_count", "retry_recovered", "retry_recovery_rate",
         "citation_accuracy"],
    )
    write_csv(
        output_dir / "component_accuracy.csv",
        component_rows,
        ["model_id", "component", "correct", "applicable", "accuracy"],
    )
    per_field_csv = []
    for row in analysis["per_field"]:
        per_field_csv.append({
            **row,
            "failure_reasons": json.dumps(row["failure_reasons"], sort_keys=True),
        })
    write_csv(
        output_dir / "field_accuracy.csv",
        per_field_csv,
        ["model_id", "source", "field", "correct", "total", "accuracy", "failures",
         "failure_reasons", "status_match_failures", "value_match_failures",
         "unit_match_failures", "scope_match_failures", "source_year_match_failures",
         "citation_value_present_failures"],
    )
    local_field_csv = [
        {**row, "failure_reasons": json.dumps(row["failure_reasons"], sort_keys=True)}
        for row in analysis["local_field_summary"]
    ]
    write_csv(
        output_dir / "local_field_summary.csv",
        local_field_csv,
        ["field", "correct", "total", "accuracy", "failures", "failure_reasons"],
    )
    runtime_fields = ["model_id"] + sorted({key for row in runtime_rows for key in row if key != "model_id"})
    write_csv(output_dir / "runtime_comparison.csv", runtime_rows, runtime_fields)
    write_csv(
        output_dir / "schema_adherence.csv",
        schema_rows,
        ["model_id", "runs", "first_pass_valid_runs", "retry_count", "retry_recovered",
         "retry_failed", "final_schema_valid_runs"],
    )
    write_csv(
        output_dir / "paired_comparisons.csv",
        analysis["paired_comparisons"],
        ["model_a", "model_b", "paired_decisions", "both_correct", "a_correct_b_wrong",
         "a_wrong_b_correct", "both_wrong", "discordant_pairs", "exact_two_sided_p_value",
         "interpretation"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-root", type=Path, default=DEFAULT_LOCAL_ROOT,
        help="Local Phase-7 test artifact root",
    )
    parser.add_argument(
        "--cloud-root", type=Path, default=DEFAULT_CLOUD_ROOT,
        help="Cloud Phase-7 test artifact root",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Directory for Phase-8 outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis = analyze(args.local_root, args.cloud_root)
    write_outputs(analysis, args.output_dir)
    best_local = next(row for row in analysis["model_ranking"] if row["source"] == "local")
    schema = analysis["local_schema_overall"]
    print("Phase 8 analysis completed.")
    print(f"Output directory: {args.output_dir}")
    print(
        f'Best local model: {best_local["model_id"]} — '
        f'{best_local["strict_correct_fields"]}/{best_local["total_fields"]} '
        f'({100 * best_local["strict_field_accuracy"]:.1f}%)'
    )
    print(
        f'Local schema adherence: first pass {schema["first_pass_valid_runs"]}/{schema["runs"]}; '
        f'final {schema["final_valid_runs"]}/{schema["runs"]}; '
        f'retry recovery {schema["retry_recovered"]}/{schema["retry_count"]}'
    )


if __name__ == "__main__":
    main()
