"""Read-only adapters for the designated final V1.1 and V2 analyses."""

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .paths import RAG_CONFIG, V1_ANALYSIS, V2_ANALYSIS


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _number(value: str | None) -> int | float | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return float(value)


def _bool(value: str | None) -> bool | None:
    if value is None or value == "":
        return None
    return value.lower() == "true"


def headline() -> list[dict[str, Any]]:
    rows = _csv(V2_ANALYSIS / "headline_comparison.csv")
    return [{
        "system": row["system"], "input": row["input"],
        "prompt": row["prompt"], "documents": int(row["documents"]),
        "correct": int(row["fully_correct_fields"]),
        "total": int(row["total_fields"]),
        "accuracy": float(row["strict_accuracy"]),
        "valid_json_rate": float(row["valid_json_rate"]),
        "citation_accuracy": float(row["mean_report_citation_accuracy"]),
    } for row in rows]


def models() -> list[dict[str, Any]]:
    analysis = _json(V1_ANALYSIS / "analysis.json")
    return [{
        "id": row["model_id"], "source": row["source"],
        "model_tag": row.get("model_tag"),
        "correct": row["strict_correct_fields"],
        "total": row["total_fields"],
        "accuracy": row["strict_field_accuracy"],
        "valid_json_rate": row["final_schema_valid_rate"],
        "first_pass_valid_rate": row.get("first_pass_valid_rate"),
        "citation_accuracy": row.get("components", {}).get("citation_value_present", {}).get("accuracy"),
        "latency_seconds": row.get("runtime", {}).get("total_duration_s_mean", row.get("runtime", {}).get("latency_s_mean")),
        "tokens_per_second": row.get("runtime", {}).get("tokens_per_second_mean"),
        "peak_vram_mb": row.get("runtime", {}).get("gpu_memory_peak_used_mb_mean"),
        "selected_for_v2": row["model_id"] == "ministral3_3b",
    } for row in analysis["model_ranking"]]


def errors() -> dict[str, Any]:
    rows = _csv(V2_ANALYSIS / "rag_error_analysis.csv")
    counts = Counter(row["category"] for row in rows)
    return {"counts": [{"category": k, "count": v} for k, v in counts.items()], "rows": rows}


def reports() -> list[dict[str, Any]]:
    result = []
    for row in _csv(V2_ANALYSIS / "rag_per_report.csv"):
        result.append({
            "report": row["doc_id"], "system": "RAG-4 + Ministral 3B",
            "input": "Complete report", "correct": int(row["fully_correct_fields"]),
            "total": 7, "accuracy": int(row["fully_correct_fields"]) / 7,
            "valid_json": _bool(row["valid_json"]),
            "first_pass_valid": _bool(row["first_pass_valid"]),
            "latency_seconds": _number(row["end_to_end_seconds"]),
            "theoretical_standard_cost_usd": None,
        })
    for row in _csv(V2_ANALYSIS / "gemini_per_report.csv"):
        score_path = V2_ANALYSIS.parent / "phase5_gemini_full_pdf" / "test" / f"{row['doc_id']}__gemini_flash" / "score.json"
        score = _json(score_path)["summary"] if score_path.exists() else {}
        result.append({
            "report": row["doc_id"], "system": "Gemini Flash",
            "input": "Complete PDF", "correct": score.get("fully_correct_fields"), "total": 7,
            "accuracy": score.get("fully_correct_rate"), "valid_json": score.get("valid_json"), "first_pass_valid": None,
            "latency_seconds": _number(row["end_to_end_latency_s"]),
            "theoretical_standard_cost_usd": _number(row["standard_cost_usd"]),
        })
    return result


def summary() -> dict[str, Any]:
    analysis = _json(V2_ANALYSIS / "analysis.json")
    import yaml
    rag = yaml.safe_load(RAG_CONFIG.read_text(encoding="utf-8"))
    rag_reports = _csv(V2_ANALYSIS / "rag_per_report.csv")
    local_operational = dict(analysis["rag_operational"])
    local_operational["average_retrieval_seconds"] = sum(float(row["retrieval_seconds"]) for row in rag_reports) / len(rag_reports)
    return {
        "headline": headline(), "models": models(), "errors": errors(),
        "reports": reports(),
        "retrieval": {
            "configuration": "RAG-4", "embedding_model": rag["embedding_model"],
            "chunking": rag["chunking"], "pages_per_query": rag["pages_per_query"],
            "maximum_pages": rag["maximum_pages_before_deduplication"],
            "candidate_k": rag["candidate_k"],
            "authoritative_page_recall": analysis["rag_operational"]["average_authoritative_page_recall"],
            "field_evidence_recall": analysis["rag_operational"]["average_field_evidence_recall"],
            "contamination_rate": analysis["rag_operational"]["average_contamination_rate"],
        },
        "local_operational": local_operational,
        "cloud_operational": analysis["gemini_operational"],
        "sources": [
            "artifacts/v2/phase6_final_analysis/headline_comparison.csv",
            "artifacts/phase8_v1_1/analysis.json",
            "artifacts/v2/phase6_final_analysis/rag_error_analysis.csv",
            "artifacts/v2/phase6_final_analysis/rag_per_report.csv",
            "artifacts/v2/phase6_final_analysis/gemini_per_report.csv",
            "artifacts/v2/phase5_gemini_full_pdf/test/*/score.json",
            "artifacts/v2/phase6_final_analysis/analysis.json",
            "configs/rag_v1.yaml",
        ],
    }
