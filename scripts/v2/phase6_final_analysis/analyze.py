from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "v2"
    / "phase6_final_analysis"
)

RAG_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "v2"
    / "phase3_rag_benchmark_v1_1"
    / "cold"
    / "test"
)

GEMINI_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "v2"
    / "phase5_gemini_full_pdf"
    / "test"
)

CONTROLLED_LOCAL_RESULTS = (
    PROJECT_ROOT
    / "artifacts"
    / "phase7_v1_1"
    / "test"
    / "results.jsonl"
)

CONTROLLED_GEMINI_RESULTS = (
    PROJECT_ROOT
    / "artifacts"
    / "phase7_v1_1"
    / "cloud"
    / "test"
    / "results.jsonl"
)

ANNOTATIONS_DIR = (
    PROJECT_ROOT
    / "data"
    / "annotations"
)

DOCS = [
    "amen_2024",
    "ATB_2023",
    "attijari_2024",
    "bh_2024",
    "bna_2024",
    "BT_2024",
]

FIELDS = [
    "total_assets",
    "total_equity",
    "net_banking_income",
    "operating_income",
    "net_income",
    "customer_deposits",
    "net_customer_loans",
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as handle:
        for line in handle:
            line = line.strip()

            if line:
                rows.append(
                    json.loads(line)
                )

    return rows


def mean(values):
    values = [
        value for value in values
        if value is not None
    ]

    if not values:
        return None

    return statistics.mean(values)


def median(values):
    values = [
        value for value in values
        if value is not None
    ]

    if not values:
        return None

    return statistics.median(values)


def percentage(
    numerator: float,
    denominator: float,
) -> float | None:

    if not denominator:
        return None

    return (
        numerator
        / denominator
        * 100.0
    )


def field_map(
    obj: dict[str, Any],
) -> dict[str, dict[str, Any]]:

    return {
        field["field"]: field
        for field in obj.get(
            "fields",
            []
        )
    }


def match_distractor(
    prediction_field: dict[str, Any],
    annotation_field: dict[str, Any],
) -> str | None:

    if (
        prediction_field.get("status")
        != "found"
    ):
        return None

    predicted_value = (
        prediction_field.get("value")
    )

    predicted_unit = (
        prediction_field.get(
            "unit_multiplier"
        )
    )

    if predicted_value is None:
        return None

    for distractor in (
        annotation_field.get(
            "distractors",
            []
        )
    ):

        if (
            distractor.get("value")
            == predicted_value
            and
            distractor.get(
                "unit_multiplier"
            )
            == predicted_unit
        ):
            return distractor.get(
                "reason",
                "known_distractor",
            )

    return None


def classify_failure(
    score_field: dict[str, Any],
    prediction_field: dict[str, Any] | None,
    annotation_field: dict[str, Any],
    valid_json: bool,
    field_evidence_recall: float | None,
    contamination_rate: float | None,
) -> tuple[str, str]:

    if not valid_json:
        return (
            "schema_failure",
            "Final response remained invalid after validation/retry.",
        )

    if prediction_field is None:
        return (
            "schema_failure",
            "No validated prediction exists for this field.",
        )

    distractor = match_distractor(
        prediction_field,
        annotation_field,
    )

    if distractor:

        reason_lower = (
            distractor.lower()
        )

        if "consolidated" in reason_lower:
            return (
                "scope_contamination",
                distractor,
            )

        if "prior_year" in reason_lower:
            return (
                "prior_year_distractor",
                distractor,
            )

        return (
            "semantic_distractor",
            distractor,
        )

    # With the current evaluation artifacts this is
    # report-level evidence recall. It is conservative:
    # only call retrieval failure when evidence recall
    # explicitly indicates missing evidence.
    if (
        field_evidence_recall
        is not None
        and field_evidence_recall < 1.0
    ):
        return (
            "possible_retrieval_failure",
            "Authoritative field evidence was not fully retrieved.",
        )
    failure_reasons = score_field.get(
        "failure_reasons",
        [],
    )

    if "unit_mismatch" in failure_reasons:
        return (
            "unit_mismatch",
            "Correct/near-correct value, but the reported unit did not match the annotation.",
        )

    if (
        contamination_rate
        is not None
        and contamination_rate > 0
    ):
        return (
            "generation_or_contamination",
            (
                "Correct evidence was retrieved, but the "
                "context also contained known distractor "
                "material."
            ),
        )

    return (
        "generation_failure",
        (
            "Correct evidence was available, but the "
            "model produced an incorrect field."
        ),
    )


def find_controlled_ministral(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    result = []

    for row in rows:

        model = str(
            row.get(
                "model_key",
                row.get(
                    "model_id",
                    "",
                ),
            )
        ).lower()

        if "ministral" in model:
            result.append(row)

    return result


def aggregate_simple(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:

    correct = sum(
        int(
            row.get(
                "fully_correct_fields",
                0,
            )
        )
        for row in rows
    )

    total = sum(
        int(
            row.get(
                "total_fields",
                0,
            )
        )
        for row in rows
    )

    valid = sum(
        1
        for row in rows
        if row.get("valid_json") is True
    )

    citations = [
        row.get("citation_accuracy")
        for row in rows
        if row.get(
            "citation_accuracy"
        ) is not None
    ]

    return {
        "documents": len(rows),
        "fully_correct_fields": correct,
        "total_fields": total,
        "strict_accuracy":
            correct / total
            if total
            else None,
        "valid_json_rate":
            valid / len(rows)
            if rows
            else None,
        "mean_report_citation_accuracy":
            mean(citations),
    }


def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rag_results = load_jsonl(
        RAG_DIR / "results.jsonl"
    )

    gemini_results = load_jsonl(
        GEMINI_DIR / "results.jsonl"
    )

    controlled_local_all = (
        load_jsonl(
            CONTROLLED_LOCAL_RESULTS
        )
    )

    controlled_local = (
        find_controlled_ministral(
            controlled_local_all
        )
    )

    controlled_gemini = (
        load_jsonl(
            CONTROLLED_GEMINI_RESULTS
        )
    )

    if len(rag_results) != 6:
        raise RuntimeError(
            "Expected 6 V1.1 RAG test results."
        )

    if len(gemini_results) != 6:
        raise RuntimeError(
            "Expected 6 full-PDF Gemini results."
        )

    rag_total_correct = sum(
        row["fully_correct_fields"]
        for row in rag_results
    )

    rag_total_fields = sum(
        row["total_fields"]
        for row in rag_results
    )

    # Guard against accidentally analyzing
    # the old 26/42 V1 RAG directory.
    if (
        rag_total_correct != 30
        or rag_total_fields != 42
    ):
        raise RuntimeError(
            "Wrong RAG artifact set. "
            f"Expected V1.1 = 30/42, got "
            f"{rag_total_correct}/"
            f"{rag_total_fields}."
        )

    gemini_total_correct = sum(
        row["fully_correct_fields"]
        for row in gemini_results
    )

    if gemini_total_correct != 41:
        raise RuntimeError(
            "Unexpected Gemini full-PDF "
            f"result: {gemini_total_correct}/42."
        )

    # -------------------------------------------------
    # Detailed local RAG operational metrics
    # -------------------------------------------------

    rag_runtime_rows = []
    rag_errors = []

    for doc_id in DOCS:

        run_dir = (
            RAG_DIR
            / f"{doc_id}__ministral3_3b"
        )

        meta = load_json(
            run_dir / "run_meta.json"
        )

        score = load_json(
            run_dir / "score.json"
        )

        annotation = load_json(
            ANNOTATIONS_DIR
            / f"{doc_id}.json"
        )

        prediction_path = (
            run_dir / "prediction.json"
        )

        prediction = (
            load_json(prediction_path)
            if prediction_path.exists()
            else None
        )

        runtime = meta.get(
            "runtime_metrics",
            {}
        )

        retrieval = meta.get(
            "retrieval_summary",
            {}
        )

        rag_runtime_rows.append(
            {
                "doc_id": doc_id,

                "valid_json":
                    meta.get("valid_json"),

                "first_pass_valid":
                    meta.get(
                        "first_pass_valid"
                    ),

                "retry_used":
                    meta.get(
                        "retry_used"
                    ),

                "end_to_end_seconds":
                    runtime.get(
                        "end_to_end_seconds"
                    ),

                "generation_seconds":
                    runtime.get(
                        "ministral_generation_wall_seconds"
                    ),

                "embedding_seconds":
                    runtime.get(
                        "embedding_seconds"
                    ),

                "retrieval_seconds":
                    runtime.get(
                        "retrieval_seconds"
                    ),

                "input_tokens":
                    runtime.get(
                        "input_tokens"
                    ),

                "output_tokens":
                    runtime.get(
                        "output_tokens"
                    ),

                "tokens_per_second":
                    runtime.get(
                        "generation_tokens_per_second"
                    ),

                "gpu_baseline_mb":
                    runtime.get(
                        "gpu_memory_baseline_mb"
                    ),

                "gpu_peak_mb":
                    runtime.get(
                        "gpu_memory_peak_used_mb"
                    ),

                "gpu_incremental_peak_mb":
                    runtime.get(
                        "gpu_memory_incremental_peak_mb"
                    ),

                "index_size_bytes":
                    runtime.get(
                        "faiss_index_size_bytes"
                    ),

                "authoritative_page_recall":
                    retrieval.get(
                        "authoritative_page_recall"
                    ),

                "field_evidence_recall":
                    retrieval.get(
                        "field_evidence_recall"
                    ),

                "contamination_rate":
                    retrieval.get(
                        "contamination_rate"
                    ),

                "fully_correct_fields":
                    meta.get(
                        "score_summary",
                        {},
                    ).get(
                        "fully_correct_fields"
                    ),
            }
        )

        score_fields = field_map(score)

        annotation_fields = field_map(
            annotation
        )

        prediction_fields = (
            field_map(prediction)
            if prediction is not None
            else {}
        )

        for field_name in FIELDS:

            scored = score_fields.get(
                field_name
            )

            if (
                scored is not None
                and scored.get("correct")
                is True
            ):
                continue

            predicted = (
                prediction_fields.get(
                    field_name
                )
            )

            category, detail = (
                classify_failure(
                    score_field=(
                        scored or {}
                    ),
                    prediction_field=(
                        predicted
                    ),
                    annotation_field=(
                        annotation_fields[
                            field_name
                        ]
                    ),
                    valid_json=bool(
                        meta.get(
                            "valid_json"
                        )
                    ),
                    field_evidence_recall=(
                        retrieval.get(
                            "field_evidence_recall"
                        )
                    ),
                    contamination_rate=(
                        retrieval.get(
                            "contamination_rate"
                        )
                    ),
                )
            )

            rag_errors.append(
                {
                    "doc_id": doc_id,
                    "field": field_name,
                    "category": category,
                    "detail": detail,

                    "failure_reasons":
                        ";".join(
                            (
                                scored
                                or {}
                            ).get(
                                "failure_reasons",
                                [],
                            )
                        ),

                    "predicted_status":
                        (
                            predicted
                            or {}
                        ).get(
                            "status"
                        ),

                    "predicted_value":
                        (
                            predicted
                            or {}
                        ).get(
                            "value"
                        ),

                    "expected_value":
                        annotation_fields[
                            field_name
                        ].get(
                            "value"
                        ),

                    "contamination_rate":
                        retrieval.get(
                            "contamination_rate"
                        ),
                }
            )

    # -------------------------------------------------
    # Gemini operational metrics + error analysis
    # -------------------------------------------------

    gemini_runtime_rows = []
    gemini_errors = []

    for doc_id in DOCS:

        run_dir = (
            GEMINI_DIR
            / f"{doc_id}__gemini_flash"
        )

        meta = load_json(
            run_dir / "run_meta.json"
        )

        score = load_json(
            run_dir / "score.json"
        )

        prediction = load_json(
            run_dir / "prediction.json"
        )

        annotation = load_json(
            ANNOTATIONS_DIR
            / f"{doc_id}.json"
        )

        runtime = meta.get(
            "runtime_metrics",
            {}
        )

        cost = meta.get(
            "cost",
            {}
        )

        gemini_runtime_rows.append(
            {
                "doc_id": doc_id,

                "pdf_pages":
                    meta.get(
                        "pdf_metadata",
                        {}
                    ).get(
                        "pdf_page_count"
                    ),

                "pdf_size_bytes":
                    meta.get(
                        "pdf_metadata",
                        {}
                    ).get(
                        "pdf_size_bytes"
                    ),

                "upload_latency_s":
                    runtime.get(
                        "upload_latency_s"
                    ),

                "generation_latency_s":
                    runtime.get(
                        "generation_latency_s"
                    ),

                "end_to_end_latency_s":
                    runtime.get(
                        "end_to_end_latency_s"
                    ),

                "input_tokens":
                    runtime.get(
                        "input_tokens"
                    ),

                "output_tokens":
                    runtime.get(
                        "output_tokens"
                    ),

                "thought_tokens":
                    runtime.get(
                        "thought_tokens"
                    ),

                "standard_cost_usd":
                    cost.get(
                        "theoretical_standard_usd"
                    ),

                "batch_cost_usd":
                    cost.get(
                        "theoretical_batch_usd"
                    ),
            }
        )

        score_fields = field_map(score)

        prediction_fields = field_map(
            prediction
        )

        annotation_fields = field_map(
            annotation
        )

        for field_name in FIELDS:

            scored = score_fields[
                field_name
            ]

            if scored.get("correct"):
                continue

            predicted = (
                prediction_fields[
                    field_name
                ]
            )

            annotation_field = (
                annotation_fields[
                    field_name
                ]
            )

            distractor = match_distractor(
                predicted,
                annotation_field,
            )

            gemini_errors.append(
                {
                    "doc_id": doc_id,
                    "field": field_name,

                    "predicted_value":
                        predicted.get(
                            "value"
                        ),

                    "expected_value":
                        annotation_field.get(
                            "value"
                        ),

                    "failure_reasons":
                        ";".join(
                            scored.get(
                                "failure_reasons",
                                [],
                            )
                        ),

                    "matched_distractor":
                        distractor,

                    "classification":
                        (
                            "semantic_distractor"
                            if distractor
                            else
                            "generation_failure"
                        ),
                }
            )

    # -------------------------------------------------
    # Headline conditions
    # -------------------------------------------------

    controlled_local_summary = (
        aggregate_simple(
            controlled_local
        )
    )

    controlled_gemini_summary = (
        aggregate_simple(
            controlled_gemini
        )
    )

    rag_summary = (
        aggregate_simple(
            rag_results
        )
    )

    gemini_summary = (
        aggregate_simple(
            gemini_results
        )
    )

    headline = [
        {
            "system":
                "Ministral 3B",
            "input":
                "Manual financial pages",
            "prompt":
                "V1.1",
            **controlled_local_summary,
        },
        {
            "system":
                "Gemini Flash",
            "input":
                "Manual financial pages",
            "prompt":
                "V1.1",
            **controlled_gemini_summary,
        },
        {
            "system":
                "RAG-4 + Ministral 3B",
            "input":
                "Complete report",
            "prompt":
                "V1.1",
            **rag_summary,
        },
        {
            "system":
                "Gemini Flash",
            "input":
                "Complete PDF",
            "prompt":
                "V1.1",
            **gemini_summary,
        },
    ]

    # -------------------------------------------------
    # Aggregated operations
    # -------------------------------------------------

    rag_operational = {
        "average_end_to_end_seconds":
            mean(
                [
                    x["end_to_end_seconds"]
                    for x in rag_runtime_rows
                ]
            ),

        "median_end_to_end_seconds":
            median(
                [
                    x["end_to_end_seconds"]
                    for x in rag_runtime_rows
                ]
            ),

        "average_generation_seconds":
            mean(
                [
                    x["generation_seconds"]
                    for x in rag_runtime_rows
                ]
            ),

        "average_embedding_seconds":
            mean(
                [
                    x["embedding_seconds"]
                    for x in rag_runtime_rows
                ]
            ),

        "average_tokens_per_second":
            mean(
                [
                    x["tokens_per_second"]
                    for x in rag_runtime_rows
                ]
            ),

        "maximum_gpu_peak_mb":
            max(
                x["gpu_peak_mb"]
                for x in rag_runtime_rows
                if x["gpu_peak_mb"]
                is not None
            ),

        "maximum_gpu_incremental_peak_mb":
            max(
                x[
                    "gpu_incremental_peak_mb"
                ]
                for x in rag_runtime_rows
                if x[
                    "gpu_incremental_peak_mb"
                ]
                is not None
            ),

        "average_input_tokens":
            mean(
                [
                    x["input_tokens"]
                    for x in rag_runtime_rows
                ]
            ),

        "average_output_tokens":
            mean(
                [
                    x["output_tokens"]
                    for x in rag_runtime_rows
                ]
            ),

        "average_authoritative_page_recall":
            mean(
                [
                    x[
                        "authoritative_page_recall"
                    ]
                    for x in rag_runtime_rows
                ]
            ),

        "average_field_evidence_recall":
            mean(
                [
                    x[
                        "field_evidence_recall"
                    ]
                    for x in rag_runtime_rows
                ]
            ),

        "average_contamination_rate":
            mean(
                [
                    x[
                        "contamination_rate"
                    ]
                    for x in rag_runtime_rows
                ]
            ),

        "valid_json_documents":
            sum(
                1
                for x in rag_runtime_rows
                if x["valid_json"]
            ),

        "total_documents":
            len(rag_runtime_rows),

        "system_ram_peak_mb":
            None,

        "system_ram_note":
            (
                "Not measured by the frozen "
                "V1 utilities; do not report "
                "as zero."
            ),
    }

    gemini_operational = {
        "average_end_to_end_seconds":
            mean(
                [
                    x[
                        "end_to_end_latency_s"
                    ]
                    for x in gemini_runtime_rows
                ]
            ),

        "median_end_to_end_seconds":
            median(
                [
                    x[
                        "end_to_end_latency_s"
                    ]
                    for x in gemini_runtime_rows
                ]
            ),

        "average_generation_seconds":
            mean(
                [
                    x[
                        "generation_latency_s"
                    ]
                    for x in gemini_runtime_rows
                ]
            ),

        "average_upload_seconds":
            mean(
                [
                    x[
                        "upload_latency_s"
                    ]
                    for x in gemini_runtime_rows
                ]
            ),

        "total_input_tokens":
            sum(
                x["input_tokens"] or 0
                for x in gemini_runtime_rows
            ),

        "average_input_tokens":
            mean(
                [
                    x["input_tokens"]
                    for x in gemini_runtime_rows
                ]
            ),

        "total_output_tokens":
            sum(
                x["output_tokens"] or 0
                for x in gemini_runtime_rows
            ),

        "average_output_tokens":
            mean(
                [
                    x["output_tokens"]
                    for x in gemini_runtime_rows
                ]
            ),

        "total_standard_cost_usd":
            sum(
                x[
                    "standard_cost_usd"
                ] or 0.0
                for x in gemini_runtime_rows
            ),

        "average_standard_cost_usd":
            mean(
                [
                    x[
                        "standard_cost_usd"
                    ]
                    for x in gemini_runtime_rows
                ]
            ),

        "total_batch_cost_usd":
            sum(
                x[
                    "batch_cost_usd"
                ] or 0.0
                for x in gemini_runtime_rows
            ),
    }

    # -------------------------------------------------
    # Research deltas
    # -------------------------------------------------

    local_manual_accuracy = (
        controlled_local_summary[
            "strict_accuracy"
        ]
    )

    local_rag_accuracy = (
        rag_summary[
            "strict_accuracy"
        ]
    )

    cloud_manual_accuracy = (
        controlled_gemini_summary[
            "strict_accuracy"
        ]
    )

    cloud_pdf_accuracy = (
        gemini_summary[
            "strict_accuracy"
        ]
    )

    deltas = {
        "local_retrieval_delta_percentage_points":
            (
                local_rag_accuracy
                - local_manual_accuracy
            ) * 100,

        "gemini_full_pdf_delta_percentage_points":
            (
                cloud_pdf_accuracy
                - cloud_manual_accuracy
            ) * 100,

        "complete_report_accuracy_gap_percentage_points":
            (
                cloud_pdf_accuracy
                - local_rag_accuracy
            ) * 100,
    }

    final = {
        "headline_conditions":
            headline,

        "rag_operational":
            rag_operational,

        "gemini_operational":
            gemini_operational,

        "deltas":
            deltas,

        "rag_error_count":
            len(rag_errors),

        "gemini_error_count":
            len(gemini_errors),
    }

    # Sanity checks.
    if len(rag_errors) != 12:
        raise RuntimeError(
            "Expected exactly 12 incorrect "
            "RAG fields for the 30/42 result, "
            f"found {len(rag_errors)}."
        )

    if len(gemini_errors) != 1:
        raise RuntimeError(
            "Expected exactly 1 incorrect "
            "Gemini field for the 41/42 result, "
            f"found {len(gemini_errors)}."
        )

    # -------------------------------------------------
    # Save JSON
    # -------------------------------------------------

    (
        OUTPUT_DIR
        / "analysis.json"
    ).write_text(
        json.dumps(
            final,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------
    # Save CSV helpers
    # -------------------------------------------------

    def save_csv(
        name: str,
        rows: list[dict[str, Any]],
    ) -> None:

        if not rows:
            return

        path = OUTPUT_DIR / name

        with path.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as handle:

            writer = csv.DictWriter(
                handle,
                fieldnames=list(
                    rows[0].keys()
                ),
            )

            writer.writeheader()
            writer.writerows(rows)

    save_csv(
        "headline_comparison.csv",
        headline,
    )

    save_csv(
        "rag_per_report.csv",
        rag_runtime_rows,
    )

    save_csv(
        "gemini_per_report.csv",
        gemini_runtime_rows,
    )

    save_csv(
        "rag_error_analysis.csv",
        rag_errors,
    )

    save_csv(
        "gemini_error_analysis.csv",
        gemini_errors,
    )

    # -------------------------------------------------
    # Console summary
    # -------------------------------------------------

    print(
        "\n"
        + "=" * 72
    )

    print(
        "V2 FINAL ANALYSIS"
    )

    print(
        "=" * 72
    )

    for row in headline:

        accuracy = (
            row["strict_accuracy"]
            * 100
        )

        print(
            f"{row['system']:25s} | "
            f"{row['input']:25s} | "
            f"{row['fully_correct_fields']}/"
            f"{row['total_fields']} "
            f"({accuracy:.1f}%)"
        )

    print(
        "\nLocal retrieval delta: "
        f"{deltas['local_retrieval_delta_percentage_points']:.1f} pp"
    )

    print(
        "Gemini full-PDF delta: "
        f"{deltas['gemini_full_pdf_delta_percentage_points']:.1f} pp"
    )

    print(
        "Complete-report accuracy gap: "
        f"{deltas['complete_report_accuracy_gap_percentage_points']:.1f} pp"
    )

    print(
        "\nRAG errors classified:",
        len(rag_errors),
    )

    print(
        "Gemini errors classified:",
        len(gemini_errors),
    )

    print(
        "\nRAG avg cold latency:",
        round(
            rag_operational[
                "average_end_to_end_seconds"
            ],
            2,
        ),
        "s",
    )

    print(
        "RAG max GPU peak:",
        round(
            rag_operational[
                "maximum_gpu_peak_mb"
            ],
            1,
        ),
        "MB",
    )

    print(
        "Gemini avg full-PDF latency:",
        round(
            gemini_operational[
                "average_end_to_end_seconds"
            ],
            2,
        ),
        "s",
    )

    print(
        "Gemini total standard cost:",
        f"${gemini_operational['total_standard_cost_usd']:.6f}",
    )

    print(
        "\nOutput:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()