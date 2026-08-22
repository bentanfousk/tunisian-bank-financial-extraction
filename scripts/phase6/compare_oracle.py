from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_score(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Score file not found: {path}"
        )

    # utf-8-sig also accepts files written by
    # PowerShell with a UTF-8 BOM.
    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise TypeError(
            f"Expected JSON object in {path}."
        )

    return data


def index_fields(
    score: dict[str, Any],
    label: str,
) -> dict[str, dict[str, Any]]:
    fields = score.get("fields")

    if not isinstance(fields, list):
        raise ValueError(
            f"{label} score is missing "
            "a valid 'fields' list."
        )

    indexed: dict[str, dict[str, Any]] = {}

    for field in fields:
        if not isinstance(field, dict):
            raise TypeError(
                f"{label} contains a field "
                "that is not an object."
            )

        field_name = field.get("field")
        correct = field.get("correct")

        if not isinstance(field_name, str):
            raise ValueError(
                f"{label} field has invalid name."
            )

        if not isinstance(correct, bool):
            raise ValueError(
                f"{label} field {field_name!r} "
                "has invalid 'correct' verdict."
            )

        if field_name in indexed:
            raise ValueError(
                f"{label} contains duplicate field "
                f"{field_name!r}."
            )

        indexed[field_name] = field

    return indexed

def get_valid_json(
    score: dict[str, Any],
    label: str,
) -> bool:
    summary = score.get("summary")

    if not isinstance(summary, dict):
        raise ValueError(
            f"{label} score is missing "
            "a valid 'summary' object."
        )

    valid_json = summary.get("valid_json")

    if not isinstance(valid_json, bool):
        raise ValueError(
            f"{label} score has invalid "
            "'valid_json' status."
        )

    return valid_json


def classify_transition(
    pipeline_correct: bool,
    oracle_correct: bool,
) -> str:
    """
    Interpret one field's pipeline/oracle transition.

    wrong -> right:
        context/page-selection failure

    wrong -> wrong:
        generation/extraction failure

    right -> right:
        correct in both

    right -> wrong:
        oracle regression; do not attribute this
        to page selection or generation automatically.
    """

    if pipeline_correct and oracle_correct:
        return "correct_both"

    if (
        not pipeline_correct
        and oracle_correct
    ):
        return "context_failure"

    if (
        not pipeline_correct
        and not oracle_correct
    ):
        return "generation_failure"

    return "oracle_regression"


def compare_scores(
    pipeline_score: dict[str, Any],
    oracle_score: dict[str, Any],
) -> dict[str, Any]:
    pipeline_doc_id = pipeline_score.get(
        "doc_id"
    )

    oracle_doc_id = oracle_score.get(
        "doc_id"
    )

    if pipeline_doc_id != oracle_doc_id:
        raise ValueError(
            "Pipeline and oracle scores refer "
            "to different documents: "
            f"{pipeline_doc_id!r} vs "
            f"{oracle_doc_id!r}."
        )

    pipeline_valid = get_valid_json(
        pipeline_score,
        "pipeline",
    )

    oracle_valid = get_valid_json(
        oracle_score,
        "oracle",
    )

    # Field-level attribution is meaningful only
    # when both model responses passed the frozen
    # schema and therefore have trustworthy
    # Phase-5 field verdicts.
    if not pipeline_valid or not oracle_valid:

        if (
            not pipeline_valid
            and oracle_valid
        ):
            comparison_status = (
                "pipeline_invalid_oracle_valid"
            )

        elif (
            not pipeline_valid
            and not oracle_valid
        ):
            comparison_status = (
                "both_invalid"
            )

        else:
            comparison_status = (
                "pipeline_valid_oracle_invalid"
            )

        return {
            "doc_id": pipeline_doc_id,
            "fields": [],
            "summary": {
                "comparison_status": (
                    comparison_status
                ),
                "field_comparison_available": False,
                "pipeline_valid_json": (
                    pipeline_valid
                ),
                "oracle_valid_json": (
                    oracle_valid
                ),
                "total_fields": None,
                "correct_both": None,
                "context_failure": None,
                "generation_failure": None,
                "oracle_regression": None,
            },
        }

    pipeline_fields = index_fields(
        pipeline_score,
        "pipeline",
    )

    oracle_fields = index_fields(
        oracle_score,
        "oracle",
    )

    if set(pipeline_fields) != set(oracle_fields):
        raise ValueError(
            "Pipeline and oracle scores do not "
            "contain the same field set."
        )

    comparisons: list[dict[str, Any]] = []

    counts = {
        "correct_both": 0,
        "context_failure": 0,
        "generation_failure": 0,
        "oracle_regression": 0,
    }

    # Preserve Phase-5 field order.
    for field in pipeline_score["fields"]:
        field_name = field["field"]

        pipeline_correct = (
            pipeline_fields[
                field_name
            ]["correct"]
        )

        oracle_correct = (
            oracle_fields[
                field_name
            ]["correct"]
        )

        classification = classify_transition(
            pipeline_correct,
            oracle_correct,
        )

        counts[
            classification
        ] += 1

        comparisons.append(
            {
                "field": field_name,
                "pipeline_correct": (
                    pipeline_correct
                ),
                "oracle_correct": (
                    oracle_correct
                ),
                "classification": (
                    classification
                ),
            }
        )

    return {
        "doc_id": pipeline_doc_id,
        "fields": comparisons,
        "summary": {
            "comparison_status": "both_valid",
            "field_comparison_available": True,
            "pipeline_valid_json": True,
            "oracle_valid_json": True,
            "total_fields": len(comparisons),
            **counts,
        },
    }

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare frozen Phase-5 field verdicts "
            "for a normal pipeline run and its "
            "Phase-6 oracle run."
        )
    )

    parser.add_argument(
        "--pipeline-score",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--oracle-score",
        required=True,
        type=Path,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    pipeline_score = load_score(
        args.pipeline_score
    )

    oracle_score = load_score(
        args.oracle_score
    )

    result = compare_scores(
        pipeline_score=pipeline_score,
        oracle_score=oracle_score,
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()