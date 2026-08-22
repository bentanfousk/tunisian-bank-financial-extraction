from pathlib import Path
from typing import Any

import mlflow


REPO_ROOT = Path(__file__).resolve().parents[2]

MLFLOW_DIR = (
    REPO_ROOT
    / "artifacts"
    / "mlflow"
)

MLFLOW_DB_PATH = (
    MLFLOW_DIR
    / "mlflow.db"
)

MLFLOW_ARTIFACTS_DIR = (
    MLFLOW_DIR
    / "artifacts"
)

TRACKING_URI = (
    f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
)

ARTIFACT_LOCATION = (
    MLFLOW_ARTIFACTS_DIR
    .resolve()
    .as_uri()
)

EXPERIMENT_NAME = "track_a_v1_dev"


def configure_mlflow(
    experiment_name: str = EXPERIMENT_NAME,
) -> None:
    """
    Configure the local MLflow tracking backend.
    """

    MLFLOW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MLFLOW_ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    mlflow.set_tracking_uri(
        TRACKING_URI
    )

    experiment = (
        mlflow.get_experiment_by_name(
            experiment_name
        )
    )

    if experiment is None:
        mlflow.create_experiment(
            name=experiment_name,
            artifact_location=(
                ARTIFACT_LOCATION
            ),
        )

    mlflow.set_experiment(
        experiment_name
    )


def log_extraction_run(
    doc_id: str,
    model_id: str,
    model_digest: str,
    bank: str,
    target_year: int,
    selection: dict[str, Any],
    generation_settings: dict[str, Any],
    artifact_hashes: dict[str, str],
    generation_result: dict[str, Any],
    validation: dict[str, Any],
    retry_info: dict[str, Any] | None = None,
    experiment_name: str = EXPERIMENT_NAME,
) -> str:
    """
    Log one successful model-generation attempt.

    "Successful" here means the model call itself
    completed. The final JSON may still be invalid.

    If retry_info is provided, the MLflow run also
    preserves:
      - first-pass response,
      - first-pass validation,
      - whether a repair retry was used,
      - retry response,
      - retry validation.

    The normal artifacts:
      raw_response.txt
      validation.json
      prediction.json

    always represent the FINAL response after the
    optional one-retry policy.

    Returns the MLflow run ID.
    """

    configure_mlflow(experiment_name)

    run_name = (
        f"{doc_id}__{model_id}"
    )

    # -----------------------------------------
    # Retry-policy metadata
    # -----------------------------------------

    retry_used = bool(
        retry_info
        and retry_info.get(
            "retry_used"
        )
    )

    if retry_info is not None:
        first_validation = (
            retry_info[
                "first_validation"
            ]
        )

        first_pass_valid = bool(
            first_validation[
                "valid"
            ]
        )

    else:
        first_validation = None

        first_pass_valid = bool(
            validation["valid"]
        )

    with mlflow.start_run(
        run_name=run_name
    ) as run:

        # -------------------------------------
        # Fixed experimental configuration
        # -------------------------------------

        mlflow.log_params(
            {
                "model_id": (
                    model_id
                ),
                "model_digest": (
                    model_digest
                ),
                **generation_settings,
            }
        )

        # -------------------------------------
        # Document / run metadata
        # -------------------------------------

        mlflow.set_tags(
            {
                "doc_id": (
                    doc_id
                ),

                "bank": (
                    bank
                ),

                "model_digest": (
                    model_digest
                ),

                **artifact_hashes,

                "target_year": str(
                    target_year
                ),

                "page_set": (
                    selection[
                        "page_set"
                    ]
                ),

                "report_pages": ",".join(
                    str(page)
                    for page in selection[
                        "report_pages"
                    ]
                ),

                "pdf_pages": ",".join(
                    str(page)
                    for page in selection[
                        "pdf_pages"
                    ]
                ),

                # Final response validity.
                "valid": str(
                    validation[
                        "valid"
                    ]
                ).lower(),

                # First extraction attempt.
                "first_pass_valid": str(
                    first_pass_valid
                ).lower(),

                # Whether the deterministic
                # repair call was needed.
                "retry_used": str(
                    retry_used
                ).lower(),

                "done_reason": str(
                    generation_result[
                        "done_reason"
                    ]
                ),

                "error_type": str(
                    validation[
                        "error_type"
                    ]
                ),
            }
        )

        # -------------------------------------
        # Final / total numeric measurements
        # -------------------------------------
        #
        # If a repair retry happened,
        # generation_result contains the combined
        # cost of:
        #
        #   first extraction + repair generation
        #
        # This means these metrics represent the
        # actual total cost of producing the final
        # answer.

        mlflow.log_metrics(
            {
                "input_tokens": (
                    generation_result[
                        "input_tokens"
                    ]
                ),

                "output_tokens": (
                    generation_result[
                        "output_tokens"
                    ]
                ),

                "total_duration_s": (
                    generation_result[
                        "total_duration_ns"
                    ]
                    / 1_000_000_000
                ),

                "load_duration_s": (
                    generation_result[
                        "load_duration_ns"
                    ]
                    / 1_000_000_000
                ),

                "prompt_eval_duration_s": (
                    generation_result[
                        "prompt_eval_duration_ns"
                    ]
                    / 1_000_000_000
                ),

                "generation_duration_s": (
                    generation_result[
                        "eval_duration_ns"
                    ]
                    / 1_000_000_000
                ),

                "tokens_per_second": (
                    generation_result[
                        "tokens_per_second"
                    ]
                ),

                "gpu_memory_baseline_mb": (
                    generation_result[
                        "gpu_memory_baseline_mb"
                    ]
                ),

                "gpu_memory_peak_used_mb": (
                    generation_result[
                        "gpu_memory_peak_used_mb"
                    ]
                ),

                "gpu_memory_incremental_peak_mb": (
                    generation_result[
                        "gpu_memory_incremental_peak_mb"
                    ]
                ),
            }
        )

        # -------------------------------------
        # Retry-specific measurements
        # -------------------------------------

        if retry_info is not None:

            first_result = (
                retry_info[
                    "first_result"
                ]
            )

            first_metrics = {
                "first_input_tokens": (
                    first_result[
                        "input_tokens"
                    ]
                ),

                "first_output_tokens": (
                    first_result[
                        "output_tokens"
                    ]
                ),

                "first_total_duration_s": (
                    first_result[
                        "total_duration_ns"
                    ]
                    / 1_000_000_000
                ),

                "first_generation_duration_s": (
                    first_result[
                        "eval_duration_ns"
                    ]
                    / 1_000_000_000
                ),
            }

            mlflow.log_metrics(
                first_metrics
            )

            if retry_used:

                retry_result = (
                    retry_info[
                        "retry_result"
                    ]
                )

                retry_metrics = {
                    "retry_input_tokens": (
                        retry_result[
                            "input_tokens"
                        ]
                    ),

                    "retry_output_tokens": (
                        retry_result[
                            "output_tokens"
                        ]
                    ),

                    "retry_total_duration_s": (
                        retry_result[
                            "total_duration_ns"
                        ]
                        / 1_000_000_000
                    ),

                    "retry_generation_duration_s": (
                        retry_result[
                            "eval_duration_ns"
                        ]
                        / 1_000_000_000
                    ),
                }

                mlflow.log_metrics(
                    retry_metrics
                )

        # -------------------------------------
        # Final response
        # -------------------------------------
        #
        # This is the response that is actually
        # passed to Phase 5.

        mlflow.log_text(
            generation_result[
                "text"
            ],
            "raw_response.txt",
        )

        # -------------------------------------
        # First-pass artifacts
        # -------------------------------------

        if retry_info is not None:

            first_result = (
                retry_info[
                    "first_result"
                ]
            )

            first_validation = (
                retry_info[
                    "first_validation"
                ]
            )

            mlflow.log_text(
                first_result[
                    "text"
                ],
                "first_raw_response.txt",
            )

            mlflow.log_dict(
                {
                    "valid": (
                        first_validation[
                            "valid"
                        ]
                    ),

                    "error_type": (
                        first_validation[
                            "error_type"
                        ]
                    ),

                    "error_message": (
                        first_validation[
                            "error_message"
                        ]
                    ),
                },
                "first_validation.json",
            )

        # -------------------------------------
        # Retry artifacts
        # -------------------------------------

        if (
            retry_info is not None
            and retry_used
        ):

            retry_result = (
                retry_info[
                    "retry_result"
                ]
            )

            retry_validation = (
                retry_info[
                    "retry_validation"
                ]
            )

            mlflow.log_text(
                retry_result[
                    "text"
                ],
                "retry_raw_response.txt",
            )

            mlflow.log_dict(
                {
                    "valid": (
                        retry_validation[
                            "valid"
                        ]
                    ),

                    "error_type": (
                        retry_validation[
                            "error_type"
                        ]
                    ),

                    "error_message": (
                        retry_validation[
                            "error_message"
                        ]
                    ),
                },
                "retry_validation.json",
            )

        # -------------------------------------
        # Final validation
        # -------------------------------------

        mlflow.log_dict(
            {
                "valid": (
                    validation[
                        "valid"
                    ]
                ),

                "error_type": (
                    validation[
                        "error_type"
                    ]
                ),

                "error_message": (
                    validation[
                        "error_message"
                    ]
                ),
            },
            "validation.json",
        )

        # -------------------------------------
        # Final prediction
        # -------------------------------------
        #
        # As before, prediction.json exists only
        # when the FINAL response passes the frozen
        # schema.

        if validation["valid"]:

            mlflow.log_dict(
                validation[
                    "parsed"
                ],
                "prediction.json",
            )

        return run.info.run_id


def log_generation_failure(
    doc_id: str,
    model_id: str,
    model_digest: str,
    bank: str,
    target_year: int,
    selection: dict[str, Any],
    generation_settings: dict[str, Any],
    artifact_hashes: dict[str, str],
    error: Exception,
    experiment_name: str = EXPERIMENT_NAME,
) -> str:
    """
    Log an actual generation/API failure.

    This means model generation itself failed,
    rather than merely producing schema-invalid JSON.
    """

    configure_mlflow(experiment_name)

    run_name = (
        f"{doc_id}__{model_id}"
    )

    with mlflow.start_run(
        run_name=run_name
    ) as run:

        mlflow.log_params(
            {
                "model_id": (
                    model_id
                ),

                "model_digest": (
                    model_digest
                ),

                **generation_settings,
            }
        )

        mlflow.set_tags(
            {
                "doc_id": (
                    doc_id
                ),

                "bank": (
                    bank
                ),

                "model_digest": (
                    model_digest
                ),

                **artifact_hashes,

                "target_year": str(
                    target_year
                ),

                "page_set": (
                    selection[
                        "page_set"
                    ]
                ),

                "report_pages": ",".join(
                    str(page)
                    for page in selection[
                        "report_pages"
                    ]
                ),

                "pdf_pages": ",".join(
                    str(page)
                    for page in selection[
                        "pdf_pages"
                    ]
                ),

                "valid": "false",

                "failure_stage": (
                    "generation"
                ),

                "error_type": (
                    type(error).__name__
                ),
            }
        )

        mlflow.log_text(
            str(error),
            "error.txt",
        )

        return run.info.run_id


if __name__ == "__main__":

    configure_mlflow()

    experiment = (
        mlflow.get_experiment_by_name(
            EXPERIMENT_NAME
        )
    )

    print(
        "Tracking URI:",
        mlflow.get_tracking_uri(),
    )

    print(
        "Experiment:",
        experiment.name,
    )

    print(
        "Artifact location:",
        experiment.artifact_location,
    )