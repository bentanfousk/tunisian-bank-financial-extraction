from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from google import genai

from scripts.phase2_page_select.page_select import select_pages
from scripts.benchmark_versions import get_benchmark_version
from scripts.phase4.prompt_builder import build_messages
from scripts.phase4.response_validation import parse_and_validate


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"

REPORTS_TEXT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "reports_txt"
)

FIELD_SCHEMA_PATH = (
    PROJECT_ROOT
    / "schemas"
    / "field_schema.json"
)

RUN_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "run.yaml"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "manifests"
    / "page_manifest.json"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "phase7"
    / "cloud"
)


# Frozen cloud baseline for Track A v1.
MODEL_ID = "gemini-3.6-flash"
MODEL_LABEL = "gemini_flash"


def load_json(
    path: Path,
) -> dict[str, Any]:

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise TypeError(
            f"Expected JSON object in {path}"
        )

    return data


def load_yaml(
    path: Path,
) -> dict[str, Any]:

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise TypeError(
            f"Expected YAML object in {path}"
        )

    return data


def write_json(
    path: Path,
    data: Any,
) -> None:

    path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as f:

        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def sha256_json(
    data: Any,
) -> str:

    canonical = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    return hashlib.sha256(
        canonical
    ).hexdigest()


def parse_doc_id(
    doc_id: str,
) -> tuple[str, int]:

    bank, year_text = doc_id.rsplit(
        "_",
        1,
    )

    return (
        bank,
        int(year_text),
    )


def read_split(
    split_name: str,
) -> list[str]:

    path = (
        SPLITS_DIR
        / f"{split_name}.txt"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}"
        )

    documents = []

    for raw_line in path.read_text(
        encoding="utf-8"
    ).splitlines():

        line = (
            raw_line
            .split("#", 1)[0]
            .strip()
        )

        if line:
            documents.append(line)

    return documents


def build_generation_schema() -> dict[str, Any]:
    field_names = [
        "total_assets",
        "total_equity",
        "net_banking_income",
        "operating_income",
        "net_income",
        "customer_deposits",
        "net_customer_loans",
    ]

    return {
        "type": "object",
        "properties": {
            "doc_id": {
                "type": "string",
            },
            "target_year": {
                "type": "integer",
            },
            "target_scope": {
                "type": "string",
                "enum": ["individual"],
            },
            "currency": {
                "type": "string",
                "enum": ["TND"],
            },
            "fields": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "field": {
                            "type": "string",
                            "enum": field_names,
                        },
                        "status": {
                            "type": "string",
                            "enum": [
                                "found",
                                "not_found",
                                "ambiguous",
                            ],
                        },
                        "value": {
                            "type": [
                                "number",
                                "null",
                            ],
                        },
                        "unit_multiplier": {
                            "type": [
                                "integer",
                                "null",
                            ],
                        },
                        "scope": {
                            "type": "string",
                        },
                        "source_year": {
                            "type": "integer",
                        },
                        "page": {
                            "type": [
                                "integer",
                                "null",
                            ],
                        },
                        "evidence": {
                            "type": "string",
                        },
                    },
                    "required": [
                        "field",
                        "status",
                        "value",
                        "unit_multiplier",
                        "scope",
                        "source_year",
                        "page",
                        "evidence",
                    ],
                },
            },
        },
        "required": [
            "doc_id",
            "target_year",
            "target_scope",
            "currency",
            "fields",
        ],
    }
FIELD_SCHEMA = load_json(
    FIELD_SCHEMA_PATH
)

RUN_CONFIG = load_yaml(
    RUN_CONFIG_PATH
)

GENERATION_SETTINGS = (
    RUN_CONFIG.get(
        "generation"
    )
)

if not isinstance(
    GENERATION_SETTINGS,
    dict,
):
    raise ValueError(
        "configs/run.yaml must contain "
        "a 'generation' mapping."
    )


GENERATION_SCHEMA = (
    build_generation_schema()
)


def build_cloud_artifact_hashes(
    prompt_version: str = "v1",
) -> dict[str, str]:
    version = get_benchmark_version(
        prompt_version
    )

    return {
        "system_prompt_sha256":
            sha256_file(
                version.system_prompt_path
            ),

        "user_template_sha256":
            sha256_file(
                version.user_template_path
            ),

        "field_schema_sha256":
            sha256_file(
                FIELD_SCHEMA_PATH
            ),

        "run_config_sha256":
            sha256_file(
                RUN_CONFIG_PATH
            ),

        "cloud_generation_schema_sha256":
            sha256_json(
                GENERATION_SCHEMA
            ),
    }


def get_cloud_output_root(
    prompt_version: str = "v1",
) -> Path:
    return (
        get_benchmark_version(
            prompt_version
        )
        .cloud_output_root
    )


# Backward-compatible frozen V1 hashes for existing imports.
ARTIFACT_HASHES = build_cloud_artifact_hashes(
    "v1"
)


def validate_inputs(
    doc_id: str,
) -> None:

    required_paths = [

        ANNOTATIONS_DIR
        / f"{doc_id}.json",

        REPORTS_TEXT_DIR
        / f"{doc_id}.jsonl",

        MANIFEST_PATH,
    ]

    missing = [
        str(path)
        for path in required_paths
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required cloud baseline "
            "input(s):\n"
            + "\n".join(missing)
        )


def is_completed(
    run_dir: Path,
) -> bool:

    meta_path = (
        run_dir
        / "run_meta.json"
    )

    score_path = (
        run_dir
        / "score.json"
    )

    if (
        not meta_path.exists()
        or not score_path.exists()
    ):
        return False

    try:
        meta = load_json(
            meta_path
        )

    except Exception:
        return False

    return (
        meta.get("status")
        == "completed"
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

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    (
        run_dir
        / "phase5_stdout.txt"
    ).write_text(
        result.stdout,
        encoding="utf-8",
    )

    (
        run_dir
        / "phase5_stderr.txt"
    ).write_text(
        result.stderr,
        encoding="utf-8",
    )

    if result.returncode != 0:

        raise RuntimeError(
            "Frozen Phase-5 scorer failed. "
            "See phase5_stderr.txt."
        )

    try:
        return json.loads(
            result.stdout
        )

    except json.JSONDecodeError as exc:

        raise RuntimeError(
            "Phase 5 exited successfully "
            "but did not return valid JSON."
        ) from exc


def interaction_to_dict(
    interaction: Any,
) -> dict[str, Any]:
    """
    Store the complete provider response when
    supported by the installed SDK.
    """

    if hasattr(
        interaction,
        "model_dump",
    ):

        try:

            data = (
                interaction.model_dump(
                    mode="json",
                    exclude_none=True,
                )
            )

            if isinstance(
                data,
                dict,
            ):
                return data

        except Exception:
            pass

    return {

        "id": getattr(
            interaction,
            "id",
            None,
        ),

        "model": getattr(
            interaction,
            "model",
            None,
        ),

        "status": str(
            getattr(
                interaction,
                "status",
                None,
            )
        ),

        "output_text": getattr(
            interaction,
            "output_text",
            None,
        ),
    }


def usage_value(
    usage: Any,
    field_name: str,
):

    if usage is None:
        return None

    return getattr(
        usage,
        field_name,
        None,
    )


def provider_status_value(
    interaction: Any,
) -> str | None:

    status = getattr(
        interaction,
        "status",
        None,
    )

    if status is None:
        return None

    if hasattr(
        status,
        "value",
    ):
        return str(
            status.value
        )

    return str(status)


def rebuild_results_jsonl(
    split_dir: Path,
) -> None:

    records = []

    for meta_path in sorted(
        split_dir.glob(
            "*/run_meta.json"
        )
    ):

        meta = load_json(
            meta_path
        )

        if (
            meta.get("status")
            != "completed"
        ):
            continue

        score_path = (
            meta_path.parent
            / "score.json"
        )

        if not score_path.exists():
            continue

        score = load_json(
            score_path
        )

        records.append(
            {

                "split":
                    meta["split"],

                "doc_id":
                    meta["doc_id"],

                "model_id":
                    meta["model_id"],

                "provider_interaction_id":
                    meta.get(
                        "provider_interaction_id"
                    ),

                "sdk_version":
                    meta["sdk_version"],

                "prompt_version":
                    meta.get(
                        "prompt_version",
                        "v1",
                    ),

                **score.get(
                    "summary",
                    {},
                ),
            }
        )

    output_path = (
        split_dir
        / "results.jsonl"
    )

    with output_path.open(
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


def parse_args(
) -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Run the Track-A Gemini "
            "controlled cloud baseline "
            "using frozen Phase-2, "
            "Phase-3 and Phase-5 machinery."
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
            "Run only one document "
            "from the selected split."
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

    # Load project-root .env.
    load_dotenv(
        PROJECT_ROOT
        / ".env"
    )

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:

        raise EnvironmentError(
            "GEMINI_API_KEY was not found. "
            "Add it to the project-root "
            ".env file."
        )


    args = parse_args()

    benchmark_version = (
        get_benchmark_version(
            args.prompt_version
        )
    )

    artifact_hashes = (
        build_cloud_artifact_hashes(
            args.prompt_version
        )
    )

    documents = read_split(
        args.split
    )


    if args.only_doc:

        if (
            args.only_doc
            not in documents
        ):

            raise ValueError(
                f"{args.only_doc} is not "
                f"in {args.split}.txt"
            )

        documents = [
            args.only_doc
        ]


    split_dir = (
        get_cloud_output_root(
            args.prompt_version
        )
        / args.split
    )

    split_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    sdk_version = (
        importlib.metadata.version(
            "google-genai"
        )
    )


    client = genai.Client(
        api_key=api_key
    )


    try:

        print(
            f"\nSplit: {args.split}"
            f"\nDocuments: {len(documents)}"
            f"\nCloud model: {MODEL_ID}"
            f"\nPrompt version: {args.prompt_version}"
            f"\nAPI: Interactions"
            f"\nSDK: google-genai "
            f"{sdk_version}"
        )


        for index, doc_id in enumerate(
            documents,
            start=1,
        ):

            validate_inputs(
                doc_id
            )


            run_dir = (
                split_dir
                / (
                    f"{doc_id}"
                    f"__{MODEL_LABEL}"
                )
            )


            print(
                "\n"
                + "=" * 70
                + (
                    f"\n[{index}/"
                    f"{len(documents)}] "
                    f"{doc_id} × "
                    f"{MODEL_ID}"
                )
                + "\n"
                + "=" * 70
            )


            if is_completed(
                run_dir
            ):

                print(
                    "Already completed "
                    "-> skipping."
                )

                continue


            if run_dir.exists():

                shutil.rmtree(
                    run_dir
                )


            run_dir.mkdir(
                parents=True,
                exist_ok=True,
            )


            annotation_path = (
                ANNOTATIONS_DIR
                / f"{doc_id}.json"
            )

            report_text_path = (
                REPORTS_TEXT_DIR
                / f"{doc_id}.jsonl"
            )


            bank, target_year = (
                parse_doc_id(
                    doc_id
                )
            )


            # Frozen Phase-2 context.
            selection = select_pages(
                doc_id=doc_id,
                core_only=False,
            )


            # Frozen Phase-3 prompt construction.
            messages = build_messages(
                doc_id=doc_id,
                bank=bank,
                target_year=target_year,
                context=selection[
                    "text"
                ],
                prompt_version=(
                    args.prompt_version
                ),
            )


            system_prompt = (
                messages[0][
                    "content"
                ]
            )

            user_prompt = (
                messages[1][
                    "content"
                ]
            )


            # Gemini-specific settings.
            #
            # Temperature is intentionally NOT passed.
            # Gemini 3.6 Flash uses its provider default.
            #
            # We retain the frozen seed/output limit
            # where supported and use minimal thinking.
            generation_config = {

                "seed": int(
                    GENERATION_SETTINGS[
                        "seed"
                    ]
                ),

                "max_output_tokens": int(
                    GENERATION_SETTINGS[
                        "num_predict"
                    ]
                ),

                "thinking_level":
                    "minimal",
            }


            # Current Interactions API structured-output
            # configuration.
            response_format = {

                "type": "text",

                "mime_type":
                    "application/json",

                "schema":
                    GENERATION_SCHEMA,
            }


            started = (
                time.perf_counter()
            )


            try:

                interaction = (
                    client.interactions.create(

                        model=MODEL_ID,

                        input=user_prompt,

                        system_instruction=(
                            system_prompt
                        ),

                        generation_config=(
                            generation_config
                        ),

                        response_format=(
                            response_format
                        ),

                        # This benchmark is single-turn.
                        # No provider-side conversation
                        # state is required.
                        store=False,
                    )
                )


            except Exception as exc:

                elapsed = (
                    time.perf_counter()
                    - started
                )

                (
                    run_dir
                    / "api_error.txt"
                ).write_text(
                    (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                    encoding="utf-8",
                )


                write_json(
                    run_dir
                    / "run_meta.json",
                    {

                        "status":
                            "api_failed",

                        "split":
                            args.split,

                        "prompt_version":
                            args.prompt_version,

                        "doc_id":
                            doc_id,

                        "model_id":
                            MODEL_ID,

                        "api":
                            "interactions",

                        "sdk_version":
                            sdk_version,

                        "latency_s":
                            elapsed,

                        "report_pages":
                            selection[
                                "report_pages"
                            ],

                        "pdf_pages":
                            selection[
                                "pdf_pages"
                            ],

                        "page_set":
                            selection[
                                "page_set"
                            ],

                        "generation_settings":
                            generation_config,

                        "artifact_hashes":
                            artifact_hashes,
                    },
                )

                raise


            elapsed = (
                time.perf_counter()
                - started
            )


            raw_text = (
                interaction.output_text
                or ""
            )


            (
                run_dir
                / "raw_response.txt"
            ).write_text(
                raw_text,
                encoding="utf-8",
            )


            # Preserve provider response separately.
            provider_response = (
                interaction_to_dict(
                    interaction
                )
            )

            write_json(
                run_dir
                / "provider_response.json",
                provider_response,
            )


            # Frozen Phase-4 validation.
            validation = (
                parse_and_validate(
                    raw_text=raw_text,
                    schema=FIELD_SCHEMA,
                )
            )


            write_json(
                run_dir
                / "validation.json",
                {

                    "valid":
                        validation[
                            "valid"
                        ],

                    "error_type":
                        validation[
                            "error_type"
                        ],

                    "error_message":
                        validation[
                            "error_message"
                        ],
                },
            )


            prediction_path = None


            if validation[
                "valid"
            ]:

                prediction_path = (
                    run_dir
                    / "prediction.json"
                )

                write_json(
                    prediction_path,
                    validation[
                        "parsed"
                    ],
                )


            # Frozen Phase-5 scoring.
            score = run_phase5(

                annotation_path=(
                    annotation_path
                ),

                validation_path=(
                    run_dir
                    / "validation.json"
                ),

                report_text_path=(
                    report_text_path
                ),

                prediction_path=(
                    prediction_path
                ),

                run_dir=run_dir,
            )


            write_json(
                run_dir
                / "score.json",
                score,
            )


            usage = getattr(
                interaction,
                "usage",
                None,
            )


            provider_status = (
                provider_status_value(
                    interaction
                )
            )


            run_meta = {

                "status":
                    "completed",

                "split":
                    args.split,

                "prompt_version":
                    args.prompt_version,

                "doc_id":
                    doc_id,

                "model_id":
                    MODEL_ID,

                "provider_model":
                    getattr(
                        interaction,
                        "model",
                        MODEL_ID,
                    ),

                "provider_interaction_id":
                    getattr(
                        interaction,
                        "id",
                        None,
                    ),

                "provider_status":
                    provider_status,

                "api":
                    "interactions",

                "sdk_version":
                    sdk_version,

                "report_pages":
                    selection[
                        "report_pages"
                    ],

                "pdf_pages":
                    selection[
                        "pdf_pages"
                    ],

                "page_set":
                    selection[
                        "page_set"
                    ],

                "character_count":
                    selection[
                        "character_count"
                    ],

                "generation_settings": {

                    "temperature":
                        "provider_default",

                    "seed":
                        generation_config[
                            "seed"
                        ],

                    "max_output_tokens":
                        generation_config[
                            "max_output_tokens"
                        ],

                    "thinking_level":
                        generation_config[
                            "thinking_level"
                        ],

                    "local_num_ctx_reference":
                        int(
                            GENERATION_SETTINGS[
                                "num_ctx"
                            ]
                        ),

                    "response_mime_type":
                        "application/json",

                    "store":
                        False,
                },


                "runtime_metrics": {

                    "latency_s":
                        elapsed,

                    "input_tokens":
                        usage_value(
                            usage,
                            "total_input_tokens",
                        ),

                    "output_tokens":
                        usage_value(
                            usage,
                            "total_output_tokens",
                        ),

                    "thought_tokens":
                        usage_value(
                            usage,
                            "total_thought_tokens",
                        ),

                    "total_tokens":
                        usage_value(
                            usage,
                            "total_tokens",
                        ),

                    "cached_tokens":
                        usage_value(
                            usage,
                            "total_cached_tokens",
                        ),
                },


                "artifact_hashes":
                    artifact_hashes,

                "valid_json":
                    validation[
                        "valid"
                    ],

                "raw_response":
                    "raw_response.txt",

                "provider_response":
                    "provider_response.json",

                "validation":
                    "validation.json",

                "prediction": (
                    "prediction.json"
                    if prediction_path
                    is not None
                    else None
                ),

                "score":
                    "score.json",
            }


            write_json(
                run_dir
                / "run_meta.json",
                run_meta,
            )


            rebuild_results_jsonl(
                split_dir
            )


            print(
                "Completed."
            )

            print(
                "Provider status:",
                provider_status,
            )

            print(
                "Valid JSON:",
                validation[
                    "valid"
                ],
            )

            print(
                "Score summary:",
                json.dumps(
                    score.get(
                        "summary",
                        {},
                    ),
                    ensure_ascii=False,
                ),
            )

            print(
                "Latency (s):",
                round(
                    elapsed,
                    2,
                ),
            )

            print(
                "Input tokens:",
                usage_value(
                    usage,
                    "total_input_tokens",
                ),
            )

            print(
                "Output tokens:",
                usage_value(
                    usage,
                    "total_output_tokens",
                ),
            )

            print(
                "Thought tokens:",
                usage_value(
                    usage,
                    "total_thought_tokens",
                ),
            )


        rebuild_results_jsonl(
            split_dir
        )


        print(
            "\nCloud baseline "
            "command completed."
        )

        print(
            "Results:",
            split_dir,
        )


    finally:

        client.close()


if __name__ == "__main__":
    main()
