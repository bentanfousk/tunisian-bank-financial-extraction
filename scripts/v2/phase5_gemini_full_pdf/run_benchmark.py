from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

import fitz
from dotenv import load_dotenv
from google import genai

from scripts.benchmark_versions import get_benchmark_version
from scripts.phase4.prompt_builder import build_messages
from scripts.phase4.response_validation import parse_and_validate

from scripts.phase7.run_cloud_baseline import (
    PROJECT_ROOT,
    ANNOTATIONS_DIR,
    REPORTS_TEXT_DIR,
    FIELD_SCHEMA,
    GENERATION_SETTINGS,
    GENERATION_SCHEMA,
    MANIFEST_PATH,
    MODEL_ID,
    MODEL_LABEL,
    load_json,
    write_json,
    parse_doc_id,
    read_split,
    run_phase5,
    interaction_to_dict,
    usage_value,
    provider_status_value,
    build_cloud_artifact_hashes,
)


PDF_ROOT = PROJECT_ROOT / "data" / "raw"

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "v2"
    / "phase5_gemini_full_pdf"
)

PROMPT_VERSION = "v1_1"

# Frozen V2 test set.
V2_TEST_DOCS = [
    "amen_2024",
    "ATB_2023",
    "attijari_2024",
    "bh_2024",
    "bna_2024",
    "BT_2024",
]

# Pricing snapshot for Gemini 3.6 Flash,
# valid through 2026-12-31.
STANDARD_INPUT_PER_M = 0.75
STANDARD_OUTPUT_PER_M = 3.75

BATCH_INPUT_PER_M = 0.375
BATCH_OUTPUT_PER_M = 1.875


def normalize_name(value: str) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "",
        value.lower(),
    )


def find_pdf(doc_id: str) -> Path:
    pdfs = list(
        PDF_ROOT.rglob("*.pdf")
    )

    target = normalize_name(doc_id)

    exact = [
        path
        for path in pdfs
        if normalize_name(path.stem) == target
    ]

    if len(exact) == 1:
        return exact[0]

    bank, year = parse_doc_id(doc_id)

    bank_key = normalize_name(bank)

    possible = [
        path
        for path in pdfs
        if (
            bank_key
            in normalize_name(path.stem)
            and str(year) in path.stem
        )
    ]

    if len(possible) == 1:
        return possible[0]

    raise FileNotFoundError(
        f"Could not resolve exactly one PDF "
        f"for {doc_id} under {PDF_ROOT}. "
        f"Candidates: {[str(p) for p in possible]}"
    )


def is_completed(run_dir: Path) -> bool:
    meta_path = run_dir / "run_meta.json"
    score_path = run_dir / "score.json"

    if not meta_path.exists() or not score_path.exists():
        return False

    try:
        meta = load_json(meta_path)
    except Exception:
        return False

    return meta.get("status") == "completed"


def pdf_metadata(pdf_path: Path) -> dict[str, Any]:
    with fitz.open(pdf_path) as document:
        page_count = document.page_count

    return {
        "pdf_path": str(
            pdf_path.relative_to(PROJECT_ROOT)
        ),
        "pdf_file_name": pdf_path.name,
        "pdf_size_bytes": pdf_path.stat().st_size,
        "pdf_page_count": page_count,
    }


def theoretical_cost(
    input_tokens: int | None,
    output_tokens: int | None,
    thought_tokens: int | None,
) -> dict[str, float | None]:

    if input_tokens is None:
        return {
            "standard_usd": None,
            "batch_usd": None,
        }

    output = int(output_tokens or 0)
    thought = int(thought_tokens or 0)

    # Gemini pricing states that thinking tokens
    # are billed at the output-token rate.
    billable_output = output + thought

    standard = (
        input_tokens
        / 1_000_000
        * STANDARD_INPUT_PER_M
        +
        billable_output
        / 1_000_000
        * STANDARD_OUTPUT_PER_M
    )

    batch = (
        input_tokens
        / 1_000_000
        * BATCH_INPUT_PER_M
        +
        billable_output
        / 1_000_000
        * BATCH_OUTPUT_PER_M
    )

    return {
        "standard_usd": standard,
        "batch_usd": batch,
    }


def rebuild_results_jsonl(
    split_dir: Path,
) -> None:
    rows = []

    for meta_path in sorted(
        split_dir.glob("*/run_meta.json")
    ):
        meta = load_json(meta_path)

        if meta.get("status") != "completed":
            continue

        score_path = (
            meta_path.parent / "score.json"
        )

        if not score_path.exists():
            continue

        score = load_json(score_path)

        rows.append(
            {
                "split": meta["split"],
                "doc_id": meta["doc_id"],
                "model_id": meta["model_id"],
                "prompt_version": meta[
                    "prompt_version"
                ],
                "input_mode": "complete_pdf",
                "pdf_page_count": meta[
                    "pdf_metadata"
                ]["pdf_page_count"],
                "pdf_size_bytes": meta[
                    "pdf_metadata"
                ]["pdf_size_bytes"],
                "runtime_metrics": meta[
                    "runtime_metrics"
                ],
                "cost": meta["cost"],
                **score.get("summary", {}),
            }
        )

    output_path = (
        split_dir / "results.jsonl"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "V2 complete-PDF Gemini baseline "
            "for Track A."
        )
    )

    parser.add_argument(
        "--split",
        required=True,
        choices=["dev", "test"],
    )

    parser.add_argument(
        "--only-doc",
        help="Run only one document.",
    )

    return parser.parse_args()


def main() -> None:

    load_dotenv(PROJECT_ROOT / ".env")

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not found in .env"
        )

    args = parse_args()

    documents = read_split(args.split)

    if args.split == "test":
        documents = [
            doc
            for doc in documents
            if doc in V2_TEST_DOCS
        ]

        missing = [
            doc
            for doc in V2_TEST_DOCS
            if doc not in documents
        ]

        if missing:
            raise ValueError(
                "Frozen V2 test documents missing "
                f"from test split: {missing}"
            )

        # Preserve frozen order.
        documents = V2_TEST_DOCS.copy()

    if args.only_doc:
        if args.only_doc not in documents:
            raise ValueError(
                f"{args.only_doc} is not in "
                f"the selected split."
            )

        documents = [args.only_doc]

    split_dir = (
        OUTPUT_ROOT / args.split
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

    artifact_hashes = (
        build_cloud_artifact_hashes(
            PROMPT_VERSION
        )
    )

    print(
        f"\nSplit: {args.split}"
        f"\nDocuments: {len(documents)}"
        f"\nCloud model: {MODEL_ID}"
        f"\nPrompt version: {PROMPT_VERSION}"
        f"\nInput mode: COMPLETE PDF"
        f"\nSDK: google-genai {sdk_version}"
    )

    try:

        for index, doc_id in enumerate(
            documents,
            start=1,
        ):

            print(
                "\n"
                + "=" * 72
                + f"\n[{index}/{len(documents)}] "
                + f"{doc_id} × {MODEL_ID}"
                + "\n"
                + "=" * 72
            )

            run_dir = (
                split_dir
                / f"{doc_id}__{MODEL_LABEL}"
            )

            if is_completed(run_dir):
                print(
                    "Already completed -> skipping."
                )
                continue

            if run_dir.exists():
                shutil.rmtree(run_dir)

            run_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            pdf_path = find_pdf(doc_id)

            annotation_path = (
                ANNOTATIONS_DIR
                / f"{doc_id}.json"
            )

            report_text_path = (
                REPORTS_TEXT_DIR
                / f"{doc_id}.jsonl"
            )

            for required in [
                annotation_path,
                report_text_path,
                MANIFEST_PATH,
                pdf_path,
            ]:
                if not required.exists():
                    raise FileNotFoundError(
                        required
                    )

            metadata = pdf_metadata(
                pdf_path
            )

            bank, target_year = (
                parse_doc_id(doc_id)
            )

            # IMPORTANT:
            # No selected pages are inserted.
            # The complete PDF attached below is
            # the only report context.
            full_pdf_context = (
                "[RAPPORT ANNUEL COMPLET FOURNI "
                "SOUS FORME DE DOCUMENT PDF]\n"
                "Utilise uniquement le rapport PDF "
                "complet joint à cette requête. "
                "Pour le champ 'page', cite le numéro "
                "de page imprimé/visible dans le rapport "
                "lorsqu'il est disponible, et non "
                "l'index physique du lecteur PDF."
            )

            messages = build_messages(
                doc_id=doc_id,
                bank=bank,
                target_year=target_year,
                context=full_pdf_context,
                prompt_version=PROMPT_VERSION,
            )

            system_prompt = (
                messages[0]["content"]
            )

            user_prompt = (
                messages[1]["content"]
            )

            generation_config = {
                "seed": int(
                    GENERATION_SETTINGS["seed"]
                ),
                "max_output_tokens": int(
                    GENERATION_SETTINGS[
                        "num_predict"
                    ]
                ),
                "thinking_level": "minimal",
            }

            response_format = {
                "type": "text",
                "mime_type": "application/json",
                "schema": GENERATION_SCHEMA,
            }

            overall_started = (
                time.perf_counter()
            )

            # Upload the COMPLETE original PDF.
            upload_started = (
                time.perf_counter()
            )

            uploaded_file = (
                client.files.upload(
                    file=str(pdf_path)
                )
            )

            upload_latency = (
                time.perf_counter()
                - upload_started
            )

            generation_started = (
                time.perf_counter()
            )

            interaction = (
                client.interactions.create(
                    model=MODEL_ID,

                    input=[
                        {
                            "type": "document",
                            "uri":
                                uploaded_file.uri,
                            "mime_type":
                                uploaded_file.mime_type
                                or "application/pdf",
                        },
                        {
                            "type": "text",
                            "text": user_prompt,
                        },
                    ],

                    system_instruction=(
                        system_prompt
                    ),

                    generation_config=(
                        generation_config
                    ),

                    response_format=(
                        response_format
                    ),

                    store=False,
                )
            )

            generation_latency = (
                time.perf_counter()
                - generation_started
            )

            end_to_end_latency = (
                time.perf_counter()
                - overall_started
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

            write_json(
                run_dir
                / "provider_response.json",
                interaction_to_dict(
                    interaction
                ),
            )

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
                        validation["valid"],
                    "error_type":
                        validation["error_type"],
                    "error_message":
                        validation[
                            "error_message"
                        ],
                },
            )

            prediction_path = None

            if validation["valid"]:
                prediction_path = (
                    run_dir
                    / "prediction.json"
                )

                write_json(
                    prediction_path,
                    validation["parsed"],
                )

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
                run_dir / "score.json",
                score,
            )

            usage = getattr(
                interaction,
                "usage",
                None,
            )

            input_tokens = usage_value(
                usage,
                "total_input_tokens",
            )

            output_tokens = usage_value(
                usage,
                "total_output_tokens",
            )

            thought_tokens = usage_value(
                usage,
                "total_thought_tokens",
            )

            total_tokens = usage_value(
                usage,
                "total_tokens",
            )

            costs = theoretical_cost(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                thought_tokens=thought_tokens,
            )

            run_meta = {
                "status": "completed",

                "split": args.split,

                "prompt_version":
                    PROMPT_VERSION,

                "input_mode":
                    "complete_pdf",

                "doc_id": doc_id,

                "model_id": MODEL_ID,

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
                    provider_status_value(
                        interaction
                    ),

                "api": "interactions",

                "sdk_version":
                    sdk_version,

                "pdf_metadata":
                    metadata,

                "generation_settings": {
                    "temperature":
                        "provider_default",
                    "seed":
                        generation_config["seed"],
                    "max_output_tokens":
                        generation_config[
                            "max_output_tokens"
                        ],
                    "thinking_level":
                        "minimal",
                    "response_mime_type":
                        "application/json",
                    "store": False,
                },

                "runtime_metrics": {
                    "upload_latency_s":
                        upload_latency,

                    "generation_latency_s":
                        generation_latency,

                    "end_to_end_latency_s":
                        end_to_end_latency,

                    "input_tokens":
                        input_tokens,

                    "output_tokens":
                        output_tokens,

                    "thought_tokens":
                        thought_tokens,

                    "total_tokens":
                        total_tokens,

                    "cached_tokens":
                        usage_value(
                            usage,
                            "total_cached_tokens",
                        ),
                },

                "cost": {
                    "pricing_snapshot":
                        "2026-09-20",

                    "standard_input_usd_per_m":
                        STANDARD_INPUT_PER_M,

                    "standard_output_usd_per_m":
                        STANDARD_OUTPUT_PER_M,

                    "batch_input_usd_per_m":
                        BATCH_INPUT_PER_M,

                    "batch_output_usd_per_m":
                        BATCH_OUTPUT_PER_M,

                    "theoretical_standard_usd":
                        costs["standard_usd"],

                    "theoretical_batch_usd":
                        costs["batch_usd"],

                    "actual_charged_usd":
                        None,
                },

                "artifact_hashes":
                    artifact_hashes,

                "valid_json":
                    validation["valid"],

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
                run_dir / "run_meta.json",
                run_meta,
            )

            rebuild_results_jsonl(
                split_dir
            )

            print("Completed.")

            print(
                "PDF:",
                pdf_path.name,
            )

            print(
                "PDF pages:",
                metadata[
                    "pdf_page_count"
                ],
            )

            print(
                "Valid JSON:",
                validation["valid"],
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
                "Upload latency (s):",
                round(
                    upload_latency,
                    2,
                ),
            )

            print(
                "Generation latency (s):",
                round(
                    generation_latency,
                    2,
                ),
            )

            print(
                "End-to-end latency (s):",
                round(
                    end_to_end_latency,
                    2,
                ),
            )

            print(
                "Input tokens:",
                input_tokens,
            )

            print(
                "Output tokens:",
                output_tokens,
            )

            print(
                "Thought tokens:",
                thought_tokens,
            )

            print(
                "Theoretical standard cost ($):",
                costs["standard_usd"],
            )

        rebuild_results_jsonl(
            split_dir
        )

        print(
            "\nV2 full-PDF Gemini "
            "baseline completed."
        )

        print(
            "Results:",
            split_dir,
        )

    finally:
        client.close()


if __name__ == "__main__":
    main()