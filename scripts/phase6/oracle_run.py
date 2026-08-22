from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]

# Make the project root importable when this script
# is executed directly.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts.phase2_page_select.page_select import (
    MANIFEST_PATH,
    REPORTS_TEXT_DIR,
    get_document_entry,
    get_page_refs,
    load_json,
    load_report_pages,
)
from scripts.phase4.prompt_builder import build_messages
from scripts.phase4.ollama_adapter import generate
from scripts.phase4.response_validation import parse_and_validate
from scripts.phase4.mlflow_tracking import (
    log_extraction_run,
    log_generation_failure,
)
from scripts.phase4.gpu_monitor import GpuMemoryMonitor
from scripts.phase4.run_extract import (
    FIELD_SCHEMA,
    GENERATION_SETTINGS,
    ARTIFACT_HASHES,
    get_locked_model,
    verify_model_digest,
    parse_doc_id,
    sha256_file,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

ANNOTATIONS_DIR = (
    REPO_ROOT
    / "data"
    / "annotations"
)


def get_oracle_report_pages(
    annotation: dict[str, Any],
) -> list[int]:
    """
    Return the unique authoritative report pages
    cited by the hand annotation.

    Only annotation page numbers are used.
    Gold values, evidence text, and distractors are
    never copied into model context.
    """

    fields = annotation.get("fields")

    if not isinstance(fields, list) or not fields:
        raise ValueError(
            "Annotation must contain a non-empty "
            "'fields' list."
        )

    pages: set[int] = set()

    for field in fields:
        if not isinstance(field, dict):
            raise TypeError(
                "Every annotation field must be "
                "a JSON object."
            )

        field_name = field.get(
            "field",
            "<unknown>",
        )

        status = field.get("status")
        page = field.get("page")

        if status == "found" and page is None:
            raise ValueError(
                f"{field_name}: status is 'found' "
                "but annotation page is null."
            )

        # not_found / other legitimate null-page
        # cases do not contribute an oracle page.
        if page is None:
            continue

        # bool is a subclass of int in Python,
        # so reject it explicitly.
        if (
            isinstance(page, bool)
            or not isinstance(page, int)
            or page < 1
        ):
            raise ValueError(
                f"{field_name}: invalid annotation "
                f"page {page!r}."
            )

        pages.add(page)

    if not pages:
        raise ValueError(
            "Annotation contains no authoritative "
            "report pages for the oracle diagnostic."
        )

    # Stable ordering makes runs reproducible.
    return sorted(pages)


def build_oracle_selection(
    doc_id: str,
    annotation_path: Path | None = None,
    manifest_path: Path = MANIFEST_PATH,
    reports_text_dir: Path = REPORTS_TEXT_DIR,
) -> dict[str, Any]:
    """
    Build model context from the authoritative pages
    cited in the hand annotation.

    This intentionally bypasses Phase-2 select_pages().
    Phase-2 helpers are reused only to:
      - read the frozen manifest,
      - map report pages to physical PDF/JSONL pages,
      - load page-indexed report text.
    """

    if annotation_path is None:
        annotation_path = (
            ANNOTATIONS_DIR
            / f"{doc_id}.json"
        )

    annotation = load_json(annotation_path)

    annotation_doc_id = annotation.get("doc_id")

    if annotation_doc_id != doc_id:
        raise ValueError(
            f"Annotation doc_id "
            f"{annotation_doc_id!r} does not match "
            f"requested doc_id {doc_id!r}."
        )

    oracle_report_pages = (
        get_oracle_report_pages(annotation)
    )

    manifest = load_json(manifest_path)

    document = get_document_entry(
        manifest=manifest,
        doc_id=doc_id,
    )

    # Get the manually verified individual page
    # references only to obtain the frozen
    # report_page -> pdf_page mapping.
    page_refs = get_page_refs(
        document=document,
        doc_id=doc_id,
        scope="individual",
        core_only=False,
    )

    report_page_map: dict[int, dict[str, Any]] = {}

    for page_ref in page_refs:
        report_page = page_ref["report_page"]

        existing = report_page_map.get(
            report_page
        )

        if existing is not None:
            if (
                existing["pdf_page"]
                != page_ref["pdf_page"]
            ):
                raise ValueError(
                    f"{doc_id}: conflicting mappings "
                    f"for REPORT_PAGE {report_page}: "
                    f"PDF pages "
                    f"{existing['pdf_page']} and "
                    f"{page_ref['pdf_page']}."
                )

            # Same report page -> same physical page.
            # Multiple statement types may legitimately
            # reference the same page.
            continue

        report_page_map[
            report_page
        ] = page_ref

    unmapped_pages = [
        page
        for page in oracle_report_pages
        if page not in report_page_map
    ]

    if unmapped_pages:
        raise ValueError(
            f"{doc_id}: authoritative annotation "
            f"page(s) {unmapped_pages} are not mapped "
            "in the manually verified individual "
            "statement manifest."
        )

    report_jsonl_path = (
        reports_text_dir
        / f"{doc_id}.jsonl"
    )

    report_text = load_report_pages(
        jsonl_path=report_jsonl_path,
        expected_doc_id=doc_id,
    )

    selected_refs = [
        report_page_map[report_page]
        for report_page in oracle_report_pages
    ]

    missing_pdf_pages = [
        page_ref["pdf_page"]
        for page_ref in selected_refs
        if page_ref["pdf_page"] not in report_text
    ]

    if missing_pdf_pages:
        raise ValueError(
            f"{doc_id}: physical PDF/JSONL pages "
            f"{missing_pdf_pages} required by the "
            "oracle are missing."
        )

    page_blocks: list[str] = []

    for page_ref in selected_refs:
        report_page = page_ref["report_page"]
        pdf_page = page_ref["pdf_page"]

        page_text = report_text[
            pdf_page
        ].strip()

        page_blocks.append(
            f"===== REPORT_PAGE {report_page} =====\n"
            f"{page_text}"
        )

    combined_text = "\n\n".join(
        page_blocks
    )

    return {
        "doc_id": doc_id,
        "page_set": "oracle_pages",
        "report_pages": [
            page_ref["report_page"]
            for page_ref in selected_refs
        ],
        "pdf_pages": [
            page_ref["pdf_page"]
            for page_ref in selected_refs
        ],
        "statement_types": [
            page_ref["statement_type"]
            for page_ref in selected_refs
        ],
        "page_count": len(selected_refs),
        "text": combined_text,
        "character_count": len(combined_text),
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Track-A oracle diagnostic for "
            "one document and one frozen Ollama model."
        )
    )

    parser.add_argument(
        "doc_id",
        help="Document ID, e.g. UIB_2024.",
    )

    parser.add_argument(
        "model_id",
        help=(
            "Exact model ID from "
            "configs/models.lock.json."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    doc_id = args.doc_id
    model_id = args.model_id

    bank, target_year = parse_doc_id(
        doc_id
    )

    annotation_path = (
        ANNOTATIONS_DIR
        / f"{doc_id}.json"
    )

    # Phase 6:
    # authoritative annotation pages instead of
    # Phase-2 select_pages().
    selection = build_oracle_selection(
        doc_id=doc_id,
        annotation_path=annotation_path,
    )

    # Everything below stays identical in behavior
    # to the frozen Phase-4 extraction harness.
    locked_model = get_locked_model(
        model_id
    )

    model_digest = locked_model.get(
        "digest"
    )

    if not isinstance(model_digest, str):
        raise ValueError(
            f"Frozen model '{model_id}' does not "
            "have a valid digest."
        )

    verify_model_digest(
        model_id=model_id,
        expected_digest=model_digest,
    )

    messages = build_messages(
        doc_id=doc_id,
        bank=bank,
        target_year=target_year,
        context=selection["text"],
    )

    # Preserve the frozen hashes and additionally
    # fingerprint the oracle annotation.
    oracle_artifact_hashes = {
        **ARTIFACT_HASHES,
        "diagnostic_mode": "oracle",
        "oracle_annotation_sha256": sha256_file(
            annotation_path
        ),
    }

    print("Document:", doc_id)
    print("Model:", model_id)
    print("Diagnostic mode: oracle")
    print(
        "Oracle report pages:",
        selection["report_pages"],
    )
    print(
        "Oracle PDF/JSONL pages:",
        selection["pdf_pages"],
    )
    print(
        "Page count:",
        selection["page_count"],
    )
    print(
        "Character count:",
        selection["character_count"],
    )
    print(
        "Generation settings:",
        GENERATION_SETTINGS,
    )

    print("\nCalling Ollama...")

    try:
        with GpuMemoryMonitor() as gpu:
            result = generate(
                model_id=model_id,
                messages=messages,
                schema=FIELD_SCHEMA,
                settings=GENERATION_SETTINGS,
            )

    except Exception as exc:
        failure_run_id = (
            log_generation_failure(
                doc_id=doc_id,
                model_id=model_id,
                model_digest=model_digest,
                bank=bank,
                target_year=target_year,
                selection=selection,
                generation_settings=(
                    GENERATION_SETTINGS
                ),
                artifact_hashes=(
                    oracle_artifact_hashes
                ),
                error=exc,
            )
        )

        print(
            "\n--- GENERATION FAILED ---"
        )
        print(
            "Error type:",
            type(exc).__name__,
        )
        print("Error:", str(exc))
        print(
            "MLflow Run ID:",
            failure_run_id,
        )

        raise

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

    result[
        "gpu_memory_incremental_peak_mb"
    ] = gpu.incremental_peak_mb

    print("\n--- MODEL RESPONSE ---")
    print(result["text"])

    validation = parse_and_validate(
        raw_text=result["text"],
        schema=FIELD_SCHEMA,
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
            validation[
                "error_message"
            ],
        )

    mlflow_run_id = log_extraction_run(
        doc_id=doc_id,
        model_id=model_id,
        model_digest=model_digest,
        bank=bank,
        target_year=target_year,
        selection=selection,
        generation_settings=(
            GENERATION_SETTINGS
        ),
        artifact_hashes=(
            oracle_artifact_hashes
        ),
        generation_result=result,
        validation=validation,
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
        "Tokens/sec:",
        round(
            result["tokens_per_second"],
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

    print("\n--- MLFLOW ---")
    print(
        "Run ID:",
        mlflow_run_id,
    )


if __name__ == "__main__":
    main()