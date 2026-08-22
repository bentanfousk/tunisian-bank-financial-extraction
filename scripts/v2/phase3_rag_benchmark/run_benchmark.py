from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from time import perf_counter
from typing import Any

import faiss
import mlflow
import ollama
import torch
import yaml

from scripts.phase4.generation_policy import (
    generate_with_one_repair,
)
from scripts.phase4.gpu_monitor import GpuMemoryMonitor
from scripts.phase4.mlflow_tracking import configure_mlflow
from scripts.phase4.prompt_builder import build_messages
from scripts.v2.phase1_rag.build_index import (
    create_faiss_index,
    encode_chunks,
    get_model_revision,
    load_embedding_model,
)
from scripts.v2.phase1_rag.chunking import (
    build_page_chunks,
)
from scripts.v2.phase1_rag.ingestion import (
    ingest_document,
    parse_doc_id,
    read_split,
)
from scripts.v2.phase1_rag.retrieval_queries import (
    build_retrieval_queries,
)
from scripts.v2.phase1_rag.retrieve import (
    encode_query,
    merge_query_results,
    retrieve_query,
)
from scripts.v2.phase3_rag_benchmark.context_builder import (
    build_retrieved_context,
)
from scripts.v2.phase3_rag_benchmark.frozen_config import (
    FROZEN_RAG_CONFIG,
    PAGES_PER_QUERY,
    PROJECT_ROOT,
    RAG_CONFIG_PATH,
)

from scripts.benchmark_versions import get_benchmark_version

MODEL_KEY = "ministral3_3b"

MODEL_TAG = (
    "ministral-3:3b-instruct-2512-q4_K_M"
)

V1_EXPERIMENT_NAME = (
    "track_a_v2_phase3_rag_ministral"
)

V1_1_EXPERIMENT_NAME = (
    "track_a_v2_phase3_rag_ministral_v1_1"
)
EXPECTED_TEST_DOCUMENTS = [
    "amen_2024",
    "ATB_2023",
    "attijari_2024",
    "bh_2024",
    "bna_2024",
    "BT_2024",
]

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "v2"
    / "phase3_rag_benchmark"
)

V1_1_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "artifacts"
    / "v2"
    / "phase3_rag_benchmark_v1_1"
)

ANNOTATIONS_DIR = (
    PROJECT_ROOT / "data" / "annotations"
)

REPORTS_TEXT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "reports_txt"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "manifests"
    / "page_manifest.json"
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

MODELS_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "models.yaml"
)

MODELS_LOCK_PATH = (
    PROJECT_ROOT
    / "configs"
    / "models.lock.json"
)



USER_TEMPLATE_PATH = (
    PROJECT_ROOT
    / "prompts"
    / "extract_user_v1.txt"
)

REQUIRED_COMPLETED_ARTIFACTS = (
    "retrieval.json",
    "retrieved_context.txt",
    "raw_response.txt",
    "validation.json",
    "score.json",
    "retrieval_evaluation.json",
    "run_meta.json",
)


def load_json(
    path: Path,
) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(
            encoding="utf-8-sig"
        )
    )

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected a JSON object in {path}."
        )

    return payload


def load_yaml(
    path: Path,
) -> dict[str, Any]:
    payload = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected a YAML mapping in {path}."
        )

    return payload


def write_json(
    path: Path,
    payload: Any,
) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def frozen_artifact_hashes(
    *,
    system_prompt_path: Path,
) -> dict[str, str]:
    files = {
        "rag_config_sha256": (
            RAG_CONFIG_PATH
        ),
        "system_prompt_sha256": (
            system_prompt_path
        ),
        "user_template_sha256": (
            USER_TEMPLATE_PATH
        ),
        "field_schema_sha256": (
            FIELD_SCHEMA_PATH
        ),
        "run_config_sha256": (
            RUN_CONFIG_PATH
        ),
        "models_lock_sha256": (
            MODELS_LOCK_PATH
        ),
        "retrieval_queries_sha256": (
            PROJECT_ROOT
            / "scripts"
            / "v2"
            / "phase1_rag"
            / "retrieval_queries.py"
        ),
        "retrieval_implementation_sha256": (
            PROJECT_ROOT
            / "scripts"
            / "v2"
            / "phase1_rag"
            / "retrieve.py"
        ),
        "context_builder_sha256": (
            PROJECT_ROOT
            / "scripts"
            / "v2"
            / "phase3_rag_benchmark"
            / "context_builder.py"
        ),
    }

    return {
        name: sha256_file(path)
        for name, path in files.items()
    }


def load_frozen_model(
) -> dict[str, Any]:
    models_config = load_yaml(
        MODELS_CONFIG_PATH
    )

    model_entries = models_config.get(
        "local_models"
    )

    if not isinstance(model_entries, list):
        raise ValueError(
            "models.yaml has no local_models list."
        )

    model_entry = next(
        (
            entry
            for entry in model_entries
            if entry.get("id") == MODEL_KEY
        ),
        None,
    )

    if (
        model_entry is None
        or model_entry.get("tag")
        != MODEL_TAG
    ):
        raise ValueError(
            "Frozen Ministral mapping is "
            "missing or changed."
        )

    lock = load_json(MODELS_LOCK_PATH)

    lock_entry = next(
        (
            entry
            for entry in lock.get(
                "models",
                [],
            )
            if entry.get("name")
            == MODEL_TAG
        ),
        None,
    )

    if (
        lock_entry is None
        or not isinstance(
            lock_entry.get("digest"),
            str,
        )
    ):
        raise ValueError(
            "Frozen Ministral digest is missing."
        )

    return {
        "model_key": MODEL_KEY,
        "model_tag": MODEL_TAG,
        "model_digest": (
            lock_entry["digest"]
        ),
    }


def verify_installed_model(
    model_spec: dict[str, Any],
) -> None:
    response = ollama.list()

    for installed in response.models:
        if (
            installed.model
            == model_spec["model_tag"]
        ):
            if (
                installed.digest
                != model_spec[
                    "model_digest"
                ]
            ):
                raise RuntimeError(
                    "Installed Ministral digest "
                    "does not match the frozen lock."
                )

            return

    raise RuntimeError(
        "Frozen model is not installed: "
        f"{model_spec['model_tag']}"
    )


def validate_execution_request(
    *,
    split: str,
    only_doc: str | None,
    force: bool,
) -> None:
    if (
        split == "test"
        and only_doc is not None
    ):
        raise ValueError(
            "The untouched test split must run "
            "as one six-report batch. "
            "--only-doc is forbidden."
        )

    if split == "test" and force:
        raise ValueError(
            "--force is disabled for the final "
            "test split."
        )


def validate_input_paths(
    doc_id: str,
) -> None:
    required = (
        REPORTS_TEXT_DIR
        / f"{doc_id}.jsonl",
        ANNOTATIONS_DIR
        / f"{doc_id}.json",
        MANIFEST_PATH,
    )

    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required input(s):\n"
            + "\n".join(missing)
        )


def prepare_run_directory(
    *,
    output_root: Path,
    split: str,
    doc_id: str,
    force: bool,
) -> tuple[Path, bool]:
    run_dir = (
        output_root
        / "cold"
        / split
        / f"{doc_id}__{MODEL_KEY}"
    )

    meta_path = (
        run_dir / "run_meta.json"
    )

    if meta_path.exists():
        try:
            meta = load_json(meta_path)
        except Exception:
            meta = {}

        if (
            meta.get("status")
            == "completed"
            and not force
        ):
            return run_dir, True

    if run_dir.exists():
        if not force:
            raise FileExistsError(
                "Incomplete artifacts already "
                f"exist: {run_dir}. "
                "Use --force only for a "
                "development smoke rerun."
            )

        shutil.rmtree(run_dir)

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    return run_dir, False


def save_chunks(
    chunks: list[dict[str, Any]],
    path: Path,
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
    ) as stream:
        for position, chunk in enumerate(
            chunks
        ):
            stream.write(
                json.dumps(
                    {
                        "faiss_position": (
                            position
                        ),
                        **chunk,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def without_text(
    result: dict[str, Any],
) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key != "text"
    }

def attach_faiss_positions(
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Add the deterministic vector position expected by
    the frozen Phase 1 search implementation.
    """

    return [
        {
            **chunk,
            "faiss_position": position,
        }
        for position, chunk in enumerate(
            chunks
        )
    ]

def run_online_pipeline(
    *,
    doc_id: str,
    run_dir: Path,
    model_spec: dict[str, Any],
    field_schema: dict[str, Any],
    generation_settings: dict[str, Any],
    prompt_version: str,
) -> dict[str, Any]:
    """
    Complete report -> RAG-4 -> Ministral
    -> validation.

    This function does not load evaluation
    ground truth.
    """

    end_to_end_start = perf_counter()

    bank, target_year = parse_doc_id(
        doc_id
    )

    timings: dict[str, float] = {}

    # Complete report loading
    start = perf_counter()

    pages = ingest_document(doc_id)

    timings[
        "report_page_data_loading_seconds"
    ] = perf_counter() - start

    # BGE-M3 model loading
    start = perf_counter()

    embedding_model = (
        load_embedding_model()
    )

    torch.cuda.synchronize()

    timings[
        "embedding_model_load_seconds"
    ] = perf_counter() - start

    # One-page chunks for every usable page
    start = perf_counter()

    chunks = attach_faiss_positions(
        build_page_chunks(
            pages,
            embedding_model.tokenizer,
        )
    )

    timings["chunking_seconds"] = (
        perf_counter() - start
    )

    # Embed every page
    start = perf_counter()

    embeddings = encode_chunks(
        model=embedding_model,
        chunks=chunks,
        batch_size=1,
    )

    torch.cuda.synchronize()

    timings["embedding_seconds"] = (
        perf_counter() - start
    )

    # Construct exact FAISS index
    start = perf_counter()

    index = create_faiss_index(
        embeddings
    )

    timings[
        "faiss_index_construction_seconds"
    ] = perf_counter() - start

    index_path = (
        run_dir / "index.faiss"
    )

    chunks_path = (
        run_dir / "indexed_chunks.jsonl"
    )

    start = perf_counter()

    faiss.write_index(
        index,
        str(index_path),
    )

    save_chunks(
        chunks,
        chunks_path,
    )

    timings[
        "index_serialization_seconds"
    ] = perf_counter() - start

    # Two frozen queries
    queries = build_retrieval_queries(
        bank=bank,
        fiscal_year=target_year,
    )

    results_by_query: dict[
        str,
        list[dict[str, Any]],
    ] = {}

    query_timings: dict[
        str,
        dict[str, float],
    ] = {}

    for query in queries:
        torch.cuda.synchronize()

        start = perf_counter()

        query_vector = encode_query(
            embedding_model,
            query["text"],
        )

        torch.cuda.synchronize()

        query_embedding_seconds = (
            perf_counter() - start
        )

        start = perf_counter()

        results = retrieve_query(
            index=index,
            chunks=chunks,
            query_vector=query_vector,
            query_id=query["query_id"],
            query_text=query["text"],
            fiscal_year=target_year,
            top_k=PAGES_PER_QUERY,
            candidate_k=int(
                FROZEN_RAG_CONFIG[
                    "candidate_k"
                ]
            ),
        )

        retrieval_seconds = (
            perf_counter() - start
        )

        results_by_query[
            query["query_id"]
        ] = results

        query_timings[
            query["query_id"]
        ] = {
            "query_embedding_seconds": (
                query_embedding_seconds
            ),
            "retrieval_seconds": (
                retrieval_seconds
            ),
        }

    start = perf_counter()

    merged_results = (
        merge_query_results(
            results_by_query
        )
    )

    timings[
        "merge_deduplication_seconds"
    ] = perf_counter() - start

    timings[
        "query_embedding_seconds"
    ] = sum(
        item[
            "query_embedding_seconds"
        ]
        for item in query_timings.values()
    )

    timings[
        "retrieval_seconds"
    ] = sum(
        item["retrieval_seconds"]
        for item in query_timings.values()
    )

    pdf_path = (
        PROJECT_ROOT
        / chunks[0]["pdf_path"]
    )

    # Reliable REPORT_PAGE markers
    start = perf_counter()

    context_payload = (
        build_retrieved_context(
            pdf_path=pdf_path,
            merged_results=(
                merged_results
            ),
        )
    )

    timings[
        "context_construction_seconds"
    ] = perf_counter() - start

    retrieval_payload = {
        "doc_id": doc_id,
        "configuration": "RAG-4",
        "retrieval_method": (
            FROZEN_RAG_CONFIG[
                "retrieval_method"
            ]
        ),
        "candidate_k": (
            FROZEN_RAG_CONFIG[
                "candidate_k"
            ]
        ),
        "pages_per_query": (
            PAGES_PER_QUERY
        ),
        "maximum_pages_before_deduplication": (
            FROZEN_RAG_CONFIG[
                "maximum_pages_before_deduplication"
            ]
        ),
        "embedding_model": (
            FROZEN_RAG_CONFIG[
                "embedding_model"
            ]
        ),
        "embedding_revision": (
            FROZEN_RAG_CONFIG[
                "embedding_revision"
            ]
        ),
        "resolved_embedding_revision": (
            get_model_revision(
                embedding_model
            )
        ),
        "total_pdf_pages": len(pages),
        "indexable_pages": len(chunks),
        "queries": queries,
        "query_timings": query_timings,
        "results_by_query": {
            query_id: [
                without_text(result)
                for result in results
            ]
            for query_id, results
            in results_by_query.items()
        },
        "merged_results": [
            without_text(result)
            for result in merged_results
        ],
        "context_pages": (
            context_payload["pages"]
        ),
        "unresolved_context_pdf_pages": (
            context_payload[
                "unresolved_pdf_pages"
            ]
        ),
    }

    # These online artifacts are written before
    # evaluation begins.
    write_json(
        run_dir / "retrieval.json",
        retrieval_payload,
    )

    (
        run_dir
        / "retrieved_context.txt"
    ).write_text(
        context_payload["text"],
        encoding="utf-8",
    )

    # Release BGE-M3 before Ollama generation.
    del embeddings
    del embedding_model

    gc.collect()
    torch.cuda.empty_cache()

    messages = build_messages(
        doc_id=doc_id,
        bank=bank,
        target_year=target_year,
        context=context_payload["text"],
        prompt_version=prompt_version,
    )

    start = perf_counter()

    generation_policy = (
        generate_with_one_repair(
            model_id=(
                model_spec[
                    "model_tag"
                ]
            ),
            messages=messages,
            schema=field_schema,
            settings=(
                generation_settings
            ),
        )
    )

    timings[
        "ministral_generation_wall_seconds"
    ] = perf_counter() - start

    timings[
        "end_to_end_seconds"
    ] = (
        perf_counter()
        - end_to_end_start
    )

    generation_result = (
        generation_policy[
            "generation_result"
        ]
    )

    validation = (
        generation_policy[
            "validation"
        ]
    )

    generation_seconds = (
        generation_result[
            "eval_duration_ns"
        ]
        / 1_000_000_000
    )

    tokens_per_second = (
        generation_result[
            "output_tokens"
        ]
        / generation_seconds
        if generation_seconds > 0
        else 0.0
    )

    timings["retry_seconds"] = (
        generation_policy[
            "retry_result"
        ]["total_duration_ns"]
        / 1_000_000_000
        if generation_policy[
            "retry_used"
        ]
        else 0.0
    )

    (
        run_dir
        / "raw_response.txt"
    ).write_text(
        generation_result["text"],
        encoding="utf-8",
    )

    (
        run_dir
        / "first_raw_response.txt"
    ).write_text(
        generation_policy[
            "first_result"
        ]["text"],
        encoding="utf-8",
    )

    write_json(
        run_dir
        / "first_validation.json",
        generation_policy[
            "first_validation"
        ],
    )

    write_json(
        run_dir / "validation.json",
        validation,
    )

    if generation_policy["retry_used"]:
        (
            run_dir
            / "retry_raw_response.txt"
        ).write_text(
            generation_policy[
                "retry_result"
            ]["text"],
            encoding="utf-8",
        )

        write_json(
            run_dir
            / "retry_validation.json",
            generation_policy[
                "retry_validation"
            ],
        )

    prediction_path: Path | None = None

    if validation["valid"]:
        prediction_path = (
            run_dir / "prediction.json"
        )

        write_json(
            prediction_path,
            validation["parsed"],
        )

    selected_chunk_ids = {
        result["chunk_id"]
        for result in merged_results
    }

    runtime = {
        **timings,
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
        "generation_tokens_per_second": (
            tokens_per_second
        ),
        "ollama_total_duration_seconds": (
            generation_result[
                "total_duration_ns"
            ]
            / 1_000_000_000
        ),
        "ollama_prompt_eval_seconds": (
            generation_result[
                "prompt_eval_duration_ns"
            ]
            / 1_000_000_000
        ),
        "ollama_generation_seconds": (
            generation_seconds
        ),
        "faiss_index_size_bytes": (
            index_path.stat().st_size
        ),
        "context_character_count": (
            context_payload[
                "character_count"
            ]
        ),
        "retrieved_bge_tokens": sum(
            chunk[
                "embedding_token_count"
            ]
            for chunk in chunks
            if chunk["chunk_id"]
            in selected_chunk_ids
        ),
    }

    return {
        "bank": bank,
        "target_year": target_year,
        "chunks": chunks,
        "merged_results": (
            merged_results
        ),
        "context": context_payload,
        "generation_policy": (
            generation_policy
        ),
        "generation_result": (
            generation_result
        ),
        "validation": validation,
        "prediction_path": (
            prediction_path
        ),
        "runtime": runtime,
    }


def run_phase5_after_prediction(
    *,
    doc_id: str,
    run_dir: Path,
    validation: dict[str, Any],
    prediction_path: Path | None,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "scripts.phase5.score_prediction",
        "--annotation",
        str(
            ANNOTATIONS_DIR
            / f"{doc_id}.json"
        ),
        "--validation",
        str(
            run_dir
            / "validation.json"
        ),
        "--manifest",
        str(MANIFEST_PATH),
        "--report-text",
        str(
            REPORTS_TEXT_DIR
            / f"{doc_id}.jsonl"
        ),
    ]

    if (
        validation["valid"]
        and prediction_path is not None
    ):
        command.extend(
            [
                "--prediction",
                str(prediction_path),
            ]
        )

    completed = subprocess.run(
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
        completed.stdout,
        encoding="utf-8",
    )

    (
        run_dir
        / "phase5_stderr.txt"
    ).write_text(
        completed.stderr,
        encoding="utf-8",
    )

    if completed.returncode != 0:
        raise RuntimeError(
            "Frozen Phase 5 scoring failed. "
            "See phase5_stderr.txt."
        )

    score = json.loads(
        completed.stdout
    )

    write_json(
        run_dir / "score.json",
        score,
    )

    return score


def evaluate_retrieval_after_prediction(
    *,
    doc_id: str,
    chunks: list[dict[str, Any]],
    merged_results: list[
        dict[str, Any]
    ],
) -> dict[str, Any]:
    """
    Ground truth is imported and loaded only
    after prediction.
    """

    from scripts.phase2_page_select.page_select import (
        get_document_entry,
        load_json as load_manifest,
    )

    from scripts.v2.phase2_rag_evaluation.evaluate_retrieval import (
        evaluate_configuration,
        load_annotation,
    )

    manifest = load_manifest(
        MANIFEST_PATH
    )

    document = get_document_entry(
        manifest=manifest,
        doc_id=doc_id,
    )

    annotation = load_annotation(
        doc_id
    )

    chunks_by_id = {
        chunk["chunk_id"]: chunk
        for chunk in chunks
    }

    return evaluate_configuration(
        configuration_name="RAG-4",
        pages_per_query=(
            PAGES_PER_QUERY
        ),
        merged_results=(
            merged_results
        ),
        chunks_by_id=chunks_by_id,
        annotation=annotation,
        document=document,
        doc_id=doc_id,
    )


def log_completed_run(
    *,
    run_dir: Path,
    run_meta: dict[str, Any],
    score: dict[str, Any],
    retrieval_evaluation: (
        dict[str, Any]
    ),
    generation_settings: (
        dict[str, Any]
    ),
    experiment_name: str,
) -> str:
    configure_mlflow()

    mlflow.set_experiment(
        experiment_name
    )

    with mlflow.start_run(
        run_name=(
            "phase3__cold__"
            f"{run_meta['doc_id']}"
            f"__{MODEL_KEY}"
        )
    ) as run:
        run_id = run.info.run_id

        run_meta[
            "mlflow_run_id"
        ] = run_id

        write_json(
            run_dir / "run_meta.json",
            run_meta,
        )

        mlflow.log_params(
            {
                "model_key": MODEL_KEY,
                "model_tag": MODEL_TAG,
                "run_mode": "cold",
                "prompt_version": run_meta["prompt_version"],
                "rag_configuration": (
                    "RAG-4"
                ),
                "candidate_k": (
                    FROZEN_RAG_CONFIG[
                        "candidate_k"
                    ]
                ),
                "pages_per_query": (
                    PAGES_PER_QUERY
                ),
                **generation_settings,
            }
        )

        mlflow.set_tags(
            {
                "doc_id": (
                    run_meta["doc_id"]
                ),
                "split": (
                    run_meta["split"]
                ),
                "first_pass_valid": str(
                    run_meta[
                        "first_pass_valid"
                    ]
                ).lower(),
                "retry_used": str(
                    run_meta[
                        "retry_used"
                    ]
                ).lower(),
                "valid_json": str(
                    run_meta[
                        "valid_json"
                    ]
                ).lower(),
                **run_meta[
                    "artifact_hashes"
                ],
            }
        )

        numeric_metrics = {
            key: float(value)
            for key, value
            in run_meta[
                "runtime_metrics"
            ].items()
            if isinstance(
                value,
                (int, float),
            )
        }

        numeric_metrics.update(
            {
                "fully_correct_fields": (
                    score["summary"][
                        "fully_correct_fields"
                    ]
                ),
                "fully_correct_rate": (
                    score["summary"][
                        "fully_correct_rate"
                    ]
                ),
                "authoritative_page_recall": (
                    retrieval_evaluation[
                        "authoritative_page_recall"
                    ]
                ),
                "field_evidence_recall": (
                    retrieval_evaluation[
                        "field_evidence_recall"
                    ]
                ),
                "contamination_rate": (
                    retrieval_evaluation[
                        "contamination_rate"
                    ]
                ),
            }
        )

        mlflow.log_metrics(
            numeric_metrics
        )

        mlflow.log_artifacts(
            str(run_dir)
        )

        return run_id


def run_document(
    *,
    split: str,
    doc_id: str,
    run_dir: Path,
    model_spec: dict[str, Any],
    field_schema: dict[str, Any],
    generation_settings: dict[str, Any],
    artifact_hashes: dict[str, str],
    prompt_version: str,
    experiment_name: str,
) -> dict[str, Any]:
    with GpuMemoryMonitor() as gpu:
        online = run_online_pipeline(
            doc_id=doc_id,
            run_dir=run_dir,
            model_spec=model_spec,
            field_schema=field_schema,
            generation_settings=(
                generation_settings
            ),
            prompt_version=prompt_version,
        )

    online["runtime"].update(
        {
            "gpu_memory_baseline_mb": (
                gpu.baseline_mb
            ),
            "gpu_memory_peak_used_mb": (
                gpu.peak_mb
            ),
            "gpu_memory_incremental_peak_mb": (
                gpu.incremental_peak_mb
            ),
        }
    )

    # Evaluation begins only after the online
    # retrieval/prediction pipeline has finished.
    score = run_phase5_after_prediction(
        doc_id=doc_id,
        run_dir=run_dir,
        validation=(
            online["validation"]
        ),
        prediction_path=(
            online["prediction_path"]
        ),
    )

    retrieval_evaluation = (
        evaluate_retrieval_after_prediction(
            doc_id=doc_id,
            chunks=online["chunks"],
            merged_results=(
                online[
                    "merged_results"
                ]
            ),
        )
    )

    write_json(
        run_dir
        / "retrieval_evaluation.json",
        retrieval_evaluation,
    )

    policy = online[
        "generation_policy"
    ]

    run_meta = {
        "status": "completed",
        "split": split,
        "doc_id": doc_id,
        "prompt_version": prompt_version,
        "bank": online["bank"],
        "target_year": (
            online["target_year"]
        ),
        "model_key": (
            model_spec["model_key"]
        ),
        "model_tag": (
            model_spec["model_tag"]
        ),
        "model_digest": (
            model_spec["model_digest"]
        ),
        "run_mode": "cold",
        "cold_run": True,
        "warm_run": False,
        "rag_configuration": "RAG-4",
        "first_pass_valid": (
            policy[
                "first_validation"
            ]["valid"]
        ),
        "retry_used": (
            policy["retry_used"]
        ),
        "retry_recovered": (
            policy["retry_used"]
            and online[
                "validation"
            ]["valid"]
        ),
        "valid_json": (
            online[
                "validation"
            ]["valid"]
        ),
        "runtime_metrics": (
            online["runtime"]
        ),
        "artifact_hashes": (
            artifact_hashes
        ),
        "retrieved_pdf_pages": (
            online["context"][
                "retrieved_pdf_pages"
            ]
        ),
        "retrieved_report_pages": (
            online["context"][
                "retrieved_report_pages"
            ]
        ),
        "unresolved_context_pdf_pages": (
            online["context"][
                "unresolved_pdf_pages"
            ]
        ),
        "score_summary": (
            score["summary"]
        ),
        "retrieval_summary": {
            "authoritative_page_recall": (
                retrieval_evaluation[
                    "authoritative_page_recall"
                ]
            ),
            "field_evidence_recall": (
                retrieval_evaluation[
                    "field_evidence_recall"
                ]
            ),
            "contamination_rate": (
                retrieval_evaluation[
                    "contamination_rate"
                ]
            ),
        },
        "system_ram_peak_mb": None,
        "system_ram_measurement": (
            "not_supported_by_frozen_v1_utilities"
        ),
        "artifacts": {
            "retrieval": (
                "retrieval.json"
            ),
            "context": (
                "retrieved_context.txt"
            ),
            "raw_response": (
                "raw_response.txt"
            ),
            "retry_raw_response": (
                "retry_raw_response.txt"
                if policy["retry_used"]
                else None
            ),
            "prediction": (
                "prediction.json"
                if online[
                    "validation"
                ]["valid"]
                else None
            ),
            "validation": (
                "validation.json"
            ),
            "score": "score.json",
            "retrieval_evaluation": (
                "retrieval_evaluation.json"
            ),
        },
        "mlflow_run_id": None,
    }

    run_id = log_completed_run(
        run_dir=run_dir,
        run_meta=run_meta,
        experiment_name=experiment_name,
        score=score,
        retrieval_evaluation=(
            retrieval_evaluation
        ),
        generation_settings=(
            generation_settings
        ),
    )

    run_meta[
        "mlflow_run_id"
    ] = run_id

    write_json(
        run_dir / "run_meta.json",
        run_meta,
    )

    return run_meta


def rebuild_results_jsonl(
    split_dir: Path,
) -> None:
    records = []

    for meta_path in sorted(
        split_dir.glob(
            "*/run_meta.json"
        )
    ):
        meta = load_json(meta_path)

        if (
            meta.get("status")
            != "completed"
        ):
            continue

        records.append(
            {
                "split": (
                    meta["split"]
                ),
                "doc_id": (
                    meta["doc_id"]
                ),
                "model_key": (
                    meta["model_key"]
                ),
                "mlflow_run_id": (
                    meta[
                        "mlflow_run_id"
                    ]
                ),
                "first_pass_valid": (
                    meta[
                        "first_pass_valid"
                    ]
                ),
                "retry_used": (
                    meta["retry_used"]
                ),
                "retry_recovered": (
                    meta[
                        "retry_recovered"
                    ]
                ),
                "valid_json": (
                    meta["valid_json"]
                ),
                **meta[
                    "score_summary"
                ],
                **meta[
                    "retrieval_summary"
                ],
            }
        )

    output_path = (
        split_dir / "results.jsonl"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as stream:
        for record in records:
            stream.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def parse_arguments(
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen V2 Phase 3 "
            "RAG-4 + Ministral benchmark."
        )
    )

    parser.add_argument(
        "--split",
        required=True,
        choices=[
            "dev",
            "test",
        ],
    )
    parser.add_argument(
        "--prompt-version",
        choices=[
            "v1",
            "v1_1",
        ],
        default="v1",
        help=(
            "Extraction prompt version. "
            "Default: v1."
        ),
    )

    parser.add_argument(
        "--only-doc",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Replace a development smoke run."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    benchmark_version = (
        get_benchmark_version(
            args.prompt_version
        )
    )

    if args.prompt_version == "v1":
        output_root = OUTPUT_ROOT
        experiment_name = (
            V1_EXPERIMENT_NAME
        )
    else:
        output_root = (
            V1_1_OUTPUT_ROOT
        )
        experiment_name = (
            V1_1_EXPERIMENT_NAME
        )

    validate_execution_request(
        split=args.split,
        only_doc=args.only_doc,
        force=args.force,
    )

    documents = read_split(
        args.split
    )

    if (
        args.split == "test"
        and documents
        != EXPECTED_TEST_DOCUMENTS
    ):
        raise ValueError(
            "test.txt does not exactly match "
            "the frozen six-report roster."
        )

    if args.only_doc is not None:
        if args.only_doc not in documents:
            raise ValueError(
                f"{args.only_doc} is not "
                f"in {args.split}.txt."
            )

        documents = [
            args.only_doc
        ]

    model_spec = (
        load_frozen_model()
    )

    verify_installed_model(
        model_spec
    )

    field_schema = load_json(
        FIELD_SCHEMA_PATH
    )

    run_config = load_yaml(
        RUN_CONFIG_PATH
    )

    generation_settings = (
        run_config.get(
            "generation"
        )
    )

    if not isinstance(
        generation_settings,
        dict,
    ):
        raise ValueError(
            "configs/run.yaml has no "
            "generation mapping."
        )

    artifact_hashes = (
        frozen_artifact_hashes(
            system_prompt_path=(
                benchmark_version
                .system_prompt_path
            ),
        )
    )

    split_dir = (
        output_root
        / "cold"
        / args.split
    )

    split_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    failures: list[str] = []

    print(
        f"Split: {args.split}"
    )
    print(
        f"Documents: {len(documents)}"
    )
    print(
        f"Model: {MODEL_TAG}"
    )
    print(
        f"Prompt version: "
        f"{args.prompt_version}"
    )
    print(
        "System prompt SHA256:",
        artifact_hashes[
            "system_prompt_sha256"
        ],
    )
    print(
        "RAG configuration: RAG-4"
    )
    print(
        f"Python: {sys.executable}"
    )

    for number, doc_id in enumerate(
        documents,
        start=1,
    ):
        print(
            "\n"
            + "=" * 72
        )
        print(
            f"[{number}/{len(documents)}] "
            f"{doc_id} × {MODEL_KEY}"
        )
        print(
            "=" * 72
        )

        validate_input_paths(
            doc_id
        )

        (
            run_dir,
            already_completed,
        ) = prepare_run_directory(
            output_root=output_root,
            split=args.split,
            doc_id=doc_id,
            force=args.force,
        )

        if already_completed:
            print(
                "Already completed -> skipping."
            )
            continue

        try:
            meta = run_document(
                split=args.split,
                doc_id=doc_id,
                prompt_version=args.prompt_version,
                experiment_name=experiment_name,
                run_dir=run_dir,
                model_spec=model_spec,
                field_schema=(
                    field_schema
                ),
                generation_settings=(
                    generation_settings
                ),
                artifact_hashes=(
                    artifact_hashes
                ),
            )

            print("Completed.")
            print(
                "Retrieved PDF pages:",
                meta[
                    "retrieved_pdf_pages"
                ],
            )
            print(
                "First-pass valid:",
                meta[
                    "first_pass_valid"
                ],
            )
            print(
                "Retry used:",
                meta["retry_used"],
            )
            print(
                "Valid JSON:",
                meta["valid_json"],
            )
            print(
                "Score:",
                meta[
                    "score_summary"
                ],
            )
            print(
                "Retrieval:",
                meta[
                    "retrieval_summary"
                ],
            )
            print(
                "Artifacts:",
                run_dir,
            )

        except Exception as error:
            failures.append(
                doc_id
            )

            (
                run_dir
                / "error.txt"
            ).write_text(
                traceback.format_exc(),
                encoding="utf-8",
            )

            write_json(
                run_dir
                / "run_meta.json",
                {
                    "status": "failed",
                    "split": args.split,
                    "doc_id": doc_id,
                    "model_key": (
                        MODEL_KEY
                    ),
                    "model_tag": (
                        MODEL_TAG
                    ),
                    "run_mode": "cold",
                    "error_type": (
                        type(error).__name__
                    ),
                    "error_message": (
                        str(error)
                    ),
                    "artifact_hashes": (
                        artifact_hashes
                    ),
                },
            )

            print(
                "FAILED:",
                type(error).__name__,
                str(error),
            )
            print(
                "See:",
                run_dir
                / "error.txt",
            )

        rebuild_results_jsonl(
            split_dir
        )

    rebuild_results_jsonl(
        split_dir
    )

    if failures:
        raise SystemExit(
            "Phase 3 finished with failed "
            "reports: "
            + ", ".join(failures)
        )

    print(
        "\nPhase 3 command completed."
    )
    print(
        "Results:",
        split_dir,
    )


if __name__ == "__main__":
    main()