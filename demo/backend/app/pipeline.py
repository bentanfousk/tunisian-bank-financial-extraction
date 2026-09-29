"""Live upload adapter using the frozen research implementation."""

import gc
import json
import logging
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import pymupdf
import yaml

# The live demo uses the pinned BGE-M3 snapshot cached by the research project.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from .paths import MODELS_CONFIG, MODELS_LOCK, RAG_CONFIG, RUN_CONFIG, SCHEMA

log = logging.getLogger(__name__)
Progress = Callable[[str, str, dict[str, Any] | None], None]


class DemoPipelineError(Exception):
    pass


def load_spec(*, validate_frozen: bool = True) -> dict[str, Any]:
    """Resolve the V2 model tag from config and its frozen digest from lock."""
    rag = yaml.safe_load(RAG_CONFIG.read_text(encoding="utf-8"))
    if validate_frozen:
        from scripts.v2.phase3_rag_benchmark.frozen_config import load_frozen_rag_config
        rag = load_frozen_rag_config()
    models = yaml.safe_load(MODELS_CONFIG.read_text(encoding="utf-8"))
    entries = [item for item in models["local_models"] if item["id"] == "ministral3_3b"]
    if len(entries) != 1:
        raise DemoPipelineError("The selected Ministral model is missing from models.yaml.")
    tag = entries[0]["tag"]
    lock = json.loads(MODELS_LOCK.read_text(encoding="utf-8-sig"))
    locked = [item for item in lock["models"] if item["name"] == tag]
    if len(locked) != 1:
        raise DemoPipelineError("The configured Ministral model has no frozen lock entry.")
    return {"model": tag, "digest": locked[0]["digest"], "rag": rag}


def inspect_pdf(pdf_path: Path) -> int:
    try:
        with pymupdf.open(pdf_path) as pdf:
            if not pdf.is_pdf or pdf.page_count < 1:
                raise DemoPipelineError("This file is not a readable PDF report.")
            if pdf.needs_pass:
                raise DemoPipelineError("Password-protected PDFs are not supported.")
            page_count = pdf.page_count
            if page_count > 1000:
                raise DemoPipelineError("This PDF exceeds the 1,000-page demo limit.")
            if not any(pdf[i].get_text().strip() for i in range(page_count)):
                raise DemoPipelineError(
                    "This report does not contain a usable text layer. OCR is not part of the frozen demo pipeline."
                )
            return page_count
    except DemoPipelineError:
        raise
    except Exception as exc:
        raise DemoPipelineError("This file is not a readable PDF report.") from exc


def _citation(field: dict[str, Any], context_pages: list[dict[str, Any]]) -> tuple[int | None, int | None]:
    report_page = field.get("page")
    if report_page is None:
        return None, None
    matching = [page for page in context_pages if page["report_page"] == report_page]
    pdf_page = matching[0]["pdf_page"] if len(matching) == 1 else None
    return pdf_page, report_page


def serialize_result(
    *, job_id: str, filename: str, page_count: int, bank: str, prediction: dict[str, Any],
    merged: list[dict[str, Any]], context_pages: list[dict[str, Any]],
    spec: dict[str, Any], timings: dict[str, float], generation: dict[str, Any],
    policy: dict[str, Any], peak_vram_mb: float | None,
) -> dict[str, Any]:
    labels = {
        "total_assets": "Total assets", "total_equity": "Total equity",
        "net_banking_income": "Net banking income", "operating_income": "Operating income",
        "net_income": "Net income", "customer_deposits": "Customer deposits",
        "net_customer_loans": "Net customer loans",
    }
    fields = []
    for field in prediction["fields"]:
        pdf_page, report_page = _citation(field, context_pages)
        fields.append({"key": field["field"], "label": labels[field["field"]],
                       "status": field["status"], "value": field["value"],
                       "unit_multiplier": field["unit_multiplier"],
                       "scope": field["scope"], "source_year": field["source_year"],
                       "pdf_page": pdf_page, "report_page": report_page,
                       "evidence": field["evidence"], "notes": field.get("notes")})
    context_by_pdf = {page["pdf_page"]: page for page in context_pages}
    retrieval = [{"pdf_page": row["pdf_page"],
                  "report_page": context_by_pdf[row["pdf_page"]]["report_page"],
                  "score": row["best_score"], "matches": row["matches"],
                  "excerpt": row["text"][:600]}
                 for row in merged]
    generation_seconds = generation.get("total_duration_ns")
    eval_seconds = generation.get("eval_duration_ns")
    output_tokens = generation.get("output_tokens")
    tps = output_tokens / (eval_seconds / 1e9) if output_tokens is not None and eval_seconds else None
    return {
        "job_id": job_id,
        "document": {"filename": filename, "page_count": page_count,
                     "bank": bank, "target_year": prediction["target_year"]},
        "system": {"model": spec["model"], "embedding_model": spec["rag"]["embedding_model"],
                   "rag_configuration": "RAG-4", "local": True},
        "retrieval": {"pages": retrieval}, "fields": fields, "raw_prediction": prediction,
        "runtime": {"pdf_loading_seconds": timings.get("pdf_loading"),
                    "page_extraction_seconds": timings.get("page_extraction"),
                    "embedding_seconds": timings.get("embedding"),
                    "indexing_seconds": timings.get("indexing"),
                    "retrieval_seconds": timings.get("retrieval"),
                    "generation_seconds": generation_seconds / 1e9 if generation_seconds else None,
                    "total_seconds": timings.get("total"),
                    "peak_vram_mb": peak_vram_mb, "tokens_per_second": tps},
        "validation": {"first_pass_valid": policy["first_validation"]["valid"],
                       "final_valid": policy["validation"]["valid"],
                       "retry_used": policy["retry_used"]},
    }


def run_pipeline(
    *, job_id: str, job_dir: Path, pdf_path: Path, filename: str,
    bank: str, year: int, progress: Progress, prevalidated_page_count: int | None = None,
    prevalidated_pdf_loading_seconds: float | None = None,
) -> dict[str, Any]:
    """Same research steps and frozen settings, with upload paths and progress hooks."""
    from scripts.phase0.extract_reports import extract_report
    from scripts.phase4.generation_policy import generate_with_one_repair
    from scripts.phase4.gpu_monitor import GpuMemoryMonitor
    from scripts.phase4.prompt_builder import build_messages
    from scripts.v2.phase1_rag.build_index import (
        create_faiss_index, encode_chunks, load_embedding_model,
    )
    from scripts.v2.phase1_rag.chunking import build_page_chunks
    from scripts.v2.phase1_rag.ingestion import ingest_document
    from scripts.v2.phase1_rag.retrieval_queries import build_retrieval_queries
    from scripts.v2.phase1_rag.retrieve import encode_query, merge_query_results, retrieve_query
    from scripts.v2.phase3_rag_benchmark.context_builder import build_retrieved_context
    import faiss
    import torch
    import ollama

    started = perf_counter()
    timings: dict[str, float] = {}
    spec = load_spec()
    try:
        installed = next((item for item in ollama.list().models if item.model == spec["model"]), None)
    except Exception as exc:
        raise DemoPipelineError("Ollama is unavailable. Start Ollama and try again.") from exc
    if installed is None:
        raise DemoPipelineError(f"Ministral is not installed in Ollama: {spec['model']}")
    if installed.digest != spec["digest"]:
        raise DemoPipelineError("The installed Ministral digest differs from the frozen research model.")

    start = perf_counter()
    page_count = prevalidated_page_count if prevalidated_page_count is not None else inspect_pdf(pdf_path)
    timings["pdf_loading"] = (
        prevalidated_pdf_loading_seconds
        if prevalidated_pdf_loading_seconds is not None else perf_counter() - start
    )
    progress("loading_pdf", "PDF loaded", {"page_count": page_count})
    doc_id = f"{bank}_{year}"
    progress("extracting_pages", "Extracting text from all PDF pages", {"page_count": page_count})
    start = perf_counter()
    try:
        extracted = extract_report(pdf_path, doc_id, job_dir)
        pages = ingest_document(doc_id, pdf_path=pdf_path, jsonl_path=Path(extracted["output_path"]))
    except Exception as exc:
        raise DemoPipelineError("Page extraction failed. Check that the PDF has readable text and is not damaged.") from exc
    if not any(page["indexable"] for page in pages):
        raise DemoPipelineError("This report has no usable extracted text. OCR is not part of the frozen demo pipeline.")
    timings["page_extraction"] = perf_counter() - start
    progress("extracting_pages", "Pages extracted", {"page_count": len(pages)})

    progress("embedding", "Loading frozen BGE-M3 and embedding pages", None)
    start = perf_counter()
    try:
        model = load_embedding_model()
        chunks = [{**chunk, "faiss_position": i} for i, chunk in enumerate(build_page_chunks(pages, model.tokenizer))]
    except Exception as exc:
        raise DemoPipelineError("BGE-M3 could not load or prepare the report pages. Verify the pinned model and CUDA setup.") from exc
    if not chunks:
        raise DemoPipelineError("No pages contain indexable text.")
    try:
        embeddings = encode_chunks(model, chunks, batch_size=1)
    except Exception as exc:
        raise DemoPipelineError("BGE-M3 failed while embedding report pages. Check the backend log for details.") from exc
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    timings["embedding"] = perf_counter() - start
    progress("embedding", "BGE-M3 embeddings generated", {"indexable_pages": len(chunks)})

    progress("indexing", "Building FAISS index", None)
    start = perf_counter()
    try:
        index = create_faiss_index(embeddings)
        faiss.write_index(index, str(job_dir / "index.faiss"))
    except Exception as exc:
        raise DemoPipelineError("FAISS could not build the report index. Check the backend log for details.") from exc
    timings["indexing"] = perf_counter() - start
    progress("indexing", "FAISS index built", {"indexed_pages": index.ntotal})
    del embeddings

    progress("retrieving", "Running frozen semantic queries", None)
    start = perf_counter()
    by_query = {}
    try:
        for query in build_retrieval_queries(bank=bank, fiscal_year=year):
            vector = encode_query(model, query["text"])
            by_query[query["query_id"]] = retrieve_query(
                index=index, chunks=chunks, query_vector=vector,
                query_id=query["query_id"], query_text=query["text"],
                fiscal_year=year, top_k=int(spec["rag"]["pages_per_query"]),
                candidate_k=int(spec["rag"]["candidate_k"]),
            )
        merged = merge_query_results(by_query)
    except Exception as exc:
        raise DemoPipelineError("Semantic retrieval failed. Check the backend log for details.") from exc
    if not merged:
        raise DemoPipelineError("Semantic retrieval found no evidence pages.")
    timings["retrieval"] = perf_counter() - start
    progress("retrieving", "Relevant pages retrieved", {"retrieved_pages": len(merged)})
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    progress("building_context", "Resolving report-page citations", None)
    context = build_retrieved_context(pdf_path=pdf_path, merged_results=merged)
    progress("building_context", "Extraction context built", {"retrieved_pages": len(merged)})
    messages = build_messages(doc_id=doc_id, bank=bank, target_year=year,
                              context=context["text"], prompt_version="v1_1")
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    settings = yaml.safe_load(RUN_CONFIG.read_text(encoding="utf-8"))["generation"]
    progress("generating", "Ministral generating structured extraction", None)
    try:
        with GpuMemoryMonitor() as gpu:
            policy = generate_with_one_repair(
                model_id=spec["model"], messages=messages, schema=schema, settings=settings,
                on_retry=lambda _: progress("retrying", "Validation failed; one format repair is running", None),
            )
    except Exception as exc:
        raise DemoPipelineError("Ministral generation failed. Verify Ollama is running and inspect the backend log.") from exc
    generation = policy["generation_result"]
    progress("validating", "Model output validated", {"valid": policy["validation"]["valid"]})
    if not policy["validation"]["valid"]:
        raise DemoPipelineError("Ministral returned invalid structured JSON after the permitted repair attempt.")
    timings["total"] = perf_counter() - started + (prevalidated_pdf_loading_seconds or 0.0)
    result = serialize_result(
        job_id=job_id, filename=filename, page_count=page_count, bank=bank,
        prediction=policy["validation"]["parsed"], merged=merged,
        context_pages=context["pages"], spec=spec, timings=timings,
        generation=generation, policy=policy, peak_vram_mb=gpu.peak_mb,
    )
    (job_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
