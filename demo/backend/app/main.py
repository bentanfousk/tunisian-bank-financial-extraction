"""FastAPI surface for live local extraction and frozen benchmark artifacts."""

import asyncio
import logging
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from . import benchmarks
from .models import ExtractionAccepted, JobState
from .paths import RUNTIME
from .pipeline import DemoPipelineError, inspect_pdf, load_spec, run_pipeline

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()
EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="demo-extraction")

app = FastAPI(title="Financial Report Intelligence", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _snapshot(job_id: str) -> dict[str, Any]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, "Extraction job not found.")
        return {k: v for k, v in job.items() if k not in {"pdf_path", "job_dir"}}


def _event(job_id: str, status: str, message: str, data: dict[str, Any] | None = None) -> None:
    with JOBS_LOCK:
        job = JOBS[job_id]
        job["status"] = status
        event = {"status": status, "message": message,
                 "timestamp": datetime.now(timezone.utc).isoformat(), "data": data or {}}
        job["events"].append(event)


def _run(job_id: str, bank: str, year: int) -> None:
    with JOBS_LOCK:
        job = JOBS[job_id]
        pdf_path, job_dir, filename = job["pdf_path"], job["job_dir"], job["filename"]
        page_count = job["page_count"]
        pdf_loading_seconds = job["pdf_loading_seconds"]
    try:
        result = run_pipeline(
            job_id=job_id, job_dir=job_dir, pdf_path=pdf_path,
            filename=filename, bank=bank, year=year,
            prevalidated_page_count=page_count,
            prevalidated_pdf_loading_seconds=pdf_loading_seconds,
            progress=lambda status, message, data: _event(job_id, status, message, data),
        )
        with JOBS_LOCK:
            JOBS[job_id]["result"] = result
        _event(job_id, "completed", "Extraction complete", {"fields": len(result["fields"])})
    except DemoPipelineError as exc:
        log.exception("Extraction %s failed: %s", job_id, exc)
        with JOBS_LOCK:
            JOBS[job_id]["error"] = str(exc)
        _event(job_id, "failed", str(exc))
    except Exception:
        log.exception("Extraction %s failed unexpectedly", job_id)
        with JOBS_LOCK:
            JOBS[job_id]["error"] = "Extraction failed. Check the backend log for details."
        _event(job_id, "failed", "Extraction failed. Check the backend log for details.")


@app.get("/api/health")
def health() -> dict[str, Any]:
    checks: dict[str, Any] = {}
    issues = []
    try:
        spec = load_spec(validate_frozen=False)
        checks.update(model=spec["model"], embedding_model=spec["rag"]["embedding_model"],
                      rag_config="RAG-4", frozen_config=True)
        try:
            from huggingface_hub import try_to_load_from_cache
            cached = try_to_load_from_cache(
                spec["rag"]["embedding_model"], "modules.json",
                revision=spec["rag"]["embedding_revision"],
            )
            checks["embedding_cached"] = isinstance(cached, str) and Path(cached).is_file()
        except Exception:
            checks["embedding_cached"] = False
        if not checks["embedding_cached"]:
            issues.append("The pinned BGE-M3 revision is not cached locally. Cache it before live extraction.")
    except Exception as exc:
        spec = None
        checks["frozen_config"] = False
        issues.append(f"Frozen configuration unavailable: {exc}")
    try:
        import faiss  # noqa: F401
        checks["faiss"] = True
    except ImportError:
        checks["faiss"] = False
        issues.append("FAISS is not installed.")
    try:
        import torch
        checks["cuda_available"] = torch.cuda.is_available()
        if not checks["cuda_available"]:
            issues.append("CUDA is unavailable; the frozen BGE-M3 loader requires a GPU.")
    except ImportError:
        checks["cuda_available"] = False
        issues.append("PyTorch is not installed.")
    try:
        import ollama
        response = ollama.Client(timeout=3).list()
        checks["ollama"] = True
        model = next((item for item in response.models if spec and item.model == spec["model"]), None)
        checks["model_available"] = model is not None and model.digest == spec["digest"]
        if not checks["model_available"]:
            issues.append("The configured frozen Ministral model is missing or has a different digest.")
    except Exception:
        checks["ollama"] = False
        checks["model_available"] = False
        issues.append("Ollama is unavailable. Start Ollama to run live extraction.")
    checks["status"] = "ready" if not issues else "needs_attention"
    checks["issues"] = issues
    return checks


@app.post("/api/extractions", status_code=202, response_model=ExtractionAccepted)
async def create_extraction(
    file: UploadFile = File(...), bank: str = Form(...), year: int = Form(...),
) -> dict[str, str]:
    if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(422, "Select a PDF annual report.")
    if file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(422, "Select a PDF annual report.")
    bank = bank.strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{2,40}", bank):
        raise HTTPException(422, "Enter a bank identifier using letters, digits, underscores, or hyphens.")
    if not 2000 <= year <= 2100:
        raise HTTPException(422, "Enter a fiscal year between 2000 and 2100.")
    job_id = uuid.uuid4().hex
    job_dir = RUNTIME / job_id
    job_dir.mkdir(parents=True, exist_ok=False)
    pdf_path = job_dir / "report.pdf"
    size = 0
    try:
        with pdf_path.open("wb") as stream:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "PDF exceeds the 100 MB demo upload limit.")
                stream.write(chunk)
        with pdf_path.open("rb") as stream:
            if stream.read(5) != b"%PDF-":
                raise HTTPException(422, "This file is not a valid PDF.")
        try:
            loading_start = perf_counter()
            page_count = await asyncio.to_thread(inspect_pdf, pdf_path)
            pdf_loading_seconds = perf_counter() - loading_start
        except DemoPipelineError as exc:
            raise HTTPException(422, str(exc)) from exc
    except Exception:
        pdf_path.unlink(missing_ok=True)
        job_dir.rmdir()
        raise
    safe_name = Path(file.filename.replace("\\", "/")).name
    with JOBS_LOCK:
        JOBS[job_id] = {
            "job_id": job_id, "status": "queued", "filename": safe_name,
            "bank": bank, "year": year, "result": None, "error": None,
            "page_count": page_count,
            "pdf_loading_seconds": pdf_loading_seconds,
            "events": [{"status": "queued", "message": "Waiting for local pipeline",
                        "timestamp": datetime.now(timezone.utc).isoformat(), "data": {}}],
            "pdf_path": pdf_path, "job_dir": job_dir,
        }
    EXECUTOR.submit(_run, job_id, bank, year)
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/extractions/{job_id}", response_model=JobState)
def get_extraction(job_id: str) -> dict[str, Any]:
    return _snapshot(job_id)


@app.get("/api/extractions/{job_id}/pdf")
def get_pdf(job_id: str) -> FileResponse:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, "Extraction job not found.")
        path = job["pdf_path"]
    return FileResponse(path, media_type="application/pdf", filename="report.pdf", content_disposition_type="inline")


@app.get("/api/extractions/{job_id}/events")
async def extraction_events(job_id: str) -> StreamingResponse:
    _snapshot(job_id)

    async def stream():
        import json
        position = 0
        while True:
            state = _snapshot(job_id)
            events = state["events"]
            for event in events[position:]:
                yield f"data: {json.dumps(event)}\n\n"
            position = len(events)
            if state["status"] in {"completed", "failed"} and position == len(events):
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.get("/api/benchmarks/summary")
def benchmark_summary() -> dict[str, Any]:
    try:
        return benchmarks.summary()
    except Exception:
        log.exception("Could not load frozen benchmark artifacts")
        raise HTTPException(503, "Frozen benchmark artifacts are unavailable.")


@app.get("/api/benchmarks/v1")
def benchmark_v1() -> dict[str, Any]:
    return {"models": benchmarks.models()}


@app.get("/api/benchmarks/v2")
def benchmark_v2() -> dict[str, Any]:
    data = benchmarks.summary()
    return {key: data[key] for key in ("headline", "retrieval", "local_operational", "cloud_operational", "reports")}


@app.get("/api/benchmarks/errors")
def benchmark_errors() -> dict[str, Any]:
    return benchmarks.errors()
