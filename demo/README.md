# Financial Report Intelligence demo

This local application exposes the completed V2 Tunisian bank annual-report extraction system and visualizes the completed V1/V2 research results. The live workflow extracts seven structured fields from a complete text-based PDF. It does not rerun or modify benchmark experiments, annotations, the test split, or their artifacts. Live uploads and indices are isolated in the ignored `demo_runtime/` directory.

## Architecture

```text
React + TypeScript + Vite
        ↓ REST / SSE
FastAPI → existing page extraction → frozen BGE-M3 / FAISS retrieval
                                   → V1.1 prompt → frozen Ministral via Ollama
                                   → existing JSON validation and one repair attempt

FastAPI → saved final-analysis artifacts → Benchmark dashboard
```

The live adapter imports `scripts/phase0/extract_reports.py`, `scripts/v2/phase1_rag/{ingestion,chunking,build_index,retrieval_queries,retrieve}.py`, `scripts/v2/phase3_rag_benchmark/{frozen_config,context_builder}.py`, and `scripts/phase4/{prompt_builder,generation_policy,gpu_monitor}.py`. The selected model tag is resolved from `configs/models.yaml` and checked against `configs/models.lock.json`. The frozen retrieval settings come from `configs/rag_v1.yaml`. The V1.1 system prompt, user template, output schema, and generation settings are loaded from their existing files.

The only research-code changes are an optional upload-path argument in V2 ingestion and an optional retry-progress callback in the existing generation policy. Defaults and CLI behavior remain unchanged.

## Prerequisites

- Windows PowerShell, Node.js 20+, npm, and the project's Python 3.10+ virtual environment (the tested `.venv` is Python 3.10.11).
- Ollama running locally with the exact locked `ministral3_3b` tag and digest in `configs/models.yaml` / `configs/models.lock.json`.
- NVIDIA CUDA GPU and the same CUDA-enabled PyTorch environment used for V2. The frozen BGE-M3 loader currently requires CUDA; CPU-only extraction is not supported by that research code. The benchmark dashboard still works without CUDA or Ollama.
- The pinned BGE-M3 revision must already be cached locally. Live extraction sets Hugging Face offline mode to avoid Hub requests during the demo. No Gemini API key is needed.

From the repository root:

```powershell
.\.venv\Scripts\pip.exe install -r demo\backend\requirements-demo.txt
cd demo\frontend
npm install
cd ..\..
```

Install the project's original dependencies in the same Python environment if they are not already present. The demo requirements add API and test packages; the V2 research imports also require the project's CUDA-enabled `torch`, `pymupdf4llm`, `faiss-cpu`, `sentence-transformers`, `ollama`, `PyYAML`, `jsonschema`, and NVIDIA monitor dependencies. Do not replace an existing CUDA PyTorch build with a generic CPU wheel.

## Run

From the repository root, run the convenience script:

```powershell
.\demo\run_demo.ps1
```

Open `http://127.0.0.1:5173`. The script starts both servers, writes logs under `demo_runtime/`, and stops them when you press Enter. Alternatively use two terminals:

```powershell
# Terminal 1, from repository root
.\.venv\Scripts\python.exe -m uvicorn demo.backend.app.main:app --host 127.0.0.1 --port 8000

# Terminal 2
cd demo\frontend
npm run dev
```

Check `http://127.0.0.1:8000/api/health` for model, config, CUDA, FAISS, and Ollama readiness. Bank identifier and fiscal year are requested because the frozen two-query retrieval and extraction prompt require them. A filename such as `UIB_2024.pdf` pre-fills both; users can correct them before submission. The application accepts one text-based PDF up to 100 MB and 1,000 pages. Image-only reports are rejected because OCR is outside the frozen pipeline.

## API

- `GET /api/health`: readiness checks and issues.
- `POST /api/extractions`: multipart `file`, `bank`, and `year`; returns a queued job ID.
- `GET /api/extractions/{job_id}`: job state, events, and result.
- `GET /api/extractions/{job_id}/events`: server-sent stage events.
- `GET /api/extractions/{job_id}/pdf`: only that uploaded PDF, for evidence viewing.
- `GET /api/benchmarks/summary`, `/v1`, `/v2`, `/errors`: read-only research data.

One worker runs jobs serially to avoid concurrent model loads exhausting GPU memory. Jobs are stored in memory, so restarting the API clears job handles; completed result files remain in `demo_runtime/`. The demo does not implement OCR, cancellation, authentication, or scheduled retention cleanup. The PDF viewer uses the browser's built-in PDF support and physical page fragment navigation. A field's physical PDF page appears only when its reported page maps uniquely to retrieved context metadata.

Evidence text is shown exactly as returned by Ministral. It may abbreviate a table row's Markdown separators or neighboring columns, so the evidence panel also links to the mapped physical PDF page for source verification.

## Benchmark sources

The final V1.1 analysis is used rather than the earlier V1 prompt outputs. The dashboard reads these files at request time:

| Dashboard section | Frozen source |
| --- | --- |
| Four-condition headline | `artifacts/v2/phase6_final_analysis/headline_comparison.csv` |
| V1 model ranking, validity, citations, resources | `artifacts/phase8_v1_1/analysis.json` |
| V2 error taxonomy and field drill-down | `artifacts/v2/phase6_final_analysis/rag_error_analysis.csv` |
| V2 local per-report accuracy and runtime | `artifacts/v2/phase6_final_analysis/rag_per_report.csv` |
| Cloud per-report runtime and theoretical cost | `artifacts/v2/phase6_final_analysis/gemini_per_report.csv` |
| Cloud per-report accuracy | `artifacts/v2/phase5_gemini_full_pdf/test/*/score.json` |
| V2 retrieval/resource aggregates | `artifacts/v2/phase6_final_analysis/analysis.json` and `configs/rag_v1.yaml` |

The final analysis records 38/42 controlled local fields, 42/42 controlled cloud fields, 30/42 complete-report local fields, and 41/42 complete-PDF cloud fields. The dashboard computes presentation values from the saved artifacts. Gemini cost is labeled **theoretical**, since an actual charge was not recorded. System RAM and live metrics that were not measured show N/A.

The headline cards use mean per-report citation accuracy from V2 final analysis. The V1 model table uses field-level citation accuracy from the V1.1 analysis; these have different denominators and can differ slightly.

## Verify

```powershell
.\.venv\Scripts\python.exe -m pytest demo\backend\tests -q
cd demo\frontend
npm run typecheck
npm run build
cd ..\..
```

For a manual complete local extraction using a known research report without writing to research directories:

```powershell
.\.venv\Scripts\python.exe -m demo.backend.manual_extract data\raw\UIB_2024.pdf --bank UIB --year 2024
```

That command copies the report into a unique `demo_runtime/` job directory, runs the complete frozen V2 path, and prints the seven fields and validation status. It needs the local model and CUDA prerequisites above.
