# Financial Information Extraction from Banking Annual Reports

An experimental NLP pipeline for automatically extracting structured financial information from French-language Tunisian banking annual reports using **Small Language Models (SLMs)**, **Retrieval-Augmented Generation (RAG)**, and cloud-based LLMs.

## Overview

This project investigates whether lightweight, locally deployed language models can reliably extract financial indicators from complex annual reports while reducing dependence on cloud-based LLM APIs.

The system extracts seven financial indicators from individual (non-consolidated) financial statements:

- Total Assets
- Total Equity
- Net Banking Income (PNB)
- Operating Income
- Net Income
- Customer Deposits
- Net Customer Loans

Results are returned as structured JSON containing extracted values, units, fiscal years, and source-page references.

## Technologies

- **Language:** Python
- **Local LLM Runtime:** Ollama
- **Local Models:** Ministral 3B, Qwen3 4B, Gemma 3 4B, Mistral 7B
- **Cloud Baseline:** Google Gemini Flash
- **Embeddings:** BGE-M3
- **Vector Search:** FAISS
- **PDF Processing:** PyMuPDF
- **Experiment Tracking:** MLflow
- **Hardware:** NVIDIA RTX 3050 (6 GB VRAM)

## Methodology

The project consists of two experimental versions.

### V1 — Controlled Financial Extraction

Four local SLMs are benchmarked against Gemini Flash using manually selected financial-statement pages.

All models are evaluated using consistent prompts, structured output requirements, and scoring criteria.

**Objective:** Identify the best-performing local model while isolating financial extraction performance from document retrieval.

### V2 — Full-Report Extraction with Semantic RAG

The best-performing local model, Ministral 3B, is integrated into an automatic retrieval pipeline.

**Local pipeline:**

```text
Complete Annual Report (PDF)
          |
    PDF Text Extraction
          |
    BGE-M3 Embeddings
          |
      FAISS Index
          |
    Semantic Retrieval
          |
      Ministral 3B
          |
    Structured JSON
```

**Cloud baseline:**

```text
Complete Annual Report (PDF)
          |
      Gemini Flash
          |
    Structured JSON
```

Both approaches receive complete annual reports, allowing an end-to-end comparison of extraction accuracy, latency, resource consumption, and estimated API costs.

## Benchmark Results

Strict accuracy measures the percentage of financial fields correctly extracted, including value, unit, scope, and fiscal year.

| System | Input | Strict Accuracy |
|---|---|---|
| Ministral 3B | Manual financial pages | 90.5% |
| Gemini Flash | Manual financial pages | 100.0% |
| RAG-4 + Ministral 3B | Complete report | 71.4% |
| Gemini Flash | Complete PDF | 97.6% |

*Results are based on six test reports (42 financial fields).*

The experiments demonstrate the feasibility of local financial extraction on consumer-grade hardware while highlighting the challenges of automatic evidence retrieval, document scope identification, and structured output reliability.

## Repository Structure

```text
├── configs/          # Model and pipeline configurations
├── data/             # Reports, annotations and dataset splits
├── prompts/          # Versioned extraction prompts
├── schemas/          # Structured JSON schemas
├── scripts/
│   ├── v1/           # Controlled extraction experiments
│   └── v2/           # RAG and full-report benchmarks
├── artifacts/        # Predictions, logs and evaluation results
└── requirements.txt  # Python dependencies
```

## Evaluation

The evaluation framework considers:

- Strict field extraction accuracy
- JSON schema validity
- Retrieval evidence recall
- Scope contamination and unit mismatches
- End-to-end inference latency
- RAM and GPU memory consumption
- Token usage and theoretical cloud API costs

## Project Context

Developed as part of an internship research project exploring the feasibility of **resource-efficient financial document understanding** using local language models.

The primary focus is reproducible benchmarking and comparative analysis of local and cloud-based extraction systems.

## License

License information to be added.
