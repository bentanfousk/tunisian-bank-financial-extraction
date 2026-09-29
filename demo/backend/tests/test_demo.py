import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from demo.backend.app import benchmarks, main
from demo.backend.app.pipeline import _citation, serialize_result


client = TestClient(main.app)


def test_frozen_artifacts_are_normalized_without_mutation():
    before = (benchmarks.V2_ANALYSIS / "headline_comparison.csv").read_bytes()
    data = benchmarks.summary()
    assert [(row["correct"], row["total"]) for row in data["headline"]] == [
        (38, 42), (42, 42), (30, 42), (41, 42)
    ]
    assert {row["category"]: row["count"] for row in data["errors"]["counts"]} == {
        "schema_failure": 7, "unit_mismatch": 1, "scope_contamination": 4,
    }
    assert next(row for row in data["reports"] if row["report"] == "bna_2024" and row["system"] == "Gemini Flash")["correct"] == 6
    assert (benchmarks.V2_ANALYSIS / "headline_comparison.csv").read_bytes() == before


def test_health_route(monkeypatch):
    monkeypatch.setattr(main, "load_spec", lambda **_: {
        "model": "configured-model", "digest": "digest", "rag": {"embedding_model": "BAAI/bge-m3"}
    })
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["model"] == "configured-model"
    assert response.json()["status"] in {"ready", "needs_attention"}


@pytest.mark.parametrize("filename,content,expected", [
    ("report.txt", b"hello", 422),
    ("report.pdf", b"not a pdf", 422),
])
def test_invalid_upload_rejected(filename, content, expected, monkeypatch, tmp_path):
    monkeypatch.setattr(main, "RUNTIME", tmp_path)
    response = client.post("/api/extractions", data={"bank": "UIB", "year": "2024"},
                           files={"file": (filename, content, "application/pdf")})
    assert response.status_code == expected
    assert list(tmp_path.rglob("*.pdf")) == []


def test_image_only_pdf_rejected_before_queue(monkeypatch, tmp_path):
    import pymupdf
    monkeypatch.setattr(main, "RUNTIME", tmp_path)
    pdf = pymupdf.open()
    pdf.new_page()
    content = pdf.tobytes()
    pdf.close()
    response = client.post("/api/extractions", data={"bank": "UIB", "year": "2024"},
                           files={"file": ("scan.pdf", content, "application/pdf")})
    assert response.status_code == 422
    assert "text layer" in response.json()["detail"]
    assert list(tmp_path.iterdir()) == []


def test_citation_mapping_does_not_invent_physical_page():
    pages = [{"report_page": 140, "pdf_page": 137}, {"report_page": 142, "pdf_page": 139}]
    assert _citation({"page": 140}, pages) == (137, 140)
    assert _citation({"page": None}, pages) == (None, None)
    assert _citation({"page": 999}, pages) == (None, 999)


def test_result_serialization_preserves_all_fields():
    keys = ["total_assets", "total_equity", "net_banking_income", "operating_income",
            "net_income", "customer_deposits", "net_customer_loans"]
    prediction = {"target_year": 2024, "fields": [
        {"field": key, "status": "found", "value": i + 1, "unit_multiplier": 1000,
         "scope": "individual", "source_year": 2024, "page": 140, "evidence": "source text"}
        for i, key in enumerate(keys)
    ]}
    result = serialize_result(
        job_id="job", filename="sample.pdf", page_count=200, bank="UIB", prediction=prediction,
        merged=[{"pdf_page": 137, "best_score": .8, "matches": [], "text": "source text"}],
        context_pages=[{"pdf_page": 137, "report_page": 140}],
        spec={"model": "configured-model", "rag": {"embedding_model": "BAAI/bge-m3"}},
        timings={"total": 10.5}, generation={"total_duration_ns": 2_000_000_000,
                                            "eval_duration_ns": 1_000_000_000, "output_tokens": 20},
        policy={"first_validation": {"valid": True}, "validation": {"valid": True},
                "retry_used": False}, peak_vram_mb=None,
    )
    assert [field["key"] for field in result["fields"]] == keys
    assert result["fields"][0]["pdf_page"] == 137
    assert result["document"]["bank"] == "UIB"
    assert result["runtime"]["peak_vram_mb"] is None
    assert result["runtime"]["tokens_per_second"] == 20


def test_extraction_job_wiring(monkeypatch, tmp_path):
    job_id = "mock-job"
    fake_result = {"fields": [{"key": "total_assets"}]}
    def fake_pipeline(**kwargs):
        assert kwargs["prevalidated_page_count"] == 2
        assert kwargs["prevalidated_pdf_loading_seconds"] == 0.2
        kwargs["progress"]("embedding", "Embeddings generated", {"indexable_pages": 2})
        return fake_result
    monkeypatch.setattr(main, "run_pipeline", fake_pipeline)
    main.JOBS[job_id] = {"job_id": job_id, "status": "queued", "filename": "report.pdf",
                         "bank": "UIB", "year": 2024, "result": None, "error": None,
                         "events": [], "pdf_path": tmp_path / "report.pdf", "job_dir": tmp_path,
                         "page_count": 2, "pdf_loading_seconds": 0.2}
    try:
        main._run(job_id, "UIB", 2024)
        response = client.get(f"/api/extractions/{job_id}")
        assert response.json()["status"] == "completed"
        assert [event["status"] for event in response.json()["events"]] == ["embedding", "completed"]
        assert response.json()["result"] == fake_result
    finally:
        del main.JOBS[job_id]
