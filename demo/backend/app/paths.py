from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNTIME = ROOT / "demo_runtime"
V1_ANALYSIS = ROOT / "artifacts" / "phase8_v1_1"
V2_ANALYSIS = ROOT / "artifacts" / "v2" / "phase6_final_analysis"
RAG_CONFIG = ROOT / "configs" / "rag_v1.yaml"
MODELS_CONFIG = ROOT / "configs" / "models.yaml"
MODELS_LOCK = ROOT / "configs" / "models.lock.json"
RUN_CONFIG = ROOT / "configs" / "run.yaml"
SCHEMA = ROOT / "schemas" / "field_schema.json"
