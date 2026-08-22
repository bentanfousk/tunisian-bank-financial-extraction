from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from scripts.v2.phase1_rag.build_index import (
    MODEL_NAME,
    MODEL_REVISION,
)
from scripts.v2.phase1_rag.retrieve import (
    CANDIDATE_K,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAG_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "rag_v1.yaml"
)


def load_frozen_rag_config() -> dict[str, Any]:
    """
    Load and verify the development-frozen RAG-4
    configuration without modifying Phase 1 or Phase 2.
    """

    if not RAG_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Frozen RAG configuration not found: "
            f"{RAG_CONFIG_PATH}"
        )

    config = yaml.safe_load(
        RAG_CONFIG_PATH.read_text(encoding="utf-8")
    )

    if not isinstance(config, dict):
        raise ValueError(
            "configs/rag_v1.yaml must contain a mapping."
        )

    expected = {
        "embedding_model": "BAAI/bge-m3",
        "embedding_revision": (
            "5617a9f61b028005a4858fdac845db406aefb181"
        ),
        "chunking": "one_page",
        "similarity": "cosine",
        "faiss_index": "IndexFlatIP",
        "candidate_k": 64,
        "pages_per_query": 2,
        "maximum_pages_before_deduplication": 4,
        "retrieval_method": (
            "bge_m3_faiss_authority_rerank_v1"
        ),
    }

    for key, expected_value in expected.items():
        actual_value = config.get(key)

        if actual_value != expected_value:
            raise ValueError(
                f"Frozen RAG mismatch for {key!r}: "
                f"expected {expected_value!r}, "
                f"found {actual_value!r}."
            )

    if (
        config["maximum_pages_before_deduplication"]
        != config["pages_per_query"] * 2
    ):
        raise ValueError(
            "Frozen page budget is inconsistent."
        )

    if config["embedding_model"] != MODEL_NAME:
        raise ValueError(
            "rag_v1.yaml embedding model disagrees "
            "with the frozen Phase 1 implementation."
        )

    if (
        config["embedding_revision"]
        != MODEL_REVISION
    ):
        raise ValueError(
            "rag_v1.yaml embedding revision disagrees "
            "with the frozen Phase 1 implementation."
        )

    if config["candidate_k"] != CANDIDATE_K:
        raise ValueError(
            "rag_v1.yaml candidate pool disagrees "
            "with the frozen Phase 1 implementation."
        )

    return config


FROZEN_RAG_CONFIG = load_frozen_rag_config()

PAGES_PER_QUERY = int(
    FROZEN_RAG_CONFIG["pages_per_query"]
)

MAXIMUM_PAGES_BEFORE_DEDUPLICATION = int(
    FROZEN_RAG_CONFIG[
        "maximum_pages_before_deduplication"
    ]
)