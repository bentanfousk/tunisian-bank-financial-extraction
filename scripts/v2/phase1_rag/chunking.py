from __future__ import annotations

from typing import Any


BGE_M3_MAX_TOKENS = 8192


def build_page_chunks(
    pages: list[dict[str, Any]],
    tokenizer: Any,
    max_tokens: int = BGE_M3_MAX_TOKENS,
) -> list[dict[str, Any]]:
    """
    Convert normalized, indexable pages into page-level chunks.

    One complete physical PDF page becomes one chunk. Pages
    exceeding the embedding-model limit cause an explicit error
    rather than being silently truncated.
    """

    if max_tokens < 1:
        raise ValueError("max_tokens must be positive.")

    chunks: list[dict[str, Any]] = []
    chunk_ids: set[str] = set()

    for page in pages:
        if not page["indexable"]:
            continue

        text = page["text"]

        if not text.strip():
            raise ValueError(
                f"{page['doc_id']} PDF page "
                f"{page['pdf_page']} is marked indexable "
                "but contains no searchable text."
            )

        token_ids = tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=False,
        )
        token_count = len(token_ids)

        if token_count > max_tokens:
            raise ValueError(
                f"{page['doc_id']} PDF page "
                f"{page['pdf_page']} contains "
                f"{token_count} tokens, exceeding the "
                f"{max_tokens}-token embedding limit."
            )

        chunk_id = (
            f"{page['doc_id']}"
            f"::pdf_page::{page['pdf_page']}"
        )

        if chunk_id in chunk_ids:
            raise ValueError(
                f"Duplicate chunk identifier: {chunk_id}"
            )

        chunk_ids.add(chunk_id)

        chunks.append(
            {
                "chunk_id": chunk_id,
                "chunk_index": 0,
                "doc_id": page["doc_id"],
                "bank": page["bank"],
                "fiscal_year": page["fiscal_year"],
                "pdf_path": page["pdf_path"],
                "pdf_page": page["pdf_page"],
                "report_page": page["report_page"],
                "page_label": page["page_label"],
                "text": text,
                "character_count": page["character_count"],
                "embedding_token_count": token_count,
            }
        )

    return chunks