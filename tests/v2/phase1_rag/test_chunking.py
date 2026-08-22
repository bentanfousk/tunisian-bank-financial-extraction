import unittest

from scripts.v2.phase1_rag.chunking import (
    build_page_chunks,
)


class FakeTokenizer:

    def encode(
        self,
        text,
        add_special_tokens=True,
        truncation=False,
    ):
        token_count = len(text.split())

        if add_special_tokens:
            token_count += 2

        return list(range(token_count))


def make_page(
    *,
    pdf_page=1,
    text="financial statement page",
    indexable=True,
):
    return {
        "doc_id": "UIB_2024",
        "bank": "UIB",
        "fiscal_year": 2024,
        "pdf_path": "data/raw/UIB_2024.pdf",
        "pdf_page": pdf_page,
        "report_page": None,
        "page_label": None,
        "text": text,
        "character_count": len(text),
        "indexable": indexable,
        "skip_reason": (
            None if indexable else "empty_text"
        ),
    }


class TestPageChunking(unittest.TestCase):

    def test_one_page_becomes_one_chunk(self):
        chunks = build_page_chunks(
            [make_page()],
            FakeTokenizer(),
        )

        self.assertEqual(len(chunks), 1)

        chunk = chunks[0]

        self.assertEqual(
            chunk["chunk_id"],
            "UIB_2024::pdf_page::1",
        )
        self.assertEqual(chunk["pdf_page"], 1)
        self.assertEqual(chunk["chunk_index"], 0)
        self.assertEqual(
            chunk["embedding_token_count"],
            5,
        )

    def test_non_indexable_page_is_excluded(self):
        chunks = build_page_chunks(
            [
                make_page(
                    text="",
                    indexable=False,
                )
            ],
            FakeTokenizer(),
        )

        self.assertEqual(chunks, [])

    def test_over_limit_page_is_rejected(self):
        with self.assertRaises(ValueError):
            build_page_chunks(
                [
                    make_page(
                        text="one two three four",
                    )
                ],
                FakeTokenizer(),
                max_tokens=5,
            )

    def test_duplicate_page_is_rejected(self):
        page = make_page()

        with self.assertRaises(ValueError):
            build_page_chunks(
                [page, page.copy()],
                FakeTokenizer(),
            )


if __name__ == "__main__":
    unittest.main()