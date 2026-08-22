import unittest

from scripts.v2.phase1_rag.retrieval_queries import (
    build_retrieval_queries,
)
from scripts.v2.phase3_rag_benchmark.frozen_config import (
    FROZEN_RAG_CONFIG,
    MAXIMUM_PAGES_BEFORE_DEDUPLICATION,
    PAGES_PER_QUERY,
)


class TestPhase3FrozenConfiguration(unittest.TestCase):

    def test_rag4_budget_is_frozen(self):
        self.assertEqual(PAGES_PER_QUERY, 2)
        self.assertEqual(
            MAXIMUM_PAGES_BEFORE_DEDUPLICATION,
            4,
        )
        self.assertEqual(
            FROZEN_RAG_CONFIG["candidate_k"],
            64,
        )

    def test_exactly_two_fixed_queries_are_used(self):
        queries = build_retrieval_queries(
            bank="UIB",
            fiscal_year=2024,
        )

        self.assertEqual(
            [
                query["query_id"]
                for query in queries
            ],
            [
                "balance_sheet",
                "income_statement",
            ],
        )


if __name__ == "__main__":
    unittest.main()