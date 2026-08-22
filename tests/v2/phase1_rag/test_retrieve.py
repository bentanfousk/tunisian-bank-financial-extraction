import unittest

import faiss
import numpy as np

from scripts.v2.phase1_rag.retrieve import (
    candidate_features,
    merge_query_results,
    rerank_candidates,
    search_index,
)
from scripts.v2.phase1_rag.retrieval_queries import (
    build_retrieval_queries,
)


def make_chunk(position, page):
    return {
        "faiss_position": position,
        "chunk_id": (
            f"UIB_2024::pdf_page::{page}"
        ),
        "doc_id": "UIB_2024",
        "pdf_page": page,
        "report_page": None,
        "text": f"Text for page {page}",
    }


def make_result(
    *,
    rank,
    score,
    page,
    text,
    query_id="balance_sheet",
):
    return {
        "query_id": query_id,
        "query_text": "query",
        "rank": rank,
        "score": score,
        "faiss_position": rank - 1,
        "chunk_id": (
            f"UIB_2024::pdf_page::{page}"
        ),
        "doc_id": "UIB_2024",
        "pdf_page": page,
        "report_page": None,
        "text": text,
    }


class TestSemanticRetrieval(unittest.TestCase):

    def test_fixed_queries_include_scope_year_and_titles(
        self,
    ):
        queries = build_retrieval_queries(
            bank="UIB",
            fiscal_year=2024,
        )

        self.assertEqual(len(queries), 2)

        for query in queries:
            self.assertIn(
                "individuel",
                query["text"].casefold(),
            )
            self.assertIn(
                "2024",
                query["text"],
            )

        self.assertIn(
            "bilan",
            queries[0]["text"].casefold(),
        )
        self.assertIn(
            "état de résultat",
            queries[1]["text"].casefold(),
        )

    def test_search_maps_vectors_to_chunks(self):
        vectors = np.zeros(
            (3, 1024),
            dtype=np.float32,
        )
        vectors[0, 0] = 1.0
        vectors[1, 1] = 1.0
        vectors[2, 2] = 1.0

        index = faiss.IndexFlatIP(1024)
        index.add(vectors)

        query = np.zeros(
            (1, 1024),
            dtype=np.float32,
        )
        query[0, 1] = 1.0

        chunks = [
            make_chunk(0, 10),
            make_chunk(1, 20),
            make_chunk(2, 30),
        ]

        results = search_index(
            index=index,
            chunks=chunks,
            query_vector=query,
            query_id="test",
            query_text="test query",
            top_k=2,
        )

        self.assertEqual(
            results[0]["pdf_page"],
            20,
        )
        self.assertEqual(
            results[0]["faiss_position"],
            1,
        )
        self.assertAlmostEqual(
            results[0]["score"],
            1.0,
            places=6,
        )

    def test_consolidated_statement_is_detected(self):
        features = candidate_features(
            text=(
                "GROUPE UBCI\n"
                "BILAN CONSOLIDÉ\n"
                "Arrêté au 31 décembre 2024"
            ),
            query_id="balance_sheet",
            fiscal_year=2024,
        )

        self.assertTrue(
            features["is_consolidated"]
        )

    def test_reranker_rejects_consolidated_page(self):
        consolidated = make_result(
            rank=1,
            score=0.90,
            page=155,
            text=(
                "GROUPE UBCI BILAN CONSOLIDÉ 2024 "
                "AC1 PA1 CP1 Total actif Capitaux "
                "propres Dépôts clientèle Créances "
                "clientèle"
            ),
        )
        individual = make_result(
            rank=12,
            score=0.65,
            page=95,
            text=(
                "BILAN APRÈS AFFECTATION DES BÉNÉFICES "
                "ARRÊTÉ AU 31 DÉCEMBRE 2024 "
                "AC1 PA1 CP1 Total actif Capitaux "
                "propres Dépôts clientèle Créances "
                "clientèle"
            ),
        )

        reranked = rerank_candidates(
            candidates=[
                consolidated,
                individual,
            ],
            query_id="balance_sheet",
            fiscal_year=2024,
            top_k=2,
        )

        self.assertEqual(len(reranked), 1)
        self.assertEqual(
            reranked[0]["pdf_page"],
            95,
        )
        self.assertEqual(
            reranked[0]["dense_rank"],
            12,
        )

    def test_statement_table_beats_high_dense_summary(self):
        summary = make_result(
            rank=1,
            score=0.90,
            page=46,
            text=(
                "CHIFFRES CLÉS 2024 Total bilan "
                "Capitaux propres Dépôts clientèle "
                "Créances clientèle"
            ),
        )
        statement = make_result(
            rank=43,
            score=0.64,
            page=37,
            text=(
                "BILAN ARRÊTÉ AU 31 DÉCEMBRE 2024 "
                "AC1 PA1 CP1 Total actif Capitaux "
                "propres Dépôts clientèle Créances "
                "clientèle"
            ),
        )

        reranked = rerank_candidates(
            candidates=[summary, statement],
            query_id="balance_sheet",
            fiscal_year=2024,
            top_k=2,
        )

        self.assertEqual(
            reranked[0]["pdf_page"],
            37,
        )

    def test_income_statement_table_is_preferred(self):
        summary = make_result(
            rank=1,
            score=0.91,
            page=128,
            text=(
                "CHIFFRES CLÉS 2024 Produit net "
                "bancaire Résultat d'exploitation "
                "Résultat net"
            ),
            query_id="income_statement",
        )
        statement = make_result(
            rank=48,
            score=0.62,
            page=39,
            text=(
                "ÉTAT DE RÉSULTAT 2024 PR1 PR2 CH1 "
                "CH2 Produit net bancaire Résultat "
                "d'exploitation Résultat de l'exercice"
            ),
            query_id="income_statement",
        )

        reranked = rerank_candidates(
            candidates=[summary, statement],
            query_id="income_statement",
            fiscal_year=2024,
            top_k=2,
        )

        self.assertEqual(
            reranked[0]["pdf_page"],
            39,
        )

    def test_merge_deduplicates_same_page(self):
        shared_result = make_result(
            rank=1,
            score=0.9,
            page=140,
            text="Financial statement",
        )

        second_result = {
            **shared_result,
            "query_id": "income_statement",
            "score": 0.8,
            "rank": 2,
        }

        merged = merge_query_results(
            {
                "balance_sheet": [shared_result],
                "income_statement": [second_result],
            }
        )

        self.assertEqual(len(merged), 1)
        self.assertEqual(
            len(merged[0]["matches"]),
            2,
        )
        self.assertEqual(
            merged[0]["best_score"],
            0.9,
        )
        self.assertEqual(
            merged[0]["best_rank"],
            1,
        )


    def test_fragmented_consolidated_heading_is_detected(
        self,
    ):
        features = candidate_features(
            text=(
                "Société Tunisienne de Banque "
                "BILAN CONSOL<br>"
                "ARRÊTÉ AU 31 DÉCEMBRE 2024"
            ),
            query_id="balance_sheet",
            fiscal_year=2024,
        )

        self.assertTrue(
            features["is_consolidated"]
        )


    def test_markdown_balance_row_codes_are_detected(
        self,
    ):
        features = candidate_features(
            text=(
                "BILAN APRÈS AFFECTATION DES BÉNÉFICES "
                "ARRÊTÉ AU 31 DÉCEMBRE 2024 "
                "|**_AC 1 -_** Caisse | "
                "|**_PA 1 -_** Banque Centrale | "
                "|**_CP 1 -_** Capital | "
                "Total actif Capitaux propres "
                "Dépôts clientèle Créances clientèle"
            ),
            query_id="balance_sheet",
            fiscal_year=2024,
        )

        self.assertEqual(
            features["structure_coverage"],
            3,
        )


    def test_markdown_income_row_codes_are_detected(
        self,
    ):
        features = candidate_features(
            text=(
                "ÉTAT DE RÉSULTAT 2024 "
                "|**_PR 1 -_** Intérêts | "
                "|**_CH 1 -_** Charges | "
                "Produit net bancaire "
                "Résultat d'exploitation "
                "Résultat de l'exercice"
            ),
            query_id="income_statement",
            fiscal_year=2024,
        )

        self.assertEqual(
            features["structure_coverage"],
            2,
        )


if __name__ == "__main__":
    unittest.main()