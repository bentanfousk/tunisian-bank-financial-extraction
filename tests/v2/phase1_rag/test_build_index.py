import unittest

import faiss
import numpy as np

from scripts.v2.phase1_rag.build_index import (
    create_faiss_index,
)


class TestFaissIndexConstruction(unittest.TestCase):

    def test_vectors_are_added_in_order(self):
        embeddings = np.zeros(
            (2, 1024),
            dtype=np.float32,
        )
        embeddings[0, 0] = 1.0
        embeddings[1, 1] = 1.0

        index = create_faiss_index(embeddings)

        query = np.zeros(
            (1, 1024),
            dtype=np.float32,
        )
        query[0, 0] = 1.0

        scores, indices = index.search(query, 2)

        self.assertEqual(index.ntotal, 2)
        self.assertEqual(indices[0, 0], 0)
        self.assertAlmostEqual(
            scores[0, 0],
            1.0,
            places=6,
        )

    def test_empty_embeddings_are_rejected(self):
        embeddings = np.empty(
            (0, 1024),
            dtype=np.float32,
        )

        with self.assertRaises(ValueError):
            create_faiss_index(embeddings)

    def test_wrong_dimension_is_rejected(self):
        embeddings = np.ones(
            (2, 10),
            dtype=np.float32,
        )

        with self.assertRaises(ValueError):
            create_faiss_index(embeddings)

    def test_non_finite_embeddings_are_rejected(self):
        embeddings = np.ones(
            (1, 1024),
            dtype=np.float32,
        )
        embeddings[0, 0] = np.nan

        with self.assertRaises(ValueError):
            create_faiss_index(embeddings)

    def test_unnormalized_embeddings_are_rejected(self):
        embeddings = np.ones(
            (1, 1024),
            dtype=np.float32,
        )

        with self.assertRaises(ValueError):
            create_faiss_index(embeddings)


if __name__ == "__main__":
    unittest.main()