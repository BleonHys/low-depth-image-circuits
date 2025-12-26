import unittest
import numpy as np

from circuit_optimization.encodings.registry import get_permutation, inverse_permutation, list_encodings
from circuit_optimization.utils.image_encodings import hierarchical_index


class TestPermutations(unittest.TestCase):
    def test_permutations_bijective(self):
        m = n = 8
        for name in list_encodings():
            with self.subTest(name=name):
                perm = get_permutation(m, n, name)
                self.assertEqual(len(perm), m * n)
                self.assertTrue(np.array_equal(np.sort(perm), np.arange(m * n)))

    def test_inverse_permutation(self):
        m = n = 8
        perm = get_permutation(m, n, "snake")
        inv = inverse_permutation(perm)
        self.assertTrue(np.array_equal(perm[inv], np.arange(m * n)))

    def test_morton_hilbert_constraints(self):
        with self.assertRaises(ValueError):
            get_permutation(8, 16, "morton")
        with self.assertRaises(ValueError):
            get_permutation(10, 10, "morton")
        with self.assertRaises(ValueError):
            get_permutation(8, 16, "hilbert")
        with self.assertRaises(ValueError):
            get_permutation(10, 10, "hilbert")

    def test_morton_matches_hierarchical(self):
        n = 8
        perm = get_permutation(n, n, "morton")
        legacy = hierarchical_index(int(np.log2(n))).reshape(-1)
        legacy_perm = np.argsort(legacy, kind="stable")
        self.assertTrue(np.array_equal(perm, legacy_perm))


if __name__ == "__main__":
    unittest.main()
