import unittest
import numpy as np
from ips.neighbors import build_model
from ips.neighbors.formulas import Formulas


def _pred(name, d, k):
    m = build_model(name).fit(np.zeros(3, int), np.array([0., 10, 0]), np.array([0., 0, 10]), 1)
    p = m.predict(np.array([[0, 1, 2]]), np.array([d], float), k)
    return (float(p.x[0]), float(p.y[0])), p.n_d0


class TestWeights(unittest.TestCase):
    def test_hand_example(self):
        # d = (2, 4, 8), P1=(0,0) P2=(10,0) P3=(0,10)
        self.assertTrue(np.allclose(_pred("knn", [2, 4, 8], 3)[0], (10 / 3, 10 / 3)))
        self.assertTrue(np.allclose(_pred("wknn_inv_d", [2, 4, 8], 3)[0], (20 / 7, 10 / 7)))
        self.assertTrue(np.allclose(_pred("wknn_inv_d2", [2, 4, 8], 3)[0], (40 / 21, 10 / 21)))

    def test_k1_equals_knn(self):
        base = _pred("knn", [2, 4, 8], 1)[0]
        for n in ("wknn_inv_d", "wknn_inv_d2"):
            self.assertTrue(np.allclose(_pred(n, [2, 4, 8], 1)[0], base))

    def test_zero_distance(self):
        for n in ("wknn_inv_d", "wknn_inv_d2"):
            xy, n0 = _pred(n, [0, 4, 8], 3)
            self.assertTrue(np.allclose(xy, (0, 0))); self.assertEqual(n0, 1)

    def test_weights_sum_to_one_and_decrease(self):
        d, mask = np.array([[2., 4, 8]]), np.ones((1, 3), bool)
        for p in (1, 2):
            w, _ = Formulas.inverse_power_weights(d, mask, p)
            self.assertAlmostEqual(w.sum(), 1.0); self.assertTrue(np.all(np.diff(w[0]) < 0))

    def test_floor_vote_restricts_neighbors(self):
        m = build_model("knn").fit(np.array([0, 0, 1]), np.array([0., 10, 100]), np.array([0., 0, 0]), 2)
        p = m.predict(np.array([[0, 1, 2]]), np.array([[1., 2, 3]]), 3)
        self.assertEqual(int(p.floor[0]), 0); self.assertAlmostEqual(float(p.x[0]), 5.0)   # bỏ láng giềng tầng 1

    def test_sorensen_search_matches_formula(self):
        rng = np.random.default_rng(1); R, Q = rng.random((50, 8)), rng.random((4, 8))
        dist, _ = Formulas.sorensen_search(Q, R, 3)
        self.assertTrue(np.allclose(dist[0], np.sort([Formulas.sorensen(Q[0], r) for r in R])[:3]))


if __name__ == "__main__":
    unittest.main()
