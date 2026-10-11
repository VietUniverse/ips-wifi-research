"""Lớp CÔNG THỨC: phép toán thuần (biểu diễn, khoảng cách, vote, trọng số, sai số). Không giữ trạng thái."""
import numpy as np
from sklearn.metrics import pairwise_distances

TOL = 1e-12     # d <= TOL coi như d = 0


class Formulas:
    # ---------- Biểu diễn RSSI ----------
    @staticmethod
    def to_positive(X, min_rss=-104, not_detected=100):
        out = X.copy().astype(float)
        det = X != not_detected
        out[det] = X[det] - (min_rss - 1)
        out[~det] = 0.0
        return out

    @staticmethod
    def to_powed(X, min_rss=-104, beta=np.e, not_detected=100):
        return (Formulas.to_positive(X, min_rss, not_detected) ** beta) / ((-(min_rss - 1)) ** beta)

    @staticmethod
    def missing_to_zero(X):
        """Plugin tiền xử lý của repo đánh dấu AP vắng = -1; khoảng cách Sørensen cần 0."""
        return np.where(X < 0, 0.0, X).astype(float)

    # ---------- Khoảng cách Sørensen + tìm láng giềng ----------
    @staticmethod
    def sorensen(u, v):
        return np.sum(np.abs(u - v)) / (np.sum(u + v) + 1e-12)

    @staticmethod
    def sorensen_search(Q, R, K, chunk=500):
        n = len(Q)
        idx = np.empty((n, K), np.int32)
        dist = np.empty((n, K))
        for s in range(0, n, chunk):
            D = np.nan_to_num(pairwise_distances(Q[s:s + chunk], R, metric="braycurtis", n_jobs=-1), nan=0.0)
            o = np.argsort(D, axis=1, kind="stable")[:, :K]
            idx[s:s + chunk] = o
            dist[s:s + chunk] = np.take_along_axis(D, o, 1)
        return dist, idx

    # ---------- Vote tòa -> tầng (không trọng số) ----------
    @staticmethod
    def vote_building_floor(nb_building, nb_floor, n_buildings, n_floors):
        pb = np.stack([(nb_building == c).sum(1) for c in range(n_buildings)], 1).argmax(1)
        mb = nb_building == pb[:, None]
        pf = np.stack([((nb_floor == c) & mb).sum(1) for c in range(n_floors)], 1).argmax(1)
        return pb, pf, mb & (nb_floor == pf[:, None])

    # ---------- Trọng số (Σw = 1) ----------
    @staticmethod
    def uniform_weights(mask):
        return mask / mask.sum(1, keepdims=True), 0

    @staticmethod
    def inverse_power_weights(dists, mask, p, tol=TOL):
        """w_i = (1/d_i^p)/Σ(1/d_j^p); có láng giềng d≈0 thì lấy trung bình các láng giềng đó."""
        z = mask & (dists <= tol)
        has0 = z.any(1)
        inv = np.where(mask & ~z, 1.0 / np.maximum(dists, tol) ** p, 0.0)
        w = np.where(has0[:, None], z / np.maximum(z.sum(1, keepdims=True), 1),
                     inv / np.maximum(inv.sum(1, keepdims=True), 1e-300))
        return w, int(has0.sum())

    @staticmethod
    def weighted_position(w, neighbor_x, neighbor_y):
        return (w * neighbor_x).sum(1), (w * neighbor_y).sum(1)

    # ---------- Sai số (giao thức repo: tính trên TẤT CẢ mẫu) ----------
    @staticmethod
    def position_error(px, py, tx, ty):
        return np.hypot(px - tx, py - ty)

    @staticmethod
    def error_summary(e):
        return dict(mean=float(e.mean()), median=float(np.median(e)), rmse=float(np.sqrt((e ** 2).mean())),
                    p90=float(np.quantile(e, .9)), le1=float((e <= 1).mean() * 100),
                    le3=float((e <= 3).mean() * 100), le5=float((e <= 5).mean() * 100))
