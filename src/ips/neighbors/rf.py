"""Mục "rf" trong danh sách model: chọn trước tập con WAP bằng Random Forest rồi mới đưa vào kNN/WKNN.

Đây KHÔNG phải model định vị. Chọn mục này thì runner chạy như thường (kNN, WKNN 1/d, WKNN 1/d²) rồi chạy thêm luồng
quét tỉ lệ WAP trong ips.experiments.knn_wap_sweep với xếp hạng WAP của file này. Không chọn thì RF không chạy.
Xếp hạng chỉ dùng train: hai RF (phân loại tòa + tầng, hồi quy x y), importance chuẩn hóa rồi trộn theo w_cls."""
import time
from pathlib import Path
import numpy as np

MODEL_NAME = "rf"


class RFSelector:
    is_wap_selector = True            # runner dựa vào cờ này để không đưa mục này vào so sánh theo tòa
    display_name = "Random Forest (chọn WAP)"
    color = "#C44E52"
    out_dir = "wap_rf"
    defaults = dict(n_trees=100, w_cls=0.5)

    def rank(self, Xtr, ref, cols, s, seed):
        from ips.experiments.knn_wap_sweep import RankResult
        tb, tf, tx, ty = ref
        meta = np.array([len(Xtr), Xtr.shape[1], s["n_trees"], seed, s["w_cls"]])
        cache = Path(s["rank_cache"]) if s["rank_cache"] else None
        z = np.load(cache) if cache is not None and cache.exists() else None
        if z is not None and {"imp_cls", "imp_reg", "t_rf", "meta"} <= set(z.files) and np.array_equal(z["meta"][:4], meta[:4]):
            imp_cls, imp_reg, t_rf = z["imp_cls"], z["imp_reg"], float(z["t_rf"])
            print(f"[m-sweep] Đã nạp xếp hạng RF từ {cache} (T_RF = {t_rf:.1f}s)")
        else:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            nf = int(tf.max()) + 1
            t0 = time.perf_counter()
            rf_c = RandomForestClassifier(n_estimators=s["n_trees"], max_features="sqrt", random_state=seed, n_jobs=-1).fit(Xtr, tb * nf + tf)
            rf_r = RandomForestRegressor(n_estimators=s["n_trees"], max_features="sqrt", random_state=seed, n_jobs=-1).fit(Xtr, np.column_stack([tx, ty]))
            t_rf = time.perf_counter() - t0
            imp_cls, imp_reg = rf_c.feature_importances_, rf_r.feature_importances_
            if cache is not None: np.savez(cache, imp_cls=imp_cls, imp_reg=imp_reg, t_rf=t_rf, meta=meta)
            print(f"[m-sweep] Đã huấn luyện RF (2 mô hình x {s['n_trees']} cây): T_RF = {t_rf:.1f}s")
        imp = s["w_cls"] * imp_cls / imp_cls.sum() + (1 - s["w_cls"]) * imp_reg / imp_reg.sum()
        order = np.argsort(-imp, kind="stable")
        import pandas as pd
        table = pd.DataFrame({"hang": np.arange(1, len(order) + 1), "WAP": np.array(cols)[order], "importance": imp[order],
                              "imp_phan_loai": imp_cls[order] / imp_cls.sum(), "imp_hoi_quy": imp_reg[order] / imp_reg.sum()})
        return RankResult({"rf": order}, {"rf": t_rf}, {"rf": "RF"}, {"rf_importance.csv": table})


def build_model(**params):
    return RFSelector()
