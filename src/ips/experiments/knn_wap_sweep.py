"""Quét số WAP (m) theo luồng Colab Lần 6.5, chạy ngay sau phần so sánh kNN/WKNN của knn_runner.run.

Thứ tự: (1) đánh giá/xếp hạng WAP chỉ bằng train (Random Forest và Độ phủ)
        (2) với mỗi m trong lưới: giữ top-m WAP, tìm 20 láng giềng một lần
        (3) đưa từng m vào các model của ips.neighbors (kNN, WKNN 1/d, WKNN 1/d²) và quét k = 1..k_max
        (4) bảng từng m, bảng tổng hợp + Wilcoxon (Bonferroni) so với baseline dùng toàn bộ WAP, chọn m, biểu đồ.

Giao thức theo Lần 6.5, KHÁC phần so sánh theo tòa của knn_runner: tham chiếu là TOÀN BỘ train, tự vote tòa -> tầng,
đánh giá trên official test (validationData.csv), sai số chỉ tính trên mẫu đúng cả tòa lẫn tầng; k và m được chọn
trên chính tập này nên kết quả hơi lạc quan (UJIIndoorLoc chỉ có train và validation). Kết quả nằm riêng trong
thư mục wap_sweep, không dùng để thay thế bảng chính thức của runner.
Cấu hình qua khóa `wap_sweep` của preset knn_*.json (xem configs/knn_full.json).
"""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from ips.neighbors import available_models, build_model
from ips.neighbors.formulas import Formulas

DEFAULTS = dict(enabled=True, rankings=["rf", "coverage"], m_values=list(range(200, 301, 10)), n_trees=100,
                w_cls=0.5, n_rep=3, sr_tol=0.5, sr_small=1.0, err_small=5.0, rank_cache=None)
RNAME = {"rf": "RF", "coverage": "Độ phủ"}
NOT_DETECTED = 100


def sweep_settings(cfg):
    raw = dict(cfg.wap_sweep or {})
    bad = set(raw) - set(DEFAULTS)
    if bad:
        raise ValueError(f"Khóa wap_sweep không hợp lệ: {sorted(bad)}")
    s = {**DEFAULTS, **raw}
    unknown = set(s["rankings"]) - set(RNAME)
    if not s["rankings"] or unknown:
        raise ValueError(f"wap_sweep.rankings phải thuộc {sorted(RNAME)}; nhận {s['rankings']}")
    return s


# ---------------------------------------------------------------- dữ liệu
def prepare(train, test, cfg):
    """Biểu diễn bằng đúng bộ tiền xử lý đã chọn (fit trên TOÀN BỘ train), AP vắng -> 0 như runner."""
    from ips.data.preprocessors.factory import create_preprocessor
    if cfg.max_train: train = train.sample(n=min(cfg.max_train, len(train)), random_state=cfg.seed).reset_index(drop=True)
    if cfg.max_test: test = test.sample(n=min(cfg.max_test, len(test)), random_state=cfg.seed).reset_index(drop=True)
    pre = create_preprocessor(cfg.preprocess, **cfg.preprocess_params).fit(train)
    cols = list(pre.feature_columns)
    Xtr, Xva = (Formulas.missing_to_zero(pre.transform(f)) for f in (train, test))
    if all(c in train.columns for c in cols):
        cover = (train[cols].to_numpy() != NOT_DETECTED).sum(0)        # số mẫu train bắt được mỗi WAP
    else:
        cover = (Xtr > 0).sum(0)
    lab = lambda f: (f.BUILDINGID.to_numpy(int), f.FLOOR.to_numpy(int), f.LONGITUDE.to_numpy(float), f.LATITUDE.to_numpy(float))
    return Xtr, Xva, cover, lab(train), lab(test)


# ---------------------------------------------------------------- xếp hạng WAP
def rank_wap(Xtr, ref, cover, s, seed):
    """Trả (thứ tự WAP theo từng cách, thời gian tạo xếp hạng)."""
    tb, tf, tx, ty = ref
    orders, times = {}, {}
    if "rf" in s["rankings"]:
        meta = np.array([len(Xtr), Xtr.shape[1], s["n_trees"], seed])
        cache = Path(s["rank_cache"]) if s["rank_cache"] else None
        z = np.load(cache) if cache is not None and cache.exists() else None
        if z is not None and {"imp_cls", "imp_reg", "t_rf", "meta"} <= set(z.files) and np.array_equal(z["meta"], meta):
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
            print(f"[m-sweep] Đã huấn luyện RF: T_RF = {t_rf:.1f}s")
        imp = s["w_cls"] * imp_cls / imp_cls.sum() + (1 - s["w_cls"]) * imp_reg / imp_reg.sum()
        orders["rf"], times["rf"] = np.argsort(-imp, kind="stable"), t_rf
    if "coverage" in s["rankings"]:
        t0 = time.perf_counter()
        orders["coverage"] = np.argsort(-cover, kind="stable")
        times["coverage"] = time.perf_counter() - t0
    return orders, times


# ---------------------------------------------------------------- dự đoán và đánh giá (Lần 6.5)
def predict(model, indices, dists, k, ref, n_buildings, n_floors):
    """Vote tòa -> tầng trên toàn bộ train, trọng số do model (ips.neighbors) quyết định, rồi gộp tọa độ."""
    rb, rx, ry = ref
    nb, d = indices[:, :k], dists[:, :k]
    pb, pf, mask = Formulas.vote_building_floor(rb[nb], model.rf[nb], n_buildings, n_floors)
    w, n0 = model.weights(d, mask)
    px, py = Formulas.weighted_position(w, rx[nb], ry[nb])
    return pb, pf, px, py, n0


def evaluate(pb, pf, px, py, truth):
    vb, vf, vx, vy = truth
    ok = (pb == vb) & (pf == vf)
    e_all = Formulas.position_error(px, py, vx, vy)
    e = e_all[ok] if ok.any() else np.full(1, np.nan)
    sm = Formulas.error_summary(e)
    return dict(success_rate=ok.mean() * 100, mean_error=sm["mean"], median=sm["median"], rmse=sm["rmse"], p90=sm["p90"],
                le1=sm["le1"], le3=sm["le3"], le5=sm["le5"]), ok, e_all


# ---------------------------------------------------------------- quét m x k x model
class WapSweep:
    def __init__(self, Xtr, Xva, ref, truth, models, s, k_max):
        self.Xtr, self.Xva, self.s, self.k_max = Xtr, Xva, s, k_max
        self.tb, self.tf, self.tx, self.ty = ref
        self.truth, self.keys = truth, list(models)
        self.nb = int(max(self.tb.max(), truth[0].max())) + 1
        self.nf = int(max(self.tf.max(), truth[1].max())) + 1
        self.models = {m: build_model(m).fit(self.tf, self.tx, self.ty, self.nf) for m in self.keys}
        self.name = {m: self.models[m].display_name for m in self.keys}
        self.color = {m: self.models[m].color for m in self.keys}
        self.n_wap = Xtr.shape[1]
        self.m_list = sorted({m for m in s["m_values"] if 1 <= m < self.n_wap} | {self.n_wap})
        self.k_max = min(k_max, len(Xtr))
        self.res, self.zero, self.rows = {}, {}, []
        self._base = None                                    # baseline m = N_WAP giống nhau ở mọi cách xếp hạng

    def _one(self, rk, m, order, t_rank):
        cols = order[:m]
        t0 = time.perf_counter()
        Xt, Xv = np.ascontiguousarray(self.Xtr[:, cols]), np.ascontiguousarray(self.Xva[:, cols])
        t_sel = time.perf_counter() - t0
        self.zero[(rk, m)] = (int((Xt.sum(1) == 0).sum()), int((Xv.sum(1) == 0).sum()))
        if m == self.n_wap and self._base is not None:        # dùng lại kết quả tìm láng giềng của baseline
            dist, idx, t_srch = self._base
        else:
            t0 = time.perf_counter(); dist, idx = Formulas.sorensen_search(Xv, Xt, self.k_max); t_srch = time.perf_counter() - t0
            if m == self.n_wap: self._base = (dist, idx, t_srch)
        ref = (self.tb, self.tx, self.ty)
        for me in self.keys:
            for k in range(1, self.k_max + 1):
                ts = []
                for _ in range(self.s["n_rep"]):
                    t0 = time.perf_counter(); out = predict(self.models[me], idx, dist, k, ref, self.nb, self.nf); ts.append(time.perf_counter() - t0)
                r, ok, e_all = evaluate(*out[:4], self.truth)
                tot = t_sel + t_srch + min(ts); t_rk = t_rank if m < self.n_wap else 0.0
                self.rows.append(dict(rk=rk, m=m, pct_wap=100 * m / self.n_wap, method=me, k=k, n_query_d0=out[4], **r,
                                      t_select_s=t_sel, t_search_s=t_srch, t_predict_s=min(ts), total_s=tot,
                                      t_rank_s=t_rk, total_with_rank_s=tot + t_rk,
                                      zero_train=self.zero[(rk, m)][0], zero_val=self.zero[(rk, m)][1]))
                self.res[(rk, m, me, k)] = (ok, e_all)

    def run(self, orders, rank_t):
        for rk, order in orders.items():
            for m in self.m_list:
                self._one(rk, m, order, rank_t[rk])
            print(f"[m-sweep] [{RNAME[rk]}] xong {len(self.m_list)} mức m")
        self.sweep = pd.DataFrame(self.rows)
        self.sw = self.sweep.set_index(["rk", "m", "method", "k"]).sort_index()
        self.rks = list(orders)
        self.best = {}
        for rk in self.rks:
            for m in self.m_list:
                for me in self.keys:
                    d = self.sweep[(self.sweep.rk == rk) & (self.sweep.m == m) & (self.sweep.method == me)]
                    self.best[(rk, m, me)] = int(d.sort_values(["success_rate", "mean_error"], ascending=[False, True]).iloc[0].k)
        self.base_rk = self.rks[0]
        self.k_fix = self.best[(self.base_rk, self.n_wap, self.keys[-1])]   # k chung = k tốt nhất của model cuối ở baseline
        return self

    def get(self, rk, m, me, k=None):
        return self.sw.loc[(rk, m, me, self.best[(rk, m, me)] if k is None else k)]

    # ---------- bảng từng m ----------
    def m_table(self, rk, m):
        out = []
        for me in self.keys:
            for tag, kk in (("k tốt nhất riêng", None), (f"cùng k = {self.k_fix}", self.k_fix)):
                a, b = self.get(rk, m, me, kk), self.get(self.base_rk, self.n_wap, me, kk)
                out.append({"Phương pháp": self.name[me], "Cách chọn k": tag, "k": int(a.name[3]), "Success rate (%)": a.success_rate,
                            "d_SR (điểm)": a.success_rate - b.success_rate, "Mean (m)": a.mean_error, "Median (m)": a["median"],
                            "RMSE (m)": a.rmse, "P90 (m)": a.p90, "≤1m (%)": a.le1, "≤3m (%)": a.le3, "≤5m (%)": a.le5,
                            "d_err (m)": a.mean_error - b.mean_error, "d_err (%)": 100 * (a.mean_error / b.mean_error - 1),
                            "Tổng (s)": a.total_s, "Tổng có xếp hạng (s)": a.total_with_rank_s,
                            "% mẫu val toàn 0": 100 * self.zero[(rk, m)][1] / len(self.Xva)})
        return pd.DataFrame(out)

    def tables(self):
        self.by_m = {(rk, m): self.m_table(rk, m) for rk in self.rks for m in self.m_list}
        with pd.option_context("display.width", 250, "display.max_columns", None):
            for (rk, m), t in self.by_m.items():
                print(f"\n--- [m-sweep] Xếp hạng: {RNAME[rk]} | m = {m} ({100 * m / self.n_wap:.1f}% N_WAP) ---")
                print(t.round(3).to_string(index=False))

    # ---------- tổng hợp + Wilcoxon ----------
    def summarize(self):
        from scipy.stats import wilcoxon
        s = self.s
        self.alpha = 0.05 / max(len(self.m_list) - 1, 1)
        recs = []
        for rk in self.rks:
            for m in self.m_list:
                for me in self.keys:
                    a, b = self.get(rk, m, me, self.k_fix), self.get(self.base_rk, self.n_wap, me, self.k_fix)
                    ok1, e1 = self.res[(rk, m, me, self.k_fix)]; ok0, e0 = self.res[(self.base_rk, self.n_wap, me, self.k_fix)]
                    both = ok1 & ok0; dl = e1[both] - e0[both]; pv = np.nan
                    if m < self.n_wap and both.any():
                        try: pv = wilcoxon(e1[both], e0[both]).pvalue
                        except ValueError: pass
                    recs.append(dict(rk=rk, m=m, pct_wap=a.pct_wap, method=me, success_rate=a.success_rate, d_SR=a.success_rate - b.success_rate,
                                     mean_error=a.mean_error, d_err=a.mean_error - b.mean_error, d_err_pct=100 * (a.mean_error / b.mean_error - 1),
                                     n_paired=int(both.sum()), d_err_paired=dl.mean() if both.any() else np.nan, p_value=pv,
                                     zero_val_pct=100 * self.zero[(rk, m)][1] / len(self.Xva), total_s=a.total_s,
                                     total_with_rank_s=a.total_with_rank_s, speedup=b.total_s / a.total_s))
        self.all = pd.DataFrame(recs); self.all["p_value"] = self.all.p_value.astype(float)
        self.all["improved"] = (self.all.d_err_paired < 0) & (self.all.p_value < self.alpha) & (self.all.d_SR >= -s["sr_tol"])
        with pd.option_context("display.width", 250, "display.max_columns", None):
            print(f"\n--- [m-sweep] Tổng hợp theo m (cùng k = {self.k_fix}); ngưỡng Bonferroni = {self.alpha:.5f}, SR_TOL = {s['sr_tol']} điểm ---")
            for title, col, nd in (("Success rate (%)", "success_rate", 2), ("Sai số trung bình (m)", "mean_error", 3)):
                print(f"{title} theo m:"); print(self.all.pivot_table(index="m", columns=["rk", "method"], values=col).round(nd).to_string())
            print("Chi tiết đầy đủ:"); print(self.all.round(4).to_string(index=False))

    # ---------- chọn m ----------
    def select(self):
        s, sel = self.s, []
        for rk in self.rks:
            for me in self.keys:
                d = self.all[(self.all.rk == rk) & (self.all.method == me) & (self.all.m < self.n_wap)]
                base_err = float(self.get(self.base_rk, self.n_wap, me, self.k_fix).mean_error)
                ok_sr = d[d.d_SR >= -s["sr_tol"]]
                line = f"[m-sweep] [{RNAME[rk]} | {self.name[me]}] baseline (m={self.n_wap}): {base_err:.3f} m. "
                rec = dict(rk=rk, method=me, baseline_err=base_err)
                if len(ok_sr):
                    r = ok_sr.loc[ok_sr.mean_error.idxmin()]
                    tag = ("CẢI THIỆN có ý nghĩa thống kê" if r.improved else
                           "thấp hơn nhưng KHÔNG có ý nghĩa thống kê" if r.d_err < 0 else "KHÔNG tốt hơn baseline")
                    line += (f"m sai số thấp nhất = {int(r.m)} ({r.pct_wap:.0f}% N_WAP): {r.mean_error:.3f} m ({r.d_err:+.3f} m, {r.d_err_pct:+.1f}%), "
                             f"SR {r.success_rate:.2f}% ({r.d_SR:+.2f}), p = {r.p_value:.3g} → {tag}. ")
                    rec.update(m_best=int(r.m), best_err=r.mean_error, best_d_err_pct=r.d_err_pct, best_improved=bool(r.improved))
                else:
                    line += "không m nào giữ được success rate. "
                g = d[(d.d_SR >= -s["sr_small"]) & (d.d_err_pct <= s["err_small"])]
                if len(g):
                    r2 = g.loc[g.m.idxmin()]
                    line += f"m nhỏ nhất gần như không mất chính xác = {int(r2.m)} ({r2.pct_wap:.0f}% N_WAP, nhanh hơn {r2.speedup:.1f} lần không tính xếp hạng)."
                    rec.update(m_small=int(r2.m), m_small_pct=r2.pct_wap)
                else:
                    line += f"không m nào có SR giảm ≤ {s['sr_small']} điểm và sai số tăng ≤ {s['err_small']}%."
                print(line); sel.append(rec)
        self.sel = pd.DataFrame(sel)
        imp = self.all[self.all.improved]
        if len(imp):
            r = imp.loc[imp.mean_error.idxmin()]
            self.verdict = (f"[m-sweep] CẤU HÌNH TỐT NHẤT: xếp hạng {RNAME[r.rk]}, {self.name[r.method]}, m = {int(r.m)} ({r.pct_wap:.0f}% N_WAP): "
                            f"{r.mean_error:.3f} m so với baseline {r.mean_error - r.d_err:.3f} m ({r.d_err_pct:+.1f}%), p = {r.p_value:.3g}; "
                            f"có {len(imp)} cấu hình cải thiện có ý nghĩa thống kê.")
        else:
            self.verdict = ("[m-sweep] KHÔNG có m nào cải thiện có ý nghĩa thống kê so với dùng toàn bộ WAP; giữ toàn bộ WAP nếu ưu tiên độ chính xác, "
                            "hoặc dùng m ở cột m_small nếu ưu tiên gọn/nhanh.")
        print(self.verdict)
        print("[m-sweep] Ghi chú: m và k chọn trên validation, thử nhiều m nên kết quả hơi lạc quan; hãy đọc xu hướng cả đường cong.")

    # ---------- biểu đồ + xuất file ----------
    def plot(self, out):
        try:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[m-sweep] Thiếu matplotlib nên bỏ qua biểu đồ."); return []
        ls = {"rf": "-", "coverage": "--"}
        fig, axes = plt.subplots(3, 1, figsize=(10, 13), sharex=True)
        base_e = {me: float(self.get(self.base_rk, self.n_wap, me, self.k_fix).mean_error) for me in self.keys}
        base_sr = float(self.get(self.base_rk, self.n_wap, self.keys[0], self.k_fix).success_rate)
        P = self.all[self.all.m < self.n_wap]
        for rk in self.rks:
            for me in self.keys:
                d = P[(P.rk == rk) & (P.method == me)].sort_values("m")
                axes[0].plot(d.m, d.mean_error, "o" + ls[rk], color=self.color[me], ms=3, label=f"{self.name[me]} - {RNAME[rk]}")
                axes[1].plot(d.m, d.success_rate, "o" + ls[rk], color=self.color[me], ms=3)
            d = P[(P.rk == rk) & (P.method == self.keys[0])].sort_values("m")
            axes[2].plot(d.m, d.zero_val_pct, "o" + ls[rk], color="#8172B3", ms=3, label=RNAME[rk])
        for me in self.keys: axes[0].axhline(base_e[me], color=self.color[me], ls=":", lw=.9)
        axes[1].axhline(base_sr, color="k", ls=":", lw=.9)
        axes[0].set_ylabel("Sai số trung bình (m)"); axes[0].legend(fontsize=7, ncol=2)
        axes[0].set_title(f"Sai số vị trí theo m (cùng k = {self.k_fix}; đường chấm = baseline m = {self.n_wap})")
        axes[1].set_ylabel("Success rate (%)"); axes[1].set_title("Success rate theo m (đường chấm = baseline)")
        axes[2].set_ylabel("% mẫu validation toàn 0"); axes[2].set_title("Mẫu không bắt được WAP nào trong top-m"); axes[2].legend()
        axes[2].set_xlabel("m (số WAP giữ lại)")
        for a in axes: a.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(out / "m_sweep.png", dpi=150); plt.close(fig)
        return ["m_sweep.png"]

    def export(self, out, orders):
        out.mkdir(parents=True, exist_ok=True)
        self.sweep.to_csv(out / "khao_sat_m_k.csv", index=False)
        self.all.to_csv(out / "tong_hop_theo_m.csv", index=False)
        self.sel.to_csv(out / "chon_m.csv", index=False)
        pd.concat([t.assign(xep_hang=RNAME[rk], m=m) for (rk, m), t in self.by_m.items()]).to_csv(out / "bang_tung_m.csv", index=False)
        pd.DataFrame({RNAME[rk]: o for rk, o in orders.items()}).rename_axis("hang").to_csv(out / "thu_tu_wap.csv")
        (out / "summary.json").write_text(json.dumps(dict(models=self.keys, m_list=self.m_list, k_fix=self.k_fix, n_wap=self.n_wap,
                                                          n_eval=int(len(self.Xva)), bonferroni=self.alpha, verdict=self.verdict),
                                                     ensure_ascii=False, indent=2), encoding="utf-8")
        return ["khao_sat_m_k.csv", "tong_hop_theo_m.csv", "chon_m.csv", "bang_tung_m.csv", "thu_tu_wap.csv", "summary.json"] + self.plot(out)


def run_wap_sweep(train, test, cfg, out_dir):
    """Chạy trọn luồng Lần 6.5 và xuất file vào out_dir (thường là <run_dir>/wap_sweep)."""
    s = sweep_settings(cfg)
    if not s["enabled"]: return None
    names = list(cfg.models) or sorted(available_models())
    Xtr, Xva, cover, ref, truth = prepare(train, test, cfg)
    print(f"\n[m-sweep] Quét số WAP theo Lần 6.5: models={names}, RSS={cfg.preprocess}, N_WAP={Xtr.shape[1]}, train={len(Xtr)}, eval={len(Xva)}")
    orders, rank_t = rank_wap(Xtr, ref, cover, s, cfg.seed)
    sw = WapSweep(Xtr, Xva, ref, truth, names, s, cfg.k_max).run(orders, rank_t)
    print(f"[m-sweep] Các mức m: {sw.m_list} | đã chạy {len(sw.sweep)} lần đánh giá | k chung K_FIX = {sw.k_fix}")
    sw.tables(); sw.summarize(); sw.select()
    files = sw.export(Path(out_dir), orders)
    print(f"[m-sweep] Đã lưu vào {out_dir}: " + ", ".join(files))
    return sw
