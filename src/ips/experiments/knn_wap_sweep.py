"""Quét tỉ lệ WAP (theo %) theo luồng Colab Lần 6.5/8; chỉ chạy khi người dùng chọn mục RF hoặc HUIM trong danh sách model.

Thứ tự: (1) bộ chọn (neighbors/rf.py, neighbors/huim_upgrowth.py, neighbors/huim_efim.py) xếp hạng WAP chỉ bằng dữ liệu train
        (2) với mỗi tỉ lệ p% (mặc định 10, 20, ..., 90): giữ top-m WAP, m = round(p% x số WAP); tìm 20 láng giềng một lần
        (3) đưa từng m vào các model gốc của ips.neighbors (kNN, WKNN 1/d, WKNN 1/d²) và quét k = 1..k_max
        (4) bảng từng m, bảng tổng hợp + Wilcoxon (Bonferroni) so với baseline dùng toàn bộ WAP, chọn m, biểu đồ.

Giao thức theo Lần 6.5, KHÁC phần so sánh theo tòa của knn_runner: tham chiếu là TOÀN BỘ train, tự vote tòa -> tầng,
đánh giá trên official test (validationData.csv), sai số chỉ tính trên mẫu đúng cả tòa lẫn tầng; k và m được chọn
trên chính tập này nên kết quả hơi lạc quan (UJIIndoorLoc chỉ có train và validation). Kết quả nằm riêng trong
thư mục của bộ chọn (wap_rf, wap_huim_upgrowth, wap_huim_efim), không thay thế bảng chính thức của runner.
Khi chọn cả hai mục HUIM, `run_huim_pair()` dùng chung min_util và ghi thêm thư mục wap_huim_compare.
Tham số tùy chọn đặt trong khóa `wap_sweep` của preset (không bắt buộc; mặc định xem DEFAULTS và `defaults` của từng bộ chọn).
"""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from ips.neighbors import build_model
from ips.neighbors.formulas import Formulas

class RankResult:
    """Kết quả của một bộ chọn WAP: orders/times/labels theo khóa xếp hạng; extras = {tên file: DataFrame hoặc dict}."""
    def __init__(self, orders, times, labels, extras=None):
        self.orders, self.times, self.labels, self.extras = orders, times, labels, extras or {}


DEFAULTS = dict(percentages=list(range(10, 100, 10)), n_rep=3, sr_tol=0.5, sr_small=1.0, err_small=5.0, rank_cache=None)


def sweep_settings(cfg, selector):
    raw = dict(cfg.wap_sweep or {})
    allowed = {**DEFAULTS, **selector.defaults}
    bad = set(raw) - set(allowed)
    if bad:
        raise ValueError(f"Khóa wap_sweep không hợp lệ cho {selector.display_name}: {sorted(bad)}; hợp lệ: {sorted(allowed)}")
    s = {**allowed, **raw}
    pcts = sorted(set(s["percentages"]))
    if not pcts or not all(0 < p < 100 for p in pcts):
        raise ValueError(f"wap_sweep.percentages phải nằm trong khoảng (0, 100); nhận {s['percentages']}")
    s["percentages"] = pcts
    return s


def prepare(train, test, cfg):
    """Biểu diễn bằng đúng bộ tiền xử lý đã chọn (fit trên TOÀN BỘ train), AP vắng -> 0 như runner."""
    from ips.data.preprocessors.factory import create_preprocessor
    if cfg.max_train: train = train.sample(n=min(cfg.max_train, len(train)), random_state=cfg.seed).reset_index(drop=True)
    if cfg.max_test: test = test.sample(n=min(cfg.max_test, len(test)), random_state=cfg.seed).reset_index(drop=True)
    pre = create_preprocessor(cfg.preprocess, **cfg.preprocess_params).fit(train)
    cols = list(pre.feature_columns)
    Xtr, Xva = (Formulas.missing_to_zero(pre.transform(f)) for f in (train, test))
    lab = lambda f: (f.BUILDINGID.to_numpy(int), f.FLOOR.to_numpy(int), f.LONGITUDE.to_numpy(float), f.LATITUDE.to_numpy(float))
    return Xtr, Xva, cols, lab(train), lab(test)


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
    def __init__(self, Xtr, Xva, ref, truth, models, s, k_max, rname):
        self.Xtr, self.Xva, self.s, self.k_max, self.rname = Xtr, Xva, s, k_max, rname
        self.tb, self.tf, self.tx, self.ty = ref
        self.truth, self.keys = truth, list(models)
        self.nb = int(max(self.tb.max(), truth[0].max())) + 1
        self.nf = int(max(self.tf.max(), truth[1].max())) + 1
        self.models = {m: build_model(m).fit(self.tf, self.tx, self.ty, self.nf) for m in self.keys}
        self.name = {m: self.models[m].display_name for m in self.keys}
        self.color = {m: self.models[m].color for m in self.keys}
        self.n_wap = Xtr.shape[1]
        self.pcts = s["percentages"]                                            # m tính theo % số WAP
        self.m_list = sorted({max(1, round(p * self.n_wap / 100)) for p in self.pcts} - {self.n_wap} | {self.n_wap})
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
            print(f"[m-sweep] [{self.rname[rk]}] xong {len(self.m_list)} mức m")
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
                print(f"\n--- [m-sweep] Xếp hạng: {self.rname[rk]} | m = {m} ({100 * m / self.n_wap:.1f}% N_WAP) ---")
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
                line = f"[m-sweep] [{self.rname[rk]} | {self.name[me]}] baseline (m={self.n_wap}): {base_err:.3f} m. "
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
            self.verdict = (f"[m-sweep] CẤU HÌNH TỐT NHẤT: xếp hạng {self.rname[r.rk]}, {self.name[r.method]}, m = {int(r.m)} ({r.pct_wap:.0f}% N_WAP): "
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
        ls = dict(zip(self.rks, ["-", "--", ":"]))
        fig, axes = plt.subplots(3, 1, figsize=(10, 13), sharex=True)
        base_e = {me: float(self.get(self.base_rk, self.n_wap, me, self.k_fix).mean_error) for me in self.keys}
        base_sr = float(self.get(self.base_rk, self.n_wap, self.keys[0], self.k_fix).success_rate)
        P = self.all[self.all.m < self.n_wap]
        for rk in self.rks:
            for me in self.keys:
                d = P[(P.rk == rk) & (P.method == me)].sort_values("m")
                axes[0].plot(d.m, d.mean_error, "o" + ls[rk], color=self.color[me], ms=3, label=f"{self.name[me]} - {self.rname[rk]}")
                axes[1].plot(d.m, d.success_rate, "o" + ls[rk], color=self.color[me], ms=3)
            d = P[(P.rk == rk) & (P.method == self.keys[0])].sort_values("m")
            axes[2].plot(d.m, d.zero_val_pct, "o" + ls[rk], color="#8172B3", ms=3, label=self.rname[rk])
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

    def export(self, out, orders, extras=None):
        out.mkdir(parents=True, exist_ok=True)
        self.sweep.to_csv(out / "khao_sat_m_k.csv", index=False)
        self.all.to_csv(out / "tong_hop_theo_m.csv", index=False)
        self.sel.to_csv(out / "chon_m.csv", index=False)
        pd.concat([t.assign(xep_hang=self.rname[rk], m=m) for (rk, m), t in self.by_m.items()]).to_csv(out / "bang_tung_m.csv", index=False)
        pd.DataFrame({self.rname[rk]: o for rk, o in orders.items()}).rename_axis("hang").to_csv(out / "thu_tu_wap.csv")
        extra_files = []
        for name, obj in (extras or {}).items():
            if isinstance(obj, pd.DataFrame): obj.to_csv(out / name, index=False)
            else: (out / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
            extra_files.append(name)
        (out / "summary.json").write_text(json.dumps(dict(models=self.keys, percentages=self.pcts, m_list=self.m_list, k_fix=self.k_fix, n_wap=self.n_wap,
                                                          n_eval=int(len(self.Xva)), bonferroni=self.alpha, verdict=self.verdict),
                                                     ensure_ascii=False, indent=2), encoding="utf-8")
        return ["khao_sat_m_k.csv", "tong_hop_theo_m.csv", "chon_m.csv", "bang_tung_m.csv", "thu_tu_wap.csv", "summary.json"] + extra_files + self.plot(out)


def run_selector(selector, train, test, cfg, base_names, out_dir, prepared=None, rank=None):
    """Chạy trọn luồng cho một bộ chọn WAP và xuất file vào out_dir.

    `prepared` và `rank` cho phép chế độ chọn cả hai HUIM tái dùng dữ liệu/kết quả khai phá
    thay vì chuẩn bị lại hai lần.
    """
    s = sweep_settings(cfg, selector)
    if prepared is None:
        prepared = prepare(train, test, cfg)
    Xtr, Xva, cols, ref, truth = prepared
    print(f"\n[m-sweep] {selector.display_name}: models={base_names}, RSS={cfg.preprocess}, N_WAP={Xtr.shape[1]}, "
          f"train={len(Xtr)}, eval={len(Xva)}, tỉ lệ WAP giữ lại = {s['percentages']}%")
    if rank is None:
        rank = selector.rank(Xtr, ref, cols, s, cfg.seed)
    sw = WapSweep(Xtr, Xva, ref, truth, base_names, s, cfg.k_max, rank.labels)
    t0 = time.perf_counter(); sw.run(rank.orders, rank.times); sw.t_run_s = time.perf_counter() - t0
    print(f"[m-sweep] Các mức m: {sw.m_list} | đã chạy {len(sw.sweep)} lần đánh giá | k chung K_FIX = {sw.k_fix}")
    sw.tables(); sw.summarize(); sw.select()
    t0 = time.perf_counter(); files = sw.export(Path(out_dir), rank.orders, rank.extras); sw.t_export_s = time.perf_counter() - t0
    print(f"[m-sweep] Đã lưu vào {out_dir}: " + ", ".join(files))
    return sw


def _best_selection(sw):
    """Dòng chọn m tốt nhất của một bộ chọn, dùng cho bảng so sánh HUIM."""
    sel = sw.sel
    cand = sel[sel.best_err.notna()] if "best_err" in sel else sel.iloc[0:0]
    if not len(cand):
        return {}
    r = cand.sort_values("best_err").iloc[0]
    return dict(best_method=r.method, best_m=int(r.m_best) if pd.notna(r.m_best) else None,
                best_mean_error=float(r.best_err), baseline_error=float(r.baseline_err),
                best_d_err_pct=float(r.best_d_err_pct) if pd.notna(r.best_d_err_pct) else np.nan,
                best_improved=bool(r.best_improved) if pd.notna(r.best_improved) else False)


def export_huim_comparison(out, joint, sweeps):
    """Ghi thư mục so sánh hai thuật toán HUIM, gồm bảng tổng hợp và so sánh thời gian."""
    from ips.neighbors.huim import ALGORITHMS, DEFAULT_ORDER
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    algorithms = [alg for alg in DEFAULT_ORDER if alg in sweeps]
    labels = {alg: ALGORITHMS[alg]["label"] for alg in algorithms}

    timing_rows = []
    for alg in algorithms:
        res, sw = joint["results"][alg], sweeps[alg]
        sweep_s, export_s = float(getattr(sw, "t_run_s", 0.0)), float(getattr(sw, "t_export_s", 0.0))
        timing_rows.append(dict(
            thuat_toan=labels[alg], thoi_gian_khai_pha_s=res["time"],
            thoi_gian_wall_lan_cuoi_s=joint["walls"][alg],
            tong_thoi_gian_xep_hang_s=joint["cumulative_walls"][alg],
            thoi_gian_quet_m_s=sweep_s, thoi_gian_xuat_file_s=export_s,
            tong_thoi_gian_s=joint["cumulative_walls"][alg] + sweep_s + export_s,
            bo_nho_mb=res["mem"], so_itemset=res["n"], so_lan_thu=joint["n_attempts"],
            min_util=joint["min_util"], ty_le_min_util=joint["min_util_ratio"],
        ))
    timing = pd.DataFrame(timing_rows)
    timing.to_csv(out / "so_sanh_thoi_gian.csv", index=False)
    joint["attempts"].to_csv(out / "so_sanh_min_util.csv", index=False)

    chosen = []
    for alg in algorithms:
        sel = sweeps[alg].sel.copy(); sel.insert(0, "thuat_toan", labels[alg]); chosen.append(sel)
    pd.concat(chosen, ignore_index=True).to_csv(out / "so_sanh_chon_m.csv", index=False)

    summary_rows = []
    for alg in algorithms:
        res, sw, best = joint["results"][alg], sweeps[alg], _best_selection(sweeps[alg])
        row = dict(thuat_toan=labels[alg], min_util=joint["min_util"], ty_le_min_util=joint["min_util_ratio"],
                   n_transactions=joint["n_transactions"], so_itemset=res["n"], same_itemsets=joint["same_itemsets"],
                   same_order=joint["same_order"], bo_nho_mb=res["mem"], so_lan_thu=joint["n_attempts"],
                   k_fix=sw.k_fix, ket_luan=sw.verdict, thu_muc=ALGORITHMS[alg]["out_dir"])
        row.update(timing[timing.thuat_toan == labels[alg]].iloc[0].to_dict())
        row.update(best)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "so_sanh_tong_hop.csv", index=False)

    fast = min(algorithms, key=lambda alg: float(joint["results"][alg]["time"]))
    slow = max(algorithms, key=lambda alg: float(joint["results"][alg]["time"]))
    tf, ts = float(joint["results"][fast]["time"]), float(joint["results"][slow]["time"])
    speed = ts / max(tf, 1e-9)
    same_text = ("Hai thuật toán cho cùng tập itemset và cùng thứ tự xếp hạng."
                 if joint["same_itemsets"] and joint["same_order"] else
                 "Hai thuật toán KHÁC tập itemset hoặc khác thứ tự xếp hạng; mỗi thư mục dùng kết quả riêng.")
    info = dict(
        min_util=joint["min_util"], ty_le_min_util=joint["min_util_ratio"],
        n_transactions=joint["n_transactions"], n_attempts=joint["n_attempts"],
        cung_min_util=True, same_itemsets=joint["same_itemsets"], same_order=joint["same_order"],
        thuat_toan={alg: dict(nhan=labels[alg], thu_muc=ALGORITHMS[alg]["out_dir"],
                              thoi_gian_khai_pha_s=float(joint["results"][alg]["time"]),
                              thoi_gian_wall_lan_cuoi_s=float(joint["walls"][alg]),
                              tong_thoi_gian_xep_hang_s=float(joint["cumulative_walls"][alg]),
                              thoi_gian_quet_m_s=float(getattr(sweeps[alg], "t_run_s", 0.0)),
                              thoi_gian_xuat_file_s=float(getattr(sweeps[alg], "t_export_s", 0.0)),
                              bo_nho_mb=float(joint["results"][alg]["mem"]),
                              so_itemset=int(joint["results"][alg]["n"]))
                        for alg in algorithms},
        ket_luan=f"{labels[fast]} nhanh hơn {labels[slow]} {speed:.2f} lần khi khai phá. {same_text}",
    )
    (out / "so_sanh_thong_tin.json").write_text(json.dumps(info, ensure_ascii=False, indent=2, default=float),
                                                encoding="utf-8")
    files = ["so_sanh_tong_hop.csv", "so_sanh_thoi_gian.csv", "so_sanh_min_util.csv", "so_sanh_chon_m.csv",
             "so_sanh_thong_tin.json"]
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
        names = timing.thuat_toan.tolist()
        colors = [ALGORITHMS[alg]["color"] for alg in algorithms]
        axes[0].bar(names, timing.thoi_gian_khai_pha_s, color=colors)
        axes[0].set(title="Thời gian khai phá PAMI", ylabel="Giây")
        axes[1].bar(names, timing.bo_nho_mb, color=colors)
        axes[1].set(title="RAM đỉnh", ylabel="MB")
        bottom = np.zeros(len(timing))
        for col, label in (("tong_thoi_gian_xep_hang_s", "Xếp hạng"), ("thoi_gian_quet_m_s", "Quét m"),
                           ("thoi_gian_xuat_file_s", "Xuất file")):
            vals = timing[col].fillna(0).to_numpy(dtype=float)
            axes[2].bar(names, vals, bottom=bottom, label=label); bottom += vals
        axes[2].set(title="Tổng thời gian", ylabel="Giây"); axes[2].legend(fontsize=8)
        for ax in axes: ax.grid(alpha=.3, axis="y")
        fig.tight_layout(); fig.savefig(out / "so_sanh_thoi_gian.png", dpi=150); plt.close(fig)
        files.append("so_sanh_thoi_gian.png")
    except ImportError:
        print("[m-sweep] Thiếu matplotlib nên bỏ qua biểu đồ so sánh HUIM.")
    return files


def run_huim_pair(selectors, train, test, cfg, base_names, out_dir):
    """Chạy cả UP-Growth và EFIM ở cùng min_util, rồi xuất hai thư mục và thư mục so sánh."""
    from ips.neighbors.huim import DEFAULT_ORDER, COMPARE_DIR, rank_joint
    selectors = sorted(selectors, key=lambda x: DEFAULT_ORDER.index(x.algorithm))
    s = sweep_settings(cfg, selectors[0])
    prepared = prepare(train, test, cfg)
    Xtr, Xva, cols, ref, truth = prepared
    print(f"\n[m-sweep] HUIM cặp: chạy {', '.join(sel.display_name for sel in selectors)} ở cùng min_util")
    joint = rank_joint(Xtr, cols, s, cfg.seed)
    sweeps = {}
    for selector in selectors:
        sweeps[selector.algorithm] = run_selector(
            selector, train, test, cfg, base_names, Path(out_dir) / selector.out_dir,
            prepared=prepared, rank=joint["ranks"][selector.algorithm])
    files = export_huim_comparison(Path(out_dir) / COMPARE_DIR, joint, sweeps)
    print(f"[m-sweep] Đã lưu so sánh HUIM vào {Path(out_dir) / COMPARE_DIR}: " + ", ".join(files))
    return sweeps
