"""SO SÁNH các model họ kNN: bảng, Wilcoxon cặp, kết luận tự động, biểu đồ, xuất file."""
import json
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


class Comparator:
    def __init__(self, exp, k_ref=13):
        self.exp, self.k_ref = exp, k_ref
        self.keys, self.name, self.color = exp.keys, exp.names, exp.colors
        self.pairs = list(combinations(self.keys, 2))
        self.alpha = 0.05 / max(len(self.pairs), 1)                       # Bonferroni
        self.sweep = exp.sweep(self.keys)
        self.best = {m: exp.best_k(m) for m in self.keys}                 # chọn trên validation
        self.k_focus = self.best[self.keys[-1]]
        self.k_ref = min(self.k_ref, exp.k_max)
        self.piv = self.sweep.pivot(index="k", columns="method", values=["test_mean", "val_mean", "total_s"])
        self.tag = lambda a, b: f"{self.name[a]} − {self.name[b]}"
        self.table = self.d1 = self.d2 = None

    def _row(self, m, k):
        return self.sweep[(self.sweep.method == m) & (self.sweep.k == k)].iloc[0]

    # ---------- bảng ----------
    def tables(self):
        N = self.name
        for title, col, nd in (("B1. Độ chính xác tầng (%) theo k — test", "floor_acc", 2),
                               ("B2. Sai số trung bình (m) theo k — validation nội bộ (dùng để chọn k)", "val_mean", 3),
                               ("B3. Sai số trung bình (m) theo k — official test", "test_mean", 3),
                               (f"B4. Tổng thời gian (s) theo k (tìm láng giềng {self.exp.t_search:.2f}s dùng chung)", "total_s", 4)):
            t = self.sweep.pivot(index="k", columns="method", values=col)[self.keys].rename(columns=N)
            print(f"\n--- {title} ---"); print(t.round(nd).to_string())
        cols = ["val_mean", "test_mean", "test_median", "test_rmse", "test_p90", "test_le1", "test_le3", "test_le5",
                "floor_acc", "t_predict_s", "total_s", "us_per_sample", "n_query_d0"]
        plan = [(m, self.best[m], "k chọn trên validation") for m in self.keys]
        plan += [(m, self.k_focus, f"cùng k={self.k_focus}") for m in self.keys]
        plan += [(m, self.k_ref, f"k={self.k_ref} (tham chiếu)") for m in self.keys]
        seen, out = set(), []
        for m, k, tag in plan:
            if (m, k) in seen: continue
            seen.add((m, k))
            out.append({"Phương pháp": self.name[m], "k": k, "Ghi chú": tag, **self._row(m, k)[cols].to_dict()})
        self.table = pd.DataFrame(out)
        print("\n--- B5. Bảng tổng hợp tại các k quan trọng ---")
        with pd.option_context("display.width", 250, "display.max_columns", None):
            print(self.table.drop(columns=["Ghi chú"]).round(3).to_string(index=False))

    # ---------- so sánh + kiểm định ----------
    def _d1(self):
        self.d1 = pd.DataFrame({self.tag(a, b): self.piv["test_mean"][a] - self.piv["test_mean"][b] for a, b in self.pairs})

    def pairwise(self):
        N = self.name
        self._d1()
        print("\n--- D1. Hiệu sai số test trung bình (m) ở cùng k (dương = phương pháp thứ hai tốt hơn) ---")
        print(self.d1.round(4).to_string())
        n = len(self.d1) - 1
        for a, b in self.pairs:
            v = self.d1[self.tag(a, b)].iloc[1:]
            print(f"{self.tag(a, b)} (k=2..{self.exp.k_max}): {N[b]} tốt hơn ở {int((v > 1e-9).sum())}/{n} k, "
                  f"kém hơn ở {int((v < -1e-9).sum())}/{n}; chênh trung bình {v.mean():+.3f} m")
        res = []
        for K in sorted({self.best[m] for m in self.keys} | {self.k_ref}):
            print(f"\n=== D2. Wilcoxon cặp trên {self.exp.n_test} mẫu test tại k = {K} (ngưỡng {self.alpha:.4f}) ===")
            for a, b in self.pairs:
                ea, eb = self.exp.run(a, K)["test_err"], self.exp.run(b, K)["test_err"]
                dl = ea - eb
                try: pv = None if np.allclose(dl, 0) else float(wilcoxon(ea, eb).pvalue)
                except ValueError: pv = None
                if pv is not None and np.isnan(pv): pv = None
                kq = "không tính được" if pv is None else ("CÓ ý nghĩa thống kê" if pv < self.alpha else "KHÔNG có ý nghĩa thống kê")
                print(f"{self.tag(a, b)}: {N[b]} nhỏ hơn ở {100*(dl > 1e-9).mean():.1f}% mẫu | lớn hơn {100*(dl < -1e-9).mean():.1f}% | "
                      f"mean {ea.mean():.3f} vs {eb.mean():.3f} (chênh {dl.mean():+.3f} m); p = {'NA' if pv is None else f'{pv:.3g}'} → {kq}")
                res.append(dict(k=K, a=a, b=b, cap=self.tag(a, b), mean_a=ea.mean(), mean_b=eb.mean(), diff_a_minus_b=dl.mean(),
                                pct_b_better=100*(dl > 1e-9).mean(), pct_a_better=100*(dl < -1e-9).mean(),
                                pct_tie=100*(abs(dl) <= 1e-9).mean(), p_value=pv, significant=(pv is not None and pv < self.alpha)))
        self.d2 = pd.DataFrame(res)

    def conclusion(self):
        if self.d2 is None: self.pairwise()
        N, K0 = self.name, self.k_focus
        print("\n" + "=" * 78)
        print("1) TẦNG (vote giống hệt nhau): " + " | ".join(f"{N[m]} (k={self.best[m]}): {self._row(m, self.best[m]).floor_acc:.2f}%" for m in self.keys))
        print("2) SAI SỐ VỊ TRÍ trên official test (k chọn trên validation nội bộ)")
        print("   " + " | ".join(f"{N[m]} k={self.best[m]}: {self._row(m, self.best[m]).test_mean:.3f} m" for m in self.keys))
        print(f"   Cùng k = {K0}: " + " | ".join(f"{N[m]} {self._row(m, K0).test_mean:.3f} m" for m in self.keys))
        for a, b in self.pairs:
            r = self.d2[(self.d2.a == a) & (self.d2.b == b) & (self.d2.k == K0)]
            if r.empty: continue
            r = r.iloc[0]; diff = r.diff_a_minus_b
            v = (f"{N[b]} {'TỐT' if diff > 0 else 'KÉM'} HƠN có ý nghĩa thống kê" if r.significant
                 else "KHÁC BIỆT KHÔNG có ý nghĩa thống kê (gần như tương đương)")
            print(f"   → {self.tag(a, b)} = {diff:+.3f} m ({100*diff/r.mean_a:+.1f}%): {v}.")
        print("3) THỜI GIAN")
        print("   Tổng: " + " | ".join(f"{N[m]} {self._row(m, self.best[m]).total_s:.2f}s" for m in self.keys))
        print("   Riêng gộp tọa độ: " + " | ".join(f"{N[m]} {self._row(m, self.best[m]).t_predict_s*1000:.1f}ms" for m in self.keys))
        big = max(self._row(m, self.best[m]).total_s for m in self.keys) / min(self._row(m, self.best[m]).total_s for m in self.keys) - 1
        print("   → Tổng thời gian chênh dưới 5%." if big < 0.05 else f"   → Tổng thời gian chênh lớn nhất: {100*big:.1f}%.")
        print("=" * 78)
        print("Lưu ý: giả định đã biết tòa nhà; k chọn trên validation nội bộ, official test chỉ dùng để báo cáo.")

    def run_all(self):
        self.tables(); self.pairwise(); self.conclusion()

    # ---------- biểu đồ ----------
    def plots(self, outdir):
        try:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[plots] Thiếu matplotlib nên bỏ qua biểu đồ (pip install matplotlib để vẽ).")
            return []
        outdir.mkdir(parents=True, exist_ok=True); saved, K = [], self.exp.k_max

        def line(col, title, ylabel, fname, mark_best=False):
            fig, ax = plt.subplots(figsize=(8, 4.5))
            for m in self.keys:
                s = self.sweep[self.sweep.method == m]
                ax.plot(s.k, s[col], "o-", color=self.color[m], label=self.name[m], ms=4)
                if mark_best: ax.axvline(self.best[m], color=self.color[m], ls="--", alpha=.5, label=f"k chọn (validation) {self.name[m]} = {self.best[m]}")
            ax.set(title=title, xlabel="k (số láng giềng)", ylabel=ylabel, xticks=range(1, K + 1)); ax.grid(alpha=.3); ax.legend()
            fig.tight_layout(); fig.savefig(outdir / fname, dpi=150); plt.close(fig); saved.append(fname)

        line("floor_acc", "Độ chính xác tầng (test) theo k", "Floor accuracy (%)", "C1_floor_accuracy.png")
        line("test_mean", "Sai số vị trí trung bình (official test) theo k", "Mean error (m)", "C2_test_mean_error.png", True)
        line("val_mean", "Sai số vị trí trung bình (validation nội bộ) theo k", "Mean error (m)", "C3_val_mean_error.png", True)
        line("total_s", "Tổng thời gian chạy theo k", "Thời gian (s)", "C5_time.png")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for m in self.keys:
            e = np.sort(self.exp.run(m, self.best[m])["test_err"])
            ax.plot(e, np.arange(1, len(e) + 1) / len(e), color=self.color[m], label=f"{self.name[m]} (k={self.best[m]})")
        ax.set(xlim=(0, 40), xlabel="Sai số vị trí (m)", ylabel="Tỉ lệ mẫu có sai số ≤ x", title="CDF sai số (official test, mọi mẫu)")
        ax.grid(alpha=.3); ax.legend(); fig.tight_layout(); fig.savefig(outdir / "C4_cdf.png", dpi=150); plt.close(fig); saved.append("C4_cdf.png")
        if self.d1 is None: self._d1()
        if self.pairs:
            fig, ax = plt.subplots(figsize=(10, 4.8)); w = 0.8 / len(self.pairs)
            for j, (a, b) in enumerate(self.pairs):
                ax.bar(self.d1.index + (j - (len(self.pairs) - 1) / 2) * w, self.d1[self.tag(a, b)], width=w, label=self.tag(a, b))
            ax.axhline(0, color="k", lw=.8)
            ax.set(xlabel="k", ylabel="Chênh lệch sai số TB (m)", xticks=range(1, K + 1), title="Chênh lệch sai số test giữa các cặp (dương = phương pháp thứ hai tốt hơn)")
            ax.grid(alpha=.3, axis="y"); ax.legend(); fig.tight_layout(); fig.savefig(outdir / "C6_chenh_lech.png", dpi=150); plt.close(fig); saved.append("C6_chenh_lech.png")
        return saved

    # ---------- xuất file (bố cục giống runner của repo) ----------
    def export(self, outdir):
        if self.table is None: self.tables()
        if self.d2 is None: self.pairwise()
        outdir.mkdir(parents=True, exist_ok=True)
        self.sweep.to_csv(outdir / "sweep_by_k.csv", index=False)
        self.table.to_csv(outdir / "summary_table.csv", index=False)
        self.d1.to_csv(outdir / "pairwise_same_k.csv")
        self.d2.to_csv(outdir / "wilcoxon.csv", index=False)
        by_b, summary = [], {"selected_k": self.best, "t_search_test_s": self.exp.t_search, "official_test": {}, "by_building": []}
        for m in self.keys:
            r = self.exp.run(m, self.best[m])
            summary["official_test"][m] = {"n": int(len(r["test_err"])), "mean_m": float(r["test_err"].mean()),
                                           "median_m": float(np.median(r["test_err"])), "p90_m": float(np.quantile(r["test_err"], .9)),
                                           "floor_acc": r["floor_acc"], "k": self.best[m]}
            for blk in self.exp.blocks:
                sel = r["test_building"] == blk.building
                row = dict(model=m, building=blk.building, k=self.best[m], n=int(sel.sum()), test_mean_m=float(r["test_err"][sel].mean()),
                           test_median_m=float(np.median(r["test_err"][sel])), test_p90_m=float(np.quantile(r["test_err"][sel], .9)),
                           floor_acc=float(r["floor_ok"][sel].mean() * 100))
                by_b.append(row); summary["by_building"].append(row)
        pd.DataFrame(by_b).to_csv(outdir / "by_building.csv", index=False)
        (outdir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        for blk in self.exp.blocks:                                     # building_N/{split,predictions}.csv
            folder = outdir / f"building_{blk.building}"; folder.mkdir(exist_ok=True)
            pd.DataFrame({"source_row": np.r_[blk.train_rows, blk.val_rows],
                          "split": ["train"] * len(blk.train_rows) + ["internal_val"] * len(blk.val_rows)}).to_csv(folder / "split.csv", index=False)
            df = pd.DataFrame({"source_row": blk.test_rows, "building": blk.building, "true_x": blk.test_xy[:, 0],
                               "true_y": blk.test_xy[:, 1], "true_floor": blk.test_floor})
            for m in self.keys:
                r = self.exp.run(m, self.best[m]); i = self.exp.blocks.index(blk)
                p = r["preds"][i]
                df[f"{m}_pred_x"], df[f"{m}_pred_y"], df[f"{m}_pred_floor"] = p.x, p.y, p.floor
                df[f"{m}_error_m"] = np.hypot(p.x - blk.test_xy[:, 0], p.y - blk.test_xy[:, 1])
            df.to_csv(folder / "predictions.csv", index=False)
        return ["sweep_by_k.csv", "summary_table.csv", "pairwise_same_k.csv", "wilcoxon.csv", "by_building.csv", "summary.json",
                "building_N/split.csv", "building_N/predictions.csv"] + self.plots(outdir)
