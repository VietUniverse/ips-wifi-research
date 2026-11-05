"""Menu terminal: chọn model để chạy hoặc so sánh tất cả."""
from pathlib import Path
from .knn_report import Comparator


def _ask_k(k_max):
    s = input(f"Nhập k (1-{k_max}) hoặc Enter để quét k = 1..{k_max} (chọn k theo validation): ").strip()
    if s.isdigit() and 1 <= int(s) <= k_max:
        return int(s)
    if s:
        print("k không hợp lệ -> quét toàn bộ.")
    return None


def _run_one(exp, key):
    name, k = exp.names[key], _ask_k(exp.k_max)
    if k is None:
        df, k = exp.sweep([key]), exp.best_k(key)
        print(f"\n{name} - quét k = 1..{exp.k_max}")
        print(df[["k", "val_mean", "test_mean", "test_median", "test_p90", "floor_acc", "t_predict_s", "total_s"]].round(3).to_string(index=False))
        print(f"→ k chọn trên validation = {k}")
    r = exp.run(key, k)
    print(f"\n{name}, k = {k}")
    for lab, col, fmt in [("Val mean error (m)", "val_mean", ".3f"), ("Test mean error (m)", "test_mean", ".3f"), ("Test median (m)", "test_median", ".3f"),
                          ("Test RMSE (m)", "test_rmse", ".3f"), ("Test P90 (m)", "test_p90", ".3f"), ("≤1 m (%)", "test_le1", ".2f"),
                          ("≤3 m (%)", "test_le3", ".2f"), ("≤5 m (%)", "test_le5", ".2f"), ("Độ chính xác tầng (%)", "floor_acc", ".2f"),
                          ("Thời gian gộp tọa độ (s)", "t_predict_s", ".4f"), ("Tổng thời gian (s)", "total_s", ".3f")]:
        print(f"  {lab:<28}{r[col]:{fmt}}")


def run_menu(exp, outdir, k_ref=13):
    keys, cmp_ = exp.keys, None
    n = len(keys)
    while True:
        print("\n" + "=" * 52 + "\n  ĐỊNH VỊ TRONG NHÀ — họ kNN\n" + "=" * 52)
        for i, k in enumerate(keys, 1):
            print(f"  {i}. Chạy {exp.names[k]}")
        print(f"  {n + 1}. SO SÁNH tất cả (bảng + Wilcoxon + kết luận)\n  {n + 2}. Xuất biểu đồ & CSV vào {outdir}\n  0. Thoát")
        c = input("Chọn: ").strip()
        if c == "0": break
        elif c.isdigit() and 1 <= int(c) <= n: _run_one(exp, keys[int(c) - 1])
        elif c == str(n + 1):
            cmp_ = cmp_ or Comparator(exp, k_ref); cmp_.run_all()
        elif c == str(n + 2):
            cmp_ = cmp_ or Comparator(exp, k_ref)
            out = Path(outdir)
            if out.exists() and any(out.iterdir()): print(f"Thư mục {out} đã có file; chọn --output-dir mới."); continue
            print("Đã lưu:", ", ".join(cmp_.export(out)))
        else: print("Lựa chọn không hợp lệ.")
