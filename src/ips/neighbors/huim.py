"""Mục "huim" trong danh sách model: chọn trước tập con WAP bằng khai phá tập mục lợi ích cao (HUIM) rồi mới đưa vào kNN/WKNN.

Đây KHÔNG phải model định vị. Chọn mục này thì runner chạy như thường (kNN, WKNN 1/d, WKNN 1/d²) rồi chạy thêm luồng
quét tỉ lệ WAP trong ips.experiments.knn_wap_sweep với xếp hạng WAP của file này. Không chọn thì HUIM không chạy.

Ánh xạ (như Colab Lần 8): giao dịch = một mẫu train; item = WAP bắt được; số lượng = giá trị powed x scale (làm tròn);
lợi nhuận đơn vị = unit_profit. Chạy CẢ UP-Growth và EFIM của thư viện PAMI ở cùng min_util; hai thuật toán chính xác nên
cho cùng tập itemset, chỉ khác thời gian và bộ nhớ (được so sánh trong huim_thoi_gian.csv). Điểm của WAP = tổng utility của
các itemset chứa nó; WAP không thuộc itemset nào xếp tiếp theo utility riêng lẻ. Mỗi thuật toán chạy trong tiến trình con
có giới hạn RAM/thời gian; nếu tràn thì min_util tự tăng và thử lại. Cần thư viện `pami` (tự pip install nếu chưa có)."""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import numpy as np

MODEL_NAME = "huim"

WORKER_SRC = '''
import sys, json, time, zlib, heapq
alg, db, minutil, out, n_wap, mem_limit = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
try:
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))
except Exception:
    pass
from PAMI.highUtilityPattern.basic import UPGrowth, EFIM
cls = EFIM.EFIM if alg == 'efim' else UPGrowth.UPGrowth
obj = cls(db, minutil, chr(9))
t0 = time.perf_counter(); obj.mine(); t = time.perf_counter() - t0
pats = obj.getPatterns()
items_iter = pats.items() if isinstance(pats, dict) else pats
score, cnt, lens, chk, top, n = [0.0] * n_wap, [0] * n_wap, {}, 0, [], 0
for pat, util in items_iter:
    its = sorted(int(x) for x in (pat.split() if isinstance(pat, str) else pat)); u = float(util); n += 1
    for i in its:
        score[i - 1] += u; cnt[i - 1] += 1
    lens[len(its)] = lens.get(len(its), 0) + 1
    chk = (chk + zlib.crc32((' '.join(map(str, its)) + '|' + str(int(round(u)))).encode())) % (1 << 64)
    if len(top) < 10: heapq.heappush(top, (u, its))
    else: heapq.heappushpop(top, (u, its))
try: mem = float(obj.getMemoryRSS()) / 1e6
except Exception: mem = float('nan')
json.dump(dict(time=t, mem=mem, n=n, checksum=chk, score=score, cnt=cnt, lens=lens, top=sorted(top, reverse=True)), open(out, 'w'))
'''


def ensure_pami():
    try:
        import PAMI.highUtilityPattern.basic  # noqa: F401
    except ImportError:
        print("[m-sweep] Chưa có thư viện pami, đang cài (pip install pami)...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pami"], check=True)
        import importlib; importlib.invalidate_caches()
        import PAMI.highUtilityPattern.basic  # noqa: F401


def available_ram():
    try:
        import psutil
        return psutil.virtual_memory().available
    except Exception:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")


def write_db(path, U):
    """Định dạng PAMI: items:tổng utility giao dịch:utility từng item (ngăn cách bằng tab)."""
    n = 0
    with open(path, "w") as f:
        for row in U:
            it = np.flatnonzero(row > 0)
            if len(it) == 0: continue
            u = row[it]
            f.write("\t".join(map(str, it + 1)) + ":" + str(int(u.sum())) + ":" + "\t".join(map(str, u)) + "\n"); n += 1
    return n


def run_guarded(workdir, alg, db, min_util, n_wap, mem_limit, time_limit):
    out = Path(workdir) / f"huim_{alg}.json"
    if out.exists(): out.unlink()
    cmd = [sys.executable, str(Path(workdir) / "huim_worker.py"), alg, str(db), str(min_util), str(out), str(n_wap), str(mem_limit)]
    t0 = time.perf_counter()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=time_limit)
    except subprocess.TimeoutExpired:
        return None, f"quá {time_limit}s", time.perf_counter() - t0
    wall = time.perf_counter() - t0
    if r.returncode != 0 or not out.exists():
        tail = (r.stderr or "").strip().splitlines()[-1:] or [""]
        return None, f"tràn RAM/lỗi (mã {r.returncode}) {tail[0][:80]}", wall
    return json.loads(out.read_text()), "OK", wall


class HUIMSelector:
    is_wap_selector = True
    display_name = "HUIM (UP-Growth + EFIM, chọn WAP)"
    color = "#8172B3"
    out_dir = "wap_huim"
    defaults = dict(scale=1000, unit_profit=1, min_util_ratio=0.005, escalate=1.5, max_tries=8, n_tx=5000,
                    time_limit_s=600, mem_fraction=0.7)

    def rank(self, Xtr, ref, cols, s, seed):
        from ips.experiments.knn_wap_sweep import RankResult
        import pandas as pd
        ensure_pami()
        n_wap = Xtr.shape[1]
        rng = np.random.RandomState(seed)
        idx = np.arange(len(Xtr)) if not s["n_tx"] else np.sort(rng.choice(len(Xtr), min(s["n_tx"], len(Xtr)), replace=False))
        U = np.rint(Xtr[idx] * s["scale"]).astype(np.int64) * s["unit_profit"]       # utility từng WAP trong từng mẫu
        total, single = int(U.sum()), U.sum(0).astype(float)
        mem_limit = int(s["mem_fraction"] * available_ram())
        print(f"[m-sweep] HUIM: {len(U)} giao dịch, {int((single > 0).sum())}/{n_wap} WAP có mặt, tổng utility = {total:,}, "
              f"giới hạn RAM mỗi lần khai phá = {mem_limit / 1e9:.1f} GB")
        with tempfile.TemporaryDirectory() as work:
            (Path(work) / "huim_worker.py").write_text(WORKER_SRC)
            db = Path(work) / "huim_db.txt"
            print(f"[m-sweep] Đã ghi {write_db(db, U)} giao dịch")
            attempts, res_e, res_u = [], None, None
            for i in range(s["max_tries"]):
                ratio = s["min_util_ratio"] * s["escalate"] ** i; mu = int(ratio * total)
                print(f"[m-sweep] HUIM lần {i + 1}/{s['max_tries']}: min_util = {ratio:.2%} tổng = {mu:,}")
                re_, why_e, w_e = run_guarded(work, "efim", db, mu, n_wap, mem_limit, s["time_limit_s"])
                ru_, why_u, w_u = (None, "bỏ qua (EFIM đã lỗi)", 0.0) if re_ is None else run_guarded(work, "upgrowth", db, mu, n_wap, mem_limit, s["time_limit_s"])
                print(f"[m-sweep]    EFIM: {why_e} ({w_e:.1f}s) | UP-Growth: {why_u} ({w_u:.1f}s)")
                attempts.append(dict(lan=i + 1, ty_le_min_util=ratio, min_util=mu, EFIM=why_e, UP_Growth=why_u, thoi_gian_EFIM_s=w_e, thoi_gian_UPG_s=w_u))
                if re_ is not None and ru_ is not None:
                    res_e, res_u = re_, ru_; break
        if res_e is None:
            raise RuntimeError("Cả EFIM lẫn UP-Growth đều thất bại ở mọi min_util đã thử: giảm n_tx (vd 2000) hoặc tăng min_util_ratio/max_tries trong khóa wap_sweep.")
        same = res_e["n"] == res_u["n"] and res_e["checksum"] == res_u["checksum"]
        t_e, t_u = res_e["time"], res_u["time"]
        score, cnt = np.array(res_e["score"]), np.array(res_e["cnt"], dtype=float)
        print(f"[m-sweep] HUIM: min_util cuối = {ratio:.2%} tổng = {mu:,} | {res_e['n']} itemset | UP-Growth và EFIM cho cùng tập itemset: {same}"
              + ("" if same else " (KHÁC NHAU: xếp hạng dùng kết quả EFIM)"))
        print(f"[m-sweep] HUIM thời gian khai phá: EFIM {t_e:.2f}s, UP-Growth {t_u:.2f}s (EFIM nhanh hơn {t_u / max(t_e, 1e-9):.1f} lần)")
        if res_e["n"] == 0:
            print("[m-sweep] Không tìm được itemset nào ở min_util này: xếp hạng chỉ dựa trên utility riêng lẻ của từng WAP.")
        order = np.lexsort((-single, -score))            # điểm giảm dần; hòa thì theo utility riêng lẻ
        score_n = score / score.sum() if score.sum() > 0 else score
        table = pd.DataFrame({"hang": np.arange(1, n_wap + 1), "WAP": np.array(cols)[order], "diem": score_n[order],
                              "so_itemset_chua_WAP": cnt[order].astype(int), "utility_rieng_le": single[order]})
        timing = pd.DataFrame([dict(thuat_toan="EFIM", thoi_gian_s=t_e, bo_nho_mb=res_e["mem"], so_itemset=res_e["n"]),
                               dict(thuat_toan="UP-Growth", thoi_gian_s=t_u, bo_nho_mb=res_u["mem"], so_itemset=res_u["n"])])
        top = pd.DataFrame({"itemset (top 10 theo utility)": [" + ".join(cols[j - 1] for j in its) for _, its in res_e["top"]],
                            "utility": [u for u, _ in res_e["top"]]})
        info = dict(min_util_ratio_used=ratio, min_util=mu, n_transactions=int(len(U)), same_itemsets=bool(same),
                    t_efim_s=t_e, t_upgrowth_s=t_u, n_itemset=int(res_e["n"]), itemset_length_counts=res_e["lens"])
        return RankResult({"huim": order}, {"huim": t_e}, {"huim": "HUIM"},
                          {"huim_thoi_gian.csv": timing, "huim_nhat_ky.csv": pd.DataFrame(attempts), "huim_top_itemset.csv": top,
                           "huim_diem_wap.csv": table, "huim_thong_tin.json": info})


def build_model(**params):
    return HUIMSelector()
