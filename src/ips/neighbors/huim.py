"""Engine dùng chung cho hai mục chọn WAP bằng khai phá tập mục lợi ích cao (HUIM).

File này KHÔNG phải plugin: không có MODEL_NAME/build_model nên không xuất hiện trong
`available_models()`. Hai plugin mỏng `huim_upgrowth.py` và `huim_efim.py` đăng ký hai
mục riêng trong menu. Mỗi mục chỉ chạy đúng thuật toán của mình; chỉ khi người dùng
chọn cả hai, `rank_joint()` mới chạy cả hai ở cùng min_util để so sánh công bằng.

Ánh xạ (như Colab Lần 8): giao dịch = một mẫu train; item = WAP bắt được; số lượng =
giá trị powed x scale (làm tròn); lợi nhuận đơn vị = unit_profit. Điểm của WAP = tổng
utility của các itemset chứa nó; WAP không thuộc itemset nào xếp tiếp theo utility riêng
lẻ. Mỗi thuật toán chạy trong tiến trình con có giới hạn RAM/thời gian; nếu tràn thì
min_util tự tăng và thử lại. Cần thư viện `pami` (tự pip install nếu chưa có)."""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import numpy as np
import pandas as pd

ALGORITHMS = {
    "upgrowth": dict(
        label="UP-Growth",
        display_name="HUIM UP-Growth (chọn WAP)",
        color="#4C72B0",
        out_dir="wap_huim_upgrowth",
    ),
    "efim": dict(
        label="EFIM",
        display_name="HUIM-EFIM (chọn WAP)",
        color="#8172B3",
        out_dir="wap_huim_efim",
    ),
}
DEFAULT_ORDER = ("upgrowth", "efim")
COMPARE_DIR = "wap_huim_compare"

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
        if hasattr(os, "sysconf"):
            try:
                return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
            except Exception:
                pass
        return 2 * 1024 ** 3


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


def _prepare_utility(Xtr, s, seed):
    n_wap = Xtr.shape[1]
    rng = np.random.RandomState(seed)
    idx = np.arange(len(Xtr)) if not s["n_tx"] else np.sort(rng.choice(len(Xtr), min(s["n_tx"], len(Xtr)), replace=False))
    U = np.rint(Xtr[idx] * s["scale"]).astype(np.int64) * s["unit_profit"]       # utility từng WAP trong từng mẫu
    return U, U.sum(0).astype(float), n_wap


def _result(algorithm, res, single, cols, ratio, mu, log_df, wall, n_attempts, cumulative_wall,
            info_extra=None, n_transactions=None):
    from ips.experiments.knn_wap_sweep import RankResult
    meta = ALGORITHMS[algorithm]
    score = np.array(res["score"], dtype=float)
    cnt = np.array(res["cnt"], dtype=float)
    order = np.lexsort((-single, -score))            # điểm giảm dần; hòa thì theo utility riêng lẻ
    score_n = score / score.sum() if score.sum() > 0 else score
    table = pd.DataFrame({"hang": np.arange(1, len(cols) + 1), "WAP": np.asarray(cols)[order], "diem": score_n[order],
                          "so_itemset_chua_WAP": cnt[order].astype(int), "utility_rieng_le": single[order]})
    timing = pd.DataFrame([dict(
        thuat_toan=meta["label"], thoi_gian_khai_pha_s=res["time"], thoi_gian_wall_lan_cuoi_s=wall,
        tong_thoi_gian_xep_hang_s=cumulative_wall, so_lan_thu=n_attempts, bo_nho_mb=res["mem"],
        so_itemset=res["n"], min_util=mu, ty_le_min_util=ratio,
    )])
    top = pd.DataFrame({"itemset (top 10 theo utility)": [" + ".join(cols[j - 1] for j in its) for _, its in res["top"]],
                        "utility": [u for u, _ in res["top"]]})
    info = dict(algorithm=algorithm, label=meta["label"], min_util_ratio_used=ratio, min_util=mu,
                n_transactions=n_transactions, t_mine_s=res["time"], t_wall_s=wall, n_attempts=n_attempts,
                cumulative_rank_wall_s=cumulative_wall, n_itemset=int(res["n"]),
                itemset_length_counts=res["lens"], checksum=int(res["checksum"]))
    if info_extra:
        info.update(info_extra)
    return RankResult({algorithm: order}, {algorithm: res["time"]}, {algorithm: meta["label"]},
                      {"huim_thoi_gian.csv": timing, "huim_nhat_ky.csv": log_df.reset_index(drop=True),
                       "huim_top_itemset.csv": top, "huim_diem_wap.csv": table, "huim_thong_tin.json": info})


def rank_single(algorithm, Xtr, cols, s, seed):
    """Chạy đúng một thuật toán, tự leo min_util khi thuật toán đó thất bại."""
    ensure_pami()
    U, single, n_wap = _prepare_utility(Xtr, s, seed)
    total, meta = int(U.sum()), ALGORITHMS[algorithm]
    mem_limit = int(s["mem_fraction"] * available_ram())
    print(f"[m-sweep] HUIM {meta['label']}: {len(U)} giao dịch, {int((single > 0).sum())}/{n_wap} WAP có mặt, "
          f"tổng utility = {total:,}, giới hạn RAM mỗi lần khai phá = {mem_limit / 1e9:.1f} GB")
    with tempfile.TemporaryDirectory() as work:
        (Path(work) / "huim_worker.py").write_text(WORKER_SRC)
        db = Path(work) / "huim_db.txt"
        print(f"[m-sweep] Đã ghi {write_db(db, U)} giao dịch")
        rows, accepted, final = [], None, None
        for i in range(s["max_tries"]):
            ratio = s["min_util_ratio"] * s["escalate"] ** i; mu = int(ratio * total)
            print(f"[m-sweep] HUIM {meta['label']} lần {i + 1}/{s['max_tries']}: min_util = {ratio:.2%} tổng = {mu:,}")
            res, why, wall = run_guarded(work, algorithm, db, mu, n_wap, mem_limit, s["time_limit_s"])
            rows.append(dict(lan=i + 1, ty_le_min_util=ratio, min_util=mu, thuat_toan=meta["label"],
                             trang_thai=why, thoi_gian_wall_s=wall,
                             thoi_gian_khai_pha_s=(res["time"] if res is not None else np.nan),
                             loi="" if res is not None else why))
            print(f"[m-sweep]    {meta['label']}: {why} ({wall:.1f}s)")
            if res is not None:
                accepted, final = res, (ratio, mu, wall, i + 1)
                break
        if accepted is None:
            raise RuntimeError(f"HUIM {meta['label']} thất bại ở mọi min_util đã thử: giảm n_tx (vd 2000) "
                               f"hoặc tăng min_util_ratio/max_tries trong khóa wap_sweep.")
        ratio, mu, wall, n_attempts = final
        log_df = pd.DataFrame(rows)
        print(f"[m-sweep] HUIM {meta['label']}: min_util cuối = {ratio:.2%} tổng = {mu:,} | {accepted['n']} itemset")
        return _result(algorithm, accepted, single, cols, ratio, mu, log_df, wall, n_attempts,
                       float(log_df.thoi_gian_wall_s.sum()),
                       info_extra=dict(same_itemsets=None, shared_threshold=False), n_transactions=len(U))


def rank_joint(Xtr, cols, s, seed):
    """Chạy cả hai thuật toán ở cùng min_util; một bên lỗi thì cả hai cùng tăng."""
    ensure_pami()
    U, single, n_wap = _prepare_utility(Xtr, s, seed)
    total = int(U.sum())
    mem_limit = int(s["mem_fraction"] * available_ram())
    labels = {alg: ALGORITHMS[alg]["label"] for alg in DEFAULT_ORDER}
    print(f"[m-sweep] HUIM cặp ({' + '.join(labels[a] for a in DEFAULT_ORDER)}): {len(U)} giao dịch, "
          f"{int((single > 0).sum())}/{n_wap} WAP có mặt, tổng utility = {total:,}, "
          f"giới hạn RAM mỗi lần khai phá = {mem_limit / 1e9:.1f} GB")
    with tempfile.TemporaryDirectory() as work:
        (Path(work) / "huim_worker.py").write_text(WORKER_SRC)
        db = Path(work) / "huim_db.txt"
        print(f"[m-sweep] Đã ghi {write_db(db, U)} giao dịch")
        rows, accepted, final, final_walls, last_error = [], {}, None, {}, {}
        for i in range(s["max_tries"]):
            ratio = s["min_util_ratio"] * s["escalate"] ** i; mu = int(ratio * total)
            print(f"[m-sweep] HUIM cặp lần {i + 1}/{s['max_tries']}: min_util = {ratio:.2%} tổng = {mu:,}")
            got, walls = {}, {}
            for alg in DEFAULT_ORDER:
                res, why, wall = run_guarded(work, alg, db, mu, n_wap, mem_limit, s["time_limit_s"])
                got[alg], walls[alg], last_error[alg] = res, wall, why
                rows.append(dict(lan=i + 1, ty_le_min_util=ratio, min_util=mu, thuat_toan=labels[alg],
                                 trang_thai=why, thoi_gian_wall_s=wall,
                                 thoi_gian_khai_pha_s=(res["time"] if res is not None else np.nan),
                                 loi="" if res is not None else why))
                print(f"[m-sweep]    {labels[alg]}: {why} ({wall:.1f}s)")
            if all(got[alg] is not None for alg in DEFAULT_ORDER):
                accepted, final, final_walls = got, (ratio, mu, i + 1), walls
                break
        if len(accepted) < len(DEFAULT_ORDER):
            detail = "; ".join(f"{labels[alg]}: {last_error.get(alg, 'chưa chạy')}" for alg in DEFAULT_ORDER)
            raise RuntimeError(f"HUIM cặp thất bại ở mọi min_util đã thử ({detail}). Giảm n_tx (vd 2000) "
                               f"hoặc tăng min_util_ratio/max_tries trong khóa wap_sweep.")
        ratio, mu, n_attempts = final
        log_df = pd.DataFrame(rows)
        up, ef = accepted["upgrowth"], accepted["efim"]
        same = up["n"] == ef["n"] and up["checksum"] == ef["checksum"]
        cumulative = {alg: float(log_df[log_df.thuat_toan == labels[alg]].thoi_gian_wall_s.sum()) for alg in DEFAULT_ORDER}
        ranks = {}
        for alg in DEFAULT_ORDER:
            other = next(a for a in DEFAULT_ORDER if a != alg)
            log_alg = log_df[log_df.thuat_toan == labels[alg]].reset_index(drop=True)
            ranks[alg] = _result(alg, accepted[alg], single, cols, ratio, mu, log_alg, final_walls[alg], n_attempts,
                                 cumulative[alg], info_extra=dict(same_itemsets=bool(same), shared_threshold=True,
                                                                  other_algorithm=labels[other]), n_transactions=len(U))
        orders = {alg: ranks[alg].orders[alg] for alg in DEFAULT_ORDER}
        same_order = bool(np.array_equal(orders["upgrowth"], orders["efim"]))
        print(f"[m-sweep] HUIM cặp: min_util cuối = {ratio:.2%} tổng = {mu:,} | {up['n']} itemset | "
              f"UP-Growth và EFIM cho cùng tập itemset: {same} | cùng thứ tự: {same_order}")
        print(f"[m-sweep] HUIM cặp thời gian khai phá: UP-Growth {up['time']:.2f}s, EFIM {ef['time']:.2f}s")
        return dict(ranks=ranks, attempts=log_df, results=accepted, walls=final_walls, cumulative_walls=cumulative,
                    min_util_ratio=ratio, min_util=mu, n_transactions=len(U), same_itemsets=bool(same),
                    same_order=same_order, orders=orders, n_attempts=n_attempts, n_wap=n_wap,
                    scores={alg: np.array(accepted[alg]["score"], dtype=float) for alg in DEFAULT_ORDER})


class HUIMSelectorBase:
    is_wap_selector = True
    algorithm = None
    defaults = dict(scale=1000, unit_profit=1, min_util_ratio=0.005, escalate=1.5, max_tries=8, n_tx=5000,
                    time_limit_s=600, mem_fraction=0.7)

    def rank(self, Xtr, ref, cols, s, seed):
        return rank_single(self.algorithm, Xtr, cols, s, seed)
