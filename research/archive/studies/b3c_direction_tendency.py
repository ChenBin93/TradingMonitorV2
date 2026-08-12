#!/usr/bin/env python3
"""B3c 多根整体倾向方向性 (2026-08-04, 无未来函数, 方向性统一定义为"数根K的整体倾向")

修正 (用户指令, 2026-08-04):
  1. GBM 对照种子 10 → 30: 单种子 P(up) 波动 ±2.6pp (30 种子均值精确 50.0%),
     10 种子存在抽样运气 — B3b 部分净差可能是种子噪声
  2. 方向性度量全部以"触碰时刻 close[t]"为参照 (对称度量, GBM 上无条件
     无偏 — 已验证 E1 端点 = 50.0%); 废除"相对位带中心"的方向度量 (M1 类,
     触碰采样位置偏置: 区间内触下沿时 close 系统性偏中心上方, GBM 上
     63% ≠ 50%, 是几何/采样偏置, 不是方向)
  3. 方向性定义统一为"数根 K 的整体倾向" (用户确认): 不以单根/端点为判据

度量 (未来 24 根, 全部以 close[t] 为参照, GBM 上应无偏):
  E1 端点:   P(c[t+24] > c[t])                    — 基准 (旧定义, 对照)
  E2 占优:   P(median(seg) > c[t])                — 超过一半根数在上方
  E3 积分:   mean(Σ log(seg/c[t]) / (atr_t/c_t))   — 路径净漂移 (对数, 修
             Jensen 偏置: 算术和 Σ(seg−c_t) 在对数正态下无条件正偏,
             GBM 上实测 +1.27 ATR; 对数积分无条件对称 → 0)
  E4 摆幅:   P(log(max/c_t) > log(c_t/min))        — 向上摆得比向下远
             (对数比较: 算术 max−c_t vs c_t−min 有右尾偏置)
  E5 偏度:   mean(skew(log-diff(seg)))             — 收益偏斜 (0 为无偏)
  E6 曲线:   E[log(seg[k]/c_t)] for k=1..24         — 漂移轨迹形态 (对数)

无偏性门禁 (2026-08-04 用户明令: 随机数据任何条件化应 ≈50%):
  - GBM 条件组 n < 2000: 稀有条件簇集抽样波动大 (实测: 区间内×触≤2 n=422
    时 E1 在 30 种子 =58.5% 但新种子集 =48.7%, 纯抽样波动), 不做绝对水平
    判断, 只报告真实−GBM 净差
  - GBM 条件组 n ≥ 2000 且偏离 50% > 2pp: 标记 ⚠ 有偏置, 该组不用于方向性
  - 真偏置已修: E3/E4/E6 对数化 (算术/对数 Jensen 效应)

条件: all / 区间内 / 区间内×波动低高 / 区间内×触碰序号≤2,≥3 (确认度近似)

输出: [GBM 无偏性检验] (各条件化下应 ≈50%/0) → [真实] → [净差矩阵] → [E6 曲线]

运行: python3 research/studies/b3c_direction_tendency.py (~1.5h)
"""
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import skew as _skew

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.data_loader import load_candles, verify
from research.sim_market import gbm_dataframe
from research.studies.b3_general_levels import (collect, bucket_key, W, MIN_N)

PARAMS = [(2, 0.3), (3, 0.5)]
TIMEFRAMES = ["1h", "4h"]
N_GBM = 30

GROUPS = {
    "all": lambda b: True,
    "区间内": lambda b: b["structure"] == "区间内",
    "区间内×vol低": lambda b: b["structure"] == "区间内" and b["vol"] == "低",
    "区间内×vol高": lambda b: b["structure"] == "区间内" and b["vol"] == "高",
    "区间内×触≤2": lambda b: b["structure"] == "区间内" and b["seq"] <= 2,
    "区间内×触≥3": lambda b: b["structure"] == "区间内" and b["seq"] >= 3,
}


def event_metrics(seg, c_t, atr_t, side):
    """触碰后 24 根的整体倾向度量 (以 close[t] 为参照, 对数对称 → GBM 无偏)"""
    lg = np.log(seg / c_t[:, None])            # (n, W) 对数增量
    m = {
        "e1": (seg[:, -1] > c_t).astype(float),
        "e2": (np.median(seg, axis=1) > c_t).astype(float),
        "e3": lg.sum(axis=1) / (atr_t / c_t),  # 对数积分, ATR 归一
        "e4": (np.log(seg.max(axis=1) / c_t) > np.log(c_t / seg.min(axis=1))).astype(float),
    }
    dl = np.diff(lg, axis=1)                    # (n, W-1) 对数收益
    m["e5"] = _skew(dl, axis=1) if dl.shape[1] >= 3 else np.zeros(len(c_t))
    return m


def run_block(dfs, mt, tol, label=""):
    print(f"\n{'═' * 70}\nB3c {label} 参数(min_touch={mt}, tol={tol})\n{'═' * 70}")
    acc = {g: defaultdict(list) for g in GROUPS}
    curve = defaultdict(list)                  # 区间内组 E6 逐根
    for df in dfs:
        d = collect(df, mt, tol)
        c, atr, n = d["c"], d["atr"], d["n"]
        for i, lv in enumerate(d["lvls"]):
            t_arr = d["ev"]["touch"][i]
            if not len(t_arr):
                continue
            valid = (t_arr + W < n) & (t_arr >= 13) & (atr[t_arr] > 0)
            t_arr = t_arr[valid]
            if not len(t_arr):
                continue
            seg = c[t_arr[:, None] + np.arange(1, W + 1)]
            mm = event_metrics(seg, c[t_arr], atr[t_arr], lv.side)
            for j, t in enumerate(t_arr):
                b = bucket_key({"t": int(t), "lv": lv, "d": d, "seq": j + 1})
                b["seq"] = j + 1
                for g, pred in GROUPS.items():
                    if not pred(b):
                        continue
                    a = acc[g]
                    a["n"].append(1)
                    for k in ("e1", "e2", "e3", "e4", "e5"):
                        a[k].append(float(mm[k][j]))
                    if g == "区间内":
                        curve["real" if "真实" in label else "gbm"].append(
                            np.log(seg[j] / c[t_arr[j]]) / (atr[t_arr[j]] / c[t_arr[j]]))
    _print_table(acc, label)
    return acc, curve


def _print_table(acc, label):
    is_gbm = "GBM" in label
    print(f"{'组':<14}{'n':>8} {'E1端点':>7} {'E2占优':>7} {'E3积分':>8} "
          f"{'E4摆幅':>7} {'E5偏度':>7}" + ("  无偏性门禁" if is_gbm else ""))
    for g in GROUPS:
        a = acc[g]
        nn = len(a["n"])
        if nn < MIN_N:
            print(f"{g:<14}{nn:>8}   n 不足")
            continue
        e1, e2, e3, e4, e5 = (np.mean(a["e1"]), np.mean(a["e2"]),
                              np.mean(a["e3"]), np.mean(a["e4"]), np.mean(a["e5"]))
        line = (f"{g:<14}{nn:>8} {e1:>7.1%} {e2:>7.1%} {e3:>8.3f} "
                f"{e4:>7.1%} {e5:>7.3f}")
        if is_gbm:
            if nn < 2000:
                line += "  †稀疏(仅净差)"
            elif abs(e1 - 0.5) > 0.02 or abs(e2 - 0.5) > 0.02 or abs(e4 - 0.5) > 0.02 \
                    or abs(e3) > 0.1 or abs(e5) > 0.02:
                line += "  ⚠偏置"
            else:
                line += "  ✓无偏"
        print(line)


def diff_block(real, gbm, label=""):
    print(f"\n── 净差矩阵 (真实−GBM, 30 种子) {label} ──")
    print(f"{'组':<14}{'n_真':>8}{'n_GBM':>8} {'ΔE1':>7} {'ΔE2':>7} {'ΔE3':>8} "
          f"{'ΔE4':>7} {'ΔE5':>7}")
    for g in GROUPS:
        rd, gd = real[0][g], gbm[0][g]
        nn, gn = len(rd["n"]), len(gd["n"])
        if nn < MIN_N or gn < MIN_N:
            print(f"{g:<14}{nn:>8}{gn:>8}   n 不足")
            continue
        d1 = np.mean(rd["e1"]) - np.mean(gd["e1"])
        d2 = np.mean(rd["e2"]) - np.mean(gd["e2"])
        d3 = np.mean(rd["e3"]) - np.mean(gd["e3"])
        d4 = np.mean(rd["e4"]) - np.mean(gd["e4"])
        d5 = np.mean(rd["e5"]) - np.mean(gd["e5"])
        print(f"{g:<14}{nn:>8}{gn:>8} {d1:>7.1%} {d2:>7.1%} {d3:>8.3f} "
              f"{d4:>7.1%} {d5:>7.3f}")


def curve_block(real, gbm, label=""):
    print(f"\n── E6 区间内触碰后逐根均值曲线 (ATR 归一) {label} ──")
    rc = np.mean(np.vstack(real[1]["real"]), axis=0)
    gc = np.mean(np.vstack(gbm[1]["gbm"]), axis=0)
    ks = [0, 1, 3, 7, 11, 17, 23]
    print(f"{'k':>4}" + "".join(f"{'真实':>9}{'GBM':>9}{'Δ':>9}" for _ in ks))
    for k in ks:
        print(f"{k + 1:>4}" + "".join(f"{rc[k]:>9.3f}{gc[k]:>9.3f}{rc[k] - gc[k]:>9.3f}"
                                      for k in [k]))


def _load(tf):
    data = load_candles(timeframes=(tf,))
    out = []
    for sym, tfs in data.items():
        df = tfs.get(tf)
        if df is None:
            continue
        if verify(df, sym, tf):
            continue
        out.append(df)
    return out


def main():
    for mt, tol in PARAMS:
        for tf in TIMEFRAMES:
            print(f"\n########## 真实 {tf} (min_touch={mt}, tol={tol}) ##########")
            dfs = _load(tf)
            real = run_block(dfs, mt, tol, f"真实 {tf}")
            ref = dfs[0]
            sig = np.std(np.diff(np.log(ref["close"].values)))
            rw = [gbm_dataframe(len(ref), sig, seed=2000 + (tf == "4h") * 1000 + k)
                  for k in range(N_GBM)]
            gbm = run_block(rw, mt, tol, f"GBM {tf}")
            diff_block(real, gbm, f"{tf} 参数({mt},{tol})")
            curve_block(real, gbm, f"{tf} 参数({mt},{tol})")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
