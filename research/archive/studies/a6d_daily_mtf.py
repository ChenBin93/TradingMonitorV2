#!/usr/bin/env python3
"""A6d 日线趋势 × 4H 道氏段命运 (2026-08-04, 无未来函数, 30 GBM 种子)

问题 (用户): 日线级别的大趋势对 4H 的趋势有影响吗?

设计:
  日线数据 = 4H 重采样聚合 (每 6 根 4H = 1 日线 bar); 无未来函数:
  4H bar t 只使用"已收盘日线" (日线收盘时刻 ≤ t 时刻, searchsorted 右界)
  日线道氏段 = 同一套 dow_segments 跑日线

分层 (4H 道氏段按日线状态, 段起点时刻):
  D1 日线方向: up / down / range / warmup (4H bar 时间占比)
  D2 一致性: 日线方向 × 4H 段方向 一致 / 相反 / 无日线趋势
  D3 日线位置: 4H 段起点在日线段中的位置 早(<0.33)/中/晚(>0.67)

度量 (4H 段命运):
  段长中位/均值 | 存活 (段长≥25 占比) | 回撤恢复率 (HL 后 48 根越段内峰值)

预注册假设:
  H1 日线上升趋势中的 4H 上升段更长
  H2 日线趋势与 4H 段同向 → 恢复率更高 (大周期顺风)
  H3 4H 段处于日线段晚期 → 更脆弱 (恢复率低)

运行: python3 research/studies/a6d_daily_mtf.py
"""
import os
import sys
import gc
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.data_loader import load_candles, verify
from research.sim_market import gbm_dataframe
from research.structures import dow_segments

TF = "4h"
N_GBM = 30
RECOVER_WIN = 48


def resample_daily(df):
    idx = df.index
    day = np.array([ts.date() for ts in idx])
    out = []
    for d in sorted(set(day)):
        m = day == d
        out.append((pd.Timestamp(d), df["open"].values[np.flatnonzero(m)[0]],
                    df["high"].values[m].max(), df["low"].values[m].min(),
                    df["close"].values[m][-1]))
    di = pd.DatetimeIndex([x[0] for x in out])
    return pd.DataFrame({"open": [x[1] for x in out], "high": [x[2] for x in out],
                         "low": [x[3] for x in out], "close": [x[4] for x in out]},
                        index=di)


def collect(df, acc):
    n = len(df)
    hi = df["high"].values
    lo = df["low"].values
    d = dow_segments(df)
    daily = resample_daily(df)
    dd = dow_segments(daily)
    dstates = dd["states"]
    # 日线收盘时刻 (日线 bar 的下一日 00:00) 与 4H bar 时刻的映射
    daily_end = np.array([ts.value + 24 * 3600 * 10 ** 9 for ts in daily.index],
                         dtype="int64")
    t_vals = np.array([ts.value for ts in df.index], dtype="int64")
    d_idx = np.searchsorted(daily_end, t_vals, "right") - 1
    # 日线段位置: 逐日映射段索引
    seg_of_day = np.full(len(daily), -1)
    for si, s in enumerate(dd["segs"]):
        seg_of_day[s["start"]:s["end"] + 1] = si
    # 4H 段分层
    for s in d["segs"]:
        if s["direction"] not in ("up", "down"):
            continue
        di = d_idx[s["start"]]
        day_dir = str(dstates[di]) if 0 <= di < len(dstates) else "warmup"
        if day_dir == s["direction"]:
            cons = "一致"
        elif day_dir in ("up", "down"):
            cons = "相反"
        else:
            cons = "无日线趋势"
        rec = dict(bars=s["bars"], dir=s["direction"], day_dir=day_dir, cons=cons)
        if day_dir in ("up", "down") and 0 <= di < len(seg_of_day):
            si2 = seg_of_day[di]
            if si2 >= 0:
                dseg = dd["segs"][si2]
                pos = (di - dseg["start"]) / max(1, dseg["bars"])
                rec["dpos"] = "早" if pos < 0.33 else ("晚" if pos > 0.67 else "中")
        else:
            rec["dpos"] = "无日线段"
        acc["seg"].append(rec)
        acc["time_frac"][day_dir] += s["bars"]
    # 恢复率: 4H 回撤事件按所在 4H 段的层
    seg_of_4h = np.full(n, -1)
    for si, s in enumerate(d["segs"]):
        seg_of_4h[s["start"]:s["end"] + 1] = si
    for r in d["retraces"]:
        si = seg_of_4h[r["bar"]]
        if si < 0 or r["bar"] + RECOVER_WIN >= n:
            continue
        s = d["segs"][si]
        if s["direction"] not in ("up", "down"):
            continue
        di = d_idx[r["bar"]]
        day_dir = str(dstates[di]) if 0 <= di < len(dstates) else "warmup"
        cons = ("一致" if day_dir == s["direction"]
                else ("相反" if day_dir in ("up", "down") else "无日线趋势"))
        if r["direction"] == "up":
            rec = float(hi[r["bar"] + 1:r["bar"] + RECOVER_WIN + 1].max() > r["peak_val"])
        else:
            rec = float(lo[r["bar"] + 1:r["bar"] + RECOVER_WIN + 1].min() < r["trough_val"])
        acc["rec"][cons].append(rec)
        if "dpos" in s and s["dpos"]:
            pass
        # 位置分层恢复率: 用段起点日线位置近似
        di2 = d_idx[s["start"]]
        if day_dir in ("up", "down") and 0 <= di2 < len(seg_of_day):
            si2 = seg_of_day[di2]
            if si2 >= 0:
                dseg = dd["segs"][si2]
                pos = (di2 - dseg["start"]) / max(1, dseg["bars"])
                key = "早" if pos < 0.33 else ("晚" if pos > 0.67 else "中")
                acc["rec_pos"][key].append(rec)


def report(acc, label):
    print(f"\n═══ {label} ═══")
    segs = acc["seg"]
    if not segs:
        print("  无段")
        return
    bars = np.array([x["bars"] for x in segs])
    print(f"D1 4H bar 时间占比 (日线状态): "
          + " | ".join(f"{k}:{v / max(1, sum(acc['time_frac'].values())):.0%}"
                       for k, v in acc["time_frac"].items()))
    print("D2 一致性 × 段命运:")
    for dr in ("up", "down"):
        print(f"  {dr} 段:")
        for cons in ("一致", "相反", "无日线趋势"):
            m = [x for x in segs if x["dir"] == dr and x["cons"] == cons]
            if len(m) < 30:
                continue
            b = np.array([x["bars"] for x in m])
            surv = np.mean(b >= 25)
            print(f"    {cons:<8} n={len(m):>4} 段长中位 {np.median(b):>4.0f} 根 "
                  f"均值 {b.mean():>4.1f} | 存活(≥25根) {surv:.0%}")
    print("D3 日线位置 × 段长 (4H 段起点在日线段的位置):")
    for pos in ("早", "中", "晚", "无日线段"):
        m = [x for x in segs if x["dpos"] == pos]
        if len(m) < 30:
            continue
        b = np.array([x["bars"] for x in m])
        print(f"    {pos:<6} n={len(m):>4} 段长中位 {np.median(b):>4.0f} 根")
    print("恢复率:")
    for cons in ("一致", "相反", "无日线趋势"):
        r = acc["rec"].get(cons, [])
        if len(r) < 50:
            continue
        print(f"  {cons:<8} n={len(r):>4} {np.mean(r):.1%}")
    for pos in ("早", "中", "晚"):
        r = acc["rec_pos"].get(pos, [])
        if len(r) < 50:
            continue
        print(f"  位置{pos:<6} n={len(r):>4} {np.mean(r):.1%}")


def diff_block(real, gbm):
    print(f"\n── 净差 (真实−GBM) ──")
    for dr in ("up", "down"):
        for cons in ("一致", "相反", "无日线趋势"):
            r = [x["bars"] for x in real["seg"] if x["dir"] == dr and x["cons"] == cons]
            g = [x["bars"] for x in gbm["seg"] if x["dir"] == dr and x["cons"] == cons]
            if len(r) < 30 or len(g) < 30:
                continue
            rb, gb = np.median(r), np.median(g)
            print(f"  {dr}/{cons}: 段长中位 {rb:.0f} vs {gb:.0f} (Δ {rb - gb:+.0f})")
    for cons in ("一致", "相反", "无日线趋势"):
        r = real["rec"].get(cons, [])
        g = gbm["rec"].get(cons, [])
        if len(r) < 50 or len(g) < 50:
            continue
        print(f"  恢复率 {cons}: 真实 {np.mean(r):.1%} vs GBM {np.mean(g):.1%} "
              f"→ Δ {np.mean(r) - np.mean(g):+.1%}")


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


def run_block(dfs, label=""):
    acc = dict(seg=[], time_frac=defaultdict(int), rec=defaultdict(list),
               rec_pos=defaultdict(list))
    for df in dfs:
        collect(df, acc)
        gc.collect()
    report(acc, label)
    return acc


def main():
    dfs = _load(TF)
    real = run_block(dfs, "真实")
    ref = dfs[0]
    sig = np.std(np.diff(np.log(ref["close"].values)))
    rw = [gbm_dataframe(len(ref), sig, seed=13000 + k) for k in range(N_GBM)]
    gbm = run_block(rw, "GBM")
    diff_block(real, gbm)


if __name__ == "__main__":
    main()
