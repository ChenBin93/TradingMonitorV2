#!/usr/bin/env python3
"""A4 趋势深度画像 — 趋势状态由什么构成 (2026-08-03, 纯描述性, 无未来函数)

方向修正: 不再用 1:1 入场胜率描述市场 (那是交易方式检验, 不是市场描述)。
本研究的五个纯统计问题 (不涉及任何入场/胜率口径):

  Q1 K线构成: 趋势各阶段内顺/逆 K 的数量比例、平均幅度、实体占比
  Q2 收益分布: 趋势状态内单根收益的均值/偏度/峰度 vs 非趋势 vs 随机游走
     — 趋势的方向性来自"数量"(顺K多)还是"幅度"(正偏度)?
  Q3 推进结构: 推进段(连续顺K)与回调段的长度/幅度/占比分布 — 趋势的节奏
  Q4 阶段差异: early/accelerate/late 的上述统计如何演变
  Q5 多空镜像: 上升 vs 下降趋势的构成对称性

对照: 随机游走 (真实 OHLC 子步) 同管线 — 趋势构成的"异常"程度
无未来函数: 状态标签只用已收盘 bar (state_series 已过不变性测试)
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.data_loader import load_candles, verify
from research.sim_market import gbm_dataframe
from research.state_features import state_series

WARMUP = 200


def _load(timeframes=("1h", "4h")):
    data = load_candles(timeframes=timeframes)
    out = {}
    for sym, tfs in data.items():
        for tf in timeframes:
            df = tfs.get(tf)
            if df is None:
                continue
            if verify(df, sym, tf):
                continue
            out.setdefault(tf, []).append(df)
    return out


def direction_of(s):
    if s.startswith("trend_up"):
        return 1
    if s.startswith("trend_down"):
        return -1
    return 0


def trend_stat(df):
    """单标的趋势构成统计 — 返回按状态聚合的字典"""
    st, _ = state_series(df)
    c = df["close"].values
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    n = len(c)
    ret = np.zeros(n)
    ret[1:] = np.diff(c) / c[:-1]
    body = np.abs(c - o)
    rng = np.maximum(h - l, 1e-12)
    body_ratio = body / rng
    up_k = np.zeros(n, bool)
    up_k[1:] = c[1:] > c[:-1]
    agg = {}
    for i in range(WARMUP, n):
        s = st[i]
        if s == "unknown":
            continue
        a = agg.setdefault(s, {"n": 0, "n_up": 0, "ret": [], "up_amp": [], "dn_amp": [],
                               "body_up": [], "body_dn": []})
        a["n"] += 1
        if up_k[i]:
            a["n_up"] += 1
            a["up_amp"].append(abs(ret[i]))
            a["body_up"].append(body_ratio[i])
        else:
            a["dn_amp"].append(abs(ret[i]))
            a["body_dn"].append(body_ratio[i])
        a["ret"].append(ret[i])
    return agg


def run_stats(aggs):
    """聚合所有标的: {状态: {统计}}"""
    out = {}
    for agg in aggs:
        for s, a in agg.items():
            o = out.setdefault(s, {"n": 0, "n_up": 0, "ret": [], "up_amp": [],
                                   "dn_amp": [], "body_up": [], "body_dn": []})
            o["n"] += a["n"]
            o["n_up"] += a["n_up"]
            o["ret"].extend(a["ret"])
            o["up_amp"].extend(a["up_amp"])
            o["dn_amp"].extend(a["dn_amp"])
            o["body_up"].extend(a["body_up"])
            o["body_dn"].extend(a["body_dn"])
    return out


def summarize(stats):
    rows = []
    for s in sorted(stats):
        a = stats[s]
        n = a["n"]
        if n == 0:
            continue
        up_ratio = a["n_up"] / n
        up_amp = np.mean(a["up_amp"]) if a["up_amp"] else np.nan
        dn_amp = np.mean(a["dn_amp"]) if a["dn_amp"] else np.nan
        body_up = np.mean(a["body_up"]) if a["body_up"] else np.nan
        body_dn = np.mean(a["body_dn"]) if a["body_dn"] else np.nan
        r = np.array(a["ret"])
        rows.append({
            "s": s, "n": n, "up_ratio": up_ratio, "up_amp": up_amp, "dn_amp": dn_amp,
            "body_up": body_up, "body_dn": body_dn,
            "mean": r.mean(), "std": r.std(), "skew": skew(r) if len(r) > 10 else np.nan,
            "kurt": kurtosis(r) if len(r) > 10 else np.nan,
        })
    return rows


def run_block(dfs, label=""):
    print(f"═══ A4 趋势深度画像 {label}═══\n")
    aggs = [trend_stat(df) for df in dfs]
    stats = run_stats(aggs)
    rows = summarize(stats)

    print("Q1 K线构成 (顺K占比/顺逆K均幅/实体占比)")
    print(f"  {'状态':>24} {'n':>8} {'顺K占比':>8} {'顺K均幅':>9} {'逆K均幅':>9} "
          f"{'顺实体':>7} {'逆实体':>7}")
    for r in rows:
        print(f"  {r['s']:>24} {r['n']:>8} {r['up_ratio']:>8.1%} {r['up_amp']:>9.3%} "
              f"{r['dn_amp']:>9.3%} {r['body_up']:>7.2f} {r['body_dn']:>7.2f}")
    print()

    print("Q2 收益分布 (单根收益 均值/std/偏度/峰度)")
    print(f"  {'状态':>24} {'均值':>9} {'std':>9} {'偏度':>7} {'峰度':>7}")
    for r in rows:
        print(f"  {r['s']:>24} {r['mean']:>9.4%} {r['std']:>9.4%} {r['skew']:>7.2f} {r['kurt']:>7.2f}")
    print()

    print("Q3 推进结构 (顺K段/逆K段 run 统计)")
    runs = {s: {"fwd_len": [], "fwd_amp": [], "bak_len": [], "bak_amp": []} for s in
            ("trend_up:early", "trend_up:accelerate", "trend_up:late",
             "trend_down:early", "trend_down:accelerate", "trend_down:late")}
    for df in dfs:
        st, _ = state_series(df)
        c = df["close"].values
        n = len(c)
        up_k = np.zeros(n, bool)
        up_k[1:] = c[1:] > c[:-1]
        ret = np.zeros(n)
        ret[1:] = np.diff(c) / c[:-1]
        i = WARMUP
        while i < n - 1:
            s = st[i]
            if s in runs:
                d = direction_of(s)
                # 一段 run: 同状态连续 bar
                j = i
                while j < n and st[j] == s:
                    j += 1
                seg_len = j - i
                if seg_len >= 2:
                    # 段内按方向分 run
                    k = i
                    while k < j:
                        kind = "fwd" if (up_k[k] == (d == 1)) else "bak"
                        kk = k
                        amp = 0.0
                        while kk < j and (up_k[kk] == (d == 1)) == (kind == "fwd"):
                            amp += abs(ret[kk])
                            kk += 1
                        runs[s][kind + "_len"].append(kk - k)
                        runs[s][kind + "_amp"].append(amp)
                        k = kk
                i = j
            else:
                i += 1
    print(f"  {'状态':>24} {'顺K段均长':>9} {'逆K段均长':>9} {'顺段均幅':>9} {'逆段均幅':>9} "
          f"{'顺段数/逆段数':>12}")
    for s in runs:
        if not runs[s]["fwd_len"]:
            continue
        fl = np.mean(runs[s]["fwd_len"])
        bl = np.mean(runs[s]["bak_len"]) if runs[s]["bak_len"] else np.nan
        fa = np.mean(runs[s]["fwd_amp"])
        ba = np.mean(runs[s]["bak_amp"]) if runs[s]["bak_amp"] else np.nan
        print(f"  {s:>24} {fl:>9.2f} {bl:>9.2f} {fa:>9.3%} {ba:>9.3%} "
              f"{len(runs[s]['fwd_len']):>6}/{len(runs[s]['bak_len']):>5}")
    print()

    # Q5 多空镜像: up 聚合 vs down 聚合
    print("Q5 多空镜像 (趋势多 vs 趋势空 聚合)")
    up = {"n": 0, "n_up": 0, "ret": [], "up_amp": [], "dn_amp": []}
    dn = {"n": 0, "n_up": 0, "ret": [], "up_amp": [], "dn_amp": []}
    for s in stats:
        if s.startswith("trend_up"):
            t = up
        elif s.startswith("trend_down"):
            t = dn
        else:
            continue
        t["n"] += stats[s]["n"]
        t["n_up"] += stats[s]["n_up"]
        t["ret"].extend(stats[s]["ret"])
        t["up_amp"].extend(stats[s]["up_amp"])
        t["dn_amp"].extend(stats[s]["dn_amp"])
    for name, t in [("趋势多(up)", up), ("趋势空(dn)", dn)]:
        r = np.array(t["ret"])
        print(f"  {name:>10}: n={t['n']:>8} 顺K占比 {t['n_up']/t['n']:>7.1%} "
              f"顺幅 {np.mean(t['up_amp']):>7.3%} 逆幅 {np.mean(t['dn_amp']):>7.3%} "
              f"均值 {r.mean():>8.4%} 偏度 {skew(r):>7.2f}")
    print()


def main():
    dfs_by_tf = _load()
    for tf in ("1h", "4h"):
        run_block(dfs_by_tf[tf], label=f"({tf})")
        # 随机游走对照
        ref = dfs_by_tf[tf][0]
        n_ref = len(ref)
        sig = np.std(np.diff(np.log(ref["close"].values)))
        rw = [gbm_dataframe(n_ref, sig, seed=100 + k) for k in range(20)]
        run_block(rw, label=f"({tf} 随机游走对照)")


if __name__ == "__main__":
    main()
