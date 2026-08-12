#!/usr/bin/env python3
"""B5 非震荡区域关键位 — 方向维度 (2026-08-04, 无未来函数, 1h)

动机 (用户): 转向非震荡区域 (趋势行情) 的关键位; 先测方向维度
(顺势/逆势 + 突破延续), 再测结构维度。

状态 (A2 状态机 state_series, 触碰时刻):
  trend_up:* → 涨 | trend_down:* → 跌 | transition/range → 中性
  非震荡区域 ≈ 趋势态 (涨/跌) 中的触碰

度量 (触碰后 24 根, ATR 归一, 全部 GBM 30 种子对照):
  D1 沿趋势方向概率: 涨: P(c[t+24]>c[t]); 跌: P(c[t+24]<c[t]); 中性: P(up)
  D2 顺势/逆势位移: 沿趋势方向最大位移 vs 反向 (P50/P90)
  D3 突破延续: 确认突破后 24 根顺突破方向位移 × 趋势方向 (E4 分层重测)
  E1 波动释放按趋势层

关键组合: 趋势方向 × 触碰侧 (触下沿=支撑回调 / 触上沿=阻力触及)
  涨趋势+触下沿: 顺势回调买入预期 | 涨趋势+触上沿: 阻力拒绝(逆势)
  跌趋势+触上沿: 顺势回调卖出预期 | 跌趋势+触下沿: 支撑拒绝(逆势)

运行: python3 research/studies/b5_trend_levels.py
"""
import os
import sys
import gc
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.data_loader import load_candles, verify
from research.sim_market import gbm_dataframe
from research.state_features import state_series
from research.studies.b3_general_levels import (collect as b3collect, bucket_key,
                                                W, MIN_N)

MT, TOL = 2, 0.3
TF = "1h"
N_GBM = 30
W24 = 24


def trend_dir(states, t):
    s = states[t]
    if s.startswith("trend_up"):
        return "涨"
    if s.startswith("trend_down"):
        return "跌"
    return "中性"


def run_block(dfs, label=""):
    print(f"\n{'═' * 74}\nB5 {label} (1h, mt={MT}, tol={TOL})\n{'═' * 74}")
    acc = defaultdict(list)   # key -> list of (d1, fwd, bwd)
    break_acc = defaultdict(lambda: (0.0, 0))
    for df in dfs:
        _process_one(df, acc, break_acc)
        gc.collect()
    _report(acc, break_acc)
    return acc, break_acc


def _process_one(df, acc, break_acc):
    d = b3collect(df, MT, TOL)
    c, h, l, atr, n = d["c"], d["h"], d["l"], d["atr"], d["n"]
    states, _ = state_series(df)
    for i, lv in enumerate(d["lvls"]):
        t_arr = d["ev"]["touch"][i]
        valid = (t_arr + W24 < n) & (t_arr >= 14) & (atr[t_arr] > 0)
        t_arr = t_arr[valid]
        if len(t_arr):
            seg = c[t_arr[:, None] + np.arange(1, W24 + 1)]
            st = states[t_arr]
            up, dn = np.char.startswith(st, "trend_up"), np.char.startswith(st, "trend_down")
            c_t, hi, lo = c[t_arr], seg.max(axis=1), seg.min(axis=1)
            d1 = np.where(up | dn, np.where(up, seg[:, -1] > c_t, seg[:, -1] < c_t),
                          seg[:, -1] > c_t)
            fwd = np.where(up, hi - c_t, np.where(dn, c_t - lo, np.maximum(hi - c_t, c_t - lo)))
            bwd = np.where(up, c_t - lo, np.where(dn, hi - c_t, np.minimum(hi - c_t, c_t - lo)))
            fwd, bwd = fwd / atr[t_arr], bwd / atr[t_arr]
            key = f"{lv.side}"
            if up.any():
                acc[f"涨_{key}"].extend(zip(d1[up].tolist(), fwd[up].tolist(), bwd[up].tolist()))
            if dn.any():
                acc[f"跌_{key}"].extend(zip(d1[dn].tolist(), fwd[dn].tolist(), bwd[dn].tolist()))
            if (~(up | dn)).any():
                m = ~(up | dn)
                acc["中性"].extend(zip(d1[m].tolist(), fwd[m].tolist(), bwd[m].tolist()))
        # D3 突破延续 (在线统计 sum/count — confirmed 事件量巨大, 不能驻留列表)
        conf = d["ev"]["confirmed"][i]
        valid_c = (conf + W24 < n) & (conf >= 14) & (atr[conf] > 0)
        conf = conf[valid_c]
        if len(conf):
            if lv.side == "support":
                disp = (c[conf] - c[conf + W24]) / atr[conf]
            else:
                disp = (c[conf + W24] - c[conf]) / atr[conf]
            stc = states[conf]
            up, dn = np.char.startswith(stc, "trend_up"), np.char.startswith(stc, "trend_down")
            for td, m in (("涨", up), ("跌", dn), ("中性", ~(up | dn))):
                if m.any():
                    s, n0 = break_acc[f"{td}_{lv.side}"]
                    break_acc[f"{td}_{lv.side}"] = (s + disp[m].sum(), n0 + int(m.sum()))


def _report(acc, break_acc):
    print(f"{'组合':<22}{'n':>8} {'D1沿趋势':>9} {'顺势P50':>9} {'逆势P50':>9}")
    for key in sorted(acc):
        rows = acc[key]
        if len(rows) < MIN_N:
            print(f"{key:<22}{len(rows):>8}  n 不足")
            continue
        a = np.array([x[0] for x in rows])
        fw = np.nanmedian([x[1] for x in rows])
        bw = np.nanmedian([x[2] for x in rows])
        note = " (无条件基线)" if key == "中性" else ""
        print(f"{key:<22}{len(rows):>8} {np.mean(a):>9.1%} {fw:>9.2f} {bw:>9.2f}{note}")
    print(f"\n── 突破延续 (确认突破后 24 根顺突破方向位移, ATR 归一) ──")
    for key in sorted(break_acc):
        s, n0 = break_acc[key]
        if n0 < MIN_N:
            print(f"  {key}: n={n0} 不足")
            continue
        print(f"  {key}: n={n0} 均值 {s / n0:+.3f} ATR")


def diff_block(real, gbm):
    print(f"\n── 净差 (真实−GBM) ──")
    print(f"{'组合':<22}{'n_真':>8}{'n_G':>8} {'ΔD1':>8} {'Δ顺势P50':>9} {'Δ逆势P50':>9}")
    ra, _ = real
    ga, _ = gbm
    for key in sorted(ra):
        if key == "中性":
            continue
        r, g = ra[key], ga[key]
        if len(r) < MIN_N or len(g) < MIN_N:
            print(f"{key:<22}{len(r):>8}{len(g):>8}  n 不足")
            continue
        d1 = np.mean([x[0] for x in r]) - np.mean([x[0] for x in g])
        fw = np.nanmedian([x[1] for x in r]) - np.nanmedian([x[1] for x in g])
        bw = np.nanmedian([x[2] for x in r]) - np.nanmedian([x[2] for x in g])
        print(f"{key:<22}{len(r):>8}{len(g):>8} {d1:>8.1%} {fw:>9.3f} {bw:>9.3f}")
    # 突破延续净差
    print(f"\n── 突破延续净差 (真实−GBM, 均值) ──")
    _, rb = real
    _, gb = gbm
    for key in sorted(rb):
        rs, rn = rb[key]
        gs, gn = gb[key]
        if rn < MIN_N or gn < MIN_N:
            print(f"  {key}: n {rn}/{gn} 不足")
            continue
        d = rs / rn - gs / gn
        print(f"  {key}: Δ均值 {d:+.3f} ATR (n {rn}/{gn})")


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
    dfs = _load(TF)
    real = run_block(dfs, "真实")
    ref = dfs[0]
    sig = np.std(np.diff(np.log(ref["close"].values)))
    rw = [gbm_dataframe(len(ref), sig, seed=7000 + k) for k in range(N_GBM)]
    gbm = run_block(rw, "GBM")
    diff_block(real, gbm)


if __name__ == "__main__":
    main()
