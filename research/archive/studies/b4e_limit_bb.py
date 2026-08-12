#!/usr/bin/env python3
"""B4e limit 入场 + 布林带优化 (2026-08-04, 无未来函数, 多空一起)

动机 (用户):
  1. 高空低多的关键是在尽可能高/低的点下单 — B4 的收盘追价入场极差
     (触下沿收盘比极值高 ~0.5 ATR); 改为 limit 挂单 (成交价=位带中心)
  2. 均值回归可结合布林带 (标准 20,2) 优化

设计:
  A limit 入场: 区间内触碰 (B3d 定义), 触下沿 → limit buy 挂位带中心
     (S.price), low ≤ S.price 成交, 成交价 = S.price; 触上沿对称 (R.price)
     成交率单独统计 (挂单未成交不计入)
     判定从成交后下一根 open 开始 (open 出发), 目标 0.3×ATR / 止损
     0.7/1.0×ATR, W=6 timeout — 参数沿用 B4 已验证配置
  B 布林带位置层 (触碰时刻 close[t] vs 布林带, 标准 20,2):
     触下沿: <下轨=极值 | 下轨~中轨=中间 | >中轨=回收
     触上沿: >上轨=极值 | 中轨~上轨=中间 | <中轨=回收
  C squeeze 层: 触碰时刻布林带宽度分位 (该标的 P33/P67): 窄/中/宽

对照: GBM 30 种子 (同参数); 输出净差 + 分年 (最优层)。
运行: python3 research/studies/b4e_limit_bb.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.data_loader import load_candles, verify
from research.sim_market import gbm_dataframe
from research.studies.b3_general_levels import (collect as b3collect, bucket_key,
                                                MIN_N)
from research.studies.b4_small_target import _sim

MT, TOL = 2, 0.3
TF = "1h"
N_GBM = 30
TARGET, W = 0.3, 6
STOPS = (0.7, 1.0)


def bollinger(c):
    s = pd.Series(c)
    ma = s.rolling(20).mean()
    sd = s.rolling(20).std()
    return (ma.values, (ma + 2 * sd).values, (ma - 2 * sd).values,
            (sd / ma).values)


def run_block(dfs, label=""):
    print(f"\n{'═' * 76}\nB4e {label} (1h, mt={MT}, tol={TOL})\n{'═' * 76}")
    acc = {}   # (side, T, S, pos, sq) -> list of (year, outcome)
    fill_rate = {"support": [], "resistance": []}
    for df in dfs:
        d = b3collect(df, MT, TOL)
        c, h, l, atr, n = d["c"], d["h"], d["l"], d["atr"], d["n"]
        o = df["open"].values
        idx = df.index
        ma, up, lo, bbw = bollinger(c)
        q33, q67 = np.nanpercentile(bbw[50:], [33, 67])
        for i, lv in enumerate(d["lvls"]):
            side = lv.side
            t_arr = d["ev"]["touch"][i]
            valid = (t_arr + W < n) & (t_arr >= 20) & (atr[t_arr] > 0)
            t_arr = t_arr[valid]
            if not len(t_arr):
                continue
            keys = np.array([bucket_key({"t": int(t), "lv": lv, "d": d, "seq": j + 1})["structure"] == "区间内"
                             for j, t in enumerate(t_arr)])
            t_arr = t_arr[keys]
            if not len(t_arr):
                continue
            # limit 成交: 触下沿 buy at S.price (low<=price); 触上沿 sell at R.price
            if side == "support":
                fill = l[t_arr] <= lv.price
            else:
                fill = h[t_arr] >= lv.price
            fill_rate[side].append(fill.mean())
            ft = t_arr[fill]
            if not len(ft):
                continue
            entry = lv.price
            atr_t = atr[ft]
            for S in STOPS:
                if side == "support":
                    target = entry + TARGET * atr_t
                    stop = entry - S * atr_t
                else:
                    target = entry - TARGET * atr_t
                    stop = entry + S * atr_t
                out = np.array(_sim(ft, o, h, l, target, stop, W))
                # 位置层 / squeeze 层
                if side == "support":
                    pos = np.where(c[ft] < lo[ft], "极值",
                                   np.where(c[ft] < ma[ft], "中间", "回收"))
                else:
                    pos = np.where(c[ft] > up[ft], "极值",
                                   np.where(c[ft] > ma[ft], "中间", "回收"))
                sq = np.where(bbw[ft] < q33, "窄",
                              np.where(bbw[ft] > q67, "宽", "中"))
                for j, t in enumerate(ft):
                    y = idx[t].year
                    acc.setdefault((side, S, pos[j], sq[j]), []).append((y, int(out[j])))
                acc.setdefault((side, S, "全体", "全体"), []).extend(
                    [(idx[t].year, int(out[j])) for j, t in enumerate(ft)])
    fr_l = np.mean(fill_rate["support"]) if fill_rate["support"] else float("nan")
    fr_s = np.mean(fill_rate["resistance"]) if fill_rate["resistance"] else float("nan")
    print(f"limit 成交率: 触下沿 {fr_l:.1%} | 触上沿 {fr_s:.1%}")
    _report(acc, "全体")
    _report(acc, "pos")
    _report(acc, "sq")
    return acc


def _report(acc, dim):
    print(f"\n── 分层: {dim} ──")
    print(f"{'side':<6} {'S':>4} {'层':<6} {'n':>7} {'胜率':>8} {'期望':>8}")
    for (side, S, pos, sq), rows in sorted(acc.items()):
        if dim == "pos" and (pos == "全体" or sq != "全体"):
            continue
        if dim == "sq" and (sq == "全体" or pos != "全体"):
            continue
        if dim == "全体" and (pos != "全体" or sq != "全体"):
            continue
        a = np.array([x[1] for x in rows], dtype=float)
        h2 = a != 0
        wr = np.mean(a[h2] > 0) if h2.any() else float("nan")
        print(f"{side:<6} {S:>4} {pos if dim=='pos' else sq:<6} "
              f"{len(a):>7} {wr:>8.1%} {a.mean():>+8.4f}")


def diff_block(real, gbm):
    print(f"\n── 净差 (真实−GBM) ──")
    print(f"{'side':<6} {'S':>4} {'层':<6} {'Δ胜率':>8} {'Δ期望':>9}")
    for (side, S, pos, sq), rows in real.items():
        g = gbm.get((side, S, pos, sq))
        if g is None:
            continue
        r = np.array([x[1] for x in rows], dtype=float)
        gg = np.array([x[1] for x in g], dtype=float)
        if len(r) < MIN_N or len(gg) < MIN_N:
            print(f"{side:<6} {S:>4} {pos if pos!='全体' else sq:<6} "
                  f"{len(r):>4}/{len(gg):<4} n 不足")
            continue
        rh, gh = r[r != 0], gg[gg != 0]
        layer = pos if pos != "全体" else sq
        print(f"{side:<6} {S:>4} {layer:<6} "
              f"{np.mean(rh > 0) - np.mean(gh > 0):>+8.1%} {r.mean() - gg.mean():>+9.4f}")


def years(real, gbm):
    print(f"\n── 分年 (多空合并, S=0.7/1.0, 全体层) ──")
    for S in STOPS:
        for side in ("support", "resistance"):
            r = [x for (s2, S2, p, q), rows in real.items()
                 if s2 == side and S2 == S and p == "全体" for x in rows]
            g = [x for (s2, S2, p, q), rows in gbm.items()
                 if s2 == side and S2 == S and p == "全体" for x in rows]
            print(f"  {side} S={S}:", end="")
            for y in sorted(set(x[0] for x in r)):
                ra = np.array([x[1] for x in r if x[0] == y], dtype=float)
                ga = np.array([x[1] for x in g if x[0] == y], dtype=float)
                dh = (np.mean(ra[ra != 0] > 0) - np.mean(ga[ga != 0] > 0)
                      if len(ra) >= 100 and len(ga) >= 100 else float("nan"))
                print(f" {y}:Δ胜率{dh:+.1%}", end="")
            print()


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
    rw = [gbm_dataframe(len(ref), sig, seed=6000 + k) for k in range(N_GBM)]
    gbm = run_block(rw, "GBM")
    diff_block(real, gbm)
    years(real, gbm)


if __name__ == "__main__":
    main()
