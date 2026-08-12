#!/usr/bin/env python3
"""B3d 区间延续性直接检验 (2026-08-04, 无未来函数)

动机 (用户逻辑): 震荡有延续性 → 触碰水平关键位后更倾向回到区间内部 =
方向性。若数据与逻辑矛盾, 先查数据获取方式 (窗口尺度/参照系/条件选择)。

设计 (三处数据环节修正 + 趋势分层):
  P1 窗口梯度: 触碰后 w 根 (w=6/12/24) "收盘在区间内部 (S.price, R.price)"
     的根数占比 — 用户"回到区间内部"的直接度量; 修正 B3 用 W=24 单窗口
     与区间中位存续 15 根不匹配的问题
  P2 剩余存续: 触碰时刻 → 区间状态连续存活的根数, 真实 vs GBM (条件化
     延续性直接检验)
  P3 全体对照: 区间内所有 bar (非触碰为主) 的 w 根留存 — 分离"区间延续性"
     与"触碰事件的选择效应" (触碰=状态破坏前兆?)
  P4 趋势分层: 触碰时刻过去 120 根趋势方向 (涨/跌/中性) × 区间内触碰的
     反向方向概率 (触下沿后 P(c[t+24]>c[t]), 触上沿后 P(<)) — 回答
     "下跌中继做空/上涨中继做多"是否成立

对照: GBM 30 种子; 无偏性门禁: GBM 条件组 n<2000 标 † 仅净差

运行: python3 research/studies/b3d_range_persistence.py (~2h)
"""
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.data_loader import load_candles, verify
from research.sim_market import gbm_dataframe
from research.studies.b3_general_levels import (collect as b3collect, bucket_key,
                                                W, MIN_N)

PARAMS = [(2, 0.3), (3, 0.5)]
TIMEFRAMES = ["1h", "4h"]
N_GBM = 30
WS = (6, 12, 24)
TREND = 120
WMAX = 24


def run_block(dfs, mt, tol, label=""):
    print(f"\n{'═' * 70}\nB3d {label} 参数(min_touch={mt}, tol={tol})\n{'═' * 70}")
    touch = {w: defaultdict(list) for w in WS}      # 触碰后 w 根留存比例
    allbar = {w: defaultdict(list) for w in WS}     # 全体区间内 bar 留存比例
    remain = []                                      # P2 剩余存续
    trend = {k: {"n": [], "rev": []} for k in ("涨", "跌", "中性")}
    for df in dfs:
        d = b3collect(df, mt, tol)
        c, atr, n, active = d["c"], d["atr"], d["n"], d["active"]
        s_p = np.array([np.nan if d["s_obj"][t] is None else d["s_obj"][t].price
                        for t in range(n)])
        r_p = np.array([np.nan if d["r_obj"][t] is None else d["r_obj"][t].price
                        for t in range(n)])
        # P2: 剩余存续 (从后往前扫 active 连续段)
        remain_len = np.zeros(n)
        for t in range(n - 1, -1, -1):
            if active[t]:
                remain_len[t] = remain_len[t + 1] + 1 if t + 1 < n else 1
        # P3: 全体区间内 bar (采样步长 5)
        slice_ok = active[:n - WMAX:5] & ~np.isnan(s_p[:n - WMAX:5])
        idx = np.flatnonzero(slice_ok) * 5
        for tt in idx:
            S, R = s_p[tt], r_p[tt]
            for w in WS:
                seg = c[tt + 1:tt + w + 1]
                allbar[w]["n"].append(1)
                allbar[w]["v"].append(np.mean((seg > S) & (seg < R)))
        # 触碰事件
        for i, lv in enumerate(d["lvls"]):
            t_arr = d["ev"]["touch"][i]
            valid = (t_arr + WMAX < n) & (t_arr >= TREND) & (atr[t_arr] > 0)
            t_arr = t_arr[valid]
            if not len(t_arr):
                continue
            seg24 = c[t_arr[:, None] + np.arange(1, WMAX + 1)]
            for j, t in enumerate(t_arr):
                b = bucket_key({"t": int(t), "lv": lv, "d": d, "seq": j + 1})
                if b["structure"] != "区间内":
                    continue
                S, R = s_p[t], r_p[t]
                if np.isnan(S) or np.isnan(R):
                    continue
                remain.append(float(remain_len[t]))
                for w in WS:
                    seg = c[t + 1:t + w + 1]
                    touch[w]["n"].append(1)
                    touch[w]["v"].append(np.mean((seg > S) & (seg < R)))
                # P4: 趋势方向 + 反向概率
                trend_dir = np.sign(np.log(c[t] / c[t - TREND]))
                if lv.side == "support":
                    rev = 1.0 if seg24[j, -1] > c[t] else 0.0
                else:
                    rev = 1.0 if seg24[j, -1] < c[t] else 0.0
                key = {1: "涨", -1: "跌", 0: "中性"}[int(trend_dir)]
                trend[key]["n"].append(1)
                trend[key]["rev"].append(rev)
    # ── 输出 P1/P3 (明细在 diff_block) ──
    for w in WS:
        tr, ar = touch[w], allbar[w]
        tv, av = np.mean(tr["v"]), np.mean(ar["v"])
        tn, an = len(tr["v"]), len(ar["v"])
        print(f"  w={w:>2}: 触碰留存 {tv:.1%} (n={tn}) | 全体区间bar {av:.1%} (n={an})")
    return dict(touch=touch, allbar=allbar, remain=remain, trend=trend)


def run_block_gbm(dfs, mt, tol, label=""):
    res = run_block(dfs, mt, tol, label)
    return res


def diff_block(real, gbm, label=""):
    print(f"\n── 净差 (真实−GBM) {label} ──")
    print(f"{'w':>3} {'触碰n真':>8}{'触碰nG':>8} {'Δ触碰':>8} {'全体n真':>8}"
          f"{'全体nG':>8} {'Δ全体':>8}")
    for w in WS:
        tr, ar = real["touch"][w], gbm["touch"][w]
        ta, aa = real["allbar"][w], gbm["allbar"][w]
        tn, gn = len(tr["v"]), len(ar["v"])
        tan, gan = len(ta["v"]), len(aa["v"])
        dt = np.mean(tr["v"]) - np.mean(ar["v"]) if (tn >= MIN_N and gn >= MIN_N) else float("nan")
        da = np.mean(ta["v"]) - np.mean(aa["v"]) if (tan >= MIN_N and gan >= MIN_N) else float("nan")
        print(f"{w:>3} {tn:>8} {gn:>8} {dt:>8.1%} {tan:>8} {gan:>8} {da:>8.1%}")
    # P2 剩余存续
    rr, rg = np.array(real["remain"]), np.array(gbm["remain"])
    if len(rr) >= MIN_N and len(rg) >= MIN_N:
        print(f"P2 触碰后剩余存续 (根): 真实 中位 {np.median(rr):.0f} 均值 {rr.mean():.1f} "
              f"| GBM 中位 {np.median(rg):.0f} 均值 {rg.mean():.1f} "
              f"| Δ均值 {rr.mean() - rg.mean():+.1f} (n 真 {len(rr)}, GBM {len(rg)})")
    # P4 趋势分层
    print("P4 趋势 × 区间内触碰反向概率 (真实 | GBM | Δ):")
    for k in ("涨", "跌", "中性"):
        rt, gt = real["trend"][k], gbm["trend"][k]
        rn, gn2 = len(rt["rev"]), len(gt["rev"])
        if rn < MIN_N or gn2 < MIN_N:
            print(f"  {k}: n 真 {rn} / GBM {gn2} 不足")
            continue
        rv, gv = np.mean(rt["rev"]), np.mean(gt["rev"])
        print(f"  {k}: {rv:.1%} | {gv:.1%} | Δ {rv - gv:+.1%}")


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
            rw = [gbm_dataframe(len(ref), sig, seed=3000 + (tf == "4h") * 1000 + k)
                  for k in range(N_GBM)]
            gbm = run_block_gbm(rw, mt, tol, f"GBM {tf}")
            diff_block(real, gbm, f"{tf} 参数({mt},{tol})")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
