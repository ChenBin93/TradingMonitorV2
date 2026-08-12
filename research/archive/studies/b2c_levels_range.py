#!/usr/bin/env python3
"""B2c 震荡区间内的水平位有效性 (2026-08-03, 无未来函数)

动机: 全行情混合测量"位方向无效"与"震荡区间上下沿有效"的认知违背 —
混入了趋势行情中的触碰 (逆势接刀)。本实验限定震荡区间。

区间定义 (用户确认: c = 成对位 + 波动约束兜底, 带宽 2.5×ATR):
  在触碰时刻 t, 若 上方最近活跃阻力带 R 与 下方最近活跃支撑带 S 同时存在,
  且 R.price - S.price ≤ 2.5×ATR[t] → 视为震荡区间, 该触碰为"区间内触碰"
  否则为"区间外触碰" (对照组)
  (活跃位 = confirm_at ≤ t 且存活 < 600 根)

度量 (描述性为主, 用户确认):
  下沿触碰 (支撑带) 后 24 根向上位移: max(close[t+1..t+24]) - close[t]
  上沿触碰 (阻力带) 后 24 根向下位移: close[t] - min(close[t+1..t+24])
  均按 ATR[t] 归一; 对照: 区间内 vs 区间外 vs 无条件 vs 随机游走
"""
import os
import sys
from bisect import bisect_right

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from market_phase import _atr_series
from research.data_loader import load_candles, verify
from research.levels import cluster_levels, levels_touch_class_all
from research.sim_market import gbm_dataframe
from scipy.stats import skew

W = 24
DEPTH = 0.5
HOLD = 0.5
MT = 3
TOL = 0.5
RANGE_ATR = 2.5
LIFE = 600
RANGE_BARS = 60  # 区间存续: 过去 60 根收盘始终在上下沿之间


def _load():
    data = load_candles(timeframes=("1h",))
    out = []
    for sym, tfs in data.items():
        df = tfs.get("1h")
        if df is None:
            continue
        if verify(df, sym, "1h"):
            continue
        out.append(df)
    return out


def collect(df):
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    atr = _atr_series(df)
    n = len(c)
    lvls = cluster_levels(h, l, atr, min_touch=MT, tolerance_mult=TOL)
    ev = levels_touch_class_all(lvls, c, h, l, atr, DEPTH, W, HOLD)
    sup = sorted([lv for lv in lvls if lv.side == "support"], key=lambda x: x.price)
    res = sorted([lv for lv in lvls if lv.side == "resistance"], key=lambda x: x.price)
    sup_p = np.array([lv.price for lv in sup])
    res_p = np.array([lv.price for lv in res])
    # 过去 RANGE_BARS 根的 close 最小/最大 (滚动窗口, 向量化)
    import pandas as pd
    cs = pd.Series(c)
    past_min = cs.rolling(RANGE_BARS, min_periods=1).min().values
    past_max = cs.rolling(RANGE_BARS, min_periods=1).max().values
    return dict(c=c, h=h, l=l, atr=atr, ev=ev, lvls=lvls, sup=sup, res=res,
                sup_p=sup_p, res_p=res_p, n=n, past_min=past_min, past_max=past_max)


def in_range(levels, prices, other_prices, price, t, side, atr_t):
    """触碰 side 位的时刻 t, 评估区间条件:
    支撑触碰 → 上方最近活跃阻力; 阻力触碰 → 下方最近活跃支撑
    返回 (in_range: bool, 对侧带价 or None)
    """
    if side == "support":
        k = bisect_right(prices, price)
        # 找上方最近活跃阻力
        for i in range(k, len(levels)):
            lv = levels[i]
            if lv.confirm_at <= t < lv.confirm_at + LIFE:
                return (lv.price - price) <= RANGE_ATR * atr_t, lv
    else:
        k = bisect_right(prices, price) - 1
        for i in range(k, -1, -1):
            lv = levels[i]
            if lv.confirm_at <= t < lv.confirm_at + LIFE:
                return (price - lv.price) <= RANGE_ATR * atr_t, lv
    return False, None


def range_alive(conf_arrs, t):
    """位带在过去 RANGE_BARS 根内未被确认突破 (区间存续)"""
    for arr in conf_arrs:
        if arr.size and np.any((arr >= t - RANGE_BARS) & (arr < t)):
            return False
    return True


def run_block(dfs, label=""):
    print(f"═══ B2c 震荡区间水平位 {label}═══\n")
    stats = {"in_sup": [], "out_sup": [], "in_res": [], "out_res": []}
    atr_in = {"in_sup": [], "in_res": []}  # 触碰时刻 ATR (归一化诊断)
    n_in = n_out = 0
    for df in dfs:
        d = collect(df)
        c, atr, n = d["c"], d["atr"], d["n"]
        conf = d["ev"]["confirmed"]  # 每位 confirmed 索引 (对应 lvls 顺序)
        # 触碰索引: 按 lvls 原始顺序 (ev["touch"][i] 对应 lvls[i])
        for i, lv in enumerate(d["lvls"]):
            for t in d["ev"]["touch"][i]:
                if t + W >= n or atr[t] <= 0:
                    continue
                if lv.side == "support":
                    ok, opp = in_range(d["res"], d["res_p"], None, lv.price, t, "support", atr[t])
                    if ok:
                        # 存续: 本带与对侧带过去 60 根内均无确认突破
                        j = d["res"].index(opp)
                        ok = range_alive([conf[i]], t) and range_alive([conf[len(d["sup"]) + j]], t)
                    seg = c[t + 1:t + W + 1]
                    disp = (seg.max() - c[t]) / atr[t]
                    if ok:
                        stats["in_sup"].append(disp)
                        atr_in["in_sup"].append(atr[t])
                        n_in += 1
                    else:
                        stats["out_sup"].append(disp)
                        n_out += 1
                else:
                    ok, opp = in_range(d["sup"], d["sup_p"], None, lv.price, t, "resistance", atr[t])
                    if ok:
                        j = d["sup"].index(opp)
                        ok = range_alive([conf[i]], t) and range_alive([conf[j]], t)
                    seg = c[t + 1:t + W + 1]
                    disp = (c[t] - seg.min()) / atr[t]
                    if ok:
                        stats["in_res"].append(disp)
                        atr_in["in_res"].append(atr[t])
                    else:
                        stats["out_res"].append(disp)
    print(f"  触碰总数 (支撑+阻力): {sum(len(x) for x in stats.values())}")

    def show(name, arr, base=None):
        if len(arr) < 100:
            print(f"    {name}: n={len(arr)} 样本不足")
            return
        line = (f"    {name}: n={len(arr)} 均值 {np.mean(arr):.3f} ATR "
                f"偏度 {skew(arr):.2f} P50 {np.percentile(arr,50):.2f} P90 {np.percentile(arr,90):.2f}")
        if base is not None:
            line += f" | 无条件 {np.mean(base):.3f}"
        print(line)

    # 无条件基线 (向上/向下 24 根位移)
    up_base, dn_base = [], []
    for df in dfs:
        d = collect(df)
        c, atr, n = d["c"], d["atr"], d["n"]
        t = np.arange(60, n - W, 24)
        rows = np.arange(W)[None, :] + 1 + t[:, None]
        seg = c[rows]
        up_base.extend((seg.max(axis=1) - c[t]) / atr[t])
        dn_base.extend((c[t] - seg.min(axis=1)) / atr[t])
    if atr_in["in_sup"]:
        print(f"  触碰时刻 ATR (归一化诊断): 区间内支撑 {np.mean(atr_in['in_sup']):.3f} "
              f"| 区间内阻力 {np.mean(atr_in['in_res']):.3f} (全样本均值 "
              f"{np.mean(np.concatenate([d['atr'] for d in [collect(x) for x in dfs[:1]]])):.3f})")
    print("  下沿触碰 (支撑) 后 24 根向上位移:")
    show("区间内", stats["in_sup"], up_base)
    show("区间外", stats["out_sup"])
    print("  上沿触碰 (阻力) 后 24 根向下位移:")
    show("区间内", stats["in_res"], dn_base)
    show("区间外", stats["out_res"])
    print(f"  区间内触碰占比: {n_in}/{n_in + n_out} ({n_in / max(1, n_in + n_out):.1%})\n")


def main():
    dfs = _load()
    run_block(dfs)
    ref = dfs[0]
    sig = np.std(np.diff(np.log(ref["close"].values)))
    rw = [gbm_dataframe(len(ref), sig, seed=600 + k) for k in range(10)]
    run_block(rw, label="(随机游走对照)")


if __name__ == "__main__":
    main()
