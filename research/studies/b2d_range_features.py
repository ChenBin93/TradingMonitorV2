#!/usr/bin/env python3
"""B2d 震荡行情特征描述 + 区间回归倾向 (2026-08-03, 无未来函数)

第一部分 — 震荡行情特征 (纯描述):
  D1 区间活跃占比 / 区间段存续时长分布
  D2 区间宽度分布 (ATR 归一化)
  D3 区间内 ATR 水平 vs 无条件 (同池相对比率, 修正 B2c 归一化缺陷)

第二部分 — 区间回归倾向 (直接度量, 不用 1:1 胜率):
  R1 触碰下沿后 24 根内收盘回到区间中位线的概率与平均用时
  R2 触碰后 24 根内反弹覆盖区间宽度的比例
  R3 触碰后 24 根内收盘回到区间内侧 (带内) 的概率
  对照: 随机游走 (真实 OHLC 子步)

区间定义 (与用户确认): 上下成对活跃位带 间距 ≤2.5×ATR 且双方过去
60 根内无确认突破 (B2 突破定义: 穿透 0.5×ATR + 24 根外侧≥50%)
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

W = 24
DEPTH = 0.5
HOLD = 0.5
MT = 3
TOL = 0.5
RANGE_ATR = 2.5
LIFE = 600
RANGE_BARS = 60


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
    return dict(c=c, h=h, l=l, atr=atr, ev=ev, lvls=lvls, sup=sup, res=res, n=n)


def range_state(d, t, atr_t):
    """bar t 的区间状态: (S, R) 或 None — 上下最近活跃位, 间距 ≤2.5×ATR"""
    sup_p = np.array([lv.price for lv in d["sup"]])
    res_p = np.array([lv.price for lv in d["res"]])
    # 下方最近活跃支撑
    k = bisect_right(sup_p, d["c"][t]) - 1
    S = None
    for i in range(k, -1, -1):
        lv = d["sup"][i]
        if lv.confirm_at <= t < lv.confirm_at + LIFE:
            S = lv
            break
    # 上方最近活跃阻力
    k = bisect_right(res_p, d["c"][t])
    R = None
    for i in range(k, len(d["res"])):
        lv = d["res"][i]
        if lv.confirm_at <= t < lv.confirm_at + LIFE:
            R = lv
            break
    if S is None or R is None:
        return None
    if R.price - S.price > RANGE_ATR * atr_t:
        return None
    return S, R


def range_alive(conf_arrs, t):
    for arr in conf_arrs:
        if arr.size and np.any((arr >= t - RANGE_BARS) & (arr < t)):
            return False
    return True


def run_block(dfs, label=""):
    print(f"═══ B2d 震荡行情特征与区间回归 {label}═══\n")
    # 每标的预处理: 逐 bar 区间状态 (预计算, 提升触碰评估性能)
    syms = []
    for df in dfs:
        d = collect(df)
        n = d["n"]
        sup_p = np.array([lv.price for lv in d["sup"]])
        res_p = np.array([lv.price for lv in d["res"]])
        # 逐 bar 区间状态 (bisect 快路径)
        active = np.zeros(n, bool)
        width = np.full(n, np.nan)
        mid = np.full(n, np.nan)
        s_price = np.full(n, np.nan)
        r_price = np.full(n, np.nan)
        for t in range(n):
            st = range_state(d, t, d["atr"][t])
            if st is not None:
                S, R = st
                active[t] = True
                s_price[t], r_price[t] = S.price, R.price
                width[t] = R.price - S.price
                mid[t] = (S.price + R.price) / 2
        syms.append(dict(d=d, active=active, width=width, mid=mid,
                         s_price=s_price, r_price=r_price))
    n_sym = len(syms)
    n = syms[0]["active"].shape[0]

    # ── D1 区间活跃占比 / 存续时长 ──
    act_all = np.concatenate([s["active"] for s in syms])
    print("D1 区间活跃:")
    print(f"  活跃占比: {act_all.mean():.1%}")
    lens = []
    for s in syms:
        a = s["active"]
        cur = 0
        for v in a:
            if v:
                cur += 1
            else:
                if cur >= 10:
                    lens.append(cur)
                cur = 0
        if cur >= 10:
            lens.append(cur)
    if lens:
        print(f"  区间段存续时长: 中位 {np.median(lens):.0f} 根, "
              f"P90 {np.percentile(lens, 90):.0f} 根 (段数 {len(lens)})")
    print()

    # ── D2 区间宽度分布 ──
    width_all = np.concatenate([s["width"][s["active"]] for s in syms])
    atr_all = np.concatenate([s["d"]["atr"][s["active"]] for s in syms])
    if len(width_all):
        wr = width_all / atr_all
        print(f"D2 区间宽度 (ATR 归一化): 中位 {np.median(wr):.2f} ATR, "
              f"P25 {np.percentile(wr, 25):.2f}, P75 {np.percentile(wr, 75):.2f}")
    print()

    # ── D3 区间内 ATR vs 无条件 (同池) ──
    rat = []
    for s in syms:
        a = s["d"]["atr"]
        in_r = a[s["active"]]
        uncond = a[60:len(a) - 24:24]
        if len(in_r) and len(uncond):
            rat.append(np.mean(in_r) / np.mean(uncond))
    if rat:
        print(f"D3 区间内 ATR / 无条件 ATR: {np.mean(rat):.3f} (同标的比率)")
    print()

    # ── R1/R2/R3 区间内触碰回归倾向 ──
    print("R 区间内触碰后的回归倾向 (触碰时刻区间条件 = B2c):")
    r1_prob, r1_time, r2_cover, r3_in = [], [], [], []
    for s in syms:
        d, active, width, mid, s_price, r_price = (s["d"], s["active"], s["width"],
                                                   s["mid"], s["s_price"], s["r_price"])
        c, atr, n = d["c"], d["atr"], d["n"]
        conf = d["ev"]["confirmed"]
        for i, lv in enumerate(d["lvls"]):
            for t in d["ev"]["touch"][i]:
                if not active[t] or t + W >= n or atr[t] <= 0:
                    continue
                if not range_alive([conf[i]], t):
                    continue
                # 对侧位存续
                if lv.side == "support":
                    j = next((k for k, x in enumerate(d["res"]) if x.price == r_price[t]), None)
                else:
                    j = next((k for k, x in enumerate(d["sup"]) if x.price == s_price[t]), None)
                if j is None or not range_alive([conf[len(d["sup"]) + j] if lv.side == "support" else conf[j]], t):
                    continue
                m = mid[t]
                if lv.side == "support":
                    # 触碰下沿: 回到中位线 = close ≥ m
                    seg = c[t + 1:t + W + 1]
                    hit = np.where(seg >= m)[0]
                    r1_prob.append(1 if len(hit) else 0)
                    r1_time.append(hit[0] + 1 if len(hit) else np.nan)
                    r2_cover.append((seg.max() - c[t]) / width[t])
                    r3_in.append(1 if seg[-1] > lv.price else 0)
                else:
                    seg = c[t + 1:t + W + 1]
                    hit = np.where(seg <= m)[0]
                    r1_prob.append(1 if len(hit) else 0)
                    r1_time.append(hit[0] + 1 if len(hit) else np.nan)
                    r2_cover.append((c[t] - seg.min()) / width[t])
                    r3_in.append(1 if seg[-1] < lv.price else 0)
    if r1_prob:
        r1p = np.mean(r1_prob)
        r1t = np.nanmean(r1_time)
        r2c = np.mean(r2_cover)
        r3i = np.mean(r3_in)
        print(f"  R1 触碰后 24 根内回到区间中位线: {r1p:.1%} (平均用时 {r1t:.1f} 根, n={len(r1_prob)})")
        print(f"  R2 触碰后 24 根内反弹覆盖区间宽度: {r2c:.1%}")
        print(f"  R3 触碰后 24 根收盘仍在区间内侧: {r3i:.1%}")
    print()


def main():
    dfs = _load()
    run_block(dfs)
    ref = dfs[0]
    sig = np.std(np.diff(np.log(ref["close"].values)))
    rw = [gbm_dataframe(len(ref), sig, seed=700 + k) for k in range(10)]
    run_block(rw, label="(随机游走对照)")


if __name__ == "__main__":
    main()
