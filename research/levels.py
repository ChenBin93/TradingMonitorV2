#!/usr/bin/env python3
"""关键水平位检测器 (无未来函数) — 用户经验: 波段高低点 + 多次触碰的价格水平

两种定义 (都建, 对比哪种更有行为规律):
  pivot  : 已确认 swing 高低点, 每个 pivot 单独成位
  cluster: pivot 按价格聚类成带 (tolerance × ATR), 聚类内触次 ≥ min_touch

确认时序 (无未来函数核心):
  pivot[j] 在 bar j+K 才可用; 位在"可用 pivot 集合"上聚类, 位出现时刻 = 聚类完成时刻
  触碰 = intrabar (bar low/high 穿过 [price-band, price+band])
  破位 = 收盘确认 (close 越过带外侧, 与结构状态机语义一致)
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from research.structures import K, confirmed_pivots


@dataclass
class Level:
    price: float
    side: str             # support / resistance
    touch_count: int      # 聚类内 pivot 数 (pivot 定义: 触碰次数, 含自身)
    last_touch_idx: int   # 最近一次触碰 (pivot 位置)
    age_bars: int         # 位出现(聚类完成)距当前 bar 的根数
    band: float           # 位带半宽 (pivot 定义: 0)
    kind: str             # pivot / cluster
    confirm_at: int = 0   # 位可用时刻 (聚类完成时刻)


def _df_like(high, low):
    n = len(high)
    return pd.DataFrame({"open": np.zeros(n), "high": high, "low": low,
                         "close": np.zeros(n), "volume": np.zeros(n)})


def confirmed_swings(high, low, k=K):
    """已确认 swing (pos, price, confirm_at=pos+k) — support 与 resistance"""
    ph, pl = confirmed_pivots(_df_like(high, low), k)
    highs = [(int(j), float(high[j]), j + k) for j in np.flatnonzero(ph)]
    lows = [(int(j), float(low[j]), j + k) for j in np.flatnonzero(pl)]
    return highs, lows


def pivot_levels(high, low, k=K):
    """单点位: 每个已确认 swing 一个位 (band=0, 触次=1)"""
    highs, lows = confirmed_swings(high, low, k)
    out = []
    for pos, price, confirm in highs:
        out.append(Level(price, "resistance", 1, pos, -1, 0.0, "pivot", confirm))
    for pos, price, confirm in lows:
        out.append(Level(price, "support", 1, pos, -1, 0.0, "pivot", confirm))
    return out


def cluster_levels(high, low, atr, k=K, tolerance_mult=0.3, min_touch=2):
    """聚类带位: 已确认 pivot 按价格聚类 (tolerance = tolerance_mult × ATR)

    聚类只在使用时刻可用 pivot 上进行; 位出现时刻 = 聚类内最后一 pivot 确认时刻
    band = tolerance / 2 (触次≥min_touch 才有位)
    """
    highs, lows = confirmed_swings(high, low, k)
    atr = np.asarray(atr, float)
    out = []
    for side, swings in [("resistance", highs), ("support", lows)]:
        if not swings:
            continue
        tol = np.nanmedian(atr[swings[0][0]:swings[-1][0] + 1]) * tolerance_mult
        tol = max(tol, 1e-9)
        swings.sort(key=lambda x: x[1])
        clusters = []
        cur = [swings[0]]
        for s in swings[1:]:
            if s[1] - cur[-1][1] <= tol:
                cur.append(s)
            else:
                clusters.append(cur)
                cur = [s]
        clusters.append(cur)
        for cl in clusters:
            if len(cl) < min_touch:
                continue
            prices = [p for _, p, _ in cl]
            last_pos = max(pos for pos, _, _ in cl)
            confirm = max(c for _, _, c in cl)
            out.append(Level(
                price=float(np.median(prices)),
                side=side,
                touch_count=len(cl),
                last_touch_idx=last_pos,
                age_bars=-1,
                band=tol / 2.0,
                kind="cluster",
                confirm_at=confirm,
            ))
    out.sort(key=lambda l: l.confirm_at)
    return out


def active_levels(levels, t):
    """bar t 时可用的位 (confirm_at <= t), age 已更新"""
    out = []
    for lv in levels:
        if lv.confirm_at <= t:
            lv.age_bars = t - lv.last_touch_idx
            out.append(lv)
    return out


def touches_at(levels, t, low, high):
    """bar t 的 intrabar 触碰事件: 返回触碰的位列表

    触碰 = bar low/high 穿过 [price-band, price+band]
    """
    hit = []
    lo, hi = low[t], high[t]
    for lv in levels:
        if lv.confirm_at <= t:
            p_lo = lv.price - lv.band
            p_hi = lv.price + lv.band
            if lo <= p_hi and hi >= p_lo:
                hit.append(lv)
    return hit


def close_breakout(levels, t, close):
    """收盘确认破位: 返回刚被破位的位 (support: close < price-band; resistance: close > price+band)"""
    broken = []
    c = close[t]
    for lv in levels:
        if lv.confirm_at <= t:
            p_lo = lv.price - lv.band
            p_hi = lv.price + lv.band
            if lv.side == "support" and c < p_lo:
                broken.append(lv)
            elif lv.side == "resistance" and c > p_hi:
                broken.append(lv)
    return broken


def nearest_levels(levels, t, price):
    """当前价格下方最近支撑 / 上方最近阻力 (可用位)"""
    sup = res = None
    best_d = np.inf
    for lv in levels:
        if lv.confirm_at > t:
            continue
        d = price - lv.price
        if lv.side == "support" and 0 <= d < best_d:
            sup, best_d = lv, d
    best_d = np.inf
    for lv in levels:
        if lv.confirm_at > t:
            continue
        d = lv.price - price
        if lv.side == "resistance" and 0 <= d < best_d:
            res, best_d = lv, d
    return sup, res
