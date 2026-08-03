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
    """聚类带位 — 在线聚类 + 冻结语义 (2026-08-03 修正, 消除未来函数)

    此前版本用全样本 pivot 离线聚类: 未来 pivot 会改变位带价格/带宽并把
    confirm_at 拉长到未来 (已验证: t=30000 时全样本位带全部不可用)。
    修正后:
      - pivot 按确认时序逐个处理
      - 位带在达到 min_touch 时形成, 价格 = 当时聚类中位数, 冻结不变
      - 后续同价区 pivot 只并入触次, 不改变位带价格/确认时间
      - 无未来函数: 位带只由"当时已确认"的 pivot 决定, 一旦形成结构固定
    """
    highs, lows = confirmed_swings(high, low, k)
    atr = np.asarray(atr, float)
    events = []
    for pos, price, confirm in highs:
        events.append((confirm, pos, price, "resistance"))
    for pos, price, confirm in lows:
        events.append((confirm, pos, price, "support"))
    events.sort()

    formed = []   # 已冻结位带 (cluster), 按 (side, price) 排序维护
    pending = []  # 未形成组 [side, prices, last_pos]

    def find_near(price, side, tol):
        """二分查找同侧 price ± tol 内的已形成位带"""
        lo, hi = 0, len(formed)
        while lo < hi:
            mid = (lo + hi) // 2
            if formed[mid].side < side or (formed[mid].side == side and formed[mid].price < price):
                lo = mid + 1
            else:
                hi = mid
        # 从插入点向两侧扫 ±tol
        i = lo
        while i < len(formed) and formed[i].side == side and formed[i].price <= price + tol:
            if abs(formed[i].price - price) <= tol:
                return formed[i]
            i += 1
        i = lo - 1
        while i >= 0 and formed[i].side == side and formed[i].price >= price - tol:
            if abs(formed[i].price - price) <= tol:
                return formed[i]
            i -= 1
        return None

    def insert_formed(lv):
        """按 (side, price) 插入已形成位带 (保持排序)"""
        lo, hi = 0, len(formed)
        while lo < hi:
            mid = (lo + hi) // 2
            if formed[mid].side < lv.side or (formed[mid].side == lv.side and formed[mid].price < lv.price):
                lo = mid + 1
            else:
                hi = mid
        formed.insert(lo, lv)

    for confirm, pos, price, side in events:
        tol = atr[min(confirm, len(atr) - 1)] * tolerance_mult if len(atr) else 1e-9
        tol = max(tol, 1e-9)
        # 1) 并入已形成位带 (价格/触次冻结, 只记最近触碰 — 触次含未来信息, 不累加)
        lv = find_near(price, side, tol)
        if lv is not None:
            lv.last_touch_idx = max(lv.last_touch_idx, pos)
            continue
        # 2) 尝试并入未形成组
        merged = False
        for grp in pending:
            if grp[0] == side and abs(float(np.median(grp[1])) - price) <= tol:
                grp[1].append(price)
                grp[2] = max(grp[2], pos)
                if len(grp[1]) >= min_touch:
                    insert_formed(Level(
                        price=float(np.median(grp[1])),
                        side=side,
                        touch_count=len(grp[1]),
                        last_touch_idx=grp[2],
                        age_bars=-1,
                        band=tol / 2.0,
                        kind="cluster",
                        confirm_at=confirm,
                    ))
                    pending.remove(grp)
                merged = True
                break
        if not merged:
            pending.append([side, [price], pos])
    return formed


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
