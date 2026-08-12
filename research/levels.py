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
import bisect
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from research.structures import K, confirmed_pivots


@dataclass(frozen=True)
class Level:
    """关键位 (R1: 形成后字段不可变)

    触碰事件不写入字段, 而是追加到内部事件日志 `_touches`
    (list[(confirm_at, pos)], 按 confirm 升序)。last_touch_idx/age_bars 只是
    "形成时刻"的静态初值; active_levels(levels, t) 用 bisect 从日志重建 t 时刻
    快照, 因此追加数据不改变任何历史 t 的快照 (无未来函数)。
    """
    price: float
    side: str             # support / resistance
    touch_count: int      # 聚类内 pivot 数 (pivot 定义: 触碰次数, 含自身)
    last_touch_idx: int   # 形成时刻的最近触碰 (pivot 位置; 快照按日志重建)
    age_bars: int         # 形成时刻位出现(聚类完成)距当前 bar 的根数
    band: float           # 位带半宽 (pivot 定义: 0)
    kind: str             # pivot / cluster
    confirm_at: int = 0   # 位可用时刻 (聚类完成时刻)
    _touches: list = field(default_factory=list, repr=False, compare=False)


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
    pending = []  # 未形成组 [side, prices, events]; events: [(confirm, pos)] 升序

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
        # 1) 并入已形成位带 (价格/触次冻结; 触碰只进事件日志, 不突变字段 — R1)
        lv = find_near(price, side, tol)
        if lv is not None:
            lv._touches.append((confirm, pos))
            continue
        # 2) 尝试并入未形成组
        merged = False
        for grp in pending:
            if grp[0] == side and abs(float(np.median(grp[1])) - price) <= tol:
                grp[1].append(price)
                grp[2].append((confirm, pos))
                if len(grp[1]) >= min_touch:
                    insert_formed(Level(
                        price=float(np.median(grp[1])),
                        side=side,
                        touch_count=len(grp[1]),
                        last_touch_idx=grp[2][-1][1],
                        age_bars=-1,
                        band=tol / 2.0,
                        kind="cluster",
                        confirm_at=confirm,
                        _touches=grp[2],
                    ))
                    pending.remove(grp)
                merged = True
                break
        if not merged:
            pending.append([side, [price], [(confirm, pos)]])
    return formed


def _last_touch(lv, t):
    """confirm<=t 的最近触碰 pos — 事件日志 bisect (R1)"""
    if not lv._touches:
        return lv.last_touch_idx
    i = bisect.bisect_right([c for c, _ in lv._touches], t) - 1
    if i >= 0:
        return lv._touches[i][1]
    return -1


def active_levels(levels, t):
    """bar t 时可用的位 (confirm_at <= t), age 由事件日志按确认时序重建快照

    R1: last_touch_idx/age_bars 只反映 confirm<=t 的触碰事件 (bisect 从 _touches
    重建), 返回副本而非原地突变 — 追加数据不改变历史 t 的快照。
    """
    out = []
    for lv in levels:
        if lv.confirm_at <= t:
            last_pos = _last_touch(lv, t)
            out.append(replace(lv, last_touch_idx=last_pos, age_bars=t - last_pos))
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


def level_breakdown(lv, close, atr, depth=0.5, w=24, hold_ratio=0.5):
    """单位的穿透/确认向量 (向量化, 事后标签 — 合法使用未来数据)

    穿透 attempt: close 越出位带外侧 ≥ depth×ATR (穿透时刻的 ATR)
    确认 confirmed: 穿透后未来 w 根内 close 保持在外侧的比例 ≥ hold_ratio
    外侧 outside: close 越过位带外沿 (price ± band)
    R2: attempt/confirmed 只对 t >= confirm_at (位带形成后) 判定 —
        形成之前的"突破"不计入; outside 保持纯描述语义不门控

    返回 (attempt, confirmed, outside, ratio) 布尔/浮点数组 (长度 n)
    """
    n = len(close)
    idx = np.arange(n)
    usable = idx >= lv.confirm_at  # R2 门控
    p_lo = lv.price - lv.band
    p_hi = lv.price + lv.band
    if lv.side == "support":
        outside = close < p_lo
        attempt = (close < p_lo - depth * atr) & usable
    else:
        outside = close > p_hi
        attempt = (close > p_hi + depth * atr) & usable
    # 未来 [t+1, t+w] 的外侧比例 (后缀和差分; 越界处按 0 处理)
    suffix = np.concatenate([outside[::-1].cumsum()[::-1], np.zeros(1)])
    s_next = np.zeros(n)
    s_next[:n - 1] = suffix[1:n]
    s_end = np.zeros(n)
    end_idx = idx + (w + 1)
    valid = end_idx <= n
    s_end[valid] = suffix[end_idx[valid]]
    cnt = s_next - s_end
    ratio = cnt / w
    confirmed = attempt & (ratio >= hold_ratio)
    return attempt, confirmed, outside, ratio


def level_touch_class(lv, close, high, low, atr, depth=0.5, w=24, hold_ratio=0.5):
    """触碰进入事件 × 后续判定 (向量化)

    触碰进入: intrabar 触及位带且前一根未触及 (confirm_at 后)
    判定 (触碰后 w 根内):
      breakout: 发生确认突破 (confirmed) → 位失效
      reject:   无确认突破 → 有效拒绝 (位保持)
    假突破 (attempt 但未 confirmed) 单独统计 (reject 的子集)

    返回 dict: touch(进入 idx), reject(touch 被拒绝), breakout(touch 被突破),
               attempt(穿透 idx), confirmed(真突破 idx), false_break(假穿透 idx)
    """
    n = len(close)
    t_arr = np.arange(n)
    usable = t_arr >= lv.confirm_at
    p_lo = lv.price - lv.band
    p_hi = lv.price + lv.band
    tm = (low <= p_hi) & (high >= p_lo) & usable
    touch = tm.copy()
    touch[1:] &= ~tm[:-1]
    attempt, confirmed, outside, ratio = level_breakdown(lv, close, atr, depth, w, hold_ratio)
    # 触碰后 w 根内是否有确认突破
    suffix_conf = np.concatenate([confirmed[::-1].cumsum()[::-1], np.zeros(1)])
    s_next = np.zeros(n)
    s_next[:n - 1] = suffix_conf[1:n]
    s_end = np.zeros(n)
    end_idx = np.arange(n) + (w + 1)
    valid = end_idx <= n
    s_end[valid] = suffix_conf[end_idx[valid]]
    has_break = (s_next - s_end) > 0
    # 触碰后被突破: 触碰 t 且 [t+1, t+w] 内有 confirmed
    t_break = touch & has_break
    t_reject = touch & ~has_break
    # 穿透分类 (独立事件): attempt 中 confirmed 为真突破, 否则假突破
    false_break = attempt & ~confirmed
    return {
        "touch": np.flatnonzero(touch),
        "reject": np.flatnonzero(t_reject),
        "breakout": np.flatnonzero(t_break),
        "attempt": np.flatnonzero(attempt),
        "confirmed": np.flatnonzero(confirmed),
        "false_break": np.flatnonzero(false_break),
    }


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
