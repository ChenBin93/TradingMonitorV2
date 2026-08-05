#!/usr/bin/env python3
"""多周期关键位监控 — 关键位是最重要的 (2026-08-04 大改造)

用户观念: 关键位有大量订单, 价格到关键位要么反转要么突破, 大部分交易
机会在关键位。需要画出正确的关键位。

实现 (严格版, 无未来函数):
  - 位带: research.levels.cluster_levels — 波段高低点在线聚类+冻结
    (K=3 pivot 确认, 形成后结构固定; 参数 min_touch=3, tol=0.5 收紧版)
  - 布林带: MA20 ± 2×std20 (动态关键位补充, 各周期)
  - 关系判定: 接近 (ATR 归一) / 触碰 (intrabar 穿过带) / 突破 (收盘越出
    带外 ≥ depth×ATR)

设计原则 (用户): 系统只描述"价格 vs 关键位"的关系, 不做方向预测。
"""
import numpy as np
import pandas as pd

from research.levels import cluster_levels

MIN_TOUCH = 3        # 收紧参数 (B2b: 位更少更准)
TOLERANCE = 0.5      # 聚类容差 × ATR
APPROACH_ATR = 1.0   # L1 酝酿: 接近阈值 (≤1.0 ATR)
TOUCH_LOOKBACK = 3   # 触碰检测回看根数
BREAK_DEPTH = 0.5    # 突破深度 × ATR


def detect_levels(df: pd.DataFrame, atr: np.ndarray, min_touch=MIN_TOUCH,
                  tolerance=TOLERANCE) -> list[dict]:
    """某周期的关键位带 (在线聚类+冻结)"""
    h = df["high"].values
    l = df["low"].values
    lvls = cluster_levels(h, l, np.asarray(atr, float), min_touch=min_touch,
                          tolerance_mult=tolerance)
    return [{"price": lv.price, "side": lv.side, "touch": lv.touch_count,
             "confirm_at": lv.confirm_at, "band": lv.band} for lv in lvls]


def bollinger_bands(df: pd.DataFrame, period: int = 20,
                    std: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """布林带: (中轨, 上轨, 下轨) — 动态关键位"""
    c = pd.Series(df["close"].values)
    ma = c.rolling(period).mean().values
    sd = c.rolling(period).std().values
    return ma, ma + std * sd, ma - std * sd


def level_relation(price: float, levels: list[dict], atr: float,
                   last_bar: int) -> dict:
    """当前价格 vs 最近活跃位带: 距离 (ATR 归一) — 不预测方向, 只描述"""
    sup = res = None
    for lv in levels:
        if lv["confirm_at"] > last_bar:
            continue
        if lv["side"] == "support" and lv["price"] < price:
            if sup is None or lv["price"] > sup["price"]:
                sup = lv
        elif lv["side"] == "resistance" and lv["price"] > price:
            if res is None or lv["price"] < res["price"]:
                res = lv
    out = {}
    for name, lv in (("support", sup), ("resistance", res)):
        if lv is not None:
            d = (price - lv["price"]) if name == "support" else (lv["price"] - price)
            out[name] = {"price": lv["price"], "side": lv["side"],
                         "touch": lv["touch"], "band": lv["band"],
                         "dist_atr": d / max(atr, 1e-9)}
    return out


def recent_touches(levels: list[dict], high: np.ndarray, low: np.ndarray,
                   last_bar: int, lookback: int = TOUCH_LOOKBACK) -> list[dict]:
    """最近 lookback 根内 intrabar 触碰位带的位带 (触碰 = 分叉点)

    降噪: 只返回最近的一个触碰 (bar 最新 + 价格最近位带)
    """
    hit = []
    t0 = max(0, last_bar - lookback + 1)
    for lv in levels:
        if lv["confirm_at"] > last_bar:
            continue
        p_lo = lv["price"] - lv["band"]
        p_hi = lv["price"] + lv["band"]
        for t in range(t0, last_bar + 1):
            if low[t] <= p_hi and high[t] >= p_lo:
                hit.append({**lv, "bar": t})
                break
    if not hit:
        return []
    close_now = (high[last_bar] + low[last_bar]) / 2
    hit.sort(key=lambda x: (x["bar"], abs(x["price"] - close_now)))
    return [hit[0]]


def break_nearest(rel: dict, close: np.ndarray, atr: np.ndarray,
                  last_bar: int, depth: float = BREAK_DEPTH) -> list[dict]:
    """最近关键位 (rel) 是否被收盘突破 — 突破 = 价格到关键位的结局之一

    只检查最近支撑/阻力两个位带 (level_relation 输出), 避免远位带噪音
    """
    out = []
    c = float(close[last_bar])
    a = float(atr[last_bar]) or 1.0
    for side, lv in rel.items():
        if lv is None:
            continue
        p_lo = lv["price"] - lv["band"]
        p_hi = lv["price"] + lv["band"]
        if side == "support" and c < p_lo - depth * a:
            out.append({**lv, "type": "支撑破", "depth": (p_lo - c) / a})
        elif side == "resistance" and c > p_hi + depth * a:
            out.append({**lv, "type": "阻力破", "depth": (c - p_hi) / a})
    return out
