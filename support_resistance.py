# 支撑阻力位识别 — 活跃边界检测

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class Level:
    price: float
    touch_count: int
    side: str           # "support" or "resistance"
    strength: str        # "强" / "中" / "弱"
    last_touch_idx: int  # 最近一次触及的 K 线索引


def find_swing_levels(df: pd.DataFrame, lookback: int = 50) -> list[Level]:
    """从 DataFrame 中找出活跃支撑和阻力位"""
    if len(df) < 5:
        return []

    df = df.tail(lookback).reset_index(drop=True)
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values

    n = len(df)
    # 找局部极值点(左右各 2 根 K 线的范围)
    swing_highs = []  # (idx, price)
    swing_lows = []   # (idx, price)

    for i in range(2, n - 2):
        if highs[i] >= max(highs[i-2:i+3]):
            swing_highs.append((i, highs[i]))
        if lows[i] <= min(lows[i-2:i+3]):
            swing_lows.append((i, lows[i]))

    # 聚类：相近的极值点合并（距离 < ATR/3）
    atr = np.mean(highs - lows) if len(highs) > 0 else 1.0
    cluster_tolerance = atr * 0.3

    levels = []
    for side, swings in [("resistance", swing_highs), ("support", swing_lows)]:
        if not swings:
            continue
        swings.sort(key=lambda x: x[1])
        clusters = []
        current_cluster = [swings[0]]

        for i in range(1, len(swings)):
            if swings[i][1] - current_cluster[-1][1] <= cluster_tolerance:
                current_cluster.append(swings[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [swings[i]]
        clusters.append(current_cluster)

        for cluster in clusters:
            if len(cluster) >= 2:  # 至少触及 2 次
                avg_price = float(np.median([p for _, p in cluster]))
                last_idx = max(idx for idx, _ in cluster)
                strength = "强" if len(cluster) >= 3 else "中"
                # 根据价格量级四舍五入：>100 → 整数, >1 → 1位小数, 其余2位
                if avg_price > 100:
                    price = round(avg_price)
                elif avg_price > 1:
                    price = round(avg_price, 1)
                else:
                    price = round(avg_price, 2)
                levels.append(Level(
                    price=price,
                    touch_count=len(cluster),
                    side=side,
                    strength=strength,
                    last_touch_idx=last_idx,
                ))

    levels.sort(key=lambda x: (x.touch_count, x.last_touch_idx), reverse=True)
    return levels


def get_nearest_levels(levels: list[Level], current_price: float) -> tuple[Optional[Level], Optional[Level]]:
    """返回当前价格上方最近的阻力和下方最近的支撑"""
    nearest_resistance = None
    nearest_support = None

    resistance_levels = [l for l in levels if l.side == "resistance" and l.price > current_price]
    support_levels = [l for l in levels if l.side == "support" and l.price < current_price]

    if resistance_levels:
        nearest_resistance = min(resistance_levels, key=lambda x: x.price - current_price)
    if support_levels:
        nearest_support = max(support_levels, key=lambda x: x.price - current_price)

    return nearest_support, nearest_resistance


def get_df_summary(df: pd.DataFrame, current_price: float, lookback: int = 50) -> dict:
    """获取结构摘要：活跃边界 + 位置判断"""
    levels = find_swing_levels(df, lookback)
    support, resistance = get_nearest_levels(levels, current_price)

    result = {
        "support": None,
        "resistance": None,
        "position": "中间",
    }

    if support:
        result["support"] = {
            "price": support.price,
            "touches": support.touch_count,
            "strength": support.strength,
        }
    if resistance:
        result["resistance"] = {
            "price": resistance.price,
            "touches": resistance.touch_count,
            "strength": resistance.strength,
        }

    # 位置判断
    if support and resistance:
        total_range = resistance.price - support.price
        if total_range > 0:
            pos_in_range = (current_price - support.price) / total_range

            if pos_in_range <= 0.25:
                result["position"] = "近支撑"
            elif pos_in_range >= 0.75:
                result["position"] = "近阻力"
            else:
                result["position"] = "中间"

    return result
