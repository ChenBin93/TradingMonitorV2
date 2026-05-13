# 成交量分布 (Volume Profile) — 量加权防线节点
# VPoC = Volume Point of Control, HVN = High Volume Node, LVN = Low Volume Node

import pandas as pd
import numpy as np


def compute_volume_profile(df: pd.DataFrame, lookback: int = 200, n_bins: int = 40) -> dict | None:
    """
    从 OHLCV 数据计算成交量分布。
    返回: {poc, hvns, lvns, bins: [(price_low, price_high, volume)]}
    """
    if len(df) < lookback:
        return None

    window = df.tail(lookback)
    high = window["high"].max()
    low = window["low"].min()
    if high <= low:
        return None

    bin_size = (high - low) / n_bins
    bin_edges = np.linspace(low, high, n_bins + 1)
    bin_vol = np.zeros(n_bins)

    for _, bar in window.iterrows():
        bar_low = bar["low"]
        bar_high = bar["high"]
        bar_vol_total = bar["volume"]
        span = bar_high - bar_low
        if span <= 0 or bar_vol_total <= 0:
            continue

        # 体积均匀分布在该 K 线跨越的 bin 上
        start_bin = int((bar_low - low) / bin_size)
        end_bin = int((bar_high - low) / bin_size)
        start_bin = max(0, min(start_bin, n_bins - 1))
        end_bin = max(0, min(end_bin, n_bins - 1))

        n_spanned = end_bin - start_bin + 1
        vol_per_bin = bar_vol_total / n_spanned if n_spanned > 0 else bar_vol_total
        for b in range(start_bin, end_bin + 1):
            if 0 <= b < n_bins:
                bin_vol[b] += vol_per_bin

    total_vol = bin_vol.sum()
    if total_vol <= 0:
        return None

    # POC (成交量最大点)
    poc_idx = int(np.argmax(bin_vol))
    poc_price = float(bin_edges[poc_idx] + bin_size / 2)

    # HVN = 成交量 > 70 分位的区间; LVN = 成交量 < 30 分位
    vol_threshold_high = float(np.percentile(bin_vol[bin_vol > 0], 70)) if bin_vol.max() > 0 else float("inf")
    vol_threshold_low = float(np.percentile(bin_vol[bin_vol > 0], 30)) if bin_vol.max() > 0 else 0

    hvns = []
    lvns = []
    for i in range(n_bins):
        if bin_vol[i] <= 0:
            continue
        price_center = float(bin_edges[i] + bin_size / 2)
        node = {"price": price_center, "volume_pct": float(bin_vol[i] / total_vol)}
        if bin_vol[i] >= vol_threshold_high:
            hvns.append(node)
        elif bin_vol[i] <= vol_threshold_low:
            lvns.append(node)

    # 按成交量排序
    hvns.sort(key=lambda x: x["volume_pct"], reverse=True)

    return {
        "poc": poc_price,
        "hvns": hvns[:5],   # Top 5 HVN
        "lvns": lvns[:3],   # Top 3 LVN
        "bin_size": float(bin_size),
    }


def get_nearest_nodes(profile: dict, current_price: float) -> dict:
    """从 profile 中提取价格上方最近的阻力和下方最近的支撑（量节点）"""
    result = {"support": None, "resistance": None}
    if not profile:
        return result

    # 从 HVN 中找最近的支撑（低于当前价的最高 HVN）和阻力（高于当前价的最低 HVN）
    below = [h for h in profile.get("hvns", []) if h["price"] < current_price]
    above = [h for h in profile.get("hvns", []) if h["price"] > current_price]

    if below:
        result["support"] = max(below, key=lambda x: x["price"])
    if above:
        result["resistance"] = min(above, key=lambda x: x["price"])

    return result


def merge_levels(swing_levels: list, vp_nodes: dict, atr: float) -> dict:
    """
    将 swing 极点防线与量节点防线合并。
    如果两者在同一位置（距离 < 0.5 ATR）→ 标记为"超强"。
    如果量节点独立出现 → 标记为"量防线"。
    """
    result = {"merged": [], "volume_only": []}

    for node in vp_nodes.get("hvns", []):
        np_price = node["price"]
        matched = False
        for lvl in swing_levels:
            if abs(lvl.price - np_price) <= atr * 0.5:
                result["merged"].append({
                    "price": lvl.price,
                    "touches": lvl.touch_count,
                    "vol_pct": node["volume_pct"],
                    "type": "超强" if lvl.touch_count >= 3 else "确认",
                })
                matched = True
                break
        if not matched:
            result["volume_only"].append({
                "price": np_price,
                "vol_pct": node["volume_pct"],
            })

    return result
