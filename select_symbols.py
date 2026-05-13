#!/usr/bin/env python3
"""币种筛选器 — 两两相关性 + 结构清晰度"""

import sys
import numpy as np
import pandas as pd
from typing import Optional

sys.path.insert(0, ".")

from okx import OKXClient
from support_resistance import find_swing_levels
from loguru import logger


def select_symbols(
    pool_size: int = 100,
    max_correlation: float = 0.8,
    tf: str = "4h",
    lookback_bars: int = 60,
    min_structure_score: float = 0.3,
    min_volume: float = 0,  # 24h最小成交量(USDT), 0=不过滤
) -> list[str]:
    """
    从 top N 币种中筛选出独立且结构清晰的标的。
    """
    okx = OKXClient()
    all_symbols = okx.get_top_symbols(pool_size)
    logger.info(f"初筛池: {len(all_symbols)} 币")

    # ── 第零步：成交量过滤 ──
    if min_volume > 0:
        volumes = okx.get_24h_volume(all_symbols)
        all_symbols = [s for s in all_symbols if volumes.get(s, 0) >= min_volume]
        logger.info(f"成交量过滤(≥{min_volume/1e6:.0f}M): {len(all_symbols)} 币")

    # ── 第一步：拉取最近 lookback_bars 根 tf K 线 ──
    candles: dict[str, list[dict]] = {}
    for sym in all_symbols:
        try:
            bars = okx.fetch_ohlcv(sym, tf, limit=lookback_bars + 5)
            if len(bars) >= lookback_bars:
                candles[sym] = bars[-lookback_bars:]
        except Exception:
            pass

    valid = list(candles.keys())
    logger.info(f"有效数据: {len(valid)} 币")

    if len(valid) < 2:
        return valid

    # ── 第二步：收益率相关性矩阵 ──
    returns = {}
    for sym in valid:
        closes = np.array([b["close"] for b in candles[sym]])
        opens = np.array([b["open"] for b in candles[sym]])
        ret = (closes - opens) / opens
        returns[sym] = ret

    syms = list(returns.keys())
    n = len(syms)
    corr_matrix = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            c = np.corrcoef(returns[syms[i]], returns[syms[j]])[0, 1]
            corr_matrix[i, j] = c
            corr_matrix[j, i] = c

    # ── 第三步：高相关对去重 ──
    # 优先保留: 结构清晰度 × 流动性(BTC>ETH>...)
    # 按交易量排序作为优先级代理
    ranked = list(syms)  # 已经是按交易量排序
    excluded: set[str] = set()

    for i in range(n):
        if syms[i] in excluded:
            continue
        for j in range(i + 1, n):
            if syms[j] in excluded:
                continue
            if corr_matrix[i, j] > max_correlation:
                # 保留 i（交易量更高），排除 j
                excluded.add(syms[j])

    independent = [s for s in ranked if s not in excluded]
    logger.info(f"相关性过滤后: {len(independent)} 币 (阈值 {max_correlation})")

    # ── 第四步：结构清晰度评分 ──
    scores: dict[str, float] = {}
    for sym in independent:
        bars = candles[sym]
        df = pd.DataFrame([{
            "open": b["open"], "high": b["high"],
            "low": b["low"], "close": b["close"],
            "volume": b["volume"],
        } for b in bars])

        levels = find_swing_levels(df, lookback=lookback_bars)
        active = [l for l in levels if l.touch_count >= 2]  # 活跃防线

        if not active:
            scores[sym] = 0.0
            continue

        avg_touches = np.mean([l.touch_count for l in active])
        price = df["close"].iloc[-1]
        price_range = df["high"].max() - df["low"].min()
        if price_range > 0 and price > 0:
            # 防线密度(每1%价格区间的防线数) × 平均触及强度
            density = len(active) / (price_range / price * 100)
            structure_score = density * avg_touches
        else:
            structure_score = 0.0
        scores[sym] = round(structure_score, 4)

    # ── 第五步：综合排序输出 ──
    final = sorted(
        [(s, scores[s]) for s in independent if scores[s] >= min_structure_score],
        key=lambda x: x[1],
        reverse=True,
    )

    print(f"\n{'币种':<24s} {'成交量(M)':>10s} {'结构分':>8s} {'活跃防线':>8s}")
    print("-" * 54)
    # 获取成交量
    volumes = okx.get_24h_volume([s for s, _ in final]) if final else {}
    for sym, score in final:
        active_count = sum(1 for l in find_swing_levels(
            pd.DataFrame([{
                "open": b["open"], "high": b["high"],
                "low": b["low"], "close": b["close"],
                "volume": b["volume"],
            } for b in candles[sym]]), lookback=lookback_bars
        ) if l.touch_count >= 2)
        vol_m = volumes.get(sym, 0) / 1e6
        print(f"{sym:<24s} {vol_m:>10.1f} {score:>8.4f} {active_count:>8}")

    result = [s for s, _ in final]
    print(f"\n推荐监控: {len(result)} 币 (从 {pool_size} 筛选)")
    print("配置格式:")
    print("watchlist:")
    for s in result:
        print(f"  - {s}")

    return result


def update_config(selected: list[str], config_path: str = "config.yaml"):
    """将筛选结果写入 config.yaml 的 watchlist"""
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cfg["watchlist"] = selected
    # 注释掉 top_n，白名单优先
    if "top_n" in cfg:
        cfg.pop("top_n", None)

    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"已写入 {config_path}: {len(selected)} 币")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="币种筛选器")
    p.add_argument("--pool", type=int, default=100)
    p.add_argument("--corr", type=float, default=0.8)
    p.add_argument("--tf", default="4h")
    p.add_argument("--bars", type=int, default=60)
    p.add_argument("--min-score", type=float, default=0.3)
    p.add_argument("--min-volume", type=float, default=0, help="24h最小成交量(USDT), 如5000000=5M")
    p.add_argument("--write-config", action="store_true", help="直接写入 config.yaml")
    args = p.parse_args()

    result = select_symbols(
        pool_size=args.pool,
        max_correlation=args.corr,
        tf=args.tf,
        lookback_bars=args.bars,
        min_structure_score=args.min_score,
        min_volume=args.min_volume,
    )

    if args.write_config and result:
        update_config(result)
