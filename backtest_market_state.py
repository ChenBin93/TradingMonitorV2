#!/usr/bin/env python3
"""回测市场状态矩阵 — 校准每种状态组合下的方向偏倚概率"""

import sqlite3, sys
from datetime import datetime
from collections import defaultdict
import pandas as pd
import numpy as np

sys.path.insert(0, ".")
from indicators import compute as compute_indicators


def load_tf_data(db_path: str, symbol: str, tfs: list[str], min_bars: int):
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    result = {}
    for tf in tfs:
        rows = db.execute("""
            SELECT timestamp, open, high, low, close, volume
            FROM history_candles WHERE symbol=? AND timeframe=?
            ORDER BY timestamp ASC
        """, (symbol, tf)).fetchall()
        if len(rows) < min_bars:
            continue
        df = pd.DataFrame([dict(r) for r in rows])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        result[tf] = df
    db.close()
    return result


def backtest(db_path: str = "data/history.db", symbol: str = "BTC/USDT:USDT",
             min_bars: int = 200, forward_hours: int = 8, atr_target: float = 1.5):
    """
    步进每个 1H K 线，计算当前市场状态，记录 forward_hours 内先碰到的方向。
    """
    tf_data = load_tf_data(db_path, symbol, ["1h", "4h"], min_bars)
    if "1h" not in tf_data or "4h" not in tf_data:
        print("数据不足")
        return

    df1, df4 = tf_data["1h"], tf_data["4h"]

    ind_params = {
        "1h": {"roc_period": 10, "rsi_period": 14, "adx_period": 14, "bb_period": 20, "bb_std": 2,
               "atr_period": 14, "volume_ma_period": 20, "ma_short": 5, "ma_mid": 20, "ma_long": 60},
        "4h": {"roc_period": 20, "rsi_period": 14, "adx_period": 14, "bb_period": 20, "bb_std": 2,
               "atr_period": 14, "volume_ma_period": 20, "ma_short": 5, "ma_mid": 20, "ma_long": 60},
    }

    results = []
    total = 0

    for i in range(min_bars, len(df1) - forward_hours):
        current_ts = df1["timestamp"].iloc[i]

        # ── 4H 指标 ──
        df4_avail = df4[df4["timestamp"] <= current_ts]
        if len(df4_avail) < 50:
            continue
        ind4 = compute_indicators(df4_avail, ind_params["4h"])
        if not ind4:
            continue

        # ── 1H 指标 ──
        df1_avail = df1.iloc[:i + 1]
        if len(df1_avail) < 30:
            continue
        ind1 = compute_indicators(df1_avail, ind_params["1h"])
        if not ind1:
            continue

        # ── 状态变量 ──
        adx_4h = ind4.get("adx", 0) or 0
        ma_align_4h = ind4.get("ma_alignment", "neutral")
        ma20_4h = ind4.get("ma20")
        close_1h = ind1.get("close")
        bb_pct_4h = ind4.get("bb_width_short_pct", 50)
        adx_1h = ind1.get("adx", 0) or 0
        ma_align_1h = ind1.get("ma_alignment", "neutral")

        # 回调深度
        pullback_pct = 0
        if ma20_4h and close_1h and ma20_4h > 0:
            pullback_pct = (close_1h - ma20_4h) / ma20_4h * 100

        # ── 离散分类 ──
        # ADX 分组
        if adx_4h < 20:
            adx_bucket = "low"
        elif adx_4h < 30:
            adx_bucket = "mid"
        else:
            adx_bucket = "high"

        # 回调分组
        if abs(pullback_pct) < 1:
            pullback_bucket = "near_ma"
        elif pullback_pct >= 1:
            pullback_bucket = "above_ma"
        else:
            pullback_bucket = "below_ma"

        # BB 分组
        if bb_pct_4h is None:
            bb_pct_4h = 50
        if bb_pct_4h <= 20:
            bb_bucket = "tight"
        elif bb_pct_4h >= 50:
            bb_bucket = "wide"
        else:
            bb_bucket = "mid"

        # ── 前瞻：未来 forward_hours 根 1H K 线，先碰到 +atr_target 还是 -atr_target ──
        atr_1h = ind1.get("atr") or 1
        entry_price = close_1h or 0
        if entry_price <= 0:
            continue

        target_up = entry_price + atr_1h * atr_target
        target_dn = entry_price - atr_1h * atr_target

        outcome = "none"
        for j in range(i + 1, min(i + 1 + forward_hours, len(df1))):
            h = df1["high"].iloc[j]
            l = df1["low"].iloc[j]
            if h >= target_up:
                outcome = "up"
                break
            if l <= target_dn:
                outcome = "dn"
                break

        if outcome == "none":
            continue

        # ── 做多胜率 = 先碰上方目标的概率 ──
        long_win = 1 if outcome == "up" else 0
        short_win = 1 if outcome == "dn" else 0

        results.append({
            "adx_bucket": adx_bucket,
            "ma_4h": ma_align_4h,
            "ma_1h": ma_align_1h,
            "pullback_bucket": pullback_bucket,
            "bb_bucket": bb_bucket,
            "adx_4h": round(adx_4h, 1),
            "adx_1h": round(adx_1h, 1),
            "pullback_pct": round(pullback_pct, 1),
            "bb_pct": round(bb_pct_4h, 1),
            "long_win": long_win,
            "short_win": short_win,
            "outcome": outcome,
        })
        total += 1
        if total % 500 == 0:
            print(f"  已处理 {total} 个时间点...")

    print(f"  总计 {total} 个有效时间点")

    df = pd.DataFrame(results)
    return df


def summarize(df: pd.DataFrame):
    if df.empty:
        print("无数据")
        return

    print(f"\n{'='*80}")
    print(f"市场状态回测 — 总体统计")
    print(f"{'='*80}")
    print(f"样本数: {len(df)}")
    print(f"整体做多胜率(先碰+1.5ATR): {df['long_win'].mean():.1%}")
    print(f"整体做空胜率(先碰-1.5ATR): {df['short_win'].mean():.1%}")

    # ── 按 MA 排列 × ADX 分组 ──
    print(f"\n{'='*80}")
    print(f"{'MA4H':<10s} {'ADX':<6s} {'样本':>6s} {'做多胜率':>8s} {'做空胜率':>8s} {'倾向':>6s}")
    print(f"{'-'*80}")

    # 组合: MA_4H × ADX_bucket × pullback_bucket
    groups = df.groupby(["ma_4h", "adx_bucket", "pullback_bucket", "bb_bucket"])
    stats = []
    for (ma4, adx_b, pb_b, bb_b), g in groups:
        n = len(g)
        if n < 20:
            continue
        long_wr = g["long_win"].mean()
        short_wr = g["short_win"].mean()

        bias = "多"
        conf = int(long_wr * 100)
        if short_wr > long_wr:
            bias = "空"
            conf = int(short_wr * 100)
        elif abs(long_wr - short_wr) < 0.03:
            bias = "中"

        stats.append({
            "ma4": ma4, "adx_b": adx_b, "pb_b": pb_b, "bb_b": bb_b,
            "n": n, "long_wr": long_wr, "short_wr": short_wr,
            "bias": bias, "conf": conf,
        })

    stats.sort(key=lambda x: -max(x["long_wr"], x["short_wr"]) * x["n"])

    for s in stats[:25]:
        bb_label = {"tight": "紧", "mid": "中", "wide": "宽"}.get(s["bb_b"], s["bb_b"])
        pb_label = {"near_ma": "近MA", "above_ma": ">MA", "below_ma": "<MA"}.get(s["pb_b"], s["pb_b"])
        bias_icon = "🔺" if s["bias"] == "多" else "🔻" if s["bias"] == "空" else "➖"
        print(f"{s['ma4']:<10s} {s['adx_b']:<6s} {s['n']:>6d} {s['long_wr']:>7.1%} "
              f"{s['short_wr']:>7.1%} {bias_icon}{s['bias']:>3s}({s['conf']}%)"
              f"  BB{bb_label} 回调{pb_label}")

    # ── 简化版：只按 MA_4H × ADX × 回调方向 ──
    print(f"\n{'='*80}")
    print(f"简化: MA_4H × ADX → 做多/做空胜率")
    print(f"{'='*80}")
    simple_groups = df.groupby(["ma_4h", "adx_bucket"])
    for (ma4, adx_b), g in simple_groups:
        n = len(g)
        if n < 30:
            continue
        print(f"  {ma4:<10s} ADX={adx_b:<6s} n={n:>5d} 做多胜率={g['long_win'].mean():.1%} 做空胜率={g['short_win'].mean():.1%}")

    # ── 关键组合高亮 ──
    print(f"\n{'='*80}")
    print(f"最有价值的市场状态（做多最优 / 做空最优）")
    print(f"{'='*80}")
    print()

    # 做多最优: 4H bullish + ADX high + 回调在 MA 附近或上方
    for ma4 in ["bullish"]:
        for adx_b in ["high", "mid"]:
            for pb_b in ["near_ma", "above_ma"]:
                g = df[(df["ma_4h"] == ma4) & (df["adx_bucket"] == adx_b) & (df["pullback_bucket"] == pb_b)]
                if len(g) >= 20:
                    print(f"  [做多] 4H{ma4} ADX_{adx_b} 回调_{pb_b} n={len(g):>4d} "
                          f"做多胜率={g['long_win'].mean():.1%} 做空胜率={g['short_win'].mean():.1%}")

    for ma4 in ["bearish"]:
        for adx_b in ["high", "mid"]:
            for pb_b in ["near_ma", "below_ma"]:
                g = df[(df["ma_4h"] == ma4) & (df["adx_bucket"] == adx_b) & (df["pullback_bucket"] == pb_b)]
                if len(g) >= 20:
                    print(f"  [做空] 4H{ma4} ADX_{adx_b} 回调_{pb_b} n={len(g):>4d} "
                          f"做多胜率={g['long_win'].mean():.1%} 做空胜率={g['short_win'].mean():.1%}")

    return stats


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="data/history.db")
    p.add_argument("--symbol", default="BTC/USDT:USDT")
    p.add_argument("--min-bars", type=int, default=200)
    p.add_argument("--forward-hours", type=int, default=8, help="前瞻窗口(1H K线根数)")
    p.add_argument("--atr-target", type=float, default=1.5, help="目标ATR倍数")
    args = p.parse_args()

    print(f"回测市场状态: {args.symbol}")
    print(f"  前瞻窗口: {args.forward_hours}h | 目标: ±{args.atr_target} ATR")
    df = backtest(args.db, args.symbol, args.min_bars, args.forward_hours, args.atr_target)
    if df is not None:
        summarize(df)
