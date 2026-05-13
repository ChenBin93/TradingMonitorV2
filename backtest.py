#!/usr/bin/env python3
"""多 TF 同步回测 — 模拟三层框架的完整决策链路"""

import sys, sqlite3
from datetime import datetime
from collections import defaultdict
import pandas as pd
import numpy as np

sys.path.insert(0, ".")

from indicators import compute as compute_indicators
from signals import SIGNALS, SignalState, get_direction, get_regime
from support_resistance import find_swing_levels, get_nearest_levels


def load_tf_data(db_path: str, symbol: str, tfs: list[str], min_bars: int) -> dict[str, pd.DataFrame]:
    """加载所有 TF 的 K 线，对齐时间轴"""
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


def backtest_symbol(
    symbol: str,
    tf_data: dict[str, pd.DataFrame],
    ind_params: dict,
    min_bars: int = 200,
    forward_bars_4h: int = 12,  # 前瞻窗口(4h)
    atr_sl_mult: float = 1.5,
) -> list[dict]:
    """
    多 TF 同步回测一个符号:
    - 4h 方向: 判断市场状态和方向
    - 1h 结构: 判断防线位置
    - 15m 执行: 寻找信号和确认
    """
    results = []
    tfs = sorted(tf_data.keys(), reverse=True)  # 4h, 1h, 15m
    if "4h" not in tf_data or "15m" not in tf_data:
        return results

    df4 = tf_data["4h"]
    df1 = tf_data.get("1h", tf_data.get("15m", df4))
    df15 = tf_data["15m"]

    # 去重：同信号在 5根15m K线(~75min) 内不重复计数
    last_trigger: dict[str, int] = {}  # key=(signal_type, direction) -> last bar index

    # 按 15m K 线步进
    for i in range(min_bars, len(df15) - 5):
        # 当前时间戳
        current_ts = df15["timestamp"].iloc[i]

        # ── 4h 方向（找当前时间之前最新的 4h K 线终点）──
        df4_avail = df4[df4["timestamp"] <= current_ts]
        if len(df4_avail) < 50:
            continue
        ind4 = compute_indicators(df4_avail, ind_params.get("4h", {}))
        if not ind4:
            continue
        dir_4h = ind4.get("ma_alignment", "neutral")  # bullish/bearish/neutral

        # ── 1h 结构 ──
        df1_avail = df1[df1["timestamp"] <= current_ts]
        if len(df1_avail) < 30:
            continue
        ind1 = compute_indicators(df1_avail, ind_params.get("1h", {}))
        if not ind1:
            continue

        # ── 15m 执行 ──
        df15_window = df15.iloc[:i + 1]
        if len(df15_window) < 30:
            continue
        ind15 = compute_indicators(df15_window, ind_params.get("15m", {}))
        if not ind15:
            continue

        # 构造 SignalState（以 15m 数据为主 + 引入 4h 方向）
        direction_15 = get_direction(ind15)
        regime_15 = get_regime(ind15)
        state = SignalState(
            symbol=symbol, timeframe="15m", ind=ind15,
            bbw_rank=ind15.get("bb_width_short_pct"),
            regime=regime_15, direction=direction_15,
        )

        # ── 检测所有信号 ──
        for sig_def in SIGNALS:
            try:
                state.params = sig_def.params
                sig_result = sig_def.check(state)
                if not sig_result:
                    continue
            except Exception:
                continue

            entry_price = ind15.get("close")
            atr_val = ind15.get("atr") or 1
            sig_dir = sig_result.get("direction", direction_15)

            # 去重: 同信号+方向在 5根15m K线(~75min) 内不重复
            dedup_key = f"{sig_def.id}_{sig_dir}"
            last_i = last_trigger.get(dedup_key, -999)
            if i - last_i < 5:
                continue
            last_trigger[dedup_key] = i

            # 止损
            sl = entry_price - atr_val * atr_sl_mult if sig_dir == "long" else entry_price + atr_val * atr_sl_mult

            # 止盈：用 1h S/R
            tp = None
            pos = "中间"
            try:
                levels = find_swing_levels(df1_avail, lookback=50)
                support, resistance = get_nearest_levels(levels, entry_price)
                if sig_dir == "long" and resistance:
                    tp = resistance.price
                    if support:
                        rng_total = resistance.price - support.price
                        if rng_total > 0:
                            pos_in = (entry_price - support.price) / rng_total
                            if pos_in <= 0.3: pos = "近支撑"
                            elif pos_in >= 0.7: pos = "近阻力"
                elif sig_dir == "short" and support:
                    tp = support.price
            except Exception:
                pass
            if tp is None:
                tp = entry_price + atr_val * 2.5 if sig_dir == "long" else entry_price - atr_val * 2.5

            # ── 前瞻判断 ──
            # 看后续 forward_bars_4h 根 4h K 线的结果
            future_idx = df4[df4["timestamp"] > current_ts]
            future_4h = future_idx.head(forward_bars_4h)
            outcome = "open"
            for _, fb in future_4h.iterrows():
                if sig_dir == "long":
                    if fb["low"] <= sl: outcome = "miss"; break
                    if fb["high"] >= tp: outcome = "hit"; break
                else:
                    if fb["high"] >= sl: outcome = "miss"; break
                    if fb["low"] <= tp: outcome = "hit"; break

            # ── 多 TF 方向一致性 ──
            mtf_confirm = (dir_4h == "bullish" and sig_dir == "long") or (dir_4h == "bearish" and sig_dir == "short")

            rr = abs(tp - entry_price) / abs(sl - entry_price) if abs(sl - entry_price) > 0 else 0
            adx_val = ind15.get("adx") or 0
            market_regime = "trend" if adx_val >= 25 else "range" if adx_val < 20 else "transition"

            results.append({
                "symbol": symbol,
                "signal_type": sig_def.id,
                "direction": sig_dir,
                "entry": round(entry_price, 2),
                "sl": round(sl, 2),
                "tp": round(tp, 2),
                "rr": round(rr, 2),
                "outcome": outcome,
                "regime": market_regime,
                "dir_4h": dir_4h,
                "mtf_confirm": mtf_confirm,
                "position": pos,
                "adx": round(adx_val, 1),
                "timestamp": str(current_ts),
            })

    return results


def report(results: pd.DataFrame):
    if results.empty:
        print("无信号。")
        return

    closed = results[results["outcome"] != "open"]
    if closed.empty:
        print("所有信号未决。")
        return

    closed["hit"] = (closed["outcome"] == "hit").astype(int)

    print(f"\n总信号: {len(results)}, 已决: {len(closed)}")
    print(f"整体胜率: {closed['hit'].mean():.0%}")
    print(f"多TF确认胜率: {closed[closed['mtf_confirm']]['hit'].mean():.0%}  (n={len(closed[closed['mtf_confirm']])})")
    print(f"无确认胜率:   {closed[~closed['mtf_confirm']]['hit'].mean():.0%}  (n={len(closed[~closed['mtf_confirm']])})")

    print("\n" + "=" * 80)
    print(f"{'信号':<22s} {'方向':<6s} {'次数':>5s} {'胜率':>6s} {'RR':>6s} {'趋势':>6s} {'震荡':>6s} {'多TF':>6s}")
    print("-" * 80)

    for sig_id in sorted(closed["signal_type"].unique()):
        for d in ["long", "short"]:
            sub = closed[(closed["signal_type"] == sig_id) & (closed["direction"] == d)]
            if len(sub) < 5:
                continue
            wr = sub["hit"].mean()
            rr = sub["rr"].mean()
            trend_wr = sub[sub["regime"] == "trend"]["hit"].mean() if len(sub[sub["regime"] == "trend"]) >= 3 else 0
            range_wr = sub[sub["regime"] == "range"]["hit"].mean() if len(sub[sub["regime"] == "range"]) >= 3 else 0
            mtf_wr = sub[sub["mtf_confirm"]]["hit"].mean() if len(sub[sub["mtf_confirm"]]) >= 3 else 0
            print(f"{sig_id:<22s} {d:<6s} {len(sub):>5d} {wr:>5.0%} {rr:>5.2f} "
                  f"{trend_wr:>5.0%} {range_wr:>5.0%} {mtf_wr:>5.0%}")

    # 清单叠加效果
    print("\n── 多TF确认 vs 位置过滤 ──")
    for mtf in [True, False]:
        for pos in ["近支撑", "近阻力", "中间"]:
            sub = closed[(closed["mtf_confirm"] == mtf) & (closed["position"] == pos)]
            if len(sub) < 3:
                continue
            print(f"  MTF={'✓' if mtf else '✗'} pos={pos:<6s}: n={len(sub):>4d} wr={sub['hit'].mean():.0%}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="data/history.db")
    p.add_argument("--symbols", nargs="*", default=["BTC/USDT:USDT", "ETH/USDT:USDT"])
    p.add_argument("--min-bars", type=int, default=200)
    args = p.parse_args()

    ind_params = {
        "15m": {"roc_period": 5, "rsi_period": 14, "adx_period": 14, "bb_period": 20, "bb_std": 2,
                "atr_period": 14, "volume_ma_period": 20, "ma_short": 5, "ma_mid": 20, "ma_long": 60},
        "1h":  {"roc_period": 10, "rsi_period": 14, "adx_period": 14, "bb_period": 20, "bb_std": 2,
                "atr_period": 14, "volume_ma_period": 20, "ma_short": 5, "ma_mid": 20, "ma_long": 60},
        "4h":  {"roc_period": 20, "rsi_period": 14, "adx_period": 14, "bb_period": 20, "bb_std": 2,
                "atr_period": 14, "volume_ma_period": 20, "ma_short": 5, "ma_mid": 20, "ma_long": 60},
    }

    all_results = []
    for sym in args.symbols:
        print(f"\n回测: {sym}")
        tf_data = load_tf_data(args.db, sym, ["15m", "1h", "4h"], args.min_bars)
        if not tf_data:
            print(f"  数据不足")
            continue
        print(f"  K线: " + " ".join(f"{tf}:{len(df)}" for tf, df in tf_data.items()))

        results = backtest_symbol(sym, tf_data, ind_params, min_bars=args.min_bars)
        all_results.extend(results)
        print(f"  信号: {len(results)}")

    df = pd.DataFrame(all_results)
    report(df)
