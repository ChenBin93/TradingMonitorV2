#!/usr/bin/env python3
"""逐条件 edge 分析 — 每个过滤条件对 1:1 胜率的独立贡献

用法:
  python3 backtest_analyze.py --symbols BTC/USDT:USDT ETH/USDT:USDT SOL/USDT:USDT
  python3 backtest_analyze.py   # 默认全部标的
"""
import argparse
import sys
from collections import defaultdict

import yaml

sys.path.insert(0, ".")

from backtest_engine import load_all, detect_signals, simulate_trades


def wr_stats(events, data, symbols, dist=1.0):
    """1:1 对称模拟, 返回 (胜率, 笔数, 收益)"""
    trades = simulate_trades(events, data, symbols,
                             atr_sl_buffer=dist, rr_min=0.3,
                             forward_hours=48, tp_mode="atr",
                             atr_tp_mult=dist, symmetric=True)
    closed = [t for t in trades if t.outcome in ("win", "loss")]
    if not closed:
        return None
    wins = sum(1 for t in closed if t.outcome == "win")
    ret = sum(t.rr if t.outcome == "win" else -1 for t in closed)
    return wins / len(closed), len(closed), ret


def analyze(events, data, symbols):
    """逐条件 1:1 胜率分析"""
    print(f"{'条件':<40} {'笔数':>7} {'胜率':>7} {'收益':>8}")
    print("-" * 70)

    def show(name, evs):
        s = wr_stats(evs, data, symbols)
        if s is None:
            print(f"{name:<40} {'0':>7} {'-':>7}")
            return
        wr, n, ret = s
        print(f"{name:<40} {n:>7} {wr*100:>6.1f}% {ret:>8.1f}")

    # ── 基线 ──
    show("基线 (全部信号)", events)

    # ── 1H 方向过滤 ──
    show("1H多头+信号多 (方向一致)", [e for e in events if e.ma_1h == "bullish" and e.direction == "long"])
    show("1H空头+信号空 (方向一致)", [e for e in events if e.ma_1h == "bearish" and e.direction == "short"])
    show("1H多头+信号空 (方向相反)", [e for e in events if e.ma_1h == "bullish" and e.direction == "short"])
    show("1H空头+信号多 (方向相反)", [e for e in events if e.ma_1h == "bearish" and e.direction == "long"])

    # ── 4H 宏观 bias 过滤 ──
    show("bias多头+信号多 (趋势一致)", [e for e in events if e.macro_bias == "long" and e.direction == "long"])
    show("bias空头+信号空 (趋势一致)", [e for e in events if e.macro_bias == "short" and e.direction == "short"])
    show("bias多头+信号空 (趋势相反)", [e for e in events if e.macro_bias == "long" and e.direction == "short"])
    show("bias空头+信号多 (趋势相反)", [e for e in events if e.macro_bias == "short" and e.direction == "long"])

    # ── 组合: 方向+趋势 都一致 ──
    show("1H方向+bias 都一致", [e for e in events
                                if ((e.ma_1h == "bullish" and e.direction == "long") or
                                    (e.ma_1h == "bearish" and e.direction == "short"))
                                and e.macro_bias != "neutral"
                                and ((e.macro_bias == "long" and e.direction == "long") or
                                     (e.macro_bias == "short" and e.direction == "short"))])
    show("1H方向一致+bias相反", [e for e in events
                                if ((e.ma_1h == "bullish" and e.direction == "long") or
                                    (e.ma_1h == "bearish" and e.direction == "short"))
                                and e.macro_bias != "neutral"
                                and ((e.macro_bias == "long" and e.direction == "short") or
                                     (e.macro_bias == "short" and e.direction == "long"))])

    # ── ADX 强弱 ──
    show("1H ADX>=25", [e for e in events if e.adx_1h >= 25])
    show("1H ADX 15-25", [e for e in events if 15 <= e.adx_1h < 25])
    show("1H ADX<15", [e for e in events if e.adx_1h < 15])

    # ── 4H ADX ──
    show("4H ADX>=25", [e for e in events if e.adx_4h >= 25])
    show("4H ADX 18-25", [e for e in events if 18 <= e.adx_4h < 25])
    show("4H ADX<18", [e for e in events if e.adx_4h < 18])

    # ── 位置 ──
    show("位置: 贴支撑 (pos<0.3)", [e for e in events if e.pos_in_range is not None and e.pos_in_range < 0.3])
    show("位置: 中间 (0.3-0.7)", [e for e in events if e.pos_in_range is not None and 0.3 <= e.pos_in_range <= 0.7])
    show("位置: 贴阻力 (pos>0.7)", [e for e in events if e.pos_in_range is not None and e.pos_in_range > 0.7])

    # ── 按信号 × 方向过滤后 ──
    print()
    print("按信号 × 方向 (全部):")
    by_sig_dir = defaultdict(list)
    for e in events:
        by_sig_dir[(e.signal, e.direction)].append(e)
    for (sig, d), evs in sorted(by_sig_dir.items(), key=lambda kv: -len(kv[1])):
        show(f"  {sig}/{d}", evs)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="*", default=None)
    args = p.parse_args()

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    data = load_all("data/backtest.db")
    symbols = args.symbols or list(data.keys())

    events = detect_signals(data, symbols, cfg)
    print(f"Events: {len(events)}\n")
    analyze(events, data, symbols)
