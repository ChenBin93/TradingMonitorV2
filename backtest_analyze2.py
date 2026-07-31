#!/usr/bin/env python3
"""修复未来函数后的全面逐条件 edge 分析 — 重建真实结论

用法:
  python3 backtest_analyze2.py --symbols BTC/USDT:USDT ETH/USDT:USDT
  python3 backtest_analyze2.py   # 默认全部
"""
import argparse
import pickle
import sys
from collections import Counter, defaultdict

import yaml

sys.path.insert(0, ".")

from backtest_engine import load_all, simulate_trades


def wr1(events, data, symbols, dist=1.0):
    """1:1 对称模拟, 返回 (胜率, 笔数)"""
    if len(events) < 30:
        return 0, 0
    trades = simulate_trades(events, data, symbols,
                             atr_sl_buffer=dist, rr_min=0.3,
                             forward_hours=48, tp_mode="atr",
                             atr_tp_mult=dist, symmetric=True)
    closed = [t for t in trades if t.outcome in ("win", "loss")]
    if not closed:
        return 0, 0
    wins = sum(1 for t in closed if t.outcome == "win")
    return wins / len(closed), len(closed)


def show(name, evs, data, symbols):
    wr, n = wr1(evs, data, symbols)
    flag = ""
    if n >= 200:
        if wr >= 0.52:
            flag = " ★"
        elif wr <= 0.48:
            flag = " ✗"
    print(f"{name:<42} {n:>6} {wr*100:>5.1f}%{flag}", flush=True)


def analyze(events, data, symbols, cache_path):
    print(f"事件总数: {len(events)}")
    print(f"{'条件':<42} {'n':>6} {'1:1胜率':>8}")
    print("-" * 60)

    # ── 基线 ──
    show("基线 (全部信号)", events, data, symbols)

    # ── 1H 方向 (用户重点: 1H 趋势到底有没有影响) ──
    print()
    show("1H多头+信号多 (方向一致)", [e for e in events if e.ma_1h == "bullish" and e.direction == "long"], data, symbols)
    show("1H空头+信号空 (方向一致)", [e for e in events if e.ma_1h == "bearish" and e.direction == "short"], data, symbols)
    show("1H多头+信号空 (方向相反)", [e for e in events if e.ma_1h == "bullish" and e.direction == "short"], data, symbols)
    show("1H空头+信号多 (方向相反)", [e for e in events if e.ma_1h == "bearish" and e.direction == "long"], data, symbols)
    show("1H中性+信号", [e for e in events if e.ma_1h == "neutral"], data, symbols)

    # ── 4H bias ──
    print()
    show("bias一致 (4H趋势与信号同向)", [e for e in events if e.macro_bias == e.direction], data, symbols)
    show("bias相反", [e for e in events if e.macro_bias != "neutral" and e.macro_bias != e.direction], data, symbols)
    show("bias中性", [e for e in events if e.macro_bias == "neutral"], data, symbols)

    # ── 1H×bias 交叉 (1H 在 bias 下是否有增量) ──
    print()
    ev_b = [e for e in events if e.macro_bias == e.direction]
    show("bias一致 + 1H同向", [e for e in ev_b
                                if (e.ma_1h == "bullish" and e.direction == "long") or
                                   (e.ma_1h == "bearish" and e.direction == "short")], data, symbols)
    show("bias一致 + 1H反向", [e for e in ev_b
                                if (e.ma_1h == "bullish" and e.direction == "short") or
                                   (e.ma_1h == "bearish" and e.direction == "long")], data, symbols)
    show("bias一致 + 1H中性", [e for e in ev_b if e.ma_1h == "neutral"], data, symbols)

    # ── ADX ──
    print()
    show("1H ADX>=25", [e for e in events if e.adx_1h >= 25], data, symbols)
    show("1H ADX 15-25", [e for e in events if 15 <= e.adx_1h < 25], data, symbols)
    show("1H ADX<15", [e for e in events if e.adx_1h < 15], data, symbols)
    show("4H ADX>=25", [e for e in events if e.adx_4h >= 25], data, symbols)
    show("4H ADX 18-25", [e for e in events if 18 <= e.adx_4h < 25], data, symbols)
    show("4H ADX<18", [e for e in events if e.adx_4h < 18], data, symbols)

    # ── 顺势距离 (修复后重新验证) ──
    print()
    def align_dist(e):
        if not e.ma20_4h or not e.close_4h or not e.atr_4h:
            return None
        p = (e.close_4h - e.ma20_4h) / e.atr_4h
        return p if e.direction == "long" else -p
    show("顺势距离>=1.0", [e for e in events if align_dist(e) is not None and align_dist(e) >= 1.0], data, symbols)
    show("顺势距离0.5-1.0", [e for e in events if align_dist(e) is not None and 0.5 <= align_dist(e) < 1.0], data, symbols)
    show("顺势距离0-0.5", [e for e in events if align_dist(e) is not None and 0 <= align_dist(e) < 0.5], data, symbols)
    show("逆势-1~0", [e for e in events if align_dist(e) is not None and -1.0 <= align_dist(e) < 0], data, symbols)
    show("逆势<-1", [e for e in events if align_dist(e) is not None and align_dist(e) < -1.0], data, symbols)

    # ── 位置 (S/R 区间) ──
    print()
    show("位置: 贴支撑 (pos<0.3)", [e for e in events if e.pos_in_range is not None and e.pos_in_range < 0.3], data, symbols)
    show("位置: 中间 (0.3-0.7)", [e for e in events if e.pos_in_range is not None and 0.3 <= e.pos_in_range <= 0.7], data, symbols)
    show("位置: 贴阻力 (pos>0.7)", [e for e in events if e.pos_in_range is not None and e.pos_in_range > 0.7], data, symbols)

    # ── 信号 × 方向 ──
    print()
    by_sd = defaultdict(list)
    for e in events:
        by_sd[(e.signal, e.direction)].append(e)
    for (sig, d), evs in sorted(by_sd.items(), key=lambda kv: -len(kv[1])):
        show(f"信号: {sig}/{d}", evs, data, symbols)

    # ── 按标的 ──
    print()
    by_sym = defaultdict(list)
    for e in events:
        by_sym[e.symbol.replace("/USDT:USDT", "")].append(e)
    for sym, evs in sorted(by_sym.items(), key=lambda kv: -len(kv[1])):
        show(f"标的: {sym}", evs, data, symbols)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="*", default=None)
    p.add_argument("--cache", default="data/events_cache/events_9782cc34b9fac2f7.pkl")
    args = p.parse_args()

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    data = load_all("data/backtest.db")
    symbols = args.symbols or list(data.keys())

    with open(args.cache, "rb") as f:
        events = pickle.load(f)
    if args.symbols:
        sym_set = set(args.symbols)
        events = [e for e in events if e.symbol in sym_set]

    analyze(events, data, symbols, args.cache)
