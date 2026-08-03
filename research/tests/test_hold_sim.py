#!/usr/bin/env python3
"""持仓模拟器测试 — 黄金测试 (手算) + 无未来函数不变性

场景: 跟踪止损上移 / 固定ATR止损 / late退出 / 超时 / down 对称
"""
import numpy as np
import pandas as pd
import pytest

from research.hold_sim import simulate_holds


def mk(closes, atr=3.0, lows=None, highs=None):
    n = len(closes)
    lows = np.array(lows) if lows is not None else np.array(closes) - 1.0
    highs = np.array(highs) if highs is not None else np.array(closes) + 1.0
    return (np.array(closes, float), highs, lows, np.full(n, atr))


def test_late_exit():
    """early 入场 → 一路上涨 → late 出现即平仓"""
    closes = [100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 116, 114, 112, 110, 108]
    c, h, l, a = mk(closes, atr=5.0)
    states = np.array(["up:early"] * 4 + ["up:mid"] * 4 + ["up:late"] * 7)
    entries = np.array([True] + [False] * 14)
    trades = simulate_holds(c, h, l, a, states, entries, "long", "hl", exit_late=True, w=96)
    assert len(trades) == 1
    t = trades[0]
    assert t.reason == "late"
    assert t.exit_idx == 8 and t.exit_px == 116.0
    assert abs(t.r_mult - 3.2) < 1e-9  # (116-100)/5


def test_trailing_stop_trailing():
    """跟踪止损: 新确认 HL 上移 → 跌破上移后的止损触发"""
    closes = [100, 103, 106, 104, 101, 98, 96, 99, 102, 105, 103, 100, 97, 94, 91]
    lows = [99, 102, 105, 103, 100, 97, 95, 98, 101, 104, 102, 99, 96, 93, 90]
    c, h, l, a = mk(closes, atr=3.0, lows=lows)
    states = np.array(["up:early"] * 3 + ["up:mid"] * 6 + ["up:late"] * 6)
    entries = np.array([True] + [False] * 14)
    # 入场 bar 0: 无已确认 pivot → ATR 兜底 stop = 100-3 = 97
    # bar 9 确认 pivot low @6 (95): 95 < 97 不上移
    trades = simulate_holds(c, h, l, a, states, entries, "long", "hl", exit_late=False, w=96)
    assert len(trades) == 1
    t = trades[0]
    assert t.reason == "stop"
    assert abs(t.exit_px - 97.0) < 1e-9  # 初始 ATR 止损
    assert abs(t.r_mult - (-1.0)) < 1e-9


def test_trailing_stop_moves_up():
    """跟踪止损上移: 入场后确认更高 HL → 止损上移 → 在上移止损触发"""
    # 段1 跌到 92 (pivot A @4, 确认@7) → 段2 涨 → 回调 103 (pivot B @11, 确认@14)
    # 入场在 bar 8 (A 已确认, stop=92); B 确认后 stop 上移至 103; 破位跌破 103 触发
    closes = [100, 98, 96, 94, 93, 96, 99, 102, 105, 108, 106, 104, 105, 106, 108, 110, 108, 105, 102, 99, 96, 93]
    lows = [99, 97, 95, 93, 92, 95, 98, 101, 104, 107, 105, 103, 104, 105, 107, 109, 107, 104, 101, 98, 95, 92]
    c, h, l, a = mk(closes, atr=3.0, lows=lows)
    states = np.array(["range"] * 8 + ["up:early"] * 4 + ["up:mid"] * 6 + ["up:late"] * 4)
    entries = np.array([False] * 8 + [True] + [False] * 13)
    trades = simulate_holds(c, h, l, a, states, entries, "long", "hl", exit_late=False, w=96)
    assert len(trades) == 1
    t = trades[0]
    # 入场 bar8 (close 105), 初始 stop = 已确认 pivot A(92)
    # bar 14 确认 B(103) > 92 → stop 上移至 103; bar 18 close=102 < 103 → 触发
    assert t.entry_idx == 8 and t.entry_px == 105.0
    assert t.reason == "stop"
    assert t.exit_idx == 18
    assert abs(t.exit_px - 103.0) < 1e-9
    assert abs(t.r_mult - (103 - 105) / 13.0) < 1e-9  # R 用初始止损距离 13


def test_fixed_atr_stop():
    """固定 ATR 止损: close 跌破 entry-ATR 触发"""
    closes = [100, 101, 102, 101, 100, 99, 98, 97, 96]
    c, h, l, a = mk(closes, atr=2.0)
    states = np.array(["up:early"] * 9)
    entries = np.array([True] + [False] * 8)
    trades = simulate_holds(c, h, l, a, states, entries, "long", "atr", exit_late=False, w=96)
    assert len(trades) == 1
    t = trades[0]
    assert t.reason == "stop"
    assert abs(t.exit_px - 98.0) < 1e-9  # entry-2
    assert abs(t.r_mult - (-1.0)) < 1e-9


def test_timeout():
    """超时: w 根内无触发 → timeout"""
    closes = [100, 101, 102, 101, 100, 101, 102, 101]
    c, h, l, a = mk(closes, atr=5.0)
    states = np.array(["up:early"] * 8)
    entries = np.array([True] + [False] * 7)
    trades = simulate_holds(c, h, l, a, states, entries, "long", "hl", exit_late=False, w=5)
    assert len(trades) == 1
    assert trades[0].reason == "timeout"


def test_short_mirror():
    """down 对称: late 退出"""
    closes = [100, 98, 96, 94, 92, 90, 88, 92, 94]
    c, h, l, a = mk(closes, atr=5.0)
    states = np.array(["down:early"] * 4 + ["down:mid"] * 3 + ["down:late"] * 2)
    entries = np.array([True] + [False] * 8)
    trades = simulate_holds(c, h, l, a, states, entries, "short", "hl", exit_late=True, w=96)
    assert len(trades) == 1
    t = trades[0]
    assert t.reason == "late"
    assert t.exit_idx == 7
    assert abs(t.r_mult - (100 - 92) / 5.0) < 1e-9  # +1.6


def test_peak_trail_exit():
    """峰值回撤退出: 先涨到峰值 → 回撤 peak_trail×ATR 触发"""
    # entry bar0 close=100, atr=1 → stop=99; peak_trail=2 (回撤 2 退出)
    closes = [100.0, 103.0, 102.0, 101.0, 99.5, 99.0]
    c, h, l, a = mk(closes, atr=1.0)
    # bar1 峰值 104 → trail=102; bar2 峰值 104.5 → trail=102.5
    h = np.array([100.0, 104.0, 104.5, 103.0, 102.0, 100.0])
    states = np.array(["up:early"] * 6)
    entries = np.array([True] + [False] * 5)
    # bar2 close=102 < 102.5 → 触发@2
    trades = simulate_holds(c, h, l, a, states, entries, "long", "atr",
                            exit_late=False, w=96, peak_trail=2.0)
    assert len(trades) == 1
    t = trades[0]
    assert t.reason == "trail"
    assert t.exit_idx == 2
    assert abs(t.exit_px - 102.5) < 1e-9
    assert abs(t.r_mult - 2.5) < 1e-9  # (102.5-100)/1


def test_peak_trail_tracks_up():
    """峰值回撤线随峰值上移: 更高峰值 → 回撤线更高"""
    closes = [100.0, 103.0, 105.0, 104.0, 103.0, 102.0]
    c, h, l, a = mk(closes, atr=1.0)
    h = np.array([100.0, 104.0, 106.0, 105.0, 104.0, 103.0])  # bar2 峰值 106
    states = np.array(["up:early"] * 6)
    entries = np.array([True] + [False] * 5)
    # bar2: peak=106 → trail=106-2=104; bar3 close=104 不 < 104; bar4 close=103 < 104 → 触发@4
    trades = simulate_holds(c, h, l, a, states, entries, "long", "atr",
                            exit_late=False, w=96, peak_trail=2.0)
    assert len(trades) == 1
    t = trades[0]
    assert t.reason == "trail"
    assert t.exit_idx == 4
    assert abs(t.exit_px - 104.0) < 1e-9
    assert abs(t.r_mult - 4.0) < 1e-9  # (104-100)/1


def test_peak_trail_no_exit_below_stop():
    """回撤线不低于初始止损: 未创新高时 trail=stop, 止损触发"""
    closes = [100.0, 99.5, 99.0, 98.5, 98.0]
    c, h, l, a = mk(closes, atr=1.0)
    h = np.array([100.0, 100.0, 99.5, 99.0, 98.5])
    states = np.array(["up:early"] * 5)
    entries = np.array([True] + [False] * 4)
    trades = simulate_holds(c, h, l, a, states, entries, "long", "atr",
                            exit_late=False, w=96, peak_trail=2.0)
    assert len(trades) == 1
    t = trades[0]
    assert t.reason == "stop"  # 未创新高, 回撤线=stop=99, close<99 触发
    assert abs(t.exit_px - 99.0) < 1e-9


def test_peak_trail_short_mirror():
    """short 镜像: 从最低点回撤触发"""
    closes = [100.0, 97.0, 98.0, 99.0, 99.5, 100.0]
    c, h, l, a = mk(closes, atr=1.0)
    l = np.array([100.0, 96.0, 97.5, 98.5, 99.0, 99.5])  # bar1 谷值 96
    states = np.array(["down:early"] * 6)
    entries = np.array([True] + [False] * 5)
    # bar1: peak(谷)=96 → trail=96+2=98; bar2 close=98 不 > 98; bar3 close=99 > 98 → 触发@3
    trades = simulate_holds(c, h, l, a, states, entries, "short", "atr",
                            exit_late=False, w=96, peak_trail=2.0)
    assert len(trades) == 1
    t = trades[0]
    assert t.reason == "trail"
    assert t.exit_idx == 3
    assert abs(t.exit_px - 98.0) < 1e-9
    assert abs(t.r_mult - 2.0) < 1e-9  # (100-98)/1


def test_invariance_real_data():
    """无未来函数: 追加 K 线不改变历史交易"""
    import os
    if not os.path.exists("data/backtest.db"):
        pytest.skip("backtest.db 不存在")
    from research.data_loader import load_candles
    from research.structures import structural_states
    data = load_candles(timeframes=("1h",))
    sym = list(data)[0]
    df = data[sym]["1h"]
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    atr = np.full(len(c), 30.0)
    for cut in (3000, 10000, len(df) - 300):
        st_full = structural_states(df)
        st_trunc = structural_states(df.iloc[:cut])
        ent_full = np.zeros(len(c), bool)
        ent_trunc = np.zeros(cut, bool)
        for i in range(cut - 1):
            if st_full[i] == "up:early" and (i == 0 or st_full[i - 1] != "up:early"):
                ent_full[i] = True
                ent_trunc[i] = True
        tr_full = simulate_holds(c, h, l, atr, st_full, ent_full, "long", "hl", False, 96)
        tr_trunc = simulate_holds(c[:cut], h[:cut], l[:cut], atr[:cut], st_trunc, ent_trunc,
                                  "long", "hl", False, 96)
        for a_, b_ in zip(tr_full, tr_trunc):
            if a_.exit_idx < cut - 50:
                assert a_.entry_idx == b_.entry_idx
                assert a_.reason == b_.reason
                assert abs(a_.r_mult - b_.r_mult) < 1e-9
