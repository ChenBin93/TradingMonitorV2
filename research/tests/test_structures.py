#!/usr/bin/env python3
"""结构状态机测试 — 黄金测试 (手算) + 未来函数不变性

黄金场景: 区间 → 向上突破(early) → 创新高(mid) → 停止创新高(late) → 破位回区间
         以及向下对称 (mirror)
构造: 温和 close 序列 + 单根尖峰 bar (high/low 延伸) — 保证 pivot 唯一可预测
"""
import numpy as np
import pandas as pd
import pytest

from research.structures import confirmed_pivots, structural_states


def mk(closes, spikes=None, hi_margin=0.5, lo_margin=0.5):
    """由 close 序列构造 K 线; spikes: {idx: (hi, lo)} 单根尖峰覆盖, None 表示不覆盖"""
    n = len(closes)
    highs = [c + hi_margin for c in closes]
    lows = [c - lo_margin for c in closes]
    for i, (hi, lo) in (spikes or {}).items():
        if hi is not None:
            highs[i] = max(highs[i], hi)
        if lo is not None:
            lows[i] = min(lows[i], lo)
    opens = [closes[0]] + list(closes[:-1])
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"open": opens, "high": highs, "low": lows,
                         "close": closes, "volume": 1.0}, index=idx)


def tri_wave(lo, hi, n_swings=4, bars_per=4):
    """三角波 close 序列 (区间): lo↔hi 来回 n_swings 次"""
    out = []
    for s in range(n_swings):
        if s % 2 == 0:
            out += list(np.linspace(lo, hi, bars_per))
        else:
            out += list(np.linspace(hi, lo, bars_per))
    return [float(x) for x in out]


def build_up_scenario():
    """区间(摆动 105/96) → 突破 105 → 创新高 109 → 停止 107 → 破位"""
    closes = tri_wave(98, 102, n_swings=3, bars_per=6)          # bars 0-17
    spikes = {
        3: (105.0, None),   # 摆动高点 105 @3 (确认 @6; 需 j>=k=3)
        8: (None, 96.0),    # 摆动低点 96 @8 (确认 @11)
        14: (104.5, None),  # 次高点 104.5 @14
    }
    closes += [98, 99, 100, 101, 102, 101]                      # bars 18-23 收尾区间
    closes += [105.5, 106.5, 107.5, 108.5]                      # bars 24-27 突破 (close>105)
    closes += [107, 106, 105.5]                                 # bars 28-30 回调
    closes += [106, 107.5, 108.5]                               # bars 31-33 继续升
    spikes[32] = (109.5, None)                                  # 新高 109.5 @32 (确认@35)
    closes += [107.5, 106.5, 106, 105.5, 105]                   # bars 34-38 回落
    closes += [105.5, 106.5, 107, 107.5]                        # bars 39-42 回升
    spikes[41] = (107.0, None)                                  # 停止创新高 107 @41 (确认@44)
    closes += [106.5, 105.5, 104.5, 103.5, 102.5]               # bars 43-47 下跌
    spikes[44] = (None, 102.0)                                  # 低点 102 @44
    closes += [101.5, 100.5]                                    # bars 48-49 破位下行
    return mk(closes, spikes)


def test_up_cycle_golden():
    df = build_up_scenario()
    st = structural_states(df)
    # 区间
    assert st[12] == "range", st[12]
    # 突破 (bar 25 起 close>105) → up:early
    assert st[25] == "up:early", st[25]
    # 新高 109.5 确认 (32+3=35 起) → up:mid
    assert st[36] == "up:mid", st[36]
    # 停止创新高 108 确认 (42+3=45 起) → up:late
    assert st[45] == "up:late", st[45]
    # 破位: close 跌破最近确认 pivot low (104.5) → range
    assert st[46] == "range", st[46]


def test_down_cycle_golden():
    df = build_up_scenario()
    # 镜像: 价格反转 (close/high/low 全部取反)
    closes = [1000 - c for c in df["close"].values]
    # 位置索引构造镜像 spikes (DatetimeIndex 上 iterrows 的 key 是 Timestamp, 用位置)
    spikes = {}
    for pos in range(len(df)):
        spikes[pos] = (1000 - df["low"].values[pos], 1000 - df["high"].values[pos])
    df2 = mk(closes, spikes)
    st = structural_states(df2)
    assert st[12] == "range", st[12]
    assert st[25] == "down:early", st[25]
    assert st[36] == "down:mid", st[36]
    assert st[45] == "down:late", st[45]


def test_pivot_confirmation_timing():
    """pivot[j] 在 j+K 前不可用 — 未确认的 105 spike (bar 3) 不得提前触发突破"""
    df = build_up_scenario()
    st = structural_states(df)
    # 105 的 spike 在 bar 3, 确认于 6 — bar 3-6 内状态不得依赖它
    assert st[3] == "warmup"  # 数据太早, 无确认 pivot
    ph, pl = confirmed_pivots(df)
    for j in np.flatnonzero(ph):
        assert df["high"].values[j] > df["high"].values[j - 3:j].max()
        assert df["high"].values[j] > df["high"].values[j + 1:j + 4].max()


def test_invariance_synthetic():
    df = build_up_scenario()
    full = structural_states(df)
    for cut in (20, 30, 40, 45):
        trunc = structural_states(df.iloc[:cut])
        for i in range(cut - 1):
            assert full[i] == trunc[i], f"cut={cut} i={i}: {full[i]} vs {trunc[i]}"


def test_invariance_real_data():
    import os
    if not os.path.exists("data/backtest.db"):
        pytest.skip("backtest.db 不存在")
    from research.data_loader import load_candles
    data = load_candles(timeframes=("1h",))
    syms = list(data)[:2]
    for sym in syms:
        df = data[sym]["1h"]
        full = structural_states(df)
        for cut in (3000, 10000, len(df) - 200):
            trunc = structural_states(df.iloc[:cut])
            for i in range(cut - 50, cut - 10):
                assert full[i] == trunc[i], f"{sym} cut={cut} i={i}: {full[i]} vs {trunc[i]}"
