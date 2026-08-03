#!/usr/bin/env python3
"""关键位检测器测试 — 黄金测试 (手算) + 确认时序 + 不变性 + live 对照

场景: 聚类带位 (两相近高点/低点聚类) / 孤立 pivot 不入聚类 / 触碰 / 破位 / 时序
"""
import numpy as np
import pandas as pd
import pytest

from research.levels import (active_levels, cluster_levels, close_breakout,
                             confirmed_swings, pivot_levels, touches_at)


def mk(closes, spikes, atr_val=1.0, hi_margin=0.5, lo_margin=0.5):
    n = len(closes)
    highs = [c + hi_margin for c in closes]
    lows = [c - lo_margin for c in closes]
    for i, (hi, lo) in (spikes or {}).items():
        if hi is not None:
            highs[i] = max(highs[i], hi)
        if lo is not None:
            lows[i] = min(lows[i], lo)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"open": [closes[0]] + list(closes[:-1]),
                         "high": highs, "low": lows, "close": closes,
                         "volume": np.ones(n)}, index=idx), atr_val


def tri_wave(lo, hi, n_swings=4, bars_per=4):
    out = []
    for s in range(n_swings):
        if s % 2 == 0:
            out += list(np.linspace(lo, hi, bars_per))
        else:
            out += list(np.linspace(hi, lo, bars_per))
    return [float(x) for x in out]


def build():
    """区间 98-102 (24 根); spikes: 高点 105.5@3, 105.8@9 (聚类), 110@15 (孤立);
    低点 96.0@6, 96.3@12 (聚类), 92@18 (孤立)"""
    closes = tri_wave(98, 102, n_swings=6, bars_per=4)
    spikes = {
        3: (105.5, None), 9: (105.8, None), 15: (110.0, None),
        6: (None, 96.0), 12: (None, 96.3), 18: (None, 92.0),
    }
    return mk(closes, spikes)


def test_cluster_levels_golden():
    df, atr = build()
    # tol = 0.3 × atr(1.0) = 0.3 → 105.5/105.8 聚类 (差0.3), 96.0/96.3 聚类; 110/92 孤立
    lvls = cluster_levels(df["high"].values, df["low"].values,
                          np.full(len(df), atr), min_touch=2)
    res = [l for l in lvls if l.side == "resistance"]
    sup = [l for l in lvls if l.side == "support"]
    assert len(res) == 1 and len(sup) == 1
    assert abs(res[0].price - 105.65) < 1e-9
    assert abs(sup[0].price - 96.15) < 1e-9
    assert res[0].touch_count == 2 and sup[0].touch_count == 2
    # 确认时序: 聚类完成 = 最后一 pivot 确认 (9+3=12, 12+3=15)
    assert res[0].confirm_at == 12
    assert sup[0].confirm_at == 15


def test_confirmation_timing():
    """位在确认前不可用 (无未来函数)"""
    df, atr = build()
    lvls = cluster_levels(df["high"].values, df["low"].values,
                          np.full(len(df), atr), min_touch=2)
    res = [l for l in lvls if l.side == "resistance"][0]
    # bar 11 (确认于 12) 前不可用
    assert len(active_levels(lvls, 11)) == 0
    assert len(active_levels(lvls, 12)) == 1


def test_pivot_levels_all_swings():
    df, atr = build()
    lvls = pivot_levels(df["high"].values, df["low"].values)
    res = [l for l in lvls if l.side == "resistance"]
    sup = [l for l in lvls if l.side == "support"]
    # 3 个高点 swing + 3 个低点 swing (三角波内部也可能有小 swing, 至少 3 个)
    assert len(res) >= 3 and len(sup) >= 3
    prices = sorted(l.price for l in res)
    assert prices[-1] == 110.0  # 孤立高点也在


def test_touches_and_breakout():
    df, atr = build()
    lvls = cluster_levels(df["high"].values, df["low"].values,
                          np.full(len(df), atr), min_touch=2)
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    # 构造触碰: bar 20 高点触及 105.5+ (位带 [105.5, 105.8])
    h[20] = 105.7
    hit = touches_at(lvls, 20, l, h)
    assert len(hit) == 1 and hit[0].side == "resistance"
    # 收盘破位: bar 22 close 高于带外侧 105.8
    c[22] = 106.5
    brk = close_breakout(lvls, 22, c)
    assert any(b.side == "resistance" for b in brk)


def test_invariance_real_data():
    """追加 K 线不改变历史位结构 (confirm_at/touch_count/price 不变)"""
    import os
    if not os.path.exists("data/backtest.db"):
        pytest.skip("backtest.db 不存在")
    from market_phase import _atr_series
    from research.data_loader import load_candles
    data = load_candles(timeframes=("1h",))
    sym = list(data)[0]
    df = data[sym]["1h"]
    for cut in (3000, 10000, len(df) - 300):
        sub = df.iloc[:cut]
        full_lv = cluster_levels(df["high"].values, df["low"].values,
                                 _atr_series(df), min_touch=2)
        cut_lv = cluster_levels(sub["high"].values, sub["low"].values,
                                _atr_series(sub), min_touch=2)
        f = {(l.price, l.side, l.confirm_at, l.touch_count) for l in full_lv if l.confirm_at < cut - 100}
        g = {(l.price, l.side, l.confirm_at, l.touch_count) for l in cut_lv if l.confirm_at < cut - 100}
        # 截断后新增位不要求一致, 但截断前的位必须一致 (价格相同, 聚类可合并)
        assert f <= g, f"cut={cut}: 历史位结构被改变 ({len(f - g)} 个位消失)"


def test_live_consistency():
    """live find_swing_levels vs 研究聚类 — 最近位价格对照 (差异应小)"""
    import os
    if not os.path.exists("data/backtest.db"):
        pytest.skip("backtest.db 不存在")
    from market_phase import _atr_series
    from research.data_loader import load_candles
    from support_resistance import find_swing_levels
    data = load_candles(timeframes=("1h",))
    sym = list(data)[0]
    df = data[sym]["1h"]
    tail = df.tail(600)
    live = find_swing_levels(tail, lookback=600)
    # 研究版: 只看最近可用位 (在数据末尾时刻)
    lvls = cluster_levels(df["high"].values, df["low"].values,
                          _atr_series(df), min_touch=2)
    t = len(df) - 1
    act = [l for l in active_levels(lvls, t) if l.age_bars < 300]
    if not live or not act:
        pytest.skip("无位可对照")
    live_prices = sorted(l.price for l in live)
    act_prices = sorted(l.price for l in act)
    # 主要位应大致对齐: live 位中 ≥50% 在研究版 3×ATR 内有对应
    # (live 用未确认 swing + 粗糙 ATR 近似, 差异容忍 3×ATR)
    atr = np.median(_atr_series(df)[-100:])
    matched = sum(1 for lp in live_prices[:8] if any(abs(lp - ap) < 3 * atr for ap in act_prices))
    assert matched >= max(1, len(live_prices[:8]) * 0.5), \
        f"live 位与研究版共识不足: {matched}/{len(live_prices[:8])}"
