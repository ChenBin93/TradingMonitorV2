#!/usr/bin/env python3
"""对拍测试 — numpy 参考引擎 vs vectorbt 在真实 backtest.db 数据上必须完全一致

这是"胜率计算正确"的最终防线: 两个独立实现产出相同结果, 才算通过。
"""
import os

import numpy as np
import pytest

from market_phase import _atr_series
from research.data_loader import load_candles, verify
from research.outcome import evaluate_forward, evaluate_forward_vbt

DB = "data/backtest.db"

pytestmark = pytest.mark.skipif(not os.path.exists(DB), reason="backtest.db 不存在")


def test_cross_check_real_data_1h():
    data = load_candles(timeframes=("1h",))
    assert data, "backtest.db 无 1h 数据"
    checked = 0
    for sym, tfs in data.items():
        df = tfs["1h"]
        problems = verify(df, sym, "1h")
        assert not problems, f"{sym} 数据不干净: {problems}"
        c = df["close"].values
        h = df["high"].values
        l = df["low"].values
        atr = _atr_series(df)
        n = len(c)
        entries = np.zeros(n, bool)
        entries[::97] = True  # 确定性抽样入场
        for direction in ("long", "short"):
            o1, r1 = evaluate_forward(c, h, l, atr, entries, direction, t_mult=1.0, w=24)
            o2, r2 = evaluate_forward_vbt(c, h, l, atr, entries, direction, t_mult=1.0, w=24)
            assert o1 == o2, f"{sym} {direction}: {o1} vs {o2}"
            assert len(r1) == len(r2), f"{sym} {direction}: 记录数 {len(r1)} vs {len(r2)}"
            for a, b in zip(r1, r2):
                assert a.entry_idx == b.entry_idx
                assert a.outcome == b.outcome
                assert abs(a.entry_px - b.entry_px) < 1e-6
                if a.outcome in ("win", "loss"):
                    assert a.exit_idx == b.exit_idx, f"{sym} {direction}: {a} vs {b}"
                    # 成交价差异仅来自跳空: 胜→vbt 按开盘价更优; 负→vbt 按开盘价更差
                    if a.outcome == "win":
                        ok = b.exit_px >= a.exit_px - 1e-6 if direction == "long" \
                            else b.exit_px <= a.exit_px + 1e-6
                    else:
                        ok = b.exit_px <= a.exit_px + 1e-6 if direction == "long" \
                            else b.exit_px >= a.exit_px - 1e-6
                    assert ok, f"{sym} {direction}: {a} vs {b}"
        checked += 1
    assert checked >= 10, f"对拍覆盖不足: 只有 {checked} 个标的"


def test_5m_cross_check_small_sample():
    """5m 数据抽 2 标的 × 前 20000 根对拍 (5m 全量 276s 太慢, 留到研究时再全跑)"""
    data = load_candles(timeframes=("5m",))
    if not data:
        pytest.skip("无 5m 数据")
    syms = list(data)[:2]
    for sym in syms:
        df = data[sym]["5m"].iloc[:20000]
        c = df["close"].values
        h = df["high"].values
        l = df["low"].values
        atr = _atr_series(df)
        entries = np.zeros(len(c), bool)
        entries[::500] = True
        o1, _ = evaluate_forward(c, h, l, atr, entries, "long", t_mult=1.0, w=24)
        o2, _ = evaluate_forward_vbt(c, h, l, atr, entries, "long", t_mult=1.0, w=24)
        assert o1 == o2, f"{sym} 5m: {o1} vs {o2}"
