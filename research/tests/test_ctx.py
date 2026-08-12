#!/usr/bin/env python3
"""ctx.py 测试 — 截断对齐 (PLAN.md §2 L2)

make_ctx 截断对齐 (长度/bar_offset/分年) + entries_from_events 全长度。
"""
import numpy as np
import pandas as pd
import pytest

from research.ctx import Ctx, entries_from_events, make_ctx


def make_df(n=400, start="2023-01-01"):
    rng = np.random.default_rng(1)
    idx = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    o = np.concatenate([[c[0]], c[:-1]])
    h = np.maximum(c, o) * (1 + rng.uniform(0, 0.003, n))
    l = np.minimum(c, o) * (1 - rng.uniform(0, 0.003, n))
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c,
                         "volume": 1.0}, index=idx)


def test_make_ctx_truncation_alignment():
    df = make_df(n=400)
    warmup = 100
    ctx = make_ctx(df, warmup, {
        "ma20": lambda d: d["close"].rolling(20).mean().values,
        "trend": lambda d: np.where(d["close"] > d["close"].rolling(20).mean(), "up", "down"),
    })
    assert ctx.bar_offset == warmup
    assert ctx.n == 400 - warmup
    # 全数组统一截断长度
    assert len(ctx.close) == len(ctx.high) == len(ctx.low) == len(ctx.open) == ctx.n
    assert len(ctx.atr) == ctx.n
    for name, seq in ctx.states.items():
        assert len(seq) == ctx.n, name
    assert len(ctx.years) == ctx.n
    # 截断坐标 ↔ 原始坐标
    assert ctx.close[0] == df["close"].iloc[warmup]
    assert ctx.close[-1] == df["close"].iloc[-1]
    # 分年归属: 截断坐标为准
    assert ctx.years[0] == df.index[warmup].year
    assert ctx.years[-1] == df.index[-1].year
    # 状态值验证: state_fns 在截断 df 上计算 (统一截断契约)
    trunc_ma = df.iloc[warmup:]["close"].rolling(20).mean().values
    assert np.allclose(ctx.states["ma20"], trunc_ma, equal_nan=True)
    # 尾窗特征不变性: 特征自身 warm-up (前 19 根 NaN) 之后与全量计算逐位一致
    full_ma = df["close"].rolling(20).mean().values[warmup:]
    assert np.allclose(ctx.states["ma20"][19:], full_ma[19:], equal_nan=True)


def test_make_ctx_state_fn_length_mismatch():
    df = make_df(n=200)
    with pytest.raises(ValueError, match="长度"):
        make_ctx(df, 50, {"bad": lambda d: np.zeros(len(d) - 1)})


def test_make_ctx_warmup_validation():
    df = make_df(n=200)
    with pytest.raises(ValueError):
        make_ctx(df, 200, {})
    with pytest.raises(ValueError):
        make_ctx(df, -1, {})


def test_entries_from_events_full_length():
    states = np.array(["range", "up", "up", "range", "up", "down", "up"])
    got = entries_from_events(states, "up")
    assert len(got) == len(states)
    assert np.array_equal(got, [False, True, False, False, True, False, True])
    got_range = entries_from_events(states, "range")
    assert np.array_equal(got_range, [True, False, False, True, False, False, False])
    # 不存在的状态 → 全 False
    assert not entries_from_events(states, "missing").any()
    # 空数组 → 空布尔
    empty = entries_from_events(np.array([], dtype=object), "x")
    assert len(empty) == 0


def test_entries_from_events_edge_cases():
    # 首元素即目标
    got = entries_from_events(np.array(["up", "up", "down", "up"]), "up")
    assert np.array_equal(got, [True, False, False, True])
    # 单元素
    got = entries_from_events(np.array(["up"]), "up")
    assert np.array_equal(got, [True])
