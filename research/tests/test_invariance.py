#!/usr/bin/env python3
"""未来函数不变性测试

核心规则: 往序列后面追加 K 线, 历史位置的特征值必须不变
  feature(series[:i+k])[i] == feature(series[:i])[i]
这能自动抓出 center 对齐窗口、跨序列错位等一切 lookahead。
"""
import numpy as np
import pandas as pd
import pytest

from market_phase import _adx_series, _atr_series


def make_df(n=1500):
    rng = np.random.default_rng(42)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    h = np.maximum(c * (1 + rng.uniform(0, 0.003, n)), c)
    l = np.minimum(c * (1 - rng.uniform(0, 0.003, n)), c)
    o = np.concatenate([[c[0]], c[:-1]])
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": 1.0}, index=idx)


def assert_invariant(fn, df, cuts=None):
    if cuts is None:
        cuts = (100, 300, 700, len(df) - 50)
    full = fn(df)
    for cut in cuts:
        trunc = fn(df.iloc[:cut])
        a = full[cut - 1]
        b = trunc[-1]
        if np.isnan(a) and np.isnan(b):
            continue
        assert np.allclose(a, b, equal_nan=True), \
            f"cut={cut}: full={a} truncated={b} — 特征值被未来数据改变 (lookahead!)"


def test_atr_series_invariant():
    assert_invariant(lambda df: _atr_series(df), make_df())


def test_adx_series_invariant():
    assert_invariant(lambda df: _adx_series(df), make_df())


def test_rolling_mean_invariant():
    assert_invariant(lambda df: df["close"].rolling(20).mean().values, make_df())


def test_ewm_invariant():
    assert_invariant(lambda df: df["close"].ewm(span=20, adjust=False).mean().values, make_df())


def test_rolling_roc_invariant():
    assert_invariant(lambda df: (df["close"] / df["close"].shift(5) - 1).values, make_df())


# ── 测试本身必须能抓到 lookahead: center 对齐窗口应该失败 ──
def test_center_aligned_is_caught():
    df = make_df()
    broken = lambda d: d["close"].rolling(5, center=True).mean().values
    with pytest.raises(AssertionError):
        assert_invariant(broken, df)


def test_shift_backward_is_caught():
    """用未来收盘价的特征 (close.shift(-1)) 必须被抓到"""
    df = make_df()
    broken = lambda d: (d["close"].shift(-1) / d["close"] - 1).values
    with pytest.raises(AssertionError):
        assert_invariant(broken, df)
