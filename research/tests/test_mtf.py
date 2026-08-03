#!/usr/bin/env python3
"""MTF 对齐测试 — 高位 bar 未收盘不得使用 (未来函数边界)

规则: 低位 bar 时间 t 只能使用 open + tf 时长 <= t 的高位 bar
"""
import numpy as np
import pandas as pd

from research.data_loader import align_higher


def _df(ts_list, close_list):
    idx = pd.DatetimeIndex(ts_list, tz="UTC")
    return pd.DataFrame({"close": close_list, "high": close_list,
                         "low": close_list, "open": close_list}, index=idx)


def test_forming_4h_bar_not_used():
    # 4H bar: 00:00(收盘 04:00), 04:00(收盘 08:00)
    higher = _df(["2024-01-01 00:00", "2024-01-01 04:00"], [100.0, 200.0])
    # 1H 低位 bar
    lower = pd.DatetimeIndex([
        "2024-01-01 03:00",   # 00:00 的 4H bar 收盘于 04:00 → 尚无已收盘 4H → NaN
        "2024-01-01 04:00",   # 00:00 bar 恰在 04:00 收盘 → 100
        "2024-01-01 07:00",   # 仍用 00:00 的 (100)
        "2024-01-01 08:00",   # 04:00 bar 已收盘 → 200
        "2024-01-02 00:00",   # 最晚一根 04:00 的 (200)
    ], tz="UTC")
    res = align_higher(higher, "4h", lower)
    assert np.isnan(res["close"].iloc[0])
    assert res["close"].tolist()[1:] == [100.0, 100.0, 200.0, 200.0]


def test_no_closed_higher_bar_yet_is_nan():
    higher = _df(["2024-01-01 04:00"], [200.0])  # 收盘 08:00
    lower = pd.DatetimeIndex(["2024-01-01 03:00"], tz="UTC")
    res = align_higher(higher, "4h", lower)
    assert np.isnan(res["close"].iloc[0])


def test_exact_close_time_usable():
    higher = _df(["2024-01-01 00:00"], [100.0])  # 4H bar 收盘 04:00
    lower = pd.DatetimeIndex(["2024-01-01 04:00"], tz="UTC")
    res = align_higher(higher, "4h", lower)
    assert res["close"].iloc[0] == 100.0


def test_select_cols():
    higher = pd.DataFrame(
        {"close": [100.0, 200.0], "atr": [1.0, 2.0]},
        index=pd.DatetimeIndex(["2024-01-01 00:00", "2024-01-01 04:00"], tz="UTC"))
    lower = pd.DatetimeIndex(["2024-01-01 08:00"], tz="UTC")
    res = align_higher(higher, "4h", lower, cols=["atr"])
    assert list(res.columns) == ["atr"]
    assert res["atr"].iloc[0] == 2.0


def test_empty_higher_returns_nan():
    higher = _df([], [])
    lower = pd.DatetimeIndex(["2024-01-01 00:00"], tz="UTC")
    res = align_higher(higher, "4h", lower)
    assert np.isnan(res["close"].iloc[0])
