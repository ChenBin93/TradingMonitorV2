#!/usr/bin/env python3
"""状态特征 — 统计定义状态序列 (A2 实现抽出, A3 复用)

- 趋势态: live market_phase.classify 逐 bar 复现 (trend_up/trend_down/range/transition + 阶段)
- 波动态: ATR 滚动 z 三分位 (低/中/高); 短分位 120 根判当前, 长分位 720 根判背景

无未来函数: 特征逐 bar 用已收盘数据 (rolling/ewm 左对齐, 已过不变性测试);
  波动 z 用滚动窗口 (只含过去数据)
"""
import numpy as np
import pandas as pd

from market_phase import _adx_series, _atr_series, classify


def state_series(df):
    """逐 bar 趋势态 (与 live analyze_market_state 语义一致, 向量化)

    返回 (states: np.ndarray[str], feats: DataFrame)
    states 格式: range / transition / trend_up:{early|accelerate|late} / trend_down:{...}
    """
    c = pd.Series(df["close"].values)
    o = pd.Series(df["open"].values)
    v = pd.Series(df["volume"].values)
    atr = pd.Series(_atr_series(df))
    adx = pd.Series(_adx_series(df))
    ma20 = c.rolling(20).mean()
    ma60 = c.rolling(60).mean()
    dev = (c - ma20) / atr
    slope = (ma20 - ma20.shift(10)) / atr
    spread = (ma20 - ma60) / atr
    mom = (c - c.shift(10)) / atr
    adx_prev = adx.shift(10)
    body = (c - o).abs()
    body_recent = body.rolling(3).mean()
    body_prior = body.rolling(13).mean().shift(3)
    vol_ratio = v / v.rolling(20).mean()
    bbw = c.rolling(20).std() / c.rolling(20).mean()

    states = []
    n = len(df)
    for i in range(n):
        r = classify(atr.iloc[i], adx.iloc[i], adx_prev.iloc[i], slope.iloc[i],
                     spread.iloc[i], dev.iloc[i], mom.iloc[i],
                     body_recent.iloc[i], body_prior.iloc[i])
        s = r["state"]
        if r["stage"] and s.startswith("trend"):
            s = f"{s}:{r['stage']}"
        states.append(s)

    feats = pd.DataFrame({
        "adx": adx.values, "dev": dev.values, "slope": slope.values,
        "mom": mom.values, "body_ratio": (body_recent / body_prior).values,
        "bbw": bbw.values, "vol_ratio": vol_ratio.values,
        "atr_c": atr.values / c.values,
    }, index=df.index)
    return np.array(states), feats


def vol_z_states(atr, window):
    """ATR 滚动 z 三分位状态 (低/中/高) + z 值序列

    阈值 ±0.5 (正态近似三分位); 前 window-1 根为 NaN (z 无效)
    """
    s = pd.Series(atr)
    z = ((s - s.rolling(window).mean()) / s.rolling(window).std()).values
    st = np.where(z < -0.5, "低", np.where(z > 0.5, "高", "中"))
    st = np.where(np.isnan(z), "未知", st)
    return st, z
