#!/usr/bin/env python3
"""合成市场生成器 — 无信息对照市场 (真实 OHLC 结构, 无未来函数)

2026-08-03 修正: 此前各研究脚本的 GBM 用 high=close×(1±2σ), open=close 的
简化构造 — 乘法结构 + open 折叠使 intrabar 判定产生方向偏差 (无条件 1:1
基线 51.3% 而非理论 50%)。本模块用 bar 内子步随机游走生成真实 OHLC:
  open[j] = close[j-1] (连续市场), bar 内 sub 步 → high/low/close

用途: 随机游走对照 (性质测试要求: 无信息市场 1:1 基线必须 ≈ 理论值 50%)
"""
import numpy as np


def gbm_ohlc(n, sig, seed=0, sub=8, start=100.0):
    """几何布朗运动 OHLC — bar 内 sub 子步随机游走

    返回 (open, high, low, close) 数组
    """
    rng = np.random.default_rng(seed)
    sig_sub = sig / np.sqrt(sub)
    o = np.empty(n)
    h = np.empty(n)
    l = np.empty(n)
    c = np.empty(n)
    prev = start
    for j in range(n):
        o[j] = prev
        p = prev
        hi = lo = prev
        for _ in range(sub):
            p *= np.exp(rng.normal(0.0, sig_sub))
            if p > hi:
                hi = p
            if p < lo:
                lo = p
        h[j], l[j], c[j] = hi, lo, p
        prev = p
    return o, h, l, c


def gbm_dataframe(n, sig, seed=0, sub=8, start=100.0):
    """GBM OHLC → DataFrame (索引与真实数据同构, 供研究管线直接使用)"""
    import pandas as pd
    o, h, l, c = gbm_ohlc(n, sig, seed, sub, start)
    idx = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c,
                         "volume": np.ones(n)}, index=idx)
