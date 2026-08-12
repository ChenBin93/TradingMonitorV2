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


def gbm_dataframe(n, sig, seed=0, sub=8, start=100.0, freq="1h", start_time="2023-01-01"):
    """GBM OHLC → DataFrame (索引与真实数据同构, 供研究管线直接使用)

    freq/start_time: 索引周期与起点 (默认 "1h"/"2023-01-01" 保持旧行为);
    需要锚定真实数据索引时请用 gbm_matching (锚定 ref_df.index, σ 自动估计)。
    """
    import pandas as pd
    o, h, l, c = gbm_ohlc(n, sig, seed, sub, start)
    idx = pd.date_range(pd.Timestamp(start_time, tz="UTC"), periods=n, freq=freq)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c,
                         "volume": np.ones(n)}, index=idx)


def gbm_matching(ref_df, seed=0, sub=8):
    """GBM OHLC DataFrame 与真实数据匹配 (时间锚定, 无信息对照)

    - 索引 = ref_df.index (锚定 ref 首根, 长度与 ref 相同 — 修复 GBM 年份错位)
    - σ 由 ref 对数收益样本 std (ddof=1) 估计 (替代各研究手写的 σ 估计模式)
    - 起始价格锚定 ref 首根 close
    """
    import pandas as pd
    c = ref_df["close"].values
    n = len(c)
    if n < 2:
        raise ValueError("gbm_matching: ref_df 至少需要 2 根")
    sig = float(np.std(np.diff(np.log(c)), ddof=1))
    o, h, l, c2 = gbm_ohlc(n, sig, seed, sub, start=float(c[0]))
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c2,
                         "volume": np.ones(n)}, index=ref_df.index)
