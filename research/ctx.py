#!/usr/bin/env python3
"""对齐安全上下文 (research/PLAN.md §2) — 截断对齐的唯一构造路径

make_ctx 是研究脚本构造特征/状态上下文的唯一入口 (check_study L3 禁手动切片):
内部统一 df.iloc[warmup:] 后再取 values, 脚本永远见不到未对齐组合。
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from market_phase import _atr_series


@dataclass
class Ctx:
    """截断对齐后的研究上下文 — 全部数组长度 = n (统一截断坐标)

    - close/high/low/open/atr: 价格数组, 长度 n
    - states: {名称: 状态序列}, 每个长度 n
    - years : 长度 n, 分年归属以截断坐标为准 (years[i] = 原始索引 warmup+i 的年份)
    - bar_offset: 截断索引 → 原始索引的偏移 (= warmup)
    - n     : 截断后长度 = len(df) - warmup
    """
    close: np.ndarray
    high: np.ndarray
    low: np.ndarray
    open: np.ndarray
    atr: np.ndarray
    states: dict = field(default_factory=dict)
    years: np.ndarray = field(default_factory=lambda: np.array([], int))
    bar_offset: int = 0
    n: int = 0


def make_ctx(df: pd.DataFrame, warmup: int, state_fns: dict) -> Ctx:
    """从 DataFrame 构造 Ctx — 唯一截断对齐路径

    - 内部统一 df.iloc[warmup:] 后再取 values (价格/atr/states/years 全走截断)
    - atr 用 market_phase._atr_series 计算 (左对齐 ewm, 无未来函数)
    - state_fns: {名称: fn(df) -> 数组(长度 len(df))}, fn 在**截断后**的 df 上计算
      (统一截断契约: 脚本永远见不到 warmup 之前的数据)。fn 只允许用已收盘/尾窗
      数据; 特征自身 warm-up (如 rolling 前 window-1 根 NaN) 需由 warmup 覆盖。
      正确实现满足不变性: 在截断 df 上计算与全量计算逐位一致 (仅差特征自身 warm-up)。
    - years 取自 DatetimeIndex 的年份 (截断坐标); 非 DatetimeIndex → ValueError
    """
    if warmup < 0 or warmup >= len(df):
        raise ValueError(f"warmup={warmup} 越界 (len(df)={len(df)})")
    trunc = df.iloc[warmup:]
    n = len(trunc)
    close = trunc["close"].values.astype(float)
    high = trunc["high"].values.astype(float)
    low = trunc["low"].values.astype(float)
    open_px = trunc["open"].values.astype(float)
    atr = np.asarray(_atr_series(trunc), float)

    states = {}
    for name, fn in state_fns.items():
        seq = np.asarray(fn(trunc))
        if len(seq) != n:
            raise ValueError(
                f"state_fn '{name}' 返回长度 {len(seq)} != 截断长度 {n}")
        states[name] = seq

    if not isinstance(trunc.index, pd.DatetimeIndex):
        raise ValueError("df.index 必须是 DatetimeIndex (分年归属需要)")
    years = np.fromiter((ts.year for ts in trunc.index), dtype=int, count=n)

    return Ctx(close=close, high=high, low=low, open=open_px, atr=atr,
               states=states, years=years, bar_offset=warmup, n=n)


def entries_from_events(states, target) -> np.ndarray:
    """状态进入事件 → 全长度布尔: target 状态每段连续出现的第 1 根为 True

    states 长度 n → 返回同长度布尔数组。
    """
    states = np.asarray(states)
    n = len(states)
    is_t = states == target
    first = np.zeros(n, bool)
    if n > 0:
        first[0] = is_t[0]
        first[1:] = is_t[1:] & ~is_t[:-1]
    return first
