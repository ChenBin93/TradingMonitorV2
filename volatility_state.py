#!/usr/bin/env python3
"""波动状态监控 — 呼吸节奏 (2026-08-04 大改造)

用户观念: 市场的主要特征是波动, 低波动和高波动像呼吸一样 — 关注低波动
时刻, 在高波动启动前判断可能的方向并入场 (止损更小, 抓住大波动)。

实现:
  - vol_z: ATR 滚动 z (z120 短分位 / z720 长背景)
  - squeeze: 布林带宽度收窄 (压缩检测)
  - vol_start: 波动启动检测 (ATR 跳升)
  - 输出: 状态标签 + 持续时长

设计原则 (用户): 系统只报告波动状态, 不做方向预测。
"""
import numpy as np
import pandas as pd


def vol_z(atr: np.ndarray, window: int = 120) -> float:
    """ATR 滚动 z (最后一个值): < -0.5 低 / > +0.5 高"""
    s = pd.Series(atr)
    z = ((s - s.rolling(window).mean()) / s.rolling(window).std()).values
    return float(z[-1]) if len(z) else np.nan


def vol_state(z: float, low_t: float = -0.5, high_t: float = 0.5) -> str:
    if np.isnan(z):
        return "未知"
    return "低" if z < low_t else ("高" if z > high_t else "中")


def squeeze(bbw: np.ndarray, percentile: float = 30.0,
            window: int = 120) -> bool:
    """布林带宽度收窄: 当前宽度 ≤ 近 window 根的 percentile 分位"""
    w = np.asarray(bbw, float)
    if len(w) < min(30, window):
        return False
    q = np.nanpercentile(w[-min(len(w), window):], percentile)
    cur = w[-1]
    return bool(cur <= q) if not np.isnan(q) else False


def vol_start(atr: np.ndarray, window: int = 6, jump: float = 1.5) -> bool:
    """波动启动: 近 window 根 ATR 均值 ≥ jump × 前 window 根均值"""
    a = np.asarray(atr, float)
    if len(a) < 2 * window:
        return False
    recent = a[-window:].mean()
    prior = a[-2 * window:-window].mean()
    return bool(recent >= jump * prior) if prior > 0 else False


def vol_state_duration(vol_states: np.ndarray) -> int:
    """当前波动状态已持续的根数 (从末尾向前数)"""
    if len(vol_states) == 0:
        return 0
    cur = vol_states[-1]
    n = 0
    for v in vol_states[::-1]:
        if v == cur:
            n += 1
        else:
            break
    return n
