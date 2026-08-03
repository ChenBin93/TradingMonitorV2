#!/usr/bin/env python3
"""结构状态机 — 用户模型: 震荡/趋势二元 + 趋势初/中/末期 (无未来函数)

定义 (v1):
  K = 3: pivot 左右确认根数 — pivot[j] 在 bar j+K 已收盘后才可用
  震荡带 band: 最近确认的 pivot 高点 / 低点构成的区间
  状态: warmup / range / up:early|mid|late / down:early|mid|late

规则:
  range:      close 在震荡带内 (band 随新确认 pivot 动态更新)
  up:early:   close 收盘突破 band 高点, 尚无新高 pivot
  up:mid:     突破后确认 pivot 高点 > 突破点 (结构创新高), 之后继续创新高保持
  up:late:    mid 之后最新确认 pivot 高点 ≤ 前一个 (停止创新高), 之后创新高回 mid
  up → range: close 跌破最近确认 pivot 低点 (趋势支撑破位)
  down 对称

无未来函数: bar t 只使用 pivot[j] (j+K <= t) 和 close[<=t];
  状态机确定性, 只用过去数据 — 追加 K 线不改变历史状态 (不变性测试锁定)
"""
import numpy as np

from market_phase import _atr_series

K = 3


def confirmed_pivots(df, k=K):
    """pivot 判定 (全样本计算; 但调用方只在 j+k <= t 时使用 — 确认时序保证无未来)

    严格定义: high[j] 严格大于左右各 k 根的高点 (相等不算, 避免相邻双 pivot)
    """
    n = len(df)
    hi = df["high"].values
    lo = df["low"].values
    pivot_hi = np.zeros(n, bool)
    pivot_lo = np.zeros(n, bool)
    for j in range(k, n - k):
        if hi[j] > hi[j - k:j].max() and hi[j] > hi[j + 1:j + k + 1].max():
            pivot_hi[j] = True
        if lo[j] < lo[j - k:j].min() and lo[j] < lo[j + 1:j + k + 1].min():
            pivot_lo[j] = True
    return pivot_hi, pivot_lo


def structural_states(df, k=K):
    """结构状态序列 (逐 bar, 只用已收盘数据)

    返回 np.ndarray[str]: warmup/range/up:early/up:mid/up:late/
                          down:early/down:mid/down:late
    """
    n = len(df)
    hi = df["high"].values
    lo = df["low"].values
    cl = df["close"].values
    atr = _atr_series(df)
    pivot_hi, pivot_lo = confirmed_pivots(df, k)

    ph = [j for j in range(n) if pivot_hi[j]]
    pl = [j for j in range(n) if pivot_lo[j]]

    states = np.full(n, "warmup", dtype=object)
    iph = ipl = 0
    last_ph = None  # (pos, val) 最新确认 pivot high
    last_pl = None  # (pos, val) 最新确认 pivot low
    phase = "range"  # range/up/down
    stage = None    # early/mid/late
    ref = 0.0       # 突破点 (up: band_hi, down: band_lo)
    prev_piv = None  # 突破后上一确认 pivot 值 (阶段推进)
    break_ts = -1    # 最近突破 bar
    proc_ph = -1     # 已处理的最新 pivot high 位置 (防重复推进)
    proc_pl = -1

    for t in range(n):
        # 推进可用 pivots (确认于 j+k)
        while iph < len(ph) and ph[iph] + k <= t:
            last_ph = (ph[iph], hi[ph[iph]])
            iph += 1
        while ipl < len(pl) and pl[ipl] + k <= t:
            last_pl = (pl[ipl], lo[pl[ipl]])
            ipl += 1

        if last_ph is None or last_pl is None or not np.isfinite(atr[t]) or atr[t] <= 0:
            states[t] = "warmup"
            continue

        if phase == "range":
            band_hi, band_lo = last_ph[1], last_pl[1]
            if cl[t] > band_hi:
                phase, stage = "up", "early"
                ref, prev_piv, break_ts = band_hi, None, t
                proc_ph = proc_pl = -1
            elif cl[t] < band_lo:
                phase, stage = "down", "early"
                ref, prev_piv, break_ts = band_lo, None, t
                proc_ph = proc_pl = -1
            states[t] = "range"
            continue

        if phase == "up":
            if cl[t] < last_pl[1]:  # 破位: 跌破最近确认 pivot low
                phase, stage = "range", None
                proc_ph = proc_pl = -1
                states[t] = "range"
                continue
            if last_ph[0] > break_ts and last_ph[0] > proc_ph:  # 新确认 pivot high
                proc_ph = last_ph[0]
                v = last_ph[1]
                if prev_piv is None:
                    if v > ref:
                        stage = "mid"
                        prev_piv = v
                else:
                    if v > prev_piv:
                        stage = "mid"
                    elif stage == "mid":
                        stage = "late"
                    prev_piv = v
            states[t] = f"up:{stage}"
            continue

        # down
        if cl[t] > last_ph[1]:  # 破位: 上破最近确认 pivot high
            phase, stage = "range", None
            proc_ph = proc_pl = -1
            states[t] = "range"
            continue
        if last_pl[0] > break_ts and last_pl[0] > proc_pl:  # 新确认 pivot low
            proc_pl = last_pl[0]
            v = last_pl[1]
            if prev_piv is None:
                if v < ref:
                    stage = "mid"
                    prev_piv = v
            else:
                if v < prev_piv:
                    stage = "mid"
                elif stage == "mid":
                    stage = "late"
                prev_piv = v
        states[t] = f"down:{stage}"

    return states
