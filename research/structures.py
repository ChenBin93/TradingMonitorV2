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
    向量化 (rolling max, 2026-08-03 优化 — 原逐点切片循环慢 ~3s/标的)
    """
    n = len(df)
    hi = df["high"].values
    lo = df["low"].values
    pivot_hi = np.zeros(n, bool)
    pivot_lo = np.zeros(n, bool)
    if n <= 2 * k:
        return pivot_hi, pivot_lo
    import pandas as pd
    sh = pd.Series(hi)
    sl = pd.Series(lo)
    rmax = sh.rolling(k, min_periods=1).max()
    rmin = sl.rolling(k, min_periods=1).min()
    left_max_h = rmax.shift(1).values   # max(hi[j-k:j])
    right_max_h = rmax.shift(-k).values  # max(hi[j+1:j+k+1])
    left_min_l = rmin.shift(1).values   # min(lo[j-k:j])
    right_min_l = rmin.shift(-k).values  # min(lo[j+1:j+k+1])
    idx = np.arange(n)
    valid = (idx >= k) & (idx + k < n)
    pivot_hi = valid & (hi > left_max_h) & (hi > right_max_h)
    pivot_lo = valid & (lo < left_min_l) & (lo < right_min_l)
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


def dow_segments(df, k=K):
    """道氏趋势段 + 回撤事件 (v2 扩展, 无未来函数)

    道氏视角 (用户): 上升趋势 = 高点和低点持续抬升 (HH+HL); 趋势结束 =
    价格打破前低 (跌破最近确认 HL); 之后进入震荡 (range) 或反转。

    基于 v1 状态机扩展:
      - up 段内跟踪 HH/HL 序列 (pivot 确认即可用, K=3)
      - 回撤事件 = 确认新 HL (up) / 新 LH (down) 时记录 (深度/时长)
      - 段生命周期: 起止 bar/方向/时长/幅度 (ATR 归一)/HH·HL 计数
      - 破位参照: 最近确认 HL (道氏) 而非任意 pivot low — 比 v1 严格

    返回 dict:
      states:  逐 bar "range/up/down/warmup" (段级, 不含阶段)
      segs:    list[dict] start/end/bars/direction/amp_atr/n_hh/n_hl
      retraces: list[dict] bar/direction/depth_atr/dur_bars (HL/LH 确认点)
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
    segs, retraces = [], []
    iph = ipl = 0
    last_ph = last_pl = None
    phase = "range"
    seg = None
    peak_pos = peak_val = None       # 段内最高 pivot high (up)
    trough_pos = trough_val = None   # 段内最低 pivot low (down)
    last_hl = None                   # 最近确认 HL 值 (up 段回撤低点序列)
    last_lh = None                   # 最近确认 LH 值 (down 段)
    n_hh = n_hl = 0
    ref = 0.0
    break_ts = -1
    proc_ph = proc_pl = -1

    def close_seg(end):
        nonlocal seg, peak_pos, peak_val, trough_pos, trough_val
        nonlocal last_hl, last_lh, n_hh, n_hl
        if seg is not None:
            if seg["direction"] == "up":
                amp = (peak_val - trough_val) if (peak_val and trough_val) else 0.0
            else:
                amp = (trough_val - peak_val) if (peak_val and trough_val) else 0.0
            seg.update(end=end, bars=end - seg["start"] + 1,
                       amp_atr=amp / max(1e-9, atr[seg["start"]]),
                       n_hh=n_hh, n_hl=n_hl)
            segs.append(seg)
            seg = None
        peak_pos = peak_val = trough_pos = trough_val = None
        last_hl = last_lh = None
        n_hh = n_hl = 0

    for t in range(n):
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
            if cl[t] > last_ph[1]:
                phase = "up"
                seg = dict(start=t, direction="up")
                ref, break_ts = last_ph[1], t
                proc_ph = proc_pl = -1
            elif cl[t] < last_pl[1]:
                phase = "down"
                seg = dict(start=t, direction="down")
                ref, break_ts = last_pl[1], t
                proc_ph = proc_pl = -1
            states[t] = "range"
            continue

        if phase == "up":
            break_level = last_hl if last_hl is not None else (
                last_pl[1] if last_pl is not None else None)
            if break_level is not None and cl[t] < break_level:
                close_seg(t)
                phase = "range"
                states[t] = "range"
                continue
            if seg is None:  # 理论不可达, 防御
                phase = "range"
                states[t] = "range"
                continue
            if last_ph is not None and last_ph[0] > break_ts and last_ph[0] > proc_ph:
                proc_ph = last_ph[0]
                v = last_ph[1]
                if peak_val is None or v > peak_val:
                    peak_val, peak_pos = v, last_ph[0]
                if v > ref:
                    n_hh += 1
            if last_pl is not None and last_pl[0] > break_ts and last_pl[0] > proc_pl:
                proc_pl = last_pl[0]
                v = last_pl[1]
                if last_hl is None or v > last_hl:
                    # 新 HL: 回撤事件
                    if last_hl is not None and peak_val is not None and peak_pos is not None:
                        retraces.append(dict(bar=last_pl[0], direction="up",
                                             depth_atr=(peak_val - v) / max(1e-9, atr[last_pl[0]]),
                                             dur_bars=last_pl[0] - peak_pos,
                                             peak_val=peak_val))
                    last_hl = v
                    if trough_val is None or v < trough_val:
                        trough_val, trough_pos = v, last_pl[0]
                    n_hl += 1
            states[t] = "up"
            continue

        # down
        break_level = last_lh if last_lh is not None else (
            last_ph[1] if last_ph is not None else None)
        if break_level is not None and cl[t] > break_level:
            close_seg(t)
            phase = "range"
            states[t] = "range"
            continue
        if seg is None:
            phase = "range"
            states[t] = "range"
            continue
        if last_pl is not None and last_pl[0] > break_ts and last_pl[0] > proc_pl:
            proc_pl = last_pl[0]
            v = last_pl[1]
            if trough_val is None or v < trough_val:
                trough_val, trough_pos = v, last_pl[0]
            if v < ref:
                n_hl += 1
        if last_ph is not None and last_ph[0] > break_ts and last_ph[0] > proc_ph:
            proc_ph = last_ph[0]
            v = last_ph[1]
            if last_lh is None or v < last_lh:
                if last_lh is not None and trough_val is not None and trough_pos is not None:
                    retraces.append(dict(bar=last_ph[0], direction="down",
                                         depth_atr=(v - trough_val) / max(1e-9, atr[last_ph[0]]),
                                         dur_bars=last_ph[0] - trough_pos,
                                         trough_val=trough_val))
                last_lh = v
                if peak_val is None or v > peak_val:
                    peak_val, peak_pos = v, last_ph[0]
                n_hh += 1
        states[t] = "down"

    close_seg(n - 1)
    return dict(states=states, segs=segs, retraces=retraces)
