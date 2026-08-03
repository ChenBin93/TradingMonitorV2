#!/usr/bin/env python3
"""持仓模拟器 — 趋势持有策略 (early 入场, 状态/止损退出, 无未来函数)

入场: 状态进入 bar 收盘 (entry = close[i])
止损 (long):
  hl  : 最近已确认 swing low (HL), 随新确认 pivot low 上移 (跟踪止损)
  atr : 入场时 1×ATR, 固定
触发: close 收盘跌破止损线 (与结构状态机破位语义一致, 避免插针)
退出:
  exit_late=True : 状态变为 late 时 bar 收盘平仓
  止损触发      : 按止损价成交 (保守, gap 不计优势)
  超时 (w 根)   : 强制平仓 → expired (单独统计, 计入分子分母可选)
R 倍数 = (exit - entry) / 初始止损距离 (long), 符号化; 跟踪止损移动不影响 R 基准

无未来函数: bar t 只用 close[<=t] 和 pivot (j+k<=t 已确认) 与 atr[<=t]
"""
from dataclasses import dataclass

import numpy as np

from research.structures import K, confirmed_pivots


@dataclass
class HoldTrade:
    entry_idx: int
    exit_idx: int
    entry_px: float
    exit_px: float
    r_mult: float
    reason: str   # stop / late / timeout


def _initial_stop(entry_px, atr_i, sl_mode, pivot_lows, pl_idx, direction, sl_mult=1.0):
    """初始止损价 (long: 止损线下; short: 止损线上)

    sl_mult: 止损距离倍数 (ATR×sl_mult) — 宽止损用于尾部捕捉实验
    """
    dist = atr_i * sl_mult
    if sl_mode == "atr":
        return entry_px - dist if direction == "long" else entry_px + dist
    # hl: 最近已确认 pivot low (long) / pivot high (short)
    if pl_idx > 0:
        return pivot_lows[pl_idx - 1][1]
    return entry_px - dist if direction == "long" else entry_px + dist  # 无 HL → ATR 兜底


def simulate_holds(close, high, low, atr, states, entries,
                   direction="long", sl_mode="hl", exit_late=False, w=96,
                   sl_mult=1.0):
    """逐笔模拟 (事件式, 每笔独立 — 允许重叠, 与 A3 事件口径一致)

    返回 (trades: list[HoldTrade], 分年 R: dict[year, list[float]])
    """
    close = np.asarray(close, float)
    atr = np.asarray(atr, float)
    n = len(close)
    states = np.asarray(states)
    entries = np.asarray(entries, bool)

    # 已确认 pivot lows/highs (pos, val)
    ph, pl = confirmed_pivots(_df_like(high, low))
    piv_lows = [(j, low[j]) for j in np.flatnonzero(pl)]
    piv_highs = [(j, high[j]) for j in np.flatnonzero(ph)]

    late_tag = "up:late" if direction == "long" else "down:late"

    trades = []
    for i in np.flatnonzero(entries):
        if i + 1 >= n or not np.isfinite(close[i]) or not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        entry_px = close[i]
        # 持仓起始指针 (只可用 j+k<=i 的 pivot)
        pl_i = np.searchsorted([p[0] for p in piv_lows], i - K, side="right")
        ph_i = np.searchsorted([p[0] for p in piv_highs], i - K, side="right")
        stop = _initial_stop(entry_px, atr[i], sl_mode,
                             piv_lows if direction == "long" else piv_highs,
                             pl_i if direction == "long" else ph_i, direction,
                             sl_mult=sl_mult)
        r_base = abs(stop - entry_px)
        if not np.isfinite(r_base) or r_base <= 0:
            # 兜底: pivot 低点与入场价重合 (低价币 tick 粒度) → 用 ATR 距离
            stop = entry_px - atr[i] if direction == "long" else entry_px + atr[i]
            r_base = atr[i]
        exit_px, reason = None, "timeout"
        for t in range(i + 1, min(i + w + 1, n)):
            # 推进已确认 pivots (j+k <= t)
            if direction == "long":
                while pl_i < len(piv_lows) and piv_lows[pl_i][0] + K <= t:
                    if piv_lows[pl_i][1] > stop:
                        stop = piv_lows[pl_i][1]  # 跟踪止损上移
                    pl_i += 1
            else:
                while ph_i < len(piv_highs) and piv_highs[ph_i][0] + K <= t:
                    if piv_highs[ph_i][1] < stop:
                        stop = piv_highs[ph_i][1]  # 跟踪止损下移
                    ph_i += 1
            # late 退出 (持仓中首次出现 late 状态即平仓)
            if exit_late and states[t] == late_tag:
                exit_px, reason = close[t], "late"
                break
            # 止损触发 (收盘破位)
            if (direction == "long" and close[t] < stop) or \
               (direction == "short" and close[t] > stop):
                exit_px, reason = stop, "stop"
                break
        if exit_px is None:
            exit_px, reason = close[min(i + w, n - 1)], "timeout"
        r = (exit_px - entry_px) / r_base if direction == "long" else \
            (entry_px - exit_px) / r_base
        trades.append(HoldTrade(i, t if exit_px is not None else min(i + w, n - 1),
                                entry_px, exit_px, r, reason))
    return trades


def _df_like(high, low):
    """临时 DataFrame 供 confirmed_pivots 使用"""
    import pandas as pd
    n = len(high)
    return pd.DataFrame({"open": np.zeros(n), "high": high, "low": low,
                         "close": np.zeros(n), "volume": np.zeros(n)})


def summarize(trades, label=""):
    """统计: n/胜率/平均R/盈亏比/期望R (只计 win/loss/stop/late, timeout 单独)"""
    rs = [t.r_mult for t in trades if t.reason != "timeout"]
    to = [t for t in trades if t.reason == "timeout"]
    if not rs:
        return f"{label}: 无有效持仓 (timeout {len(to)})"
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r <= 0]
    wr = len(wins) / len(rs)
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = np.mean(losses) if losses else 0.0
    exp_r = np.mean(rs)
    pf = (avg_win * len(wins)) / (abs(avg_loss) * len(losses)) if losses else np.inf
    return (f"{label}: n={len(rs)} 胜率{wr:.1%} 平均R{exp_r:+.3f} "
            f"(赢{avg_win:.2f}R/亏{avg_loss:.2f}R) 盈亏比{pf:.2f} timeout{len(to)}")
