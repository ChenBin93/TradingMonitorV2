#!/usr/bin/env python3
"""市场结构标签 — 第一批 live 增强 (2026-08-04)

来源 (全部 GBM 对照验证):
  A1 4H 道氏段龄 + 存活率      (A6c: 段长中位 25 根, 存活曲线)
  A2 日线一致性 顺风/逆风/无风  (A6d: 回撤恢复率 66.4/77.3/75.7%)
  A4 段内位置 早/中/晚          (A6c: 回撤集中中后段 63%)
  B1 区间内标记 (简化版)        (B3d: 区间内触碰后留区间内 +5.3pp)
  B2 触碰检测                  (B2b/B3: 触碰后 12 根波动 +6%)

描述性信息层 — 不改信号/场景逻辑, 只加标签与上下文。
"""
import numpy as np
import pandas as pd

from research.structures import dow_segments

# A6c 4h 真实存活曲线: P(段长 >= k)
_SURVIVAL = [(5, 0.98), (10, 0.88), (15, 0.75), (20, 0.60), (25, 0.49),
             (30, 0.41), (40, 0.26), (50, 0.16), (60, 0.10), (80, 0.04),
             (100, 0.01)]


def _survival(k: int) -> float:
    ks = [x[0] for x in _SURVIVAL]
    vs = [x[1] for x in _SURVIVAL]
    if k <= ks[0]:
        return vs[0]
    if k >= ks[-1]:
        return vs[-1]
    return float(np.interp(k, ks, vs))


def resample_daily(df: pd.DataFrame) -> pd.DataFrame:
    """4H → 日线 (每 6 根 4H = 1 日线 bar; 日线收盘于当日最后一根 4H 之后)"""
    idx = df.index
    day = np.array([ts.date() for ts in idx])
    out = []
    for dd in sorted(set(day)):
        m = day == dd
        j0 = np.flatnonzero(m)[0]
        out.append((pd.Timestamp(dd), df["open"].values[j0],
                    df["high"].values[m].max(), df["low"].values[m].min(),
                    df["close"].values[m][-1]))
    di = pd.DatetimeIndex([x[0] for x in out])
    return pd.DataFrame({"open": [x[1] for x in out],
                         "high": [x[2] for x in out],
                         "low": [x[3] for x in out],
                         "close": [x[4] for x in out]}, index=di)


def compute_dow_info(df_4h: pd.DataFrame | None) -> dict:
    """4H 道氏段信息 (当前 bar): 段方向/段龄/存活率/段位置/日线一致性"""
    if df_4h is None or len(df_4h) < 30:
        return {}
    try:
        d = dow_segments(df_4h)
    except Exception:
        return {}
    n = len(df_4h)
    cur = str(d["states"][-1])
    info = {"seg_dir": cur}
    for s in d["segs"]:
        if s["start"] <= n - 1 <= s["end"]:
            age = n - 1 - s["start"] + 1
            info["seg_age"] = age
            info["seg_surv"] = _survival(age)
            info["seg_pos"] = (n - 1 - s["start"]) / max(1, s["bars"])
            break
    try:
        daily = resample_daily(df_4h)
        dd = dow_segments(daily)
        daily_dir = str(dd["states"][-1])
        info["daily_dir"] = daily_dir
        if daily_dir == cur and cur in ("up", "down"):
            info["daily_cons"] = "顺风"
        elif daily_dir in ("up", "down"):
            info["daily_cons"] = "逆风"
        else:
            info["daily_cons"] = "无风"
    except Exception:
        info["daily_cons"] = ""
    return info


def check_range_bounds(levels, current_price: float, atr: float,
                       max_width: float = 2.5) -> bool:
    """B1 简化区间判定: 最近支撑与阻力间距 <= max_width×ATR

    研究版 (B3d) 还要求双方 60 根无确认突破 — live 无该基建, 简化为
    成对位带间距条件 (保守标记, 只提示不决策)
    """
    sup = res = None
    for lv in levels:
        if lv.side == "support" and lv.price < current_price:
            if sup is None or lv.price > sup.price:
                sup = lv
        elif lv.side == "resistance" and lv.price > current_price:
            if res is None or lv.price < res.price:
                res = lv
    if sup is None or res is None:
        return False
    return (res.price - sup.price) <= max_width * (atr or 1)


def check_recent_touch(levels, df: pd.DataFrame | None):
    """B2 触碰检测: 最近一根 bar 穿过位带中心线 (intrabar)"""
    if df is None or len(df) < 2:
        return None
    lo = df["low"].values[-1]
    hi = df["high"].values[-1]
    for lv in levels:
        if lo <= lv.price <= hi:
            return lv
    return None
