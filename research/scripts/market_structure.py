"""道氏结构状态识别 — 粗糙版 (2026-08-03)

设计决策 (用户确认):
  1. 日线为主 + 4H 递进
  2. 阶段先两档 (late vs healthy), 后续三档对比
  3. 最小幅度过滤: 低于 0.5×ATR 的摆动视为突破失败/噪声, 不构成有效 swing

道氏映射:
  - 主要趋势 = 日线 swing 结构 (HH+HL=bull / LH+LL=bear / 其他=range)
  - 阶段 late = 过冲(dev>2ATR) 或 摆动衰减(最近摆动<0.6×前一摆动)
  - 震荡时长 = 从最后一个有效结构被破坏起计

无未来函数: 逐根状态只使用 idx <= i-2 的已确认 swing
(rolling(5,center) 的极值需未来 2 根确认, 故延迟 2 根)
"""
import numpy as np
import pandas as pd

SWING_WINDOW = 5          # 极值确认窗口 (中心, 左右各 2)
MIN_SIZE_RATIO = 0.75     # 最小摆动幅度 = 0.75 × ATR (0.5 抖动过多, 0.75 过滤噪声摆动)
LATE_DEV = 2.0            # 过冲阈值 (ATR)
LATE_SHRINK = 0.6         # 摆动衰减阈值 (最近/前一)
CONFIRM_LAG = 2           # swing 确认延迟根数 (center 窗口半宽)
MIN_SEG_BARS = 5          # 状态最小持续根数 (短于阈值合并进前段, 抑制抖动)


def _atr_series(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    n = len(df)
    tr = np.zeros(n)
    if n > 1:
        tr[1:] = np.maximum(h[1:] - l[1:],
                            np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    return pd.Series(tr).ewm(alpha=1 / period, adjust=False).mean().values


def build_swing_series(df: pd.DataFrame, min_size: float) -> list:
    """swing 序列: (type, price, idx), 交替, 摆动幅度 >= min_size

    同型取更极端; 异型但幅度不足 -> 忽略 (突破失败语义)
    """
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)
    is_high = highs >= pd.Series(highs).rolling(SWING_WINDOW, center=True).max().values
    is_low = lows <= pd.Series(lows).rolling(SWING_WINDOW, center=True).min().values

    seq = []  # (type, price, idx)
    for i in range(n):
        if is_high[i]:
            if seq and seq[-1][0] == "H":
                if highs[i] > seq[-1][1]:
                    seq[-1] = ("H", float(highs[i]), i)
            else:
                if not seq or (highs[i] - seq[-1][1]) >= min_size:
                    seq.append(("H", float(highs[i]), i))
        if is_low[i]:
            if seq and seq[-1][0] == "L":
                if lows[i] < seq[-1][1]:
                    seq[-1] = ("L", float(lows[i]), i)
            else:
                if not seq or (seq[-1][1] - lows[i]) >= min_size:
                    seq.append(("L", float(lows[i]), i))
    return seq


def _classify(swings: list, dev: float) -> tuple:
    """最近 swing 集合 → (state, phase)
    state: bull / bear / range
    phase: late / healthy (仅趋势态; 两档粗糙版)
    """
    highs = [s for s in swings if s[0] == "H"]
    lows = [s for s in swings if s[0] == "L"]
    if len(highs) >= 2 and len(lows) >= 2:
        H1, H2 = highs[-1][1], highs[-2][1]
        L1, L2 = lows[-1][1], lows[-2][1]
        if H1 > H2 and L1 > L2:
            state = "bull"
        elif H1 < H2 and L1 < L2:
            state = "bear"
        else:
            state = "range"
    else:
        state = "range"

    phase = None
    if state in ("bull", "bear"):
        late = False
        if abs(dev) > LATE_DEV:
            late = True
        if len(swings) >= 3:
            w1 = abs(swings[-1][1] - swings[-2][1])
            w2 = abs(swings[-2][1] - swings[-3][1])
            if w2 > 0 and w1 < LATE_SHRINK * w2:
                late = True
        phase = "late" if late else "healthy"
    return state, phase


def structure_series(df: pd.DataFrame, min_size_ratio: float = MIN_SIZE_RATIO) -> dict:
    """逐根结构状态 (无未来函数: 只使用 idx <= i-CONFIRM_LAG 的 swing)

    返回: {
      "state": np.ndarray[str],      # bull/bear/range 逐根
      "phase": np.ndarray[obj],      # late/healthy/None 逐根
      "range_bars": np.ndarray[int], # 震荡持续根数 (仅 range 态)
      "swings": list,                # 全 swing 序列
    }
    """
    n = len(df)
    closes = df["close"].values
    atr = _atr_series(df)
    atr_now = float(atr[-1]) if n else 0.0
    min_size = min_size_ratio * max(atr_now, 1e-9)
    ma20 = pd.Series(closes).rolling(20).mean().values

    seq = build_swing_series(df, min_size)

    states = np.empty(n, dtype=object)
    phases = np.empty(n, dtype=object)
    range_bars = np.zeros(n, dtype=int)

    # 指针: 最后一个 idx <= i-CONFIRM_LAG 的 swing 位置
    p = 0
    last_state = None
    range_start = 0
    for i in range(n):
        while p < len(seq) and seq[p][2] <= i - CONFIRM_LAG:
            p += 1
        if p == 0:
            state, phase = "range", None
        else:
            dev = (closes[i] - ma20[i]) / max(atr[i], 1e-9) if not np.isnan(ma20[i]) else 0.0
            state, phase = _classify(seq[:p], dev)
        states[i] = state
        phases[i] = phase
        if state == "range":
            if last_state != "range":
                range_start = i
            range_bars[i] = i - range_start
        last_state = state

    return {"state": states, "phase": phases, "range_bars": range_bars, "swings": seq}


def structure_series_fast(df: pd.DataFrame, min_size_ratio: float = MIN_SIZE_RATIO) -> dict:
    """优化版: 只记录状态翻转点 (状态持续段), 用于回测按时间查询

    返回: {"segments": [(start_idx, end_idx, state, phase), ...], "swings": seq}
    """
    n = len(df)
    closes = df["close"].values
    atr = _atr_series(df)
    atr_now = float(atr[-1]) if n else 0.0
    min_size = min_size_ratio * max(atr_now, 1e-9)
    ma20 = pd.Series(closes).rolling(20).mean().values

    seq = build_swing_series(df, min_size)

    segments = []
    p = 0
    cur_state = None
    cur_phase = None
    seg_start = 0
    for i in range(n):
        while p < len(seq) and seq[p][2] <= i - CONFIRM_LAG:
            p += 1
        if p == 0:
            state, phase = "range", None
        else:
            dev = (closes[i] - ma20[i]) / max(atr[i], 1e-9) if not np.isnan(ma20[i]) else 0.0
            state, phase = _classify(seq[:p], dev)
        if state != cur_state or phase != cur_phase:
            if cur_state is not None:
                segments.append((seg_start, i - 1, cur_state, cur_phase))
            cur_state, cur_phase = state, phase
            seg_start = i
    if cur_state is not None:
        segments.append((seg_start, n - 1, cur_state, cur_phase))

    # 最小段长: 短段合并进前段 (抑制抖动)
    if segments:
        merged = [segments[0]]
        for s, e, st, ph in segments[1:]:
            last = merged[-1]
            if (e - s + 1) < MIN_SEG_BARS:
                merged[-1] = (last[0], e, last[2], last[3])
            else:
                merged.append((s, e, st, ph))
        segments = merged
    return {"segments": segments, "swings": seq}


def state_at(segments: list, i: int) -> tuple:
    """在 segments 中查询时刻 i 的状态 (二分)"""
    lo, hi = 0, len(segments) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        s, e, st, ph = segments[mid]
        if s <= i <= e:
            return st, ph
        if i < s:
            hi = mid - 1
        else:
            lo = mid + 1
    return "range", None


def daily_ma_states(df4, threshold: float = 0.5):
    """日线 MA 三分类状态序列 (无未来函数查询准备)

    dev = (close - MA20) / ATR_d, 阈值 ±threshold → 多/空/中
    返回: (states[np.ndarray[str]], close_times[np.ndarray[ns]])
    查询时刻 t 的状态: j = searchsorted(close_times, t, side="right") - 1
    注意: close_times = 日线开盘 + 24h (该日收盘时刻), 故查询只用已收盘日线
    """
    daily = df4.resample("1D").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    ma20d = daily["close"].rolling(20).mean()
    atr_d = (daily["high"] - daily["low"]).rolling(14).mean()
    dc = daily["close"].values
    states = np.full(len(daily), "中", dtype=object)
    for j, (ts, row) in enumerate(daily.iterrows()):
        if pd.isna(ma20d[ts]) or pd.isna(atr_d[ts]) or atr_d[ts] <= 0:
            continue
        dev = (row["close"] - ma20d[ts]) / atr_d[ts]
        states[j] = "多" if dev > threshold else "空" if dev < -threshold else "中"
    close_times = daily.index.values.astype("datetime64[ns]") + np.timedelta64(24, "h")
    return states, close_times


def daily_dev_series(df4, threshold: float = 0.5):
    """日线 dev (偏离 MA20 的 ATR 倍数) 序列 + 动量衰减, 无未来函数查询准备

    返回: {"dev": np.ndarray, "mom_turn": np.ndarray, "close_times": np.ndarray}
    mom_turn = 当前10日动量 - 10日前的10日动量 (负值=动能衰减)
    """
    daily = df4.resample("1D").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    dc = daily["close"].values
    ma20d = daily["close"].rolling(20).mean().values
    atr_d = (daily["high"] - daily["low"]).rolling(14).mean().values
    nd = len(daily)
    dev = np.full(nd, np.nan)
    for j in range(nd):
        if not np.isnan(ma20d[j]) and atr_d[j] > 0:
            dev[j] = (dc[j] - ma20d[j]) / atr_d[j]
    mom = np.full(nd, np.nan)
    for j in range(10, nd):
        if atr_d[j] > 0:
            mom[j] = (dc[j] - dc[j - 10]) / atr_d[j]
    mom_turn = np.full(nd, np.nan)
    for j in range(20, nd):
        if not np.isnan(mom[j]) and not np.isnan(mom[j - 10]):
            mom_turn[j] = mom[j] - mom[j - 10]
    close_times = daily.index.values.astype("datetime64[ns]") + np.timedelta64(24, "h")
    return {"dev": dev, "mom_turn": mom_turn, "close_times": close_times}
