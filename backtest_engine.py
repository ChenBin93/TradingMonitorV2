#!/usr/bin/env python3
"""5m 回测引擎 — 与 live 系统同逻辑的参数优化基础

时间对齐: 每根 5m K 线 = 一个扫描点
  5m: 当前已收盘 bar (trailing window 计算指标)
  1H: 时间戳 ≤ 当前 5m 时间的最近已收盘 1H bar → ind_1h
  4H: 时间戳 ≤ 当前 5m 时间的最近已收盘 4H bar → ind_4h
入场: 信号触发后按该 5m bar 收盘价入场
出场: SL/TP 取 1H S/R + ATR 缓冲, forward-walk 检查先碰哪个
"""
import hashlib
import os
import pickle
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from indicators import compute as compute_indicators, compute_series
from signals import SIGNALS, SignalState, get_regime, get_direction
from support_resistance import find_swing_levels, get_nearest_levels


# ═══════════════════════════════════════════════════════════════════
# find_swing_levels memoize — 同一 df 只算一次 (回测时同一 1H bar 被多次引用)
# ═══════════════════════════════════════════════════════════════════

_sr_memo: dict[tuple, object] = {}
_sr_memo_order: list[tuple] = []
_series_ncol_cache: dict[int, dict] = {}
_series_ncol_order: list[int] = []


def _find_swing_levels_memo(df, lookback: int = 50):
    key = (id(df), lookback, len(df))
    if key in _sr_memo:
        return _sr_memo[key]
    result = find_swing_levels(df, lookback)
    _sr_memo[key] = result
    _sr_memo_order.append(key)
    if len(_sr_memo_order) > 5000:
        old = _sr_memo_order.pop(0)
        _sr_memo.pop(old, None)
    return result


# ═══════════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════════

def load_all(db_path: str = "data/backtest.db") -> dict[str, dict[str, pd.DataFrame]]:
    """{symbol: {tf: DataFrame}} — 按时间排序"""
    conn = sqlite3.connect(db_path)
    data: dict[str, dict[str, pd.DataFrame]] = {}
    for sym, tf, ts, o, h, l, c, v in conn.execute(
            "SELECT symbol, timeframe, timestamp, open, high, low, close, volume FROM candles ORDER BY timestamp"):
        sym_data = data.setdefault(sym, {})
        tf_data = sym_data.setdefault(tf, {"timestamp": [], "open": [], "high": [],
                                           "low": [], "close": [], "volume": []})
        tf_data["timestamp"].append(pd.Timestamp(ts))
        tf_data["open"].append(o)
        tf_data["high"].append(h)
        tf_data["low"].append(l)
        tf_data["close"].append(c)
        tf_data["volume"].append(v)
    conn.close()

    result = {}
    for sym, tfs in data.items():
        result[sym] = {}
        for tf, d in tfs.items():
            result[sym][tf] = pd.DataFrame(d).set_index("timestamp").sort_index()
    return result


# ═══════════════════════════════════════════════════════════════════
# 时间对齐: 找 ≤ ts 的最近 bar 的指标
# ═══════════════════════════════════════════════════════════════════

class MiniSeries:
    """轻量 Series 替代: .iloc[idx] 直接索引 numpy 数组 (支持负索引)"""
    __slots__ = ("_arr",)

    def __init__(self, arr):
        self._arr = arr

    @property
    def iloc(self):
        return self

    @property
    def values(self):
        return self._arr

    def __getitem__(self, idx):
        return self._arr[idx]

    def __len__(self):
        return len(self._arr)


class MiniDf:
    """轻量 DataFrame 替代: 支持信号用到的 open/high/low/close 列访问 + tail()"""
    __slots__ = ("_cols", "_n")

    def __init__(self, df: pd.DataFrame, pos: int):
        self._n = pos + 1
        self._cols = {
            "open": df["open"].values,
            "high": df["high"].values,
            "low": df["low"].values,
            "close": df["close"].values,
            "volume": df["volume"].values,
        }

    def __len__(self):
        return self._n

    def __getitem__(self, col: str) -> MiniSeries:
        return MiniSeries(self._cols[col])

    def tail(self, n: int) -> "MiniDf":
        """返回最近 n 根的视图 (截断到当前已收盘范围)"""
        start = max(0, self._n - n)
        new = MiniDf.__new__(MiniDf)
        new._n = self._n
        new._cols = {k: v[start:self._n] for k, v in self._cols.items()}
        return new

    def reset_index(self, drop: bool = True) -> "MiniDf":
        """与 pandas API 兼容 — 数据本就按序存储"""
        return self


def _row_to_light(series: pd.DataFrame, pos: int, df: pd.DataFrame,
                  light_df: bool = True) -> dict:
    """从 compute_series 结果取第 pos 行的轻量指标 (信号检查够用, 跳过 O(n) 字段)

    信号检查所需: close/atr/volume_ratio/roc/rsi/body_dir/body_pct/pinbar/adx/df/ma20/bb_width
    """
    s = series
    o, h, l, c = df["open"].iloc[pos], df["high"].iloc[pos], df["low"].iloc[pos], df["close"].iloc[pos]

    body_top = max(o, c)
    body_bottom = min(o, c)
    body = body_top - body_bottom
    total_range = h - l
    body_pct = body / total_range if total_range > 0 else 0
    body_dir = "bullish" if c > o else "bearish" if c < o else "neutral"

    lower_wick = body_bottom - l
    upper_wick = h - body_top
    pinbar = None
    if total_range > 0:
        if lower_wick >= total_range * 0.6 and upper_wick <= total_range * 0.2:
            pinbar = "bullish"
        elif upper_wick >= total_range * 0.6 and lower_wick <= total_range * 0.2:
            pinbar = "bearish"

    # numpy 列缓存 — 按 id(series) 缓存, 避免重复 pandas 列访问
    s_id = id(s)
    ncols = _series_ncol_cache.get(s_id)
    if ncols is None:
        ncols = {col: s[col].values for col in
                 ("roc", "rsi", "adx", "plus_di", "minus_di", "bb_width",
                  "atr", "volume_ratio", "ma5", "ma20", "ma60",
                  "ma_alignment", "bb_state")}
        _series_ncol_cache[s_id] = ncols
        _series_ncol_order.append(s_id)
        if len(_series_ncol_order) > 200:
            old = _series_ncol_order.pop(0)
            _series_ncol_cache.pop(old, None)

    def _v(col):
        arr = ncols[col]
        x = arr[pos] if pos < len(arr) else np.nan
        return float(x) if x == x else None

    return {
        "close": float(c),
        "roc": _v("roc"),
        "rsi": _v("rsi"),
        "adx": _v("adx"),
        "plus_di": _v("plus_di"),
        "minus_di": _v("minus_di"),
        "bb_width": _v("bb_width"),
        "atr": _v("atr"),
        "volume_ratio": _v("volume_ratio"),
        "ma5": _v("ma5"),
        "ma20": _v("ma20"),
        "ma60": _v("ma60"),
        "ma_alignment": str(ncols["ma_alignment"][pos]),
        "bb_state": str(ncols["bb_state"][pos]) if ncols["bb_state"][pos] else "unknown",
        "pinbar": pinbar,
        "body_pct": body_pct,
        "body_dir": body_dir,
        "df": MiniDf(df, pos) if light_df else df.iloc[:pos + 1],
    }


class TFContext:
    """walk-forward: 5m 滑窗 + 1H/4H 上下文, 全部用 compute_series 预计算"""

    def __init__(self, sym_data: dict[str, pd.DataFrame], tf_cfg: dict,
                 ind_params_5m: dict, ind_params_1h: dict, ind_params_4h: dict,
                 max_5m_bars: int = 300):
        self.df_5m = sym_data.get("5m", pd.DataFrame())
        self.df_1h = sym_data.get("1h", pd.DataFrame())
        self.df_4h = sym_data.get("4h", pd.DataFrame())
        self.ind_params_5m = ind_params_5m
        self.ind_params_1h = ind_params_1h
        self.ind_params_4h = ind_params_4h

        # 全量预计算指标序列 (每 TF 一次, O(n) 向量化)
        self._series_5m = compute_series(self.df_5m, ind_params_5m) if not self.df_5m.empty else None
        self._series_1h = compute_series(self.df_1h, ind_params_1h) if not self.df_1h.empty else None
        self._series_4h = compute_series(self.df_4h, ind_params_4h) if not self.df_4h.empty else None

        self._5m_index = list(self.df_5m.index) if not self.df_5m.empty else []
        self._1h_index = list(self.df_1h.index) if not self.df_1h.empty else []
        self._4h_index = list(self.df_4h.index) if not self.df_4h.empty else []

        # 按 pos 缓存指标 dict — 同一 1H/4H bar 内所有 5m bar 共享
        self._ind_1h_by_pos: dict[int, dict] = {}
        self._ind_4h_by_pos: dict[int, dict] = {}

    def _ind_at_series(self, df: pd.DataFrame, series: pd.DataFrame | None,
                       ts: pd.Timestamp, index: list, cache: dict) -> dict:
        """≤ ts 的最近 bar 指标 (按 pos 缓存, 同 bar 内复用)"""
        if series is None:
            return {}
        import bisect
        pos = bisect.bisect_right(index, ts) - 1
        if pos < 0 or pos >= len(series):
            return {}
        if pos in cache:
            return cache[pos]
        ind = _row_to_light(series, pos, df, light_df=False)
        cache[pos] = ind
        return ind

    def last_1h_ind(self, ts: pd.Timestamp) -> dict:
        return self._ind_at_series(self.df_1h, self._series_1h, ts, self._1h_index,
                                   self._ind_1h_by_pos)

    def last_4h_ind(self, ts: pd.Timestamp) -> dict:
        return self._ind_at_series(self.df_4h, self._series_4h, ts, self._4h_index,
                                   self._ind_4h_by_pos)

    def window_5m_ind(self, ts: pd.Timestamp) -> dict:
        """截至 ts 的 5m bar 指标 (从预计算序列取值)"""
        import bisect
        if self._series_5m is None:
            return {}
        pos = bisect.bisect_right(self._5m_index, ts) - 1
        if pos < 0 or pos >= len(self._series_5m):
            return {}
        return _row_to_light(self._series_5m, pos, self.df_5m)


# ═══════════════════════════════════════════════════════════════════
# 信号回测 — 两阶段: 检测(慢, 一次) + 模拟(快, 任意参数)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SignalEvent:
    """信号检测结果 — 与 SL/TP/RR 参数无关, 可缓存复用"""
    symbol: str
    signal: str
    direction: str
    ts: datetime
    entry_price: float
    atr_1h: float
    atr_5m: float
    support_price: float | None = None
    resistance_price: float | None = None
    # ── 触发时的市场状态 (供逐条件分析) ──
    ma_1h: str = "neutral"          # 1H MA 排列 bullish/bearish/neutral
    adx_1h: float = 0.0             # 1H ADX
    ma_4h: str = "neutral"          # 4H MA 排列
    adx_4h: float = 0.0             # 4H ADX
    atr_4h: float = 0.0             # 4H ATR (行情状态机用)
    close_4h: float | None = None   # 4H 收盘 (宏观 bias 用)
    ma20_4h: float | None = None
    ma60_4h: float | None = None
    macro_bias: str = "neutral"     # _symbol_bias 同款逻辑: long/short/neutral
    adx_5m: float = 0.0             # 5m ADX (gate 判断)
    pos_in_range: float | None = None  # 价格在 S/R 区间的位置 0=贴支撑 1=贴阻力


def _macro_bias(ind_1h: dict, ind_4h: dict) -> str:
    """与 live main._symbol_bias 同款: 4H MA20/MA60+ADX 稳定趋势, 1H 确认"""
    if not ind_4h or not ind_1h:
        return "neutral"
    close_4h = ind_4h.get("close")
    ma20_4h = ind_4h.get("ma20")
    ma60_4h = ind_4h.get("ma60")
    adx_4h = ind_4h.get("adx", 0) or 0
    adx_1h = ind_1h.get("adx", 0) or 0
    ma_1h = ind_1h.get("ma_alignment", "neutral")

    if close_4h is None:
        return "neutral"
    above_ma20 = ma20_4h and close_4h > ma20_4h
    above_ma60 = ma60_4h and close_4h > ma60_4h

    if above_ma20 and above_ma60 and adx_4h >= 20 and ma_1h == "bullish":
        return "long"
    if not above_ma20 and not above_ma60 and adx_4h >= 20 and ma_1h == "bearish":
        return "short"
    if above_ma20 and above_ma60 and ma_1h == "bullish" and adx_1h >= 20:
        return "long"
    if not above_ma20 and not above_ma60 and ma_1h == "bearish" and adx_1h >= 20:
        return "short"
    if above_ma60 and adx_4h >= 18:
        return "long"
    if not above_ma60 and adx_4h >= 18:
        return "short"
    return "neutral"


@dataclass
class Trade:
    symbol: str
    signal: str
    direction: str
    entry_ts: datetime
    entry_price: float
    sl: float
    tp: float
    rr: float
    exit_ts: datetime | None = None
    exit_price: float | None = None
    outcome: str = "open"  # win / loss / open


EVENTS_CACHE_DIR = "data/events_cache"


def _events_cache_key(symbols: list[str], tf_cfg: dict, signal_overrides: dict | None,
                      dedup_minutes: int) -> str:
    """缓存 key: symbols + 指标参数 + 信号覆盖 + dedup + SIGNALS 指纹(含实现源码)"""
    import inspect
    sig_fp = hashlib.md5(pickle.dumps([
        (s.id, s.params, s.gate, inspect.getsource(s.check))
        for s in SIGNALS
    ])).hexdigest()[:8]
    payload = {
        "symbols": sorted(symbols),
        "indicators": tf_cfg.get("indicators", {}),
        "overrides": signal_overrides or {},
        "dedup_minutes": dedup_minutes,
        "signals_fp": sig_fp,
        "engine_version": 7,
    }
    raw = pickle.dumps(payload)
    return hashlib.md5(raw).hexdigest()[:16]


def detect_signals(
    all_data: dict[str, dict[str, pd.DataFrame]],
    symbols: list[str],
    tf_cfg: dict,
    signal_overrides: dict | None = None,
    dedup_minutes: int = 30,
    use_cache: bool = True,
) -> list[SignalEvent]:
    """阶段一: 只检测信号, 不计算 SL/TP/RR, 不 forward 判定 (与参数无关, 可缓存)

    use_cache=True 时结果持久化到 data/events_cache/, 同参数二次调用直接加载
    """
    if not use_cache:
        return _detect_signals_inner(all_data, symbols, tf_cfg, signal_overrides, dedup_minutes)

    os.makedirs(EVENTS_CACHE_DIR, exist_ok=True)
    key = _events_cache_key(symbols, tf_cfg, signal_overrides, dedup_minutes)
    cache_path = os.path.join(EVENTS_CACHE_DIR, f"events_{key}.pkl")

    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                events = pickle.load(f)
            print(f"[cache] loaded {len(events)} events from {cache_path}", flush=True)
            return events
        except Exception as e:
            print(f"[cache] load failed: {e}, re-detecting", flush=True)

    events = _detect_signals_inner(all_data, symbols, tf_cfg, signal_overrides, dedup_minutes)
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(events, f)
        print(f"[cache] saved {len(events)} events → {cache_path}", flush=True)
    except Exception as e:
        print(f"[cache] save failed: {e}", flush=True)
    return events


def _detect_signals_inner(
    all_data: dict[str, dict[str, pd.DataFrame]],
    symbols: list[str],
    tf_cfg: dict,
    signal_overrides: dict | None = None,
    dedup_minutes: int = 30,
) -> list[SignalEvent]:
    """detect_signals 内部实现 (无缓存)"""
    ind_params = tf_cfg.get("indicators", {})
    ind_5m_params = ind_params.get("5m", {})
    ind_1h_params = ind_params.get("1h", {})
    ind_4h_params = ind_params.get("4h", {})

    import signals as signals_mod
    _orig_fsl = signals_mod.find_swing_levels
    signals_mod.find_swing_levels = _find_swing_levels_memo
    try:
        events: list[SignalEvent] = []
        last_signal_ts: dict[str, pd.Timestamp] = {}

        for sym in symbols:
            if sym not in all_data:
                continue
            ctx = TFContext(all_data[sym], tf_cfg, ind_5m_params, ind_1h_params, ind_4h_params)
            if ctx.df_5m.empty:
                continue

            all_5m_ts = list(ctx.df_5m.index)
            for i in range(30, len(all_5m_ts)):
                ts = all_5m_ts[i]
                ind = ctx.window_5m_ind(ts)
                if not ind:
                    continue

                ind_1h = ctx.last_1h_ind(ts)
                if not ind_1h:
                    continue
                ind_4h = ctx.last_4h_ind(ts)

                regime = get_regime(ind)
                direction = get_direction(ind)
                state = SignalState(
                    symbol=sym, timeframe="5m", ind=ind,
                    regime=regime, direction=direction,
                    params={}, ind_1h=ind_1h, ind_4h=ind_4h,
                )

                adx_val = ind.get("adx", 0) or 0
                for sig_def in SIGNALS:
                    if sig_def.gate == "trend" and adx_val < 25:
                        continue
                    if sig_def.gate == "range" and adx_val >= 20:
                        continue
                    params = dict(sig_def.params)
                    if signal_overrides:
                        # 支持定向: {"signal_id": {"param": v}} 或 全局: {"param": v}
                        sig_ov = signal_overrides.get(sig_def.id)
                        if isinstance(sig_ov, dict):
                            params.update(sig_ov)
                        else:
                            params.update(signal_overrides)
                    state.params = params
                    try:
                        result = sig_def.check(state)
                    except Exception:
                        continue
                    if not result:
                        continue

                    sig_dir = result.get("direction", direction)
                    dedup_key = f"{sym}|{sig_def.id}|{sig_dir}"
                    last_ts = last_signal_ts.get(dedup_key)
                    if last_ts and (ts - last_ts).total_seconds() < dedup_minutes * 60:
                        continue
                    last_signal_ts[dedup_key] = ts

                    entry_price = ind.get("close", 0)
                    atr_1h = ind_1h.get("atr") or ind.get("atr") or 1
                    atr_5m = ind.get("atr") or 1

                    # 记录信号时的 S/R (供模拟阶段计算 SL/TP)
                    support_price = resistance_price = None
                    df_1h = ind_1h.get("df")
                    if df_1h is not None and len(df_1h) > 0 and entry_price:
                        levels = find_swing_levels(df_1h, lookback=50)
                        support, resistance = get_nearest_levels(levels, entry_price)
                        if support:
                            support_price = support.price
                        if resistance:
                            resistance_price = resistance.price

                    # 触发时的市场状态
                    pos_in_range = None
                    if support_price and resistance_price and resistance_price > support_price:
                        pos_in_range = (entry_price - support_price) / (resistance_price - support_price)

                    events.append(SignalEvent(
                        symbol=sym, signal=sig_def.id, direction=sig_dir,
                        ts=ts, entry_price=entry_price,
                        atr_1h=atr_1h, atr_5m=atr_5m,
                        support_price=support_price, resistance_price=resistance_price,
                        ma_1h=ind_1h.get("ma_alignment", "neutral"),
                        adx_1h=ind_1h.get("adx", 0) or 0,
                        ma_4h=ind_4h.get("ma_alignment", "neutral"),
                        adx_4h=ind_4h.get("adx", 0) or 0,
                        atr_4h=ind_4h.get("atr", 0) or 0,
                        close_4h=ind_4h.get("close"),
                        ma20_4h=ind_4h.get("ma20"),
                        ma60_4h=ind_4h.get("ma60"),
                        macro_bias=_macro_bias(ind_1h, ind_4h),
                        adx_5m=ind.get("adx", 0) or 0,
                        pos_in_range=pos_in_range,
                    ))
        return events
    finally:
        signals_mod.find_swing_levels = _orig_fsl


def simulate_trades(
    events: list[SignalEvent],
    all_data: dict[str, dict[str, pd.DataFrame]],
    symbols: list[str],
    atr_sl_buffer: float = 0.3,
    rr_min: float = 1.2,
    forward_hours: float = 48,
    tp_mode: str = "sr",
    atr_tp_mult: float = 2.5,
    symmetric: bool = False,
) -> list[Trade]:
    """阶段二: 用任意 SL/TP/RR 参数模拟交易 — 快, 适合参数扫描

    tp_mode: "sr" — TP=1H S/R 对侧; "atr" — TP=entry±ATR×atr_tp_mult; "min" — 取较近者
    symmetric: True 时 SL/TP 以入场价对称按 ATR 距离, 忽略 S/R (edge 测试用)
    """
    # 预加载 5m df 的 numpy 数组 (每 symbol 一次)
    df_by_sym: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        if sym in all_data:
            df_by_sym[sym] = all_data[sym].get("5m", pd.DataFrame())

    trades: list[Trade] = []
    for ev in events:
        df = df_by_sym.get(ev.symbol)
        if df is None or df.empty:
            continue

        entry_price = ev.entry_price
        if symmetric:
            # 对称模式: SL/TP 以入场价为基准按 ATR 距离, 忽略 S/R (用于 edge 测试)
            if ev.direction == "long":
                sl = entry_price - ev.atr_1h * atr_sl_buffer
                tp = entry_price + ev.atr_1h * atr_tp_mult
            else:
                sl = entry_price + ev.atr_1h * atr_sl_buffer
                tp = entry_price - ev.atr_1h * atr_tp_mult
        elif ev.direction == "long":
            if ev.support_price:
                sl = ev.support_price - ev.atr_1h * atr_sl_buffer
            else:
                sl = entry_price - ev.atr_5m * 1.5
            tp_sr = ev.resistance_price
            tp_atr = entry_price + ev.atr_1h * atr_tp_mult
            if tp_mode == "atr":
                tp = tp_atr
            elif tp_mode == "min":
                tp = min(tp_sr, tp_atr) if tp_sr else tp_atr
            else:
                tp = tp_sr if tp_sr else tp_atr
            if tp <= sl or tp <= entry_price:
                tp = tp_atr
        else:
            if ev.resistance_price:
                sl = ev.resistance_price + ev.atr_1h * atr_sl_buffer
            else:
                sl = entry_price + ev.atr_5m * 1.5
            tp_sr = ev.support_price
            tp_atr = entry_price - ev.atr_1h * atr_tp_mult
            if tp_mode == "atr":
                tp = tp_atr
            elif tp_mode == "min":
                tp = max(tp_sr, tp_atr) if tp_sr else tp_atr
            else:
                tp = tp_sr if tp_sr else tp_atr
            if tp >= sl or tp >= entry_price:
                tp = tp_atr

        sl_dist = abs(entry_price - sl)
        tp_dist = abs(tp - entry_price)
        rr = tp_dist / sl_dist if sl_dist > 0 else 0
        if rr < rr_min:
            continue

        trade = Trade(
            symbol=ev.symbol, signal=ev.signal, direction=ev.direction,
            entry_ts=ev.ts, entry_price=entry_price, sl=sl, tp=tp, rr=rr,
        )
        result = resolve_forward_df(df, ev.ts, entry_price, sl, tp, forward_hours)
        if result is None:
            trade.outcome = "open"
        else:
            trade.exit_ts, trade.exit_price, trade.outcome = result
        trades.append(trade)

    return trades


def resolve_forward_df(df: pd.DataFrame, entry_ts: pd.Timestamp, entry_price: float,
                       sl: float, tp: float, forward_hours: float) -> tuple | None:
    """numpy 向量化前向判定 — 用 searchsorted 定位起点, 切片后 argmax"""
    idx = df.index
    ts_vals = idx.values.astype("datetime64[ns]")
    entry_dt = np.datetime64(entry_ts.to_datetime64()) if hasattr(entry_ts, "to_datetime64") else np.datetime64(entry_ts)
    cutoff_dt = entry_dt + np.timedelta64(int(forward_hours * 3600), "s")

    pos = int(np.searchsorted(ts_vals, entry_dt, side="right"))
    end = int(np.searchsorted(ts_vals, cutoff_dt, side="left"))
    if pos >= end or pos >= len(df):
        return None

    highs = df["high"].values[pos:end]
    lows = df["low"].values[pos:end]
    long_dir = tp > entry_price

    if long_dir:
        hit_tp = np.argmax(highs >= tp)
        hit_sl = np.argmax(lows <= sl)
    else:
        hit_tp = np.argmax(lows <= tp)
        hit_sl = np.argmax(highs >= sl)

    tp_hit = highs[hit_tp] >= tp if long_dir else lows[hit_tp] <= tp
    sl_hit = lows[hit_sl] <= sl if long_dir else highs[hit_sl] >= sl

    if tp_hit and (not sl_hit or hit_tp < hit_sl):
        return idx[pos + hit_tp], tp, "win"
    if sl_hit and (not tp_hit or hit_sl < hit_tp):
        return idx[pos + hit_sl], sl, "loss"
    return None


def run_backtest(
    all_data: dict[str, dict[str, pd.DataFrame]],
    symbols: list[str],
    tf_cfg: dict,
    signal_params: dict | None = None,
    atr_sl_buffer: float = 0.3,
    atr_tp_fallback: float = 2.5,
    rr_min: float = 1.2,
    forward_hours: float = 48,
    dedup_minutes: int = 30,
    signal_overrides: dict | None = None,
    tp_mode: str = "sr",
    atr_tp_mult: float = 2.5,
) -> list[Trade]:
    """全量回测: 检测 + 模拟 (兼容旧接口)"""
    events = detect_signals(all_data, symbols, tf_cfg, signal_overrides, dedup_minutes)
    return simulate_trades(events, all_data, symbols, atr_sl_buffer, rr_min,
                           forward_hours, tp_mode, atr_tp_mult)


# ═══════════════════════════════════════════════════════════════════
# 统计
# ═══════════════════════════════════════════════════════════════════

def summarize(trades: list[Trade]) -> dict:
    if not trades:
        return {"count": 0}
    closed = [t for t in trades if t.outcome in ("win", "loss")]
    wins = [t for t in closed if t.outcome == "win"]
    losses = [t for t in closed if t.outcome == "loss"]
    win_rate = len(wins) / len(closed) if closed else 0

    # PnL: 每笔按 RR 结算 (风险 1 单位, 赢 = rr, 输 = -1)
    pnls = [t.rr if t.outcome == "win" else -1.0 for t in closed]
    equity = np.cumsum(pnls)
    max_dd = 0
    peak = 0
    for e in equity:
        peak = max(peak, e)
        max_dd = min(max_dd, e - peak)

    total_return = sum(pnls)
    avg_rr = np.mean([t.rr for t in closed]) if closed else 0
    avg_win = np.mean([t.rr for t in wins]) if wins else 0
    avg_loss = 1.0
    sharpe = (np.mean(pnls) / np.std(pnls) * np.sqrt(252 * 24 * 12)) if len(pnls) > 1 and np.std(pnls) > 0 else 0
    calmar = (total_return / abs(max_dd)) if max_dd != 0 else 0

    return {
        "count": len(trades),
        "closed": len(closed),
        "open": len(trades) - len(closed),
        "win": len(wins),
        "loss": len(losses),
        "win_rate": round(win_rate, 4),
        "total_return": round(total_return, 2),
        "avg_rr": round(float(avg_rr), 3),
        "avg_win": round(float(avg_win), 3),
        "avg_loss": avg_loss,
        "max_drawdown": round(float(max_dd), 2),
        "sharpe": round(float(sharpe), 2),
        "calmar": round(float(calmar), 2),
    }


if __name__ == "__main__":
    import json
    import yaml

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    data = load_all()
    symbols = list(data.keys())
    print(f"Loaded {len(symbols)} symbols")

    trades = run_backtest(data, symbols, cfg)
    stats = summarize(trades)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
