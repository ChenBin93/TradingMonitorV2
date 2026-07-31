#!/usr/bin/env python3
"""5m 回测引擎 — 与 live 系统同逻辑的参数优化基础

时间对齐: 每根 5m K 线 = 一个扫描点
  5m: 当前已收盘 bar (trailing window 计算指标)
  1H: 时间戳 ≤ 当前 5m 时间的最近已收盘 1H bar → ind_1h
  4H: 时间戳 ≤ 当前 5m 时间的最近已收盘 4H bar → ind_4h
入场: 信号触发后按该 5m bar 收盘价入场
出场: SL/TP 取 1H S/R + ATR 缓冲, forward-walk 检查先碰哪个
"""
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
    """轻量 Series 替代: .iloc[idx] 直接索引 numpy 数组"""
    __slots__ = ("_arr",)

    def __init__(self, arr):
        self._arr = arr

    @property
    def iloc(self):
        return self

    def __getitem__(self, idx):
        return self._arr[idx]

    def __len__(self):
        return len(self._arr)


class MiniDf:
    """轻量 DataFrame 替代: 只支持信号用到的 open/high/low/close 列访问"""
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
# 信号回测
# ═══════════════════════════════════════════════════════════════════

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
) -> list[Trade]:
    """跑全量回测, 返回所有模拟交易

    signal_overrides: {param_key: value} — 覆盖信号参数 (如 {"threshold": 3.0})
    """
    ind_params = tf_cfg.get("indicators", {})
    ind_5m_params = ind_params.get("5m", {})
    ind_1h_params = ind_params.get("1h", {})
    ind_4h_params = ind_params.get("4h", {})

    # 打补丁: signals 内部调用的 find_swing_levels → memoize 版
    import signals as signals_mod
    _orig_fsl = signals_mod.find_swing_levels
    signals_mod.find_swing_levels = _find_swing_levels_memo
    try:
        return _run_backtest_inner(
            all_data, symbols, tf_cfg, signal_params, atr_sl_buffer,
            atr_tp_fallback, rr_min, forward_hours, dedup_minutes,
            ind_5m_params, ind_1h_params, ind_4h_params, signal_overrides)
    finally:
        signals_mod.find_swing_levels = _orig_fsl


def _run_backtest_inner(
    all_data, symbols, tf_cfg, signal_params, atr_sl_buffer,
    atr_tp_fallback, rr_min, forward_hours, dedup_minutes,
    ind_5m_params, ind_1h_params, ind_4h_params, signal_overrides=None,
) -> list[Trade]:
    """run_backtest 内部实现 (find_swing_levels 已 memoize)"""
    trades: list[Trade] = []
    last_signal_ts: dict[str, pd.Timestamp] = {}  # dedup: symbol|signal|direction

    for sym in symbols:
        if sym not in all_data:
            continue
        ctx = TFContext(all_data[sym], tf_cfg, ind_5m_params, ind_1h_params, ind_4h_params)

        if ctx.df_5m.empty:
            continue

        # 只遍历 5m bars, 且 5m 指标需要 ≥ 30 根才有意义
        all_5m_ts = list(ctx.df_5m.index)
        for i in range(30, len(all_5m_ts)):
            ts = all_5m_ts[i]
            ind = ctx.window_5m_ind(ts)
            if not ind:
                continue

            ind_1h = ctx.last_1h_ind(ts)
            ind_4h = ctx.last_4h_ind(ts)

            # ind_1h 缺失时 5m MiniDf 无法支撑 S/R 信号 — 跳过 (live 中 1H 数据恒在)
            if not ind_1h:
                continue

            regime = get_regime(ind)
            direction = get_direction(ind)

            state = SignalState(
                symbol=sym, timeframe="5m", ind=ind,
                regime=regime, direction=direction,
                params=signal_params or {},
                ind_1h=ind_1h,
            )

            # ── gate 过滤 (与 live 一致) ──
            adx_val = ind.get("adx", 0) or 0
            for sig_def in SIGNALS:
                if sig_def.gate == "trend" and adx_val < 25:
                    continue
                if sig_def.gate == "range" and adx_val >= 20:
                    continue
                params = dict(sig_def.params)
                if signal_overrides:
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

                # ── 入场/止损/止盈 ──
                entry_price = ind.get("close", 0)
                atr_1h = ind_1h.get("atr") or ind.get("atr") or 1
                atr_5m = ind.get("atr") or 1

                sr_info = {}
                df_1h = ind_1h.get("df")
                if df_1h is not None and len(df_1h) > 0 and entry_price:
                    levels = find_swing_levels(df_1h, lookback=50)
                    support, resistance = get_nearest_levels(levels, entry_price)
                    if support:
                        sr_info["support"] = support
                    if resistance:
                        sr_info["resistance"] = resistance

                if sig_dir == "long":
                    if "support" in sr_info:
                        sl = sr_info["support"].price - atr_1h * atr_sl_buffer
                    else:
                        sl = entry_price - atr_5m * 1.5
                    if "resistance" in sr_info:
                        tp = sr_info["resistance"].price
                    else:
                        tp = entry_price + atr_1h * atr_tp_fallback
                    if tp <= sl or tp <= entry_price:
                        tp = entry_price + atr_1h * atr_tp_fallback
                else:
                    if "resistance" in sr_info:
                        sl = sr_info["resistance"].price + atr_1h * atr_sl_buffer
                    else:
                        sl = entry_price + atr_5m * 1.5
                    if "support" in sr_info:
                        tp = sr_info["support"].price
                    else:
                        tp = entry_price - atr_1h * atr_tp_fallback
                    if tp >= sl or tp >= entry_price:
                        tp = entry_price - atr_1h * atr_tp_fallback

                sl_dist = abs(entry_price - sl)
                tp_dist = abs(tp - entry_price)
                rr = tp_dist / sl_dist if sl_dist > 0 else 0

                # ── RR 过滤 (新信号才检查, 与 live 一致) ──
                if rr < rr_min:
                    continue

                trade = Trade(
                    symbol=sym, signal=sig_def.id, direction=sig_dir,
                    entry_ts=ts, entry_price=entry_price, sl=sl, tp=tp, rr=rr,
                )
                self_ = resolve_forward(ctx, ts, entry_price, sl, tp, forward_hours)
                if self_ is None:
                    trade.outcome = "open"
                else:
                    trade.exit_ts, trade.exit_price, trade.outcome = self_
                trades.append(trade)

    return trades


def resolve_forward(ctx: TFContext, entry_ts: pd.Timestamp, entry_price: float,
                    sl: float, tp: float, forward_hours: float) -> tuple | None:
    """从 entry_ts 之后逐根 5m bar 检查: 先碰 TP (win) 还是 SL (loss), 超时 open"""
    import bisect
    cutoff = entry_ts + timedelta(hours=forward_hours)
    start = bisect.bisect_right(ctx._5m_index, entry_ts)
    long_dir = tp > entry_price
    for ts in ctx._5m_index[start:]:
        if ts > cutoff:
            return None
        row = ctx.df_5m.loc[ts]
        if long_dir:
            if row["high"] >= tp:
                return ts, tp, "win"
            if row["low"] <= sl:
                return ts, sl, "loss"
        else:
            if row["low"] <= tp:
                return ts, tp, "win"
            if row["high"] >= sl:
                return ts, sl, "loss"
    return None


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
