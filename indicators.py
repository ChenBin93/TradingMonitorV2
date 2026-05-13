# 技术指标计算
# 从 V1 src/signals/indicators.py 复用

import pandas as pd
import numpy as np


def compute(df: pd.DataFrame, params: dict) -> dict | None:
    """计算所有技术指标，返回 dict"""

    if df is None or len(df) < 30:
        return None

    roc = _calc_roc(df, params.get("roc_period", 10))
    rsi = _calc_rsi(df, params.get("rsi_period", 14))
    adx = _calc_adx(df, params.get("adx_period", 14))
    plus_di, minus_di = _calc_di(df, params.get("adx_period", 14))
    bb_mid, bb_width = _calc_bb(df, params.get("bb_period", 20), params.get("bb_std", 2))
    atr = _calc_atr(df, params.get("atr_period", 14))
    ma_s = params.get("ma_short", 5)
    ma_m = params.get("ma_mid", 20)
    ma_l = params.get("ma_long", 60)
    mas = _calc_ma(df, [ma_s, ma_m, ma_l])
    ma5, ma20, ma60 = mas[f"ma_{ma_s}"], mas[f"ma_{ma_m}"], mas[f"ma_{ma_l}"]
    vol_ma = _calc_vol_ma(df, params.get("volume_ma_period", 20))
    vol_ratio = _calc_vol_ratio(df, vol_ma)
    macd_line, signal_line, histogram = _calc_macd(df)
    ma_converge = _calc_ma_converge(df, atr, ma5, ma20, ma60)
    ttm_squeeze = _calc_ttm_squeeze(df)
    rsi_div = _detect_rsi_divergence(df, rsi)
    macd_div = _detect_macd_divergence(df, macd_line, signal_line)
    vol_breakout = _check_vol_breakout(df, vol_ma, vol_ratio)

    idx = len(df) - 1
    v = lambda s: s.iloc[idx] if len(s) > idx and s.iloc[idx] == s.iloc[idx] else None

    prev_h = histogram.iloc[idx - 1] if idx > 0 and not np.isnan(histogram.iloc[idx - 1]) else 0
    curr_h = v(histogram) or 0
    if prev_h <= 0 < curr_h:
        macd_cross = "golden"
    elif prev_h >= 0 > curr_h:
        macd_cross = "death"
    else:
        macd_cross = None

    # 布林带宽度短期百分位 (最近 20 根 K 线内)
    bb_short_pct = _local_percentile(bb_width, lookback=20)

    # K线实体与形态
    o, h, l, c = df["open"].iloc[idx], df["high"].iloc[idx], df["low"].iloc[idx], df["close"].iloc[idx]
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

    return {
        "close": v(df["close"]), "roc": v(roc), "rsi": v(rsi), "adx": v(adx),
        "plus_di": v(plus_di), "minus_di": v(minus_di),
        "bb_width": v(bb_width), "bb_width_short_pct": bb_short_pct, "atr": v(atr),
        "volume_ratio": v(vol_ratio), "ma5": v(ma5), "ma20": v(ma20), "ma60": v(ma60),
        "ma_converge": ma_converge, "macd_hist": v(histogram), "macd_cross": macd_cross,
        "macd_line": v(macd_line), "signal_line": v(signal_line),
        "ttm_squeeze": ttm_squeeze, "rsi_divergence": rsi_div,
        "macd_divergence": macd_div, "volume_breakout": vol_breakout,
        # 新增：结构判断
        "ma_alignment": _ma_alignment(ma5, ma20, ma60),
        "bb_state": _bb_state(bb_width) if len(bb_width) > 10 else "unknown",
        "price_vs_ma60": "above" if v(df["close"]) and v(ma60) and v(df["close"]) > v(ma60) else "below",
        "compression_bars": _count_compression_bars(df),
        "adx_trend": "up" if len(adx) > 5 and adx.iloc[-1] > adx.iloc[-5] else "down",
        "pinbar": pinbar,
        "body_pct": body_pct,
        "body_dir": body_dir,
        "df": df,
    }


# --- 基础指标 ---

def _calc_roc(df, period):
    if len(df) < period + 1:
        return pd.Series([np.nan] * len(df))
    return (df["close"] - df["close"].shift(period)) / df["close"].shift(period) * 100


def _calc_rsi(df, period=14):
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs.fillna(float("inf"))).fillna(100.0)


def _calc_adx(df, period=14):
    plus_di, minus_di = _calc_di(df, period)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def _calc_di(df, period=14):
    h, l, c = df["high"], df["low"], df["close"]
    plus_dm = (h - h.shift(1)).where(lambda x: (x > l.shift(1) - l) & (x > 0), 0.0)
    minus_dm = (l.shift(1) - l).where(lambda x: (x > h - h.shift(1)) & (x > 0), 0.0)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean().replace(0, np.nan)
    return (100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr).fillna(0), \
           (100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr).fillna(0)


def _calc_bb(df, period=20, std=2):
    mid = df["close"].rolling(period).mean()
    s = df["close"].rolling(period).std()
    return mid, ((mid + s * std - (mid - s * std)) / mid.replace(0, np.nan) * 100).fillna(0)


def _calc_atr(df, period=14):
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - df["close"].shift(1)).abs(),
                    (df["low"] - df["close"].shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _calc_ma(df, periods):
    return {f"ma_{p}": df["close"].rolling(p).mean() for p in periods}


def _calc_vol_ma(df, period=20):
    return df["volume"].rolling(period).mean()


def _calc_vol_ratio(df, vol_ma):
    return df["volume"] / vol_ma


def _calc_macd(df, fast=12, slow=26, signal=9):
    ema_f = df["close"].ewm(span=fast, adjust=False).mean()
    ema_s = df["close"].ewm(span=slow, adjust=False).mean()
    line = ema_f - ema_s
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def _calc_ma_converge(df, atr, ma5, ma20, ma60):
    try:
        d = abs(df["close"].iloc[-1] - ma5.iloc[-1]) + abs(ma5.iloc[-1] - ma20.iloc[-1]) + abs(ma20.iloc[-1] - ma60.iloc[-1])
        return min(d / (atr.iloc[-1] * 3), 10.0)
    except Exception:
        return 0.0


def _calc_ttm_squeeze(df, bb_period=20, bb_std=2, kc_period=20, kc_mul=1.5, min_bars=5):
    try:
        mid = df["close"].rolling(bb_period).mean()
        s = df["close"].rolling(bb_period).std()
        bu, bl = mid + s * bb_std, mid - s * bb_std
        tr = pd.concat([df["high"] - df["low"],
                        (df["high"] - df["close"].shift(1)).abs(),
                        (df["low"] - df["close"].shift(1)).abs()], axis=1).max(axis=1)
        kc_m = df["close"].ewm(span=kc_period, adjust=False).mean()
        kc_a = tr.ewm(span=kc_period, adjust=False).mean()
        ku, kl = kc_m + kc_a * kc_mul, kc_m - kc_a * kc_mul
        squeeze = (bu < ku) & (bl > kl)
        cnt = 0
        for i in range(len(squeeze) - 1, -1, -1):
            if squeeze.iloc[i]: cnt += 1
            else: break
        return {
            "squeeze_active": bool(squeeze.iloc[-1]),
            "squeeze_bars": cnt,
            "is_fired": cnt >= min_bars and not squeeze.iloc[-1],
            "direction": "bullish" if not squeeze.iloc[-1] and df["close"].iloc[-1] > mid.iloc[-1]
                         else "bearish" if not squeeze.iloc[-1] and df["close"].iloc[-1] < mid.iloc[-1]
                         else None,
        }
    except Exception:
        return None


def _detect_rsi_divergence(df, rsi, pl=5, pr=5, min_dist=2.0):
    try:
        lookback = pl + pr + 1
        p = df["close"].tail(lookback).values
        r = rsi.tail(lookback).values
        pp, rp = p[pl], r[pl]
        pr_min, pr_max = np.min(p[pl + 1:]), np.max(p[pl + 1:])
        dist = abs(pr_max - pr_min) / pr_max * 100 if pr_max != 0 else 0
        div = None
        if dist >= min_dist:
            if pr_min < pp and np.min(r[:pl]) > rp: div = "bullish"
            elif pr_max > pp and np.max(r[:pl]) < rp: div = "bearish"
        return {"divergence": div, "rsi_value": float(rp), "price_distance_pct": round(dist, 2)}
    except Exception:
        return None


def _detect_macd_divergence(df, macd_line, signal_line, pl=5, pr=5, min_dist=2.0):
    try:
        lookback = pl + pr + 1
        p = df["close"].tail(lookback).values
        m = macd_line.tail(lookback).values
        pp, mp = p[pl], m[pl]
        pr_min, pr_max = np.min(p[pl + 1:]), np.max(p[pl + 1:])
        dist = abs(pr_max - pr_min) / pr_max * 100 if pr_max != 0 else 0
        div = None
        if dist >= min_dist:
            if pr_min < pp and mp > np.min(m[:pl]): div = "bullish"
            elif pr_max > pp and mp < np.max(m[:pl]): div = "bearish"
        return {"divergence": div, "macd_value": float(mp), "price_distance_pct": round(dist, 2)}
    except Exception:
        return None


def _check_vol_breakout(df, vol_ma, vol_ratio, threshold=1.5):
    try:
        return {
            "confirmed": vol_ratio.iloc[-1] >= threshold,
            "vol_ratio": float(vol_ratio.iloc[-1]),
            "price_change_pct": round(abs(df["close"].iloc[-1] - df["close"].iloc[-2]) / df["close"].iloc[-2] * 100, 2),
        }
    except Exception:
        return None


def _ma_alignment(ma5, ma20, ma60) -> str:
    try:
        if ma5.iloc[-1] > ma20.iloc[-1] > ma60.iloc[-1]:
            return "bullish"
        elif ma5.iloc[-1] < ma20.iloc[-1] < ma60.iloc[-1]:
            return "bearish"
        return "neutral"
    except Exception:
        return "neutral"


def _bb_state(bb_width) -> str:
    """判断布林带状态：expanding(扩张) / contracting(收缩) / flat(平直)"""
    try:
        recent = bb_width.iloc[-10:]
        first_half = recent.iloc[:5].mean()
        second_half = recent.iloc[5:].mean()
        if second_half > first_half * 1.15:
            return "expanding"
        elif second_half < first_half * 0.85:
            return "contracting"
        return "flat"
    except Exception:
        return "unknown"


def _count_compression_bars(df) -> int:
    """统计最近连续小实体 K 线根数（压缩)"""
    try:
        count = 0
        closes = df["close"].values
        for i in range(len(df) - 1, -1, -1):
            body = abs(closes[i] - df["open"].values[i])
            avg_body = np.mean(abs(np.diff(closes[max(0, i-20):i+1]))) if i > 5 else body
            if avg_body > 0 and body < avg_body * 0.6:
                count += 1
            else:
                break
        return count
    except Exception:
        return 0


def _local_percentile(series, lookback: int = 20) -> float | None:
    """计算最近 lookback 根 K 线内的百分位 (0-100, 越小越压缩)"""
    try:
        recent = series.tail(lookback).dropna()
        if len(recent) < 5:
            return None
        current = series.iloc[-1]
        if pd.isna(current):
            return None
        return float((recent < current).sum() / len(recent) * 100)
    except Exception:
        return None
