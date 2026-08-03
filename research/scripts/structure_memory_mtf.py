#!/usr/bin/env python3
"""结构记忆多周期验证 — 4H/日线的"贴而未破"锚点效应与年龄窗口

问题: 有效贴锚窗口按根数普适 (≤100) 还是按时间?
- 4H: 插曲(日线顺势+4H逆) + 无条件, 沿日线方向 1:1 (W=12)
- 日线: 无条件, 沿日线方向 1:1 (W=10)
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from backtest_engine import load_all
from market_phase import _atr_series

data = load_all(timeframes=["4h"])
syms = list(data.keys())

# 日线合成 + 状态
daily_data = {}
daily_states = {}
for sym in syms:
    df4 = data[sym].get("4h")
    if df4 is None or len(df4) < 300:
        continue
    daily = df4.resample("1D").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    if len(daily) < 200:
        continue
    daily_data[sym] = daily
    ma20d = daily["close"].rolling(20).mean()
    d = {}
    for ts, row in daily.iterrows():
        if pd.isna(ma20d[ts]):
            continue
        d[ts.date()] = "bull" if row["close"] > ma20d[ts] else "bear"
    daily_states[sym] = d


def run_tf(tf_name, df_pool, atr_pool, horizon, step, env_mode):
    """env_mode: 'episode' (需上层方向) 或 'flat' (沿本层日线方向)"""
    print(f"\n{'='*58}\n═══ {tf_name} (env={env_mode}) ═══\n{'='*58}", flush=True)
    samples = []  # (age_hi, dist_hi, age_lo, dist_lo, long_side, out)
    for sym in df_pool:
        df = df_pool[sym]
        c = df["close"].values
        h = df["high"].values
        l = df["low"].values
        n = len(df)
        atr = atr_pool[sym]
        ds = daily_states.get(sym, {})
        h_roll = pd.Series(h).rolling(5, center=True).max().values
        l_roll = pd.Series(l).rolling(5, center=True).min().values
        is_high = h >= h_roll
        is_low = l <= l_roll
        idx_arr = np.arange(n)
        last_high = np.maximum.accumulate(np.where(is_high, idx_arr, -1))
        last_low = np.maximum.accumulate(np.where(is_low, idx_arr, -1))
        # 4H 状态 (本层 vs MA20) — episode 模式用
        ma20_self = pd.Series(c).rolling(20).mean().values
        for i in range(60, n - horizon - 1, step):
            ts = pd.Timestamp(df.index[i])
            sd = ds.get(ts.date())
            if sd not in ("bull", "bear"):
                continue
            a = atr[i]
            if a <= 0 or np.isnan(a):
                continue
            if env_mode == "episode":
                s_self = "bull" if c[i] > ma20_self[i] else "bear"
                if np.isnan(ma20_self[i]) or s_self == sd:
                    continue  # 插曲: 本层逆上层
            entry = c[i]
            ih = last_high[i]
            il = last_low[i]
            age_hi = i - ih if ih >= 0 else 99999
            age_lo = i - il if il >= 0 else 99999
            dist_hi = (h[ih] - entry) / a if ih >= 0 else 99.0
            dist_lo = (entry - l[il]) / a if il >= 0 else 99.0
            long_side = sd == "bull"
            hit = 0
            for k in range(1, horizon + 1):
                if long_side:
                    if h[i+k] >= entry + a:
                        hit = 1; break
                    if l[i+k] <= entry - a:
                        hit = -1; break
                else:
                    if l[i+k] <= entry - a:
                        hit = 1; break
                    if h[i+k] >= entry + a:
                        hit = -1; break
            if hit == 0:
                continue
            samples.append((age_hi, dist_hi, age_lo, dist_lo, long_side, 1 if hit == 1 else 0))
    print(f"样本: {len(samples)}")
    base_long = np.mean([s[5] for s in samples if s[4]]) * 100
    base_short = np.mean([s[5] for s in samples if not s[4]]) * 100
    print(f"基线: 做多 {base_long:.1f}% / 做空 {base_short:.1f}%")

    def show(tag, sub, base):
        w = sum(1 for s in sub if s[5] == 1)
        l = sum(1 for s in sub if s[5] == 0)
        nn = w + l
        if nn < 100:
            print(f"{tag:<42} n={nn:>5} 样本不足")
        else:
            print(f"{tag:<42} n={nn:>6} 胜率={w/nn*100:>5.1f}% Δ={w/nn*100-base:>+5.1f}pp")

    print("\n做多 × 距最近低点 (贴而未破, 0<=dist<0.5, 按年龄):")
    for lo_, hi_, name in [(0, 5, "0-5根(刚形成)"), (5, 50, "5-50根"), (50, 100, "50-100根"),
                           (100, 250, "100-250根"), (250, 99999, "250+根")]:
        show(f"  低点年龄 {name}", [s for s in samples if s[4] and lo_ <= s[2] < hi_ and 0 <= s[3] < 0.5], base_long)

    print("\n做空 × 距最近高点 (贴而未破, 0<=dist<0.5, 按年龄):")
    for lo_, hi_, name in [(0, 5, "0-5根(刚形成)"), (5, 50, "5-50根"), (50, 100, "50-100根"),
                           (100, 250, "100-250根"), (250, 99999, "250+根")]:
        show(f"  高点年龄 {name}", [s for s in samples if not s[4] and lo_ <= s[0] < hi_ and 0 <= s[1] < 0.5], base_short)

    print("\n距离梯度 (全部年龄):")
    for tag, cond, base in [
        ("做多 贴(0-0.5)", lambda s: s[4] and 0 <= s[3] < 0.5, base_long),
        ("做多 0.5-1.0", lambda s: s[4] and 0.5 <= s[3] < 1.0, base_long),
        ("做多 1.0-1.5", lambda s: s[4] and 1.0 <= s[3] < 1.5, base_long),
        ("做多 1.5-3.0", lambda s: s[4] and 1.5 <= s[3] < 3.0, base_long),
        ("做空 贴(0-0.5)", lambda s: not s[4] and 0 <= s[1] < 0.5, base_short),
        ("做空 0.5-1.0", lambda s: not s[4] and 0.5 <= s[1] < 1.0, base_short),
        ("做空 1.0-1.5", lambda s: not s[4] and 1.0 <= s[1] < 1.5, base_short),
        ("做空 1.5-3.0", lambda s: not s[4] and 1.5 <= s[1] < 3.0, base_short),
    ]:
        show(tag, [s for s in samples if cond(s)], base)

    print("\n已跌破 (负距离) 对照:")
    show("做多 已跌破低点(dist<0)", [s for s in samples if s[4] and s[3] < 0], base_long)
    show("做空 已突破高点(dist<0)", [s for s in samples if not s[4] and s[1] < 0], base_short)


# ── 4H 周期 ──
df_4h = {}
atr_4h = {}
for sym in daily_data:
    df4 = data[sym]["4h"]
    df_4h[sym] = df4
    atr_4h[sym] = _atr_series(df4)
run_tf("4H", df_4h, atr_4h, horizon=12, step=2, env_mode="episode")

# ── 日线周期 ──
atr_d = {}
for sym in daily_data:
    atr_d[sym] = _atr_series(daily_data[sym])
run_tf("日线", daily_data, atr_d, horizon=10, step=1, env_mode="flat")
