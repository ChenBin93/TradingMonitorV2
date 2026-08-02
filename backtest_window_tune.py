#!/usr/bin/env python3
"""窗口调优 v2 — 预计算+索引采样 (O(1) 查询, 秒级)

对每标的每周期一次向量化预计算完整指标序列:
  atr/adx/ma20/ma60/body/多空累计和
采样点直接用索引区间取值, 窗口扫描只换索引 — 零重复计算

评估:
  A. power窗口: 插曲环境 + 力量同向极端(±40) 的 1:1 胜率增量 (1H + 4H)
  B. phase窗口: trend_up沿向做单 vs trend_down沿向做单 区分度 (1H + 4H)
"""
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from backtest_engine import load_all
from market_phase import _atr_series, _adx_series, classify
from power_balance import score_from_components

data = load_all(timeframes=['1h', '4h'])
syms = list(data.keys())

daily_states = {}
for sym in syms:
    df4 = data[sym].get("4h")
    if df4 is None or len(df4) < 100:
        continue
    daily = df4.resample("1D").last()
    ma20 = daily["close"].rolling(20).mean()
    for ts, row in daily.iterrows():
        if pd.isna(ma20[ts]):
            continue
        daily_states.setdefault(sym, {})[ts.date()] = "bull" if row["close"] > ma20[ts] else "bear"


def t4_state(sym, ts):
    df4 = data[sym].get("4h")
    if df4 is None or len(df4) < 70:
        return "N"
    idx = df4.index.values.astype("datetime64[ns]")
    pos = int(np.searchsorted(idx, np.datetime64(ts) - np.timedelta64(240, "m"), side="right")) - 1
    if pos < 60:
        return "N"
    c = df4["close"].values
    ma20 = np.mean(c[pos-19:pos+1])
    return "bull" if c[pos] > ma20 else "bear"


def precompute(df):
    """一次向量化预计算完整指标序列"""
    o = df["open"].values; h = df["high"].values
    l = df["low"].values; c = df["close"].values
    v = df["volume"].values if "volume" in df.columns else np.ones(len(df))
    n = len(df)
    atr = _atr_series(df)
    adx = _adx_series(df)
    ma20 = pd.Series(c).rolling(20).mean().values
    ma60 = pd.Series(c).rolling(60).mean().values
    body = c - o
    bull = body > 0
    bear = body < 0
    bull_body = np.where(bull, body, 0.0)
    bear_body = np.where(bear, -body, 0.0)
    upper_wick = np.maximum(0, h - c)
    lower_wick = np.maximum(0, o - l)
    bull_wick = np.where(bull, upper_wick, 0.0)
    bear_wick = np.where(bear, lower_wick, 0.0)
    bull_vol = np.where(bull, v, 0.0)
    bear_vol = np.where(bear, v, 0.0)
    return {
        "o": o, "h": h, "l": l, "c": c, "v": v, "n": n,
        "atr": atr, "adx": adx, "ma20": ma20, "ma60": ma60,
        "body": body, "abody": np.abs(body),
        "cum_bb": np.concatenate([[0], np.cumsum(bull_body)]),
        "cum_rb": np.concatenate([[0], np.cumsum(bear_body)]),
        "cum_bw": np.concatenate([[0], np.cumsum(bull_wick)]),
        "cum_rw": np.concatenate([[0], np.cumsum(bear_wick)]),
        "cum_bv": np.concatenate([[0], np.cumsum(bull_vol)]),
        "cum_rv": np.concatenate([[0], np.cumsum(bear_vol)]),
    }


def seg(cum, lo, hi):
    """累计和区间 [lo, hi) — O(1)"""
    return cum[hi] - cum[lo]


def power_at(pre, i, pw, sh):
    """O(1) 力量评分 — 与 analyze_power 同逻辑"""
    if i < pw + 1 or pre["atr"][i] <= 0:
        return 0.0
    bb = seg(pre["cum_bb"], i - pw, i)
    rb = seg(pre["cum_rb"], i - pw, i)
    bw = seg(pre["cum_bw"], i - pw, i)
    rw = seg(pre["cum_rw"], i - pw, i)
    bv = seg(pre["cum_bv"], i - pw, i)
    rv = seg(pre["cum_rv"], i - pw, i)
    h = pre["h"]; l = pre["l"]
    return score_from_components(bb, rb, bw, rw, bv, rv, pre["atr"][i],
                                 h[i], l[i], h[i-5], l[i-5])


def phase_at(pre, i, tw):
    """O(1) 市场状态 — 与 analyze_market_state 同逻辑 (window=tw)"""
    atr = pre["atr"]; adx = pre["adx"]
    ma20 = pre["ma20"]; ma60 = pre["ma60"]
    c = pre["c"]
    if i < 70 or atr[i] <= 0 or np.isnan(ma20[i]) or np.isnan(ma60[i]):
        return classify(atr[i], 0, 0, 0, 0, 0, 0, 0, 0) if i >= 0 else None
    dev = (c[i] - ma20[i]) / atr[i]
    slope = (ma20[i] - ma20[i-10]) / atr[i]
    ma60_now = ma60[i] if not np.isnan(ma60[i]) else ma20[i]
    spread = (ma20[i] - ma60_now) / atr[i]
    mom = (c[i] - c[i-10]) / atr[i]
    body = pre["abody"]
    body_recent = np.mean(body[i-2:i+1])
    body_prior = np.mean(body[i-12:i-2]) if i >= 13 else body_recent
    return classify(atr[i], adx[i], adx[i-10], slope, spread, dev, mom,
                    body_recent, body_prior)


t0 = time.time()

# ═══ 预计算 + 采样 (1H) ═══
pre_1h = {}
samples_1h = []  # (sym, i, episode, long_side, out)
for sym in syms:
    df1 = data[sym].get("1h")
    if df1 is None or len(df1) < 260:
        continue
    pre = precompute(df1)
    pre_1h[sym] = pre
    c1 = df1["close"].values; h1 = df1["high"].values; l1 = df1["low"].values
    n1 = len(df1)
    atr1 = pre["atr"]
    idx1 = df1.index.values.astype("datetime64[ns]")
    for i in range(200, n1 - 25, 6):
        ts = pd.Timestamp(idx1[i])
        sd = daily_states.get(sym, {}).get(ts.date())
        if sd not in ("bull", "bear"):
            continue
        s4 = t4_state(sym, ts)
        if s4 == "N":
            continue
        episode = s4 != sd
        entry = c1[i]; a = atr1[i]
        if a <= 0 or np.isnan(a):
            continue
        long_side = sd == "bull"
        out = 0
        for k in range(1, 25):
            if i+k >= n1:
                break
            if long_side:
                if h1[i+k] >= entry + a:
                    out = 1; break
                if l1[i+k] <= entry - a:
                    out = -1; break
            else:
                if l1[i+k] <= entry - a:
                    out = 1; break
                if h1[i+k] >= entry + a:
                    out = -1; break
        samples_1h.append((sym, i, episode, long_side, out))

# ═══ 预计算 + 采样 (4H) ═══
pre_4h = {}
samples_4h = []  # (sym, i, long_side, out)
for sym in syms:
    df4 = data[sym].get("4h")
    if df4 is None or len(df4) < 220:
        continue
    pre = precompute(df4)
    pre_4h[sym] = pre
    c4 = df4["close"].values; h4 = df4["high"].values; l4 = df4["low"].values
    n4 = len(df4)
    atr4 = pre["atr"]
    idx4 = df4.index.values.astype("datetime64[ns]")
    for i in range(200, n4 - 12, 2):
        ts = pd.Timestamp(idx4[i])
        sd = daily_states.get(sym, {}).get(ts.date())
        if sd not in ("bull", "bear"):
            continue
        entry = c4[i]; a = atr4[i]
        if a <= 0 or np.isnan(a):
            continue
        long_side = sd == "bull"
        out = 0
        for k in range(1, 13):
            if i+k >= n4:
                break
            if long_side:
                if h4[i+k] >= entry + a:
                    out = 1; break
                if l4[i+k] <= entry - a:
                    out = -1; break
            else:
                if l4[i+k] <= entry - a:
                    out = 1; break
                if h4[i+k] >= entry + a:
                    out = -1; break
        samples_4h.append((sym, i, long_side, out))
print(f"预计算完成: 1H={len(samples_1h)} 4H={len(samples_4h)} 采样, {time.time()-t0:.1f}s", flush=True)


def tune_power(samples, pre, env_ep_only, label):
    print(f"\n--- {label} ---")
    if env_ep_only:
        pool = [s for s in samples if s[2]]
        base_w = sum(1 for s in pool if s[4] == 1); base_l = sum(1 for s in pool if s[4] == -1)
        base = base_w / (base_w + base_l) * 100
    else:
        pool = samples
        base_w = sum(1 for s in pool if s[3] == 1); base_l = sum(1 for s in pool if s[3] == -1)
        base = base_w / (base_w + base_l) * 100
    print(f"基线: n={len(pool)} 胜率={base:.1f}%")
    best = None
    for pw in [15, 20, 30, 40, 50]:
        for sh in [5, 10, 15]:
            w = l = 0
            for s in pool:
                sym, i = s[0], s[1]
                long_side = s[3] if env_ep_only else s[2]
                out = s[4] if env_ep_only else s[3]
                p = pre[sym]
                if i < pw + 1:
                    continue
                score = power_at(p, i, pw, sh)
                if long_side and score <= 40:
                    continue
                if not long_side and score >= -40:
                    continue
                if out == 1:
                    w += 1
                elif out == -1:
                    l += 1
            n = w + l
            if n < 400:
                continue
            wr = w / n * 100
            d = wr - base
            print(f"PW={pw:>2} SH={sh:>2}: n={n:>6} 胜率={wr:>5.1f}% Δ={d:>+5.1f}pp")
            if best is None or d > best[0]:
                best = (d, pw, sh, wr, n)
    if best:
        print(f"最优: PW={best[1]} SH={best[2]} Δ={best[0]:+.1f}pp 胜率={best[3]:.1f}% (n={best[4]})")


def tune_phase(samples, pre, label, horizon_w=None):
    print(f"\n--- {label} ---")
    print(f"{'TW':>4} {'n_up':>7} {'up胜率':>7} {'n_dn':>7} {'dn胜率':>7} {'区分度':>6}")
    best = None
    for tw in [50, 70, 90, 120, 150]:
        up_w = up_l = dn_w = dn_l = 0
        for s in samples:
            sym, i = s[0], s[1]
            p = pre[sym]
            if i < tw:
                continue
            ms = phase_at(p, i, tw)
            state = ms.get("state", "")
            out = s[4] if len(s) == 5 else s[3]
            if state == "trend_up":
                if out == 1:
                    up_w += 1
                elif out == -1:
                    up_l += 1
            elif state == "trend_down":
                if out == -1:
                    dn_w += 1
                elif out == 1:
                    dn_l += 1
        n_up = up_w + up_l
        n_dn = dn_w + dn_l
        if n_up < 300 or n_dn < 300:
            continue
        up_wr = up_w / n_up * 100
        dn_wr = dn_w / n_dn * 100
        diff = up_wr - dn_wr
        print(f"{tw:>4} {n_up:>7} {up_wr:>6.1f}% {n_dn:>7} {dn_wr:>6.1f}% {diff:>+5.1f}")
        if best is None or diff > best[0]:
            best = (diff, tw, up_wr, dn_wr)
    if best:
        print(f"最优: TW={best[1]} 区分度={best[0]:+.1f}pp (up {best[2]:.1f}% vs dn {best[3]:.1f}%)")


print("═══ A. 多空力量窗口 ═══")
tune_power(samples_1h, pre_1h, True, "1H 插曲+力量极端")
tune_power(samples_4h, pre_4h, False, "4H 力量极端")

print("\n═══ B. 趋势判定窗口 ═══")
tune_phase(samples_1h, pre_1h, "1H 周期")
tune_phase(samples_4h, pre_4h, "4H 周期")

print(f"\n总耗时: {time.time()-t0:.1f}s")
