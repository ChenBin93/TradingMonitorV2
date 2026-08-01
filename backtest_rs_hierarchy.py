#!/usr/bin/env python3
"""416天: 三层趋势 × RS 增量验证

在日线×4H×1H 细分组合内, 按 4H RS 分档测顺势方向 1:1 胜率,
验证 RS 是否在细分组合下具备改变胜率的优势。
"""
import sys
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, ".")

from backtest_engine import load_all

with open("config.yaml") as f:
    import yaml
    cfg = yaml.safe_load(f)

data = load_all()
syms = list(data.keys())

# ── 日线状态 (4H 合成) ──
daily_states = {}
for sym in syms:
    df4 = data[sym].get("4h")
    if df4 is None or len(df4) < 100:
        continue
    daily = df4.resample("1D").last()
    ma20 = daily["close"].rolling(20).mean()
    ma60 = daily["close"].rolling(60).mean()
    states = {}
    for ts, row in daily.iterrows():
        if pd.isna(ma60[ts]):
            continue
        if row["close"] > ma20[ts] and row["close"] > ma60[ts]:
            states[ts.date()] = "bull"
        elif row["close"] < ma20[ts] and row["close"] < ma60[ts]:
            states[ts.date()] = "bear"
        else:
            states[ts.date()] = "neut"
    daily_states[sym] = states

# ── 4H/1H 趋势状态 (已收盘) ──
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
    ma60 = np.mean(c[pos-59:pos+1])
    if c[pos] > ma20 and c[pos] > ma60:
        return "bull"
    if c[pos] < ma20 and c[pos] < ma60:
        return "bear"
    return "neut"

def t1_state(sym, ts):
    df1 = data[sym].get("1h")
    if df1 is None or len(df1) < 70:
        return "N"
    idx = df1.index.values.astype("datetime64[ns]")
    pos = int(np.searchsorted(idx, np.datetime64(ts) - np.timedelta64(60, "m"), side="right")) - 1
    if pos < 60:
        return "N"
    c = df1["close"].values
    ma5 = np.mean(c[pos-4:pos+1])
    ma20 = np.mean(c[pos-19:pos+1])
    ma60 = np.mean(c[pos-59:pos+1])
    if ma5 > ma20 > ma60:
        return "bull"
    if ma5 < ma20 < ma60:
        return "bear"
    return "neut"

# ── 横截面 4H RS (live 混合公式, 已收盘) ──
df4s = {s: data[s].get("4h") for s in syms if data[s].get("4h") is not None and len(data[s].get("4h")) > 200}
idx4 = {s: d.index.values.astype("datetime64[ns]") for s, d in df4s.items()}
close4 = {s: d["close"].values for s, d in df4s.items()}
high4 = {s: d["high"].values for s, d in df4s.items()}
low4 = {s: d["low"].values for s, d in df4s.items()}

def atr_series(closes, highs, lows, period=50):
    n = len(closes)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
    out = np.zeros(n)
    e = 0.0
    alpha = 1 / period
    for i in range(n):
        e = tr[i] if i == 0 else alpha * tr[i] + (1 - alpha) * e
        out[i] = e
    return out

atr4 = {s: atr_series(close4[s], high4[s], low4[s]) for s in df4s}
btc_sym = "BTC/USDT:USDT"
btc_idx = idx4[btc_sym]
n4 = len(btc_idx)
windows = [1, 2, 3]
ws = np.array([1.0 / w for w in windows])
ws /= ws.sum()

rs_at_ts = {}
for t in range(3, n4):
    btc_c = close4[btc_sym][t]
    if btc_c <= 0:
        continue
    btc_atr_pct = atr4[btc_sym][t] / btc_c * 100
    btc_zs = []
    for w in windows:
        if t < w or close4[btc_sym][t-w] <= 0:
            continue
        btc_zs.append(((btc_c - close4[btc_sym][t-w]) / close4[btc_sym][t-w] * 100) / max(btc_atr_pct, 0.01))
    if not btc_zs:
        continue
    btc_z = sum(btc_zs[j] * ws[j] for j in range(len(btc_zs)))
    zd = {}
    for sym, d in df4s.items():
        pos = int(np.searchsorted(idx4[sym], btc_idx[t], side="right")) - 1
        if pos < 3 or pos >= len(d):
            continue
        c = close4[sym]
        if c[pos] <= 0:
            continue
        ap = atr4[sym][pos] / c[pos] * 100
        zs = []
        for w in windows:
            if pos < w or c[pos-w] <= 0:
                continue
            zs.append(((c[pos] - c[pos-w]) / c[pos-w] * 100) / max(ap, 0.01))
        z_avg = sum(zs[j] * ws[j] for j in range(len(zs))) if zs else 0
        zd[sym] = z_avg - btc_z
    if len(zd) < 10:
        continue
    vals = np.array(list(zd.values()))
    scores = {}
    for sym, z in zd.items():
        rank = (vals < z).sum() / max(len(vals) - 1, 1)
        rank_dev = (rank - 0.5) * 2
        norm_z = np.clip(z / 3.0, -1, 1)
        scores[sym] = (norm_z * 0.5 + rank_dev * 0.5) * 100
    rs_at_ts[t] = scores
print(f"4H RS precomputed at {len(rs_at_ts)} timestamps", flush=True)

def get_rs(sym, ts):
    t = int(np.searchsorted(btc_idx, np.datetime64(ts) - np.timedelta64(240, "m"), side="right")) - 1
    if t not in rs_at_ts:
        return None
    return rs_at_ts[t].get(sym)

# ── 主循环: 三层组合 × RS 分档 × 顺势方向 1:1 ──
results = defaultdict(lambda: {"w": 0, "l": 0})
for sym in syms:
    df1 = data[sym].get("1h")
    if df1 is None or len(df1) < 200:
        continue
    st_d = daily_states.get(sym, {})
    if not st_d:
        continue
    idx = df1.index.values.astype("datetime64[ns]")
    c = df1["close"].values
    h = df1["high"].values
    l = df1["low"].values
    n = len(df1)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
    atr = np.zeros(n)
    e = 0.0
    for i in range(n):
        e = tr[i] if i == 0 else (1/14) * tr[i] + (13/14) * e
        atr[i] = e
    for i in range(100, n - 25):
        ts = pd.Timestamp(idx[i])
        sd = st_d.get(ts.date(), None)
        if sd is None or sd == "neut":
            continue
        s4 = t4_state(sym, ts)
        s1 = t1_state(sym, ts)
        if s4 == "N" or s1 == "N":
            continue
        rs = get_rs(sym, ts)
        if rs is None:
            continue
        entry = c[i]
        a = atr[i]
        if a <= 0:
            continue
        # 只测顺势方向: 日线多头做多, 日线空头做空
        win = loss = False
        if sd == "bull":
            for k in range(1, 25):
                if i+k >= n:
                    break
                if h[i+k] >= entry + a:
                    win = True
                    break
                if l[i+k] <= entry - a:
                    loss = True
                    break
        else:
            for k in range(1, 25):
                if i+k >= n:
                    break
                if l[i+k] <= entry - a:
                    win = True
                    break
                if h[i+k] >= entry + a:
                    loss = True
                    break
        if not win and not loss:
            continue
        # RS 分档
        if rs < -40:
            rb = "RS弱<-40"
        elif rs < -15:
            rb = "RS偏弱-40~-15"
        elif rs < 15:
            rb = "RS中-15~15"
        elif rs < 40:
            rb = "RS偏强15~40"
        else:
            rb = "RS强>40"
        key = (sd, s4, s1, rb)
        if win:
            results[key]["w"] += 1
        else:
            results[key]["l"] += 1

# ── 输出: 三层组合 × RS ──
print()
print("=== 日线顺势入场 × RS 分档 (416天, 1:1) ===")
print(f"{'组合':<32} {'RS':<14} {'n':>7} {'WR':>7} {'基差':>6}")
print("-" * 70)

# 先算每组基础胜率 (不含RS)
base_wr = defaultdict(lambda: {"w": 0, "l": 0})
for (sd, s4, s1, rb), d in results.items():
    base_wr[(sd, s4, s1)]["w"] += d["w"]
    base_wr[(sd, s4, s1)]["l"] += d["l"]

for (sd, s4, s1), bd in sorted(base_wr.items(), key=lambda kv: -(kv[1]["w"]+kv[1]["l"])):
    bn = bd["w"] + bd["l"]
    if bn < 3000:
        continue
    bwr = bd["w"] / bn * 100
    print(f"{sd}×{s4}×{s1:<24} {'(基线)':<14} {bn:>7} {bwr:>6.1f}% {'':>6}")
    for rb in ["RS弱<-40", "RS偏弱-40~-15", "RS中-15~15", "RS偏强15~40", "RS强>40"]:
        d = results.get((sd, s4, s1, rb))
        if not d:
            continue
        n = d["w"] + d["l"]
        if n < 500:
            continue
        wr = d["w"] / n * 100
        diff = wr - bwr
        print(f"{'':<32} {rb:<14} {n:>7} {wr:>6.1f}% {diff:>+5.1f}pp", flush=True)
    print()
