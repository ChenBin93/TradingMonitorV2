#!/usr/bin/env python3
"""四重条件: 日线顺势 + 强S/R(touch>=3) + RS × 价格行为增量 (416天, 1H入场 1:1)

层级:
  L0: 日线顺势 + 贴强S/R (无RS过滤)
  L1: L0 + RS同向 (做多RS>0 / 做空RS<0)
  L2: L0 + RS极端 (做多RS>60 / 做空RS<-60)
每层内对比价格行为: 基线 / 收阳收阴 / pin / 吞没 / 连续 / 双重确认 / 反确认
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from backtest_engine import load_all
from support_resistance import find_swing_levels

with open("config.yaml") as f:
    import yaml
    cfg = yaml.safe_load(f)

data = load_all()
syms = list(data.keys())

# ── 日线状态 (4H resample 1D, close vs MA20) ──
daily_states = {}
for sym in syms:
    df4 = data[sym].get("4h")
    if df4 is None or len(df4) < 100:
        continue
    daily = df4.resample("1D").last()
    ma20 = daily["close"].rolling(20).mean()
    states = {}
    for ts, row in daily.iterrows():
        if pd.isna(ma20[ts]):
            continue
        states[ts.date()] = "bull" if row["close"] > ma20[ts] else "bear"
    daily_states[sym] = states

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

rs_at = {}
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
            zs.append(((c[pos]-c[pos-w]) / c[pos-w] * 100) / max(ap, 0.01))
        z_avg = sum(zs[j] * ws[j] for j in range(len(zs))) if zs else 0
        zd[sym] = z_avg - btc_z
    if len(zd) < 10:
        continue
    vals = np.array(list(zd.values()))
    scores = {}
    for sym, z in zd.items():
        rank = (vals < z).sum() / max(len(vals)-1, 1)
        rank_dev = (rank - 0.5) * 2
        norm_z = np.clip(z / 3.0, -1, 1)
        scores[sym] = (norm_z * 0.5 + rank_dev * 0.5) * 100
    rs_at[t] = scores
print(f"[RS] {len(rs_at)} 个 4H 时间点", flush=True)

def get_rs(sym, ts):
    t = int(np.searchsorted(btc_idx, np.datetime64(ts) - np.timedelta64(240, "m"), side="right")) - 1
    if t not in rs_at:
        return None
    return rs_at[t].get(sym)

# ── 收集: 日线顺势 + 贴强S/R 的所有 1H bar ──
TOUCH_MIN = 3      # 强结构
NEAR_ATR = 0.3     # 贴水平位距离 (×1H ATR)

rows = []
for sym in syms:
    df1 = data[sym].get("1h")
    if df1 is None or len(df1) < 130:
        continue
    o1 = df1["open"].values; c1 = df1["close"].values
    h1 = df1["high"].values; l1 = df1["low"].values
    n1 = len(df1)
    tr1 = np.zeros(n1)
    for i in range(1, n1):
        tr1[i] = max(h1[i]-l1[i], abs(h1[i]-c1[i-1]), abs(l1[i]-c1[i-1]))
    atr1 = np.zeros(n1); e = 0.0
    for i in range(n1):
        e = tr1[i] if i == 0 else (1/14) * tr1[i] + (13/14) * e
        atr1[i] = e

    idx1 = df1.index.values.astype("datetime64[ns]")
    for i in range(100, n1 - 25):
        ts = pd.Timestamp(idx1[i])
        sd = daily_states[sym].get(ts.date())
        if sd is None:
            continue
        # 只在日线顺势方向测: bull→做多, bear→做空
        levels = find_swing_levels(df1.iloc[:i+1], 50)
        if not levels:
            continue
        price = c1[i]; a = atr1[i]
        if a <= 0:
            continue
        near = False
        for lv in levels:
            if abs(price - lv.price) <= NEAR_ATR * a and lv.touch_count >= TOUCH_MIN:
                near = True
                break
        if not near:
            continue
        rs = get_rs(sym, ts)
        if rs is None:
            continue
        # 价格行为特征
        rng = h1[i] - l1[i]
        body = abs(c1[i] - o1[i])
        body_bull = c1[i] > o1[i]
        lower_wick = min(o1[i], c1[i]) - l1[i]
        upper_wick = h1[i] - max(o1[i], c1[i])
        prev_bull = c1[i-1] > o1[i-1]
        prev_body = abs(c1[i-1] - o1[i-1])
        pa = {
            "cur_bull": body_bull,
            "pin": (lower_wick >= 0.6 * rng and upper_wick <= 0.2 * rng),
            "pin_up": (upper_wick >= 0.6 * rng and lower_wick <= 0.2 * rng),
            "engulf": (body >= prev_body and body_bull != prev_bull and body > 0),
            "prev_same": (prev_bull == body_bull),
            "strong_body": (body >= 0.6 * rng and body > 0),
        }
        rows.append({
            "sym": sym, "i": i, "entry": price, "atr": a,
            "sd": sd, "rs": rs, "pa": pa,
        })
print(f"[收集] 日线顺势+贴强S/R(touch>={TOUCH_MIN}): {len(rows)} 样本", flush=True)

# ── 模拟 1:1 (entry±ATR, 24bar) ──
HL = {}
for sym in syms:
    df1 = data[sym].get("1h")
    if df1 is not None:
        HL[sym] = (df1["high"].values, df1["low"].values, len(df1))

def simulate(subset):
    w = l = 0
    for r in subset:
        h1_, l1_, n1_ = HL[r["sym"]]
        i = r["i"]; entry = r["entry"]; a = r["atr"]
        win = loss = False
        for k in range(1, 25):
            if i + k >= n1_:
                break
            if r["sd"] == "bull":      # 做多
                if h1_[i+k] >= entry + a: win = True; break
                if l1_[i+k] <= entry - a: loss = True; break
            else:                      # 做空
                if l1_[i+k] <= entry - a: win = True; break
                if h1_[i+k] >= entry + a: loss = True; break
        if win: w += 1
        elif loss: l += 1
    return w, l

def ztest(w1, l1, w0, l0):
    """两比例 z 检验: 子集 vs 基线"""
    n1 = w1 + l1; n0 = w0 + l0
    if n1 < 60 or n0 == 0:
        return None
    p1 = w1 / n1; p0 = w0 / n0
    p = (w1 + w0) / (n1 + n0)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n0))
    if se == 0:
        return None
    return (p1 - p0) / se

def show(label, w, l, base_w, base_l):
    n = w + l
    if n < 60:
        print(f"{label:<44} n={n:>6}  样本不足", flush=True)
        return
    pct = w / n * 100
    z = ztest(w, l, base_w, base_l)
    zs = f"  z={z:+.2f} {'***' if z and abs(z) >= 3 else '**' if z and abs(z) >= 2 else ''}" if z is not None else ""
    print(f"{label:<44} n={n:>6}  胜率={pct:>5.1f}%{zs}", flush=True)

def layer(name, subset):
    print()
    print(f"═══ {name} (n={len(subset)}) ═══", flush=True)
    w0, l0 = simulate(subset)
    show("  基线 (全部)", w0, l0, w0, l0)
    for tag, cond in [
        ("+收阳(多)/收阴(空)", lambda r: r["pa"]["cur_bull"] == (r["sd"] == "bull")),
        ("+pin 长下影/长上影", lambda r: r["pa"]["pin"] if r["sd"] == "bull" else r["pa"]["pin_up"]),
        ("+吞没 (反向包吞)", lambda r: r["pa"]["engulf"]),
        ("+前根同向 (连续)", lambda r: r["pa"]["prev_same"]),
        ("+实体≥60% (强K)", lambda r: r["pa"]["strong_body"]),
        ("+双重 (方向+pin)", lambda r: (r["pa"]["cur_bull"] == (r["sd"] == "bull")) and (r["pa"]["pin"] if r["sd"] == "bull" else r["pa"]["pin_up"])),
    ]:
        w, l = simulate([r for r in subset if cond(r)])
        show(f"  {tag}", w, l, w0, l0)
    w, l = simulate([r for r in subset if r["pa"]["cur_bull"] != (r["sd"] == "bull")])
    show(f"  反确认 (逆K方向)", w, l, w0, l0)

bull = [r for r in rows if r["sd"] == "bull"]
bear = [r for r in rows if r["sd"] == "bear"]

for name, sub in [("日线多 + 强支撑 (做多)", bull), ("日线空 + 强阻力 (做空)", bear)]:
    layer(name, sub)

    is_long = name.startswith("日线多")
    aligned = [r for r in sub if (r["rs"] > 0 if is_long else r["rs"] < 0)]
    layer(name + " + RS同向", aligned)

    extreme = [r for r in sub if (r["rs"] > 60 if is_long else r["rs"] < -60)]
    layer(name + " + RS极端", extreme)
