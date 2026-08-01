#!/usr/bin/env python3
"""实盘溢价验证 V2: 多根价格行为形态 (双顶双底/旗形/三推) 胜率统计 (416天)

形态 → 确认入场, TP=1ATR/SL=1ATR/W=48 (1:1), 同bar双命中跳过
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from backtest_engine import load_all

data = load_all()
syms = list(data.keys())

daily_states = {}
for sym in syms:
    df4 = data[sym].get("4h")
    if df4 is None or len(df4) < 100:
        continue
    daily = df4.resample("1D").last()
    ma20 = daily["close"].rolling(20).mean()
    atr_d = (daily["high"] - daily["low"]).rolling(14).mean()
    for ts, row in daily.iterrows():
        if pd.isna(ma20[ts]) or pd.isna(atr_d[ts]) or atr_d[ts] <= 0:
            continue
        dev = (row["close"] - ma20[ts]) / atr_d[ts]
        if dev > 0.5:
            daily_states.setdefault(sym, {})[ts.date()] = "bull"
        elif dev < -0.5:
            daily_states.setdefault(sym, {})[ts.date()] = "bear"
        else:
            daily_states.setdefault(sym, {})[ts.date()] = "neut"

# 收集所有形态事件
events = []  # (sym, i, entry, a, form, ds)
for sym in syms:
    df4 = data[sym].get("4h")
    df1 = data[sym].get("1h")
    if df4 is None or df1 is None or len(df4) < 70 or len(df1) < 200:
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

    # swing 极值 (±2 bar)
    sw_hi = [(i, h1[i]) for i in range(2, n1-2) if h1[i] >= max(h1[i-2:i+3])]
    sw_lo = [(i, l1[i]) for i in range(2, n1-2) if l1[i] <= min(l1[i-2:i+3])]

    # ── 双顶 (做空): 两高点相近, 跌破颈线确认 ──
    for a_idx in range(len(sw_hi)-1):
        i1, p1 = sw_hi[a_idx]
        i2, p2 = sw_hi[a_idx+1]
        if not (3 <= i2 - i1 <= 40):
            continue
        if abs(p1 - p2) > 0.3 * atr1[i2]:
            continue
        # 颈线 = 两高点之间最低点
        neck = min(l1[i1:i2+1])
        if p1 - neck < 0.4 * atr1[i2]:
            continue
        # 确认: i2 之后 close 跌破颈线
        for k in range(1, 8):
            if i2 + k >= n1 - 50:
                break
            if c1[i2+k] < neck:
                entry = c1[i2+k]; a = atr1[i2+k]
                ts = pd.Timestamp(idx1[i2+k])
                ds = daily_states.get(sym, {}).get(ts.date())
                events.append((sym, i2+k, entry, a, "双顶", ds))
                break

    # ── 双底 (做多): 对称 ──
    for a_idx in range(len(sw_lo)-1):
        i1, p1 = sw_lo[a_idx]
        i2, p2 = sw_lo[a_idx+1]
        if not (3 <= i2 - i1 <= 40):
            continue
        if abs(p1 - p2) > 0.3 * atr1[i2]:
            continue
        neck = max(h1[i1:i2+1])
        if neck - p1 < 0.4 * atr1[i2]:
            continue
        for k in range(1, 8):
            if i2 + k >= n1 - 50:
                break
            if c1[i2+k] > neck:
                entry = c1[i2+k]; a = atr1[i2+k]
                ts = pd.Timestamp(idx1[i2+k])
                ds = daily_states.get(sym, {}).get(ts.date())
                events.append((sym, i2+k, entry, a, "双底", ds))
                break

    # ── 三推 (做空): 三个接近高点, 第三推后破第二推低点 ──
    for a_idx in range(len(sw_hi)-2):
        i1, p1 = sw_hi[a_idx]
        i2, p2 = sw_hi[a_idx+1]
        i3, p3 = sw_hi[a_idx+2]
        if not (3 <= i2 - i1 <= 25 and 3 <= i3 - i2 <= 25):
            continue
        if abs(p1-p2) > 0.3*atr1[i3] or abs(p2-p3) > 0.3*atr1[i3]:
            continue
        # 第二推低点 (i2 与 i3 之间最低)
        pull = min(l1[i2:i3+1])
        for k in range(1, 8):
            if i3 + k >= n1 - 50:
                break
            if c1[i3+k] < pull:
                entry = c1[i3+k]; a = atr1[i3+k]
                ts = pd.Timestamp(idx1[i3+k])
                ds = daily_states.get(sym, {}).get(ts.date())
                events.append((sym, i3+k, entry, a, "三推", ds))
                break

    # ── 牛旗 (做多): 旗杆(5bar涨>2ATR) + 旗面(3-15bar横盘<0.8ATR) + 突破 ──
    for i in range(100, n1 - 60):
        if atr1[i] <= 0:
            continue
        pole = c1[i] - c1[i-5]
        if pole < 2.0 * atr1[i]:
            continue
        # 旗面: i+1..i+f 横盘
        for f in range(3, 16):
            if i + f >= n1 - 50:
                break
            seg = h1[i+1:i+f+1]; seg_l = l1[i+1:i+f+1]
            if max(seg) - min(seg_l) > 0.8 * atr1[i]:
                break
            # 突破旗面上沿
            top = max(seg)
            for k in range(1, 6):
                if i + f + k >= n1 - 50:
                    break
                if c1[i+f+k] > top:
                    entry = c1[i+f+k]; a = atr1[i+f+k]
                    ts = pd.Timestamp(idx1[i+f+k])
                    ds = daily_states.get(sym, {}).get(ts.date())
                    events.append((sym, i+f+k, entry, a, "牛旗", ds))
                    break
            break

    # ── 熊旗 (做空): 对称 ──
    for i in range(100, n1 - 60):
        if atr1[i] <= 0:
            continue
        pole = c1[i-5] - c1[i]
        if pole < 2.0 * atr1[i]:
            continue
        for f in range(3, 16):
            if i + f >= n1 - 50:
                break
            seg = h1[i+1:i+f+1]; seg_l = l1[i+1:i+f+1]
            if max(seg) - min(seg_l) > 0.8 * atr1[i]:
                break
            bot = min(seg_l)
            for k in range(1, 6):
                if i + f + k >= n1 - 50:
                    break
                if c1[i+f+k] < bot:
                    entry = c1[i+f+k]; a = atr1[i+f+k]
                    ts = pd.Timestamp(idx1[i+f+k])
                    ds = daily_states.get(sym, {}).get(ts.date())
                    events.append((sym, i+f+k, entry, a, "熊旗", ds))
                    break
            break

print(f"形态事件: {len(events)}", flush=True)
from collections import Counter
print(Counter(ev[4] for ev in events))

HL = {}
for sym in syms:
    df1 = data[sym].get("1h")
    if df1 is not None:
        HL[sym] = (df1["high"].values, df1["low"].values, df1["close"].values, len(df1))

def sim(sub, tp=1.0, sl=1.0, w=48):
    rets = []
    for sym, i, entry, a, form, ds in sub:
        h1_, l1_, c1_, n1_ = HL[sym]
        wp = lp = skip = False
        for k in range(1, w+1):
            if i+k >= n1_:
                break
            if form in ("双顶", "三推", "熊旗"):
                hit_p = l1_[i+k] <= entry - tp*a
                hit_l = h1_[i+k] >= entry + sl*a
            else:
                hit_p = h1_[i+k] >= entry + tp*a
                hit_l = l1_[i+k] <= entry - sl*a
            if hit_p and hit_l:
                skip = True; break
            if hit_p:
                wp = True; break
            if hit_l:
                lp = True; break
        if skip:
            continue
        if wp:
            rets.append(tp)
        elif lp:
            rets.append(-sl)
        else:
            kk = min(w, n1_-i-1)
            rets.append((entry - c1_[i+kk]) / a if form in ("双顶", "三推", "熊旗") else (c1_[i+kk] - entry) / a)
    return rets

print()
print(f"{'形态':<6} {'环境':<8} {'n':>6} {'胜率':>7} {'不亏率':>7} {'EV':>7}")
for form in ["双顶", "双底", "三推", "牛旗", "熊旗"]:
    sub = [e for e in events if e[4] == form]
    for env_name, env in [("全部", None), ("日线顺势", "auto"), ("日线逆势", "anti"), ("日线中性", "neut")]:
        if env == "auto":
            ss = [e for e in sub if (e[5] == "bull" and e[4] in ("双底", "牛旗")) or (e[5] == "bear" and e[4] in ("双顶", "三推", "熊旗"))]
        elif env == "anti":
            ss = [e for e in sub if e[5] not in (None, "neut") and not ((e[5] == "bull" and e[4] in ("双底", "牛旗")) or (e[5] == "bear" and e[4] in ("双顶", "三推", "熊旗")))]
        elif env == "neut":
            ss = [e for e in sub if e[5] == "neut"]
        else:
            ss = sub
        if len(ss) < 150:
            continue
        rets = sim(ss)
        n = len(rets)
        wr = sum(1 for r in rets if r > 0)/n*100
        wr0 = sum(1 for r in rets if r >= 0)/n*100
        ev = np.mean(rets)
        print(f"{form:<6} {env_name:<8} {n:>6} {wr:>6.1f}% {wr0:>6.1f}% {ev:>+7.3f}")
