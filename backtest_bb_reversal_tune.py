#!/usr/bin/env python3
"""布林带反转系统参数调优 — 上轨做空

触发: 1H close 破上轨 (cross) 或 破轨后下一根收回 (reclaim)
环境: 日线空 / 日线中性
出场: TP/SL × 1H ATR 先到, 时间平仓; 每笔 1% 风险资金曲线
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from backtest_engine import load_all

with open("config.yaml") as f:
    import yaml
    cfg = yaml.safe_load(f)

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

MAXB = 72
# 收集: 破轨样本 (f,g 空头有利/不利序列), 含收回变体
raw = []  # (ds, ts, sym, f, g, reclaim_f, reclaim_g)
for sym in syms:
    df4 = data[sym].get("4h")
    df1 = data[sym].get("1h")
    if df4 is None or df1 is None or len(df4) < 70 or len(df1) < MAXB + 130:
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
    c1s = pd.Series(c1)
    for P in [20, 25]:
        ma = c1s.rolling(P).mean().values
        sd = c1s.rolling(P).std().values
        for i in range(100, n1 - MAXB - 2):
            if np.isnan(ma[i]) or np.isnan(sd[i]) or sd[i] <= 0:
                continue
            ts = pd.Timestamp(idx1[i])
            ds = daily_states.get(sym, {}).get(ts.date())
            if ds not in ("bear", "neut"):
                continue
            rng = h1[i]-l1[i]
            if rng <= 0 or atr1[i] <= 0:
                continue
            entry = c1[i]; a = atr1[i]
            for sigma in [2.0, 2.5, 3.0]:
                up = ma[i] + sigma*sd[i]
                if c1[i] <= up:
                    continue
                f = np.zeros(MAXB); g = np.zeros(MAXB)
                for k in range(1, MAXB+1):
                    f[k-1] = (entry - l1[i+k]) / a
                    g[k-1] = (h1[i+k] - entry) / a
                # 收回变体: 下一根 close 收回轨内 → 在该根收盘入场
                reclaim = None
                if i + 1 < n1 and c1[i+1] < ma[i+1] + sigma*sd[i+1]:
                    f2 = np.zeros(MAXB); g2 = np.zeros(MAXB)
                    entry2 = c1[i+1]
                    for k in range(1, MAXB+1):
                        j = i + 1 + k
                        if j >= n1:
                            break
                        f2[k-1] = (entry2 - l1[j]) / a
                        g2[k-1] = (h1[j] - entry2) / a
                    reclaim = (f2, g2)
                raw.append((ds, P, sigma, "cross", f, g, ts))
                if reclaim is not None:
                    raw.append((ds, P, sigma, "reclaim", reclaim[0], reclaim[1], ts))
print(f"破轨样本: {len(raw)}", flush=True)

def trade(sub, tp, sl, w):
    rets = []
    for s in sub:
        f = s[4]; g = s[5]
        wp = lp = False
        for k in range(min(w, MAXB)):
            hp = f[k] >= tp; hl = g[k] >= sl
            if hp and hl:
                break
            if hp:
                wp = True; break
            if hl:
                lp = True; break
        if wp:
            rets.append(tp)
        elif lp:
            rets.append(-sl)
        else:
            rets.append(f[min(w, MAXB)-1])
    return rets

def evaluate(sub, tp, sl, w):
    rets = trade(sub, tp, sl, w)
    n = len(rets)
    wr = sum(1 for r in rets if r > 0)/n*100
    pnl = np.array(rets)/sl*0.01
    eq = np.cumsum(pnl)
    mdd = np.min(eq - np.maximum.accumulate(eq))
    return n, wr, np.mean(pnl), eq[-1], mdd

results = []
for env_name, env in [("日线空", "bear"), ("日线中性", "neut")]:
    sub_env = [s for s in raw if s[0] == env]
    print()
    print(f"═══ {env_name} (n={len(sub_env)}) ═══")
    best = None
    for P in [20, 25]:
        for sigma in [2.0, 2.5, 3.0]:
            for trig in ["cross", "reclaim"]:
                sub = [s for s in sub_env if s[1] == P and s[2] == sigma and s[3] == trig]
                if len(sub) < 200:
                    continue
                for tp in [0.3, 0.5, 0.75]:
                    for sl in [0.5, 0.75, 1.0]:
                        for w in [12, 24, 48, 72]:
                            n, wr, avg, total, mdd = evaluate(sub, tp, sl, w)
                            rec = (avg, n, wr, total, mdd, P, sigma, trig, tp, sl, w)
                            if best is None or avg > best[0]:
                                best = rec
                            # 记录高胜率+正EV
                            if wr >= 62 and avg > 0.003 and n >= 300:
                                results.append((avg, n, wr, total, mdd, P, sigma, trig, tp, sl, w, env_name))
    print(f"最优每笔: EV={best[0]*100:.3f}% n={best[1]} 胜率={best[2]:.1f}% 总={best[3]*100:+.0f}% 回撤={best[4]*100:.1f}% "
          f"[P={best[5]} σ={best[6]} {best[7]} TP={best[8]} SL={best[9]} W={best[10]}]")

print()
print("=== 全环境高胜率档 (胜率>=62%, n>=300) TOP20 ===")
results.sort(key=lambda r: -r[0])
for r in results[:20]:
    print(f"{r[11]:<8} P={r[5]} σ={r[6]} {r[7]:<7} TP={r[8]} SL={r[9]} W={r[10]}: "
          f"n={r[1]:>5} 胜率={r[2]:>5.1f}% 每笔={r[0]*100:+.3f}% 总={r[3]*100:+.0f}% 回撤={r[4]*100:.1f}%")
