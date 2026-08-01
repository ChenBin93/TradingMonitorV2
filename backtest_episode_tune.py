#!/usr/bin/env python3
"""趋势回调系统 (顺大逆小) 最优参数组 — TP/SL/窗口/强K确认 网格

入场: 日线顺势 + 4H逆向(插曲) + 贴强S/R(touch>=3, 0.3ATR) [+ 强K确认可选]
出场: TP×ATR / SL×ATR 先到者 (同bar双命中跳过), 窗口限时
评估: EV/ATR, 简单夏普(收益均值/std), 胜率, 平均赢/平均亏, 频率
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
    ma20 = np.mean(c[pos-19:pos+1]); ma60 = np.mean(c[pos-59:pos+1])
    if c[pos] > ma20 and c[pos] > ma60:
        return "bull"
    if c[pos] < ma20 and c[pos] < ma60:
        return "bear"
    return "neut"

# 收集: 插曲+贴强SR 样本, 72bar 前向序列
samples = []
for sym in syms:
    df1 = data[sym].get("1h")
    if df1 is None or len(df1) < 180:
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
    for i in range(100, n1 - 73):
        ts = pd.Timestamp(idx1[i])
        sd = daily_states.get(sym, {}).get(ts.date())
        if sd not in ("bull", "bear"):
            continue
        s4 = t4_state(sym, ts)
        if not (s4 in ("bull", "bear") and s4 != sd):
            continue
        rng = h1[i]-l1[i]
        if rng <= 0 or atr1[i] <= 0:
            continue
        bp = abs(c1[i]-o1[i])/rng
        entry = c1[i]; a = atr1[i]
        long_side = sd == "bull"
        sign = 1.0 if long_side else -1.0
        f = np.zeros(72); g = np.zeros(72)
        for k in range(1, 73):
            f[k-1] = (h1[i+k]-entry)/a * sign
            g[k-1] = (entry-l1[i+k])/a * sign
        near_sr = False
        levels = find_swing_levels(df1.iloc[:i+1], 50)
        for lv in levels:
            if abs(entry - lv.price) <= 0.3*a and lv.touch_count >= 3:
                near_sr = True
                break
        if not near_sr:
            continue
        samples.append((f, g, bp >= 0.6, ts, sym))
print(f"插曲+贴强SR 样本: {len(samples)}", flush=True)

TP_GRID = [3.0, 4.0, 5.0, 6.0]
SL_GRID = [1.0, 1.25, 1.5, 2.0]
W_GRID = [72, 120, 168, 240]

def trade(sub, tp, sl, w):
    """返回 (wins, losses, avg_win, avg_loss) — 收益按 ATR 计
    窗口内未到达 TP/SL → 时间平仓 (按第 w bar 收盘对齐收益)"""
    rets = []
    for f, g, *_ in sub:
        wp = lp = False
        for k in range(min(w, 72)):
            hit_p = f[k] >= tp; hit_l = g[k] >= sl
            if hit_p and hit_l:
                break
            if hit_p:
                wp = True; break
            if hit_l:
                lp = True; break
        if wp:
            rets.append(tp)
        elif lp:
            rets.append(-sl)
        else:
            rets.append(f[min(w, 72)-1])
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]
    return wins, losses

results = []
for tp in TP_GRID:
    for sl in SL_GRID:
        for w in W_GRID:
            for kreq, kname in [(False, "任意K"), (True, "强K")]:
                sub = [s for s in samples if not kreq or s[2]]
                wins, losses = trade(sub, tp, sl, w)
                n = len(wins) + len(losses)
                if n < 100:
                    continue
                ev = (sum(wins) + sum(losses)) / n
                all_ret = wins + losses
                std = np.std(all_ret) if len(all_ret) > 1 else 1e-9
                sharpe = ev / std if std > 0 else 0
                avg_w = np.mean(wins) if wins else 0
                avg_l = -np.mean(losses) if losses else 0
                results.append({
                    "tp": tp, "sl": sl, "w": w, "k": kname,
                    "n": n, "wr": len(wins)/n*100,
                    "ev": ev, "sharpe": sharpe,
                    "avg_w": avg_w, "avg_l": avg_l,
                    "rr": avg_w/avg_l if avg_l > 0 else 0,
                })
results.sort(key=lambda r: -r["ev"])
print()
print(f"{'TP':>4} {'SL':>4} {'W':>4} {'K':<4} {'n':>5} {'胜率':>6} {'EV/ATR':>7} {'夏普':>6} {'均赢':>6} {'均亏':>6} {'盈亏比':>6}")
for r in results[:15]:
    print(f"{r['tp']:>4} {r['sl']:>4} {r['w']:>4} {r['k']:<4} {r['n']:>5} {r['wr']:>5.1f}% {r['ev']:>7.3f} {r['sharpe']:>6.2f} {r['avg_w']:>6.2f} {r['avg_l']:>6.2f} {r['rr']:>6.2f}")

print()
print("=== TOP5 按夏普 ===")
results.sort(key=lambda r: -r["sharpe"])
print(f"{'TP':>4} {'SL':>4} {'W':>4} {'K':<4} {'n':>5} {'胜率':>6} {'EV/ATR':>7} {'夏普':>6} {'盈亏比':>6}")
for r in results[:5]:
    print(f"{r['tp']:>4} {r['sl']:>4} {r['w']:>4} {r['k']:<4} {r['n']:>5} {r['wr']:>5.1f}% {r['ev']:>7.3f} {r['sharpe']:>6.2f} {r['rr']:>6.2f}")

# 最优组合按年稳定性
best = results[0] if results else None
if best:
    print()
    print(f"=== 最优(EV): TP={best['tp']} SL={best['sl']} W={best['w']} K={best['k']} — 按年 ===")
    sub_all = [s for s in samples if (best["k"] == "任意K") or s[2]]
    years = {}
    for s in sub_all:
        years.setdefault(s[3].year, []).append(s)
    for yr in sorted(years):
        wins, losses = trade(years[yr], best["tp"], best["sl"], best["w"])
        n = len(wins) + len(losses)
        if n >= 30:
            ev = (sum(wins) + sum(losses)) / n
            print(f"{yr}: n={n:>4} 胜率={len(wins)/n*100:>5.1f}% EV/ATR={ev:>+.3f}")
