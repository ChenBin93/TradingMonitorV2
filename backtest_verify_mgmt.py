#!/usr/bin/env python3
"""实盘溢价验证 V1+V3: 固定结构止损 + 移动止损/订单管理 (416天)

事件: 日线空/中性 + 1H BB(20,2.5σ) 破上轨做空
V1: 结构止损(近6根高点) vs ATR止损; 更优入场价模拟(5M级 -0.15/-0.3 ATR)
V3: 移动止损(盈利后保本/锁定) + 时间止损 vs 静态
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

events = []  # (sym, i, entry, atr, sl_struct)
for sym in syms:
    df4 = data[sym].get("4h")
    df1 = data[sym].get("1h")
    if df4 is None or df1 is None or len(df4) < 70 or len(df1) < 130:
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
    ma = c1s.rolling(20).mean().values
    sd = c1s.rolling(20).std().values
    for i in range(100, n1 - 25):
        if np.isnan(ma[i]) or np.isnan(sd[i]) or sd[i] <= 0:
            continue
        ts = pd.Timestamp(idx1[i])
        ds = daily_states.get(sym, {}).get(ts.date())
        if ds not in ("bear", "neut"):
            continue
        up = ma[i] + 2.5 * sd[i]
        if c1[i] <= up:
            continue
        entry = c1[i]; a = atr1[i]
        if a <= 0:
            continue
        sl_struct = max(h1[max(0, i-5):i+1])  # 近6根(含破轨bar)高点
        events.append((sym, i, entry, a, sl_struct))
print(f"事件: {len(events)}", flush=True)

HL = {}
for sym in syms:
    df1 = data[sym].get("1h")
    if df1 is not None:
        HL[sym] = (df1["high"].values, df1["low"].values, df1["close"].values, len(df1))

def sim_static(sub, tp, sl, w, entry_adj=0.0):
    """静态 TP/SL, entry_adj = 入场价调整(ATR单位, 正=卖更高)"""
    rets = []
    for sym, i, entry, a, _ in sub:
        h1_, l1_, c1_, n1_ = HL[sym]
        e2 = entry + entry_adj * a
        wp = lp = skip = False
        for k in range(1, w+1):
            if i+k >= n1_:
                break
            hit_p = l1_[i+k] <= e2 - tp * a
            hit_l = h1_[i+k] >= e2 + sl * a
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
            rets.append((e2 - c1_[i+kk]) / a)
    return rets

def sim_struct_sl(sub, tp, w, entry_adj=0.0):
    """结构止损: SL = 近6根高点, TP = tp ATR"""
    rets = []; sl_dists = []
    for sym, i, entry, a, sl_s in sub:
        h1_, l1_, c1_, n1_ = HL[sym]
        e2 = entry + entry_adj * a
        sl_dist = (sl_s - e2) / a  # 止损距离(ATR)
        sl_dists.append(sl_dist)
        wp = lp = skip = False
        for k in range(1, w+1):
            if i+k >= n1_:
                break
            hit_p = l1_[i+k] <= e2 - tp * a
            hit_l = h1_[i+k] >= e2 + sl_dist * a
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
            rets.append(-sl_dist)
        else:
            kk = min(w, n1_-i-1)
            rets.append((e2 - c1_[i+kk]) / a)
    return rets, np.mean(sl_dists)

print()
print("═══ V1: 结构止损 vs ATR止损 (TP=0.5, W=48) ═══")
for tag, adj in [("1H收盘入场", 0.0), ("入场-0.15ATR(5M)", 0.15), ("入场-0.3ATR(5M)", 0.3)]:
    rets = sim_static(events, 0.5, 0.5, 48, adj)
    n = len(rets)
    wr = sum(1 for r in rets if r > 0)/n*100
    ev = np.mean(rets)
    print(f"ATR止损(SL=0.5) {tag:<18}: 胜率={wr:>5.1f}% EV={ev:+.3f}")

for tag, adj in [("1H收盘入场", 0.0), ("入场-0.15ATR(5M)", 0.15), ("入场-0.3ATR(5M)", 0.3)]:
    rets, avg_sl = sim_struct_sl(events, 0.5, 48, adj)
    n = len(rets)
    wr = sum(1 for r in rets if r > 0)/n*100
    ev = np.mean(rets)
    print(f"结构止损(近6高) {tag:<18}: 胜率={wr:>5.1f}% EV={ev:+.3f} 平均止损距离={avg_sl:.2f}ATR")

print()
print("═══ V3: 订单管理模拟 (TP=0.5/SL=0.5 基线) ═══")
def sim_mgmt(sub, tp, sl, w, mgmt):
    """mgmt: 'static' | 'breakeven'(盈利0.25后移保本) | 'lock'(盈利0.25后移+0.1) | 'time12' | 'time24'"""
    rets = []
    for sym, i, entry, a, _ in sub:
        h1_, l1_, c1_, n1_ = HL[sym]
        sl_level = entry + sl * a      # 当前止损价
        protected = False
        exit_ret = None; skip = False
        for k in range(1, w+1):
            if i+k >= n1_:
                break
            # 先检查盈利移动
            if not protected:
                profit = (entry - l1_[i+k]) / a
                if profit >= 0.25:
                    if mgmt == "breakeven":
                        sl_level = entry
                    elif mgmt == "lock":
                        sl_level = entry - 0.1 * a
                    protected = True
            if mgmt == "time12" and k >= 12:
                exit_ret = (entry - c1_[i+k]) / a
                break
            if mgmt == "time24" and k >= 24:
                exit_ret = (entry - c1_[i+k]) / a
                break
            hit_p = l1_[i+k] <= entry - tp * a
            hit_l = h1_[i+k] >= sl_level
            if hit_p and hit_l:
                skip = True; break
            if hit_p:
                exit_ret = tp; break
            if hit_l:
                exit_ret = -(sl_level - entry) / a
                break
        if skip:
            continue
        if exit_ret is None:
            kk = min(w, n1_-i-1)
            exit_ret = (entry - c1_[i+kk]) / a
        rets.append(exit_ret)
    return rets

for mgmt, name in [("static", "静态(基线)"), ("breakeven", "盈利0.25→保本"), ("lock", "盈利0.25→锁定+0.1"), ("time12", "时间止损12bar"), ("time24", "时间止损24bar")]:
    rets = sim_mgmt(events, 0.5, 0.5, 48, mgmt)
    n = len(rets)
    wr = sum(1 for r in rets if r > 0)/n*100
    wr0 = sum(1 for r in rets if r >= 0)/n*100
    ev = np.mean(rets)
    avg_win = np.mean([r for r in rets if r > 0]) if any(r > 0 for r in rets) else 0
    avg_loss = -np.mean([r for r in rets if r < 0]) if any(r < 0 for r in rets) else 0
    print(f"{name:<20}: 胜率(>0)={wr:>5.1f}% 不亏率(>=0)={wr0:>5.1f}% EV={ev:+.3f} 均赢={avg_win:.2f} 均亏={avg_loss:.2f}")
