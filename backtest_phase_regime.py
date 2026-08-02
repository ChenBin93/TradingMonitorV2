#!/usr/bin/env python3
"""切换灵敏度实验 — 判定当前K状态的最优窗口

A. 合成数据 (有 ground truth): 趋势→震荡→趋势, 测各窗口切换响应延迟
B. 真实数据 (事后标签): 用未来走势定义真实状态, 测各窗口 滞后+准确
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from market_phase import analyze_market_state

# ═══════════════ A. 合成数据 ═══════════════
def make_regime_series(segments, seed=0):
    """segments: [(n, 'trend_up'|'trend_down'|'range', vol)]"""
    rng = np.random.default_rng(seed)
    o = []; h = []; l = []; c = []; v = []
    price = 100.0
    for n, kind, vol in segments:
        for _ in range(n):
            if kind == "trend_up":
                drift = 0.35 * vol
            elif kind == "trend_down":
                drift = -0.35 * vol
            else:
                drift = rng.normal(0, 0.02 * vol)
            o.append(price)
            price += drift + rng.normal(0, 0.15 * vol)
            c.append(price)
            h.append(price + vol * (0.2 + abs(rng.normal(0, 0.1))))
            l.append(price - vol * (0.2 + abs(rng.normal(0, 0.1))))
            v.append(100)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v})

# 序列: 趋势上300 → 震荡300 → 趋势下300 → 震荡300 → 趋势上300
df = make_regime_series([
    (300, "trend_up", 1.0), (300, "range", 1.0),
    (300, "trend_down", 1.0), (300, "range", 1.0),
    (300, "trend_up", 1.0),
])
TRUE = []
for n, kind, vol in [(300, "trend_up", 1.0), (300, "range", 1.0),
                     (300, "trend_down", 1.0), (300, "range", 1.0),
                     (300, "trend_up", 1.0)]:
    TRUE.extend([kind] * n)
TRUE = np.array(TRUE)
switches = [i for i in range(1, len(TRUE)) if TRUE[i] != TRUE[i-1]]
print(f"合成序列: {len(df)} 根, 切换点: {switches}", flush=True)

print("\n═══ A. 合成数据: 窗口 vs 切换响应延迟 ═══")
print(f"{'窗口':>5} {'切换后判对(根)':>16} {'稳态误判率':>12}")
for tw in [20, 30, 40, 50, 60, 80, 100, 150]:
    preds = []
    for i in range(tw, len(df)):
        ms = analyze_market_state(df.iloc[max(0, i - tw - 70):i + 1].reset_index(drop=True), window=tw)
        st = ms.get("state", "")
        if st == "trend_up":
            preds.append("trend_up")
        elif st == "trend_down":
            preds.append("trend_down")
        elif st == "range":
            preds.append("range")
        else:
            preds.append("transition")
    preds = np.array(preds)
    # 切换响应: 每个切换点后第几根 pred 变成新状态
    lags = []
    for sw in switches:
        new_state = TRUE[sw]
        if new_state == "range":
            # 震荡状态判定为 range 或 transition 都算"离开趋势"
            hit = [j for j in range(sw, min(sw + 120, len(preds)))
                   if preds[j] in ("range", "transition")]
        else:
            hit = [j for j in range(sw, min(sw + 120, len(preds))) if preds[j] == new_state]
        lags.append(hit[0] - sw if hit else 120)
    # 稳态误判: 远离切换点(±40)的区域, 判定与真实不符的比例
    stable_mask = np.ones(len(TRUE), bool)
    for sw in switches:
        stable_mask[max(0, sw-40):sw+40] = False
    wrong = 0
    total = 0
    for i in range(tw, len(preds)):
        if not stable_mask[i]:
            continue
        t = TRUE[i]
        p = preds[i]
        if t == "range":
            ok = p in ("range", "transition")
        else:
            ok = p == t
        total += 1
        if not ok:
            wrong += 1
    lag_med = int(np.median(lags))
    print(f"{tw:>5} {lag_med:>12}根  {wrong/max(total,1)*100:>10.1f}%")

# ═══════════════ B. 真实数据 (1H) ═══════════════
print("\n═══ B. 真实数据 (1H): 事后标签验证 ═══")
sys.path.insert(0, ".")
from backtest_engine import load_all

data = load_all(timeframes=["1h"])
syms = list(data.keys())

# 事后标签: 未来30根的净移动/ATR — 定义该时刻真实状态
def true_state_from_future(df, i, w=30, atr=None):
    c = df["close"].values
    a = atr[i] if atr is not None else 1
    if i + w >= len(c) or a <= 0:
        return None
    move = (c[i + w] - c[i]) / a
    if abs(move) > w * 0.25:
        return "trend_up" if move > 0 else "trend_down"
    if abs(move) < w * 0.12:
        return "range"
    return "transition"

results = {tw: {"lag": [], "acc": 0, "n": 0} for tw in [20, 30, 40, 50, 60, 80, 100, 150]}
true_by_sym = {}

for sym in syms:
    df1 = data[sym].get("1h")
    if df1 is None or len(df1) < 300:
        continue
    from market_phase import _atr_series
    atr = _atr_series(df1)
    n = len(df1)
    # 事后标签序列
    tlabels = np.array([true_state_from_future(df1, i, 30, atr) for i in range(n)])
    # 切换点 (事后标签变化)
    sw = [i for i in range(1, n) if tlabels[i] is not None and tlabels[i-1] is not None and tlabels[i] != tlabels[i-1]]
    sw_set = set(sw)
    for tw in results:
        lag_acc = {"near_ok": 0, "near_n": 0, "far_ok": 0, "far_n": 0}
        for i in range(150, n - 31):
            tl = tlabels[i]
            if tl is None:
                continue
            ms = analyze_market_state(df1.iloc[i - tw - 70:i + 1].reset_index(drop=True), window=tw)
            st = ms.get("state", "")
            if st == "trend_up":
                p = "trend_up"
            elif st == "trend_down":
                p = "trend_down"
            elif st == "range":
                p = "range"
            else:
                p = "transition"
            # 判定-标签一致 (range 与 transition 互通)
            if tl in ("range", "transition"):
                ok = p in ("range", "transition")
            else:
                ok = p == tl
            near = any(abs(i - s) < 10 for s in sw)
            if near:
                lag_acc["near_n"] += 1
                lag_acc["near_ok"] += 1 if ok else 0
            else:
                lag_acc["far_n"] += 1
                lag_acc["far_ok"] += 1 if ok else 0
        results[tw]["near_ok"] = lag_acc["near_ok"]
        results[tw]["near_n"] = lag_acc["near_n"]
        results[tw]["far_ok"] = lag_acc["far_ok"]
        results[tw]["far_n"] = lag_acc["far_n"]

print(f"{'窗口':>5} {'切换点±10根符合率':>18} {'稳态符合率':>12} {'n(切换附近)':>12}")
for tw in sorted(results):
    r = results[tw]
    if r["near_n"] == 0:
        continue
    near = r["near_ok"] / r["near_n"] * 100
    far = r["far_ok"] / r["far_n"] * 100
    print(f"{tw:>5} {near:>15.1f}% {far:>10.1f}% {r['near_n']:>12}")
