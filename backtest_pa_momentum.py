#!/usr/bin/env python3
"""价格行为 = 移动开始信号? — 对齐收益/速度/MFU 测试 (416天)

核心指标 (入场K收盘价入场, 对齐方向 = 入场K方向):
  aligned_ret(N) = 对齐方向 N bar 后收益 (阳K=+ret, 阴K=-ret), 单位 ATR
  t_to_atr      = 对齐方向首次达到 +1ATR 的 bar 数 (中位数, 24bar 上限)
  mfu_24h       = 24bar 内最大有利波动 (对齐方向, 单位 ATR)

如果"强K=移动开始": 顺K强度↑ → aligned_ret↑, t_to_atr↓, mfu↑
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

groups = {
    "顺K强(实体>=60%)": [],
    "顺K中(30-60%)": [],
    "顺K弱(<30%)": [],
    "逆K": [],
}
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
    for i in range(100, n1 - 25):
        rng = h1[i] - l1[i]
        if rng <= 0 or atr1[i] <= 0:
            continue
        body = abs(c1[i] - o1[i])
        bull = c1[i] > o1[i]
        bp = body / rng
        sign = 1.0 if bull else -1.0
        entry = c1[i]; a = atr1[i]
        # 后续 6/12/24 bar 对齐收益
        r6 = r12 = r24 = t_atr = np.nan; mfu = np.nan
        if i + 24 < n1:
            for N, name in [(6, "r6"), (12, "r12"), (24, "r24")]:
                ret = (c1[i+N] - entry) / a * sign
                if name == "r6": r6 = ret
                elif name == "r12": r12 = ret
                else: r24 = ret
            max_fav = 0.0; t_atr = np.nan
            for k in range(1, 25):
                if bull:
                    fav = (h1[i+k] - entry) / a * sign
                    if h1[i+k] >= entry + a and np.isnan(t_atr):
                        t_atr = k
                else:
                    fav = (entry - l1[i+k]) / a * sign
                    if l1[i+k] <= entry - a and np.isnan(t_atr):
                        t_atr = k
                max_fav = max(max_fav, fav)
            mfu = max_fav
        row = (r6, r12, r24, t_atr, mfu)
        if bp >= 0.6:
            groups["顺K强(实体>=60%)"].append(row)
        elif bp >= 0.3:
            groups["顺K中(30-60%)"].append(row)
        elif bp > 0:
            groups["顺K弱(<30%)"].append(row)
        else:
            groups["逆K"].append(row)

def stat(rows, col):
    vals = [r[col] for r in rows if not (isinstance(r[col], float) and np.isnan(r[col]))]
    if not vals:
        return None
    return np.mean(vals), np.median(vals), len(vals)

print(f"{'组':<20} {'n':>9} | {'r6均值':>8} {'r12均值':>9} {'r24均值':>9} | {'t_atr中位':>10} {'mfu24均值':>10}", flush=True)
for name, rows in groups.items():
    n = len(rows)
    if n < 100:
        print(f"{name:<20} {n:>9}  样本不足", flush=True)
        continue
    r6 = stat(rows, 0); r12 = stat(rows, 1); r24 = stat(rows, 2)
    t = stat(rows, 3); mfu = stat(rows, 4)
    print(f"{name:<20} {n:>9} | {r6[0]:>8.4f} {r12[0]:>9.4f} {r24[0]:>9.4f} | {t[1]:>10.1f} {mfu[0]:>10.4f}", flush=True)

# 差异检验: 顺K强 vs 顺K弱
from scipy import stats as st
a = [r[2] for r in groups["顺K强(实体>=60%)"]]
b = [r[2] for r in groups["顺K弱(<30%)"]]
if a and b:
    t_stat, p = st.ttest_ind(a, b, equal_var=False)
    print(f"\nr24 顺K强 vs 顺K弱: 均值差={np.mean(a)-np.mean(b):+.4f} ATR, t={t_stat:+.1f}, p={p:.3g} (N={len(a)}/{len(b)})", flush=True)
a = [r[2] for r in groups["顺K强(实体>=60%)"]]
c = [r[2] for r in groups["逆K"]]
if a and c:
    t_stat, p = st.ttest_ind(a, c, equal_var=False)
    print(f"r24 顺K强 vs 逆K: 均值差={np.mean(a)-np.mean(c):+.4f} ATR, t={t_stat:+.1f}, p={p:.3g} (N={len(a)}/{len(c)})", flush=True)
