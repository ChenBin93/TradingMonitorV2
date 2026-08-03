#!/usr/bin/env python3
"""快照 vs 序列 — 研判当前行情用"当前K指标值"还是"窗口内K的指标轨迹"?

问题1: 指标窗口多大 (MA/ADX/动量 计算窗口)
问题2: 分析窗口多大 + 用快照还是序列特征

A. 合成数据 (ground truth): 快照特征 vs 序列特征 vs 组合 → 3分类准确率
B. 真实数据 (插曲环境): 快照 vs 序列特征对 1:1 的区分度
"""
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

sys.path.insert(0, ".")
from market_phase import _atr_series, _adx_series

# ═══════════════ A. 合成数据 ═══════════════
def make_regime_series(segments, seed=0):
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

segments = [(300, "trend_up", 1.0), (300, "range", 1.0),
            (300, "trend_down", 1.0), (300, "range", 1.0),
            (300, "trend_up", 1.0)]
df = make_regime_series(segments)
TRUE = []
for n, kind, vol in segments:
    TRUE.extend([kind] * n)
TRUE = np.array(TRUE)
c = df["close"].values
ret = np.diff(c) / c[:-1] * 100
n = len(ret)

# 标签: 0=down, 1=range, 2=up
def label(i):
    t = TRUE[i + 1]
    return 2 if t == "trend_up" else 0 if t == "trend_down" else 1

W = 20  # 分析窗口
ma20 = pd.Series(c).rolling(20).mean().values
adx = _adx_series(df)

rows = []
for i in range(120, n - 1, 2):
    # ── 快照特征 (当前K指标值) ──
    snap = [
        ret[i] / 0.2,                      # 当前收益 (归一)
        abs(ret[i]) / 0.2,                 # 当前波动
        (c[i+1] - ma20[i+1]) / 0.3,        # 偏离 MA20
        adx[i+1] / 30,                     # 当前 ADX
    ]
    # ── 序列特征 (窗口内所有K的指标轨迹) ──
    win_ret = ret[i-W+1:i+1]
    seq = [
        np.sum(win_ret) / (0.2 * np.sqrt(W)),   # 窗口累计动量
        np.sum(np.abs(win_ret)) / (0.2 * W),    # 窗口平均波动
        (ma20[i+1] - ma20[i+1-W]) / 0.3,        # MA20 窗口斜率
        adx[i+1] - adx[i+1-W],                  # ADX 窗口变化
        np.sum(win_ret > 0) / W,                # 同向占比
        (c[i+1] - np.min(c[i+1-W:i+2])) / max(np.ptp(c[i+1-W:i+2]), 1e-9),  # 窗口内位置
    ]
    rows.append((snap, seq, label(i)))
print(f"合成样本: {len(rows)}")

X_snap = np.array([r[0] for r in rows])
X_seq = np.array([r[1] for r in rows])
X_all = np.hstack([X_snap, X_seq])
y = np.array([r[2] for r in rows])

for name, X in [("快照(当前K)", X_snap), ("序列(窗口轨迹)", X_seq), ("快照+序列", X_all)]:
    clf = LogisticRegression(max_iter=500)
    scores = cross_val_score(clf, X, y, cv=5)
    print(f"  {name:<14} 3分类准确率: {scores.mean()*100:.1f}% ± {scores.std()*100:.1f}")

# ═══════════════ B. 真实数据 (插曲环境, 1H) ═══════════════
print("\n═══ B. 真实数据 (20标的 1H, 插曲环境): 快照 vs 序列 区分度 ═══")
from backtest_engine import load_all
data = load_all(timeframes=["1h", "4h"])
syms = list(data.keys())

daily_states = {}
for sym in syms:
    df4 = data[sym].get("4h")
    if df4 is None or len(df4) < 100:
        continue
    daily = df4.resample("1D").last()
    ma20_d = daily["close"].rolling(20).mean()
    d = {}
    for ts, row in daily.iterrows():
        if pd.isna(ma20_d[ts]):
            continue
        d[ts.date()] = "bull" if row["close"] > ma20_d[ts] else "bear"
    daily_states[sym] = d

samples = []  # (snap_feat, seq_feat, out)
for sym in syms:
    df1 = data[sym].get("1h")
    df4 = data[sym].get("4h")
    if df1 is None or df4 is None or len(df1) < 300:
        continue
    c1 = df1["close"].values
    hi = df1["high"].values
    lo = df1["low"].values
    n1 = len(df1)
    atr = _atr_series(df1)
    adx = _adx_series(df1)
    ma20_1 = pd.Series(c1).rolling(20).mean().values
    idx1 = df1.index.values.astype("datetime64[ns]")
    idx4 = df4.index.values.astype("datetime64[ns]")
    c4 = df4["close"].values
    ma20_4 = pd.Series(c4).rolling(20).mean().values
    ds = daily_states.get(sym, {})
    ret1 = np.diff(c1) / c1[:-1] * 100

    for i in range(200, n1 - 25, 4):
        ts = pd.Timestamp(idx1[i])
        sd = ds.get(ts.date())
        if sd not in ("bull", "bear"):
            continue
        t4 = int(np.searchsorted(idx4, idx1[i] - np.timedelta64(240, "m"), side="right")) - 1
        if t4 < 20 or np.isnan(ma20_4[t4]):
            continue
        s4 = "bull" if c4[t4] > ma20_4[t4] else "bear"
        if s4 == sd:
            continue  # 只测插曲
        a = atr[i]
        if a <= 0 or np.isnan(a) or np.isnan(adx[i]):
            continue
        # 快照: 当前指标
        snap = [
            (c1[i] - ma20_1[i]) / a,     # 1H 偏离
            adx[i] / 40,                 # ADX
            ret1[i-1] / (a / c1[i] * 100),  # 当前收益 (ATR归一)
        ]
        # 序列: 20根轨迹
        W2 = 20
        if i < W2:
            continue
        seq = [
            (ma20_1[i] - ma20_1[i-W2]) / a,       # MA20 斜率
            adx[i] - adx[i-W2],                    # ADX 变化
            np.mean(ret1[i-W2:i]) / (a / c1[i] * 100),  # 窗口动量
            np.mean(np.abs(ret1[i-W2:i])) / (a / c1[i] * 100),  # 窗口波动
            (c1[i] - np.min(c1[i-W2:i+1])) / max(np.ptp(c1[i-W2:i+1]), 1e-9),  # 位置
        ]
        long_side = sd == "bull"
        entry = c1[i]
        hit = 0
        for k in range(1, 25):
            if long_side:
                if hi[i+k] >= entry + a:
                    hit = 1; break
                if lo[i+k] <= entry - a:
                    hit = -1; break
            else:
                if lo[i+k] <= entry - a:
                    hit = 1; break
                if hi[i+k] >= entry + a:
                    hit = -1; break
        if hit != 0:
            samples.append((snap, seq, 1 if hit == 1 else 0))

print(f"插曲样本: {len(samples)}")
base = np.mean([s[2] for s in samples]) * 100
print(f"基线胜率: {base:.1f}%")

# 特征分组区分度: 按各特征集的高分/低分分组, 对比胜率差
def group_wr(feat_idx, hi=True):
    vals = [s[feat_idx] for s in samples]
    m = np.mean(vals)
    if hi:
        sub = [s for s in samples if s[feat_idx] >= m]
    else:
        sub = [s for s in samples if s[feat_idx] < m]
    return np.mean([s[2] for s in sub]) * 100 if sub else 0

# 快照综合分: 偏离+ADX 同向强度
snap_scores = []
for s in samples:
    # 做多时: 偏离>0 和 ADX>0.6 有利
    snap_scores.append(s[0][0] * 0.5 + (s[0][1] - 0.6) * 1.0)
seq_scores = []
for s in samples:
    seq_scores.append(s[1][1] * 0.5 + s[1][0] * 0.5)  # ADX变化 + MA斜率

print(f"\n快照综合分 (偏离+ADX): 高分组 {np.mean([s[2] for s, v in zip(samples, snap_scores) if v >= np.median(snap_scores)])*100:.1f}% vs 低分组 {np.mean([s[2] for s, v in zip(samples, snap_scores) if v < np.median(snap_scores)])*100:.1f}%")
print(f"序列综合分 (ADX变化+MA斜率): 高分组 {np.mean([s[2] for s, v in zip(samples, seq_scores) if v >= np.median(seq_scores)])*100:.1f}% vs 低分组 {np.mean([s[2] for s, v in zip(samples, seq_scores) if v < np.median(seq_scores)])*100:.1f}%")
