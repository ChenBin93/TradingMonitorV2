#!/usr/bin/env python3
"""A1.5 市场记忆结构 — 当前 K 线受多大窗口影响 (2026-08-03, 纯描述性, 无未来函数)

目标: 不预测方向, 刻画市场的时间记忆结构 (为 A2 状态特征 / B 关键价位提供尺度依据)
  ① ACF 衰减曲线族: |r|/r²/log(ATR)/范围/实体/成交量 自相关 → 影响窗口 + 半衰期
  ② PACF (Levinson-Durbin 手写): 排除中间传导后"直接影响"的最远窗口
  ③ DFA-Hurst (手写): 趋势记忆 vs 均值回归 vs 无记忆
  ④ 小波 DWT 能量谱 (pywt): 波动由哪些时间尺度驱动
  ⑤ Granger 因果 (statsmodels): 量能→波动 / 范围→波动 (过去→现在, 单向)
  ⑥ 波动状态持续性: ATR 分位三态转移矩阵 + 平均持续时长

无未来函数: 全部为描述性统计 (数据集性质描述, 不产生交易决策)。
若后续把记忆度量用作特征, 必须滚动估计 (见研究规范)。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from market_phase import _atr_series
from research.data_loader import load_candles, verify

MAX_LAG = 168
SIGNIF = 1.96  # z 分位


def _load(timeframes):
    data = load_candles(timeframes=timeframes)
    out = {}
    for sym, tfs in data.items():
        for tf in timeframes:
            df = tfs.get(tf)
            if df is None:
                continue
            if verify(df, sym, tf):
                print(f"跳过 {sym} {tf}")
                continue
            out.setdefault(tf, []).append(df)
    return out


# ────────────────────────────────────────────────
# 序列构造
# ────────────────────────────────────────────────
def build_series(df, warmup=60):
    """序列构造; warmup 丢弃指标未收敛段 (ATR 前 13 根=0 → log 尖刺)"""
    c = df["close"].values
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    v = df["volume"].values
    r = np.diff(c) / c[:-1]
    atr = _atr_series(df)
    s = {
        "|r|": np.abs(r),
        "r2": r ** 2,
        "log(ATR)": np.log(np.maximum(atr, 1e-12)),
        "range": (h - l)[1:] / c[:-1],
        "body": np.abs(c - o)[1:] / c[:-1],
        "volume": v[1:],
        "log vol": np.log(np.maximum(v[1:], 1e-12)),
    }
    return {k: x[warmup:] for k, x in s.items()}


# ────────────────────────────────────────────────
# ① ACF
# ────────────────────────────────────────────────
def acf_series(x, max_lag):
    x = x - x.mean()
    var = np.mean(x * x)
    if var <= 0:
        return np.full(max_lag, np.nan)
    x = x / np.sqrt(var)
    return np.array([np.mean(x[:-k] * x[k:]) for k in range(1, max_lag + 1)])


def half_life(acf, unit_hours):
    below = np.where(acf < 0.5)[0]
    if len(below):
        lag = int(below[0]) + 1
        return f"{lag}根({lag * unit_hours}h)"
    return f">{len(acf)}根(>{len(acf) * unit_hours}h)"


# ────────────────────────────────────────────────
# ② PACF (Levinson-Durbin)
# ────────────────────────────────────────────────
def levinson_pacf(x, max_lag):
    x = x - x.mean()
    n = len(x)
    r = np.array([np.mean(x * x)] +
                 [np.mean(x[:-k] * x[k:]) for k in range(1, max_lag + 1)])
    phi = np.zeros(max_lag + 1)
    pacf = np.zeros(max_lag)
    v = r[0]
    for k in range(1, max_lag + 1):
        num = r[k] - sum(phi[j] * r[k - j] for j in range(1, k))
        phi_k = num / v if v > 0 else 0.0
        pacf[k - 1] = phi_k
        if k > 1:
            new_phi = np.zeros(k + 1)
            new_phi[1:k] = phi[1:k] - phi_k * phi[1:k][::-1]
            new_phi[k] = phi_k
            phi = new_phi
        else:
            phi[1] = phi_k
        v *= (1 - phi_k ** 2)
    return pacf


# ────────────────────────────────────────────────
# ③ DFA-Hurst
# ────────────────────────────────────────────────
def dfa_hurst(x, n_scales=18):
    y = np.cumsum(x - x.mean())
    n = len(y)
    scales = np.geomspace(16, max(32, n // 16), n_scales).astype(int)
    scales = np.unique(scales)
    fs, ss = [], []
    for s in scales:
        n_blocks = n // s
        if n_blocks < 10:
            continue
        t = np.arange(s)
        f2 = 0.0
        for b in range(n_blocks):
            seg = y[b * s:(b + 1) * s]
            seg_det = seg - np.polyval(np.polyfit(t, seg, 1), t)
            f2 += np.mean(seg_det ** 2)
        fs.append(np.sqrt(f2 / n_blocks))
        ss.append(s)
    if len(ss) < 5:
        return float("nan")
    return np.polyfit(np.log(ss), np.log(fs), 1)[0]


# ────────────────────────────────────────────────
# ⑥ 波动状态持续性
# ────────────────────────────────────────────────
def atr_state_persistence(atr, qs=(1 / 3, 2 / 3)):
    q1, q2 = np.quantile(atr, qs)
    s = np.digitize(atr, [q1, q2])  # 0/1/2
    n = len(s)
    trans = np.zeros((3, 3))
    for i in range(n - 1):
        trans[s[i], s[i + 1]] += 1
    row_sum = trans.sum(axis=1, keepdims=True)
    trans = trans / np.maximum(row_sum, 1)
    # 平均持续时长: 状态 run 的平均长度
    runs = []
    cur, cnt = s[0], 1
    for v in s[1:]:
        if v == cur:
            cnt += 1
        else:
            runs.append((cur, cnt))
            cur, cnt = v, 1
    runs.append((cur, cnt))
    dur = {}
    for st in range(3):
        lens = [c for s_, c in runs if s_ == st]
        dur[st] = np.mean(lens) if lens else np.nan
    return trans, dur, np.bincount(s, minlength=3) / n


if __name__ == "__main__":
    dfs_by_tf = _load(timeframes=("1h", "4h"))

    print("═══ A1.5 市场记忆结构 (描述性, 无未来函数) ═══\n")

    # ── ① ACF 族 ──
    print("① 影响窗口 (两个口径):")
    print("   显著窗口 = 自相关仍显著 (|ACF|>1.96/√n) 的最远 lag — 弱相关长尾也计入")
    print("   半衰期 = ACF 首次 <0.5 的 lag (仅 ACF@1>0.5 的序列有意义)\n")
    n1 = len(dfs_by_tf["1h"][0])
    n4 = len(dfs_by_tf["4h"][0])
    header = f"{'序列':>10} | {'1h 显著窗口':>12} | {'4h 显著窗口':>12} | {'1h 半衰期':>16} | {'4h 半衰期':>18}"
    print(header)
    print("-" * len(header))
    acf_cache = {}
    for seq_name in ["|r|", "r2", "log(ATR)", "range", "body", "volume", "log vol"]:
        row = []
        for tf in ("1h", "4h"):
            acfs = [acf_series(build_series(df)[seq_name], MAX_LAG) for df in dfs_by_tf[tf]]
            acf_mean = np.nanmean(np.stack(acfs), axis=0)
            acf_cache[(tf, seq_name)] = acf_mean
            n = n1 if tf == "1h" else n4
            thr = SIGNIF / np.sqrt(n)
            above = np.where(np.abs(acf_mean) > thr)[0]
            sig = int(above.max()) + 1 if len(above) else 0
            hl = half_life(acf_mean, 1 if tf == "1h" else 4) if acf_mean[0] > 0.5 else "—(<0.5)"
            row.append((sig, hl))
        print(f"{seq_name:>10} | {row[0][0]:>12} | {row[1][0]:>12} | {row[0][1]:>16} | {row[1][1]:>18}")
    print()
    # 时间对齐: log(ATR) 半衰期小时数 (1h vs 4h), 差异大 = 按根数/多尺度
    a1 = acf_cache[("1h", "log(ATR)")]
    a4 = acf_cache[("4h", "log(ATR)")]
    h1 = np.where(a1 < 0.5)[0]
    h4 = np.where(a4 < 0.5)[0]
    t1 = (int(h1[0]) + 1) if len(h1) else None
    t4 = (int(h4[0]) + 1) * 4 if len(h4) else None
    note = "—(168根内均未跌破0.5, 无法比较)" if t1 is None and t4 is None else \
        f"1h半衰期>{t1 or 168}h vs 4h半衰期>{t4 or 672}h"
    print(f"   时间对齐检查 (log(ATR) 半衰期): {note} — 若按时间对齐应接近, 差异大=多尺度")
    print()

    # 衰减曲线采样
    print("   ACF 采样 (lag: 1/4/16/64/128):")
    for seq_name in ["|r|", "log(ATR)", "range"]:
        parts = []
        for tf in ("1h", "4h"):
            a = acf_cache[(tf, seq_name)]
            vals = " ".join(f"{a[k - 1]:.3f}" for k in [1, 4, 16, 64, 128])
            parts.append(f"{tf}[{vals}]")
        print(f"   {seq_name:>10}: {'  '.join(parts)}")
    print()

    # ── ② PACF ──
    print("② PACF 直接影响窗口 (Levinson-Durbin; 显著阈值 ±1.96/√n)")
    print("   '直接影响' = 排除中间 lag 传导后仍有自相关; 看显著 lag 的最远范围\n")
    for seq_name in ["|r|", "log(ATR)"]:
        parts = []
        for tf in ("1h", "4h"):
            n = len(dfs_by_tf[tf][0])
            thr = SIGNIF / np.sqrt(n)
            sigs = []
            for df in dfs_by_tf[tf]:
                pacf = levinson_pacf(build_series(df)[seq_name], 48)
                above = np.where(np.abs(pacf) > thr)[0]
                sigs.append(int(above.max()) + 1 if len(above) else 0)
            parts.append(f"{tf}: 显著最远lag中位 {int(np.median(sigs))} (标的区间 {min(sigs)}-{max(sigs)})")
        print(f"   {seq_name:>10}: {' | '.join(parts)}")
    print()

    # ── ③ DFA-Hurst ──
    print("③ DFA-Hurst (H>0.5 趋势记忆 / <0.5 均值回归 / ≈0.5 无记忆)")
    print("   收益 DFA = log价格累计 (随机游走应≈0.5)")
    print("   波动 DFA = |r| 累计 (波动能量长记忆应 >0.5; 不用 log(ATR) 避免 ewm 平滑伪影)\n")
    for tf, dfs in dfs_by_tf.items():
        hs_ret, hs_vol = [], []
        for df in dfs:
            s = build_series(df)
            r = s["|r|"] * np.sign(np.diff(df["close"].values)[60:])  # 带符号收益 (warmup 对齐)
            h_ret = dfa_hurst(r)
            h_vol = dfa_hurst(s["|r|"])
            if not np.isnan(h_ret):
                hs_ret.append(h_ret)
            if not np.isnan(h_vol):
                hs_vol.append(h_vol)
        print(f"   {tf} 收益: H = {np.mean(hs_ret):.3f} (标的 {min(hs_ret):.3f}-{max(hs_ret):.3f})")
        print(f"   {tf} 波动(|r|累计): H = {np.mean(hs_vol):.3f} (标的 {min(hs_vol):.3f}-{max(hs_vol):.3f})")
    print()

    # ── ④ 小波能量谱 ──
    import pywt
    print("④ 小波 DWT 能量谱 (db4, level=5) — 波动由哪些时间尺度驱动")
    print("   1h: 细节层 2/4/8/16/32h + 近似层>32h; 4h: ×4 换算\n")
    for tf, dfs in dfs_by_tf.items():
        levels = 5
        energies = []
        for df in dfs:
            x = build_series(df)["log(ATR)"]
            coefs = pywt.wavedec(x, "db4", level=levels)
            e = np.array([np.sum(c ** 2) for c in coefs])
            energies.append(e / e.sum())
        em = np.mean(np.stack(energies), axis=0)
        # pywt.wavedec 返回 [cA5, cD5, cD4, cD3, cD2, cD1] — 近似层在前, 最高频在最后
        labels = ["近似>32h", "细节32h", "细节16h", "细节8h", "细节4h", "细节2h"] if tf == "1h" \
            else ["近似>128h", "细节128h", "细节64h", "细节32h", "细节16h", "细节8h"]
        parts = "  ".join(f"{lb}:{v:.1%}" for lb, v in zip(labels, em))
        print(f"   {tf}: {parts}")
    print()

    # ── ⑤ Granger 因果 ──
    from statsmodels.tsa.stattools import grangercausalitytests
    print("⑤ Granger 因果 (过去→现在, 单向; maxlag=12, p<0.01 视为显著)")
    print("   量能→波动 |r|, 范围→波动 |r|; 报告显著标的占比 (20 标的)\n")
    for tf, dfs in dfs_by_tf.items():
        for cause, effect in [("volume", "|r|"), ("range", "|r|")]:
            sig_lags = []
            n_sig = 0
            for df in dfs:
                s = build_series(df)
                X = np.column_stack([s[effect], s[cause]])
                X = X[~np.isnan(X).any(axis=1)]
                if len(X) < 500:
                    continue
                try:
                    res = grangercausalitytests(X, maxlag=12, verbose=False)
                    pmin = min(res[l][0]["ssr_ftest"][1] for l in res)
                    if pmin < 0.01:
                        n_sig += 1
                        best = min((l for l in res if res[l][0]["ssr_ftest"][1] < 0.01),
                                   key=lambda l: res[l][0]["ssr_ftest"][1])
                        sig_lags.append(best)
                except Exception:
                    pass
            med = int(np.median(sig_lags)) if sig_lags else "-"
            print(f"   {tf} {cause}→波动: {n_sig}/{len(dfs)} 标的显著 (最佳lag中位 {med})")
    print()

    # ── ⑥ 状态持续性 ──
    print("⑥ 波动状态持续性 (ATR 三分位: 低/中/高)")
    print("   转移矩阵 = 状态在下一根保持的概率; 平均持续时长 (根 / 时间)\n")
    for tf, dfs in dfs_by_tf.items():
        unit = 1 if tf == "1h" else 4
        tms, durs, occs = [], [], []
        for df in dfs:
            atr = _atr_series(df)
            trans, dur, occ = atr_state_persistence(atr)
            tms.append(trans)
            durs.append(dur)
            occs.append(occ)
        tm = np.mean(np.stack(tms), axis=0)
        print(f"   {tf} 转移矩阵 (行=当前, 列=下一根, 均值):")
        for i, name in enumerate(["低波动", "中波动", "高波动"]):
            row = " ".join(f"{tm[i, j]:.2f}" for j in range(3))
            keep = tm[i, i]
            mean_dur = np.mean([d[i] for d in durs])
            occ = np.mean([o[i] for o in occs])
            print(f"     {name}: [{row}] 自保持 {keep:.2f} | 平均持续 {mean_dur:.0f}根({mean_dur * unit}h) | 占比 {occ:.0%}")
        print()

    print("═══ 完成 ═══")
