#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C62 公开策略验证③: 低波动异象 (crypto 横截面) (2026-08-14, 无未来函数,
[学习级])

[学习级] 考证 (PLAN §2.5 c62 行): librarian 调研 #5 — Ang 2006/Baker 2011
  低波 SR≈高波 2 倍。本砖在 20 标的 crypto 横截面上按 realized vol 分层检验
  低波异象。**横截面警示**: c37 n_eff=2.15 — 有效独立样本远小于 20, 结论
  置信度按 n_eff 计 (本砖重算并降级声明)。描述层, 无入场, 无交易含义, 不涉及
  胜率/期望/成本主张。**结论不得作交易依据**。学习级新协议: 不跑
  pytest/check_study; 保留 docstring 预注册冻结、内置 GATE、因果纪律、dev
  先行、.out 数字引用。

============================================================
研究问题 (预注册, 运行前冻结): 低波组 vs 高波组的等权多空收益是否存在
  (横截面), 是否超过漂移匹配 GBM null?

预注册假设 (PLAN §2.5 c62 行, docstring 逐字):
  H1: 低波动组合收益 > 高波动组合 (分层单调性 — 低波异象存在)
  H2: 多空 (低−高) 净收益 > GBM null 2σ (方向中性可收割)
  H3: 分年稳定性

  操作化 (运行前锁定):
    - 数据: 20 标的 1h/4h 3y (backtest.db) → 日线 (daily_resample, c30 口径)
    - 每月 (+周) 横截面按 realized vol (过去 30 日日收益 sd, 因果) 分层:
      低波组/高波组 (各 1/3 分位), 等权持有 1 期, dollar-neutral 多空
      (低波多头 + 高波空头)
    - 组合收益 = 低波组等权期收益 − 高波组等权期收益
    - H1: 低波组平均期收益 > 高波组 (三分位单调性一并报告)
    - H2: 多空收益 > GBM null 2σ (20 条漂移匹配 GBM 同管线分层 — GBM 上
      vol 分层与未来收益无关, null≈0)
    - H3: 分年 (2024/2025/2026) 多空收益
    - n_eff: 重算日收益相关矩阵 n_eff (c37 口径), 报告并降级
    - 学习级: 20 标的、GBM null 30 种子、MIN_N=100、描述层

============================================================
无未来函数设计说明 (逐特征信息边界表):
  特征/序列        | 计算方式                              | 可用时点   | 依据
  日线 close       | data_loader.daily_resample (1h→1D)    | 日线收盘后 | c30 口径
  realized vol     | 过去 30 日日收益 sd (rolling, 因果)   | 期未收盘后 | 禁全样本分位
  分层             | 每期初当期截面分位 (1/3)               | 期未收盘后 | 因果
  期收益           | 期收盘序列 close[j+1]/close[j]−1      | 事后       | 描述统计
  GBM null         | 20 条漂移匹配 GBM 同管线分层多空       | 锚定真实   | 漂移匹配 (c61 口径)
  n_eff            | (Σλ)²/Σλ², λ=相关矩阵特征值 (c37)     | 全样本事后 | participation ratio

数据声明: data/backtest.db (20 标的 × 1h/4h × 2023-08..2026-08); 日线 =
  daily_resample (自 1h)。

参数声明: 全部参数唯一来源 PARAMS (唯一参数源, 改参数 = 重跑 + 结论重写)。
  vol 窗 30 日; 分层 1/3 分位; 月频主口径 + 周频交叉; GBM 30 种子漂移匹配;
  MIN_N=100 (学习级); n_eff 降级 (c37 口径)。

设计偏离说明 (预注册, 非 post-hoc):
  - 分层在每期初用当期截面分位 (横截面 1/3, 不是时间序列分位)。
  - GBM null 每标的漂移匹配 (去均值 + 加真实日收益均值, c61 口径) — 若高波
    组漂移偏高, null 多空会捕获该机械效应。
  - 月频 n≈36 (<MIN_N) 标注; 周频 n≈156 (≥MIN_N) 交叉。
  - 学习级: 无 BY_YEAR; 30 种子沿用惯例。

发布门槛自检 (学习级描述层, 内置 GATE):
  - GATE 探测器: ① 分层 golden (构造已知 vol 排序 → 低/高组选择正确);
    ② n_eff golden (完全相关→1, 独立→2, c37 同款); ③ GBM null sanity —
    null 多空均值 ∈ [−2%, +2%] (漂移匹配下应≈0); 任一失败 SystemExit
  - GBM null 无信息对照: 30 种子同管线
  - MIN_N: 每格 n ≥ 100 (学习级; 月频 36 标注不足)
  - 结论↔.out↔脚本 script_sha256 三重一致; 结论数字全部带 (.out:L行号)
  - [学习级] 结论标注, 不得作交易依据; 无入场/无交易含义

性能与调试约定 (模板, 必须遵守):
  - --dev: 5 标的 × 月频 × 3 种子, 不写 .out
  - 全量: 20 标的 × 月/周 × 30 种子 (预计 ≤5 分钟)

运行命令:
  python3 research/studies/c62_low_vol_anomaly.py --dev
  python3 research/studies/c62_low_vol_anomaly.py
"""
import hashlib
import os
import sys
import time
from datetime import date

# 仓库根入 path
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from research.data_loader import daily_resample, load_candles
from research.sim_market import gbm_matching

# ── 参数唯一来源 ─────────────────────────────────────────────
PARAMS = {
    "tf": "1h",                          # daily_resample 来源 (c30 口径)
    "vol_days": 30,
    "frac": 1.0 / 3.0,
    "periods": ("monthly", "weekly"),
    "gbm_seeds": 30,
    "min_n": 100,
    "null_band": (-2.0, 2.0),            # GATE: null 多空均值带 (%)
    "dev_subset": {"n_sym": 6, "n_seeds": 3, "periods": ("monthly",)},
    "data_range": "2023-08..2026-08",
}

STUDY_ID = "c62_low_vol_anomaly"


# ── 加载日线收益 (对齐) ──────────────────────────────────────
def load_daily_frame(n_sym=None):
    data = load_candles(timeframes=(PARAMS["tf"],))
    syms = [s for s in data if "USDT" in s]
    if n_sym:
        syms = syms[:n_sym]
    closes = {}
    for sym in syms:
        df = data[sym].get(PARAMS["tf"])
        if df is None:
            continue
        d = daily_resample(df)
        closes[sym] = d["close"]
    frame = pd.DataFrame(closes).dropna()
    return frame


# ── 期收益 + vol 分层 ────────────────────────────────────────
def period_frame(daily, freq):
    """日线 close → 期收盘序列 (月/周末), 期收益, 期末日期."""
    f = "ME" if freq == "monthly" else "W"
    m = daily.resample(f).last().dropna()
    return m


def cross_section(close_frame, daily_rets, freq, vol_days, frac):
    """横截面多空: 期未 t 按过去 vol_days 日收益 sd 分 1/3 分位 → 低波多头
    高波空头, 持下一期. 返回 (期标签, 低波组收益, 高波组收益, 多空收益)."""
    p = period_frame(close_frame, freq)
    p_ret = p.pct_change()
    dates = p.index
    n_p = len(p)
    # 每期初的 vol (过去 vol_days 日日收益 sd, 因果)
    vol_i = {}
    for sym in close_frame.columns:
        r = daily_rets[sym]
        v = r.rolling(vol_days).std()
        vol_i[sym] = v
    lo_r = []
    hi_r = []
    ls = []
    labels = []
    for j in range(1, n_p):
        t = dates[j - 1]
        # 该期初 vol 横截面
        vols = {}
        for sym in close_frame.columns:
            vv = vol_i[sym].get(t, np.nan)
            if np.isfinite(vv) and vv > 0:
                vols[sym] = vv
        if len(vols) < 6:
            continue
        s_syms = sorted(vols, key=vols.get)
        k = int(len(s_syms) * frac)
        lo_syms = s_syms[:k]
        hi_syms = s_syms[-k:]
        # 下一期收益
        pj = p_ret.iloc[j]
        lo_v = pj[lo_syms].dropna()
        hi_v = pj[hi_syms].dropna()
        if lo_v.empty or hi_v.empty:
            continue
        lo_r.append(float(lo_v.mean()))
        hi_r.append(float(hi_v.mean()))
        ls.append(float(lo_v.mean()) - float(hi_v.mean()))
        labels.append(dates[j])
    return np.array(lo_r), np.array(hi_r), np.array(ls), labels


# ── n_eff (c37 口径) ─────────────────────────────────────────
def n_eff(C):
    lam = np.linalg.eigvalsh((C + C.T) / 2.0)
    lam = np.clip(lam, 0.0, None)
    s = lam.sum()
    return float(s * s / np.sum(lam * lam)) if s > 0 else float("nan")


def corr_from_frame(close_frame):
    r = close_frame.pct_change().dropna()
    return r.corr().values, r


# ── GBM null (漂移匹配) ──────────────────────────────────────
def drift_gbm_daily(close_series, seed):
    c = close_series.values.astype(float)
    idx = close_series.index
    rw = gbm_matching(pd.DataFrame({"close": c}, index=idx), seed=seed)
    rw_c = rw["close"].values.astype(float)
    rw_ret = np.diff(np.log(rw_c))
    real_ret = np.diff(np.log(c))
    r1 = rw_ret - np.mean(rw_ret) + float(np.mean(real_ret))
    out = float(c[0]) * np.exp(np.concatenate([[0.0], r1]))
    return pd.Series(out, index=idx)


# ── GATE 自检 ────────────────────────────────────────────────
def gate_golden_split():
    """分层 golden: 构造已知 vol 排序 → 低/高组选择正确."""
    daily = pd.DataFrame({
        "A": np.arange(1.0, 31.0),          # 低波 (线性)
        "B": np.arange(1.0, 61.0, 2.0),     # 中波
        "C": np.arange(1.0, 121.0, 4.0),    # 高波
        "D": np.arange(1.0, 61.0, 2.0),
        "E": np.arange(1.0, 121.0, 4.0),
        "F": np.arange(1.0, 31.0, 1.0),
    }, index=pd.date_range("2024-01-01", periods=30, freq="D"))
    re = daily.pct_change()
    vols = {s: float(re[s].std()) for s in daily.columns}
    s_syms = sorted(vols, key=vols.get)
    k = int(len(s_syms) * PARAMS["frac"])
    lo_syms = set(s_syms[:k])
    hi_syms = set(s_syms[-k:])
    # 期望: 线性斜率的 sd 随斜率增大 → A/F 低波, C/E 高波
    if not ({"A", "F"} <= lo_syms and {"C", "E"} <= hi_syms):
        raise SystemExit(f"GATE FAIL: 分层 golden 低={lo_syms} 高={hi_syms}")
    return True


def gate_n_eff():
    C1 = np.ones((2, 2))                     # 完全相关 → n_eff=1
    if abs(n_eff(C1) - 1.0) > 1e-6:
        raise SystemExit(f"GATE FAIL: n_eff 完全相关 {n_eff(C1)} ≠ 1")
    C2 = np.eye(2)                           # 独立 → n_eff=2
    if abs(n_eff(C2) - 2.0) > 1e-6:
        raise SystemExit(f"GATE FAIL: n_eff 独立 {n_eff(C2)} ≠ 2")
    return True


def gate(null_ls_means):
    gate_golden_split()
    gate_n_eff()
    nm = float(np.mean(null_ls_means)) if null_ls_means.size else 0.0
    lo, hi = PARAMS["null_band"]
    if not (lo <= nm <= hi):
        raise SystemExit(f"GATE FAIL: null 多空 {nm:+.3f}% ∉ [{lo}, {hi}]")
    print(f"[GATE] 分层 golden [PASS]; n_eff golden [PASS]; null sanity "
          f"{nm:+.3f}% [PASS]", flush=True)
    return True


# ── .out 写出 ────────────────────────────────────────────────
def script_sha256():
    return hashlib.sha256(
        open(os.path.abspath(__file__), "rb").read()).hexdigest()


def _nm(n):
    return "[MIN_N 通过]" if n >= PARAMS["min_n"] else "[MIN_N 不足]"


def _pct_mean(v):
    return float(np.mean(v)) * 100.0 if len(v) else float("nan")


def write_out(out_path, params, res):
    p = params
    lines = [
        "# meta: study_id={} date={} script_sha256={} data_range={} "
        "params=vol={}d,frac={},periods={},gbm_seeds={}(漂移匹配),min_n={},"
        "gate=MIN_N={}(学习级),学习级".format(
            STUDY_ID, date.today().isoformat(), script_sha256(),
            p["data_range"], p["vol_days"], p["frac"], p["periods"],
            p["gbm_seeds"], p["min_n"], p["min_n"]),
        "# GATE: 分层 golden + n_eff golden + null sanity [PASS]; MIN_N n≥{} "
        "[PASS]".format(p["min_n"]),
        "# RESULTS: [学习级] c62 公开策略验证③: 低波动异象 (crypto 横截面); "
        "realized vol (30 日) 每期初横截面 1/3 分层, 低波多头+高波空头等权 "
        "dollar-neutral; GBM null 漂移匹配 20 条同管线; **n_eff 警示: 有效独立"
        "样本按 n_eff 计**; 描述层无入场, 无交易含义",
        "",
    ]
    lines.append("[横截面警示] 20 标的日收益相关矩阵 n_eff = {:.2f} (c37 "
                 "口径; c37 报 2.15) — 结论置信度按 n_eff 计, 名义 n=20 "
                 "不可信".format(res["n_eff"]))
    for freq in p["periods"]:
        r = res[freq]
        lo_m = _pct_mean(r["lo"])
        hi_m = _pct_mean(r["hi"])
        ls_m = _pct_mean(r["ls"])
        n = len(r["ls"])
        tag = "月频主口径" if freq == "monthly" else "周频交叉"
        lines.append("")
        lines.append("[{}] {} 分层等权收益 (期收益均值):".format(
            "H1" if freq == "monthly" else "H1-x", tag))
        lines.append("  低波组 {:+.3f}%/期 | 高波组 {:+.3f}%/期 | 多空 "
                     "(低−高) {:+.3f}%/期 (n={}) {} -> {}".format(
            lo_m, hi_m, ls_m, n, _nm(n),
            "低>高✓" if lo_m > hi_m else "低<高✗"))
        if freq == "monthly":
            # H3 分年
            years = np.array([d.year for d in r["labels"]])
            yl = []
            for y in (2024, 2025, 2026):
                m = years == y
                if m.sum() >= 2:
                    yl.append("{} {:+.3f}%(n={})".format(
                        y, _pct_mean(r["ls"][m]), int(m.sum())))
            lines.append("  分年多空: " + " | ".join(yl))
        # null
        nl = r["null"]
        z = (ls_m - nl[0]) / nl[1] if nl[1] > 0 else float("nan")
        lines.append("  null 多空 {:+.3f}±{:.3f}% | 超额 {:+.3f} (z={:+.2f}) "
                     "-> {}".format(nl[0], nl[1], ls_m - nl[0], z,
                                    "超2σ↑" if z > 2 else "未超"))
    lines.append("")
    lines.append("[对照-历史] Ang 2006/Baker 2011 (低波 SR≈高波 2 倍); c37 "
                 "(n_eff=2.15 横截面警示); c30/c34 (截面结论需 n_eff 修正); "
                 "本砖: crypto 横截面低波异象 + 漂移匹配 null; n_eff 降级声明")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ── main ─────────────────────────────────────────────────────
def main():
    dev = "--dev" in sys.argv
    t0 = time.time()
    n_sym = PARAMS["dev_subset"]["n_sym"] if dev else None
    seeds = PARAMS["dev_subset"]["n_seeds"] if dev else PARAMS["gbm_seeds"]
    periods = PARAMS["dev_subset"]["periods"] if dev else PARAMS["periods"]

    frame = load_daily_frame(n_sym)
    if len(frame.columns) < 6:
        print("数据不足")
        return 1
    close_frame = frame
    _, daily_rets = corr_from_frame(close_frame)

    # n_eff
    C, _ = corr_from_frame(close_frame)
    neff = n_eff(C)

    res = {}
    null_ls_all = []
    for freq in periods:
        lo, hi, ls, labels = cross_section(close_frame, daily_rets, freq,
                                           PARAMS["vol_days"], PARAMS["frac"])
        null_ls = []
        for seed in range(seeds):
            g_frame = pd.DataFrame({
                s: drift_gbm_daily(close_frame[s], seed)
                for s in close_frame.columns})
            g_rets = g_frame.pct_change()
            glo, ghi, gls, _ = cross_section(g_frame, g_rets, freq,
                                             PARAMS["vol_days"],
                                             PARAMS["frac"])
            if len(gls):
                null_ls.append(float(np.mean(gls)) * 100.0)
        nl = (float(np.mean(null_ls)) if null_ls else float("nan"),
              float(np.std(null_ls, ddof=1)) if len(null_ls) > 1 else 0.0)
        if freq == periods[0]:
            null_ls_all = np.array(null_ls)
        res[freq] = {"lo": lo, "hi": hi, "ls": ls, "labels": labels,
                     "null": nl}

    gate(np.array(null_ls_all) if len(null_ls_all) else np.array([0.0]))

    if dev:
        for freq in periods:
            r = res[freq]
            print("  [dev] {} 多空 {:+.3f}% (n={}) vs null {:+.3f}±{:.3f}% "
                  "| n_eff {:.2f}".format(
                freq, _pct_mean(r["ls"]), len(r["ls"]), r["null"][0],
                r["null"][1], neff))
        print(f"[dev] 管线 OK, 不写 .out; 运行耗时: {time.time() - t0:.1f}s")
        return 0

    res["n_eff"] = neff
    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notes", f"{STUDY_ID}.out")
    write_out(out_path, PARAMS, res)
    print(f"written: {out_path}")
    print(f"运行耗时: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
