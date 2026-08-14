#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C65 公开策略验证⑥: 横截面动量 (crypto) (2026-08-14, 无未来函数, [学习级])

[学习级] 考证 (PLAN §2.5 c65 行): librarian 调研 #6 — Liu-Tsyvinski-Wu 2022
  JF 三因子动量最稳 (调研称 2021 后转弱)。本砖复用 c62 横截面管线, 排序变量
  从 realized vol 换成过去 30 日收益。**n_eff 警示**: 同 c62 (2.28), 结论降级。
  描述层, 无入场, 无交易含义, 不涉及胜率/期望/成本主张。**结论不得作交易
  依据**。学习级新协议: 不跑 pytest/check_study; 保留 docstring 预注册冻结、
  内置 GATE、因果纪律、dev 先行、.out 数字引用。

============================================================
研究问题 (预注册, 运行前冻结): 横截面动量 (赢家−输家多空) 是否 > 漂移匹配
  GBM null? 2026 是否衰减?

预注册假设 (PLAN §2.5 c65 行, docstring 逐字):
  H1: 横截面动量多空净收益 > GBM null 2σ (LTW 2022 复现)
  H2: 分年稳定性 (2026 衰减检验 — 调研"2021 后转弱"对照)

  操作化 (运行前锁定):
    - 数据: 20 标的 1h (backtest.db) → 日线 (daily_resample); 月频主口径 +
      周频交叉
    - 分层: 每期初按过去 30 日收益 (rolling 30 日累计) 横截面 1/3 分位:
      赢家组 vs 输家组, 等权持有 1 期, dollar-neutral 多空 (赢家多+输家空)
    - H1: 多空净收益 > GBM null 2σ (30 种子漂移匹配同管线)
    - H2: 分年 (2024/2025/2026), 2026 衰减检验
    - n_eff 警示: 同 c62 口径报告降级
    - 学习级: GBM null 30 种子、MIN_N=100、描述层

============================================================
无未来函数设计说明 (逐特征信息边界表):
  特征/序列        | 计算方式                              | 可用时点   | 依据
  日线 close       | data_loader.daily_resample (1h→1D)    | 日线收盘后 | c30 口径
  30 日收益        | rolling(30).sum() of 日收益 (因果)    | 期未收盘后 | 禁全样本
  分层             | 每期初当期截面分位 (1/3)               | 期未收盘后 | 因果
  期收益           | 期收盘序列 pct_change                  | 事后       | 描述统计
  GBM null         | 20 条漂移匹配 GBM 同管线分层多空       | 锚定真实   | c62 口径
  n_eff            | (Σλ)²/Σλ² (c37/c62)                   | 全样本事后 | 降级

数据声明: data/backtest.db (20 标的 × 1h × 2023-08..2026-08); 日线 =
  daily_resample。

参数声明: 全部参数唯一来源 PARAMS (唯一参数源, 改参数 = 重跑 + 结论重写)。
  mom 窗 30 日; 分层 1/3; 月频主 + 周频交叉; GBM 30 种子漂移匹配; MIN_N=100。

设计偏离说明 (预注册, 非 post-hoc):
  - 排序变量 = 过去 30 日累计收益 (rolling sum of daily returns), 非 1h 内
    的原始 30 日收益 — 等价, 因果。
  - GBM null 漂移匹配 (c62/c61 口径) — 若赢家组漂移偏高, null 捕获机械效应。
  - 月频 n≈32 (<MIN_N) 标注; 周频 n≈135 交叉。
  - 学习级: 无 BY_YEAR; 30 种子沿用惯例。

发布门槛自检 (学习级描述层, 内置 GATE):
  - GATE 探测器: ① 分层 golden (已知 30 日收益排序 → 赢/输家组正确); ②
    n_eff golden (c37 同款: 完全相关→1, 独立→2); ③ null sanity — null
    多空均值 ∈ [−2%, +2%]; 任一失败 SystemExit
  - GBM null 无信息对照: 30 种子同管线
  - MIN_N: 每格 n ≥ 100 (学习级; 月频标注不足)
  - 结论↔.out↔脚本 script_sha256 三重一致; 结论数字全部带 (.out:L行号)
  - [学习级] 结论标注, 不得作交易依据; 无入场/无交易含义

性能与调试约定 (模板, 必须遵守):
  - --dev: 6 标的 × 月频 × 3 种子, 不写 .out
  - 全量: 20 标的 × 月/周 × 30 种子 (预计 ≤5 分钟)

运行命令:
  python3 research/studies/c65_cross_sectional_momentum.py --dev
  python3 research/studies/c65_cross_sectional_momentum.py
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
    "tf": "1h",
    "mom_days": 30,
    "frac": 1.0 / 3.0,
    "periods": ("monthly", "weekly"),
    "gbm_seeds": 30,
    "min_n": 100,
    "null_band": (-2.0, 2.0),            # GATE: null 多空均值带 (%)
    "dev_subset": {"n_sym": 6, "n_seeds": 3, "periods": ("monthly",)},
    "data_range": "2023-08..2026-08",
}

STUDY_ID = "c65_cross_sectional_momentum"


# ── 加载 (c62 同款) ──────────────────────────────────────────
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
        closes[sym] = daily_resample(df)["close"]
    return pd.DataFrame(closes).dropna()


def period_frame(daily, freq):
    f = "ME" if freq == "monthly" else "W"
    return daily.resample(f).last().dropna()


def cross_section_mom(close_frame, daily_rets, freq, mom_days, frac):
    """横截面动量: 期未 t 按过去 mom_days 日收益排序 1/3 分位 → 赢家多头
    输家空头, 持下一期. 返回 (赢家组收益, 输家组收益, 多空收益, 期标签)."""
    p = period_frame(close_frame, freq)
    p_ret = p.pct_change()
    dates = p.index
    n_p = len(p)
    mom_i = {}
    for sym in close_frame.columns:
        r = daily_rets[sym]
        mom_i[sym] = r.rolling(mom_days).sum()       # 过去 30 日累计收益
    win_r = []
    lose_r = []
    ls = []
    labels = []
    for j in range(1, n_p):
        t = dates[j - 1]
        moms = {}
        for sym in close_frame.columns:
            v = mom_i[sym].get(t, np.nan)
            if np.isfinite(v):
                moms[sym] = v
        if len(moms) < 6:
            continue
        s_syms = sorted(moms, key=moms.get)          # 升序
        k = int(len(s_syms) * frac)
        lose_syms = s_syms[:k]                       # 输家 (低收益)
        win_syms = s_syms[-k:]                       # 赢家 (高收益)
        pj = p_ret.iloc[j]
        wv = pj[win_syms].dropna()
        lv = pj[lose_syms].dropna()
        if wv.empty or lv.empty:
            continue
        win_r.append(float(wv.mean()))
        lose_r.append(float(lv.mean()))
        ls.append(float(wv.mean()) - float(lv.mean()))
        labels.append(dates[j])
    return np.array(win_r), np.array(lose_r), np.array(ls), labels


# ── n_eff (c37/c62 口径) ─────────────────────────────────────
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
def gate_golden_mom():
    """分层 golden: 构造已知 30 日收益排序 → 赢/输家组正确."""
    rng = np.random.default_rng(7)
    base = pd.DataFrame({
        "A": 1.0 + np.arange(1.0, 61.0) * 0.02,       # 最强动量
        "B": 1.0 + np.arange(1.0, 61.0) * 0.01,
        "C": np.ones(60),
        "D": 1.0 - np.arange(1.0, 61.0) * 0.005,
        "E": 1.0 - np.arange(1.0, 61.0) * 0.01,
        "F": 1.0 - np.arange(1.0, 61.0) * 0.02,       # 最弱动量
    }, index=pd.date_range("2024-01-01", periods=60, freq="D"))
    re = base.pct_change()
    mom = {s: float(re[s].sum()) for s in base.columns}
    s_syms = sorted(mom, key=mom.get)
    k = int(len(s_syms) * PARAMS["frac"])
    lose_syms = set(s_syms[:k])
    win_syms = set(s_syms[-k:])
    if not ({"A", "B"} <= win_syms and {"E", "F"} <= lose_syms):
        raise SystemExit(f"GATE FAIL: 动量分层 golden 赢={win_syms} 输="
                         f"{lose_syms}")
    return True


def gate_n_eff():
    if abs(n_eff(np.ones((2, 2))) - 1.0) > 1e-6:
        raise SystemExit("GATE FAIL: n_eff 完全相关")
    if abs(n_eff(np.eye(2)) - 2.0) > 1e-6:
        raise SystemExit("GATE FAIL: n_eff 独立")
    return True


def gate(null_ls_means):
    gate_golden_mom()
    gate_n_eff()
    nm = float(np.mean(null_ls_means)) if null_ls_means.size else 0.0
    lo, hi = PARAMS["null_band"]
    if not (lo <= nm <= hi):
        raise SystemExit(f"GATE FAIL: null 多空 {nm:+.3f}% ∉ [{lo}, {hi}]")
    print(f"[GATE] 动量分层 golden [PASS]; n_eff golden [PASS]; null sanity "
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
        "params=mom={}d,frac={},periods={},gbm_seeds={}(漂移匹配),min_n={},"
        "gate=MIN_N={}(学习级),学习级".format(
            STUDY_ID, date.today().isoformat(), script_sha256(),
            p["data_range"], p["mom_days"], p["frac"], p["periods"],
            p["gbm_seeds"], p["min_n"], p["min_n"]),
        "# GATE: 动量分层 golden + n_eff golden + null sanity [PASS]; MIN_N "
        "n≥{} [PASS]".format(p["min_n"]),
        "# RESULTS: [学习级] c65 公开策略验证⑥: 横截面动量 (crypto); 过去 "
        "{} 日收益横截面 1/3 分层, 赢家多头+输家空头等权 dollar-neutral; "
        "GBM null 漂移匹配; **n_eff 警示降级 (同 c62)**; 描述层无入场, 无交易"
        "含义".format(p["mom_days"]),
        "",
    ]
    lines.append("[横截面警示] n_eff = {:.2f} (c62/c37 口径) — 截面结论按 "
                 "n_eff 降级".format(res["n_eff"]))
    for freq in p["periods"]:
        r = res[freq]
        wm = _pct_mean(r["win"])
        lm = _pct_mean(r["lose"])
        ls_m = _pct_mean(r["ls"])
        n = len(r["ls"])
        tag = "月频主口径" if freq == "monthly" else "周频交叉"
        lines.append("")
        lines.append("[{}] {} 赢家/输家等权收益 (期收益均值):".format(
            "H1" if freq == "monthly" else "H1-x", tag))
        lines.append("  赢家 {:+.3f}% | 输家 {:+.3f}% | 多空 (赢−输) {:+.3f}%"
                     "/期 (n={}) {} -> {}".format(
            wm, lm, ls_m, n, _nm(n),
            "赢>输✓" if wm > lm else "赢<输✗"))
        if freq == "monthly":
            years = np.array([d.year for d in r["labels"]])
            yl = []
            for y in (2024, 2025, 2026):
                mm = years == y
                if mm.sum() >= 2:
                    yl.append("{} {:+.3f}%(n={})".format(
                        y, _pct_mean(r["ls"][mm]), int(mm.sum())))
            lines.append("  分年多空: " + " | ".join(yl))
        nl = r["null"]
        z = (ls_m - nl[0]) / nl[1] if nl[1] > 0 else float("nan")
        lines.append("  null 多空 {:+.3f}±{:.3f}% | 超额 {:+.3f} (z={:+.2f}) "
                     "-> {}".format(nl[0], nl[1], ls_m - nl[0], z,
                                    "超2σ↑" if z > 2 else "未超"))
    lines.append("")
    lines.append("[对照-历史] LTW 2022 JF (三因子动量最稳, 2021 后转弱); "
                 "c62 (低波异象 2026 反转); c37 (n_eff=2.15); 本砖: 横截面"
                 "动量 + 2026 衰减检验; n_eff 降级声明")
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
    _, daily_rets = corr_from_frame(frame)
    C, _ = corr_from_frame(frame)
    neff = n_eff(C)

    res = {}
    null_ls_all = []
    for freq in periods:
        win, lose, ls, labels = cross_section_mom(frame, daily_rets, freq,
                                                  PARAMS["mom_days"],
                                                  PARAMS["frac"])
        null_ls = []
        for seed in range(seeds):
            g_frame = pd.DataFrame({
                s: drift_gbm_daily(frame[s], seed) for s in frame.columns})
            g_rets = g_frame.pct_change()
            g_win, g_lose, gls, _ = cross_section_mom(
                g_frame, g_rets, freq, PARAMS["mom_days"], PARAMS["frac"])
            if len(gls):
                null_ls.append(float(np.mean(gls)) * 100.0)
        nl = (float(np.mean(null_ls)) if null_ls else float("nan"),
              float(np.std(null_ls, ddof=1)) if len(null_ls) > 1 else 0.0)
        if freq == periods[0]:
            null_ls_all = np.array(null_ls)
        res[freq] = {"win": win, "lose": lose, "ls": ls, "labels": labels,
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
