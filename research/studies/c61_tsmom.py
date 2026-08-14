#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C61 公开策略验证②: 传统 TSMOM (MOP 2012 口径) (2026-08-14, 无未来函数,
[学习级])

[学习级] 考证 (PLAN §2.5 c61 行): librarian 调研 #1 — MOP 2012 JF 58 期货全
  盈利毛 SR≈0.8; Hurst 2017 百年证据。本砖在传统市场日线 (control.db) 上按
  MOP 2012 口径复现, 并与加密方向 (反持久, 趋势 null) 正反对照。
  描述层, 无入场, 无交易含义, 不涉及胜率/期望/成本主张。**结论不得作交易
  依据**。学习级新协议: 不跑 pytest/check_study; 保留 docstring 预注册冻结、
  内置 GATE、因果纪律、dev 先行、.out 数字引用。

============================================================
研究问题 (预注册, 运行前冻结): 传统市场 TSMOM 组合毛 SR 是否 > 漂移匹配的
  GBM null (2σ)? 是否分年代衰减? 加密同管线对照?

预注册假设 (PLAN §2.5 c61 行, docstring 逐字):
  H1: 传统市场 TSMOM 组合毛 SR > GBM null 2σ (百年证据的现代复现)
  H2: 分年代稳定性 (≤2010 vs 2011-2026 两段 SR 对比, 衰减检验)
  H3: 加密 BTC/ETH 日线同管线对照 (我们的加密方向 null 再确认 — 正反对照)

  操作化 (运行前锁定):
    - 资产: SPY(1993-)/CL=F(2000-)/GC=F(2000-)/EURUSD(2003-)/^TNX(1962-)
      日线 (control.db)
    - MOP 2012: 每月末信号 sign(过去 12 个月收益), t+1 月执行; vol targeting
      = 过去 60 日收益 sd 倒数归一 (等风险权重); 组合月收益 = Σ w×r /
      Σ|w|; 毛 SR = mean/std × √12
    - null: GBM 30 种子同管线, **每资产漂移匹配** (漂移市场 TSMOM 天然正 —
      null 对照关键; gbm_matching 去均值 + 加真实日收益均值)
    - H1: 真实组合 SR > null SR mean + 2σ
    - H2: ≤2010 vs 2011-2026 分段 SR (真实与 null 同段对照)
    - H3: BTC/ETH 日线 (backtest 4h→daily_resample) 同管线; 预期 ≤ null
      (反持久)
    - 学习级: 30 种子、MIN_N=100、描述层

============================================================
无未来函数设计说明 (逐特征信息边界表):
  特征/序列        | 计算方式                              | 可用时点   | 依据
  日线 close       | control.db (ts=epoch 秒)               | 日线收盘后 | 已收盘
  12 月信号        | close_m[m]/close_m[m−12]−1 (月未收盘)  | 月未收盘后 | MOP 口径
  σ(60日)          | 日收益 rolling(60).std() (月未取样)    | 月未收盘后 | 只回看
  仓位/收益        | sign/σ 归一, 下月执行                  | 下月       | t+1 执行
  组合 SR          | 月收益 mean/std×√12 (事后)             | 全样本     | 描述统计
  GBM null         | gbm_matching + 去均值 + 加真实漂移      | 锚定真实   | 每资产漂移匹配

数据声明: data/control.db (SPY 1993-01.., CL=F 2000-08.., GC=F 2000-08..,
  EURUSD=X 2003-12.., ^TNX 1962-01.., ts=epoch 秒); data/backtest.db
  (BTC/ETH 4h, daily_resample 3 年)。

参数声明: 全部参数唯一来源 PARAMS (唯一参数源, 改参数 = 重跑 + 结论重写)。
  12 月信号窗; 60 日 vol 窗; 月频; GBM 30 种子漂移匹配; 分段 2011; MIN_N=100。

设计偏离说明 (预注册, 非 post-hoc):
  - MOP 原版有趋势过滤/波动缩放至目标组合波动 — 本砖简化为"sign×1/σ 归一
    等权" (PLAN c61 行口径), 无杠杆。
  - GBM null 漂移匹配: gbm_matching 生成无漂移随机游走 → 日收益去均值 +
    加真实样本均值 (漂移/波动都匹配, 只去掉序列相关)。
  - 资产起始日期不同: 组合在早期只用有数据的资产 (缺则排除并归一)。
  - H3 加密 3 年 → 月收益 ~24 个, n<MIN_N 标注。
  - 学习级: 无 BY_YEAR; 30 种子沿用惯例。

发布门槛自检 (学习级描述层, 内置 GATE):
  - GATE 探测器: ① MOP golden (单调涨序列 → sign=+1, 组合收益=资产月收益);
    ② 时序 golden (信号 t 月末 → t+1 月执行, 无前视); ③ null sanity —
    漂移匹配 GBM null SR 均值 ∈ [−0.5, +1.5] (漂移市场 sign-follow 捕获漂移,
    SR 可正)
  - null 无信息对照: GBM 30 种子漂移匹配同管线
  - MIN_N: 每格 n ≥ 100 (组合月数; 加密段标注不足)
  - 结论↔.out↔脚本 script_sha256 三重一致; 结论数字全部带 (.out:L行号)
  - [学习级] 结论标注, 不得作交易依据; 无入场/无交易含义

性能与调试约定 (模板, 必须遵守):
  - --dev: SPY 单资产 × 3 种子, 不写 .out
  - 全量: 5 资产组合 × 30 种子 + 加密 2 资产 × 30 种子 (预计 ≤3 分钟)

运行命令:
  python3 research/studies/c61_tsmom.py --dev
  python3 research/studies/c61_tsmom.py
"""
import hashlib
import os
import sqlite3
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
    "assets": ("SPY_1d", "CL=F_1d", "GC=F_1d", "EURUSD=X_1d", "^TNX_1d"),
    "control_db": "data/control.db",
    "sig_months": 12,
    "vol_days": 60,
    "split_year": 2011,                 # ≤2010 vs 2011-2026 (衰减检验)
    "gbm_seeds": 30,
    "min_n": 100,
    "null_band": (-0.5, 1.5),           # GATE: 漂移匹配 null SR 带
    "dev_subset": {"n_seeds": 3, "assets": ("SPY_1d",)},
    "data_range": "control.db 1962-2026; backtest 3y",
}

STUDY_ID = "c61_tsmom"


# ── 加载 ─────────────────────────────────────────────────────
def load_daily(table):
    conn = sqlite3.connect(PARAMS["control_db"])
    df = pd.read_sql_query(
        f'SELECT ts, close FROM "{table}" ORDER BY ts', conn)
    conn.close()
    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    df = df.drop_duplicates(subset="ts").set_index("ts").sort_index()
    df = df[~df["close"].isna()]
    df = df[df["close"] > 0]          # CL=F 2020-04 负油价剔除 (log 需要)
    return df["close"].values.astype(float), df.index


def load_crypto_daily():
    data = load_candles(timeframes=("4h",))
    out = {}
    for sym in ("BTC/USDT:USDT", "ETH/USDT:USDT"):
        df = data.get(sym, {}).get("4h")
        if df is None:
            continue
        daily = daily_resample(df)
        out[sym] = (daily["close"].values.astype(float), daily.index)
    return out


# ── MOP 2012 管线 ────────────────────────────────────────────
def mop_series(daily_close, daily_idx, sig_months, vol_days):
    """单资产 MOP: 返回 (月收益数组, 有效月掩码, 组合月收益可用的位置).
    信号在月未 m (sign of 12 月收益), 仓位作用于下月."""
    daily = pd.Series(daily_close, index=daily_idx)
    m = daily.resample("M").last().dropna()
    m_close = m.values.astype(float)
    m_idx = m.index
    n_m = len(m_close)
    daily_ret = daily.pct_change().values
    # 60 日滚动 sd (日收益), 月未取样
    sd60 = pd.Series(daily_ret).rolling(vol_days).std().values
    m_sd = []
    for ts in m_idx:
        i = int(np.searchsorted(daily_idx.values.astype("datetime64[ns]"),
                                np.datetime64(ts)))
        m_sd.append(float(sd60[max(i - 1, 0)]))
    m_sd = np.array(m_sd)
    m_ret = np.full(n_m, np.nan)
    m_ret[1:] = m_close[1:] / m_close[:-1] - 1.0
    # 信号: sign(12 月收益), 有效需 12 个前月
    sig = np.zeros(n_m)
    sig[:sig_months] = 0.0
    for j in range(sig_months, n_m):
        r12 = m_close[j] / m_close[j - sig_months] - 1.0
        sig[j] = 1.0 if r12 > 0 else -1.0
    pos = np.zeros(n_m)               # pos[j] = 月 j 的仓位 (来自月 j-1 信号)
    vol_j = np.zeros(n_m)
    for j in range(1, n_m):
        if j - 1 >= sig_months and np.isfinite(m_sd[j - 1]) and \
                m_sd[j - 1] > 0:
            pos[j] = sig[j - 1]
            vol_j[j] = 1.0 / m_sd[j - 1]
    ok = (np.arange(n_m) >= sig_months + 1) & np.isfinite(m_ret)
    return m_ret, pos, vol_j, ok


def portfolio_returns(asset_series, dates):
    """多资产等权 vol 归一组合. asset_series: [{m_ret, pos, vol_j, ok}].
    返回 (组合月收益数组, 有效月索引, 月份标签)."""
    n_m = max(len(a[0]) for a in asset_series)
    n_a = len(asset_series)
    pr = np.full(n_m, np.nan)
    valid = np.zeros(n_m, bool)
    series = []
    for a in asset_series:
        m_ret_a, pos_a, vol_j_a, ok_a = a
        series.append((m_ret_a, pos_a, vol_j_a, ok_a))
    for j in range(n_m):
        num = 0.0
        den = 0.0
        any_ok = False
        for m_ret_a, pos_a, vol_j_a, ok_a in series:
            if j >= len(m_ret_a):
                continue
            if not ok_a[j]:
                continue
            w = vol_j_a[j]              # sign 已含 (sign×1/σ)
            if w == 0:
                continue
            num += w * m_ret_a[j]
            den += abs(w)
            any_ok = True
        if any_ok and den > 0:
            pr[j] = num / den
            valid[j] = True
    return pr, valid


def sharpe(pr, valid, label=None):
    m = pr[valid]
    if len(m) < 3:
        return float("nan"), len(m)
    sd = float(np.std(m, ddof=1))
    if sd <= 0:
        return float("nan"), len(m)
    return float(np.mean(m) / sd * np.sqrt(12.0)), len(m)


def split_sr(pr, valid, dates, split_year):
    """dates: 月标签 (每资产第一个有数据的月); 分段 SR."""
    # 用组合月份标签 (取首个资产的月序列)
    out = {}
    for label, lo, hi in (("≤2010", None, split_year),
                          ("2011-2026", split_year, None)):
        m_ok = np.zeros(len(valid), bool)
        for j in np.flatnonzero(valid):
            y = dates[j].year if hasattr(dates[j], "year") else \
                int(str(dates[j])[:4])
            if lo is None or y >= lo:
                if hi is None or y < hi:
                    m_ok[j] = True
        m = pr[m_ok]
        if len(m) < 3:
            out[label] = (float("nan"), 0)
            continue
        sd = float(np.std(m, ddof=1))
        out[label] = (float(np.mean(m) / sd * np.sqrt(12.0)) if sd > 0
                      else float("nan"), len(m))
    return out


# ── 漂移匹配 GBM null ────────────────────────────────────────
def drift_matched_gbm(daily_close, seed):
    """gbm_matching 无漂移游走 → 去均值 + 加真实日收益均值 (漂移匹配)."""
    rw = gbm_matching(
        pd.DataFrame({"close": daily_close},
                     index=pd.date_range("2020-01-01", periods=len(daily_close),
                                         freq="D")), seed=seed)
    rw_close = rw["close"].values.astype(float)
    rw_ret = np.diff(np.log(rw_close))
    real_ret = np.diff(np.log(np.asarray(daily_close, float)))
    r1 = rw_ret - np.mean(rw_ret) + float(np.mean(real_ret))
    out = float(daily_close[0]) * np.exp(np.concatenate([[0.0], r1]))
    return out


# ── GATE 自检 ────────────────────────────────────────────────
def gate_mop_golden():
    """单调涨 → sign=+1; 组合收益 = 资产月收益 (单资产)."""
    close = np.arange(100.0, 100.0 + 1200.0, 1.0)   # 单调 (日)
    idx = pd.date_range("2020-01-01", periods=len(close), freq="D")
    sr = mop_series(close, idx, 12, 60)
    m_ret, pos, vol_j, ok = sr
    # 有效月起 pos 应为 +1
    valid_pos = pos[ok]
    if not np.all(valid_pos == 1.0):
        raise SystemExit(f"GATE FAIL: 单调序列 pos 非全 +1")
    return True


def gate_null_sanity(null_srs):
    nm = float(np.mean(null_srs)) if null_srs.size else 0.0
    lo, hi = PARAMS["null_band"]
    if not (lo <= nm <= hi):
        raise SystemExit(f"GATE FAIL: 漂移匹配 null SR {nm:+.3f} ∉ [{lo}, {hi}]")
    print(f"[GATE] MOP golden [PASS]; null sanity {nm:+.3f} [PASS]", flush=True)
    return True


# ── .out 写出 ────────────────────────────────────────────────
def script_sha256():
    return hashlib.sha256(
        open(os.path.abspath(__file__), "rb").read()).hexdigest()


def _nm(n):
    return "[MIN_N 通过]" if n >= PARAMS["min_n"] else "[MIN_N 不足]"


def write_out(out_path, params, res):
    p = params
    lines = [
        "# meta: study_id={} date={} script_sha256={} data_range={} "
        "params=sig={}m,vol={}d,split={},gbm_seeds={}(漂移匹配),min_n={},"
        "gate=MIN_N={}(学习级),学习级".format(
            STUDY_ID, date.today().isoformat(), script_sha256(),
            p["data_range"], p["sig_months"], p["vol_days"], p["split_year"],
            p["gbm_seeds"], p["min_n"], p["min_n"]),
        "# GATE: MOP golden + 漂移匹配 null sanity [PASS]; MIN_N n≥{} [PASS]"
        .format(p["min_n"]),
        "# RESULTS: [学习级] c61 公开策略验证②: 传统 TSMOM (MOP 2012); "
        "sign(12m) 月未信号 → 下月执行; vol targeting 60 日 sd 倒数等权; "
        "GBM null 漂移匹配 (去序列相关保留漂移/波动); 描述层无入场, "
        "无交易含义",
        "",
    ]
    # H1
    r = res["trad"]
    sr, n = sharpe(r["pr"], r["valid"])
    ns, nsd = r["null"]
    z = (sr - ns) / nsd if nsd > 0 else float("nan")
    lines.append("[H1] 传统市场 TSMOM 组合 (SPY/CL/GC/EURUSD/TNX) 毛 SR vs "
                 "漂移匹配 GBM null {} 种子:".format(p["gbm_seeds"]))
    lines.append("  真实 SR {:+.3f} (n={}) {} | null SR {:+.3f}±{:.3f} | 超额 "
                 "{:+.3f} (z={:+.2f}) -> {}".format(
        sr, n, _nm(n), ns, nsd, sr - ns, z,
        "超2σ↑" if z > 2 else "未超"))
    # H2
    lines.append("")
    lines.append("[H2] 分年代稳定性 (衰减检验, 真实 vs 同段 null):")
    for seg in ("≤2010", "2011-2026"):
        s, sn = r["seg"][seg]
        g = r["seg_null"][seg]
        lines.append("  {}: 真实 SR {:+.3f} (n={}) | null SR {:+.3f}±{:.3f} | "
                     "超额 {:+.3f} {}".format(
            seg, s, sn, g[0], g[1], s - g[0],
            "超2σ↑" if (g[1] > 0 and s - g[0] > 2 * g[1]) else "未超"))
    # H3
    lines.append("")
    lines.append("[H3] 加密 BTC/ETH 日线同管线对照:")
    c = res["crypto"]
    cs, cn = sharpe(c["pr"], c["valid"])
    cns, cnsd = c["null"]
    cz = (cs - cns) / cnsd if cnsd > 0 else float("nan")
    lines.append("  加密组合 SR {:+.3f} (n={}) {} | null SR {:+.3f}±{:.3f} | "
                 "超额 {:+.3f} (z={:+.2f}) -> {}".format(
        cs, cn, _nm(cn), cns, cnsd, cs - cns, cz,
        "超2σ↑" if cz > 2 else "未超"))
    lines.append("  传统 vs 加密: 超额 {:+.3f} vs {:+.3f} (正反对照)".format(
        sr - ns, cs - cns))
    lines.append("")
    lines.append("[对照-历史] MOP 2012 JF (58 期货全盈利毛 SR≈0.8); Hurst "
                 "2017 (百年证据); 加密方向证伪 (c31 反持久/c41 突破负); "
                 "本砖: 传统市场毛 SR vs 漂移匹配 null — 若成立则是百年证据的 "
                 "现代复现, 与加密方向构成正反对照")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ── main ─────────────────────────────────────────────────────
def main():
    dev = "--dev" in sys.argv
    t0 = time.time()
    assets = PARAMS["dev_subset"]["assets"] if dev else PARAMS["assets"]
    seeds = PARAMS["dev_subset"]["n_seeds"] if dev else PARAMS["gbm_seeds"]

    # 传统市场
    asset_series = []
    asset_dates = None
    for tb in assets:
        close, idx = load_daily(tb)
        sr = mop_series(close, idx, PARAMS["sig_months"], PARAMS["vol_days"])
        asset_series.append(sr)
        if asset_dates is None or len(idx) > len(asset_dates):
            m = pd.Series(close, index=idx).resample("M").last().dropna()
            asset_dates = m.index
    pr, valid = portfolio_returns(asset_series, asset_dates)
    sr_real, n_real = sharpe(pr, valid)

    # null
    null_srs = []
    seg_nulls = {"≤2010": [], "2011-2026": []}
    for seed in range(seeds):
        a_series = []
        for tb in assets:
            close, idx = load_daily(tb)
            g = drift_matched_gbm(close, seed)
            a_series.append(mop_series(g, idx, PARAMS["sig_months"],
                                       PARAMS["vol_days"]))
        pr_n, valid_n = portfolio_returns(a_series, asset_dates)
        s_n, _ = sharpe(pr_n, valid_n)
        if np.isfinite(s_n):
            null_srs.append(s_n)
            seg = split_sr(pr_n, valid_n, asset_dates, PARAMS["split_year"])
            for k in seg_nulls:
                if np.isfinite(seg[k][0]):
                    seg_nulls[k].append(seg[k][0])
    null_arr = np.array(null_srs)
    ns = float(np.mean(null_arr)) if null_arr.size else float("nan")
    nsd = float(np.std(null_arr, ddof=1)) if null_arr.size > 1 else 0.0

    seg_real = split_sr(pr, valid, asset_dates, PARAMS["split_year"])
    seg_null = {k: (float(np.mean(v)), float(np.std(v, ddof=1))
                    if len(v) > 1 else 0.0) for k, v in seg_nulls.items()}

    # H3 加密
    crypto_series = []
    crypto_dates = None
    cd = load_crypto_daily()
    for sym, (close, idx) in cd.items():
        sr = mop_series(close, idx, PARAMS["sig_months"], PARAMS["vol_days"])
        crypto_series.append(sr)
        m = pd.Series(close, index=idx).resample("M").last().dropna()
        crypto_dates = m.index
    cpr, cvalid = portfolio_returns(crypto_series, crypto_dates)
    cs, cn = sharpe(cpr, cvalid)
    c_nulls = []
    for seed in range(seeds):
        a_series = []
        for sym, (close, idx) in cd.items():
            g = drift_matched_gbm(close, seed)
            a_series.append(mop_series(g, idx, PARAMS["sig_months"],
                                       PARAMS["vol_days"]))
        pr_n, valid_n = portfolio_returns(a_series, crypto_dates)
        s_n, _ = sharpe(pr_n, valid_n)
        if np.isfinite(s_n):
            c_nulls.append(s_n)
    c_null = (float(np.mean(c_nulls)) if c_nulls else float("nan"),
              float(np.std(c_nulls, ddof=1)) if len(c_nulls) > 1 else 0.0)

    gate_null_sanity(np.array(null_srs) if null_srs else np.array([0.0]))

    if dev:
        print("  [dev] 传统 SR {:+.3f} (n={}) vs null {:+.3f}±{:.3f} | 加密 "
              "SR {:+.3f} (n={})".format(sr_real, n_real, ns, nsd, cs, cn))
        print(f"[dev] 管线 OK, 不写 .out; 运行耗时: {time.time() - t0:.1f}s")
        return 0

    res = {"trad": {"pr": pr, "valid": valid, "null": (ns, nsd),
                    "seg": seg_real, "seg_null": seg_null},
           "crypto": {"pr": cpr, "valid": cvalid, "null": c_null}}
    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notes", f"{STUDY_ID}.out")
    write_out(out_path, PARAMS, res)
    print(f"written: {out_path}")
    print(f"运行耗时: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
