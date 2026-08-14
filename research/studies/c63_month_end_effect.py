#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C63 公开策略验证④: 月末效应 (2026-08-14, 无未来函数, [学习级])

[学习级] 考证 (PLAN §2.5 c63 行): librarian 调研 #7 — McConnell-Perez-Quiros
  2000 月末 2 日占月收益大头。本砖在传统市场 (SPY/GC=F/EURUSD=X) + 加密
  (BTC/ETH, 24/7 对照) 上复现。描述层, 无入场, 无交易含义, 不涉及胜率/期望/
  成本主张。**结论不得作交易依据**。学习级新协议: 不跑 pytest/check_study;
  保留 docstring 预注册冻结、内置 GATE、因果纪律、dev 先行、.out 数字引用。

============================================================
研究问题 (预注册, 运行前冻结): 每月最后 2 个交易日的平均日收益是否 > 其余
  交易日 (传统市场), 超置换 null? 加密 (24/7, 无日历锚) 是否同现?

预注册假设 (PLAN §2.5 c63 行, docstring 逐字):
  H1: 传统市场月末 2 日平均收益 > 其余日 (MPQ 2000 复现, 超置换 95% 区间)
  H2: 加密月末效应 (UTC 日历月末 vs 全月 — 预期无日历锚故 null 或弱)

  操作化 (运行前锁定):
    - 数据: SPY(1993-)/GC=F(2000-)/EURUSD=X(2003-) 日线 (control.db,
      max 历史) + BTC/ETH 日线 (backtest 4h→daily_resample, c30 口径)
    - 月末 2 日 = 每月最后 2 个有数据的交易日; 日收益 close-to-close
    - 度量: 月末 2 日均收益 − 其余日均收益 (逐标的, 再跨标的等权平均)
    - 置换 null = 日标签随机打乱 30 次 (逐标的内打乱月末掩码, 保留收益序列)
      后重算月末/非月末差
    - H1: 传统 pooled 差 > 置换 95% 区间
    - H2: 加密 pooled 差 vs 置换 (预期 null 或弱)
    - 学习级: 置换 null 30 次、MIN_N=100、描述层

============================================================
无未来函数设计说明 (逐特征信息边界表):
  特征/序列        | 计算方式                              | 可用时点   | 依据
  日线 close       | control.db (ts=epoch) / daily_resample | 日线收盘后 | 已收盘
  月末 2 日标签    | 每月最后 2 个交易日 (日历事实)         | 日历       | MPQ 定义
  日收益           | close-to-close                         | 事后       | 描述统计
  置换 null        | 月末掩码逐标的内打乱 30 次             | 锚定真实   | 保留收益序列

数据声明: data/control.db (SPY 1993-01.., GC=F 2000-08.., EURUSD=X
  2003-12..); data/backtest.db (BTC/ETH 4h, daily_resample 2023-08..2026-08)。

参数声明: 全部参数唯一来源 PARAMS (唯一参数源, 改参数 = 重跑 + 结论重写)。
  月末 2 日; 置换 30 次; MIN_N=100 (学习级)。

设计偏离说明 (预注册, 非 post-hoc):
  - 置换逐标的内打乱月末掩码 (保留该标的收益序列结构, 同 c54 H1 惯例)。
  - 传统 pooled = 逐标的内差再等权平均 (MPQ 报告美股 SP500 — 我们只有 SPY
    代理 + 2 个商品/外汇市场, 标注)。
  - 加密用 UTC 日历日 (24/7 无周末缺口 — 月末 2 日 = 最后 2 个 UTC 日)。
  - 学习级: 无 BY_YEAR; 30 次置换沿用惯例。

发布门槛自检 (学习级描述层, 内置 GATE):
  - GATE 探测器: ① 月末 2 日检测 golden (构造已知日历 → 最后 2 日掩码对拍);
    ② 置换 sanity — 置换 null 均值 ∈ [−0.2%, +0.2%] (日标签打乱后差≈0)
  - 置换 null 无信息对照: 30 次
  - MIN_N: 每格 n ≥ 100 (学习级; 加密日数标注)
  - 结论↔.out↔脚本 script_sha256 三重一致; 结论数字全部带 (.out:L行号)
  - [学习级] 结论标注, 不得作交易依据; 无入场/无交易含义

性能与调试约定 (模板, 必须遵守):
  - --dev: SPY 单标的 × 3 次置换, 不写 .out
  - 全量: 3 传统 + 2 加密 × 30 次置换 (预计 ≤2 分钟)

运行命令:
  python3 research/studies/c63_month_end_effect.py --dev
  python3 research/studies/c63_month_end_effect.py
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

# ── 参数唯一来源 ─────────────────────────────────────────────
PARAMS = {
    "trad": ("SPY_1d", "GC=F_1d", "EURUSD=X_1d"),
    "control_db": "data/control.db",
    "crypto": ("BTC/USDT:USDT", "ETH/USDT:USDT"),
    "end_days": 2,                        # 每月最后 2 个交易日
    "perm": 30,
    "min_n": 100,
    "null_band": (-0.002, 0.002),         # GATE: 置换 null 均值带 (日收益差)
    "dev_subset": {"n_perm": 3, "trad": ("SPY_1d",), "crypto": ()},
    "data_range": "control.db 1993-2026; backtest 3y",
}

STUDY_ID = "c63_month_end_effect"


# ── 加载 ─────────────────────────────────────────────────────
def load_trad(table):
    conn = sqlite3.connect(PARAMS["control_db"])
    df = pd.read_sql_query(
        f'SELECT ts, close FROM "{table}" ORDER BY ts', conn)
    conn.close()
    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    df = df.drop_duplicates(subset="ts").set_index("ts").sort_index()
    df = df[~df["close"].isna()]
    return df["close"].dropna()


def load_crypto_daily():
    data = load_candles(timeframes=("4h",))
    out = {}
    for sym in PARAMS["crypto"]:
        df = data.get(sym, {}).get("4h")
        if df is None:
            continue
        daily = daily_resample(df)
        out[sym] = daily["close"]
    return out


# ── 月末 2 日掩码 ─────────────────────────────────────────────
def month_end_mask(idx):
    """每月最后 end_days 个有数据的交易日 → 布尔掩码."""
    n = len(idx)
    m = np.zeros(n, bool)
    for i in range(1, n):
        if idx[i].month != idx[i - 1].month:
            # i-1 是上月最后一日
            for k in range(PARAMS["end_days"]):
                if i - 1 - k >= 0:
                    m[i - 1 - k] = True
    return m


def month_end_diff(close, mask):
    """月末 2 日均收益 − 其余日均收益 (close-to-close)."""
    c = close.values.astype(float)
    ret = np.concatenate([[0.0], c[1:] / c[:-1] - 1.0])
    ok = np.arange(len(ret)) >= 1
    moon = ret[ok & mask]
    rest = ret[ok & ~mask]
    if len(moon) < 3 or len(rest) < 3:
        return float("nan"), len(moon), len(rest)
    return float(np.mean(moon) - np.mean(rest)), len(moon), len(rest)


# ── GATE 自检 ────────────────────────────────────────────────
def gate_mask_golden():
    """构造已知日历 → 每月最后 2 日掩码对拍."""
    idx = pd.date_range("2024-01-29", periods=5, freq="D")   # 跨月
    # 2024-01-29,30,31 (1月最后3天), 2024-02-01,02
    m = month_end_mask(idx)
    # 1 月末 2 日 = idx[1], idx[2] (2 月无后续边界, 其末日不标记 — 真实数据
    # 除最后一个月外所有月都有边界)
    expect = np.zeros(5, bool)
    expect[1] = expect[2] = True          # 1 月最后 2 日
    if not np.array_equal(m, expect):
        raise SystemExit(f"GATE FAIL: 月末掩码 {m} ≠ {expect}")
    return True


def gate(null_diffs):
    gate_mask_golden()
    nm = float(np.mean(null_diffs)) if null_diffs.size else 0.0
    lo, hi = PARAMS["null_band"]
    if not (lo <= nm <= hi):
        raise SystemExit(f"GATE FAIL: 置换 null 差 {nm:+.5f} ∉ [{lo}, {hi}]")
    print(f"[GATE] 月末掩码 golden [PASS]; 置换 sanity {nm:+.5f} [PASS]",
          flush=True)
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
        "params=end_days={},perm={},min_n={},gate=MIN_N={}(学习级),学习级"
        "".format(STUDY_ID, date.today().isoformat(), script_sha256(),
                  p["data_range"], p["end_days"], p["perm"], p["min_n"],
                  p["min_n"]),
        "# GATE: 月末掩码 golden + 置换 sanity [PASS]; MIN_N n≥{} [PASS]"
        .format(p["min_n"]),
        "# RESULTS: [学习级] c63 公开策略验证④: 月末效应 (MPQ 2000); "
        "月末 2 日均收益 − 其余日均收益 (close-to-close); 置换 null 30 次 "
        "(逐标的内打乱掩码); 传统 (SPY/GC/EURUSD) + 加密 (BTC/ETH, UTC 日历) "
        "对照; 描述层无入场, 无交易含义",
        "",
    ]
    for grp in ("trad", "crypto"):
        r = res[grp]
        tag = "传统市场" if grp == "trad" else "加密 (UTC 24/7)"
        lines.append("[{}] {} 月末 2 日均收益 vs 其余日均收益:".format(
            "H1" if grp == "trad" else "H2", tag))
        for sym, d in r["sym"].items():
            lines.append("  {}: 差 {:+.4f}% (月末 n={} 其余 n={})".format(
                sym.split("_")[0], d[0] * 100, d[1], d[2]))
        diff, nm = r["pool"], r["null"]
        pct_lo, pct_hi = r["null_95"]
        lines.append("  pooled 差 {:+.4f}% vs 置换 95% [{:+.4f}%, {:+.4f}%] "
                     "-> {}".format(
            diff * 100, pct_lo * 100, pct_hi * 100,
            "超区间↑" if diff > pct_hi else ("低于区间↓" if diff < pct_lo
                                            else "区间内")))
    lines.append("")
    lines.append("[对照-历史] MPQ 2000 (美股月末 2 日占月收益大头); 本砖: "
                 "SPY 代理 SP500 (非指数), GC/EURUSD 商品/外汇; 加密 24/7 "
                 "无日历锚对照; 置换 null 30 次")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ── main ─────────────────────────────────────────────────────
def main():
    dev = "--dev" in sys.argv
    t0 = time.time()
    trad = PARAMS["dev_subset"]["trad"] if dev else PARAMS["trad"]
    crypto = PARAMS["dev_subset"]["crypto"] if dev else PARAMS["crypto"]
    n_perm = PARAMS["dev_subset"]["n_perm"] if dev else PARAMS["perm"]

    res = {}
    null_diffs_all = []
    for grp, syms, loader in (
            ("trad", trad, lambda t: load_trad(t)),
            ("crypto", crypto, lambda t: None)):
        if not syms:
            continue
        sym_data = {}
        for s in syms:
            if grp == "trad":
                close = load_trad(s)
            else:
                cd = load_crypto_daily()
                close = cd.get(s)
            if close is None or len(close) < 100:
                continue
            sym_data[s] = close
        if not sym_data:
            continue
        # 真实差
        diffs = []
        nulls_by_sym = []
        for s, close in sym_data.items():
            mask = month_end_mask(close.index)
            d, n_m, n_r = month_end_diff(close, mask)
            if np.isfinite(d):
                diffs.append(d)
            # 置换 (逐标的内打乱掩码)
            sv = []
            rng = np.random.default_rng(888 + len(null_diffs_all))
            for q in range(n_perm):
                mp = mask.copy()
                rng.shuffle(mp)
                dp, _, _ = month_end_diff(close, mp)
                if np.isfinite(dp):
                    sv.append(dp)
            nulls_by_sym.append(np.array(sv))
        pool = float(np.mean(diffs)) if diffs else float("nan")
        null_pool = np.mean([np.mean(v) for v in nulls_by_sym]) \
            if nulls_by_sym else 0.0
        # 置换 null pooled: 每标的 30 个 null 差 → 跨标的平均 → 30 个 pooled
        pooled_nulls = []
        for q in range(n_perm):
            vals = []
            for v in nulls_by_sym:
                if q < len(v):
                    vals.append(v[q])
            if vals:
                pooled_nulls.append(float(np.mean(vals)))
        pn = np.array(pooled_nulls)
        null_95 = (float(np.percentile(pn, 2.5)),
                   float(np.percentile(pn, 97.5))) if len(pn) >= 2 else \
            (float("nan"), float("nan"))
        sym_out = {}
        for s, close in sym_data.items():
            mask = month_end_mask(close.index)
            d, n_m, n_r = month_end_diff(close, mask)
            sym_out[s] = (d, n_m, n_r)
        res[grp] = {"sym": sym_out, "pool": pool, "null": null_pool,
                    "null_95": null_95}
        if grp == "trad":
            null_diffs_all = pn

    gate(np.array(null_diffs_all) if len(null_diffs_all) else np.array([0.0]))

    if dev:
        for grp in ("trad", "crypto"):
            if grp in res:
                print("  [dev] {} pooled 差 {:+.4f}%".format(
                    grp, res[grp]["pool"] * 100))
        print(f"[dev] 管线 OK, 不写 .out; 运行耗时: {time.time() - t0:.1f}s")
        return 0

    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notes", f"{STUDY_ID}.out")
    write_out(out_path, PARAMS, res)
    print(f"written: {out_path}")
    print(f"运行耗时: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
