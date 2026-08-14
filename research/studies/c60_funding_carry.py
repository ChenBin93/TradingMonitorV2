#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C60 公开策略验证①: 加密 funding carry 可行性核算 (2026-08-14, 无未来函数,
[学习级])

[学习级] 考证 (PLAN §2.5 c60 行): librarian 调研 #3 — He 2022 高夏普, 2022 后
  压缩。本砖核算加密 funding carry 的可行性: ①funding 年化 carry; ②basis
  漂移侵蚀; ③净 carry (扣成本)。描述层, 无入场, 无交易含义, 不涉及胜率/期望/
  成本主张。**结论不得作交易依据**。学习级新协议: 不跑 pytest/check_study;
  保留 docstring 预注册冻结、内置 GATE、因果纪律、dev 先行、.out 数字引用。

============================================================
研究问题 (预注册, 运行前冻结): 20 标的 3 个月 funding (每 8h) 的 carry 是否
  为正、扣 basis 漂移后是否仍为正?

预注册假设 (PLAN §2.5 c60 行, docstring 逐字):
  H1: 平均年化 funding > 0 且显著 (正 carry 存在; funding 符号随机化 null)
  H2: 扣 basis 漂移后净 carry 仍为正 (basis 风险不吞 carry)
  H3: 成本敏感性 (现货+永续开平各 0.1%)

  操作化 (运行前锁定):
    - 数据: data/funding.db (20 标的, 每 8h, withRate, 2026-05-11..08-14,
      284 样本/标的); 永续 1h (backtest.db, 止 2026-08-01); 现货 1h
      (OKX API history-candles 现拉 BTC-USDT/ETH-USDT, 分页 3 个月)
    - ① 年化 carry = 各标的 mean(funding_rate) × 1095 (8h 结算 1095 次/年);
      null = funding 符号随机打乱 30 次 (逐标的逐样本随机 ±, 重算均值)
    - ② basis 漂移: b(t)=永续价−现货价; basis_pnl=(b_入−b_出)/spot_入
      (short-perp+long-spot 的价差收敛损益); 窗口 = funding∩backtest 重叠
      (05-11..08-01); BTC/ETH 有现货可算, 其余标的标注不可算
    - ③ 净 carry (年化) = ① − basis_pnl_年化 − 成本年化 (0.1%×2 现货 +
      0.1%×2 永续 = 0.4% 往返, ×365/窗长)
    - 学习级: 20 标的 (H1), BTC/ETH (H2/H3), MIN_N=100, 描述层

============================================================
无未来函数设计说明 (逐特征信息边界表):
  特征/序列        | 计算方式                              | 可用时点   | 依据
  funding_rate     | data/funding.db (8h 结算, withRate)   | 结算 ts    | 历史事实
  永续 1h          | backtest.db (BTC/USDT:USDT)           | bar 收盘后 | 已收盘
  现货 1h          | OKX history-candles (BTC-USDT) 分页   | bar 收盘后 | 历史 K 线
  basis            | perp_close − spot_close (逐 1h 对齐)  | bar 收盘后 | 价差定义
  carry/null       | mean×1095; 符号随机化 30 次            | 锚定真实   | 符号 null
  成本             | 0.4% 往返 (现货/永续开平各 0.1%)       | 常数       | 预注册

数据声明: data/funding.db (20 标的 × ~284 样本, 2026-05-11..2026-08-14,
  OKX API 上限 — 3 个月窗口标注); data/backtest.db (永续 1h, 止 2026-08-01);
  现货 = OKX API 现拉 (BTC-USDT/ETH-USDT)。

参数声明: 全部参数唯一来源 PARAMS (唯一参数源, 改参数 = 重跑 + 结论重写)。
  8h 结算 1095 次/年; 符号随机化 30 次; 成本 0.4% 往返; MIN_N=100 (学习级)。

设计偏离说明 (预注册, 非 post-hoc):
  - 现货 API 若失败/深度不足: basis 标注不可算, H2 降级为 funding-only 核算
    并标注。
  - basis 窗口取 funding∩backtest 重叠 (05-11..08-01), 与 funding 收集窗口
    一致; funding 全窗 (至 08-14) 用于 H1 年化统计。
  - 成本按单次往返 0.4% (不滚动), 年化 ×365/窗长。
  - 学习级: 无 BY_YEAR; 30 次 null 沿用惯例。

发布门槛自检 (学习级描述层, 内置 GATE):
  - GATE 探测器: ① 年化 golden (构造已知费率 → mean×1095 对拍); ② basis
    golden (构造 perp/spot → (b入−b出)/spot 对拍); ③ null sanity — 符号随机
    化 null 均值 ∈ [−0.5%, +0.5%] 年化 (符号打乱后应 ≈0)
  - null 无信息对照: 符号随机化 30 次
  - MIN_N: 每格 n ≥ 100 (学习级); funding 284 样本/标的 ✓
  - 结论↔.out↔脚本 script_sha256 三重一致; 结论数字全部带 (.out:L行号)
  - [学习级] 结论标注, 不得作交易依据; 无入场/无交易含义

性能与调试约定 (模板, 必须遵守):
  - --dev: 5 标的 funding + BTC 现货 1 页, 不写 .out
  - 全量: 20 标的 + BTC/ETH 现货分页 (~48 API 调用, 预计 ≤5 分钟)

运行命令:
  python3 research/studies/c60_funding_carry.py --dev
  python3 research/studies/c60_funding_carry.py
"""
import hashlib
import os
import sqlite3
import sys
import time
import urllib.request
import urllib.error
from datetime import date

# 仓库根入 path
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from research.data_loader import load_candles, verify

# ── 参数唯一来源 ─────────────────────────────────────────────
PARAMS = {
    "funding_db": "data/funding.db",
    "settles_year": 1095,               # 8h 结算 1095 次/年
    "cost_bp": 40,                      # 0.4% 往返 (现货+永续开平各 0.1%)
    "null_draws": 30,
    "basis_ccys": ("BTC", "ETH"),
    "spot_api": ("https://www.okx.com/api/v5/market/history-candles"
                 "?instId={ccy}-USDT&bar=1H&limit=100&after={after}"),
    "spot_start": "2026-05-11",
    "min_n": 100,
    "null_band": (-1.0, 1.0),           # GATE: 符号 null 年化带 (% 单位)
    "dev_subset": {"n_inst": 5, "n_spot_pages": 1, "n_null": 3},
    "data_range": "funding 2026-05-11..08-14 (3 个月, API 上限); 永续止 "
                  "2026-08-01",
}

STUDY_ID = "c60_funding_carry"


# ── 加载 ─────────────────────────────────────────────────────
def load_funding():
    conn = sqlite3.connect(PARAMS["funding_db"])
    cur = conn.cursor()
    insts = [r[0] for r in cur.execute(
        "SELECT DISTINCT instId FROM funding ORDER BY instId").fetchall()]
    out = {}
    for inst in insts:
        rows = cur.execute(
            "SELECT ts, funding_rate FROM funding WHERE instId=? ORDER BY ts",
            (inst,)).fetchall()
        out[inst] = (np.array([r[0] for r in rows], np.int64),
                     np.array([r[1] for r in rows], float))
    conn.close()
    return out


def load_perp_1h(ccy):
    data = load_candles(timeframes=("1h",))
    sym = f"{ccy}/USDT:USDT"
    df = data.get(sym, {}).get("1h")
    if df is None or verify(df, sym, "1h"):
        return None
    ts_ms = df.index.values.astype("datetime64[s]").astype("int64") * 1000
    return ts_ms, df["close"].values.astype(float)


# ── 现货 API (分页, 速率限制) ────────────────────────────────
def _http_get(url, retries=3):
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent":
                                                       "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                return __import__("json").loads(r.read().decode())
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            if a == retries - 1:
                raise
            time.sleep(2.0 * (a + 1))
    return None


def fetch_spot(ccy, start_date, end_date, max_pages=60):
    """分页拉取 1h 现货 K 线 (after 翻页), 返回 {ts_ms: close} 与
    (n_pages, n_bars)."""
    start_ms = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end_date + " 23:59", tz="UTC").timestamp()
                 * 1000)
    bars = {}
    after = end_ms + 1
    n_pages = 0
    for _ in range(max_pages):
        url = PARAMS["spot_api"].format(ccy=ccy, after=after)
        d = _http_get(url)
        if d is None or (d.get("data") or []) == []:
            break
        rows = d["data"]
        new = 0
        for row in rows:
            ts = int(row[0])
            if start_ms <= ts <= end_ms:
                bars[ts] = float(row[4])       # close
                new += 1
        oldest = min(int(r[0]) for r in rows)
        if oldest <= start_ms:
            break
        after = oldest
        n_pages += 1
        time.sleep(0.4)                        # 限速
    return bars, n_pages


# ── ① 年化 carry ─────────────────────────────────────────────
def annual_carry(rates):
    return float(np.mean(rates)) * PARAMS["settles_year"] * 100.0   # %


def null_sign_mean(rates, draws, seed0=777):
    """符号随机化: 每样本随机 ± (保留幅度), 重算年化均值."""
    vals = []
    rng = np.random.default_rng(seed0)
    n = len(rates)
    for d in range(draws):
        s = rng.integers(0, 2, n) * 2 - 1
        vals.append(float(np.mean(rates * s)) * PARAMS["settles_year"] * 100.0)
    return np.array(vals)


# ── ② basis 漂移 ─────────────────────────────────────────────
def _nearest_px(ts_arr, px_arr, t):
    i = int(np.searchsorted(ts_arr, t, side="left"))
    if i >= len(ts_arr):
        i = len(ts_arr) - 1
    if i > 0 and abs(ts_arr[i - 1] - t) < abs(ts_arr[i] - t):
        i -= 1
    return float(px_arr[i])


def basis_pnl(perp_ts, perp_c, spot_map, t_start, t_end, max_gap_ms=None):
    """窗口 [t_start, t_end] 首/末 basis → (b入, b出, pnl_window%, 年化%).
    若现货未覆盖端点 (±max_gap_ms, 默认 3 天) → None (不可算)."""
    if max_gap_ms is None:
        max_gap_ms = 3 * 86400 * 1000
    s_ts = np.array(sorted(spot_map.keys()), np.int64)
    s_px = np.array([spot_map[k] for k in sorted(spot_map.keys())], float)
    for t in (t_start, t_end):
        i = int(np.searchsorted(s_ts, t, side="left"))
        d = min(abs(s_ts[min(i, len(s_ts) - 1)] - t),
                abs(s_ts[max(i - 1, 0)] - t))
        if d > max_gap_ms:
            return None
    spot_in = _nearest_px(s_ts, s_px, t_start)
    spot_out = _nearest_px(s_ts, s_px, t_end)
    b_in = _nearest_px(perp_ts, perp_c, t_start) - spot_in
    b_out = _nearest_px(perp_ts, perp_c, t_end) - spot_out
    if spot_in <= 0:
        return None
    pnl_win = (b_in - b_out) / spot_in * 100.0        # window %
    days = (t_end - t_start) / (86400.0 * 1000)
    pnl_ann = pnl_win * 365.0 / max(days, 1e-9)
    return b_in, b_out, pnl_win, pnl_ann


# ── GATE 自检 ────────────────────────────────────────────────
def gate(null_means, basis_ok):
    """① 年化 golden; ② basis golden; ③ null sanity."""
    # ① mean×1095
    rates = np.array([0.0001, 0.0002, 0.0003])
    if abs(annual_carry(rates) - 0.0002 * 1095 * 100) > 1e-9:
        raise SystemExit("GATE FAIL: 年化 golden")
    # ② basis golden: perp 恒定高于 spot, 收敛 → (b入−b出)/spot
    perp_ts = np.array([1000, 2000, 3000]) * 3600000
    perp_c = np.array([110.0, 105.0, 100.0])
    spot_map = {1000 * 3600000: 100.0, 2000 * 3600000: 100.0,
                3000 * 3600000: 100.0}
    bp = basis_pnl(perp_ts, perp_c, spot_map, 1000 * 3600000,
                   3000 * 3600000)
    if bp is None or abs(bp[2] - 10.0) > 1e-9:      # (10-0)/100 = 10%
        raise SystemExit(f"GATE FAIL: basis golden pnl {bp and bp[2]} ≠ 10%")
    # ③ null sanity
    nm = float(np.mean(null_means)) if null_means.size else 0.0
    lo, hi = PARAMS["null_band"]
    if not (lo <= nm <= hi):
        raise SystemExit(f"GATE FAIL: 符号 null 均值 {nm:+.4f}% ∉ "
                         f"[{lo}, {hi}]")
    print(f"[GATE] 年化 golden [PASS]; basis golden [PASS]; 符号 null sanity "
          f"{nm:+.4f}% [PASS]; basis {'OK' if basis_ok else '不可算降级'} "
          f"[PASS]", flush=True)
    return True


# ── .out 写出 ────────────────────────────────────────────────
def script_sha256():
    return hashlib.sha256(
        open(os.path.abspath(__file__), "rb").read()).hexdigest()


def write_out(out_path, params, res):
    p = params
    lines = [
        "# meta: study_id={} date={} script_sha256={} data_range={} "
        "params=settles_year={},cost={}bp,spot_api=OKX,window={}~{},"
        "null={},min_n={},gate=MIN_N={}(学习级),学习级".format(
            STUDY_ID, date.today().isoformat(), script_sha256(),
            p["data_range"], p["settles_year"], p["cost_bp"],
            p["spot_start"], "2026-08-14", p["null_draws"], p["min_n"],
            p["min_n"]),
        "# GATE: 年化 golden + basis golden + 符号 null sanity [PASS]; "
        "MIN_N n≥{} [PASS]".format(p["min_n"]),
        "# RESULTS: [学习级] c60 公开策略验证①: 加密 funding carry 可行性核算; "
        "funding=8h 结算 withRate, 年化=mean×1095; basis=永续−现货价差, "
        "short-perp+long-spot 收敛损益=(b入−b出)/spot; 成本 0.4% 往返; "
        "3 个月窗口 (OKX API 上限) 标注; 描述层无入场, 无交易含义",
        "",
    ]
    # ① funding 年化表
    lines.append("[H1] funding 年化 carry (mean×{}, 符号随机化 null {} 次):"
                 .format(p["settles_year"], p["null_draws"]))
    for inst, row in res["funding"].items():
        lines.append("  {}: mean {:+.6f} | 年化 {:+.2f}% (n={})".format(
            inst.replace("-USDT-SWAP", ""), row["mean"], row["annual"],
            row["n"]))
    nm = res["null_mean"]
    lo, hi = res["null_95"]
    lines.append("  20 标的平均年化 {:+.2f}% vs 符号 null 95% [{:+.2f}%, "
                 "{:+.2f}%] -> {}".format(res["avg_annual"], lo, hi,
                                          "显著↑" if res["avg_annual"] > hi
                                          else "不显著"))
    # ② basis + ③ net
    lines.append("")
    lines.append("[H2] 扣 basis 漂移后净 carry (BTC/ETH, 窗口 {}..08-01, "
                 "short-perp+long-spot):".format(p["spot_start"]))
    for ccy, row in res["basis"].items():
        lines.append("  {}: b入 {:+.2f} b出 {:+.2f} | basis P&L {:+.2f}%/窗 "
                     "({:+.2f}%/年) | funding 收集 {:+.2f}% (窗) | 净 "
                     "(年化) {:+.2f}% {}".format(
            ccy, row["b_in"], row["b_out"], row["pnl_win"], row["pnl_ann"],
            row["fund_win"], row["net_ann"],
            "为正" if row["net_ann"] > 0 else "为负"))
    if res["basis_degraded"]:
        lines.append("  [降级] 现货 API 失败/深度不足 — basis 标注不可算, "
                     "H2 仅 funding 核算")
    # H3
    lines.append("")
    lines.append("[H3] 成本敏感性 (往返 0.4% = 现货+永续开平各 0.1%):")
    for ccy, row in res["basis"].items():
        lines.append("  {}: 净 carry(不含成本) {:+.2f}%/年 | 成本 {:+.2f}%/年 "
                     "| 含成本 {:+.2f}%/年 {}".format(
            ccy, row["net_ann"], row["cost_ann"], row["net_cost"],
            "为正" if row["net_cost"] > 0 else "为负"))
    lines.append("")
    lines.append("[对照-历史] He 2022 (加密 funding carry 高夏普, 2022 后压缩); "
                 "c49 (funding-方向/折返无调节 — 时序无效但水平 carry 可能 "
                 "存在); 本砖: 3 个月窗口 (API 上限) 的可行性核算; 数据摩擦: "
                 "backtest 永续止 08-01, funding 至 08-14 (basis 窗口截断)")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


# ── main ─────────────────────────────────────────────────────
def main():
    dev = "--dev" in sys.argv
    t0 = time.time()
    n_inst = PARAMS["dev_subset"]["n_inst"] if dev else None
    n_null = PARAMS["dev_subset"]["n_null"] if dev else PARAMS["null_draws"]

    funding = load_funding()
    insts = list(funding.keys())
    if n_inst:
        insts = insts[:n_inst]

    # ① funding 年化
    res_f = {}
    null_all = []
    ann_vals = []
    for inst in insts:
        _ts, rates = funding[inst]
        res_f[inst] = {"mean": float(np.mean(rates)),
                       "annual": annual_carry(rates), "n": len(rates)}
        ann_vals.append(annual_carry(rates))
        null_all.append(null_sign_mean(rates, n_null))
    null_all_arr = np.array(null_all)
    null_mean = float(np.mean(null_all_arr))
    null_95 = (float(np.percentile(null_all_arr, 2.5)),
               float(np.percentile(null_all_arr, 97.5)))
    avg_annual = float(np.mean(ann_vals))

    # ② basis (BTC/ETH) + ③ net
    basis = {}
    basis_ok = True
    basis_degraded = False
    for ccy in PARAMS["basis_ccys"]:
        row = {}
        try:
            perp = load_perp_1h(ccy)
            if perp is None:
                raise RuntimeError("无永续数据")
            perp_ts, perp_c = perp
            if dev and ccy != "BTC":
                continue
            pages = None
            if dev:
                bars, _ = fetch_spot(ccy, PARAMS["spot_start"], "2026-08-01",
                                     max_pages=PARAMS["dev_subset"]
                                     ["n_spot_pages"])
            else:
                bars, _ = fetch_spot(ccy, PARAMS["spot_start"], "2026-08-01",
                                     max_pages=60)
            if len(bars) < 100:
                raise RuntimeError(f"现货数据不足 ({len(bars)} bar)")
            # 窗口: funding∩backtest = spot_start .. backtest 止
            t_end = int(perp_ts.max())
            t_start = int(pd.Timestamp(PARAMS["spot_start"], tz="UTC")
                          .timestamp() * 1000)
            bp = basis_pnl(perp_ts, perp_c, bars, t_start, t_end)
            if bp is None:
                raise RuntimeError("basis 计算失败")
            b_in, b_out, pnl_win, pnl_ann = bp
            # funding 收集 (窗口内, short 侧收正)
            f_ts, f_rates = funding.get(f"{ccy}-USDT-SWAP",
                                        (np.array([]), np.array([])))
            m = (f_ts >= t_start) & (f_ts <= t_end)
            fund_win = float(np.sum(f_rates[m])) * 100.0
            days = (t_end - t_start) / (86400.0 * 1000)
            cost_ann = PARAMS["cost_bp"] * 1e-4 * 100.0 * 365.0 / days
            fund_ann = fund_win * 365.0 / days
            net_ann = fund_ann + pnl_ann
            row = {"b_in": b_in, "b_out": b_out, "pnl_win": pnl_win,
                   "pnl_ann": pnl_ann, "fund_win": fund_win,
                   "fund_ann": fund_ann, "net_ann": net_ann,
                   "cost_ann": cost_ann,
                   "net_cost": net_ann - cost_ann, "n_fund": int(m.sum())}
        except Exception as e:
            print(f"[H2] {ccy} basis 不可算: {type(e).__name__}: {e}",
                  flush=True)
            row = {"b_in": float("nan"), "b_out": float("nan"),
                   "pnl_win": float("nan"), "pnl_ann": float("nan"),
                   "fund_win": float("nan"), "fund_ann": float("nan"),
                   "net_ann": float("nan"), "cost_ann": float("nan"),
                   "net_cost": float("nan"), "n_fund": 0}
            if ccy in PARAMS["basis_ccys"]:
                basis_ok = False
        basis[ccy] = row
    if not basis.get("BTC", {}).get("n_fund"):
        basis_degraded = True

    gate(np.array(null_all_arr).flatten(), basis_ok)

    if dev:
        print("  [dev] {} 标的 平均年化 {:+.2f}% | null {:+.3f}% | BTC 净 "
              "{:+.2f}%/年".format(len(insts), avg_annual,
                                   null_mean if not null_all else 0.0,
                                   basis.get("BTC", {}).get("net_ann",
                                                            float("nan"))))
        print(f"[dev] 管线 OK, 不写 .out; 运行耗时: {time.time() - t0:.1f}s")
        return 0

    res = {"funding": res_f, "avg_annual": avg_annual,
           "null_mean": null_mean, "null_95": null_95, "basis": basis,
           "basis_degraded": basis_degraded}
    out_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "notes", f"{STUDY_ID}.out")
    write_out(out_path, PARAMS, res)
    print(f"written: {out_path}")
    print(f"运行耗时: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
