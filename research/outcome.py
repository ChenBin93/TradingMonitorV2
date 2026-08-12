#!/usr/bin/env python3
"""结果引擎 — 严格口径的双实现 (numpy 参考引擎 + vectorbt)

口径 (research/caliber.py):
- 入场 = 信号 bar 收盘价 close[i]
- TP/SL = entry ± T×ATR[i], 从 bar i+1 前向逐根判定
- bar 内判定用 open 出发语义 (2026-08-03 修正, 消除 close 基准的方向偏差):
    1. open_j 已越过上界 → 按越界判定 (跳空先成交)
    2. open_j 已越过下界 → 按越界判定
    3. 同 bar 双命中 (open 在带内, high≥TP 且 low≤SL) → skip (路径未知, 中性)
    4. 单侧触碰 → 该侧判定 (open 出发必先碰)
- W 根内未命中 → expired (不计入胜率)
- 两实现必须一致 (对拍测试), vectorbt 只负责前向执行, 分类一律由本模块重新判定

非对称口径扩展 (PLAN.md §2, C 系列 2026-08-12):
- t_target/t_stop 分别为目标/止损的 ATR 倍数; 均未给时退化为对称 t_mult×ATR
  (锁定语义不变: 1:1 对称行为与旧版逐位一致)
- 末根入场 (i+1>=n, 无前向 bar) 计入 n_truncated (数据截断), 不静默丢弃
- 引擎入口长度断言: entries/价格数组与 close 长度不一致 → ValueError (A3 类
  索引错位首次运行即死, 不再产出 .out 后才被发现)
"""
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from research.caliber import MIN_N, T, W


@dataclass
class Outcome:
    n_win: int = 0
    n_loss: int = 0
    n_expired: int = 0
    n_skip: int = 0
    n_truncated: int = 0   # 末根入场 (i+1>=n): 数据截断, 独立于 expired

    @property
    def n_eval(self) -> int:
        return self.n_win + self.n_loss

    @property
    def n_total(self) -> int:
        return self.n_eval + self.n_expired + self.n_skip

    @property
    def win_rate(self) -> float:
        return self.n_win / self.n_eval if self.n_eval else float("nan")

    @property
    def hit_rate(self) -> float:
        """在窗口内命中任一目标的比例 (胜率外的第二视角)"""
        return self.n_eval / self.n_total if self.n_total else float("nan")


@dataclass
class TradeRec:
    entry_idx: int
    exit_idx: int
    entry_px: float
    exit_px: float
    outcome: str  # win / loss / expired / skip
    tp: float = 0.0
    sl: float = 0.0


def _targets(close_i, atr_i, t_target, t_stop, direction):
    """目标/止损价 (tp, sl) — 非对称口径: tp=t_target×ATR, sl=t_stop×ATR

    对称 1:1 时 t_target == t_stop == t_mult, 与旧语义逐位一致。
    """
    if direction == "long":
        return close_i + atr_i * t_target, close_i - atr_i * t_stop
    return close_i - atr_i * t_target, close_i + atr_i * t_stop


def _default_open(close):
    """open 缺失时用 prev close 近似 (连续市场语义)"""
    n = len(close)
    o = np.empty(n)
    o[0] = close[0]
    o[1:] = close[:-1]
    return o


def evaluate_forward(close, high, low, atr, entries, direction="long",
                     t_mult=T, w=W, open_px=None, t_target=None, t_stop=None) -> tuple[Outcome, list[TradeRec]]:
    """numpy 参考引擎 — 前向逐根先碰判定 (open 出发语义, 严格口径)

    t_target/t_stop: 非对称口径的目标/止损 ATR 倍数; 均 None 时退化为
    对称 t_mult×ATR (锁定语义不变)。末根入场计入 n_truncated。
    """
    close = np.asarray(close, float)
    high = np.asarray(high, float)
    low = np.asarray(low, float)
    atr = np.asarray(atr, float)
    entries = np.asarray(entries, bool)
    n = len(close)
    if len(entries) != n:
        raise ValueError(
            "entries 与 close 长度不一致: 研究脚本禁止自行切片对齐，"
            "请用 research.ctx.make_ctx 声明 warmup")
    if len(high) != n or len(low) != n or len(atr) != n:
        raise ValueError(
            "high/low/atr 与 close 长度不一致: 研究脚本禁止自行切片对齐，"
            "请用 research.ctx.make_ctx 声明 warmup")
    if open_px is None:
        open_px = _default_open(close)
    open_px = np.asarray(open_px, float)
    if len(open_px) != n:
        raise ValueError(
            "open_px 与 close 长度不一致: 研究脚本禁止自行切片对齐，"
            "请用 research.ctx.make_ctx 声明 warmup")
    if t_target is None:
        t_target = t_mult
    if t_stop is None:
        t_stop = t_mult
    out = Outcome()
    recs: list[TradeRec] = []
    for i in np.flatnonzero(entries):
        if i + 1 >= n:                      # 末根入场: 无前向 bar → 数据截断
            out.n_truncated += 1
            continue
        if not np.isfinite(close[i]) or not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        tp, sl = _targets(close[i], atr[i], t_target, t_stop, direction)
        lo_bound, hi_bound = min(tp, sl), max(tp, sl)
        hit_j, outcome = -1, "expired"
        for j in range(i + 1, min(i + w + 1, n)):
            o, l, h = open_px[j], low[j], high[j]
            if o >= hi_bound:      # 跳空上穿上界 → 开盘即成交
                outcome = "win" if hi_bound == tp else "loss"
                hit_j = j
                break
            if o <= lo_bound:      # 跳空下穿下界 → 开盘即成交
                outcome = "loss" if lo_bound == sl else "win"
                hit_j = j
                break
            if l <= lo_bound and h >= hi_bound:  # open 在带内, 双命中 → 路径未知, 中性跳过
                outcome = "skip"
                hit_j = j
                break
            if h >= hi_bound:      # 单侧触碰上界 (open 出发必先碰)
                outcome = "win" if hi_bound == tp else "loss"
                hit_j = j
                break
            if l <= lo_bound:      # 单侧触碰下界
                outcome = "loss" if lo_bound == sl else "win"
                hit_j = j
                break
        exit_px = {"win": tp, "loss": sl}.get(outcome, close[min(hit_j, n - 1)] if hit_j >= 0 else np.nan)
        if hit_j < 0:
            exit_idx = min(i + w, n - 1)
        else:
            exit_idx = hit_j
        if outcome == "win":
            out.n_win += 1
        elif outcome == "loss":
            out.n_loss += 1
        elif outcome == "expired":
            out.n_expired += 1
        else:
            out.n_skip += 1
        recs.append(TradeRec(i, exit_idx, close[i], exit_px, outcome, tp, sl))
    return out, recs


def evaluate_forward_vbt(close, high, low, atr, entries, direction="long",
                         t_mult=T, w=W, open_px=None, chunk=64,
                         t_target=None, t_stop=None) -> tuple[Outcome, list[TradeRec]]:
    """vectorbt 实现 — 前向执行交给 vectorbt, 分类按严格口径重新判定

    每个入场占一个独立 column 模拟 (column-per-entry):
    - 每个 column 只有一次入场, 天然无阻塞/无重叠, 与 numpy 独立事件口径一致
    - 分块执行控制内存 (n × chunk 列)
    - 时间退出: 入场后第 w 根平仓 (与 numpy 窗口语义对齐)
    - win/loss/expired/skip 一律由本模块重新分类 (不信任 vectorbt 内部统计)
    - 跳空时 vectorbt 按开盘价成交: 胜时更优, 负时更差 (numpy 保守按目标价)

    非对称口径: t_target/t_stop 分别驱动 tp_stop/sl_stop (百分比 = ATR×倍数/close),
    与 numpy 引擎同族对拍语义; 均 None 时退化为对称 t_mult。
    末根入场计入 n_truncated。
    """
    import vectorbt as vbt

    close = np.asarray(close, float)
    high = np.asarray(high, float)
    low = np.asarray(low, float)
    atr = np.asarray(atr, float)
    entries = np.asarray(entries, bool)
    n = len(close)
    if len(entries) != n:
        raise ValueError(
            "entries 与 close 长度不一致: 研究脚本禁止自行切片对齐，"
            "请用 research.ctx.make_ctx 声明 warmup")
    if len(high) != n or len(low) != n or len(atr) != n:
        raise ValueError(
            "high/low/atr 与 close 长度不一致: 研究脚本禁止自行切片对齐，"
            "请用 research.ctx.make_ctx 声明 warmup")
    if open_px is None:
        open_px = _default_open(close)
    open_px = np.asarray(open_px, float)
    if len(open_px) != n:
        raise ValueError(
            "open_px 与 close 长度不一致: 研究脚本禁止自行切片对齐，"
            "请用 research.ctx.make_ctx 声明 warmup")
    if t_target is None:
        t_target = t_mult
    if t_stop is None:
        t_stop = t_mult
    idx = pd.RangeIndex(n)
    tp_pct = np.divide(atr * t_target, close, out=np.full(n, np.nan), where=close != 0)
    sl_pct = np.divide(atr * t_stop, close, out=np.full(n, np.nan), where=close != 0)
    tp_cols = pd.DataFrame(np.tile(tp_pct[:, None], (1, chunk)), index=idx)
    sl_cols = pd.DataFrame(np.tile(sl_pct[:, None], (1, chunk)), index=idx)

    n_truncated = 0
    entry_idxs = []
    for i in np.flatnonzero(entries):
        if i + 1 >= n:                      # 末根入场: 无前向 bar → 数据截断
            n_truncated += 1
            continue
        if not np.isfinite(close[i]) or not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        entry_idxs.append(i)
    out = Outcome()
    out.n_truncated = n_truncated
    recs: list[TradeRec] = []
    for ci in range(0, len(entry_idxs), chunk):
        ids = entry_idxs[ci:ci + chunk]
        m = len(ids)
        if m < chunk:
            tp_cols = tp_cols.iloc[:, :m]
            sl_cols = sl_cols.iloc[:, :m]
        ent = pd.DataFrame(np.zeros((n, m), bool), index=idx)
        exits = pd.DataFrame(np.zeros((n, m), bool), index=idx)
        for k, i in enumerate(ids):
            ent.iloc[i, k] = True
            if i + w < n:
                exits.iloc[i + w, k] = True
        long_ent = ent if direction == "long" else pd.DataFrame(np.zeros((n, m), bool), index=idx)
        short_ent = ent if direction == "short" else pd.DataFrame(np.zeros((n, m), bool), index=idx)
        close_df = pd.DataFrame(np.tile(close[:, None], (1, m)), index=idx)
        open_df = pd.DataFrame(np.tile(open_px[:, None], (1, m)), index=idx)
        high_df = pd.DataFrame(np.tile(high[:, None], (1, m)), index=idx)
        low_df = pd.DataFrame(np.tile(low[:, None], (1, m)), index=idx)
        pf = vbt.Portfolio.from_signals(
            close=close_df, open=open_df, high=high_df, low=low_df,
            entries=long_ent, exits=exits,
            short_entries=short_ent, short_exits=exits,
            size=1.0, sl_stop=sl_cols, tp_stop=tp_cols,
        )
        seen_col: set[int] = set()
        for r in pf.trades.records.to_dict("records"):
            col = int(r["col"])
            # vectorbt 止损单部分成交伪影: 同一 column 会拆出主单 + 残留单
            # 口径是独立事件, 一个入场只算一个结果 → 只保留主单 (第一条)
            if col in seen_col:
                continue
            seen_col.add(col)
            e_i = int(r["entry_idx"])
            x_i = int(r["exit_idx"])
            status = int(r["status"])
            e_px = float(r["entry_price"])
            x_px = float(r["exit_price"]) if x_i >= 0 else np.nan
            tp, sl = _targets(close[e_i], atr[e_i], t_target, t_stop, direction)
            if status != 1 or x_i < 0 or x_i - e_i > w:
                outcome = "expired"
            else:
                o = open_px[x_i]
                lo_b, hi_b = min(tp, sl), max(tp, sl)
                l, h = low[x_i], high[x_i]
                if o >= hi_b:  # 跳空上穿上界 (与 numpy 引擎同语义)
                    outcome = "win" if direction == "long" else "loss"
                elif o <= lo_b:  # 跳空下穿下界
                    outcome = "loss" if direction == "long" else "win"
                elif l <= lo_b and h >= hi_b:
                    outcome = "skip"
                else:
                    # 跳空时 vectorbt 按开盘价成交: 用不等式判定 (gap 只会更优/更差)
                    tol = 1e-6 * max(1.0, abs(tp), abs(sl))
                    if direction == "long":
                        outcome = "win" if x_px >= tp - tol else "loss" if x_px <= sl + tol else "expired"
                    else:
                        outcome = "win" if x_px <= tp + tol else "loss" if x_px >= sl - tol else "expired"
            if outcome == "win":
                out.n_win += 1
            elif outcome == "loss":
                out.n_loss += 1
            elif outcome == "expired":
                out.n_expired += 1
            else:
                out.n_skip += 1
            recs.append(TradeRec(e_i, x_i if x_i >= 0 else min(e_i + w, n - 1),
                                 e_px, x_px, outcome, tp, sl))
    return out, recs


def wilson_ci(n_win: int, n_eval: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% CI — 胜率置信区间"""
    if n_eval == 0:
        return (float("nan"), float("nan"))
    p = n_win / n_eval
    denom = 1 + z * z / n_eval
    center = (p + z * z / (2 * n_eval)) / denom
    half = z * math.sqrt(p * (1 - p) / n_eval + z * z / (4 * n_eval * n_eval)) / denom
    return (center - half, center + half)


def report_wr(outcome: Outcome, label: str = "") -> str:
    """标准胜率报告行: 含样本量/CI/命中率, MIN_N 不足标注"""
    wr = outcome.win_rate
    lo, hi = wilson_ci(outcome.n_win, outcome.n_eval)
    head = f"{label}: n={outcome.n_eval} (win {outcome.n_win} / loss {outcome.n_loss} / 过期 {outcome.n_expired} / 跳过 {outcome.n_skip} / 截断 {outcome.n_truncated})"
    if outcome.n_eval < MIN_N:
        return f"{head} — WR {wr:.1%} ⚠样本不足({MIN_N})"
    return f"{head} — WR {wr:.1%} [CI {lo:.1%}–{hi:.1%}] 命中率 {outcome.hit_rate:.1%}"
