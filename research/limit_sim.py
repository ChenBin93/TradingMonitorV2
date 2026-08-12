#!/usr/bin/env python3
"""limit 挂单成交模拟 (research/PLAN.md §2, B4e 语义合法化)

语义 (无未来函数, 与 outcome.py 严格口径对齐):
- 每个事件 (t_arr 中的一根) 挂一张独立限价单, 无阻塞/无重叠 (独立事件口径)
- 成交 = 当根 intrabar 触及挂单价: low[j] <= entry_px <= high[j]
  (找第一根满足的 j >= t); 成交价 = 挂单价 entry_px
- 成交后自下一根 open 起按 open 出发语义判定 (outcome.evaluate_forward 同款):
    1. open[k] 越过上界 → 越界判定
    2. open[k] 越过下界 → 越界判定
    3. 同 bar 双命中 (open 在带内, low<=lo_bound 且 high>=hi_bound) → skip
    4. 单侧触碰 → 该侧判定
- W 根内未命中 → expired; 数据内从未成交 → unfilled
- 方向由价格隐含: hi_bound == target → 胜侧 (long 型), hi_bound == stop → 负侧

返回与 outcome.py 兼容: (LimitOutcome, list[TradeRec])
  - LimitOutcome 继承 outcome.Outcome (同 win/loss/expired/skip 计数 + n_unfilled)
  - TradeRec 复用 outcome.TradeRec; outcome 字符串多一个 "unfilled"
"""
import numpy as np

from research.outcome import Outcome, TradeRec


class LimitOutcome(Outcome):
    """Outcome 超集: 增加 n_unfilled (挂单从未成交)

    n_total 把 unfilled 计入分母 (hit_rate = 成交且命中 / 全部尝试)。
    """

    def __init__(self, n_win=0, n_loss=0, n_expired=0, n_skip=0, n_unfilled=0):
        super().__init__(n_win=n_win, n_loss=n_loss, n_expired=n_expired, n_skip=n_skip)
        self.n_unfilled = n_unfilled

    @property
    def n_total(self) -> int:
        return self.n_eval + self.n_expired + self.n_skip + self.n_unfilled


def _broadcast(a, name, n):
    """标量或数组 → 长度 n 的 float 数组 (标量广播)"""
    if np.isscalar(a):
        return np.full(n, a, dtype=float)
    a = np.asarray(a, float)
    if a.ndim == 0:
        return np.full(n, a, dtype=float)
    if a.shape != (n,):
        raise ValueError(f"{name} 长度 {len(a)} != t_arr 长度 {n}")
    return a


def simulate_limit_entries(t_arr, open_px, high, low, entry_px, target, stop, w):
    """limit 挂单成交模拟 — 每事件一张独立限价单

    参数:
      t_arr   : 挂单事件 bar 索引数组 (int, 允许重复 — 每根独立一张单)
      open_px/high/low: 长度 n 的价格数组
      entry_px: 挂单价 (标量或长度 len(t_arr) 数组)
      target  : 目标价 (标量或数组)
      stop    : 止损价 (标量或数组)
      w       : 成交后判定窗口根数 (从成交 bar 下一根起算)

    返回 (LimitOutcome, list[TradeRec]):
      - win:    exit_px=target,  exit_idx=命中 bar
      - loss:   exit_px=stop,    exit_idx=命中 bar
      - skip:   exit_px=命中bar开盘价 (中性, 引用 bar open), exit_idx=命中 bar
      - expired:exit_px=NaN,     exit_idx=min(fill_j+w, n-1)
      - unfilled:entry_idx=挂单 bar, exit_idx=-1, exit_px=NaN
      entry_idx: 成交 bar (unfilled 为挂单 bar)
    """
    open_px = np.asarray(open_px, float)
    high = np.asarray(high, float)
    low = np.asarray(low, float)
    t_arr = np.asarray(t_arr)
    n = len(open_px)
    if len(high) != n or len(low) != n:
        raise ValueError("open_px/high/low 长度不一致")
    w = int(w)
    epx = _broadcast(entry_px, "entry_px", len(t_arr))
    tgt = _broadcast(target, "target", len(t_arr))
    stp = _broadcast(stop, "stop", len(t_arr))

    out = LimitOutcome()
    recs = []
    for k, t in enumerate(t_arr):
        t = int(t)
        e, tg, st = epx[k], tgt[k], stp[k]
        lo_b, hi_b = min(tg, st), max(tg, st)
        hi_is_target = (hi_b == tg)
        # 成交: 第一根 low<=entry_px<=high 的 bar (从挂单 bar 起)
        fill_j = -1
        for j in range(t, n):
            if low[j] <= e <= high[j]:
                fill_j = j
                break
        if fill_j < 0:
            out.n_unfilled += 1
            recs.append(TradeRec(t, -1, e, np.nan, "unfilled", tg, st))
            continue
        # 成交后判定: 自 fill_j+1 起 open 出发语义
        hit_j, outcome = -1, "expired"
        for j in range(fill_j + 1, min(fill_j + w + 1, n)):
            o, l, h = open_px[j], low[j], high[j]
            if o >= hi_b:      # 跳空上穿上界 → 开盘即成交
                outcome = "win" if hi_is_target else "loss"
                hit_j = j
                break
            if o <= lo_b:      # 跳空下穿下界
                outcome = "loss" if hi_is_target else "win"
                hit_j = j
                break
            if l <= lo_b and h >= hi_b:  # open 在带内, 双命中 → 中性
                outcome = "skip"
                hit_j = j
                break
            if h >= hi_b:      # 单侧触碰上界
                outcome = "win" if hi_is_target else "loss"
                hit_j = j
                break
            if l <= lo_b:      # 单侧触碰下界
                outcome = "loss" if hi_is_target else "win"
                hit_j = j
                break
        if outcome == "win":
            out.n_win += 1
            exit_px = tg
        elif outcome == "loss":
            out.n_loss += 1
            exit_px = st
        elif outcome == "skip":
            out.n_skip += 1
            exit_px = open_px[hit_j]
        else:  # expired
            out.n_expired += 1
            exit_px = np.nan
            hit_j = min(fill_j + w, n - 1)
        recs.append(TradeRec(fill_j, hit_j, e, exit_px, outcome, tg, st))
    return out, recs
