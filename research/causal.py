#!/usr/bin/env python3
"""因果模式库 — 所有"当特征用"的结构统计唯一出口 (research/PLAN.md §2)

全部函数无未来函数 (只回看过去已收盘数据), 且满足"追加不变性":
往序列后面追加数据, 历史位置的值必须不变。研究脚本禁止自写替代实现
(check_study L3 AST 黑名单), 任何统计特征必须从这里走。

- rolling_percentile : 滚动分位 (禁全样本分位 — B4e 教训)
- rolling_rank       : 滚动百分位排位
- causal_confirmed   : 事后标签 (confirmed) 的因果可用版 — 条件化唯一出口
- frozen_cluster     : 在线聚类 + 冻结通用封装 (levels.cluster_levels 泛化)
- align_events       : 因果窗口查询 (bisect 封装)
"""
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field

import numpy as np


def rolling_percentile(x, window, q, min_periods=None) -> np.ndarray:
    """滚动分位: x[i] = 过去 window 根 (含自身) 的 q 分位 (线性插值)

    - 窗口 = 左对齐尾窗 x[max(0, i-window+1) : i+1], 只含过去数据
    - 窗口内 NaN 剔除; 有效值数 < min_periods → NaN
    - min_periods 默认 = window (前 window-1 根为 NaN)
    - 追加数据不改变历史值 (左对齐尾窗性质, 测试锁定)
    """
    x = np.asarray(x, float)
    n = len(x)
    window = int(window)
    if min_periods is None:
        min_periods = window
    min_periods = int(min_periods)
    out = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - window + 1)
        win = x[lo:i + 1]
        valid = win[~np.isnan(win)]
        if valid.size < min_periods:
            continue
        out[i] = np.percentile(valid, float(q) * 100.0)
    return out


def rolling_rank(x, window) -> np.ndarray:
    """滚动升序百分位排位: x[i] 在过去 window 根 (含自身) 中的百分位 ∈ [0,1]

    - 平均秩 (fractional ranking): 并列取平均秩; 最小值→0, 最大值→1
    - 需要完整 window 根 (前 window-1 根为 NaN)
    - 窗口内 NaN 剔除 (有效值数 < 2 → NaN); x[i] 本身为 NaN → NaN
    - 追加数据不改变历史值 (测试锁定)
    """
    x = np.asarray(x, float)
    n = len(x)
    window = int(window)
    out = np.full(n, np.nan)
    if n < window or window < 2:
        return out
    from numpy.lib.stride_tricks import sliding_window_view

    m = n - window + 1          # 完整窗口数
    valid_start = window - 1    # 第一个有完整窗口的位置
    win = sliding_window_view(x, window)  # win[k] = x[k : k+window] = 位置 k+window-1 的尾窗
    chunk = 20000
    for s in range(0, m, chunk):
        e = min(s + chunk, m)
        w = win[s:e]                       # (rows, window) 尾窗矩阵
        xi = x[valid_start + np.arange(s, e)]
        valid_mask = ~np.isnan(w)
        n_valid = valid_mask.sum(axis=1)
        ok = n_valid >= 2
        wv = np.where(valid_mask, w, np.nan)
        less = np.sum(wv < xi[:, None], axis=1)          # NaN 比较恒 False
        eq = np.sum(wv == xi[:, None], axis=1)
        mean_rank = less + (eq + 1) / 2.0
        pct = (mean_rank - 1.0) / np.maximum(n_valid - 1, 1)
        res = np.where(ok & ~np.isnan(xi), pct, np.nan)
        out[valid_start + np.arange(s, e)] = res
    return out


def causal_confirmed(confirmed, w, lag_lo=0, lag_hi=60, confirm_cost=0):
    """事后标签 (confirmed) 的因果可用版 — 条件化唯一出口

    known[t] = 1 ⟺ ∃ c ∈ [t-lag_hi, t-lag_lo] 满足 confirmed[c]
                    且 c + confirm_cost + w <= t  (确认窗口完全收在 t 之前)

    默认 (lag_lo=0, lag_hi=60, confirm_cost=0): conf ∈ [t-60, t-24] 可用;
    [t-23, t] 内刚确认/突破的样本剔除 (B2c/B2d 泄漏的正确替代)。

    confirmed: 长度 n 的布尔 (事后标签)。w: 确认窗口长度。
    返回 (known: bool[n], usable_idxs: list[int]) —
      known[t] 为布尔序列; usable_idxs = 确认窗口在数据内完全关闭的事件索引
      (c + confirm_cost + w <= n-1), 升序。
    """
    confirmed = np.asarray(confirmed, bool)
    n = len(confirmed)
    if not (0 <= lag_lo <= lag_hi) or w < 0 or confirm_cost < 0:
        raise ValueError(f"非法参数: lag_lo={lag_lo} lag_hi={lag_hi} w={w} confirm_cost={confirm_cost}")
    prefix = np.concatenate([[0], np.cumsum(confirmed)])
    t = np.arange(n)
    # c 的上界 = min(窗口上界 t-lag_lo, 确认完成上界 t-confirm_cost-w)
    c_max = np.minimum(t - lag_lo, t - confirm_cost - w)
    c_min = t - lag_hi
    lo = np.maximum(c_min, 0)
    hi = np.minimum(c_max, n - 1)
    valid = hi >= lo
    cnt = np.zeros(n, int)
    cnt[valid] = prefix[hi[valid] + 1] - prefix[lo[valid]]
    known = cnt > 0
    usable = [int(c) for c in np.flatnonzero(confirmed)
              if c + confirm_cost + w <= n - 1]
    return known, usable


@dataclass
class FrozenGroup:
    """冻结聚类组 — 形成后 value/confirm_at 不可变

    - value     : 形成时刻 (第 min_touch 个事件 confirm_at) 的组内中位数, 冻结
    - confirm_at: 组形成时刻 = 促成形成的第 min_touch 个事件的 confirm_at
    - n_touch   : 并入该组的事件总数 (含形成前成员 + 形成后 touches)
    - touches   : 形成后并入的同组事件日志 [(confirm_at, value), ...]
    """
    key: object
    value: float
    confirm_at: int
    n_touch: int = 0
    touches: list = field(default_factory=list)


def frozen_cluster(events, tol_fn, min_touch) -> list:
    """在线聚类 + 冻结通用封装 (levels.cluster_levels 泛化, 研究脚本禁自写)

    events: 按 confirm_at 升序的 (confirm_at, value, key) 流
    tol_fn : tol_fn(confirm_at) -> 该时刻容差 (容忍随事件时刻变化, 如 atr×mult)
    min_touch: 形成组所需事件数

    语义:
      - 新事件先找已形成组 (key 相同且 |value - group.value| <= tol, value 为冻结值)
        → 只并入 touches 日志, 不改 value/confirm_at
      - 否则并入未形成组 (按当前组中位数比较); 达到 min_touch 即形成:
        value = 当时中位数, confirm_at = 促成事件时刻, 之后冻结
      - 追加数据不改变历史组集合 (value/confirm_at 不变; touches 只增前缀) — 测试锁定

    返回 list[FrozenGroup], 按形成时刻升序 (即事件处理顺序)。
    """
    formed: list = []
    pending: list = []  # [key, values, last_confirm]

    for confirm, value, key in events:
        tol = tol_fn(confirm)
        # 1) 已形成组: 与冻结值比较
        g = None
        for grp in formed:
            if grp.key == key and abs(float(value) - grp.value) <= tol:
                g = grp
                break
        if g is not None:
            g.touches.append((int(confirm), float(value)))
            g.n_touch += 1
            continue
        # 2) 未形成组: 与当前中位数比较
        merged = False
        for grp in pending:
            if grp[0] == key and abs(float(value) - float(np.median(grp[1]))) <= tol:
                grp[1].append(float(value))
                grp[2] = int(confirm)
                if len(grp[1]) >= min_touch:
                    formed.append(FrozenGroup(
                        key=key,
                        value=float(np.median(grp[1])),
                        confirm_at=int(confirm),
                        n_touch=len(grp[1]),
                    ))
                    pending.remove(grp)
                merged = True
                break
        if not merged:
            pending.append([key, [float(value)], int(confirm)])
    return formed


def align_events(event_positions, t, lag_lo, lag_hi) -> np.ndarray:
    """因果窗口查询: 事件位置排序数组按窗口 [t-lag_hi, t-lag_lo] 切片 (含边界)

    bisect 封装 + 有序校验断言。返回 event_positions 在窗口内的切片 (视图)。
    """
    pos = np.asarray(event_positions)
    assert np.all(pos[1:] >= pos[:-1]), "event_positions 必须升序"
    lo = bisect_left(pos.tolist(), t - lag_hi)
    hi = bisect_right(pos.tolist(), t - lag_lo)
    return pos[lo:hi]
