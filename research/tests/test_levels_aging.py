#!/usr/bin/env python3
"""levels R1/R2 锁测试

R1: active_levels 快照 — 追加数据后历史 t 的 age_bars/last_touch_idx 不变
    (Level 事件日志 + bisect 重建; 形成后字段不可变)
R2: level_breakdown attempt/confirmed 按 confirm_at 门控 — 位带形成前不计
"""
import numpy as np
import pandas as pd
import pytest

from research.levels import Level, active_levels, cluster_levels, level_breakdown


def mk(closes, spikes, atr_val=1.0, hi_margin=0.5, lo_margin=0.5):
    n = len(closes)
    highs = [c + hi_margin for c in closes]
    lows = [c - lo_margin for c in closes]
    for i, (hi, lo) in (spikes or {}).items():
        if hi is not None:
            highs[i] = max(highs[i], hi)
        if lo is not None:
            lows[i] = min(lows[i], lo)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"open": [closes[0]] + list(closes[:-1]),
                         "high": highs, "low": lows, "close": closes,
                         "volume": np.ones(n)}, index=idx), atr_val


# ---------------------------------------------------------------- R2 门控
def test_breakdown_gated_by_confirm_at_support():
    """confirm_at=30 的支撑位: t<30 越带外侧 → attempt/confirmed 为 False"""
    lv = Level(100.0, "support", 2, 0, -1, 0.5, "cluster", 30)
    n = 60
    close = np.full(n, 101.0)
    atr = np.full(n, 2.0)
    close[5:12] = 95.0    # 位带形成前穿透 (outside: <99.5; attempt 深度: <98.5)
    close[25:30] = 95.0
    close[35:] = 95.0     # 形成后穿透并保持外侧 → 确认
    attempt, confirmed, outside, ratio = level_breakdown(lv, close, atr, 0.5, 24, 0.5)
    for t in (5, 10, 25, 28):
        assert not attempt[t], f"t={t} (confirm_at 前) attempt 应为 False"
        assert not confirmed[t], f"t={t} (confirm_at 前) confirmed 应为 False"
        assert outside[t], f"t={t} outside 是纯描述, 应保持 True"
    assert attempt[35] and confirmed[35]  # 形成后真突破照常判定


def test_breakdown_gated_by_confirm_at_resistance():
    lv = Level(100.0, "resistance", 2, 0, -1, 0.5, "cluster", 30)
    n = 60
    close = np.full(n, 99.0)
    atr = np.full(n, 2.0)
    close[5:12] = 105.0   # 位带形成前穿透 (outside: >100.5; attempt 深度: >101.5)
    close[35:] = 105.0
    attempt, confirmed, _, _ = level_breakdown(lv, close, atr, 0.5, 24, 0.5)
    assert not attempt[5] and not confirmed[5]
    assert attempt[35] and confirmed[35]


def test_touch_class_gated_via_breakdown():
    """level_touch_class 经 level_breakdown 继承门控: 形成前无穿透/确认事件"""
    from research.levels import level_touch_class
    lv = Level(100.0, "support", 2, 0, -1, 0.5, "cluster", 30)
    n = 60
    close = np.full(n, 101.0)
    high = np.full(n, 102.0)
    low = np.full(n, 101.5)
    atr = np.full(n, 2.0)
    low[10] = 99.0
    close[10] = 95.0     # t=10 < confirm_at: 触碰+穿透 → 不产生 attempt
    low[40:55] = 99.0
    close[40:55] = 95.0  # t>=confirm_at: 触碰+穿透保持 → confirmed
    res = level_touch_class(lv, close, high, low, atr, 0.5, 24, 0.5)
    assert 10 not in res["attempt"] and 10 not in res["confirmed"]
    assert 10 not in res["false_break"]
    assert 40 in res["attempt"] and 40 in res["confirmed"]


# ---------------------------------------------------------------- R1 快照
def _band(levels, t, price=105.65):
    act = active_levels(levels, t)
    hit = [l for l in act if l.side == "resistance" and abs(l.price - price) < 1e-9]
    assert len(hit) == 1, f"t={t} 位带未找到/不唯一: {[(l.price, l.side, l.confirm_at) for l in act]}"
    return hit[0]


def test_active_levels_snapshot_append_invariance():
    """追加数据后历史 t 快照不变 (R1 锁): 晚于 t 确认的触碰不改变 t 的 age_bars"""
    n = 30
    closes = [100.0] * n
    # 105.5@3(确认6) + 105.8@9(确认12) → 聚类形成@12, price=105.65
    # 105.9@15(确认18) → 并入已形成带 (R1 只进事件日志); 106.0@21 距带 0.35>tol 不入
    spikes = {3: (105.5, None), 9: (105.8, None), 15: (105.9, None), 21: (106.0, None)}
    df, atr = mk(closes, spikes)
    full = cluster_levels(df["high"].values, df["low"].values,
                          np.full(n, atr), min_touch=2)
    # 形成@12, 事件日志 [(6,3), (12,9), (18,15)]
    assert _band(full, 30).confirm_at == 12
    for cut in (14, 20, 26):
        sub = df.iloc[:cut]
        cut_lv = cluster_levels(sub["high"].values, sub["low"].values,
                                np.full(cut, atr), min_touch=2)
        for t in range(12, cut):
            a = _band(full, t)
            b = _band(cut_lv, t)
            assert a.age_bars == b.age_bars, f"cut={cut} t={t}: {a.age_bars} vs {b.age_bars}"
            assert a.last_touch_idx == b.last_touch_idx, f"cut={cut} t={t}: last_touch 不一致"
    # 手算快照值 (最近 confirm<=t 的事件 pos):
    assert _band(full, 13).age_bars == 13 - 9      # 事件(12,9) 为最近触碰 (t<18)
    assert _band(full, 16).age_bars == 16 - 9      # 105.9@15 确认@18, t=16 时不可用
    assert _band(full, 18).age_bars == 18 - 15     # 事件(18,15) 已可用
    assert _band(full, 19).age_bars == 19 - 15


def test_level_fields_immutable_after_formation():
    """R1: 形成后字段不可变 (frozen) — 追加触碰只进事件日志"""
    n = 30
    closes = [100.0] * n
    spikes = {3: (105.5, None), 9: (105.8, None), 15: (105.9, None)}
    df, atr = mk(closes, spikes)
    lvls = cluster_levels(df["high"].values, df["low"].values,
                          np.full(n, atr), min_touch=2)
    band = [l for l in lvls if l.side == "resistance"][0]
    with pytest.raises(Exception):
        band.price = 999.0
    with pytest.raises(Exception):
        band.last_touch_idx = 0
    # 事件日志记录了全部触碰 (含形成后并入的 105.9@15)
    assert [c for c, _ in band._touches] == [6, 12, 18]
    # 静态字段仍反映形成时刻 (last_touch_idx=9, 不随追加数据突变)
    assert band.last_touch_idx == 9 and band.confirm_at == 12
