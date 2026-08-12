#!/usr/bin/env python3
"""dow_segments 测试 (L2 模块层) — 黄金手算 + 多空镜像 + 追加不变性 + 确认时序 + amp NaN

场景 (K=3, pivot[j] 确认于 j+3):
  区间(全部 close=100, 摆动高点 105@3 / 低点 96@8 确认) → close 突破 105 (bar 18,
  段 start) → 新 HH 110@21 (确认 24) → 回撤新 HL 103@24 (确认 27, 首个 HL 无事件)
  → 新高 HH 114@32 (确认 35) → 回撤新高 HL 106.0@39 (确认 42, 高于前 HL → 回撤事件)
  → close 跌破 HL (bar 44) → 段 end
"""
import os

import numpy as np
import pandas as pd
import pytest

from research.structures import K, dow_segments


def mk(closes, spikes=None, hi_margin=0.5, lo_margin=0.5):
    """由 close 序列构造 K 线; spikes: {idx: (hi, lo)} 单根尖峰覆盖, None 表示不覆盖"""
    n = len(closes)
    highs = [c + hi_margin for c in closes]
    lows = [c - lo_margin for c in closes]
    for i, (hi, lo) in (spikes or {}).items():
        if hi is not None:
            highs[i] = max(highs[i], hi)
        if lo is not None:
            lows[i] = min(lows[i], lo)
    opens = [closes[0]] + list(closes[:-1])
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"open": opens, "high": highs, "low": lows,
                         "close": closes, "volume": 1.0}, index=idx)


def build_golden():
    """黄金场景: 45 根, 区间 → up 段 (start 18, end 44), 一个回撤事件 @39"""
    closes = [100.0] * 18                              # bars 0-17 区间
    closes += [105.5, 106.5, 107.5, 108.5]             # bars 18-21 突破+爬升
    closes += [107.5, 106.5]                           # bars 22-23 回调
    closes += [105.5, 105.5, 106.0, 106.5]             # bars 24-27 (HL 103@24)
    closes += [107.5, 108.5, 109.5, 110.5]             # bars 28-31 爬升
    closes += [111.5, 110.5, 109.5]                    # bars 32-34 (HH 114@32)
    closes += [109.0, 108.5, 108.0, 107.5]             # bars 35-38 回落
    closes += [107.2, 107.8, 108.3, 108.5]             # bars 39-42 (HL 106.0@39)
    closes += [107.5, 105.5]                           # bars 43-44 跌破 HL → 段 end
    spikes = {
        3: (105.0, None),    # 区间摆动高点 (确认@6)
        8: (None, 96.0),     # 区间摆动低点 (确认@11)
        21: (110.0, None),   # 段内新 HH (确认@24)
        24: (None, 103.0),   # 首个 HL (确认@27, 无回撤事件)
        32: (114.0, None),   # 段内新高 HH (确认@35)
        39: (None, 106.0),   # 新高 HL (确认@42 → 回撤事件)
    }
    return mk(closes, spikes)


def test_up_cycle_golden():
    """手算黄金: 事件逐项断言 (段字段 + 回撤事件字段 + 状态)"""
    df = build_golden()
    res = dow_segments(df)
    # 段生命周期
    assert len(res["segs"]) == 1
    s = res["segs"][0]
    assert s["start"] == 18        # close 收盘突破 105 (bar 18)
    assert s["end"] == 44          # close 跌破最近确认 HL (106.0) → 段结束
    assert s["bars"] == 27
    assert s["direction"] == "up"
    assert s["n_hh"] == 2          # HH: 110@21, 114@32
    assert s["n_hl"] == 2          # HL: 103@24, 106.0@39
    # amp_atr = (peak 114 - trough 103) / atr[18]
    #   atr[18] = 1.3186145117716923 (TR=0.5 恒定 ewm(α=1/14) 收敛值)
    assert s["amp_atr"] == pytest.approx((114.0 - 103.0) / 1.3186145117716923)
    assert s["amp_atr"] == pytest.approx(8.342089292814157)
    # 回撤事件 (HL 106.0@39 确认@42; 前 HL 103@24 存在)
    assert len(res["retraces"]) == 1
    r = res["retraces"][0]
    assert r["bar"] == 39
    assert r["direction"] == "up"
    # depth_atr = (peak 114 - HL 106.0) / atr[39]
    #   atr[39] = 1.4668206614128756
    assert r["depth_atr"] == pytest.approx((114.0 - 106.0) / 1.4668206614128756)
    assert r["depth_atr"] == pytest.approx(5.453972806937567)
    assert r["dur_bars"] == 39 - 32  # 距峰值 HH(114@32) 的根数
    assert r["peak_val"] == pytest.approx(114.0)
    # 状态: 突破 bar 记录 range (状态机口径), 之后 up, 段结束 bar 回 range
    assert res["states"][18] == "range"
    assert res["states"][19] == "up"
    assert res["states"][27] == "up"
    assert res["states"][44] == "range"


def test_mirror_symmetry():
    """价格镜像后段方向/回撤对称 (up ↔ down, 深度/时长一致, amp 取反)"""
    df = build_golden()
    res = dow_segments(df)
    closes = [1000 - c for c in df["close"].values]
    spikes = {}
    for pos in range(len(df)):
        spikes[pos] = (1000 - df["low"].values[pos], 1000 - df["high"].values[pos])
    df2 = mk(closes, spikes)
    res2 = dow_segments(df2)
    assert len(res2["segs"]) == 1 and len(res2["retraces"]) == 1
    s2, r2 = res2["segs"][0], res2["retraces"][0]
    assert s2["direction"] == "down"
    assert (s2["start"], s2["end"], s2["bars"]) == (18, 44, 27)
    assert (s2["n_hh"], s2["n_hl"]) == (2, 2)
    # 镜像 ATR 逐点相等 (TR 在反射下不变) → 归一值可直接对照
    assert s2["amp_atr"] == pytest.approx(-res["segs"][0]["amp_atr"])
    assert r2["bar"] == 39 and r2["direction"] == "down"
    assert r2["depth_atr"] == pytest.approx(res["retraces"][0]["depth_atr"])
    assert r2["dur_bars"] == 7
    assert r2["trough_val"] == pytest.approx(1000 - 114.0)  # 峰值镜像为 trough
    # 状态对称: up ↔ down, range/warmup 不变
    for t in range(len(df)):
        st, st2 = res["states"][t], res2["states"][t]
        if st in ("up", "down"):
            assert st2 == ("down" if st == "up" else "up"), f"t={t} {st} vs {st2}"
        else:
            assert st2 == st, f"t={t} {st} vs {st2}"


def _assert_invariance(df, cuts):
    """追加数据不改变历史: states[:cut-10] / end<cut-10 的 segs / bar<cut-10 的 retraces"""
    full = dow_segments(df)

    def eq_float(a, b):
        return (np.isnan(a) and np.isnan(b)) or a == b

    for cut in cuts:
        trunc = dow_segments(df.iloc[:cut])
        for i in range(cut - 10):
            assert full["states"][i] == trunc["states"][i], f"cut={cut} states[{i}]"
        fs = {s["start"]: s for s in full["segs"] if s["end"] < cut - 10}
        ts = {s["start"]: s for s in trunc["segs"] if s["end"] < cut - 10}
        assert set(fs) == set(ts), f"cut={cut} seg starts 不一致: {set(fs)} vs {set(ts)}"
        for st in fs:
            a, b = fs[st], ts[st]
            for k in ("start", "end", "bars", "direction", "n_hh", "n_hl"):
                assert a[k] == b[k], f"cut={cut} seg@{st}.{k}: {a[k]} vs {b[k]}"
            assert eq_float(a["amp_atr"], b["amp_atr"]), f"cut={cut} seg@{st}.amp_atr"
        fr = {r["bar"]: r for r in full["retraces"] if r["bar"] < cut - 10}
        tr = {r["bar"]: r for r in trunc["retraces"] if r["bar"] < cut - 10}
        assert set(fr) == set(tr), f"cut={cut} retrace bars 不一致: {set(fr)} vs {set(tr)}"
        for rb in fr:
            a, b = fr[rb], tr[rb]
            assert a["direction"] == b["direction"], f"cut={cut} retrace@{rb} direction"
            assert eq_float(a["depth_atr"], b["depth_atr"]), f"cut={cut} retrace@{rb}.depth_atr"
            assert eq_float(a["dur_bars"], b["dur_bars"]), f"cut={cut} retrace@{rb}.dur_bars"
            key = "peak_val" if a["direction"] == "up" else "trough_val"
            assert eq_float(a[key], b[key]), f"cut={cut} retrace@{rb}.{key}"


def test_invariance_synthetic():
    _assert_invariance(build_golden(), cuts=(25, 35, 43))


def test_invariance_real_data():
    if not os.path.exists("data/backtest.db"):
        pytest.skip("backtest.db 不存在")
    from research.data_loader import load_candles
    data = load_candles(timeframes=("1h",))
    for sym in list(data)[:2]:
        df = data[sym]["1h"]
        _assert_invariance(df, cuts=(3000, 10000, len(df) - 200))


def test_confirmation_timing():
    """HL pivot 在 j=39, 确认在 j+K=42 — j..j+K-1 内不得产生该回撤事件"""
    df = build_golden()
    j = 39
    for t in range(j, j + K):
        tr = dow_segments(df.iloc[:t + 1])["retraces"]
        assert not any(r["bar"] == j for r in tr), f"回撤事件 @{j} 在 t={t} 提前出现"
    tr = dow_segments(df.iloc[:j + K + 1])["retraces"]  # 含确认 bar → 事件出现
    assert any(r["bar"] == j for r in tr)


def test_amp_atr_nan_no_pivot():
    """段内无 pivot (突破后立即破位) → amp_atr 为 NaN 而非 0.0 (a6b 修复)"""
    closes = [100.0] * 14 + [106.0, 95.0]  # bar 14 突破, bar 15 跌破区间低
    df = mk(closes, spikes={3: (105.0, None), 8: (None, 96.0)})
    res = dow_segments(df)
    assert len(res["segs"]) == 1
    s = res["segs"][0]
    assert s["start"] == 14 and s["end"] == 15 and s["bars"] == 2
    assert s["direction"] == "up"
    assert s["n_hh"] == 0 and s["n_hl"] == 0
    assert np.isnan(s["amp_atr"]), f"无 pivot 段 amp_atr 应为 NaN, 实际 {s['amp_atr']}"
    assert len(res["retraces"]) == 0
