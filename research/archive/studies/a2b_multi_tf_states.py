#!/usr/bin/env python3
"""A2b 多周期状态对比 — 1h / 4h / 日线 的趋势状态特征是否一致 (2026-08-03, 无未来函数)

问题: 日线/4H 级别的趋势状态是否由与 1H 不同的特征刻画?
方法: 同一 classify 公式对 1h/4h/1d(从4h重采样) 三周期逐 bar 复现状态,
      对比 状态占比 / 平均持续 / 特征画像 (ADX/偏离/动能/量能比/波动率/阶段特征)
无未来函数: 单周期序列只用已收盘 bar; 日线 bar 索引=当日00:00 (开盘时间),
  重采样自已收盘 4h — 与 1h/4h 相同语义, 无 lookahead
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.data_loader import daily_resample, load_candles, verify
from research.state_features import state_series

WARMUP = 200  # 日线样本少 (1094 根), 用 200 而非 730 (120 收敛 + 裕量)


def load3():
    data = load_candles(timeframes=("1h", "4h"))
    out = {"1h": [], "4h": [], "1d": []}
    for sym, tfs in data.items():
        for tf in ("1h", "4h"):
            df = tfs[tf]
            if verify(df, sym, tf):
                continue
            out[tf].append(df)
        daily = daily_resample(tfs["4h"])
        if len(daily) >= 400:
            out["1d"].append(daily)
    return out


def run_period(dfs, tf):
    hours = {"1h": 1, "4h": 4, "1d": 24}[tf]
    all_seq, all_feats = [], []
    for df in dfs:
        st, feats = state_series(df)
        all_seq.append(st[WARMUP:])
        all_feats.append(feats.iloc[WARMUP:].values)
    seq = np.concatenate(all_seq)
    feats = np.concatenate(all_feats)
    # feats 列顺序: adx dev slope mom body_ratio bbw atr_c vol_ratio
    return seq, feats, hours


def fcol(feats, idx, m):
    return feats[m, idx]


def main():
    dfs_by_tf = load3()
    print("═══ A2b 多周期状态对比 (1h / 4h / 日线) ═══\n")
    results = {}
    for tf in ("1h", "4h", "1d"):
        seq, feats, hours = run_period(dfs_by_tf[tf], tf)
        results[tf] = (seq, feats, hours)
        # 平均持续 (按状态过滤的 run length)
        runs_all, cur, cnt = [], None, 0
        for v in seq:
            if v == cur:
                cnt += 1
            else:
                if cur is not None and cnt:
                    runs_all.append((cur, cnt))
                cur, cnt = v, 1
        runs_all.append((cur, cnt))
        names = sorted(set(seq))
        print(f"── {tf} ({len(dfs_by_tf[tf])} 标的 × {seq.shape[0]} 根) ──")
        for s in names:
            m = seq == s
            occ = m.mean()
            lens = [c for s_, c in runs_all if s_ == s]
            dur = np.mean(lens) if lens else np.nan
            adx = feats[m, 0].mean()
            dev = feats[m, 1].mean()
            mom = feats[m, 3].mean()
            vr = feats[m, 6].mean()   # vol_ratio
            atrc = feats[m, 7].mean()  # atr_c
            print(f"  {s:>22}: 占{occ:5.1%} 持续{dur:5.1f}根({dur*hours:6.0f}h) "
                  f"ADX {adx:5.1f} 偏离 {dev:+5.2f} 动能 {mom:+5.2f} 量能 {vr:4.2f} 波动率 {atrc:6.3f}")
        print()

    # 阶段特征对比 (early/accelerate/late 的分离度是否随周期变化)
    print("── 阶段特征 (early/accelerate/late) 跨周期对比 ──\n")
    print(f"{'周期':>4} | {'阶段':>20} | {'偏离':>7} | {'动能':>7} | {'量能':>6}")
    for tf in ("1h", "4h", "1d"):
        seq, feats, _ = results[tf]
        for stg in ["early", "accelerate", "late"]:
            m = np.array([s.endswith(stg) for s in seq])
            if not m.sum():
                continue
            dev = feats[m, 1].mean()
            mom = feats[m, 3].mean()
            vr = feats[m, 6].mean()
            print(f"{tf:>4} | {stg:>20} | {dev:+7.2f} | {mom:+7.2f} | {vr:6.2f}")
        print()


if __name__ == "__main__":
    main()
