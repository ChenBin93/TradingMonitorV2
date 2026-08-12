#!/usr/bin/env python3
"""B4 区间触碰小目标胜率快速验证 (2026-08-04, 无未来函数)

假设 (用户): 震荡关键水平位触碰改变价格方向 (外→内, 短窗口) → 用小的
ATR 目标 (0.3~0.5) 可以给某个方向 (向内) 更大的胜率。

策略: 区间内触碰 (触下沿做多 / 触上沿做空), 入场 = 触碰 bar 收盘,
open 出发判定, 1:1 对称 T×ATR 目标/止损, 窗口 W 根 timeout。
对照: GBM 30 种子 (1:1 对称在无偏随机游走应 ≈50%); T=1.0×W=24 复现 B1。

运行: python3 research/studies/b4_small_target.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.data_loader import load_candles, verify
from research.sim_market import gbm_dataframe
from research.studies.b3_general_levels import (collect as b3collect, bucket_key,
                                                MIN_N)

PARAMS = [(2, 0.3)]
TIMEFRAMES = ["1h"]
N_GBM = 30
TS = (0.3, 0.5, 0.7, 1.0)
WS = (6, 12, 24)


def run_strategy(dfs, mt, tol, label=""):
    print(f"\n{'═' * 78}\nB4 {label} 参数(min_touch={mt}, tol={tol})\n{'═' * 78}")
    # acc[T][W] = {"long": [胜/负/timeout], "short": [...]}
    acc = {t: {w: {"long": [], "short": []} for w in WS} for t in TS}
    for df in dfs:
        d = b3collect(df, mt, tol)
        c, h, l, o, atr, n = d["c"], d["h"], d["l"], df["open"].values, d["atr"], d["n"]
        for i, lv in enumerate(d["lvls"]):
            t_arr = d["ev"]["touch"][i]
            valid = (t_arr + max(WS) < n) & (atr[t_arr] > 0)
            t_arr = t_arr[valid]
            if not len(t_arr):
                continue
            keys = []
            for j, t in enumerate(t_arr):
                b = bucket_key({"t": int(t), "lv": lv, "d": d, "seq": j + 1})
                keys.append(b["structure"] == "区间内")
            keys = np.array(keys)
            if not keys.any():
                continue
            t_arr = t_arr[keys]
            entry = c[t_arr]
            atr_t = atr[t_arr]
            for T in TS:
                for w in WS:
                    target = entry + T * atr_t if lv.side == "support" else entry - T * atr_t
                    stop = entry - T * atr_t if lv.side == "support" else entry + T * atr_t
                    out = _sim(t_arr, o, h, l, target, stop, w)
                    acc[T][w]["long" if lv.side == "support" else "short"].extend(out)
    _report(acc, label)
    return acc


def _sim(t_arr, o, h, l, target, stop, w):
    """open 出发 1:1 判定, 返回每事件结果 list (1=目标, 0=timeout, -1=止损)"""
    out = np.zeros(len(t_arr), dtype=int)
    done = np.zeros(len(t_arr), dtype=bool)
    for k in range(1, w + 1):
        if done.all():
            break
        ok = t_arr + k
        ov, hv, lv = o[ok], h[ok], l[ok]
        gap_t = ov >= target
        gap_s = ov <= stop
        hit_t = gap_t | (hv >= target)
        hit_s = gap_s | (lv <= stop)
        both = gap_t & gap_s
        two = ~both & hit_t & hit_s
        # 先碰: open 越界优先 (同根双命中除 gap 外跳过)
        m_t = ~done & ~both & ~two & hit_t
        m_s = ~done & ~both & ~two & hit_s
        out[m_t] = 1
        out[m_s] = -1
        done |= m_t | m_s
    return out.tolist()


def _report(acc, label):
    print(f"{'T':>4} {'W':>3} {'多(触下沿)':>16} {'空(触上沿)':>16} {'合并(向内)':>16}")
    for T in TS:
        for w in WS:
            lg, sh = acc[T][w]["long"], acc[T][w]["short"]
            allr = lg + sh
            rows = []
            for arr in (lg, sh, allr):
                if len(arr) < MIN_N:
                    rows.append(f"n={len(arr)}不足")
                    continue
                arr = np.array(arr)
                wr = np.mean(arr > 0)
                ex = arr.mean()
                rows.append(f"n={len(arr)} 胜率{wr:.1%} 期望{ex:+.3f}")
            print(f"{T:>4} {w:>3} {rows[0]:>16} {rows[1]:>16} {rows[2]:>16}")


def diff_block(real, gbm, label=""):
    print(f"\n── 净差 (真实−GBM) {label} ──")
    print(f"{'T':>4} {'W':>3} {'Δ多胜率':>9} {'Δ空胜率':>9} {'Δ合并胜率':>10} "
          f"{'Δ合并期望':>10}")
    for T in TS:
        for w in WS:
            lg_r, sh_r = real[T][w]["long"], real[T][w]["short"]
            lg_g, sh_g = gbm[T][w]["long"], gbm[T][w]["short"]
            allr = np.array(lg_r + sh_r)
            allg = np.array(lg_g + sh_g)
            if len(allr) < MIN_N or len(allg) < MIN_N:
                print(f"{T:>4} {w:>3} n 不足 (真 {len(allr)} / GBM {len(allg)})")
                continue
            dl = (np.mean(np.array(lg_r) > 0) - np.mean(np.array(lg_g) > 0)
                  if len(lg_r) >= MIN_N and len(lg_g) >= MIN_N else float("nan"))
            ds = (np.mean(np.array(sh_r) > 0) - np.mean(np.array(sh_g) > 0)
                  if len(sh_r) >= MIN_N and len(sh_g) >= MIN_N else float("nan"))
            dm = np.mean(allr > 0) - np.mean(allg > 0)
            de = allr.mean() - allg.mean()
            print(f"{T:>4} {w:>3} {dl:>9.1%} {ds:>9.1%} {dm:>10.1%} {de:>10.3f}")


def _load(tf):
    data = load_candles(timeframes=(tf,))
    out = []
    for sym, tfs in data.items():
        df = tfs.get(tf)
        if df is None:
            continue
        if verify(df, sym, tf):
            continue
        out.append(df)
    return out


def main():
    for mt, tol in PARAMS:
        for tf in TIMEFRAMES:
            print(f"\n########## 真实 {tf} (min_touch={mt}, tol={tol}) ##########")
            dfs = _load(tf)
            real = run_strategy(dfs, mt, tol, f"真实 {tf}")
            ref = dfs[0]
            sig = np.std(np.diff(np.log(ref["close"].values)))
            rw = [gbm_dataframe(len(ref), sig, seed=4000 + k) for k in range(N_GBM)]
            gbm = run_strategy(rw, mt, tol, f"GBM {tf}")
            diff_block(real, gbm, f"{tf} 参数({mt},{tol})")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
