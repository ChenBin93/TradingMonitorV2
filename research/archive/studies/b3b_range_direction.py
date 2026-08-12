#!/usr/bin/env python3
"""B3b 震荡区间关键位方向概率 + 触次×年龄交叉分离 (2026-08-04, 无未来函数)

动机 (用户观点): "方向无规律"可能是无条件混合的假象 — 震荡区间关键位有
明确方向倾向 (触下沿→反弹, 触上沿→回落)。本实验直接测方向概率并叠加条件。

Part 1 — 区间方向概率:
  D1 触下沿 (支撑) 后 24 根 P(close[t+W] > close[t]) — 反弹向上概率
  D2 触上沿 (阻力) 后 24 根 P(close[t+W] < close[t]) — 回落向下概率
  分层: 无条件 / 结构层 / 区间内×年龄 / 区间内×波动 / 区间内×年龄<30×波动低
  对照: 全体 bar 的无条件 P(up) + GBM 同流程

Part 2 — 触次×年龄交叉分离 (回答"多触碰位更有效"是否年龄混叠):
  X1 控制年龄 30~120: 触次 5~8 vs 9+ → 触次独立效应
  X2 控制触次 5~8: 年龄 <120 vs >120 → 年龄独立效应
  度量: M1 (留本侧) / 拒绝率 / E1 (波动释放) / M2P50

预注册假设:
  H1 区间内触下沿 P(up) 真实 > GBM (用户: 震荡区间方向倾向)
  H2 区间内 × 新位带 × 低波动 → 方向净差最大
  H3 触次独立效应: 控制年龄后触次仍分层 (或归零 → 纯年龄混叠)

运行: python3 research/studies/b3b_range_direction.py (约 30 分钟)
"""
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.data_loader import load_candles, verify
from research.sim_market import gbm_dataframe
from research.studies.b3_general_levels import (collect, bucket_key, W, MIN_N)

PARAMS = [(2, 0.3), (3, 0.5)]
TIMEFRAMES = ["1h", "4h"]

GROUPS = {
    "all": lambda b: True,
    "str_区间内": lambda b: b["structure"] == "区间内",
    "str_宽成对": lambda b: b["structure"] == "宽成对",
    "str_孤立": lambda b: b["structure"] == "孤立",
    "range_age<30": lambda b: b["structure"] == "区间内" and b["age"] == "<30",
    "range_age30-120": lambda b: b["structure"] == "区间内" and b["age"] == "30~120",
    "range_age>120": lambda b: b["structure"] == "区间内" and b["age"] == ">120",
    "range_vol低": lambda b: b["structure"] == "区间内" and b["vol"] == "低",
    "range_vol中": lambda b: b["structure"] == "区间内" and b["vol"] == "中",
    "range_vol高": lambda b: b["structure"] == "区间内" and b["vol"] == "高",
    "range_new_lowvol": lambda b: (b["structure"] == "区间内" and b["age"] == "<30"
                                   and b["vol"] == "低"),
    "X_age30-120_t5-8": lambda b: b["age"] == "30~120" and b["touch"] == "5~8",
    "X_age30-120_t9+": lambda b: b["age"] == "30~120" and b["touch"] == "9+",
    "X_t5-8_age<120": lambda b: b["touch"] == "5~8" and b["age"] != ">120",
    "X_t5-8_age>120": lambda b: b["touch"] == "5~8" and b["age"] == ">120",
}


def run_block(dfs, mt, tol, label=""):
    print(f"\n{'═' * 66}\nB3b {label} 参数(min_touch={mt}, tol={tol})\n{'═' * 66}")
    acc = {g: defaultdict(list) for g in GROUPS}   # 每组的度量 list
    base_up, base_dn = [], []
    for df in dfs:
        d = collect(df, mt, tol)
        c, atr, n = d["c"], d["atr"], d["n"]
        # 无条件方向基线 (全部 bar, 步长 3 采样)
        idx = np.arange(0, n - W, 3)
        seg = c[idx + W]
        base_up.append(np.mean(seg > c[idx]))
        base_dn.append(np.mean(seg < c[idx]))
        for i, lv in enumerate(d["lvls"]):
            t_arr = d["ev"]["touch"][i]
            if not len(t_arr):
                continue
            valid = (t_arr + W < n) & (t_arr >= 13) & (atr[t_arr] > 0)
            t_arr = t_arr[valid]
            if not len(t_arr):
                continue
            seg = c[t_arr + W]
            rej = d["ev"]["reject"][i]
            is_rej = np.isin(t_arr, rej)
            if lv.side == "support":
                dir_up = (seg > c[t_arr])
                m1 = (seg > lv.price)
            else:
                dir_up = (seg < c[t_arr])
                m1 = (seg < lv.price)
            for j, t in enumerate(t_arr):
                b = bucket_key({"t": int(t), "lv": lv, "d": d, "seq": j + 1})
                for g, pred in GROUPS.items():
                    if not pred(b):
                        continue
                    a = acc[g]
                    a["n"].append(1)
                    a["dir"].append(float(dir_up[j]))
                    a["m1"].append(float(m1[j]))
                    a["m5"].append(float(is_rej[j]))
    bup = np.mean(base_up)
    bdn = np.mean(base_dn)
    print(f"无条件方向基线: P(up) {bup:.1%} | P(dn) {bdn:.1%}")
    print(f"{'组':<20}{'n':>8} {'触反方向概率':>12} {'M1':>7} {'拒绝':>7}")
    for g in GROUPS:
        a = acc[g]
        nn = len(a["dir"])
        if nn < MIN_N:
            print(f"{g:<20}{nn:>8}   样本不足(<{MIN_N})")
            continue
        ddir = np.mean(a["dir"])
        m1 = np.mean(a["m1"])
        m5 = np.mean(a["m5"])
        print(f"{g:<20}{nn:>8} {ddir:>12.1%} {m1:>7.1%} {m5:>7.1%}")
    return acc, (bup, bdn)


def diff_block(real, gbm, label=""):
    print(f"\n── 净差矩阵 (真实−GBM) {label} ──")
    print(f"{'组':<20}{'n_真实':>8}{'n_GBM':>8} {'Δ方向概率':>9} {'ΔM1':>7} {'Δ拒绝':>7}")
    for g in GROUPS:
        rd, gd = real[0][g], gbm[0][g]
        nn, gn = len(rd["dir"]), len(gd["dir"])
        if nn < MIN_N or gn < MIN_N:
            print(f"{g:<20}{nn:>8}{gn:>8}   n 不足")
            continue
        ddir = np.mean(rd["dir"]) - np.mean(gd["dir"])
        dm1 = np.mean(rd["m1"]) - np.mean(gd["m1"])
        dm5 = np.mean(rd["m5"]) - np.mean(gd["m5"])
        print(f"{g:<20}{nn:>8}{gn:>8} {ddir:>9.1%} {dm1:>7.1%} {dm5:>7.1%}")


def main():
    for mt, tol in PARAMS:
        for tf in TIMEFRAMES:
            print(f"\n########## 真实 {tf} (min_touch={mt}, tol={tol}) ##########")
            dfs = _load(tf)
            real = run_block(dfs, mt, tol, f"真实 {tf}")
            ref = dfs[0]
            sig = np.std(np.diff(np.log(ref["close"].values)))
            rw = [gbm_dataframe(len(ref), sig, seed=1100 + (tf == "4h") * 100 + k)
                  for k in range(10)]
            gbm = run_block(rw, mt, tol, f"GBM {tf}")
            diff_block(real, gbm, f"{tf} 参数({mt},{tol})")
            sys.stdout.flush()


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


if __name__ == "__main__":
    main()
