#!/usr/bin/env python3
"""裂变 vs 非裂变 — 蒙特卡洛模拟"""

import random
import math

# ── 参数 ──
WIN_RATE = 0.45
RR = 2.0
TRADES_PER_MONTH = 20
MONTHS = 12

# 单笔期望: 0.45*2 - 0.55*1 = 0.35R
KELLY_F = (WIN_RATE * RR - (1 - WIN_RATE)) / RR  # ≈ 0.175

SCENARIOS = [
    ("裂变 Kelly", KELLY_F, True, KELLY_F),
    ("裂变 半Kelly", KELLY_F / 2, True, KELLY_F / 2),
    ("裂变 保守2%", 0.02, True, 0.02),
    ("单链 Kelly", KELLY_F, False, KELLY_F),
    ("单链 半Kelly", KELLY_F / 2, False, KELLY_F / 2),
    ("单链 保守2%", 0.02, False, 0.02),
]

FISSION_RESERVE = 300   # 老链裂变后保留的残值
FISSION_SEED = 500      # 新种子链

SIMULATIONS = 100


def sim_trade(capital, risk_fraction):
    """单笔交易。返回新资本。"""
    stake = capital * risk_fraction
    if random.random() < WIN_RATE:
        return capital + stake * RR
    else:
        return capital - stake


def run_one(sim_name, risk_fraction, fission_enabled, num_trades, seed):
    """运行一次模拟，返回每月末的总资本。"""
    random.seed(seed)

    if fission_enabled:
        chains = [500.0]  # 初始一条链
        harvest = 0.0     # 收割到保守账户的利润
        history = []

        for month in range(1, MONTHS + 1):
            for _ in range(TRADES_PER_MONTH):
                new_chains = []

                # 用索引遍历，确保修改写回列表
                idx = 0
                while idx < len(chains):
                    c = chains[idx]

                    # 残链太小不能交易 → 收割
                    if c < 50:
                        harvest += c
                        chains.pop(idx)
                        continue

                    # 单笔交易
                    c = sim_trade(c, risk_fraction)

                    # 检查裂变：赚够 300U 利润（从 500U 起）
                    if c >= 800:  # 500 + 300
                        new_chains.append(500.0)       # 子链
                        c -= 500                        # 老链残值
                        if c < FISSION_RESERVE:
                            c = FISSION_RESERVE         # 残链保底

                    # 太小的残链 → 消亡并收割
                    if c < 50:
                        harvest += c
                        chains.pop(idx)
                        continue

                    chains[idx] = c
                    idx += 1

                chains.extend(new_chains)

                # 链数上限（防爆炸 + 模拟真实市场容量）
                if len(chains) > 200:
                    chains.sort(reverse=True)
                    harvest += sum(chains[200:])
                    chains = chains[:200]

            # 每月末：收割成熟大链 (>= 10000U)
            for i in range(len(chains)):
                if chains[i] >= 10000:
                    harvest += chains[i] - FISSION_SEED
                    chains[i] = FISSION_SEED

            # 记录总资本 = 所有链 + 收割
            total = harvest + sum(chains)
            history.append(total)

        return history

    else:
        # 单链模式 — 不收割，自由增长
        capital = 500.0
        history = []

        for month in range(1, MONTHS + 1):
            for _ in range(TRADES_PER_MONTH):
                if capital < 10:
                    break
                capital = sim_trade(capital, risk_fraction)

            history.append(capital)

        return history


# ── 运行模拟 ──
print("=" * 90)
print("裂变 vs 非裂变 — 模拟对比")
print(f"胜率={WIN_RATE:.0%}  RR={RR}:1  Kelly f={KELLY_F:.1%}")
print(f"模拟次数={SIMULATIONS}  月数={MONTHS}  每月交易={TRADES_PER_MONTH}笔")
print("=" * 90)

total_trades = MONTHS * TRADES_PER_MONTH

for name, risk_frac, fission, _ in SCENARIOS:
    month_histories = [[] for _ in range(MONTHS)]
    for sim_i in range(SIMULATIONS):
        seed = sim_i * 1000 + hash(name) % 10000
        hist = run_one(name, risk_frac, fission, total_trades, seed)
        for m in range(MONTHS):
            month_histories[m].append(hist[m])

    # 统计
    medians = [sorted(h)[len(h)//2] for h in month_histories]
    final_vals = month_histories[-1]

    # 分位数
    final_sorted = sorted(final_vals)
    p10 = final_sorted[len(final_sorted)//10]
    p90 = final_sorted[len(final_sorted)*9//10]

    mean_final = sum(final_vals) / len(final_vals)
    # 年化几何增长率
    pos_vals = [v for v in final_vals if v > 1]
    if pos_vals:
        log_mean = sum(math.log10(v/500) for v in pos_vals) / len(pos_vals)
        annual_growth = 10 ** (log_mean / 2)  # 2年
    else:
        annual_growth = 0

    survived = sum(1 for v in final_vals if v > 500) / SIMULATIONS

    fission_tag = "裂变" if fission else "单链"
    print(f"\n{fission_tag} | {name.split()[-1]:8s} | 风险{risk_frac:.1%}")
    print(f"  期末中位数: {medians[-1]:,.0f}U  (P10={p10:,.0f}  P90={p90:,.0f})")
    print(f"  期末均值:   {mean_final:,.0f}U")
    print(f"  年化倍数:   {annual_growth:.1f}×/年")
    print(f"  2年 >500U 比例: {survived:.0%}")
    print(f"  路径示例 (月): ", end="")
    for m in range(0, MONTHS, MONTHS // 6):
        if m < len(medians):
            print(f"{medians[m]:,.0f} ", end="")
    print()
