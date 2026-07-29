import numpy as np
from dataclasses import dataclass


@dataclass
class RSResult:
    symbol: str
    rs_score: float       # -100 ~ +100, ATR归一化后的连续相对强弱评分
    rs_zscore: float      # 多窗口加权平均 z-score
    rs_rank: float        # 全币种百分位排名 (0~1)
    rs_level: str         # strong / mild_strong / neutral / mild_weak / weak


def compute_rs(
    sym_close_map: dict[str, float | None],
    sym_prev_maps: list[dict[str, float | None]],
    sym_atr_map: dict[str, float | None],
    btc_close: float | None,
    btc_prev_list: list[float | None],
    btc_atr: float | None,
    lookbacks: list[int] | None = None,
) -> dict[str, RSResult]:
    if not btc_close or btc_atr is None or btc_atr <= 0:
        return {}

    if lookbacks is None:
        lookbacks = [5]

    # 反比权重
    raw_weights = np.array([1.0 / w for w in lookbacks])
    weights = raw_weights / raw_weights.sum()

    btc_atr_pct = btc_atr / btc_close * 100 if btc_close > 0 else 1.0
    btc_roc = (btc_close - btc_prev_list[0]) / btc_prev_list[0] * 100 if btc_prev_list[0] and btc_prev_list[0] > 0 else 0
    btc_z = btc_roc / max(btc_atr_pct, 0.01)

    results = {}
    z_diffs_all = {}

    for sym, close in sym_close_map.items():
        atr = sym_atr_map.get(sym)
        if not close or not atr or atr <= 0:
            continue

        sym_atr_pct = atr / close * 100 if close > 0 else 1.0
        sym_zs = []
        for i, w in enumerate(lookbacks):
            prev_map = sym_prev_maps[i] if i < len(sym_prev_maps) else {}
            prev = prev_map.get(sym)
            if prev and prev > 0:
                sym_roc = (close - prev) / prev * 100
                sym_z = sym_roc / max(sym_atr_pct, 0.01)
                sym_zs.append(sym_z)

        if not sym_zs:
            continue

        sym_z_avg = sum(sym_zs[j] * weights[j] for j in range(min(len(sym_zs), len(weights))))
        z_diff = sym_z_avg - btc_z
        z_diffs_all[sym] = z_diff

        results[sym] = RSResult(
            symbol=sym,
            rs_score=0.0,
            rs_zscore=round(z_diff, 4),
            rs_rank=0.5,
            rs_level="neutral",
        )

    if len(z_diffs_all) < 2:
        return results

    z_vals = np.array(list(z_diffs_all.values()))

    for sym, z_diff in z_diffs_all.items():
        rank = (z_vals < z_diff).sum() / max(len(z_vals) - 1, 1)
        rank_dev = (rank - 0.5) * 2
        norm_z = np.clip(z_diff / 3.0, -1, 1)
        rs_score = round((norm_z * 0.5 + rank_dev * 0.5) * 100, 2)

        if rs_score >= 40:
            level = "strong"
        elif rs_score >= 15:
            level = "mild_strong"
        elif rs_score <= -40:
            level = "weak"
        elif rs_score <= -15:
            level = "mild_weak"
        else:
            level = "neutral"

        results[sym].rs_score = rs_score
        results[sym].rs_rank = round(float(rank), 4)
        results[sym].rs_level = level

    return results
