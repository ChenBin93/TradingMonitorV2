import numpy as np
from dataclasses import dataclass


@dataclass
class RSResult:
    symbol: str
    rs_score: float       # -100 ~ +100, ATR归一化后的连续相对强弱评分
    rs_momentum: float    # 原始超额收益率 (%)
    rs_zscore: float      # ATR归一化后的 z-score (超额/波动)
    rs_rank: float        # 全币种百分位排名 (0~1)
    rs_level: str         # strong / mild_strong / neutral / mild_weak / weak


def compute_rs(
    sym_close_map: dict[str, float | None],
    sym_prev_close_map: dict[str, float | None],
    sym_atr_map: dict[str, float | None],
    btc_close: float | None,
    btc_prev_close: float | None,
    btc_atr: float | None,
    momentum_period: int = 5,
) -> dict[str, RSResult]:
    if not btc_close or not btc_prev_close or btc_prev_close == 0:
        return {}

    btc_roc = (btc_close - btc_prev_close) / btc_prev_close * 100
    btc_atr_pct = (btc_atr / btc_close * 100) if btc_atr and btc_atr > 0 and btc_close > 0 else 1.0
    btc_z = btc_roc / max(btc_atr_pct, 0.01)

    results = {}
    z_diffs = {}

    for sym, close in sym_close_map.items():
        prev = sym_prev_close_map.get(sym)
        atr = sym_atr_map.get(sym)
        if not close or not prev or prev == 0:
            continue

        sym_roc = (close - prev) / prev * 100
        rs_momentum = sym_roc - btc_roc

        sym_atr_pct = (atr / close * 100) if atr and atr > 0 and close > 0 else 1.0
        sym_z = sym_roc / max(sym_atr_pct, 0.01)
        z_diff = sym_z - btc_z

        z_diffs[sym] = z_diff
        results[sym] = RSResult(
            symbol=sym,
            rs_score=0.0,
            rs_momentum=round(rs_momentum, 4),
            rs_zscore=round(z_diff, 4),
            rs_rank=0.5,
            rs_level="neutral",
        )

    if len(z_diffs) < 2:
        return results

    z_vals = np.array(list(z_diffs.values()))

    for sym, z_diff in z_diffs.items():
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
        results[sym].rs_zscore = round(z_diff, 4)
        results[sym].rs_rank = round(float(rank), 4)
        results[sym].rs_level = level

    return results
