def compute_market_state(tf_ind: dict) -> dict:
    if not tf_ind:
        return {"bias": "neutral", "confidence": 0, "desc": "无数据"}

    sym = next(iter(tf_ind.keys()), None)
    if not sym:
        return {"bias": "neutral", "confidence": 0, "desc": "无数据"}

    ind_data = tf_ind.get(sym, {})
    ind_1h = ind_data.get("1h", {})
    if not ind_1h:
        return {"bias": "neutral", "confidence": 0, "desc": "数据不足"}

    adx_1h = ind_1h.get("adx", 0) or 0
    ma_align_1h = ind_1h.get("ma_alignment", "neutral")
    bb_state = ind_1h.get("bb_state", "unknown")
    vr_1h = ind_1h.get("volume_ratio") or 1
    close_1h = ind_1h.get("close")
    ma20_1h = ind_1h.get("ma20")
    atr_1h = ind_1h.get("atr") or 1

    bias = "neutral"
    conf = 0
    bias_reason = ""

    if ma_align_1h == "bullish" and adx_1h >= 20:
        bias = "long"
        conf = min(50 + adx_1h * 0.5, 75)
        bias_reason = f"1H多头 ADX={adx_1h:.0f}"
    elif ma_align_1h == "bearish" and adx_1h >= 20:
        bias = "short"
        conf = min(50 + adx_1h * 0.5, 75)
        bias_reason = f"1H空头 ADX={adx_1h:.0f}"
    elif ma_align_1h == "neutral":
        if bb_state == "contracting":
            bias = "neutral"
            conf = 0
            bias_reason = "1H收缩-等方向"
        else:
            bias = "neutral"
            conf = 0
            bias_reason = "1H中性-无方向偏倚"
    else:
        bias = "neutral"
        conf = 0
        bias_reason = f"1H趋势弱 ADX={adx_1h:.0f}"

    pullback_pct = 0
    if ma20_1h and close_1h and ma20_1h > 0:
        pullback_pct = (close_1h - ma20_1h) / ma20_1h * 100

    state_desc = f"1H{'多头' if ma_align_1h == 'bullish' else '空头' if ma_align_1h == 'bearish' else '中性'}(ADX{adx_1h:.0f})"
    state_desc += f" | 距MA20:{pullback_pct:+.1f}%"
    state_desc += f" | BB:{bb_state}"
    if vr_1h >= 1.5:
        state_desc += " | 放量"

    return {
        "bias": bias,
        "confidence": round(conf),
        "desc": state_desc,
        "reason": bias_reason,
        "adx_1h": round(adx_1h, 1),
        "ma_1h": ma_align_1h,
        "pullback_pct": round(pullback_pct, 1),
    }
