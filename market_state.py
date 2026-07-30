def compute_market_state(tf_ind: dict) -> dict:
    """从全量指标池生成市场状态报告, 聚焦 BTC 做风向标"""
    btc_key = "BTC/USDT:USDT"
    sym = btc_key if btc_key in tf_ind else next(iter(tf_ind.keys()), None)
    if not sym:
        return {"bias": "neutral", "confidence": 0, "desc": "无数据", "btc_price": None}

    ind_data = tf_ind.get(sym, {})
    ind_4h = ind_data.get("4h", {})
    ind_1h = ind_data.get("1h", {})
    ind_5m = ind_data.get("5m", {})

    if not ind_1h:
        return {"bias": "neutral", "confidence": 0, "desc": "数据不足", "btc_price": None}

    # ── BTC 现价 ──
    btc_price = ind_1h.get("close")
    btc_atr_1h = ind_1h.get("atr") or 1
    btc_change_1h = ind_1h.get("roc", 0) or 0

    # ── 1H 指标 ──
    adx_1h = ind_1h.get("adx", 0) or 0
    ma_align_1h = ind_1h.get("ma_alignment", "neutral")
    bb_state_1h = ind_1h.get("bb_state", "unknown")
    vr_1h = ind_1h.get("volume_ratio") or 1
    ma20_1h = ind_1h.get("ma20")
    pullback_pct = 0
    if ma20_1h and btc_price and ma20_1h > 0:
        pullback_pct = (btc_price - ma20_1h) / ma20_1h * 100

    # ── 4H 趋势 ──
    adx_4h = (ind_4h or {}).get("adx", 0) or 0
    ma_align_4h = (ind_4h or {}).get("ma_alignment", "neutral")
    ma20_4h = (ind_4h or {}).get("ma20")
    ma60_4h = (ind_4h or {}).get("ma60")
    bb_state_4h = (ind_4h or {}).get("bb_state", "unknown")
    above_ma20_4h = ma20_4h and btc_price and btc_price > ma20_4h and ma20_4h > 0
    above_ma60_4h = ma60_4h and btc_price and btc_price > ma60_4h and ma60_4h > 0

    if adx_4h >= 20 and ma_align_4h == "bullish":
        trend_4h = f"📈4H多头(ADX{adx_4h:.0f})"
    elif adx_4h >= 20 and ma_align_4h == "bearish":
        trend_4h = f"📉4H空头(ADX{adx_4h:.0f})"
    elif above_ma20_4h and above_ma60_4h:
        trend_4h = f"4H偏多(ADX{adx_4h:.0f})"
    elif not above_ma20_4h and not above_ma60_4h:
        trend_4h = f"4H偏空(ADX{adx_4h:.0f})"
    else:
        trend_4h = f"4H中性(ADX{adx_4h:.0f})"

    # ── 1H S/R 位 ──
    sr_str = ""
    df_1h = ind_1h.get("df")
    if df_1h is not None and len(df_1h) >= 20 and btc_price:
        try:
            from support_resistance import find_swing_levels, get_nearest_levels
            levels = find_swing_levels(df_1h, lookback=50)
            support, resistance = get_nearest_levels(levels, btc_price)
            parts = []
            if support:
                p = support.price
                sf = f"{p:.0f}" if p > 100 else f"{p:.1f}" if p > 1 else f"{p:.5f}"
                parts.append(f"支撑:{sf}({support.strength},{support.touch_count}触)")
            if resistance:
                p = resistance.price
                sf = f"{p:.0f}" if p > 100 else f"{p:.1f}" if p > 1 else f"{p:.5f}"
                parts.append(f"阻力:{sf}({resistance.strength},{resistance.touch_count}触)")
            sr_str = " · ".join(parts) if parts else ""
        except Exception:
            pass

    # ── 1H 方向判定 ──
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
    elif adx_1h < 15:
        bias = "neutral"
        conf = 0
        bias_reason = f"1H震荡 ADX={adx_1h:.0f}"
    else:
        bias = "neutral"
        conf = 0
        bias_reason = f"1H中性 ADX={adx_1h:.0f}"

    # ── 多空计数 ──
    bull_count = sum(
        1 for s in tf_ind
        if (tf_ind[s].get("1h") or {}).get("ma_alignment") == "bullish"
    )
    bear_count = sum(
        1 for s in tf_ind
        if (tf_ind[s].get("1h") or {}).get("ma_alignment") == "bearish"
    )
    total_with_data = bull_count + bear_count + sum(
        1 for s in tf_ind
        if (tf_ind[s].get("1h") or {}).get("ma_alignment") == "neutral"
    )

    breadth_str = ""
    if total_with_data > 5:
        bull_pct = bull_count / total_with_data * 100
        bear_pct = bear_count / total_with_data * 100
        breadth_str = f"多{bull_pct:.0f}%/空{bear_pct:.0f}%"

    # ── 生成 desc ──
    btc_disp = f"{btc_price:.0f}" if btc_price and btc_price > 100 else f"{btc_price:.5f}" if btc_price else "-"
    desc_parts = [f"BTC {btc_disp}"]

    if btc_change_1h:
        desc_parts.append(f"{btc_change_1h:+.2f}%")
    desc_parts.append(trend_4h)

    # 1H 行
    h1_type = "多头" if ma_align_1h == "bullish" else "空头" if ma_align_1h == "bearish" else "中性"
    h1_line = f"1H{h1_type}(ADX{adx_1h:.0f}) | 距MA20:{pullback_pct:+.1f}%"
    if bb_state_1h == "contracting":
        h1_line += " | 收缩"
    elif bb_state_1h == "expanding":
        h1_line += " | 扩张"
    if vr_1h >= 1.5:
        h1_line += " | 放量"

    return {
        "bias": bias,
        "confidence": round(conf),
        "desc": " | ".join(desc_parts),
        "h1_line": h1_line,
        "reason": bias_reason,
        "sr": sr_str,
        "breadth": breadth_str,
        "btc_price": btc_price,
        "btc_change_1h": btc_change_1h,
        "adx_1h": round(adx_1h, 1),
        "adx_4h": round(adx_4h, 1),
        "bb_state_1h": bb_state_1h,
        "ma_1h": ma_align_1h,
        "pullback_pct": round(pullback_pct, 1),
        "atr_1h": round(btc_atr_1h, 1),
    }
