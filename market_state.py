def compute_market_state(tf_ind: dict) -> dict:
    """从全量指标池生成市场状态报告, 聚焦 BTC 做风向标"""
    from support_resistance import find_swing_levels, get_nearest_levels
    from utils import fmt_price

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

    # ── BTC 现价 (5m close 为实时价格) ──
    btc_price = (ind_5m or ind_1h).get("close")
    btc_atr_1h = ind_1h.get("atr") or 1
    btc_atr_4h = (ind_4h or {}).get("atr") or 1
    btc_change_1h = ind_1h.get("roc", 0) or 0

    def _pullback(price, ma):
        if not price or not ma or ma <= 0:
            return None
        return (price - ma) / ma * 100

    # ── 各周期 MA20 距离 ──
    ma20_5m = ind_5m.get("ma20") if ind_5m else None
    ma20_1h = ind_1h.get("ma20")
    ma20_4h = (ind_4h or {}).get("ma20")
    pb_5m = _pullback(btc_price, ma20_5m)
    pb_1h = _pullback(btc_price, ma20_1h)
    pb_4h = _pullback(btc_price, ma20_4h)

    ma20_parts = []
    for label, val in [("5m", pb_5m), ("1H", pb_1h), ("4H", pb_4h)]:
        if val is not None:
            ma20_parts.append(f"{label}{val:+.1f}%")
    ma20_line = " · ".join(ma20_parts) if ma20_parts else ""

    # ── 1H 指标 ──
    adx_1h = ind_1h.get("adx", 0) or 0
    ma_align_1h = ind_1h.get("ma_alignment", "neutral")
    bb_state_1h = ind_1h.get("bb_state", "unknown")
    vr_1h = ind_1h.get("volume_ratio") or 1

    # ── 4H 趋势 ──
    adx_4h = (ind_4h or {}).get("adx", 0) or 0
    ma_align_4h = (ind_4h or {}).get("ma_alignment", "neutral")
    ma60_4h = (ind_4h or {}).get("ma60")
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
    sr_1h_str = ""
    df_1h = ind_1h.get("df")
    if df_1h is not None and len(df_1h) >= 20 and btc_price:
        try:
            levels = find_swing_levels(df_1h, lookback=50)
            support, resistance = get_nearest_levels(levels, btc_price)
            parts = []
            if support:
                p = support.price
                parts.append(f"支撑:{fmt_price(p, btc_atr_1h)}({support.strength},{support.touch_count}触)")
            if resistance:
                p = resistance.price
                parts.append(f"阻力:{fmt_price(p, btc_atr_1h)}({resistance.strength},{resistance.touch_count}触)")
            sr_1h_str = " · ".join(parts) if parts else ""
        except Exception:
            pass

    # ── 4H S/R 位 ──
    sr_4h_str = ""
    df_4h = ind_4h.get("df")
    if df_4h is not None and len(df_4h) >= 20 and btc_price:
        try:
            levels_4h = find_swing_levels(df_4h, lookback=30)
            support_4h, resistance_4h = get_nearest_levels(levels_4h, btc_price)
            parts = []
            if support_4h:
                p = support_4h.price
                parts.append(f"支撑:{fmt_price(p, btc_atr_4h)}({support_4h.strength},{support_4h.touch_count}触)")
            if resistance_4h:
                p = resistance_4h.price
                parts.append(f"阻力:{fmt_price(p, btc_atr_4h)}({resistance_4h.strength},{resistance_4h.touch_count}触)")
            sr_4h_str = " · ".join(parts) if parts else ""
        except Exception:
            pass

    # ── 方向判定: 1H 定方向, 4H 调置信度 ──
    bias = "neutral"
    conf = 0
    bias_reason = ""
    dir_1h_label = "1H中性"
    dir_4h_label = ""

    if ma_align_1h == "bullish" and adx_1h >= 20:
        bias = "long"
        conf = min(50 + adx_1h * 0.5, 75)
        dir_1h_label = f"1H多头(ADX{adx_1h:.0f})"
    elif ma_align_1h == "bearish" and adx_1h >= 20:
        bias = "short"
        conf = min(50 + adx_1h * 0.5, 75)
        dir_1h_label = f"1H空头(ADX{adx_1h:.0f})"
    elif adx_1h < 15:
        bias = "neutral"
        conf = 0
        dir_1h_label = f"1H震荡(ADX{adx_1h:.0f})"
        bias_reason = f"1H震荡 ADX={adx_1h:.0f}"
    else:
        bias = "neutral"
        conf = 0
        dir_1h_label = f"1H中性(ADX{adx_1h:.0f})"
        bias_reason = f"1H中性 ADX={adx_1h:.0f}"

    # 4H 标签
    if adx_4h >= 20 and ma_align_4h == "bullish":
        dir_4h_label = f"4H多头(ADX{adx_4h:.0f})"
    elif adx_4h >= 20 and ma_align_4h == "bearish":
        dir_4h_label = f"4H空头(ADX{adx_4h:.0f})"
    elif above_ma20_4h and above_ma60_4h:
        dir_4h_label = f"4H偏多(ADX{adx_4h:.0f})"
    elif not above_ma20_4h and not above_ma60_4h:
        dir_4h_label = f"4H偏空(ADX{adx_4h:.0f})"
    else:
        dir_4h_label = f"4H中性(ADX{adx_4h:.0f})"

    # 4H 校验
    bias_4h = ""
    if bias != "neutral":
        align = (bias == "long" and ma_align_4h == "bullish") or (bias == "short" and ma_align_4h == "bearish")
        oppose = (bias == "long" and ma_align_4h == "bearish") or (bias == "short" and ma_align_4h == "bullish")
        if align and adx_4h >= 20:
            conf = min(conf + 15, 85)
            bias_4h = "✓4H"
        elif oppose and adx_4h >= 20:
            conf = max(conf - 20, 30)
            bias_4h = "⚠逆4H"
        elif align:
            bias_4h = "趋4H"  # weak alignment

    if not bias_reason:
        bias_reason = f"{dir_1h_label}"
    if bias_4h:
        bias_reason += f" {bias_4h}"

    # ── 1H 多空分布 ──
    bull_1h = sum(
        1 for s in tf_ind
        if (tf_ind[s].get("1h") or {}).get("ma_alignment") == "bullish"
    )
    bear_1h = sum(
        1 for s in tf_ind
        if (tf_ind[s].get("1h") or {}).get("ma_alignment") == "bearish"
    )
    neutral_1h = sum(
        1 for s in tf_ind
        if (tf_ind[s].get("1h") or {}).get("ma_alignment") == "neutral"
    )
    total_1h = bull_1h + bear_1h + neutral_1h

    breadth_1h = ""
    if total_1h > 5:
        breadth_1h = f"1H 多{bull_1h/total_1h*100:.0f}%/空{bear_1h/total_1h*100:.0f}%"

    # ── 4H 多空分布 ──
    bull_4h = sum(
        1 for s in tf_ind
        if (tf_ind[s].get("4h") or {}).get("ma_alignment") == "bullish"
    )
    bear_4h = sum(
        1 for s in tf_ind
        if (tf_ind[s].get("4h") or {}).get("ma_alignment") == "bearish"
    )
    neutral_4h = sum(
        1 for s in tf_ind
        if (tf_ind[s].get("4h") or {}).get("ma_alignment") == "neutral"
    )
    total_4h = bull_4h + bear_4h + neutral_4h

    breadth_4h = ""
    if total_4h > 5:
        breadth_4h = f"4H 多{bull_4h/total_4h*100:.0f}%/空{bear_4h/total_4h*100:.0f}%"

    # ── 生成 desc ──
    btc_disp = f"{btc_price:.0f}" if btc_price and btc_price > 100 else f"{btc_price:.5f}" if btc_price else "-"
    desc_parts = [f"BTC {btc_disp}"]

    if btc_change_1h:
        desc_parts.append(f"{btc_change_1h:+.2f}%")
    desc_parts.append(trend_4h)

    # 1H 行
    h1_type = "多头" if ma_align_1h == "bullish" else "空头" if ma_align_1h == "bearish" else "中性"
    h1_line = f"1H{h1_type}(ADX{adx_1h:.0f})"
    if bb_state_1h == "contracting":
        h1_line += " | BB收缩"
    elif bb_state_1h == "expanding":
        h1_line += " | BB扩张"
    if vr_1h >= 1.5:
        h1_line += " | 放量"

    return {
        "bias": bias,
        "confidence": round(conf),
        "desc": " | ".join(desc_parts),
        "h1_line": h1_line,
        "ma20_line": ma20_line,
        "reason": bias_reason,
        "sr_1h": sr_1h_str,
        "sr_4h": sr_4h_str,
        "breadth_1h": breadth_1h,
        "breadth_4h": breadth_4h,
        "btc_price": btc_price,
        "btc_change_1h": btc_change_1h,
        "adx_1h": round(adx_1h, 1),
        "adx_4h": round(adx_4h, 1),
        "bb_state_1h": bb_state_1h,
        "ma_1h": ma_align_1h,
        "ma_4h": ma_align_4h,
        "atr_1h": round(btc_atr_1h, 1),
    }
