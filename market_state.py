# 市场状态矩阵 — 4H+1H 组合 → 方向偏倚
# v2: 基于回测的规则判断(n=1897 BTC 1H, 16个月数据, 覆盖多空完整周期)

def compute_market_state(tf_ind: dict) -> dict:
    """
    输入: {sym: {tf: ind_dict}}  ← 注意：取第一个币的 4H+1H 数据
    输出: {bias, confidence, state_desc}
    """
    # 用第一个币（BTC）代表整体市场方向
    if not tf_ind:
        return {"bias": "neutral", "confidence": 0, "desc": "无数据"}

    sym = next(iter(tf_ind.keys()), None)
    if not sym:
        return {"bias": "neutral", "confidence": 0, "desc": "无数据"}

    ind_data = tf_ind.get(sym, {})
    ind_4h = ind_data.get("4h", {})
    ind_1h = ind_data.get("1h", {})

    if not ind_4h or not ind_1h:
        return {"bias": "neutral", "confidence": 0, "desc": "数据不足"}

    # ── 1. 4H 趋势强度 ──
    adx_4h = ind_4h.get("adx", 0) or 0
    ma_align_4h = ind_4h.get("ma_alignment", "neutral")
    ma20_4h = ind_4h.get("ma20")
    ma_slope_4h = _ma_slope(ind_4h, "ma20", lookback=3)

    # ── 2. 1H 回调深度 ──
    close_1h = ind_1h.get("close")
    pullback_pct = 0
    if ma20_4h and close_1h and ma20_4h > 0:
        pullback_pct = (close_1h - ma20_4h) / ma20_4h * 100

    # ── 3. 压缩状态 ──
    bb_pct = ind_4h.get("bb_width_short_pct", 50)

    # ── 4. 量方向 ──
    vr_1h = ind_1h.get("volume_ratio") or 1

    # ── 判定 ──
    trend_strength = ""
    if adx_4h >= 30:
        trend_strength = "强"
    elif adx_4h >= 20:
        trend_strength = "中"
    else:
        trend_strength = "弱"

    # 压缩描述
    compression = ""
    if bb_pct is not None and bb_pct <= 20:
        compression = "紧"
    elif bb_pct is not None and bb_pct >= 60:
        compression = "宽"
    else:
        compression = "中"

    # 回调深度描述
    pullback_desc = ""
    if abs(pullback_pct) < 1:
        pullback_desc = "近(+0%)" if pullback_pct >= 0 else "近(-0%)"
    else:
        direction = "+" if pullback_pct >= 0 else ""
        pullback_desc = f"深({direction}{pullback_pct:.0f}%)" if abs(pullback_pct) >= 3 else f"中({direction}{pullback_pct:.0f}%)"

    # ── 方向偏倚（回测 v2, n=1897 BTC 1H, 16个月多空完整周期）──
    # 整体: 49.6%多 / 50.4%空 — 接近随机
    # 关键发现: 趋势方向+ADX强度 组合决定偏倚方向
    #   ↑ 多头ADX高≠衰竭(52%), 空头ADX高=持续(59%)
    #   ↑ ADX低+反方向=反转信号(多头低ADX→65%多, 空头低ADX→62%多=反转)
    bias = "neutral"
    conf = 0
    bias_reason = ""

    if ma_align_4h == "bullish":
        if adx_4h >= 30:
            bias = "long"
            conf = 52
            bias_reason = "4H强多→持续(回测52%多,n=277)"
        elif adx_4h >= 20:
            bias = "long"
            conf = 57
            bias_reason = "4H温和多→持续(回测57%多,n=130)"
        else:
            bias = "long"
            conf = 65
            bias_reason = "4H弱多→强势多(回测65%多,n=141)"
    elif ma_align_4h == "bearish":
        if adx_4h >= 30:
            bias = "short"
            conf = 59
            bias_reason = "4H强空→持续(回测59%空,n=329)"
        elif adx_4h >= 20:
            bias = "short"
            conf = 60
            bias_reason = "4H温和空→持续(回测60%空,n=177)"
        else:
            bias = "long"
            conf = 62
            bias_reason = "4H弱空→反转多(回测62%多,n=86)"
    else:
        # neutral — 整体 51/49, 不加方向偏倚
        bias = "neutral"
        conf = 0
        bias_reason = "4H中性-无方向偏倚"

    # 压缩调整
    if compression == "紧" and bias != "neutral":
        conf = min(conf + 10, 90)
        bias_reason += "+压缩"

    if bb_pct is not None and bb_pct >= 70 and adx_4h < 20:
        bias = "neutral"
        conf = 0
        bias_reason = "宽幅震荡-等方向"

    state_desc = f"4H{trend_strength}{'多' if ma_align_4h == 'bullish' else '空' if ma_align_4h == 'bearish' else '中'}(ADX{adx_4h:.0f})"
    state_desc += f" | 回调:{pullback_desc}"
    state_desc += f" | 压缩:{compression}(BB{bb_pct:.0f}%)"
    if vr_1h >= 1.5:
        state_desc += " | 放量中"

    return {
        "bias": bias,
        "confidence": conf,
        "desc": state_desc,
        "reason": bias_reason,
        "adx_4h": round(adx_4h, 1),
        "ma_4h": ma_align_4h,
        "pullback_pct": round(pullback_pct, 1),
        "bb_pct": round(bb_pct, 1) if bb_pct else None,
    }


def _ma_slope(ind: dict, ma_key: str, lookback: int = 3) -> float | None:
    """计算 MA 的最近 N 根 K 的斜率 (% per bar)"""
    df = ind.get("df")
    if df is None or len(df) < lookback + 1:
        return None
    ma_vals = []
    for i in range(-lookback, 0):
        val = df[ma_key].iloc[i] if ma_key in df.columns else None
        if val and val == val:
            ma_vals.append(val)
    if len(ma_vals) < 2:
        return None
    return (ma_vals[-1] - ma_vals[0]) / ma_vals[0] * 100 / lookback
