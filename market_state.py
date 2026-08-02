# 行情状态机 — 回测验证的 5 态 (20标的60天, 1:1 胜率)
#   强顺势(d>=1.0) 60.5% | 中顺势(0.5-1.0) 55.7% | 贴均线(0-0.5) 49.4%
#   浅回调(-1~0) 44.0% | 深回调(<-1) 35.1% | 无趋势 48.7%
# 结论: 只交易 顺势距离>=0.5 (强+中顺势), 其余回避

STRONG_TREND_DIST = 1.0    # 强顺势阈值
TRADE_MIN_DIST = 0.5       # 可交易最小顺势距离 (ATR 归一化)


def market_regime(bias: str, close_4h, ma20_4h, atr_4h, direction: str) -> str:
    """行情状态判定 — 返回 strong_trend/mid_trend/at_ma/pullback/no_trend

    bias: 宏观方向 long/short/neutral
    direction: 信号方向 long/short
    顺势距离 = (close_4h - ma20_4h) / atr_4h, 按信号方向对称化
    """
    if bias == "neutral" or not close_4h or not ma20_4h or not atr_4h or atr_4h <= 0:
        return "no_trend"
    dist = (close_4h - ma20_4h) / atr_4h
    align_dist = dist if direction == "long" else -dist

    if align_dist >= STRONG_TREND_DIST:
        return "strong_trend"
    if align_dist >= TRADE_MIN_DIST:
        return "mid_trend"
    if align_dist >= 0:
        return "at_ma"
    return "pullback"


def is_tradable_regime(regime: str) -> bool:
    """该行情状态是否可交易 (回测: 只有强+中顺势有正 edge)"""
    return regime in ("strong_trend", "mid_trend")


# ────────────────────────────────────────────────────────────────
# 场景引擎 (3年1094天平衡数据验证, 1:1 严格口径, 同bar双命中跳过)
# 场景 = 信号方向 × (日线bias × 4H状态) — 顺大逆小是唯一稳定edge
# ────────────────────────────────────────────────────────────────

# 场景 → (标签, 3年胜率区间, 按TP增大趋势)
SCENE_WR = {
    "episode_long":   ("顺日逆时多",  "55-57%"),
    "episode_short":  ("顺日逆时空",  "57-61%"),
    "follow_long":    ("顺势多",  "53-55%"),
    "follow_short":   ("顺势空",  "54-57%"),
    "counter_long":   ("逆势多",  "45%"),
    "counter_short":  ("逆势空",  "47%"),
}

# 场景 → 期望增益 (置信度乘数)
SCENE_BOOST = {
    "episode_long": 1.08, "episode_short": 1.08,
    "follow_long": 1.04, "follow_short": 1.04,
    "counter_long": 0.85, "counter_short": 0.85,
}


def _t4_state(ind_4h: dict) -> str:
    """4H 状态: long/short/neutral (MA排列优先, close vs MA20 兜底)"""
    if not ind_4h:
        return "neutral"
    ma_align = ind_4h.get("ma_alignment", "neutral")
    if ma_align == "bullish":
        return "long"
    if ma_align == "bearish":
        return "short"
    close_4h = ind_4h.get("close")
    ma20_4h = ind_4h.get("ma20")
    if close_4h and ma20_4h:
        return "long" if close_4h > ma20_4h else "short"
    return "neutral"


def scene_of(direction: str, ind_4h: dict, bias: str) -> str:
    """场景判定:
    episode_* = 日线顺势 + 4H逆向(顺日逆时) — 最优 (顺大逆小)
    follow_*  = 日线+4H同向(全顺势)
    counter_* = 逆日线方向 — 3年验证 44-47% 低期望
    neutral   = 无趋势
    """
    if bias == "neutral" or direction not in ("long", "short"):
        return "neutral"
    t4 = _t4_state(ind_4h)
    if t4 == "neutral":
        return "neutral"
    if direction == "long":
        if bias == "long" and t4 == "short":
            return "episode_long"
        if bias == "long" and t4 == "long":
            return "follow_long"
        return "counter_long"
    else:
        if bias == "short" and t4 == "long":
            return "episode_short"
        if bias == "short" and t4 == "short":
            return "follow_short"
        return "counter_short"


def _empty_state(desc: str) -> dict:
    return {
        "bias": "neutral", "confidence": 0, "desc": desc, "btc_price": None,
        "h1_line": "", "ma20_line": "", "reason": "", "sr_1h": "", "sr_4h": "",
        "breadth_1h": "", "breadth_4h": "", "btc_change_1h": None,
        "adx_1h": 0, "adx_4h": 0, "bb_state_1h": "unknown",
        "ma_1h": "neutral", "ma_4h": "neutral", "atr_1h": 0,
    }


def compute_market_state(tf_ind: dict) -> dict:
    """从全量指标池生成市场状态报告, 聚焦 BTC 做风向标"""
    from support_resistance import find_swing_levels, get_nearest_levels
    from market_phase import analyze_market_state as _phase
    from market_phase import label_of as _phase_label
    from power_balance import analyze_power as _power
    from power_balance import label_of as _power_label
    from utils import fmt_price

    btc_key = "BTC/USDT:USDT"
    sym = btc_key if btc_key in tf_ind else next(iter(tf_ind.keys()), None)
    if not sym:
        return _empty_state("无数据")

    ind_data = tf_ind.get(sym, {})
    ind_4h = ind_data.get("4h", {})
    ind_1h = ind_data.get("1h", {})
    ind_5m = ind_data.get("5m", {})

    if not ind_1h:
        return _empty_state("数据不足")

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

    # ── 市场状态递进分析 (用户方法论: 日线→4H→1H, 每周期看90根) ──
    phase_line = ""
    power_line = ""
    try:
        df_4h = ind_4h.get("df")
        df_1h = ind_1h.get("df")
        parts_phase = []
        if df_4h is not None and len(df_4h) >= 90:
            # 日线 (4H 合成) — live df 是 RangeIndex, 需要 timestamp 建索引
            daily_df = None
            try:
                tmp = df_4h.copy()
                if "timestamp" in tmp.columns:
                    tmp.index = pd.to_datetime(tmp["timestamp"])
                daily_df = tmp.resample("1D").agg(
                    {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
                ).dropna()
            except Exception:
                daily_df = None
            if daily_df is not None and len(daily_df) >= 70:
                ms_d = _phase(daily_df)
                parts_phase.append(f"日线:{_phase_label(ms_d)}")
            ms_4h = _phase(df_4h.tail(90).reset_index(drop=True))
            parts_phase.append(f"4H:{_phase_label(ms_4h)}")
            pw_4h = _power(df_4h.tail(60).reset_index(drop=True))
            power_line = f"4H力量:{_power_label(pw_4h)}"
        if df_1h is not None and len(df_1h) >= 90:
            ms_1h = _phase(df_1h.tail(90).reset_index(drop=True))
            parts_phase.append(f"1H:{_phase_label(ms_1h)}")
            pw_1h = _power(df_1h.tail(60).reset_index(drop=True))
            power_line += f" 1H力量:{_power_label(pw_1h)}" if power_line else f"1H力量:{_power_label(pw_1h)}"
        phase_line = " · ".join(parts_phase)
    except Exception:
        phase_line = ""

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

    # ── 方向判定: 4H 定方向 (回测: 4H bias 是唯一方向 edge), 1H 仅展示 ──
    bias = "neutral"
    conf = 0
    bias_reason = ""
    dir_1h_label = "1H中性"
    dir_4h_label = ""

    # 1H 标签 (仅展示)
    if ma_align_1h == "bullish":
        dir_1h_label = f"1H多头(ADX{adx_1h:.0f})"
    elif ma_align_1h == "bearish":
        dir_1h_label = f"1H空头(ADX{adx_1h:.0f})"
    elif adx_1h < 15:
        dir_1h_label = f"1H震荡(ADX{adx_1h:.0f})"
    else:
        dir_1h_label = f"1H中性(ADX{adx_1h:.0f})"

    # 4H 定方向 (与 _symbol_bias/_macro_bias 同款: 4H MA20/MA60+ADX 稳定趋势)
    if adx_4h >= 20 and ma_align_4h == "bullish":
        bias = "long"
        conf = min(50 + adx_4h * 0.5, 75)
        dir_4h_label = f"4H多头(ADX{adx_4h:.0f})"
    elif adx_4h >= 20 and ma_align_4h == "bearish":
        bias = "short"
        conf = min(50 + adx_4h * 0.5, 75)
        dir_4h_label = f"4H空头(ADX{adx_4h:.0f})"
    elif above_ma20_4h and above_ma60_4h:
        bias = "long"
        conf = 45
        dir_4h_label = f"4H偏多(ADX{adx_4h:.0f})"
    elif not above_ma20_4h and not above_ma60_4h:
        bias = "short"
        conf = 45
        dir_4h_label = f"4H偏空(ADX{adx_4h:.0f})"
    else:
        bias = "neutral"
        conf = 0
        dir_4h_label = f"4H中性(ADX{adx_4h:.0f})"

    # 1H 动量确认: 与 4H 同向时提置信度, 反向时降级 (仅影响置信度, 不改变方向)
    if bias != "neutral":
        h1_align = (bias == "long" and ma_align_1h == "bullish") or (bias == "short" and ma_align_1h == "bearish")
        h1_oppose = (bias == "long" and ma_align_1h == "bearish") or (bias == "short" and ma_align_1h == "bullish")
        if h1_align and adx_1h >= 20:
            conf = min(conf + 10, 80)
            bias_reason = f"{dir_4h_label} ✓1H同向"
        elif h1_oppose and adx_1h >= 20:
            conf = max(conf - 10, 30)
            bias_reason = f"{dir_4h_label} ⚠1H反向"
        else:
            bias_reason = f"{dir_4h_label}"
    else:
        bias_reason = dir_4h_label

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
        "phase_line": phase_line,
        "power_line": power_line,
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
