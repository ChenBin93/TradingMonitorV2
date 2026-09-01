#!/usr/bin/env python3
"""K线图表生成 — 供飞书预警图片推送

设计:
- 顶部信息栏: 4H 大方向/段龄 + 日线一致性 + 统计状态 + 4H/1H 距关键位距离
- 上图: K线 + BB40±2σ + MA60 + 1h 关键位 (支撑/阻力分色+价格) + 4h 关键位 (虚线) + 预警标注
- 下图: 成交量 (颜色随涨跌)
- 输出: /tmp/charts/{symbol}_{tf}.png (或指定目录)

依赖: matplotlib (Agg 后端, 无显示环境)
"""
import os

import numpy as np
import pandas as pd

# 颜色 (飞书深色背景友好)
C_UP = "#e15241"      # 涨 (红, 国内习惯)
C_DOWN = "#2db25d"    # 跌 (绿)
C_BB = "#8a8f99"      # 布林带
C_MA60 = "#e6a23c"    # MA60
C_SUP = "#2db25d"     # 支撑 (绿)
C_RES = "#e15241"     # 阻力 (红)
C_LEVEL = "#5a5f6b"   # 4h 关键位 (灰)
C_GRID = "#2a2e37"
C_BG = "#1e222d"
C_TEXT = "#d8dce6"
C_DIM = "#7a8194"     # 次要文字


def pick_levels(levels: list[dict], price: float, atr: float,
                max_dist_pct: float = 0.04, merge_pct: float = 0.004,
                max_each_side: int = 2) -> list[dict]:
    """关键位精选 — 解决图上位标注混乱 (支撑/阻力重叠 + 远位干扰)

    距离用价格百分比 (ATR 归一在宽位距标的上失效: BTC ATR 185 但位距 400+):
      1. 过滤距当前价 > max_dist_pct×price (默认 4%) 的远位
      2. 支撑/阻力价差 < merge_pct×price (默认 0.4%) 时视为同一密集区:
         保留触碰次数多的一个, 消解红绿重叠
      3. 每侧最多 max_each_side 条 (触碰次数优先), 输出按价格排序
    只影响图表标注, 不改研究/检测逻辑。
    """
    if not levels or price <= 0:
        return []
    cur = float(price)
    max_d = max_dist_pct * cur
    merge_d = merge_pct * cur
    # 1) 距离过滤 + 信息增强
    cand = []
    for lv in levels:
        px = float(lv["price"])
        if abs(px - cur) > max_d:
            continue
        cand.append({
            "price": px, "side": lv.get("side", ""),
            "touch": int(lv.get("touch", 0) or lv.get("touch_count", 0) or 0),
            "dist": abs(px - cur),
        })
    # 2) 冲突消解: 跨 side 距离 < merge_d 的, 保留 touch 多的
    cand.sort(key=lambda x: x["price"])
    kept = []
    for lv in cand:
        merged = False
        for k in kept:
            if abs(k["price"] - lv["price"]) < merge_d:
                # 保留触碰多/距近的; 相同则保留支撑 (惯例: 下方支撑优先)
                if lv["touch"] > k["touch"] or (
                        lv["touch"] == k["touch"] and lv["dist"] < k["dist"]):
                    k["price"] = lv["price"]
                    k["side"] = lv["side"]
                    k["touch"] = lv["touch"]
                    k["dist"] = lv["dist"]
                merged = True
                break
        if not merged:
            kept.append(dict(lv))
    # 3) 每侧限数量 (触碰次数优先)
    out = []
    for side in ("support", "resistance"):
        side_lv = sorted([k for k in kept if k["side"] == side],
                         key=lambda x: (-x["touch"], x["dist"]))
        out.extend(side_lv[:max_each_side])
    return sorted(out, key=lambda x: x["price"])


def fused_levels(df: pd.DataFrame, price: float, atr: float,
                 n_bins: int = 40, merge_atr: float = 0.8,
                 max_dist_pct: float = 0.04, max_each_side: int = 1,
                 band_atr: float = 0.4) -> list[dict]:
    """融合关键位 — 成交量分布 (HVN) + swing 极点, 业界推荐做法

    原理: 关键位本质是"资金聚集处" (成交量大的价格区), swing 极点只是近似。
    输出:
      - 位价对齐真实极值: 聚类内最接近中心的 swing 极值 (影线端)
      - band 统一 = band_atr×ATR (所有周期视觉一致, 不过宽不过窄)
      - 每侧最多 max_each_side 条

    步骤:
      1. volume_profile 算 HVN 节点 + 自算 swing 极值
      2. 按支撑/阻力分组后组内聚类 (避免跨价格合并)
      3. 位价 = 聚类内最接近中心的 swing 极值; band = band_atr×ATR
    只影响图表标注, 不改研究/检测逻辑。
    """
    from volume_profile import compute_volume_profile

    if df is None or len(df) < 60 or price <= 0 or atr <= 0:
        return []
    cur = float(price)
    merge_d = merge_atr * atr

    # ── swing 极值 (含原始影线价: 波段高低点 K 线的 high/low) ──
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)
    swing_ext = []  # (price, side, idx) — price 为影线端价
    for i in range(2, n - 2):
        if highs[i] >= max(highs[i - 2:i + 3]):
            swing_ext.append((float(highs[i]), "resistance", i))
        if lows[i] <= min(lows[i - 2:i + 3]):
            swing_ext.append((float(lows[i]), "support", i))

    # ── 候选: HVN 量节点 + swing 极值 ──
    vp = compute_volume_profile(df, lookback=min(200, len(df)), n_bins=n_bins)
    cand = []  # (price, vol, src, ext_price)
    if vp:
        cand += [(float(h["price"]), float(h["volume_pct"]), "vp", None)
                 for h in vp.get("hvns", [])]
    for p, side, _i in swing_ext:
        cand.append((p, 0.0, f"swing_{side}", p))
    if not cand:
        return []

    # ── 按 side 分组聚类 (组内聚合并保留极值) ──
    def _cluster(cands):
        if not cands:
            return []
        cands.sort(key=lambda x: x[0])
        bands = []
        for p, vol, src, ext in cands:
            if bands and abs(p - bands[-1][0]) <= merge_d:
                c0, v0, s0, exts = bands[-1]
                w0 = v0 if v0 > 0 else 0.5
                w1 = vol if vol > 0 else 0.5
                new_c = (c0 * w0 + p * w1) / (w0 + w1)
                if ext is not None:
                    exts.append(ext)
                bands[-1] = (new_c, v0 + vol, s0 + "," + src, exts)
            else:
                bands.append((p, vol, src, [ext] if ext is not None else []))
        return bands

    cand_sup = [c for c in cand if c[0] < cur]
    cand_res = [c for c in cand if c[0] > cur]
    bands = [(p, v, s, e, "support") for p, v, s, e in _cluster(cand_sup)]
    bands += [(p, v, s, e, "resistance") for p, v, s, e in _cluster(cand_res)]

    # ── 位价对齐真实极值 + band 统一 (band_atr×ATR, 视觉一致) ──
    out = []
    for center, vol, src, exts, side in bands:
        if abs(center - cur) > max_dist_pct * cur:
            continue
        px = center
        if exts:
            # 最接近聚类中心的 swing 极值 → 落在影线端
            px = min(exts, key=lambda e: abs(e - center))
        out.append({
            "price": px, "side": side,
            "touch": len(exts), "vol_pct": vol,
            "dist": abs(px - cur), "src": src,
            "band": max(band_atr * atr, atr * 0.1),  # 统一带宽, 最小 0.1×ATR
        })
    final = []
    for side in ("support", "resistance"):
        side_lv = sorted([k for k in out if k["side"] == side],
                         key=lambda x: (-x["vol_pct"], -x["touch"], x["dist"]))
        final.extend(side_lv[:max_each_side])
    return sorted(final, key=lambda x: x["price"])


def make_chart(
    df: pd.DataFrame,
    symbol: str,
    tf: str,
    levels: list | None = None,
    levels_4h: list[float] | None = None,
    alerts: list[dict] | None = None,
    bb: tuple | None = None,
    ma60: pd.Series | None = None,
    info: dict | None = None,
    df_4h: pd.DataFrame | None = None,
    levels_4h_chart: list | None = None,
    df_15m: pd.DataFrame | None = None,
    levels_15m_chart: list | None = None,
    df_1d: pd.DataFrame | None = None,
    levels_1d_chart: list | None = None,
    extra_1d: list | None = None,
    extra_15m: list | None = None,
    extra_4h: list | None = None,
    extra_1h: list | None = None,
    out_dir: str = "/tmp/charts",
    n_bars: int = 48,
) -> str:
    """生成K线图 → 返回 PNG 路径

    df: DataFrame (DatetimeIndex 或 timestamp 列, 含 ohlcv) — 主图 1h
    levels: 1h 关键位, [{"price", "side", "band", "touch"}]
    levels_4h: 4h 关键位价格列表 (兼容旧调用, 已弃用)
    alerts: [{price, level, text}] 预警标注
    bb: (ma, upper, lower) 布林带三序列
    ma60: MA60 序列
    info: 顶部信息栏 dict
    df_4h: 4h K线 (右面板)
    levels_4h_chart: 4h 关键位 dict 列表 (带 band)
    df_15m: 15m K线 (最左面板)
    levels_15m_chart: 15m 关键位 dict 列表
    n_bars: 每面板显示根数 (默认 48)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    from matplotlib.patches import Rectangle
    from matplotlib import font_manager

    # 中文字体 (Noto Sans CJK, 安装: apt install fonts-noto-cjk)
    # ttc 集合需显式注册; 失败时回退 DejaVu (英文可显示, 中文变方块)
    _registered = False
    for _p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
               "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"):
        if os.path.exists(_p):
            try:
                font_manager.fontManager.addfont(_p)
                _registered = True
            except Exception:
                pass
    if _registered:
        # ttc 集合 addfont 只注册第一个 face (JP); JP face 含 CJK 字形, 简体可显示
        plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    else:
        for _f in ("Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei"):
            if any(_f in f.name for f in font_manager.fontManager.ttflist):
                plt.rcParams["font.sans-serif"] = [_f, "DejaVu Sans"]
                break
    plt.rcParams["axes.unicode_minus"] = False

    # 数据准备
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    df = df.sort_index()
    c_full = df["close"].to_numpy(float)  # 全量 close (BB/MA 用长历史)
    df = df.tail(n_bars)
    if len(df) < 20:
        return ""

    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    v = df["volume"].to_numpy(float) if "volume" in df.columns else np.ones(len(df))
    idx = df.index
    price_last = float(c[-1]) if len(c) else 0.0

    # 默认指标 (BB40/MA60 用全量历史计算, 取尾部显示)
    if bb is None:
        cs_full = pd.Series(c_full, dtype=float)
        sma = cs_full.rolling(40, min_periods=40).mean()
        sd = cs_full.rolling(40, min_periods=40).std(ddof=0)
        bb = (sma, sma + 2 * sd, sma - 2 * sd)
    ma, up, lo_ = bb
    if ma60 is None:
        ma60 = pd.Series(c_full, dtype=float).rolling(60, min_periods=60).mean()
    # 外部传入的指标序列按同一尾部对齐 (tail(n_bars) 后可能短于 df); 统一返回 numpy
    def _align(s):
        if s is None:
            return None
        if isinstance(s, pd.Series):
            s = s.to_numpy(float)
        s = np.asarray(s, float)
        return s[-len(df):] if len(s) >= len(df) else np.concatenate(
            [np.full(len(df) - len(s), np.nan), s])
    ma, up, lo_ = _align(ma), _align(up), _align(lo_)
    ma60 = _align(ma60)

    os.makedirs(out_dir, exist_ok=True)
    fname = f"{symbol.replace('/', '_').replace(':', '_')}_{tf}.png"
    path = os.path.join(out_dir, fname)

    # ── 布局: 信息栏(整宽) + 2×2 网格 (上排: 日线/4H, 下排: 1H/15M) ──
    has_15m = df_15m is not None and len(df_15m) >= 20
    has_4h = df_4h is not None and len(df_4h) >= 20
    has_1d = df_1d is not None and len(df_1d) >= 20
    # 网格: 每周期占一格 (价格+量上下)
    # 上排周期: [日线, 4H]; 下排周期: [1H, 15M]
    top_row = [x for x in (("1d", has_1d), ("4h", has_4h)) if x[1]]
    bot_row = [x for x in (("1h", True), ("15m", has_15m)) if x[1]]
    n_top = len(top_row)
    n_bot = len(bot_row)
    n_col = max(n_top, n_bot, 1)
    fig = plt.figure(figsize=(6.2 * n_col, 9.0), facecolor=C_BG)
    # 5 行网格: 0=信息栏, 1-2=上排(价格+量), 3-4=下排(价格+量)
    gs = fig.add_gridspec(5, n_col, height_ratios=[0.8, 3.0, 0.9, 3.0, 0.9],
                          hspace=0.30, wspace=0.20,
                          left=0.05, right=0.985, top=0.94, bottom=0.06)
    ax0 = fig.add_subplot(gs[0, :])   # 信息栏 (整宽)
    ax0.axis("off")
    plot_axes = []
    ax15 = ax15v = None
    ax1 = ax2 = None
    ax3 = ax4 = None
    axd = axdv = None
    # 上排: 日线 (col 0), 4H (col 1)
    col = 0
    if has_1d:
        axd = fig.add_subplot(gs[1, col])
        axdv = fig.add_subplot(gs[2, col], sharex=axd)
        plot_axes += [axd, axdv]
        col += 1
    if has_4h:
        ax3 = fig.add_subplot(gs[1, col])
        ax4 = fig.add_subplot(gs[2, col], sharex=ax3)
        plot_axes += [ax3, ax4]
    # 下排: 1H (col 0), 15M (col 1)
    col = 0
    ax1 = fig.add_subplot(gs[3, col])
    ax2 = fig.add_subplot(gs[4, col], sharex=ax1)
    plot_axes += [ax1, ax2]
    col += 1
    if has_15m:
        ax15 = fig.add_subplot(gs[3, col])
        ax15v = fig.add_subplot(gs[4, col], sharex=ax15)
        plot_axes += [ax15, ax15v]
    for ax in plot_axes:
        ax.set_facecolor(C_BG)
        ax.grid(True, color=C_GRID, linewidth=0.6, alpha=0.6)
        ax.tick_params(colors=C_TEXT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(C_GRID)

    # ── 顶部信息栏: 4H方向 / 日线一致性 / 统计状态 / 关键位距离 ──
    if info:
        def _seg_arrow(d):
            return {"up": "↑ 多头", "down": "↓ 空头"}.get(d, "— 震荡")
        seg4 = _seg_arrow(info.get("dow4h"))
        age4 = info.get("dow4h_age")
        seg4_s = f"{seg4}" + (f" {age4}根" if age4 else "")
        daily_s = {"顺风": "日线顺风", "逆风": "日线逆风", "单边": "日线单边",
                   "无风": "日线无风"}.get(info.get("cons"), "")
        stat_s = info.get("stat", "")
        d4 = str(info.get("dist4h", ""))
        d1 = str(info.get("dist1h", ""))
        line1 = (f"4H方向: {seg4_s}   |   {daily_s}   |   1H状态: {stat_s}")
        score_s = info.get("score", "")
        if score_s:
            line1 = f"🔥 {score_s}   |   " + line1
        line2 = (f"距关键位(支撑/阻力 ATR): 4H [{d4}]   1H [{d1}]")
        pos_ref = info.get("pos_ref", "")
        if pos_ref:
            line2 += f"   |   {pos_ref}"
        ax0.text(0.005, 0.72, line1, transform=ax0.transAxes,
                 fontsize=10, color=C_TEXT, va="center",
                 bbox=dict(boxstyle="round,pad=0.35", facecolor="#262b38",
                           edgecolor=C_GRID, linewidth=0.6))
        ax0.text(0.005, 0.16, line2, transform=ax0.transAxes,
                 fontsize=9, color=C_DIM, va="center",
                 bbox=dict(boxstyle="round,pad=0.35", facecolor="#262b38",
                           edgecolor=C_GRID, linewidth=0.6))

    # 辅助: 在面板上画布林带 + MA20 — 用全量历史计算, 只取尾部显示
    # close_full: 全量 close 序列 (未截断); x_arr: 面板 x 坐标 (长度 = 显示根数)
    def _draw_bb(ax, close_full, x_arr, period=20, std=2.0, n_show=None):
        n_show = n_show or len(x_arr)
        cs = pd.Series(close_full, dtype=float)
        sma = cs.rolling(period, min_periods=period).mean()
        sd = cs.rolling(period, min_periods=period).std(ddof=0)
        up = sma + std * sd
        lo = sma - std * sd
        ma20 = cs.rolling(20, min_periods=20).mean()
        # 取尾部 n_show 根 (与面板显示对齐)
        sma, up, lo, ma20 = (s.tail(n_show).to_numpy(float)
                             for s in (sma, up, lo, ma20))
        ax.fill_between(x_arr, up, lo, color=C_BB, alpha=0.10, zorder=1)
        ax.plot(x_arr, sma, color=C_BB, linewidth=0.8, alpha=0.65)
        ax.plot(x_arr, up, color=C_BB, linewidth=0.6, alpha=0.45, linestyle="--")
        ax.plot(x_arr, lo, color=C_BB, linewidth=0.6, alpha=0.45, linestyle="--")
        # MA20 (橙)
        ax.plot(x_arr, ma20, color="#d08770", linewidth=1.0, alpha=0.85,
                label="MA20")

    # 辅助: 画大周期参考位 (虚线 + 标签)
    def _draw_extra(ax, x_last, extra_list, label=""):
        for lv in (extra_list or []):
            if not isinstance(lv, dict):
                continue
            px_e = lv["price"]
            col_e = C_SUP if lv.get("side") == "support" else C_RES
            ax.axhline(px_e, color=col_e, linewidth=0.9, alpha=0.5,
                       linestyle="--", zorder=1.5)
            ax.text(x_last, px_e, f"  {label}{px_e:.0f}",
                    color=col_e, fontsize=7, va="center", ha="right",
                    alpha=0.9,
                    bbox=dict(boxstyle="round,pad=0.12", facecolor=C_BG,
                              edgecolor=col_e, linewidth=0.3, alpha=0.7))

    # ── 主图: K线 (x 轴 = 距最新 K 线的根数, 0 为最新) ──
    x = np.arange(len(df))
    width = 0.65
    for i in range(len(df)):
        color = C_UP if c[i] >= o[i] else C_DOWN
        ax1.plot([x[i], x[i]], [l[i], h[i]], color=color, linewidth=0.8, zorder=2)
        ax1.add_patch(Rectangle(
            (x[i] - width / 2, min(o[i], c[i])), width, abs(c[i] - o[i]) or 1e-6,
            facecolor=color, edgecolor=color, linewidth=0.5, zorder=3))

    # 布林带 + MA20 (全量历史, 与其他周期统一)
    _draw_bb(ax1, c_full, x, period=20, std=2.0)

    # 1h 关键位: 支撑绿 / 阻力红, 浅色 band 区域 + 清晰中线 + 价格标签
    if levels:
        for lv in levels:
            if isinstance(lv, dict):
                px = lv["price"]
                side = lv.get("side", "")
                band = lv.get("band") or 0.0
                col = C_SUP if side == "support" else C_RES if side == "resistance" else C_LEVEL
                # 位带区域: 浅色填充 (弱化, 不干扰K线)
                if band > 0:
                    ax1.axhspan(px - band, px + band, color=col, alpha=0.05,
                                linewidth=0, zorder=0.5)
                # 中线 = 真实极值 (影线端), 清晰但不过度
                ax1.axhline(px, color=col, linewidth=1.1, alpha=0.8, linestyle="-")
                # 价格标签: 线右侧, 带轻微底色
                ax1.text(x[-1], px, f"  {px:.0f}",
                         color=col, fontsize=8, va="center", ha="right",
                         alpha=1.0, fontweight="bold",
                         bbox=dict(boxstyle="round,pad=0.15", facecolor=C_BG,
                                   edgecolor=col, linewidth=0.4, alpha=0.8))
            else:
                ax1.axhline(lv, color=C_LEVEL, linewidth=0.8, alpha=0.5, linestyle=":")
    # 1H 面板: 最近的大周期位 (虚线参考, 只叠加最近的 1-2 条)
    if extra_1h:
        for lv in extra_1h:
            if isinstance(lv, dict):
                px_e = lv["price"]
                col_e = C_SUP if lv.get("side") == "support" else C_RES
                ax1.axhline(px_e, color=col_e, linewidth=0.9, alpha=0.5,
                            linestyle="--", zorder=1.5)
                ax1.text(x[-1], px_e, f"  ↑{px_e:.0f}",
                         color=col_e, fontsize=7, va="center", ha="right",
                         alpha=0.9,
                         bbox=dict(boxstyle="round,pad=0.12", facecolor=C_BG,
                                   edgecolor=col_e, linewidth=0.3, alpha=0.7))
    # 预警标注移动到所有面板渲染之后 (需 x4/x15 已定义)

    ax1.set_title(f"{symbol}  {tf}  (横轴: 距最新K线根数)",
                  color=C_TEXT, fontsize=11, loc="left")
    ax1.legend(loc="lower left", fontsize=7, facecolor=C_BG, edgecolor=C_GRID,
               labelcolor=C_TEXT)  # lower left: 避开左上角预警标注
    ax1.set_ylabel("Price", color=C_TEXT, fontsize=8)
    # K线图横轴 = 距最新 K 线的根数 (显式设置, 不依赖 VOL 的 sharex)
    ax1.xaxis.set_major_locator(mtick.MaxNLocator(6))
    ax1.xaxis.set_major_formatter(mtick.FuncFormatter(
        lambda t, _p: f"-{int(len(df) - 1 - t)}" if t < len(df) - 1 else "0"))
    ax1.tick_params(axis="x", colors=C_TEXT, labelsize=7)

    # ── 下图: 成交量 ──
    vol_colors = [C_UP if c[i] >= o[i] else C_DOWN for i in range(len(df))]
    ax2.bar(x, v, color=vol_colors, width=width, alpha=0.7)
    ax2.set_ylabel("Vol", color=C_TEXT, fontsize=8)

    # ── 左: 15M 短线面板 (最近 n_bars 根 ≈ 12小时) ──
    if ax15 is not None and df_15m is not None:
        d15_full = df_15m.copy()
        if "timestamp" in d15_full.columns:
            d15_full = d15_full.set_index("timestamp")
        d15_full = d15_full.sort_index()
        c15_full = d15_full["close"].to_numpy(float)  # 全量 close (布林/MA 用)
        d15 = d15_full.tail(n_bars)
        o15 = d15["open"].to_numpy(float)
        h15 = d15["high"].to_numpy(float)
        l15 = d15["low"].to_numpy(float)
        c15 = d15["close"].to_numpy(float)
        v15 = d15["volume"].to_numpy(float) if "volume" in d15.columns else np.ones(len(d15))
        x15 = np.arange(len(d15))  # 根数坐标, 0=最老, n-1=最新
        w15 = 0.65
        for i in range(len(d15)):
            col15 = C_UP if c15[i] >= o15[i] else C_DOWN
            ax15.plot([x15[i], x15[i]], [l15[i], h15[i]], color=col15, linewidth=0.6)
            ax15.add_patch(Rectangle(
                (x15[i] - w15 / 2, min(o15[i], c15[i])), w15,
                abs(c15[i] - o15[i]) or 1e-6,
                facecolor=col15, edgecolor=col15, linewidth=0.35))
        # 15M 布林带 + MA20 (全量历史计算, 尾部显示)
        _draw_bb(ax15, c15_full, x15)
        # 15M 关键位 (带 band)
        for lv in (levels_15m_chart or []):
            if isinstance(lv, dict):
                px15 = lv["price"]
                b15 = lv.get("band") or 0.0
                col15 = C_SUP if lv.get("side") == "support" else C_RES
                if b15 > 0:
                    ax15.axhspan(px15 - b15, px15 + b15, color=col15, alpha=0.05, linewidth=0)
                ax15.axhline(px15, color=col15, linewidth=1.0, alpha=0.8)
                ax15.text(x15[-1], px15, f"  {px15:.0f}",
                          color=col15, fontsize=7, va="center", ha="right",
                          alpha=1.0, fontweight="bold",
                          bbox=dict(boxstyle="round,pad=0.12", facecolor=C_BG,
                                    edgecolor=col15, linewidth=0.3, alpha=0.8))
            else:
                ax15.axhline(lv, color=C_LEVEL, linewidth=0.7, alpha=0.5, linestyle="--")
        ax15.axhline(price_last, color=C_MA60, linewidth=0.7, alpha=0.4, linestyle=":")
        _draw_extra(ax15, x15[-1], extra_15m, "↑")  # 最近大周期位参考
        ax15.set_title(f"15M (最近 {len(d15)} 根)", color=C_DIM, fontsize=9, loc="left")
        ax15.set_ylabel("15M", color=C_TEXT, fontsize=8)
        # K线图横轴 = 距最新 K 线的根数
        ax15.xaxis.set_major_locator(mtick.MaxNLocator(6))
        ax15.xaxis.set_major_formatter(mtick.FuncFormatter(
            lambda t, _p: f"-{int(len(d15) - 1 - t)}" if t < len(d15) - 1 else "0"))
        ax15.tick_params(axis="x", colors=C_TEXT, labelsize=7)
        vol15 = [C_UP if c15[i] >= o15[i] else C_DOWN for i in range(len(d15))]
        ax15v.bar(x15, v15, color=vol15, width=w15, alpha=0.7)
        ax15v.set_ylabel("Vol", color=C_TEXT, fontsize=8)
        # 横轴 = 距最新 K 线的根数 (0 = 最新)
        ax15v.xaxis.set_major_locator(mtick.MaxNLocator(6))
        ax15v.xaxis.set_major_formatter(mtick.FuncFormatter(
            lambda t, _p: f"-{int(len(d15) - 1 - t)}" if t < len(d15) - 1 else "0"))
        for lbl in ax15v.get_xticklabels():
            lbl.set_rotation = 0

    # ── 日线面板 (最左, 大周期) ──
    if axd is not None and df_1d is not None:
        dd_full = df_1d.copy()
        if "timestamp" in dd_full.columns:
            dd_full = dd_full.set_index("timestamp")
        dd_full = dd_full.sort_index()
        cd_full = dd_full["close"].to_numpy(float)  # 全量 close
        dd = dd_full.tail(n_bars)
        od = dd["open"].to_numpy(float)
        hd = dd["high"].to_numpy(float)
        ld = dd["low"].to_numpy(float)
        cd = dd["close"].to_numpy(float)
        vd = dd["volume"].to_numpy(float) if "volume" in dd.columns else np.ones(len(dd))
        xd = np.arange(len(dd))
        wd = 0.65
        for i in range(len(dd)):
            cold = C_UP if cd[i] >= od[i] else C_DOWN
            axd.plot([xd[i], xd[i]], [ld[i], hd[i]], color=cold, linewidth=0.8)
            axd.add_patch(Rectangle(
                (xd[i] - wd / 2, min(od[i], cd[i])), wd, abs(cd[i] - od[i]) or 1e-6,
                facecolor=cold, edgecolor=cold, linewidth=0.45))
        _draw_bb(axd, cd_full, xd)  # 日线布林带 + MA20
        # 日线关键位 (带 band)
        for lv in (levels_1d_chart or []):
            if isinstance(lv, dict):
                pxd = lv["price"]
                bd = lv.get("band") or 0.0
                cold = C_SUP if lv.get("side") == "support" else C_RES
                if bd > 0:
                    axd.axhspan(pxd - bd, pxd + bd, color=cold, alpha=0.05, linewidth=0)
                axd.axhline(pxd, color=cold, linewidth=1.0, alpha=0.8)
                axd.text(xd[-1], pxd, f"  {pxd:.0f}",
                          color=cold, fontsize=7, va="center", ha="right",
                          alpha=1.0, fontweight="bold",
                          bbox=dict(boxstyle="round,pad=0.12", facecolor=C_BG,
                                    edgecolor=cold, linewidth=0.3, alpha=0.8))
            else:
                axd.axhline(lv, color=C_LEVEL, linewidth=0.7, alpha=0.5, linestyle="--")
        axd.axhline(price_last, color=C_MA60, linewidth=0.7, alpha=0.4, linestyle=":")
        _draw_extra(axd, xd[-1], extra_1d, "↑")  # 最近大周期位参考
        axd.set_title(f"日线 (最近 {len(dd)} 根)", color=C_DIM, fontsize=9, loc="left")
        axd.set_ylabel("1D", color=C_TEXT, fontsize=8)
        axd.xaxis.set_major_locator(mtick.MaxNLocator(6))
        axd.xaxis.set_major_formatter(mtick.FuncFormatter(
            lambda t, _p: f"-{int(len(dd) - 1 - t)}" if t < len(dd) - 1 else "0"))
        axd.tick_params(axis="x", colors=C_TEXT, labelsize=7)
        vold = [C_UP if cd[i] >= od[i] else C_DOWN for i in range(len(dd))]
        axdv.bar(xd, vd, color=vold, width=wd, alpha=0.7)
        axdv.set_ylabel("Vol", color=C_TEXT, fontsize=8)
        axdv.xaxis.set_major_locator(mtick.MaxNLocator(6))
        axdv.xaxis.set_major_formatter(mtick.FuncFormatter(
            lambda t, _p: f"-{int(len(dd) - 1 - t)}" if t < len(dd) - 1 else "0"))
        for lbl in axdv.get_xticklabels():
            lbl.set_rotation = 0

    # ── 右: 4H 大周期面板 (横向并排) ──
    if ax3 is not None and df_4h is not None:
        d4_full = df_4h.copy()
        if "timestamp" in d4_full.columns:
            d4_full = d4_full.set_index("timestamp")
        d4_full = d4_full.sort_index()
        c4_full = d4_full["close"].to_numpy(float)  # 全量 close (布林/MA 用)
        d4 = d4_full.tail(n_bars)
        o4 = d4["open"].to_numpy(float)
        h4 = d4["high"].to_numpy(float)
        l4 = d4["low"].to_numpy(float)
        c4 = d4["close"].to_numpy(float)
        v4 = d4["volume"].to_numpy(float) if "volume" in d4.columns else np.ones(len(d4))
        x4 = np.arange(len(d4))  # 根数坐标
        w4 = 0.65
        for i in range(len(d4)):
            col4 = C_UP if c4[i] >= o4[i] else C_DOWN
            ax3.plot([x4[i], x4[i]], [l4[i], h4[i]], color=col4, linewidth=0.7)
            ax3.add_patch(Rectangle(
                (x4[i] - w4 / 2, min(o4[i], c4[i])), w4, abs(c4[i] - o4[i]) or 1e-6,
                facecolor=col4, edgecolor=col4, linewidth=0.4))
        # 4H 布林带 + MA20 (全量历史计算, 尾部显示)
        _draw_bb(ax3, c4_full, x4)
        # 4H 关键位 (带 band)
        for lv in (levels_4h_chart or []):
            if isinstance(lv, dict):
                px4 = lv["price"]
                b4 = lv.get("band") or 0.0
                col4 = C_SUP if lv.get("side") == "support" else C_RES
                if b4 > 0:
                    ax3.axhspan(px4 - b4, px4 + b4, color=col4, alpha=0.05, linewidth=0)
                ax3.axhline(px4, color=col4, linewidth=1.0, alpha=0.8)
                ax3.text(x4[-1], px4, f"  {px4:.0f}",
                          color=col4, fontsize=7, va="center", ha="right",
                          alpha=1.0, fontweight="bold",
                          bbox=dict(boxstyle="round,pad=0.12", facecolor=C_BG,
                                    edgecolor=col4, linewidth=0.3, alpha=0.8))
            else:
                ax3.axhline(lv, color=C_LEVEL, linewidth=0.7, alpha=0.5, linestyle="--")
        ax3.axhline(price_last, color=C_MA60, linewidth=0.7, alpha=0.4, linestyle=":")
        _draw_extra(ax3, x4[-1], extra_4h, "↑")  # 最近大周期位参考
        ax3.set_title(f"4H 大周期 (最近 {len(d4)} 根)", color=C_DIM, fontsize=9, loc="left")
        ax3.set_ylabel("4H", color=C_TEXT, fontsize=8)
        # K线图横轴 = 距最新 K 线的根数
        ax3.xaxis.set_major_locator(mtick.MaxNLocator(6))
        ax3.xaxis.set_major_formatter(mtick.FuncFormatter(
            lambda t, _p: f"-{int(len(d4) - 1 - t)}" if t < len(d4) - 1 else "0"))
        ax3.tick_params(axis="x", colors=C_TEXT, labelsize=7)
        # 4H 成交量
        vol4 = [C_UP if c4[i] >= o4[i] else C_DOWN for i in range(len(d4))]
        ax4.bar(x4, v4, color=vol4, width=w4, alpha=0.7)
        ax4.set_ylabel("Vol", color=C_TEXT, fontsize=8)
        # 横轴 = 距最新 K 线的根数
        ax4.xaxis.set_major_locator(mtick.MaxNLocator(6))
        ax4.xaxis.set_major_formatter(mtick.FuncFormatter(
            lambda t, _p: f"-{int(len(d4) - 1 - t)}" if t < len(d4) - 1 else "0"))
        for lbl in ax4.get_xticklabels():
            lbl.set_rotation = 0

    # X轴 (1H) = 距最新 K 线的根数
    ax2.xaxis.set_major_locator(mtick.MaxNLocator(6))
    ax2.xaxis.set_major_formatter(mtick.FuncFormatter(
        lambda t, _p: f"-{int(len(df) - 1 - t)}" if t < len(df) - 1 else "0"))
    for lbl in ax2.get_xticklabels():
        lbl.set_rotation = 0

    # 预警标注: 左上角文字 + K线图上当前价→目标位的竖线/箭头 + ATR 距离
    alert_pos = {"4h": 0, "1h": 0, "15m": 0}

    def _draw_alert(ax, a, x_last, cur_px):
        icon = {"L1": "🔵", "L2": "⚡", "L3": "💥"}.get(a.get("level", "L2"), "")
        color = {"L1": "#5a9cf8", "L2": "#e6a23c", "L3": "#e15241"} \
            .get(a.get("level", "L2"), "#e6a23c")
        # 左上角文字 (级别 + 描述)
        slot = alert_pos.get(a.get("tf", tf), 0)
        alert_pos[a.get("tf", tf)] = slot + 1
        y = 0.95 - slot * 0.06
        ax.text(0.015, y, f"{icon} {a.get('text', '')}",
                transform=ax.transAxes, fontsize=8, color=color,
                ha="left", va="top", alpha=0.95,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#262b38",
                          edgecolor=color, linewidth=0.5, alpha=0.85))
        # K线图上: 当前价 → 目标位 竖线 + 箭头 + 距离标注
        target = a.get("target")
        dist_atr = a.get("dist_atr")
        if target is not None and dist_atr is not None and dist_atr < 10:
            ax.annotate(
                "", xy=(x_last, target), xytext=(x_last, cur_px),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2,
                                alpha=0.85, shrinkA=0, shrinkB=0),
                zorder=6)
            # 距离标注 (线段中点, 略偏右)
            mid = (cur_px + target) / 2
            ax.text(x_last + 0.3, mid, f" {dist_atr:.2f}ATR",
                    fontsize=7.5, color=color, va="center", alpha=0.95,
                    bbox=dict(boxstyle="round,pad=0.12", facecolor=C_BG,
                              edgecolor=color, linewidth=0.3, alpha=0.8),
                    zorder=7)

    if alerts:
        for a in alerts:
            a_tf = a.get("tf", tf)
            cur_px = a.get("cur", price_last)
            if a_tf == "4h" and ax3 is not None:
                _draw_alert(ax3, a, x4[-1], cur_px)
            elif a_tf == "15m" and ax15 is not None:
                _draw_alert(ax15, a, x15[-1], cur_px)
            else:
                _draw_alert(ax1, a, x[-1], cur_px)  # 默认 1H 主图

    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    return path


def quick_test():
    """生成一张示例图 (无数据时用合成数据)"""
    import numpy as np
    n = 150
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    high = close + np.abs(rng.normal(0, 0.3, n))
    low = close - np.abs(rng.normal(0, 0.3, n))
    open_ = np.concatenate([[100], close[:-1]])
    vol = rng.uniform(100, 500, n)
    idx = pd.date_range("2026-08-01", periods=n, freq="5min", tz="UTC")
    df = pd.DataFrame({"open": open_, "high": high, "low": low,
                       "close": close, "volume": vol}, index=idx)
    path = make_chart(
        df, "TEST/USDT", "5m",
        levels=[{"price": 98.5, "side": "support", "touch": 3},
                {"price": 101.8, "side": "resistance", "touch": 2}],
        levels_4h=[97.2, 103.0],
        alerts=[{"price": 97.5, "level": "L2", "text": "关键位触及 L2"}],
        info={"dow4h": "up", "dow4h_age": 23, "dow_daily": "up",
              "cons": "顺风", "stat": "涨趋势·early",
              "dist4h": "1.2/0.8", "dist1h": "0.5/1.4"})
    print(f"图表已生成: {path}")


if __name__ == "__main__":
    quick_test()
