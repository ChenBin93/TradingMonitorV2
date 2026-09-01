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
    out_dir: str = "/tmp/charts",
    n_bars: int = 120,
) -> str:
    """生成K线图 → 返回 PNG 路径

    df: DataFrame (DatetimeIndex 或 timestamp 列, 含 ohlcv)
    levels: 1h 关键位, [{"price", "side": support/resistance, "touch"}]
    levels_4h: 4h 关键位价格列表 (画虚线)
    alerts: [{price, level: 'L1'/'L2'/'L3', text}] 预警标注
    bb: (ma, upper, lower) 布林带三序列
    ma60: MA60 序列
    info: 顶部信息栏 dict (dow4h/dow4h_age/dow_daily/cons/stat/dist4h/dist1h)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
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
    df = df.sort_index().tail(n_bars)
    if len(df) < 20:
        return ""

    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    v = df["volume"].to_numpy(float) if "volume" in df.columns else np.ones(len(df))
    idx = df.index

    # 默认指标 (若未提供)
    if bb is None:
        sma = df["close"].rolling(40, min_periods=40).mean()
        sd = df["close"].rolling(40, min_periods=40).std(ddof=0)
        bb = (sma, sma + 2 * sd, sma - 2 * sd)
    ma, up, lo_ = bb
    if ma60 is None and "close" in df.columns:
        ma60 = df["close"].rolling(60, min_periods=60).mean()
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

    # 布局: 信息栏(0) + 主图(1) + 成交量(2)
    fig = plt.figure(figsize=(12, 7.2), facecolor=C_BG)
    gs = fig.add_gridspec(3, 1, height_ratios=[0.9, 3.2, 1], hspace=0.10,
                          left=0.06, right=0.98, top=0.93, bottom=0.07)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    ax0.axis("off")
    for ax in (ax1, ax2):
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
        line2 = (f"距关键位(支撑/阻力 ATR): 4H [{d4}]   1H [{d1}]")
        ax0.text(0.005, 0.72, line1, transform=ax0.transAxes,
                 fontsize=10, color=C_TEXT, va="center",
                 bbox=dict(boxstyle="round,pad=0.35", facecolor="#262b38",
                           edgecolor=C_GRID, linewidth=0.6))
        ax0.text(0.005, 0.16, line2, transform=ax0.transAxes,
                 fontsize=9, color=C_DIM, va="center",
                 bbox=dict(boxstyle="round,pad=0.35", facecolor="#262b38",
                           edgecolor=C_GRID, linewidth=0.6))

    # ── 主图: K线 ──
    x = np.arange(len(df))
    width = 0.65
    for i in range(len(df)):
        color = C_UP if c[i] >= o[i] else C_DOWN
        ax1.plot([x[i], x[i]], [l[i], h[i]], color=color, linewidth=0.8, zorder=2)
        ax1.add_patch(Rectangle(
            (x[i] - width / 2, min(o[i], c[i])), width, abs(c[i] - o[i]) or 1e-6,
            facecolor=color, edgecolor=color, linewidth=0.5, zorder=3))

    # 布林带 (ma/up/lo_ 已统一为 numpy)
    ax1.fill_between(x, up, lo_, color=C_BB, alpha=0.12, zorder=1)
    ax1.plot(x, ma, color=C_BB, linewidth=0.9, alpha=0.7, label="BB40")
    ax1.plot(x, up, color=C_BB, linewidth=0.7, alpha=0.5, linestyle="--")
    ax1.plot(x, lo_, color=C_BB, linewidth=0.7, alpha=0.5, linestyle="--")
    # MA60
    if ma60 is not None and np.isfinite(ma60).any():
        ax1.plot(x, ma60, color=C_MA60, linewidth=1.1, label="MA60")

    # 1h 关键位: 支撑绿 / 阻力红, 带价格标签
    if levels:
        y_lo, y_hi = ax1.get_ylim()
        for lv in levels:
            if isinstance(lv, dict):
                px = lv["price"]
                side = lv.get("side", "")
                col = C_SUP if side == "support" else C_RES if side == "resistance" else C_LEVEL
                ax1.axhline(px, color=col, linewidth=0.9, alpha=0.55, linestyle="-")
                ax1.text(len(df) - 1, px, f" {px:.0f}",
                         color=col, fontsize=7.5, va="bottom", ha="right",
                         alpha=0.9)
            else:
                ax1.axhline(lv, color=C_LEVEL, linewidth=0.8, alpha=0.5, linestyle=":")
    # 4h 关键位: 灰色虚线
    if levels_4h:
        for px in levels_4h:
            ax1.axhline(px, color=C_LEVEL, linewidth=0.7, alpha=0.4, linestyle="--")

    # 预警标注
    if alerts:
        for a in alerts:
            ax1.scatter([len(df) - 1], [a.get("price", c[-1])],
                        marker="v" if a.get("level", "L2") == "L3" else "o",
                        s=90 if a.get("level") == "L3" else 60,
                        color={"L1": "#5a9cf8", "L2": "#e6a23c", "L3": "#e15241"}
                        .get(a.get("level"), "#e6a23c"),
                        zorder=5, edgecolor="white", linewidth=0.6)
            ax1.annotate(
                a.get("text", ""), (len(df) - 1, a.get("price", c[-1])),
                textcoords="offset points", xytext=(8, -10), fontsize=8,
                color=C_TEXT, zorder=6)

    ax1.set_title(f"{symbol}  {tf}  (UTC: {idx[-1]:%m-%d %H:%M})",
                  color=C_TEXT, fontsize=11, loc="left")
    ax1.legend(loc="upper left", fontsize=7, facecolor=C_BG, edgecolor=C_GRID,
               labelcolor=C_TEXT)
    ax1.set_ylabel("Price", color=C_TEXT, fontsize=8)

    # ── 下图: 成交量 ──
    vol_colors = [C_UP if c[i] >= o[i] else C_DOWN for i in range(len(df))]
    ax2.bar(x, v, color=vol_colors, width=width, alpha=0.7)
    ax2.set_ylabel("Vol", color=C_TEXT, fontsize=8)

    # X轴时间
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    for lbl in ax2.get_xticklabels():
        lbl.set_rotation=0

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
