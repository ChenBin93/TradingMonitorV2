#!/usr/bin/env python3
"""K线图表生成 — 供飞书预警图片推送

设计:
- 上图: K线 + BB40±2σ + MA60 + 关键位水平线 + 预警标注 (L1/L2/L3)
- 下图: 成交量 (颜色随涨跌)
- 输出: /tmp/charts/{symbol}_{tf}.png (或指定目录)

依赖: matplotlib (Agg 后端, 无显示环境)
"""
import os

import numpy as np
import pandas as pd

# 颜色 (飞书深色背景友好)
C_UP = "#e15241"    # 涨 (红, 国内习惯)
C_DOWN = "#2db25d"  # 跌 (绿)
C_BB = "#8a8f99"    # 布林带
C_MA60 = "#e6a23c"  # MA60
C_LEVEL = "#5a5f6b"  # 关键位
C_GRID = "#2a2e37"
C_BG = "#1e222d"
C_TEXT = "#d8dce6"


def make_chart(
    df: pd.DataFrame,
    symbol: str,
    tf: str,
    levels: list[float] | None = None,
    alerts: list[dict] | None = None,
    bb: tuple | None = None,
    ma60: pd.Series | None = None,
    out_dir: str = "/tmp/charts",
    n_bars: int = 120,
) -> str:
    """生成K线图 → 返回 PNG 路径

    df: DataFrame (DatetimeIndex 或 timestamp 列, 含 ohlcv)
    levels: 关键位价格列表 (画水平线)
    alerts: [{price, level: 'L1'/'L2'/'L3', text}] 预警标注
    bb: (ma, upper, lower) 布林带三序列
    ma60: MA60 序列
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

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 6.5), sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1], "hspace": 0.08},
    )
    fig.patch.set_facecolor(C_BG)
    for ax in (ax1, ax2):
        ax.set_facecolor(C_BG)
        ax.grid(True, color=C_GRID, linewidth=0.6, alpha=0.6)
        ax.tick_params(colors=C_TEXT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(C_GRID)

    # ── 上图: K线 ──
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
    ax1.plot(x, ma, color=C_BB, linewidth=0.9, alpha=0.7, label=f"BB40")
    ax1.plot(x, up, color=C_BB, linewidth=0.7, alpha=0.5, linestyle="--")
    ax1.plot(x, lo_, color=C_BB, linewidth=0.7, alpha=0.5, linestyle="--")
    # MA60
    if ma60 is not None and np.isfinite(ma60).any():
        ax1.plot(x, ma60, color=C_MA60, linewidth=1.1, label="MA60")

    # 关键位水平线
    if levels:
        for lv in levels:
            ax1.axhline(lv, color=C_LEVEL, linewidth=0.8, alpha=0.5, linestyle=":")

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
    path = make_chart(df, "TEST/USDT", "5m", levels=[98, 102],
                      alerts=[{"price": 97.5, "level": "L2", "text": "关键位触及"}])
    print(f"图表已生成: {path}")


if __name__ == "__main__":
    quick_test()
