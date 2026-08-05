#!/usr/bin/env python3
"""三级预警引擎 (2026-08-04 大改造, 无方向预测)

用户观念: 关键位有大量订单, 价格到关键位要么反转要么突破; 波动低→高
像呼吸; 系统提前预警机会 (关键位 + 波动 + 状态), 方向由人判断。

L1 酝酿: 接近关键位 (≤1.0 ATR) × 低波动/压缩 — 提前量最大
L2 触发: 关键位触碰 (intrabar, 分叉点) / 波动启动
L3 确认: 收盘突破
"""
import numpy as np

from key_levels import APPROACH_ATR

LEVEL_ICON = {"L1": "🔵", "L2": "⚡", "L3": "💥"}


def compose(sym: str, tfs: dict) -> list[dict]:
    """组装某标的的全部预警 (tfs: {tf: info})"""
    warns = []
    touched = set()
    for tf, info in tfs.items():
        # 先收集触碰 (更即时): (tf, side)
        for tch in info.get("touches") or []:
            touched.add((tf, tch["side"]))
            side_cn = "支撑" if tch["side"] == "support" else "阻力"
            warns.append({
                "level": "L2", "type": "触碰", "tf": tf,
                "side": tch["side"], "price": tch["price"],
                "desc": f"触碰{side_cn}位 {tch['price']:.5g}",
            })
    for tf, info in tfs.items():
        price = info.get("price")
        if price is None:
            continue
        atr = info.get("atr") or 0
        low_vol = info.get("vol_state") == "低" or bool(info.get("squeeze"))
        # L1: 接近位带 × 波动 (同侧已触碰则跳过 — 触碰信息更即时)
        rel = info.get("rel") or {}
        for side in ("support", "resistance"):
            if (tf, side) in touched:
                continue
            lv = rel.get(side)
            if lv and lv["dist_atr"] <= APPROACH_ATR:
                t = "酝酿" if low_vol else "接近"
                warns.append({
                    "level": "L1", "type": t, "tf": tf,
                    "side": "支撑" if side == "support" else "阻力",
                    "desc": f"{t}{'支撑' if side == 'support' else '阻力'} "
                            f"{lv['dist_atr']:.2f}ATR({lv['touch']}触)"
                            + ("·低波动" if low_vol else ""),
                })
        # L1: 布林带上下轨
        bb = info.get("bb")
        if bb is not None and atr > 0:
            for name, bp in (("上轨", bb[1]), ("下轨", bb[2])):
                if not np.isfinite(bp):
                    continue
                d = abs(price - bp) / atr
                if d <= 0.5:
                    warns.append({
                        "level": "L1", "type": "布林", "tf": tf, "side": name,
                        "desc": f"接近布林{name} {d:.2f}ATR",
                    })
        # L3: 突破
        for brk in info.get("breaks") or []:
            warns.append({
                "level": "L3", "type": "突破", "tf": tf,
                "side": brk["side"], "price": brk["price"],
                "desc": f"突破{brk['side']}位 {brk['price']:.5g}",
            })
        # L2: 波动启动
        if info.get("vol_start"):
            warns.append({
                "level": "L2", "type": "波动启动", "tf": tf, "side": "",
                "desc": "波动启动(ATR跳升)",
            })
    return warns


def format_warns(sym: str, warns: list[dict], dow4h: dict | None = None,
                 limit: int = 5) -> str:
    """精简文本行 (一行 = 一个预警)"""
    seg = ""
    if dow4h:
        d = dow4h.get("seg_dir")
        age = dow4h.get("seg_age")
        if d in ("up", "down") and age is not None:
            seg = f"·段{'↑' if d == 'up' else '↓'}{age}根"
    lines = []
    for w in sorted(warns, key=lambda x: (x["level"], x["type"]))[:limit]:
        icon = LEVEL_ICON.get(w["level"], "")
        lines.append(f"  {icon}{w['tf']} {w['desc']}{seg}")
    return "\n".join(lines)
