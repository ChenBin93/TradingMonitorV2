#!/usr/bin/env python3
"""参数网格搜索 — 两阶段: 信号检测一次 + 交易模拟逐参数 (毫秒级)

用法:
  python3 backtest_tune.py --param atr_tp_mult --values 1.0,1.5,2.0 --base tp_mode=atr
  python3 backtest_tune.py --param volume_threshold --values 2.5,3.0,3.5
  python3 backtest_tune.py --sweep-all
"""
import argparse
import json
import sys
import time
from datetime import datetime

import yaml

sys.path.insert(0, ".")

from backtest_engine import load_all, detect_signals, simulate_trades, summarize

# 单参数扫描定义: 参数路径 → 候选值
PARAM_SWEEPS = {
    # signals.py SignalDef 参数 (影响检测阶段 — 会触发重新检测)
    # 格式: ("signals", "信号id", "参数key", [候选值])
    "volume_threshold":     ("signals", "volume_spike", "threshold", [2.5, 3.0, 3.5]),
    "vol_min_atr_ratio":    ("signals", "volume_spike", "min_price_change", [0.2, 0.3, 0.5]),
    "rsi_oversold":         ("signals", "rsi_extreme", "oversold", [20, 25, 30]),
    "rsi_overbought":       ("signals", "rsi_extreme", "overbot", [70, 75, 80]),
    # 模拟阶段参数 (检测一次后秒级切换)
    "atr_sl_buffer":        ("engine", "atr_sl_buffer", None, [0.1, 0.2, 0.3, 0.5]),
    "rr_min":               ("engine", "rr_min", None, [1.0, 1.2, 1.5, 2.0]),
    "forward_hours":        ("engine", "forward_hours", None, [24, 48, 72]),
    "tp_mode":              ("engine", "tp_mode", None, ["sr", "atr", "min"]),
    "atr_tp_mult":          ("engine", "atr_tp_mult", None, [1.0, 1.2, 1.5, 2.0, 2.5, 3.0]),
}

# 模拟阶段参数 (检测一次后秒级切换)
ENGINE_PARAMS = {"atr_sl_buffer", "rr_min", "forward_hours", "tp_mode", "atr_tp_mult"}
SIGNAL_PARAMS = {"volume_threshold", "vol_min_atr_ratio", "rsi_oversold", "rsi_overbought"}


def run_sweep(param_name: str, values: list, cfg_path: str,
              symbols: list | None, base_overrides: dict | None = None,
              events_cache: dict | None = None):
    """单参数扫描 — 两阶段执行"""
    if param_name not in PARAM_SWEEPS:
        print(f"Unknown param {param_name}. Available: {list(PARAM_SWEEPS.keys())}")
        return
    kind, _, _, default_vals = PARAM_SWEEPS[param_name]
    if values is None:
        values = default_vals

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    data = load_all("data/backtest.db")
    syms = symbols or list(data.keys())

    results = []
    t_total = time.time()
    local_cache = {} if events_cache is None else events_cache

    for v in values:
        t0 = time.time()
        if param_name in SIGNAL_PARAMS:
            # 信号参数: 需要重新检测, 定向覆盖到指定信号
            # base 中的信号参数也一并应用
            _, sig_id, sig_key, _ = PARAM_SWEEPS[param_name]
            base_sig_overrides = {}
            if base_overrides:
                for k, bv in base_overrides.items():
                    if k in SIGNAL_PARAMS:
                        base_sig_overrides[k] = bv
            ov = {}
            for k, bv in base_sig_overrides.items():
                _, bs_id, bs_key, _ = PARAM_SWEEPS[k]
                ov.setdefault(bs_id, {})[bs_key] = bv
            ov.setdefault(sig_id, {})[sig_key] = v
            events = detect_signals(data, syms, cfg, signal_overrides=ov)
        else:
            # 引擎参数: 复用缓存事件
            if "events" not in local_cache:
                local_cache["events"] = detect_signals(data, syms, cfg)
                local_cache["symbols"] = syms
                print(f"  [detect] {len(local_cache['events'])} events, "
                      f"{time.time()-t0:.0f}s", flush=True)
            events = local_cache["events"]

        kw = {"atr_sl_buffer": 0.3, "rr_min": 1.2, "forward_hours": 48,
              "tp_mode": "sr", "atr_tp_mult": 2.5}
        if base_overrides:
            for k, bv in base_overrides.items():
                if k in ENGINE_PARAMS:
                    kw[k] = bv
        if param_name in ENGINE_PARAMS:
            kw[param_name] = v

        trades = simulate_trades(events, data, syms, **kw)
        stats = summarize(trades)
        stats["param"] = param_name
        stats["value"] = v
        stats["runtime_s"] = round(time.time() - t0, 2)
        results.append(stats)
        print(f"  {str(v):>8}: {stats['count']:>5} trades  WR={stats['win_rate']*100:5.1f}%  "
              f"RR={stats['avg_rr']:5.2f}  ret={stats['total_return']:8.1f}  "
              f"sharpe={stats['sharpe']:6.1f}  maxDD={stats['max_drawdown']:7.1f}  "
              f"[{stats['runtime_s']:.1f}s]", flush=True)

    results.sort(key=lambda r: r.get("sharpe", -99), reverse=True)
    print(f"\n=== {param_name} sorted by sharpe ===")
    print(f"{'value':>10} {'trades':>7} {'win%':>6} {'RR':>6} {'return':>8} {'sharpe':>7} {'maxDD':>7}")
    for r in results:
        print(f"{str(r['value']):>10} {r['count']:>7} {r['win_rate']*100:>5.1f}% "
              f"{r['avg_rr']:>6.2f} {r['total_return']:>8.1f} {r['sharpe']:>7.1f} "
              f"{r['max_drawdown']:>7.1f}")

    out = f"data/tune_{param_name}_{datetime.now():%Y%m%d_%H%M}.json"
    with open(out, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out}  (total {time.time()-t_total:.0f}s)")


def sweep_all(cfg_path: str, symbols: list | None,
              base_overrides: dict | None = None):
    """所有参数逐个扫描 (引擎参数共享检测缓存)"""
    cache = {}
    for param_name in PARAM_SWEEPS:
        print(f"\n### {param_name}")
        run_sweep(param_name, None, cfg_path, symbols,
                  base_overrides=base_overrides, events_cache=cache)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--param", default=None, help="参数名 (见 PARAM_SWEEPS)")
    p.add_argument("--values", default=None, help="逗号分隔候选值")
    p.add_argument("--sweep-all", action="store_true")
    p.add_argument("--symbols", nargs="*", default=None)
    p.add_argument("--base", default=None,
                   help="基础参数覆盖 key=value,key=value (如 tp_mode=atr)")
    args = p.parse_args()

    base_overrides = None
    if args.base:
        base_overrides = {}
        for kv in args.base.split(","):
            k, v = kv.split("=")
            base_overrides[k] = v if v in ("sr", "atr", "min") else float(v)

    values = None
    if args.values:
        if args.param == "tp_mode":
            values = args.values.split(",")
        else:
            values = [float(v) for v in args.values.split(",")]

    if args.sweep_all:
        sweep_all("config.yaml", args.symbols, base_overrides)
    elif args.param:
        run_sweep(args.param, values, "config.yaml", args.symbols,
                  base_overrides=base_overrides)
    else:
        p.print_help()
