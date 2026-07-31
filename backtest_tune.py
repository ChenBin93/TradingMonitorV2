#!/usr/bin/env python3
"""参数网格搜索 — 单参数扫描优先, 多进程并行

用法:
  python3 backtest_tune.py --param rsi_oversold --values 20,25,30
  python3 backtest_tune.py --sweep-all        # 所有参数逐个扫描
  python3 backtest_tune.py --grid             # 暴力网格(慎用, 组合爆炸)
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from datetime import datetime

import yaml

sys.path.insert(0, ".")

from backtest_engine import load_all, run_backtest, summarize

# 单参数扫描定义: 参数路径 → 候选值 (与 signals.py / engine 参数对应)
PARAM_SWEEPS = {
    # signals.py SignalDef 参数
    "volume_threshold":     ("signals", "volume_spike", "threshold", [2.5, 3.0, 3.5]),
    "vol_min_atr_ratio":    ("signals", "volume_spike", "min_price_change", [0.2, 0.3, 0.5]),
    # engine 参数
    "atr_sl_buffer":        ("engine", "atr_sl_buffer", None, [0.2, 0.3, 0.5]),
    "atr_tp_fallback":      ("engine", "atr_tp_fallback", None, [2.0, 2.5, 3.0]),
    "rr_min":               ("engine", "rr_min", None, [1.2, 1.5]),
    "forward_hours":        ("engine", "forward_hours", None, [24, 48, 72]),
    "dedup_minutes":        ("engine", "dedup_minutes", None, [15, 30, 60]),
}


def build_signal_params(overrides: dict) -> dict:
    """从 config 构建 SignalDef 参数 dict, 应用 overrides"""
    return overrides  # 具体信号参数在 run_backtest 里按 SignalDef.params 逐信号应用


def apply_params(tf_cfg: dict, signal_params: dict) -> dict:
    """复制 config, 应用参数覆盖"""
    import copy
    cfg = copy.deepcopy(tf_cfg)
    cfg["signal_params"] = signal_params
    return cfg


_DATA = None
_CFG = None
_SYMBOLS = None


def _init_worker(cfg_path, symbols):
    global _DATA, _CFG, _SYMBOLS
    with open(cfg_path) as f:
        _CFG = yaml.safe_load(f)
    _DATA = load_all("data/backtest.db")
    _SYMBOLS = symbols or list(_DATA.keys())


def run_one(args_tuple):
    """多进程 worker: 跑一组参数, 返回统计"""
    param_name, param_value, engine_kwargs = args_tuple
    t0 = time.time()
    trades = run_backtest(_DATA, _SYMBOLS, _CFG, **engine_kwargs)
    dt = time.time() - t0
    stats = summarize(trades)
    stats["param"] = param_name
    stats["value"] = param_value
    stats["runtime_s"] = round(dt, 1)
    return stats


def sweep_single_param(param_name: str, values: list, cfg_path: str,
                       symbols: list | None, workers: int = 4):
    """单参数扫描: 其他参数固定, 只变一个"""
    # 定位参数路径
    if param_name not in PARAM_SWEEPS:
        print(f"Unknown param {param_name}. Available: {list(PARAM_SWEEPS.keys())}")
        return
    kind, _, _, default_vals = PARAM_SWEEPS[param_name]
    if values is None:
        values = default_vals

    # 构造 worker 任务
    tasks = []
    for v in values:
        if kind == "engine":
            kw = {"atr_sl_buffer": 0.3, "atr_tp_fallback": 2.5, "rr_min": 1.2,
                  "forward_hours": 48, "dedup_minutes": 30}
            kw[param_name] = v
        else:  # signals
            kw = {"atr_sl_buffer": 0.3, "atr_tp_fallback": 2.5, "rr_min": 1.2,
                  "forward_hours": 48, "dedup_minutes": 30,
                  "signal_overrides": {param_name: v}}
        tasks.append((param_name, v, kw))

    print(f"Sweeping {param_name}: {values} ({len(tasks)} runs, {workers} workers)")
    with mp.Pool(workers, initializer=_init_worker, initargs=(cfg_path, symbols)) as pool:
        results = pool.map(run_one, tasks)

    # 输出排序
    results.sort(key=lambda r: r.get("sharpe", -99), reverse=True)
    print(f"\n{'value':>10} {'trades':>7} {'win%':>6} {'RR':>6} {'return':>8} {'sharpe':>7} {'maxDD':>7} {'time':>6}")
    for r in results:
        print(f"{str(r['value']):>10} {r['count']:>7} {r['win_rate']*100:>5.1f}% "
              f"{r['avg_rr']:>6.2f} {r['total_return']:>8.1f} {r['sharpe']:>7.1f} "
              f"{r['max_drawdown']:>7.1f} {r['runtime_s']:>5.0f}s")

    with open(f"data/tune_{param_name}_{datetime.now():%Y%m%d_%H%M}.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def sweep_all(cfg_path: str, symbols: list | None, workers: int = 4):
    """所有参数逐个扫描"""
    for param_name in PARAM_SWEEPS:
        sweep_single_param(param_name, None, cfg_path, symbols, workers)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--param", default=None, help="参数名 (见 PARAM_SWEEPS)")
    p.add_argument("--values", default=None, help="逗号分隔候选值")
    p.add_argument("--sweep-all", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--symbols", nargs="*", default=None)
    args = p.parse_args()

    values = [float(v) for v in args.values.split(",")] if args.values else None

    if args.sweep_all:
        sweep_all("config.yaml", args.symbols, args.workers)
    elif args.param:
        sweep_single_param(args.param, values, "config.yaml", args.symbols, args.workers)
    else:
        p.print_help()
