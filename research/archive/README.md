# Archive — 作废研究（2026-08-03 ~ 08-05 批次）

> ⚠️ 本目录全部研究（A/B 系列 29 项）经 2026-08-12 严格审计，因未来函数/口径违规整体作废。
> **所有数字一律不得引用。** 有效研究结论唯一来源 = `research/notes/`（C 系列，从 2026-08-12 起）。

## 作废原因概要

- **A3**（750 根索引错位 bug）：条件入场价/ATR 取自事件前 750 根，全部条件胜率无意义。
- **confirmed 未来标签泄漏线**（B2c/B2d/B3/B3b/B3c/B3d/B4/B4e/B5c）：需未来 24 根才能确定的 `confirmed` 标签被当作 t 时刻已知信息做样本条件化。
- **B4**：结论描述的运行在仓库中不存在（结论↔脚本↔.out 三重矛盾），且自写 `_sim` 引擎违反口径。
- **B4e**：BB 宽度分层用全样本分位（未来函数）。
- **B5c**：结论声称的修复未落地（脚本仍是 buggy 版本）。
- 其余：dow_segments 零测试（A6 系）、GBM 种子不足（a38 仅 6 种子）、恢复窗口起点违规（a6a/b/d）等。

完整审计与重建方案见 `research/PLAN.md`。

## 目录结构

- `studies/` — 全部 A/B 研究脚本（仅作历史参考，不再运行；内部互 import 已断裂，属预期）
- `notes/` — 全部结论与 .out 输出
- `patches/` — 归档时工作区未提交改动的存档（levels_batch_helpers_uncommitted.patch：levels.py 的 batch helpers，研究全作废故不留用）

## 例外

- `research/levels.py` 未归档：live 生产代码依赖（key_levels.py → main.py），按 PLAN.md §2 在 HEAD 基础上修复 R1/R2。
