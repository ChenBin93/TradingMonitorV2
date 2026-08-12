# 研究重启方案（C 系列）

> 2026-08-12 定稿。A/B 系列 29 项研究经严格审计发现致命缺陷（A3 750 根索引错位、confirmed 未来标签泄漏线、B4 结论不可复现等），研究结论整体作废。本方案保留经审计验证的基础设施（62 项测试全过、无未来函数），重建研究协议与路线图。

## 0. 裁决与原则

- **基础设施保留加固**：caliber.py / outcome.py / sim_market.py / data_loader.py + 62 项测试经审计无未来函数，保留；补齐缺口。
- **旧研究归档作废**：A/B 系列脚本+结论移入 `research/archive/`，数字一律不得引用；可信描述性规律仅作为"待重验 prior"，不是事实。
- **分层推进**：描述层（c1x）→ 条件层（c2x）→ 策略层（c3x），每层独立发布门槛。
- **核心教训**：旧门禁只测引擎、不测研究脚本——A3 的致命 bug 因此混过。新协议下研究脚本本身进入验证范围。

## 1. 三层门禁体系

### L1 引擎层（现有 62 测试保留 + 加固）

1. **引擎入口长度断言**（outcome.evaluate_forward / evaluate_forward_vbt / hold_sim.simulate_holds）：`len(entries) != len(close)` 即抛 ValueError——A3 类索引错位第一次运行即死，不再产出 .out 后才被发现。
2. **末根入场语义**：`i+1>=n` 不再静默丢弃，计入独立字段 `n_truncated`（非 expired）。
3. **data_loader 连续性检查**：`check_continuity(df, tf)` 报告 bar 间隔缺口，纳入 `verify()`。
4. **sim_market 时间锚定**：新增 `gbm_matching(ref_df, seed)`——索引锚定真实数据首根、长度相同、σ 由真实对数收益估计（修复 GBM 2023-01 vs 真实 2023-08 年份错位；替代各研究手写的 σ 估计模式）。
5. **caliber.py 新增 `MIN_GBM_SEEDS = 30`**（a38 6 种子、B3c 10 种子 ±2.6pp 的教训，统一取 30）。

### L2 模块层（每个公共 API 必须黄金+不变性+时序测试）

- **test_dow.py（dow_segments，当前零测试）**：手算道氏段黄金（确认 pivot→突破→HH→回撤→段结束逐事件断言）、多空镜像、真实+合成数据 cut 不变性、确认时序（HL pivot 确认前不得产生回撤事件——锁定"恢复窗口起点"类问题的结构侧）、amp_atr 无 pivot 段返回 NaN（顺手修 a6b 已披露 bug）。
- **test_levels_batch.py**：breakdown_all 按 `confirm_at` 门控（位形成前突破不计）、batch helpers 与已测单级函数对拍、`active_levels` 的 age_bars 追加数据后历史值不变（R1 修复锁）。
- **test_causal.py / test_ctx.py**：每个 helper 手算黄金 + 追加不变性。

### L3 研究脚本层（`research/check_study.py`，研究前第二道门禁）

`pytest research/tests -q` 通过后执行，任一违规即 FAIL（输出文件+行号）。检查项：

1. **import 白名单（AST）**：只允许 research 公共模块 + 第三方白名单；`from research.studies.xxx import`（研究互 import，B4e/B5c 模式）→ FAIL。
2. **禁止模式（AST）**：价格数组手动切片（必须走 `ctx.make_ctx`）；`np.percentile/nanpercentile/quantile` 作用于特征（除非 `[DESCRIPTIVE]` 分区）；自写 outcome（函数内同现 tp/sl 变量与逐 bar 循环，且不在引擎注册表）；`searchsorted(conf, ...)` 事后标签条件化（必须走 `causal.causal_confirmed`）。
3. **.out 头部校验**：meta 区块含 `study_id/script_sha256/data_range/params/gate`，sha256 与当前脚本一致（脚本改过未重跑 → FAIL）。
4. **GATE 区块校验**：存在无条件基线（真实+GBM）；GBM 种子数 ≥ MIN_GBM_SEEDS；1:1 口径研究 GBM 无条件胜率 ∈ 50%±1pp；MIN_N 检查结果打印。
5. **数字指纹**：结论文件中所有数字模式必须在对应 .out 中可查，找不到 → FAIL（B5c 的"1.00"即被此项击落）。
6. **成对性**：每个 .conclusion.md 必须有同名 .out，反之亦然。
7. **发布门槛强制**：结论含"正期望/edge/有效/可交易"等词时，必须全部门槛绿（真实−RW>0 且 .out BY_YEAR 成对分年可查 + n≥MIN_N + 成本核算小节），否则 FAIL。

**历史反例拦截对照（验收矩阵）**：

| 问题实例 | 拦截层 | 机制 |
|---|---|---|
| A3 索引错位 | L1 + L3 | 引擎长度断言（首次运行抛错）+ AST 禁手动切片 |
| B4e 全样本分位 | L2 + L3 | rolling_percentile 不变性测试 + AST 黑名单 |
| B5c 结论与脚本不符 | L3 | sha256 + 数字指纹 |
| confirmed 泄漏 | L2 | causal_confirmed 为唯一条件化出口 |
| B4 自写 _sim | L1/L3 | 官方非对称口径（见 §2）+ 引擎注册表 AST |
| a38 种子不足 | L1 + L3 | MIN_GBM_SEEDS=30 常量 + GATE 区块校验 |

## 2. 因果模式库与模块新增

**research/causal.py**（所有"当特征用"的结构统计唯一出口，全部过不变性测试）：

- `rolling_percentile(x, window, q, min_periods)` —— 滚动分位（禁全样本分位）。
- `rolling_rank(x, window)` —— 滚动百分位排位。
- `causal_confirmed(confirmed, w, lag_lo, lag_hi, confirm_cost)` —— 事后标签的因果可用版：`known[t]=1 ⟺ 存在 c∈[t-60, t-24] 满足 confirmed[c] 且确认窗口完全收在 t 之前`。默认语义 conf∈[t-60, t-24]；[t-23,t] 内突破的样本**剔除**而非视为存活（B2c/B2d 泄漏的正确替代）。
- `frozen_cluster(events, tol_fn, min_touch)` —— 在线聚类+冻结通用封装（levels.cluster_levels 泛化），研究脚本禁止自写聚类。
- `align_events(event_positions, t, lag_lo, lag_hi)` —— 因果窗口查询（bisect 封装）。

**research/ctx.py**：`make_ctx(df, warmup, state_fns)` 是截断对齐的唯一构造路径（内部统一 `iloc[warmup:]`，脚本永远见不到未对齐组合）；`entries_from_events` 保证全长度布尔。

**research/limit_sim.py**：`simulate_limit_entries(...)` limit 挂单成交模拟（B4e 语义合法化），注册引擎，黄金+性质+对拍测试。

**outcome.py 官方非对称口径扩展**：`evaluate_forward(..., t_target, t_stop)`——黄金测试 + 性质测试（GBM 上非对称结构优势必须真实/GBM 一致）+ vbt 对拍。消除 B4 自写 `_sim` 的根因：非对称需求存在但此前无官方口径。

**levels.py 修复**（**live 依赖，不得归档**：key_levels.py:19 ← main.py:581）：

- R1（:120 last_touch_idx 全样本突变）：Level 内部改为事件日志，`age_bars` 在 `active_levels(levels, t)` 内按确认时序重建快照；形成后字段不可变。live 只用最终带，输出语义不变。
- R2（:202-203/:331 未按 confirm_at 门控）：attempt/confirmed 增加 `t >= confirm_at` 掩码。
- 未提交 diff 处置：`git diff research/levels.py > research/archive/patches/levels_batch_helpers_uncommitted.patch` → `git checkout -- research/levels.py` 恢复 HEAD → 在 HEAD 基础上按 R1/R2 重写并补测。

## 3. 研究脚本与结论模板

### docstring 预注册格式（固定字段，check_study 校验）

```
"""C{id} {标题} ({date}, 无未来函数, {周期})

预注册假设 (运行前锁定, 结论逐条回应, 不得新造):
  H1 ...: 若成立, 应观察到 {具体可证伪数字}
无未来函数设计说明 (逐特征信息边界表): 特征 | 计算方式 | 可用时点 | 依据
  ([DESCRIPTIVE] 事后统计单独成段并标注, 其结论禁止进入交易含义)
数据声明 / 参数声明 (集中于 PARAMS) / 发布门槛自检 / 运行命令
"""
```

### 脚本骨架

import 白名单 + `PARAMS` 唯一参数源 + **GATE 自检**（模板内置：GBM ≥30 种子、与研究同管线重放的无条件基线断言 ≈50%，失败 `SystemExit`——"违规即停"的技术落地）。自检用同一管线重放而非只调引擎，可暴露研究脚本自己的管线级错位。

### .out 格式

```
# meta: study_id date script_sha256 data params
# GATE: gbm_seeds=30 无条件基线 真实..% GBM..% [PASS] MIN_N [PASS]
# RESULTS: ...
# BY_YEAR: 每个条件结论的 真实分年 + GBM分年 (成对出现)
```

### 结论模板

每个数字带 `(.out:L行号)` 引用；末尾固定发布门槛自查表；假设逐条回应（支持/否定/无法判断），禁止结论中新引入 .out 没有的分层或发现。

### 编号分层

| 层 | 编号 | 内容 | 门槛 |
|---|---|---|---|
| 描述层 | c1x | 市场事实，纯描述 | 无入场；[DESCRIPTIVE] 与因果特征分区；GBM 对照 |
| 条件层 | c2x | 条件 1:1 胜率/倾向 | GATE≈50% + 分年 + 净效应 + 成本行 |
| 策略层 | c3x | 策略期望（hold/limit 变体） | 全门槛 + vectorbt 二层 + live 试点 |

## 4. C 系列路线图

### 阶段 0：前置条件（加固 + 归档）

- P0-1 levels.py 未提交改动处置（patch 存档 → checkout → 重写）。
- P0-2 dow_segments / hold_sim 黄金测试补齐。
- P0-3 官方非对称口径 `evaluate_forward(t_target, t_stop)`。
- P0-4 A/B 归档 + 文档作废戳 + AGENTS.md 研究章节重写（见 §7）。
- P0-5 数据补充（可选）：历史延长至 4 年（history.py 分页）、OKX funding rate 历史（8h）。

### 阶段 1：描述层 c1x（prior 验尸）

通用 exit 模板：①GATE 过；②效应达预注册下限且 4 组合（2 参数×2 周期）同号；③真实−GBM 净差>0 且 GBM 侧无同类效应；④分年 ≥2/3 年同号；⑤每单元格 n≥MIN_N，全单元格输出（含不足者）。

| 编号 | 研究问题 | 关键假设（下限已打折） | 旧参照 | 优先级 |
|---|---|---|---|---|
| c11 | 方向不可预测 null 基线（Phase2 方向类对照锚） | H1: 各条件格方向概率 真实−GBM |Δ|≤1pp | a36+a1+B3c | 低（可并入 c18） |
| c12 | 波动长记忆+状态转移（因果口径） | H1: DFA-H≥0.70(1h)/0.65(4h)；H3: 低↔高直接转移率≤0.01（必经中）；z120 用滚动分位 | a15/a2 | 高（地基） |
| c13 | 趋势收益偏度特异性 | H1: up:late 偏度≥+1.5 且 GBM≤+0.3；H2: top5% K 贡献≥60% | a4 | 高（尾部线地基） |
| c14 | 位移约束普适（因果条件化） | H1: M2P50Δ≤-0.4ATR 全结构层同号；H3: 泄漏敏感性对照（因果版 vs B3 alive_at 版差异≤20%） | B3 | 高 |
| c15 | 触碰后波动释放 | H1: 净差≥+4pp 4 组合同号；H3: GBM 侧≤+1.5pp | B2b/B3 | 高 |
| c16 | 区间延续性（因果区间+双边界） | H1: w=6 留存净差≥+3pp；H4: 方向层净差≤0 | B3d | 高 |
| c17 | 趋势逆势折返（线索级） | H1: 沿趋势概率净差≤-2pp；H2: 阶段无梯度；H3: causal 角色层无额外信息 | B5/B5b/B5c | 中 |
| c18 | 触位 1:1 无增益复核 | H1: 4 方向 |Δ|≤1pp | B1 | 低（与 c11 合并） |
| c19 | 道氏段生命周期因果重做 | H1: 4h 存活 P(≥25) 真实−GBM≥+2pp；H2: 恢复率净差≤-8pp（打折重验）；恢复窗口起点=回撤低点确认 bar+1；气候条件全滚动 | a6 系 | 中（c22 不用道氏可砍） |
| c1a | 趋势持有正期望=纯结构 | H1: |期望差|<0.05R 全止损模式 | a35 | 低（默认砍，作 Phase3 sanity） |

交付顺序：**c12 → {c14, c15, c16} 并行 → c13 → c17 → {c11+c18 合并} → c19**。预计 10 项有效研究。

### 阶段 2：条件层 c2x（4 候选家族，全部预注册）

| 编号 | 候选 | 前提 | 主端点预注册要点 |
|---|---|---|---|
| c21 | 区间触碰小目标 + limit 入场（B4/B4e 因果重做） | c14+c16 确认；P0-3 就位 | limit buy @S.price，目标 0.3×ATR / 止损 0.7×ATR，W=6，胜率差≥+3pp 且期望差≥+0.05R；收盘入场版净差≈0（复现 B4e 伪影）；空头只报净差；成本后期望>0。参数网格预注册 + 开发/验证集分离 |
| c22 | 尾部收割因果重做（a38） | c13 确认 | 结构 early 入场+固定 1ATR 止损+峰值回撤 3ATR 退出，short 期望差≥+0.10R；GBM 30 种子、分年、4h 交叉、成本行（taker+滑点+funding） |
| c23 | 趋势逆势折返条件化 | c17 确认 | 逆势入场（涨触阻空/跌触撑多），止损=位带外+0.3ATR，目标 1ATR，W=24，胜率差≥+3pp；预注册声明：基线仅 2-4pp，成本后≤0 只能写"结构发现"非 edge |
| c24 | 波动压缩→触碰→释放 择时过滤 | c12+c15+c16 确认 | 触碰前 z120=低波动子集对 c21 有增量（≥+1.5pp）；GBM 上同过滤无同向增量（门禁）；主策略不达标自动降级为描述性报告 |

**Phase2 发布门槛清单（10 项，缺一不可）**：①真实−RW(30 种子) 超预注册下限；②分年 ≥2/3 为正、最差年 ≥-2pp；③每格 n≥MIN_N；④GATE 条件组无偏；⑤跨周期+跨参数一致；⑥holdout（末 3 月，参数冻结后一次评估）方向不变；⑦Holm 校正后仍显著；⑧成本核算后 >0；⑨结论↔.out↔脚本三重一致；⑩负结果/未达标格全部记录。

### 阶段 3：策略层 c3x

**c31 双层验证**：口径层（事件级 edge 是否存在）→ vectorbt 全回测层（可交易性：重叠持仓、资金管理、成本逐项注入，每步衰减可解释）。一致性验证：(a) 零成本单开无重叠下 vbt 与口径层逐笔对拍；(b) R 分布两层差异 ≤ 预注册容差（vbt 已知坑由测试锁定，不许"修"）；(c) 成本衰减曲线报告，单项成本吃掉 >50% 净差即单独立项。

**c32 live 试点**：影子模式（只推送不成交）观察 ≥30 信号事件 且 ≥4 周。6 项检查：事件频率偏差≤±30%、历史重放信号重合率≥95%、特征/参数/时间戳语义逐项对拍、limit 成交率偏差≤±10pp、事件延迟≤1 bar、影子 R 序列落在预回归 95% 区间。最小仓位：每笔风险 0.1~0.5% 净值，连亏 5R 退回影子，每周对账。

### Exit criteria

| 跃迁 | 条件 |
|---|---|
| Phase1→2 | ≥8 项完成且 GATE 全过；核心 prior 确认 ≥6 条（**c14/c15/c16 至少 2 条 + c12 必过**）；未确认 prior 按失败归档写明原因；若 c14-c17 全部不确认 → 停 level 线，Phase2 只留 c22/c24 |
| Phase2→3 | ≥1 候选过全部 10 项门槛，且机制与 Phase1 事实自洽；0 个通过 → 归档全部候选，返回 Phase1 扩展，明示"没有 edge 也是结论" |
| Phase3→实盘 | 影子 6 项全过 + 成本后期望>0 + holdout 未衰减 + 用户确认仓位/止损/暂停规则 |

## 5. 资源预算

| 阶段 | 估算 |
|---|---|
| Phase0 加固 | 3-5 人日（引擎断言→causal/ctx/limit_sim→levels 重写→check_study→三反例回归演示） |
| Phase1 | 8-10 项 × (pytest 4min + 运行 10-90min)；level 系 30-60min/项；**10-20 机器时，4-5 人日** |
| Phase2 | 4 家族 × (开发+验证+30 种子+holdout+成本) ≈ 3-6h/家族；**15-25 机器时，4-6 人日** |
| Phase3 | 全回测 1-3h/配置；影子模式 4 周墙钟；3-5 人日 |
| 数据 | backtest.db 够 Phase1/2 主体；建议补 4 年历史（1 天下载）+ funding 历史（Phase2 成本必需） |

## 6. 风险与反模式防线

| 反模式（历史教训） | C 系列防线 |
|---|---|
| 未来函数（整批作废之根） | 已收盘 bar；MTF align_higher；在线聚类+冻结；特征过 invariance；禁 center 窗口/shift(-1) |
| confirmed 泄漏（B 系列贯穿缺陷） | 条件化一律 causal_confirmed（conf∈[t-60,t-24]）；c14 H3 泄漏敏感性对照实验 |
| 全样本分位（B4e） | 一律 rolling_percentile；全样本统计量仅限事后标签且禁参与条件化 |
| post-hoc（B4 结论对应不存在的运行） | docstring 预注册冻结；三重一致为提交前置；脚本改动=重跑+结论重写 |
| 选择性报告 | 全单元格输出；负结果同等归档；"发现"措辞必须过门槛清单 |
| 样本内过拟合 | 参数网格预注册+开发/验证分离；GBM 固定种子序列（禁换种子刷结果）；holdout 一次评估；跨周期一致强制 |
| GBM 种子不足（a38 6 种子） | MIN_GBM_SEEDS=30，GATE 自动检查 |
| 引擎语义坑（自写 _sim、vbt 伪影） | 禁自写 outcome；非对称走官方扩展；vbt 坑测试锁定不许修 |
| 成本幻觉（B4 净差 0.07R < 成本 0.3-0.5R） | Phase2 结论附成本行；统一成本模型；成本后≤0 标"结构发现" |
| 文档不一致（LIVE_SYSTEM 旧段引用作废研究） | 文档三级状态标注（已确认/待重验/作废）；修正声明连带更新所有引用处 |
| 复现性（levels.py 未提交、B5c 修复未落地） | 每研究一个 commit（脚本+notes+.out）；禁止从脏工作树运行 |
| 方向参照偏置（位带中心参照 GBM 63%≠50%） | 方向度量一律以 close[t] 为参照；对数化防 Jensen；GATE 无条件 ≈50% 强制 |

## 7. 归档与迁移方案

### git 操作序列（按序，勿合并）

```bash
# 1. levels.py 未提交改动存档（B2+ 依赖的 batch helpers 研究全作废 → 存档不留用）
mkdir -p research/archive/patches
git diff research/levels.py > research/archive/patches/levels_batch_helpers_uncommitted.patch
git checkout -- research/levels.py          # 恢复 HEAD（live 依赖的稳定版）

# 2. 归档（保留历史，勿删）
git mv research/studies research/archive/studies
git mv research/notes   research/archive/notes

# 3. 重建空目录 + archive/README.md 作废公告

# 4. 确认 live 不受影响后提交
grep -rn "research.studies\|research.notes" --include="*.py" .   # 应只剩 archive 内部互引
git commit -m "research: archive all A/B studies (contaminated) + restart as C series"
```

注意：`research/levels.py` **不能归档**（live 依赖）；archive 内部互 import 断裂可接受（旧研究不再运行，tests/ 无 studies 依赖已核实）。

### 文档处置

| 文件 | 处置 |
|---|---|
| docs/SESSION_SUMMARY.md、docs/MARKET_STATE_STRATEGY_MAP.md、archive/notes/phase_summary.md、CONTEXT_RESTORE.md | 不删，头部加作废戳：研究结论基于 2026-08-03~05 批次，2026-08-12 整体作废，数字不可采信；有效结论唯一来源 = research/notes/（C 系列） |
| docs/LIVE_SYSTEM.md | 运行规格（端口/启动/推送时间窗）保留；SCENE_WR 等研究数字段加戳 |
| AGENTS.md | "回测与研究"整节重写：C 系列协议（编号/模板/两道门禁 `pytest research/tests -q` + `python3 research/check_study.py`/发布门槛/archive 制度/levels patch 存档与 live 依赖说明） |

## 8. 执行顺序（最小步骤 + 验收）

1. **归档**（git 序列 §7）→ 验收：git status 干净、live 冒烟不报错、旧测试全绿、无残留 import。
2. **L1 引擎加固** → 验收：新测试绿、62 用例不回归、用 A3 旧脚本演示断言触发。
3. **L2 模块**（causal/ctx/limit_sim + 测试 → levels R1/R2 重写 + test_levels_batch → test_dow 含 amp NaN 修复）→ 验收：全绿（预计 ~75 用例）、live 冒烟 key_levels 输出不变。
4. **L3 门禁**（check_study 七类检查 + 三模板）→ 验收：A3/B4e/B5c 三反例回归演示，输出"FAIL+原因+行号"。
5. **C1 试点**：c12（波动记忆+状态转移）——描述层无入场、风险最低，dogfood 全模板 → 验收：check_study 过、GATE 50%±1pp、结论数字全可溯源。
6. **分层推进**：c1x 队列 → c2x 候选 → c3x 全回测+试点，每层按 exit criteria 裁决。

---

**一句话总纲**：门禁从"测引擎"升级为"测模块+验脚本"三层；Phase1 用因果口径给旧结论验尸（确认 6 条即过关），Phase2 只允许 4 个预注册候选走完整门槛（过不了就是没有），Phase3 口径层与 vectorbt 层、回测与 live 双重对拍——宁可收敛到"关键位=围墙、无方向 edge、尾部收割待成本验证"的诚实结论，也不许再产出一个 B4。
