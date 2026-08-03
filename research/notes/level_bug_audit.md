# 关键位系列 Bug 审计 (2026-08-03)

> 目的：确认两个 bug 的影响范围，防止旧 .out 文件误导未来研究。
> 结论：**只有 4 个矩阵脚本受影响（已修复），关键位系列 8 个实验全部可信。**

## Bug 1: 显示 bug（nn = w + l，分母错误）
`res[1]` 累加的是**命中总数**（win+loss），但输出时 `nn = w + l` 把赢数又加了一遍，
分母变成 `2×win+loss` → 显示胜率 = w/(2w+loss)，**系统性偏低**（36% 假象）。

### 受影响（已修复）
| 脚本 | 修复前显示 | 真实值 |
|------|-----------|--------|
| level_state_fast.py | 做多顺日线 36.1% | **56.5%** |
| level_state_clean.py | 做多顺日线 36.1% | ~56.5% |
| level_state_matrix.py | 同 | ~56-58% |

### 未受影响（l 是真 loss 数，nn = w+l 正确）
structure_memory / structure2 / structure_memory_mtf / level_quality / level_quality_v2 /
level_quality_env / mtf_level_overlap / trendline_study / hurst_daily
—— 全部用 `out ∈ {0,1}` win 标志，`l = sum(out==0)`，做空方向也正确
（`else: if l[i+k] <= entry - a: hit = 1`）。

## Bug 2: level_target_grid 做空方向 bug
`hit_stats` 返回 first=1（上涨先到），做空桶直接用 `ft==1` 判赢 → 做空胜率**反向**。
修复：`side == "short"` 时 `first = -first`。

## 影响范围结论
- **作废结论**：矩阵"贴位无条件 30-40% 全部 <50%"、"贴位做多顺日线 36-39%"
- **可信结论**（全部经代码审查确认）：
  1. 结构锚（刚形成极值 0-5根）贴位 63-70%；已跌破 37%
  2. 关键位窗口：年龄 200-400 最优（63.5%），400+ 回落；touch 11-15 拐点
  3. 质量分单调 50.7%→61.1%（+8.9pp，level_quality_v2）
  4. 4H 重叠 +1.8pp；趋势线 -7.6pp
  5. 插曲贴锚基线 56.0-56.4%（与修复后 grid 插曲 57.1/60.6% 互证）
- **P1 遗留矛盾闭合**：level_quality_v2 的 56.0% 正确；"debug5 45.9%" 是 bug 产物（≈100%-56%）

## 新框架：两类贴位（level_touch_mode.md）
| 类型 | 定义 | 胜率 | 质量分适用 |
|------|------|------|-----------|
| A. 结构锚贴位 | 最近极值 0-0.5ATR | 63.7-67.1% | 否（age 天然小） |
| B. 水平位贴位 | ∃ 老极值贴住 | 56.1% 基础 | 是（+8.3pp） |

## 检查清单（未来脚本规范）
1. 判定 win 用 `hit == 1`，loss 用 `hit == -1`，命中总数 = win + loss
2. 输出胜率 = win / (win + loss)，绝不用 win / (win + 命中总数)
3. 做空方向：下跌先到 = 赢
4. 每次改脚本先跑小样本对照（同逻辑旧脚本）
