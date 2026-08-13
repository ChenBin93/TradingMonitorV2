#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L3 研究脚本门禁 — research/check_study.py (research/PLAN.md §1 L3)

用法:
    python3 research/check_study.py <study_script.py>

自动定位同名 .out 与 .conclusion.md (先 research/notes/ 再 research/archive/notes/,
按 basename 前缀匹配); 下划线开头文件跳过。七类检查, 任一 FAIL 输出
"文件:L行号 + 原因" 并退出码 1:

  ① import 白名单 (AST)      — 只允许 research.{...} 白名单子模块 + 第三方白名单;
                               from research.studies (研究互 import) → FAIL
  ② 禁止模式 (AST)           — 价格数组手动切片 (.values[切片/变量], 裸数组切片,
                               df.iloc/loc 切片); np.percentile/nanpercentile/
                               quantile 作用于特征 (docstring 无 [DESCRIPTIVE]);
                               自写 outcome (target/stop 或 tp/sl + 逐 bar range
                               循环, 且不在引擎注册表); searchsorted(conf 条件化
                               (必须走 causal.causal_confirmed / align_events)
  ③ .out 头部校验            — meta 含 study_id/script_sha256/data_range/params/gate;
                               script_sha256 与脚本实际 sha256 一致 (不一致=改过未重跑)
  ④ GATE 区块校验            — 无条件基线 (真实+GBM); gbm_seeds ≥ MIN_GBM_SEEDS=30;
                               1:1 口径 GBM 无条件 WR ∈ [49%, 51%] (缺失该数字则
                               WARN); MIN_N 检查存在
  ⑤ 数字指纹                 — 结论全部数字 (\d+\.\d+) 逐一在 .out 可查 (子串匹配)
  ⑥ 成对性                   — 有 .out 无结论 / 有结论无 .out → WARN (不 FAIL)
  ⑦ 发布门槛强制             — 结论含 正期望/edge/有效/可交易/捕捉 (非"无"否定语境)
                               时, .out 必须有 BY_YEAR (真实+GBM 成对) 且结论含
                               成本核算, 否则 FAIL

设计决策:
  - 纯静态检查 (AST/正则/文本), 绝不 import research 模块 → 任何模块缺失不崩溃。
  - AST 解析/遍历异常一律捕获并转 FAIL 报告 (文件+行号+原因), 不抛栈。
  - [DESCRIPTIVE] 豁免: docstring 含该标注即豁免整文件的 percentile/quantile 检查
    (描述层允许全样本统计; 因果特征区仍禁 — 无法从 AST 精确切分到"区段", 取保守
    全文件豁免并在输出中提示)。
  - 数字指纹取裸小数 \d+\.\d+ (子串覆盖 \d+\.\d+% / \d+\.\d+pp / [+-]\d\.\d+R,
    并能击落 B5c 的 "1.00")。
  - "edge" 用字母边界 (?<![a-zA-Z])edge(?![a-zA-Z]) 防 knowledge/hedge 误报;
    "有效/edge" 紧邻"无"开头的否定语境 (无 edge / 无有效) 不算发布主张。
"""
import argparse
import ast
import hashlib
import os
import re
import sys

# ── 门槛常量 (与 research/caliber.py 对齐; 静态检查不 import, 双份维护) ──
MIN_GBM_SEEDS = 30
MIN_N = 200
GBM_WR_LO, GBM_WR_HI = 49.0, 51.0

# ── 白名单 ──
RESEARCH_MODULES = frozenset({
    "caliber", "outcome", "sim_market", "data_loader", "causal", "ctx",
    "state_features", "structures", "hold_sim", "levels", "limit_sim",
})
THIRD_PARTY = frozenset({
    "numpy", "pandas", "scipy", "statsmodels", "pywt", "collections",
    "bisect", "os", "sys", "gc", "dataclasses", "math", "json", "time",
    "datetime",
    # 性能: GBM 种子循环/标的分层可并行 (机器 2 核); functools.lru_cache
    # 用于研究内确定性重算消除 (如 cluster 结果跨 grid 复用)
    "multiprocessing", "concurrent.futures", "functools",
})
# 例外: hashlib 为 .out meta 运行时计算 script_sha256 的必需模块 (PLAN §3 模板
# 骨架要求 meta 含运行时 sha256); __future__ 为语法指令 (PEP 236), 语义惰性。
THIRD_PARTY |= frozenset({"__future__", "hashlib"})

# 官方 outcome 引擎注册表 — 函数名在此则豁免"自写 outcome"检查
ENGINE_REGISTRY = frozenset({
    "evaluate_forward", "evaluate_forward_vbt",
    "simulate_holds", "simulate_limit_entries",
})

META_FIELDS = ("study_id", "script_sha256", "data_range", "params", "gate")

PUBLISH_CH_ZH = ("正期望", "有效", "可交易", "捕捉")
PUBLISH_EN = re.compile(r"(?<![a-zA-Z])edge(?![a-zA-Z])")
DECIMAL_RE = re.compile(r"\d+\.\d+")


class Report:
    """收集 FAIL / WARN / INFO, 全部带 (类别, 文件, 行号, 原因)"""

    def __init__(self, script):
        self.script = script
        self.fails = []
        self.warns = []
        self.infos = []

    def fail(self, cat, path, line, reason):
        self.fails.append((cat, path, line, reason))

    def warn(self, cat, path, line, reason):
        self.warns.append((cat, path, line, reason))

    def info(self, msg):
        self.infos.append(msg)


# ─────────────────────────────────────────────────────────────
# 定位 .out / .conclusion.md
# ─────────────────────────────────────────────────────────────
def locate_outputs(script_path):
    """按 basename 前缀匹配同名 .out 与 .conclusion.md.

    候选目录: 仓库根 research/notes → research/archive/notes → 脚本同目录 →
    脚本同目录 notes。返回 (out_path, concl_path), 找不到者为 None。
    """
    stem = os.path.splitext(os.path.basename(script_path))[0]
    sdir = os.path.dirname(os.path.abspath(script_path))

    root = None
    d = sdir
    while True:
        if os.path.isdir(os.path.join(d, "research")):
            root = d
            break
        p = os.path.dirname(d)
        if p == d:
            break
        d = p

    candidates = []
    if root:
        candidates += [os.path.join(root, "research", "notes"),
                       os.path.join(root, "research", "archive", "notes")]
    candidates += [sdir, os.path.join(sdir, "notes")]

    out = concl = None
    seen = set()
    for cdir in candidates:
        if cdir in seen or not os.path.isdir(cdir):
            continue
        seen.add(cdir)
        if out is None and os.path.exists(os.path.join(cdir, stem + ".out")):
            out = os.path.join(cdir, stem + ".out")
        if concl is None and os.path.exists(os.path.join(cdir, stem + ".conclusion.md")):
            concl = os.path.join(cdir, stem + ".conclusion.md")
    return out, concl


# ─────────────────────────────────────────────────────────────
# ① import 白名单
# ─────────────────────────────────────────────────────────────
def check_imports(tree, rep):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                name = a.name
                if name == "research":
                    continue
                if name.startswith("research."):
                    sub = name[len("research."):].split(".")[0]
                    if sub == "studies":
                        rep.fail("①import", rep.script, node.lineno,
                                 f"研究互 import 被禁 (B4e/B5c 模式): {name}")
                    elif sub in RESEARCH_MODULES:
                        continue
                    else:
                        rep.fail("①import", rep.script, node.lineno,
                                 f"research 非白名单子模块 {sub}: {name}")
                elif name.split(".")[0] in THIRD_PARTY:
                    continue
                else:
                    rep.fail("①import", rep.script, node.lineno,
                             f"非白名单模块 {name}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod == "research":
                for n in (a.name for a in node.names):
                    if n in RESEARCH_MODULES:
                        continue
                    if n == "studies":
                        rep.fail("①import", rep.script, node.lineno,
                                 "研究互 import 被禁: from research import studies")
                    else:
                        rep.fail("①import", rep.script, node.lineno,
                                 f"from research import {n} — 非白名单 (研究模块只允许 "
                                 f"{sorted(RESEARCH_MODULES)})")
            elif mod.startswith("research."):
                sub = mod[len("research."):].split(".")[0]
                if sub == "studies":
                    rep.fail("①import", rep.script, node.lineno,
                             f"研究互 import 被禁 (B4e/B5c 模式): {mod}")
                elif sub not in RESEARCH_MODULES:
                    rep.fail("①import", rep.script, node.lineno,
                             f"research 非白名单子模块 {sub}: {mod}")
            elif mod.split(".")[0] in THIRD_PARTY:
                continue
            else:
                rep.fail("①import", rep.script, node.lineno,
                         f"非白名单模块 {mod}")


# ─────────────────────────────────────────────────────────────
# ② 禁止模式
# ─────────────────────────────────────────────────────────────
def _expr_repr(node):
    """尽力还原表达式文本 (Name/Attribute/Subscript/Call)"""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_expr_repr(node.value)}.{node.attr}"
    if isinstance(node, ast.Subscript):
        return f"{_expr_repr(node.value)}[...]"
    if isinstance(node, ast.Call):
        return f"{_expr_repr(node.func)}(...)"
    return type(node).__name__


def check_forbidden(tree, rep, docstring):
    descriptive = bool(docstring) and "[DESCRIPTIVE]" in docstring
    if descriptive:
        rep.info("docstring 含 [DESCRIPTIVE] 标注 → percentile/quantile 检查豁免 "
                 "(描述层允许全样本统计; 结论禁止进入交易含义)")

    for node in ast.walk(tree):
        # ── 价格数组手动切片 ──
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice):
            base = node.value
            if isinstance(base, ast.Attribute) and base.attr == "values":
                rep.fail("②禁止模式", rep.script, node.lineno,
                         f"{_expr_repr(base)}[切片] 价格数组手动切片 — 必须走 "
                         f"research.ctx.make_ctx 对齐")
            elif isinstance(base, ast.Attribute) and base.attr in ("iloc", "loc"):
                rep.fail("②禁止模式", rep.script, node.lineno,
                         f"{_expr_repr(base)}[切片] 手动截断切片 — 必须走 "
                         f"research.ctx.make_ctx")
            elif isinstance(base, ast.Name):
                rep.fail("②禁止模式", rep.script, node.lineno,
                         f"{base.id}[切片] 手动数组切片 — 必须走 research.ctx.make_ctx")
            elif (isinstance(base, ast.Subscript)
                  and isinstance(base.slice, ast.Constant)
                  and isinstance(base.slice.value, str)):
                rep.fail("②禁止模式", rep.script, node.lineno,
                         f"DataFrame 列 {_expr_repr(base)} 手动切片 — 必须走 "
                         f"research.ctx.make_ctx")
        elif (isinstance(node, ast.Subscript)
              and isinstance(node.value, ast.Attribute)
              and node.value.attr == "values"
              and not isinstance(node.slice, ast.Constant)):
            # .values[变量] — 变量索引 (常量直接索引 .values[0] 除外)
            rep.fail("②禁止模式", rep.script, node.lineno,
                     f"{_expr_repr(node.value)}[变量] 价格数组手动索引 — 必须走 "
                     f"research.ctx.make_ctx 对齐")

        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute):
                if f.attr in ("percentile", "nanpercentile") and \
                        isinstance(f.value, ast.Name) and f.value.id == "np":
                    if not descriptive:
                        rep.fail("②禁止模式", rep.script, node.lineno,
                                 f"np.{f.attr}(...) 全样本分位作用于特征 — 必须用 "
                                 f"causal.rolling_percentile/rolling_rank (描述层才可 "
                                 f"标 [DESCRIPTIVE])")
                elif f.attr == "quantile":
                    if not descriptive:
                        rep.fail("②禁止模式", rep.script, node.lineno,
                                 f"{_expr_repr(f)} 分位作用于特征 — 必须用 "
                                 f"causal.rolling_percentile/rolling_rank (描述层才可 "
                                 f"标 [DESCRIPTIVE])")
                elif f.attr == "searchsorted":
                    rep.fail("②禁止模式", rep.script, node.lineno,
                             "searchsorted 手动条件化 confirmed 数组 — 必须走 "
                             "causal.causal_confirmed / causal.align_events")
            elif isinstance(f, ast.Name) and f.id == "searchsorted":
                rep.fail("②禁止模式", rep.script, node.lineno,
                         "searchsorted 手动条件化 confirmed 数组 — 必须走 "
                         "causal.causal_confirmed / causal.align_events")


def check_self_written_outcome(tree, rep):
    """自写 outcome: 函数内同现 target/stop(或 tp/sl) 与逐 bar range 循环,
    且函数名不在引擎注册表 → FAIL"""
    def _names(fn):
        s = set()
        for n in ast.walk(fn):
            if isinstance(n, ast.Name):
                s.add(n.id)
        return s

    def _has_range_loop(fn):
        for n in ast.walk(fn):
            if (isinstance(n, ast.For) and isinstance(n.iter, ast.Call)
                    and isinstance(n.iter.func, ast.Name)
                    and n.iter.func.id == "range"):
                return True
        return False

    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if fn.name in ENGINE_REGISTRY:
            continue  # 官方引擎豁免
        names = _names(fn)
        has_pair = ({"target", "stop"} <= names) or ({"tp", "sl"} <= names)
        if has_pair and _has_range_loop(fn):
            rep.fail("②禁止模式", rep.script, fn.lineno,
                     f"函数 {fn.name} 内同现 target/stop(或 tp/sl) 与逐 bar range "
                     f"循环 — 疑似自写 outcome 引擎; 必须用官方引擎 "
                     f"{sorted(ENGINE_REGISTRY)}")


# ─────────────────────────────────────────────────────────────
# .out 区块解析 (meta / GATE / BY_YEAR)
# ─────────────────────────────────────────────────────────────
def block_after(lines, marker):
    """返回 (起始行号, 区块行列表)。区块 = 从含 marker 的 # 行起连续 # 行。"""
    start = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith("#") and marker in ln:
            start = i
            break
    if start is None:
        return None, []
    block = []
    for ln in lines[start:]:
        if ln.strip().startswith("#"):
            block.append(ln)
        else:
            break
    return start, block


def extract_fields(text, fields):
    out = {}
    for f in fields:
        m = re.search(rf"\b{f}\s*[=:]\s*([^\s#]+)", text)
        if m:
            out[f] = m.group(1)
    return out


def sha256_of(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# ─────────────────────────────────────────────────────────────
# ③ .out 头部校验
# ─────────────────────────────────────────────────────────────
def check_out_meta(rep, out_path, script_path, out_lines):
    start, block = block_after(out_lines, "meta")
    if start is None:
        rep.fail("③.out meta", out_path, 1,
                 "meta 区块缺失 (需 study_id/script_sha256/data_range/params/gate)")
        return None
    fields = extract_fields("\n".join(block), META_FIELDS)
    missing = [f for f in META_FIELDS if f not in fields]
    if missing:
        rep.fail("③.out meta", out_path, start + 1,
                 f"meta 缺字段 {missing} (需全部: {list(META_FIELDS)})")
        return None
    actual = sha256_of(script_path)
    if fields["script_sha256"] != actual:
        rep.fail("③.out meta", out_path, start + 1,
                 f"script_sha256 不一致: .out={fields['script_sha256'][:12]}… "
                 f"脚本实际={actual[:12]}… — 脚本修改后未重跑 (.out 陈旧)")
    return fields


# ─────────────────────────────────────────────────────────────
# ④ GATE 区块校验
# ─────────────────────────────────────────────────────────────
def infer_is_1to1(tree):
    """脚本口径推断: 只用对称 evaluate_forward → 1:1; 只用非对称/limit/hold →
    非 1:1; 无法判定 → None"""
    if tree is None:
        return None
    uses_sym = uses_asym = False
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            fname = f.attr if isinstance(f, ast.Attribute) else \
                (f.id if isinstance(f, ast.Name) else "")
            if fname in ("evaluate_forward", "evaluate_forward_vbt"):
                kw = {k.arg for k in n.keywords}
                if "t_target" in kw or "t_stop" in kw:
                    uses_asym = True
                else:
                    uses_sym = True
            elif fname in ("simulate_holds", "simulate_limit_entries"):
                uses_asym = True
    if uses_sym and not uses_asym:
        return True
    if uses_asym and not uses_sym:
        return False
    return None


def check_gate(rep, out_path, out_lines, is_1to1):
    start, block = block_after(out_lines, "GATE")
    if start is None:
        rep.fail("④GATE", out_path, 1, "GATE 区块缺失")
        return
    text = "\n".join(block)

    m = re.search(r"gbm_seeds\s*=\s*(\d+)", text)
    seeds = int(m.group(1)) if m else None
    if seeds is None:
        rep.fail("④GATE", out_path, start + 1, "GATE 缺 gbm_seeds= 字段")
    elif seeds < MIN_GBM_SEEDS:
        rep.fail("④GATE", out_path, start + 1,
                 f"gbm_seeds={seeds} < MIN_GBM_SEEDS={MIN_GBM_SEEDS}")

    if "无条件基线" not in text:
        rep.fail("④GATE", out_path, start + 1, "GATE 缺无条件基线 (真实+GBM 对照)")

    mreal = re.search(r"真实\s*(\d+(?:\.\d+)?)%", text)
    mgbm = re.search(r"GBM\s*(\d+(?:\.\d+)?)%", text)
    real_wr = float(mreal.group(1)) if mreal else None
    gbm_wr = float(mgbm.group(1)) if mgbm else None

    # 口径显式标记优先; 否则用脚本推断
    if "t1:1" in text:
        is_1to1 = True
    elif re.search(r"asym|non-?1:1|limit", text, re.I):
        is_1to1 = False

    if real_wr is None or gbm_wr is None:
        rep.warn("④GATE", out_path, start + 1,
                 f"GATE 缺 真实/GBM 无条件基线百分比 (真实={real_wr}, GBM={gbm_wr}) "
                 f"— 无法核验 1:1 WR 区间")
    elif is_1to1 and not (GBM_WR_LO <= gbm_wr <= GBM_WR_HI):
        rep.fail("④GATE", out_path, start + 1,
                 f"1:1 口径 GBM 无条件基线 WR {gbm_wr:.1f}% ∉ "
                 f"[{GBM_WR_LO:.0f}%, {GBM_WR_HI:.0f}%] — 口径偏置, 须修")

    if "MIN_N" not in text:
        rep.fail("④GATE", out_path, start + 1, "GATE 缺 MIN_N 检查结果")


# ─────────────────────────────────────────────────────────────
# ⑤ 数字指纹
# ─────────────────────────────────────────────────────────────
def check_fingerprint(rep, out_path, concl_path, out_text, concl_lines):
    tokens = []  # (token, 结论行号)
    seen = set()
    for i, ln in enumerate(concl_lines, 1):
        for m in DECIMAL_RE.finditer(ln):
            tok = m.group(0)
            if (tok, i) not in seen:
                seen.add((tok, i))
                tokens.append((tok, i))
    if not tokens:
        return
    missing = [(tok, i) for tok, i in tokens if tok not in out_text]
    if missing:
        first_line = min(i for _, i in missing)
        detail = " ".join(f"{t}(L{i})" for t, i in missing[:10])
        if len(missing) > 10:
            detail += f" …共{len(missing)}处"
        rep.fail("⑤数字指纹", concl_path, first_line,
                 f"结论 {len(missing)} 个数字在 .out 中不可查 (子串匹配): {detail}")
    else:
        rep.info(f"⑤数字指纹: {len(set(t for t, _ in tokens))} 个数字全部在 .out 可查")


# ─────────────────────────────────────────────────────────────
# ⑥ 成对性
# ─────────────────────────────────────────────────────────────
def check_pairing(rep, script_path, out_path, concl_path):
    if out_path and not concl_path:
        rep.warn("⑥成对性", script_path, 1,
                 "有 .out 无同名 .conclusion.md (成对性不齐, 不 FAIL)")
    elif concl_path and not out_path:
        rep.warn("⑥成对性", script_path, 1,
                 "有 .conclusion.md 无同名 .out (成对性不齐, 不 FAIL)")


# ─────────────────────────────────────────────────────────────
# ⑦ 发布门槛强制
# ─────────────────────────────────────────────────────────────
def publish_hits(text):
    """返回 [(词, 行号)] — 排除"无"开头的否定语境 (无 edge / 无有效)"""
    hits = []
    for m in re.finditer(PUBLISH_EN, text):
        pre = text[max(0, m.start() - 2):m.start()]
        if "无" in pre:
            continue
        line = text.count("\n", 0, m.start()) + 1
        hits.append(("edge", line))
    for zh in PUBLISH_CH_ZH:
        for m in re.finditer(re.escape(zh), text):
            pre = text[max(0, m.start() - 2):m.start()]
            if "无" in pre:
                continue
            line = text.count("\n", 0, m.start()) + 1
            hits.append((zh, line))
    return hits


def check_publish(rep, out_path, concl_path, out_text, concl_lines):
    text = "\n".join(concl_lines)
    hits = publish_hits(text)
    if not hits:
        return

    start, block = block_after(out_text.splitlines(), "BY_YEAR")
    has_by_year = start is not None
    by_year_paired = False
    if has_by_year:
        bt = "\n".join(block)
        by_year_paired = ("真实" in bt or "real" in bt.lower()) and ("GBM" in bt)

    has_cost = "成本" in text
    first_line = hits[0][1]
    if not (has_by_year and by_year_paired and has_cost):
        missing = []
        if not has_by_year:
            missing.append(".out 无 BY_YEAR 区块")
        elif not by_year_paired:
            missing.append("BY_YEAR 区块缺 真实+GBM 成对分年")
        if not has_cost:
            missing.append("结论缺成本核算小节 (发布主张必须成本后仍 > 0 才有资格)")
        rep.fail("⑦发布门槛", concl_path, first_line,
                 f"结论含发布主张「{hits[0][0]}」— " + "；".join(missing))


# ─────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────
def run_check(script_path):
    rep = Report(os.path.abspath(script_path))
    stem = os.path.basename(script_path)

    if stem.startswith("_"):
        rep.info(f"跳过: 下划线开头模板文件 {stem}")
        return rep, "skip"

    try:
        src = open(script_path, "r", encoding="utf-8").read()
    except OSError as e:
        rep.fail("读取", script_path, 1, f"无法读取脚本: {e}")
        return rep, "fail"

    # AST 解析 — 异常一律转 FAIL, 绝不崩溃
    tree = None
    docstring = ""
    try:
        tree = ast.parse(src, filename=script_path)
        docstring = ast.get_docstring(tree) or ""
    except SyntaxError as e:
        rep.fail("AST", script_path, getattr(e, "lineno", 1),
                 f"语法解析失败: {e}")
    except Exception as e:  # noqa: BLE001 — 门禁必须兜底
        rep.fail("AST", script_path, 1, f"解析异常 (已转 FAIL): {e!r}")

    if tree is not None:
        try:
            check_imports(tree, rep)
            check_forbidden(tree, rep, docstring)
            check_self_written_outcome(tree, rep)
        except Exception as e:  # noqa: BLE001
            rep.fail("AST", script_path, 1, f"AST 检查异常 (已转 FAIL): {e!r}")

    out_path, concl_path = locate_outputs(script_path)
    rep.info(f".out:        {out_path or '未找到'}")
    rep.info(f"conclusion:  {concl_path or '未找到'}")

    check_pairing(rep, script_path, out_path, concl_path)

    if out_path:
        out_text = open(out_path, "r", encoding="utf-8").read()
        out_lines = out_text.splitlines()
        check_out_meta(rep, out_path, script_path, out_lines)
        check_gate(rep, out_path, out_lines, infer_is_1to1(tree))
    else:
        out_text = ""
        rep.info("③④⑤: 无 .out — 静态检查 ①② 已完成, 数据层校验跳过 (成对性见 ⑥)")

    if concl_path:
        concl_lines = open(concl_path, "r", encoding="utf-8").read().splitlines()
        if out_path:
            check_fingerprint(rep, out_path, concl_path, out_text, concl_lines)
            check_publish(rep, out_path, concl_path, out_text, concl_lines)
        else:
            rep.info("⑤⑦: 无 .out — 数字指纹/发布门槛核验跳过")

    return rep, ("fail" if rep.fails else "pass")


def main(argv=None):
    ap = argparse.ArgumentParser(description="L3 研究脚本门禁 (PLAN.md §1 L3)")
    ap.add_argument("study", help="研究脚本路径 (下划线开头模板自动跳过)")
    args = ap.parse_args(argv)

    if not os.path.isfile(args.study):
        print(f"error: 文件不存在: {args.study}")
        return 2

    rep, verdict = run_check(args.study)

    print(f"===== check_study: {args.study} =====")
    for msg in rep.infos:
        print(f"  [info] {msg}")
    for cat, path, line, reason in rep.fails:
        print(f"  [FAIL] {cat} {path}:L{line} — {reason}")
    for cat, path, line, reason in rep.warns:
        print(f"  [WARN] {cat} {path}:L{line} — {reason}")

    n_fail, n_warn = len(rep.fails), len(rep.warns)
    if verdict == "skip":
        print(f"RESULT: SKIP (模板文件) — 退出码 0")
        return 0
    if n_fail:
        print(f"RESULT: FAIL ({n_fail} FAIL / {n_warn} WARN) — 退出码 1")
        return 1
    print(f"RESULT: PASS ({n_fail} FAIL / {n_warn} WARN) — 退出码 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
