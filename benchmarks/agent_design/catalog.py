"""Keep design cases reviewable without confusing specifications with executions.

Loading and rendering this catalog performs no Agent calls, fault injection or
scoring. An execution adapter and independent state-based scorers are still
required before any of these cases can produce a benchmark result.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


CATEGORY_LABELS = {
    "tools": "工具执行与恢复",
    "state": "状态管理与幂等",
    "budget": "预算与终止",
    "context": "上下文管理",
    "a2a": "多 Agent / A2A",
}
CATEGORY_PREFIXES = {
    "tools": "TOOL", "state": "STATE", "budget": "BUDGET",
    "context": "CONTEXT", "a2a": "A2A",
}
DEFAULT_CATALOG_PATH = Path(__file__).with_name("cases.json")


def _nonempty_text(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")


def _text_list(value: Any, field: str) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a nonempty list")
    for index, item in enumerate(value):
        _nonempty_text(item, f"{field}[{index}]")


def _object(value: Any, field: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")


def validate_catalog(data: Any) -> None:
    """Reject incomplete or misleading specifications; do not run their cases."""
    _object(data, "catalog")
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValueError("schema_version must be integer 1")
    if data.get("suite") != "agent_design":
        raise ValueError("suite must be agent_design")
    if data.get("status") != "specified":
        raise ValueError("catalog status must be specified, not an execution claim")
    _nonempty_text(data.get("description"), "description")
    _object(data.get("evaluation_policy"), "evaluation_policy")
    for key in ("acceptance", "metrics", "budget_units", "trace_scope", "a2a_scope", "data"):
        _nonempty_text(data["evaluation_policy"].get(key), f"evaluation_policy.{key}")
    _text_list(data["evaluation_policy"].get("execution_layers"), "evaluation_policy.execution_layers")
    cases = data.get("cases")
    if not isinstance(cases, list) or len(cases) != 50:
        raise ValueError("cases must contain exactly 50 specifications")
    seen = set()
    category_ids = {category: set() for category in CATEGORY_LABELS}
    for index, case in enumerate(cases):
        field = f"cases[{index}]"
        _object(case, field)
        category = case.get("category")
        if not isinstance(category, str) or category not in CATEGORY_LABELS:
            raise ValueError(f"{field}.category is not a supported direction")
        case_id = case.get("id")
        _nonempty_text(case_id, f"{field}.id")
        if case_id in seen:
            raise ValueError(f"duplicate case id: {case_id}")
        if not re.fullmatch(rf"{CATEGORY_PREFIXES[category]}-(?:0[1-9]|10)", case_id):
            raise ValueError(f"{field}.id must match its category and index 01..10")
        seen.add(case_id)
        category_ids[category].add(case_id)
        if case.get("status") != "specified":
            raise ValueError(f"{case_id}.status must be specified, not an execution claim")
        for key in ("title", "prompt"):
            _nonempty_text(case.get(key), f"{case_id}.{key}")
        for key in ("expected", "metrics", "required_capabilities", "trace_requirements"):
            _text_list(case.get(key), f"{case_id}.{key}")
        if len(case["expected"]) < 2:
            raise ValueError(f"{case_id}.expected needs outcome and invariant checks")
        setup = case.get("setup")
        _object(setup, f"{case_id}.setup")
        _nonempty_text(setup.get("isolation"), f"{case_id}.setup.isolation")
        if type(setup.get("seed")) is not int:
            raise ValueError(f"{case_id}.setup.seed must be an integer")
        _text_list(setup.get("tools"), f"{case_id}.setup.tools")
        _object(setup.get("initial_state"), f"{case_id}.setup.initial_state")
        limits = setup.get("limits")
        _object(limits, f"{case_id}.setup.limits")
        for key in ("max_agent_steps", "max_total_tokens", "deadline_seconds", "max_tool_attempts_per_operation"):
            value = limits.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{case_id}.setup.limits.{key} must be finite and positive")
            if key != "deadline_seconds" and type(value) is not int:
                raise ValueError(f"{case_id}.setup.limits.{key} must be an integer")
        if "fault" not in case:
            raise ValueError(f"{case_id}.fault is required, use null for healthy controls")
        fault = case["fault"]
        if fault is not None:
            _object(fault, f"{case_id}.fault")
            for key in ("type", "trigger"):
                _nonempty_text(fault.get(key), f"{case_id}.fault.{key}")
        comparison = case.get("comparison")
        _object(comparison, f"{case_id}.comparison")
        for key in ("baseline", "intervention"):
            _nonempty_text(comparison.get(key), f"{case_id}.comparison.{key}")
        _text_list(comparison.get("controls"), f"{case_id}.comparison.controls")
        if category == "a2a":
            if "future_a2a_transport_adapter" not in case["required_capabilities"]:
                raise ValueError(f"{case_id} must declare its future A2A transport adapter")
            scope = setup.get("protocol_scope")
            _object(scope, f"{case_id}.setup.protocol_scope")
            if scope.get("implementation_status") != "future_adapter_required":
                raise ValueError(f"{case_id} must not claim A2A execution support")
            for key in ("profile", "transport", "assertion_scope"):
                _nonempty_text(scope.get(key), f"{case_id}.setup.protocol_scope.{key}")
    for category, ids in category_ids.items():
        if len(ids) != 10:
            raise ValueError(f"{category} must contain exactly 10 specifications")
        if not any(case["category"] == category and case["fault"] is None for case in cases):
            raise ValueError(f"{category} must include a healthy control")


def load_catalog(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load validated, fresh JSON data without importing a model or Agent runtime."""
    source = DEFAULT_CATALOG_PATH if path is None else Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    validate_catalog(data)
    return data


def _cell(value: str) -> str:
    return html.escape(value, quote=False).replace("|", "&#124;").replace("\n", "<br>")


def render_markdown(data: Dict[str, Any]) -> str:
    """Render the same specifications deterministically, with no result columns."""
    validate_catalog(data)
    lines: List[str] = [
        "# Agent 设计评估：50 条测试规格", "",
        "状态：**specified（仅完成规格编写）**。每个方向 10 条，不含 Skill。",
        "这些条目不是 50 条已执行测试，也不代表环境、适配器和评分器已全部实现。没有模型实测分数。",
        "完整 prompt、初始数据、故障时机、时限、验收条件、指标、消融控制和轨迹字段见 [cases.json](cases.json)。", "",
        "## 执行与判分约定", "",
        "- 机制层使用脚本模型驱动真实 Agent、存储和调度器；故障由环境注入，不能让测试函数内部自行恢复后冒充 Agent 能力。",
        "- 真实模型层使用同模型、同任务、同预算的配对消融。计划每配置重复 5 次；尚未运行。未触发的故障单独标记，不能计入恢复率分母。",
        "- 优先核验最终环境状态、必要依赖、不变量和资源上限，允许多种正确执行路径。诚实失败单独计分，不等于任务成功。",
        "- 全部写入仅发生在每次 trial 重置的本地沙箱。时间阈值由虚拟时钟确定性验证，真实进程另报墙钟和调度偏差。",
        "- 记录可观察的模型请求/响应、工具、状态、压缩和子任务事件；不把轨迹描述成模型隐藏思维链。日志脱敏后保存。",
        "- A2A 需要未来协议适配器，并在执行前锁定版本、传输和能力。普通函数委派只能验证本地编排；幂等、围栏等应用保证不等于协议自带保证。", "",
    ]
    for category, label in CATEGORY_LABELS.items():
        lines.extend([f"## {label}", "", "| 编号 | 测试场景 | 环境 / 注入 | 验收重点 | 对照设计 |",
                      "|---|---|---|---|---|"])
        selected = sorted((case for case in data["cases"] if case["category"] == category), key=lambda case: case["id"])
        for case in selected:
            fault = case["fault"]
            scenario = "正常对照；" + case["prompt"] if fault is None else fault["trigger"]
            expected = "；".join(case["expected"][:2])
            comparison = case["comparison"]["baseline"] + " → " + case["comparison"]["intervention"]
            cells = [case["id"], case["title"], scenario, expected, comparison]
            lines.append("| " + " | ".join(_cell(value) for value in cells) + " |")
        lines.append("")
    lines.extend(["## 规格校验", "", "```powershell", "python -m benchmarks.agent_design.catalog", "```", "",
                  "该命令只检查规格结构与数量，不启动 Agent、不调用模型、不产生评估得分。", ""])
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate Agent design specifications; do not execute them.")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--write-markdown", type=Path, help="Write the reviewable specification table to this path")
    args = parser.parse_args(argv)
    data = load_catalog(args.catalog)
    if args.write_markdown:
        args.write_markdown.write_text(render_markdown(data), encoding="utf-8")
    print("Validated 50 specifications, 10 per direction. Status: specified; no Agent execution or scores.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
