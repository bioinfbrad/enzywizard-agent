"""测试 Planning、确认循环和真实多工具执行工作流。

本脚本有意使用截止 Tool Calling 的测试专用图，以便大规模验证工具执行而不消耗
额外 Analysis 模型调用。完整 Analysis 与多轮恢复由 run_analysis_module_tests.py
及正式图负责。默认只生成并检查最终计划；显式传入 --execute 后才运行工具。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.types import Command

from enzywizard_agent.conf import AgentConfig
from enzywizard_agent.data_models.agent_inputs import AgentInputs
from enzywizard_agent.data_models.artifact import CleanedFiles
from enzywizard_agent.nodes.planning import planning_node
from enzywizard_agent.nodes.request_routing import request_routing_node
from enzywizard_agent.nodes.tool_calling import tool_calling_node
from enzywizard_agent.nodes.user_confirmation_interrupt import (
    user_confirmation_interrupt_node,
)
from enzywizard_agent.state import AgentState
from enzywizard_agent.utils.CLI_utils import get_single_interrupt_payload
from enzywizard_agent.utils.checkpoint_utils import create_in_memory_checkpointer
from enzywizard_agent.utils.IO_utils import create_session_id, normalize_path

from evaluation_utils import aggregate_usage, runtime_usage_payload


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES_PATH = Path(__file__).with_name("full_workflow_cases.json")
DEFAULT_BASE_URL = (
    "https://ws-gwd0ta1p7z9yx2r9.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
)


def build_parser() -> argparse.ArgumentParser:
    """创建端到端工作流测试参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "Test Planning, repeated plan revisions, and real multi-tool workflows."
        )
    )
    parser.add_argument(
        "--api_key",
        default=os.environ.get("ENZYWIZARD_API_KEY"),
        help="LLM API key; defaults to ENZYWIZARD_API_KEY.",
    )
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default="qwen3.7-plus")
    parser.add_argument(
        "--reasoning_effort",
        default=None,
        help="Optional OpenAI-compatible reasoning effort, such as 'none'.",
    )
    parser.add_argument(
        "--workdir",
        default="~/enzywizard-agent-full-workflow-tests",
        help="Parent test directory; each run creates a timestamped subdirectory.",
    )
    parser.add_argument("--cases", default=str(DEFAULT_CASES_PATH))
    parser.add_argument(
        "--case_ids",
        nargs="*",
        help="Run only the specified cases; defaults to all cases.",
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        help=(
            "Run cases matching any tag, such as smoke, docking, mutant, batch, "
            "or revision. This can be combined with --case_ids."
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Approve and execute the final ToolPlan. Without this option, only "
            "planning and revision loops are checked."
        ),
    )
    parser.add_argument(
        "--output",
        help="Result JSON path; defaults to <run_dir>/results.json.",
    )
    parser.add_argument(
        "--show_plans",
        action="store_true",
        help="Show concise plan reasons and asserted parameters without full JSON.",
    )
    parser.add_argument(
        "--verbose_internal",
        action="store_true",
        help="Show internal Planning and Tool Calling debug output.",
    )
    return parser


def route_from_state(state: AgentState) -> str:
    """测试图使用与正式图相同的 route 状态契约。"""

    route = state.get("route")
    if route is None or not route.strip():
        raise ValueError("Missing route in agent state.")
    return route


def build_workflow_test_graph():
    """构建经过 Router 且截止 Tool Calling 的专项测试图。"""

    workflow = StateGraph(AgentState)
    workflow.add_node("request_routing", request_routing_node)
    workflow.add_node("planning", planning_node)
    workflow.add_node(
        "user_confirmation_interrupt",
        user_confirmation_interrupt_node,
    )
    workflow.add_node("tool_calling", tool_calling_node)
    workflow.set_entry_point("request_routing")
    workflow.add_conditional_edges(
        "request_routing",
        route_from_state,
        {
            "request_to_planning": "planning",
            # 该测试文件中的案例都必须是新工具工作流。若 Router 把案例送往
            # History Query 或 Analysis，图会直接结束，后续的确认中断断言会失败。
            "request_to_history_query": END,
            "request_to_analysis": END,
        },
    )
    workflow.add_conditional_edges(
        "planning",
        route_from_state,
        {
            "tool_plan_generated": "user_confirmation_interrupt",
            "tool_unused": END,
        },
    )
    workflow.add_conditional_edges(
        "user_confirmation_interrupt",
        route_from_state,
        {
            "tool_execution_approved": "tool_calling",
            "tool_plan_revision_requested": "request_routing",
            "tool_execution_rejected": END,
        },
    )
    workflow.add_edge("tool_calling", END)
    return workflow.compile(checkpointer=create_in_memory_checkpointer())


def load_cases(cases_path: Path) -> list[dict[str, Any]]:
    """读取并校验声明式工作流案例。"""

    payload = json.loads(cases_path.read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(cases, list) or not cases:
        raise ValueError("The case file must contain a non-empty 'cases' list.")

    known_case_ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each workflow case must be a JSON object.")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("Each workflow case requires a non-empty case_id.")
        if case_id in known_case_ids:
            raise ValueError(f"Duplicate workflow case ID: {case_id}.")
        if not isinstance(case.get("message"), str) or not case["message"].strip():
            raise ValueError(f"Case {case_id} requires a non-empty message.")
        tags = case.get("tags", [])
        if (
            not isinstance(tags, list)
            or any(not isinstance(tag, str) or not tag.strip() for tag in tags)
            or len(tags) != len(set(tags))
        ):
            raise ValueError(
                f"Case {case_id} tags must be unique non-empty strings."
            )

        actions = case.get("confirmation_actions")
        if not isinstance(actions, list) or not actions:
            raise ValueError(f"Case {case_id} requires confirmation_actions.")
        if actions[-1].get("decision") not in {"approved", "rejected"}:
            raise ValueError(
                f"Case {case_id} must end with approved or rejected confirmation."
            )
        for action in actions:
            decision = action.get("decision")
            revision_request = action.get("revision_request")
            if decision not in {"approved", "revision_requested", "rejected"}:
                raise ValueError(
                    f"Case {case_id} contains invalid decision: {decision!r}."
                )
            if decision == "revision_requested" and not revision_request:
                raise ValueError(
                    f"Case {case_id} revision_requested requires revision_request."
                )
            if decision != "revision_requested" and revision_request is not None:
                raise ValueError(
                    f"Case {case_id} {decision} requires revision_request=null."
                )
        known_case_ids.add(case_id)
    return cases


def select_cases(
    cases: list[dict[str, Any]],
    case_ids: list[str] | None,
    tags: list[str] | None,
) -> list[dict[str, Any]]:
    """按 case_id 和标签筛选；多个标签使用任一匹配语义。"""

    selected_cases = cases
    if case_ids:
        case_map = {case["case_id"]: case for case in cases}
        unknown_case_ids = sorted(set(case_ids) - set(case_map))
        if unknown_case_ids:
            raise ValueError(f"Unknown case IDs: {unknown_case_ids}.")
        selected_cases = [case_map[case_id] for case_id in case_ids]

    if tags:
        requested_tags = {tag.strip() for tag in tags if tag.strip()}
        known_tags = {
            tag for case in cases for tag in case.get("tags", [])
        }
        unknown_tags = sorted(requested_tags - known_tags)
        if unknown_tags:
            raise ValueError(f"Unknown workflow tags: {unknown_tags}.")
        selected_cases = [
            case
            for case in selected_cases
            if requested_tags.intersection(case.get("tags", []))
        ]

    if not selected_cases:
        raise ValueError("No workflow cases matched the requested filters.")
    return selected_cases


def resolve_input_path(value: str | None) -> str | None:
    """把案例中的相对输入路径转换为项目内绝对路径。"""

    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return normalize_path(path)


def create_run_directory(parent: Path) -> Path:
    """为本次批量测试创建不会覆盖历史结果的目录。"""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = parent.expanduser().resolve() / f"run_{timestamp}_{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def build_initial_state(
    case: dict[str, Any],
    api_key: str,
    base_url: str,
    model: str,
    reasoning_effort: str | None,
    case_workdir: Path,
) -> AgentState:
    """构造与正式 Agent 首轮一致的状态，但不启动正式未完成图。"""

    inputs = case.get("inputs", {})
    actions = case["confirmation_actions"]
    return {
        "config": AgentConfig(
            api_key=api_key,
            base_url=base_url,
            model=model,
            reasoning_effort=reasoning_effort,
            workdir=case_workdir,
            max_turns=max(3, len(actions)),
        ),
        "session_id": create_session_id(),
        "current_turn_id": 1,
        "current_plan_id": 0,
        "agent_inputs": AgentInputs(
            structure_file=resolve_input_path(inputs.get("structure_file")),
            msa_file=resolve_input_path(inputs.get("msa_file")),
            mut_structure_file=resolve_input_path(inputs.get("mut_structure_file")),
            mut_msa_file=resolve_input_path(inputs.get("mut_msa_file")),
            substrate_name=(
                str(inputs["substrate_name"]).strip()
                if inputs.get("substrate_name")
                else None
            ),
            amino_acid_substitution=(
                str(inputs["amino_acid_substitution"]).strip().upper()
                if inputs.get("amino_acid_substitution")
                else None
            ),
        ),
        "current_user_message": case["message"].strip(),
        "user_messages": [],
        "user_message_summary": "",
        "planning_results": [],
        "planning_summary": "",
        "user_confirmations": [],
        "user_confirmation_summary": "",
        "tool_executions": [],
        "tool_calling_summary": "",
        "analysis_results": [],
        "analysis_summary": "",
        "cleaned_files": CleanedFiles(
            **{field_name: None for field_name in CleanedFiles.model_fields}
        ),
        "current_plan_artifacts": [],
        "artifacts": [],
        "artifact_summary": "",
    }


def unique_plan_tools(payload: dict[str, Any]) -> list[str]:
    """从确认 payload 按首次出现顺序汇总唯一工具名。"""

    return list(
        dict.fromkeys(call["tool_name"] for call in payload.get("calls", []))
    )


def call_label(call: dict[str, Any]) -> str:
    """将一次 ToolCall 转换为便于比较的 tool:side 标签。"""

    return f"{call['tool_name']}:{call['side']}"


def check_parameter_values(
    calls: list[dict[str, Any]],
    expectations: list[dict[str, Any]],
) -> list[str]:
    """检查修改意见是否落入最终 ToolCall 参数。"""

    failures: list[str] = []
    for expectation in expectations:
        matches = [
            call
            for call in calls
            if call["tool_name"] == expectation["tool_name"]
            and (
                expectation.get("side") is None
                or call["side"] == expectation["side"]
            )
        ]
        if len(matches) != 1:
            failures.append(
                "The plan must contain exactly one call for "
                f"{expectation['tool_name']}:{expectation.get('side', '*')}, "
                f"but found {len(matches)}."
            )
            continue
        parameter_name = expectation["parameter_name"]
        actual_value = matches[0]["arguments"].get(parameter_name)
        if actual_value != expectation["value"]:
            failures.append(
                f"Plan parameter {expectation['tool_name']}.{parameter_name} "
                f"must be {expectation['value']!r}, got {actual_value!r}."
            )
    return failures


def check_resolved_parameter_values(
    call_executions: list[Any],
    expectations: list[dict[str, Any]],
) -> list[str]:
    """检查延迟文件引用在真实执行前解析成了预期 CLI 参数。"""

    failures: list[str] = []
    for expectation in expectations:
        matches = [
            call_execution
            for call_execution in call_executions
            if call_execution.tool_name == expectation["tool_name"]
            and (
                expectation.get("side") is None
                or call_execution.side == expectation["side"]
            )
        ]
        if len(matches) != 1:
            failures.append(
                "The execution record must contain exactly one call for "
                f"{expectation['tool_name']}:{expectation.get('side', '*')}, "
                f"but found {len(matches)}."
            )
            continue
        parameter_name = expectation["parameter_name"]
        actual_value = matches[0].resolved_arguments.get(parameter_name)
        if actual_value != expectation["value"]:
            failures.append(
                f"Resolved runtime parameter "
                f"{expectation['tool_name']}.{parameter_name} must be "
                f"{expectation['value']!r}, got {actual_value!r}."
            )
    return failures


def evaluate_plan_payload(
    payload: dict[str, Any],
    expected_plan: dict[str, Any],
    expected_plan_id: int,
) -> list[str]:
    """验证每次确认中断展示的计划，而不是只检查最终执行结果。"""

    failures: list[str] = []
    if payload.get("interrupt_type") != "tool_plan_confirmation":
        failures.append(
            f"Unexpected confirmation interrupt type: "
            f"{payload.get('interrupt_type')!r}."
        )
    if payload.get("turn_id") != 1:
        failures.append(f"turn_id must be 1, got {payload.get('turn_id')!r}.")
    if payload.get("plan_id") != expected_plan_id:
        failures.append(
            f"plan_id must be {expected_plan_id}, "
            f"got {payload.get('plan_id')!r}."
        )

    calls = payload.get("calls", [])
    selected_tools = unique_plan_tools(payload)
    required_tools = set(expected_plan.get("required_tools", []))
    missing_tools = sorted(required_tools - set(selected_tools))
    if missing_tools:
        failures.append(f"The plan is missing required tools: {missing_tools}.")

    forbidden_tools = set(expected_plan.get("forbidden_tools", []))
    present_forbidden_tools = sorted(forbidden_tools & set(selected_tools))
    if present_forbidden_tools:
        failures.append(
            f"The plan contains forbidden tools: {present_forbidden_tools}."
        )

    if "exact_tools" in expected_plan:
        exact_tools = expected_plan["exact_tools"]
        if selected_tools != exact_tools:
            failures.append(
                f"The tool list must be {exact_tools!r}, got {selected_tools!r}."
            )
    minimum_call_count = expected_plan.get("minimum_call_count")
    if minimum_call_count is not None and len(calls) < minimum_call_count:
        failures.append(
            f"The plan must contain at least {minimum_call_count} calls, "
            f"got {len(calls)}."
        )
    failures.extend(
        check_parameter_values(calls, expected_plan.get("parameter_values", []))
    )
    return failures


def summarize_plan(payload: dict[str, Any]) -> dict[str, Any]:
    """保留计划审计所需内容，不把 API 配置写入结果。"""

    return {
        "turn_id": payload.get("turn_id"),
        "plan_id": payload.get("plan_id"),
        "purpose_type": payload.get("purpose_type"),
        "summary": payload.get("summary"),
        "plan_reason": payload.get("plan_reason"),
        "selected_tools": unique_plan_tools(payload),
        "call_sequence": [call_label(call) for call in payload.get("calls", [])],
        "calls": payload.get("calls", []),
    }


def invoke_graph(
    graph: Any,
    value: Any,
    graph_config: dict[str, Any],
    verbose: bool,
):
    """默认隐藏节点内部调试输出，由批量 runner 统一输出中文进度。"""

    if verbose:
        return graph.invoke(value, config=graph_config)
    with Path(os.devnull).open("w", encoding="utf-8") as sink:
        with redirect_stdout(sink):
            return graph.invoke(value, config=graph_config)


def _model_payload(value: Any) -> dict[str, Any] | None:
    """Serialize a Pydantic state value without exposing the full Agent config."""

    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return dict(value)
    return {"value": str(value)}


def build_terminal_diagnostics(
    result: dict[str, Any],
    initial_state: AgentState,
    *,
    failure_stage: str | None = None,
) -> dict[str, Any]:
    """Preserve the terminal Router/Planning state for pre-confirmation failures."""

    routing = result.get("current_request_routing") or initial_state.get(
        "current_request_routing"
    )
    planning = result.get("current_planning_result") or initial_state.get(
        "current_planning_result"
    )
    runtime = result.get("runtime_session") or initial_state.get("runtime_session")
    return {
        "failure_stage": failure_stage,
        "route": result.get("route") or initial_state.get("route"),
        "request_routing": _model_payload(routing),
        "planning_result": _model_payload(planning),
        "runtime": runtime_usage_payload(runtime),
    }


def missing_confirmation_failure(
    result: dict[str, Any],
    action_index: int,
) -> tuple[str, str]:
    """Return one root-cause-oriented error for a missing confirmation interrupt."""

    routing = result.get("current_request_routing")
    if routing is None:
        return (
            "request_routing",
            "Workflow ended before confirmation without a RequestRoutingResult.",
        )
    if routing.routing_status != "succeeded" or routing.output is None:
        detail = routing.error_message or routing.summary
        return (
            "request_routing",
            f"Request routing ended before confirmation: {detail}",
        )
    if routing.route != "request_to_planning":
        return (
            "request_routing",
            "Expected a tool workflow routed to Planning before confirmation, "
            f"got request_type={routing.output.request_type!r}, "
            f"route={routing.route!r}.",
        )

    planning = result.get("current_planning_result")
    if planning is None:
        return (
            "planning",
            "Planning ended before confirmation without a PlanningResult.",
        )
    if planning.planning_flag != "tool_plan_generated" or planning.tool_plan is None:
        detail = "; ".join(planning.error_messages) or planning.summary
        return (
            "planning",
            "Planning ended before confirmation: "
            f"flag={planning.planning_flag!r}; {detail}",
        )
    return (
        "confirmation",
        f"Expected a confirmation interrupt before action {action_index + 1}.",
    )


def print_plan_summary(
    plan_snapshot: dict[str, Any],
    action: dict[str, Any],
    show_details: bool,
) -> None:
    """输出计划关键路径；即使显示详情也不展开完整 JSON。"""

    plan_id = plan_snapshot["plan_id"]
    call_sequence = " -> ".join(plan_snapshot["call_sequence"])
    print(
        f"  [计划 {plan_id}] {len(plan_snapshot['calls'])} 步：{call_sequence}",
        flush=True,
    )
    decision = action["decision"]
    if decision == "revision_requested":
        print(f"  [确认] 请求修改：{action['revision_request']}", flush=True)
    elif decision == "approved":
        print("  [确认] 批准该计划", flush=True)
    else:
        print("  [确认] 拒绝该计划，不执行工具", flush=True)

    if not show_details:
        return
    reason = " ".join(str(plan_snapshot.get("plan_reason") or "").split())
    if len(reason) > 500:
        reason = f"{reason[:497]}..."
    if reason:
        print(f"  [理由] {reason}", flush=True)
    for expectation in action.get("expected_plan", {}).get(
        "parameter_values", []
    ):
        print(
            "  [参数检查] "
            f"{expectation['tool_name']}:{expectation.get('side', '*')} "
            f"{expectation['parameter_name']}={expectation['value']!r}",
            flush=True,
        )


def compact_log_tail(value: str | None, max_lines: int = 8, max_chars: int = 1200) -> str:
    """压缩错误输出，只保留最可能包含根因的尾部内容。"""

    if not value:
        return ""
    lines = [line.rstrip() for line in value.splitlines() if line.strip()]
    compact = "\n".join(lines[-max_lines:])
    if len(compact) > max_chars:
        compact = compact[-max_chars:]
    return compact


def print_case_failure_details(result: dict[str, Any]) -> None:
    """展示断言失败和真实工具失败的关键诊断信息。"""

    for failure in result["failures"]:
        print(f"  [失败原因] {failure}", flush=True)

    diagnostics = result.get("terminal_diagnostics") or {}
    if diagnostics.get("failure_stage"):
        print(f"  [失败阶段] {diagnostics['failure_stage']}", flush=True)
    if diagnostics.get("route"):
        print(f"  [终止路由] {diagnostics['route']}", flush=True)

    execution = result.get("tool_execution")
    if not execution or execution.get("execution_status") != "failed":
        print(f"  [案例目录] {result['case_workdir']}", flush=True)
        return
    failed_step_index = execution.get("failed_step_index")
    failed_calls = [
        call
        for call in execution.get("call_executions", [])
        if call.get("step_index") == failed_step_index
    ]
    if not failed_calls:
        return
    failed_call = failed_calls[0]
    print(
        "  [失败步骤] "
        f"步骤={failed_step_index}，工具={failed_call.get('tool_name')}，"
        f"侧别={failed_call.get('side')}，退出码={failed_call.get('exit_code')}",
        flush=True,
    )
    if failed_call.get("error_message"):
        print(f"  [工具错误] {failed_call['error_message']}", flush=True)
    log_tail = compact_log_tail(
        failed_call.get("log_excerpt") or failed_call.get("stderr_excerpt")
    )
    if log_tail:
        print("  [日志尾部]", flush=True)
        for line in log_tail.splitlines():
            print(f"    {line}", flush=True)
    print(f"  [案例目录] {result['case_workdir']}", flush=True)


def build_results_payload(
    args: argparse.Namespace,
    run_dir: Path,
    selected_count: int,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """构建可在每个案例结束后覆盖保存的批量结果。"""

    return {
        "mode": "execute" if args.execute else "plan_only",
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "base_url": args.base_url,
        "run_dir": str(run_dir),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "selected": selected_count,
            "completed": len(results),
            "passed": sum(result["status"] == "passed" for result in results),
            "planned": sum(result["status"] == "planned" for result in results),
            "failed": sum(result["status"] == "failed" for result in results),
            "remaining": selected_count - len(results),
        },
        "usage": aggregate_usage(result.get("usage") for result in results),
        "results": results,
    }


def save_results(output_path: Path, payload: dict[str, Any]) -> None:
    """原子更新结果，避免长任务中断后留下半份 JSON。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(output_path)


def evaluate_completed_state(
    result: dict[str, Any],
    case: dict[str, Any],
    plan_snapshots: list[dict[str, Any]],
) -> list[str]:
    """检查确认历史、只执行最终计划以及真实多工具执行结果。"""

    failures: list[str] = []
    actions = case["confirmation_actions"]
    planning_results = result.get("planning_results", [])
    confirmations = result.get("user_confirmations", [])
    tool_executions = result.get("tool_executions", [])

    if len(planning_results) != len(actions):
        failures.append(
            f"Expected {len(actions)} PlanningResult records, "
            f"got {len(planning_results)}."
        )
    if len(confirmations) != len(actions):
        failures.append(
            f"Expected {len(actions)} user confirmation records, "
            f"got {len(confirmations)}."
        )
    actual_decisions = [confirmation.decision for confirmation in confirmations]
    expected_decisions = [action["decision"] for action in actions]
    if actual_decisions != expected_decisions:
        failures.append(
            f"Confirmation decisions must be {expected_decisions!r}, "
            f"got {actual_decisions!r}."
        )
    if len(tool_executions) != 1:
        failures.append(
            f"Expected one ToolExecution, got {len(tool_executions)}."
        )

    execution = result.get("current_tool_execution")
    if execution is None:
        failures.append("current_tool_execution is missing after final approval.")
        return failures
    expected_status = case.get("expected_execution_status", "succeeded")
    if execution.execution_status != expected_status:
        failures.append(
            f"Execution status must be {expected_status!r}, "
            f"got {execution.execution_status!r}."
        )

    final_call_sequence = plan_snapshots[-1]["call_sequence"]
    executed_call_sequence = [
        f"{call_execution.tool_name}:{call_execution.side}"
        for call_execution in execution.call_executions
    ]
    if execution.execution_status == "succeeded" and (
        executed_call_sequence != final_call_sequence
    ):
        failures.append(
            "Execution order does not match the final approved plan: "
            f"{executed_call_sequence!r} != {final_call_sequence!r}."
        )
    failures.extend(
        check_resolved_parameter_values(
            execution.call_executions,
            case.get("expected_resolved_parameters", []),
        )
    )

    # 被修改掉的计划只允许计算路径，不应创建任何目录或产物。
    for old_plan in plan_snapshots[:-1]:
        existing_old_dirs = [
            call["tool_execution_dir"]
            for call in old_plan["calls"]
            if Path(call["tool_execution_dir"]).exists()
        ]
        if existing_old_dirs:
            failures.append(
                "A superseded plan must not create execution directories: "
                f"{existing_old_dirs}."
            )
    return failures


def evaluate_rejected_state(
    result: dict[str, Any],
    case: dict[str, Any],
    plan_snapshots: list[dict[str, Any]],
) -> list[str]:
    """验证最终拒绝会结束当前图且不会执行任何已生成计划。"""

    failures: list[str] = []
    actions = case["confirmation_actions"]
    planning_results = result.get("planning_results", [])
    confirmations = result.get("user_confirmations", [])
    if len(planning_results) != len(actions):
        failures.append(
            f"Expected {len(actions)} PlanningResult records, "
            f"got {len(planning_results)}."
        )
    if len(confirmations) != len(actions):
        failures.append(
            f"Expected {len(actions)} user confirmation records, "
            f"got {len(confirmations)}."
        )
    if result.get("tool_executions"):
        failures.append("A rejected plan must not produce ToolExecution records.")
    if result.get("current_tool_execution") is not None:
        failures.append("current_tool_execution must be empty after plan rejection.")

    existing_dirs = [
        call["tool_execution_dir"]
        for plan in plan_snapshots
        for call in plan["calls"]
        if Path(call["tool_execution_dir"]).exists()
    ]
    if existing_dirs:
        failures.append(
            "A rejected plan must not create execution directories: "
            f"{existing_dirs}."
        )
    return failures


def run_case(
    case: dict[str, Any],
    api_key: str,
    base_url: str,
    model: str,
    reasoning_effort: str | None,
    run_dir: Path,
    execute_tools: bool,
    show_plans: bool,
    verbose_internal: bool,
) -> dict[str, Any]:
    """运行一次包含任意次计划修改的测试案例。"""

    started_at = time.monotonic()
    case_id = case["case_id"]
    case_workdir = run_dir / case_id
    case_workdir.mkdir(parents=True, exist_ok=False)
    graph = build_workflow_test_graph()
    state = build_initial_state(
        case,
        api_key,
        base_url,
        model,
        reasoning_effort,
        case_workdir,
    )
    session_id = state["session_id"]
    graph_config = {"configurable": {"thread_id": session_id}}
    actions = case["confirmation_actions"]
    plan_snapshots: list[dict[str, Any]] = []
    failures: list[str] = []
    result: dict[str, Any] = {}
    execution_was_requested = False
    invalid_approved_plan = False
    confirmation_flow_failed = False
    failure_stage: str | None = None

    try:
        result = invoke_graph(graph, state, graph_config, verbose_internal)
        for action_index, action in enumerate(actions):
            if not result.get("__interrupt__"):
                failure_stage, failure = missing_confirmation_failure(
                    result,
                    action_index,
                )
                failures.append(failure)
                confirmation_flow_failed = True
                break

            payload = get_single_interrupt_payload(result)
            plan_failures = evaluate_plan_payload(
                payload,
                action.get("expected_plan", {}),
                expected_plan_id=action_index + 1,
            )
            failures.extend(plan_failures)
            plan_snapshot = summarize_plan(payload)
            plan_snapshots.append(plan_snapshot)
            print_plan_summary(plan_snapshot, action, show_plans)

            # 已批准的计划若不满足案例语义契约，继续执行只会浪费真实工具时间，
            # 还可能把“命令成功”误读成“规划正确”。记录失败后直接进入下一案例。
            if (
                action["decision"] == "approved"
                and execute_tools
                and plan_failures
            ):
                invalid_approved_plan = True
                print(
                    "[跳过执行] 已批准计划未通过语义检查，不调用真实工具。",
                    flush=True,
                )
                break

            # 计划模式仍会执行所有 revision_requested，从而真正测试多次
            # Planning 循环；只在最终 approved 即将触发真实工具前停止。
            if action["decision"] == "approved" and not execute_tools:
                break

            resume_value = {
                "turn_id": payload["turn_id"],
                "plan_id": payload["plan_id"],
                "decision": action["decision"],
                "revision_request": action["revision_request"],
            }
            if action["decision"] == "approved":
                execution_was_requested = True
            result = invoke_graph(
                graph,
                Command(resume=resume_value),
                graph_config,
                verbose_internal,
            )

        if execute_tools and actions[-1]["decision"] == "approved":
            if invalid_approved_plan:
                pass
            elif confirmation_flow_failed:
                pass
            elif not execution_was_requested:
                failure_stage = "confirmation"
                failures.append(
                    "The final plan was approved, but graph execution did not resume."
                )
            elif result.get("__interrupt__"):
                failure_stage = "confirmation"
                failures.append(
                    "The graph remained interrupted after final approval."
                )
            else:
                failures.extend(
                    evaluate_completed_state(result, case, plan_snapshots)
                )
        elif execute_tools and actions[-1]["decision"] == "rejected":
            if result.get("__interrupt__"):
                failure_stage = "confirmation"
                failures.append(
                    "The graph remained interrupted after final rejection."
                )
            else:
                failures.extend(
                    evaluate_rejected_state(result, case, plan_snapshots)
                )
    except Exception as error:
        failure_stage = failure_stage or "runner"
        failures.append(f"{type(error).__name__}: {error}")

    execution = result.get("current_tool_execution") if result else None
    if failures and failure_stage is None:
        if invalid_approved_plan:
            failure_stage = "plan_assertion"
        elif execution is not None and execution.execution_status == "failed":
            failure_stage = "tool_execution"
        else:
            failure_stage = "workflow_assertion"
    diagnostics = build_terminal_diagnostics(
        result,
        state,
        failure_stage=failure_stage,
    )
    status = (
        "failed"
        if failures
        else "passed"
        if execute_tools
        else "planned"
    )
    return {
        "case_id": case_id,
        "description": case.get("description"),
        "tags": case.get("tags", []),
        "status": status,
        "execute_tools": execute_tools,
        "session_id": session_id,
        "case_workdir": str(case_workdir),
        "failures": failures,
        "plan_snapshots": plan_snapshots,
        "planning_result_count": len(result.get("planning_results", [])) if result else 0,
        "confirmation_count": len(result.get("user_confirmations", [])) if result else 0,
        "tool_execution": (
            execution.model_dump(mode="json") if execution is not None else None
        ),
        "terminal_diagnostics": diagnostics,
        "usage": diagnostics["runtime"],
        "cleaned_files": (
            result["cleaned_files"].model_dump(mode="json")
            if result and result.get("cleaned_files") is not None
            else None
        ),
        "artifact_count": len(result.get("artifacts", [])) if result else 0,
        "duration_seconds": round(time.monotonic() - started_at, 3),
    }


def main() -> int:
    """批量运行案例；未显式 --execute 时绝不调用 EnzyWizard 工具。"""

    args = build_parser().parse_args()
    if not args.api_key or not str(args.api_key).strip():
        print(
            "[错误] 请传入 --api_key 或设置 ENZYWIZARD_API_KEY。",
            file=sys.stderr,
        )
        return 2

    try:
        all_cases = load_cases(Path(args.cases).expanduser().resolve())
        selected_cases = select_cases(all_cases, args.case_ids, args.tags)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"[错误] 无法读取工作流案例：{error}", file=sys.stderr)
        return 2

    run_dir = create_run_directory(Path(args.workdir))
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else run_dir / "results.json"
    )
    mode = "执行真实工具" if args.execute else "仅检查计划、确认和修改流程"
    print(f"[运行模式] {mode}", flush=True)
    print(f"[模型] {args.model}", flush=True)
    print(f"[推理强度] {args.reasoning_effort or '模型默认值'}", flush=True)
    if args.tags:
        print(f"[标签] {', '.join(args.tags)}", flush=True)
    print(f"[案例数量] {len(selected_cases)}", flush=True)
    print(f"[运行目录] {run_dir}", flush=True)
    print(f"[结果文件] {output_path}", flush=True)
    print("[失败策略] 单个案例失败后记录错误并继续", flush=True)

    results: list[dict[str, Any]] = []
    save_results(
        output_path,
        build_results_payload(args, run_dir, len(selected_cases), results),
    )
    for case_index, case in enumerate(selected_cases, start=1):
        case_id = case["case_id"]
        print(
            f"\n[案例 {case_index}/{len(selected_cases)}] 开始：{case_id}",
            flush=True,
        )
        print(f"  [说明] {case.get('description', '')}", flush=True)
        case_started_at = time.monotonic()
        try:
            result = run_case(
                case,
                str(args.api_key).strip(),
                str(args.base_url).strip(),
                str(args.model).strip(),
                args.reasoning_effort,
                run_dir,
                args.execute,
                args.show_plans,
                args.verbose_internal,
            )
        except Exception as error:
            # runner 自身的意外错误也只终止当前案例，保证长批次继续推进。
            result = {
                "case_id": case_id,
                "description": case.get("description"),
                "tags": case.get("tags", []),
                "status": "failed",
                "execute_tools": args.execute,
                "session_id": None,
                "case_workdir": str(run_dir / case_id),
                "failures": [f"{type(error).__name__}: {error}"],
                "plan_snapshots": [],
                "planning_result_count": 0,
                "confirmation_count": 0,
                "tool_execution": None,
                "terminal_diagnostics": {
                    "failure_stage": "runner",
                    "route": None,
                    "request_routing": None,
                    "planning_result": None,
                    "runtime": runtime_usage_payload(None),
                },
                "usage": runtime_usage_payload(None),
                "cleaned_files": None,
                "artifact_count": 0,
                "duration_seconds": round(time.monotonic() - case_started_at, 3),
            }
        results.append(result)
        save_results(
            output_path,
            build_results_payload(args, run_dir, len(selected_cases), results),
        )

        status = result["status"]
        duration = result["duration_seconds"]
        if status == "failed":
            print(f"[案例结果] 失败，用时 {duration:.1f} 秒", flush=True)
            print_case_failure_details(result)
            print("[继续执行] 已记录失败，开始下一个案例", flush=True)
        elif status == "planned":
            print(
                f"[案例结果] 计划检查通过，用时 {duration:.1f} 秒；"
                "未执行真实工具",
                flush=True,
            )
        else:
            execution = result.get("tool_execution")
            executed_count = (
                len(execution.get("call_executions", [])) if execution else 0
            )
            print(
                f"[案例结果] 通过，用时 {duration:.1f} 秒，"
                f"成功执行 {executed_count} 个工具调用，"
                f"登记 {result['artifact_count']} 个产物",
                flush=True,
            )
        completed_failed = sum(
            item["status"] == "failed" for item in results
        )
        print(
            f"[当前进度] 已完成 {len(results)}/{len(selected_cases)}，"
            f"失败 {completed_failed}",
            flush=True,
        )

    passed_count = sum(result["status"] == "passed" for result in results)
    planned_count = sum(result["status"] == "planned" for result in results)
    failed_count = sum(result["status"] == "failed" for result in results)
    print(
        "\n[全部完成] "
        f"总数={len(results)}，通过={passed_count}，"
        f"仅计划={planned_count}，失败={failed_count}",
        flush=True,
    )
    print(f"[完整结果] {output_path}", flush=True)
    return 1 if failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
