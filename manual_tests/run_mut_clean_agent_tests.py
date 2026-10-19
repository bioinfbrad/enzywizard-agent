"""通过真实 Agent Tool Calling 流程测试 enzywizard-mut-clean 1.0.4。"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from enzywizard_agent.conf import AgentConfig
from enzywizard_agent.data_models.agent_inputs import AgentInputs
from enzywizard_agent.data_models.artifact import CleanedFiles
from enzywizard_agent.data_models.planning import PlanningResult, ToolSelectionOutput
from enzywizard_agent.data_models.user_confirmation import UserConfirmation
from enzywizard_agent.nodes.tool_calling import tool_calling_node
from enzywizard_agent.services.planning_service import generate_default_tool_plan
from enzywizard_agent.state import AgentState


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WT_INPUT = PROJECT_ROOT / "test_inputs" / "1ZG4_WT.pdb"
DEFAULT_MUT_INPUT = PROJECT_ROOT / "test_inputs" / "1ZG6_S70G.pdb"
EXPECTED_PACKAGE_VERSION = "1.0.4"


def build_parser() -> argparse.ArgumentParser:
    """创建真实 Agent 集成测试参数。"""

    parser = argparse.ArgumentParser(
        description="Run real Agent integration tests for enzywizard-mut-clean 1.0.4."
    )
    parser.add_argument("--wt_input", default=str(DEFAULT_WT_INPUT))
    parser.add_argument("--mut_input", default=str(DEFAULT_MUT_INPUT))
    parser.add_argument(
        "--workdir",
        default="/tmp/enzywizard-mut-clean-agent-tests",
        help="Parent directory for timestamped test runs.",
    )
    parser.add_argument(
        "--output",
        help="Optional JSON result path. Defaults to <run_dir>/results.json.",
    )
    parser.add_argument(
        "--case_ids",
        nargs="*",
        choices=("successful_execution", "failed_execution"),
        help="Run selected cases. By default both cases are run.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print complete ToolExecution JSON for each case.",
    )
    return parser


def create_run_directory(parent: Path) -> Path:
    """创建独立目录，避免重复测试覆盖之前的真实工具产物。"""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = parent.expanduser().resolve() / f"run_{timestamp}_{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def empty_cleaned_files() -> CleanedFiles:
    """构造尚未执行任何 clean/mut-clean 的文件状态。"""

    return CleanedFiles(
        **{field_name: None for field_name in CleanedFiles.model_fields}
    )


def build_approved_state(
    run_dir: Path,
    wt_input: Path,
    mut_input: Path,
    mutation: str,
    add_h: bool,
) -> AgentState:
    """用真实规划规则生成仅包含 mut-clean 的已批准计划。"""

    state: AgentState = {
        # 此测试不调用 LLM，但 ToolPlan 的目录生成仍使用 AgentConfig.workdir。
        "config": AgentConfig(
            api_key="not-used",
            base_url="not-used",
            model="not-used",
            workdir=run_dir,
            max_turns=1,
        ),
        "session_id": f"mut_clean_test_{uuid.uuid4().hex}",
        "current_turn_id": 1,
        "current_plan_id": 1,
        "current_user_message": "Prepare this WT/MUT structure pair.",
        "agent_inputs": AgentInputs(
            structure_file=str(wt_input),
            msa_file=None,
            mut_structure_file=str(mut_input),
            mut_msa_file=None,
            substrate_name=None,
            amino_acid_substitution=mutation,
        ),
        "cleaned_files": empty_cleaned_files(),
        "tool_executions": [],
        "current_plan_artifacts": [],
        "artifacts": [],
    }
    planning_result = PlanningResult(
        turn_id=1,
        plan_id=1,
        planning_route="tool_plan_generated",
        planning_flag="tool_selection_succeeded",
        needs_tool=True,
        summary=None,
        purpose_type="wildtype_mutant_comparison",
        purpose_reason="The test requires paired structure preparation.",
        tool_selection=ToolSelectionOutput(
            tool_selection_status="tool_selection_succeeded",
            selected_tools=["enzywizard-mut-clean"],
            selection_reason="Run the program-managed paired preparation step.",
            error_message="",
        ),
        parameter_extraction=None,
        tool_plan=None,
        error_messages=[],
    )

    # 使用正式 Planning 辅助函数生成命令参数、输出模式和执行目录，确保测试的
    # 不是手工伪造的 ToolCall。成功案例关闭加氢以缩短运行时间；失败案例在
    # 清理前结束，因此可保留默认加氢值并测试 store_false 参数的另一分支。
    tool_plan = generate_default_tool_plan(state, planning_result)
    if len(tool_plan.calls) != 1 or tool_plan.calls[0].tool_name != "enzywizard-mut-clean":
        raise AssertionError("Expected a single enzywizard-mut-clean ToolCall.")
    tool_plan.calls[0].arguments["add_H"] = add_h

    planning_result.tool_plan = tool_plan
    planning_result.planning_flag = "tool_plan_generated"
    planning_result.summary = tool_plan.summary
    state["current_planning_result"] = planning_result
    state["current_user_confirmation"] = UserConfirmation(
        turn_id=1,
        plan_id=1,
        decision="approved",
        revision_request=None,
    )
    return state


def assert_success_state(state: AgentState) -> dict[str, Any]:
    """验证成功执行后的 ToolExecution、Artifact 和 CleanedFiles。"""

    execution = state["current_tool_execution"]
    failures: list[str] = []
    if execution.execution_status != "succeeded":
        failures.append(f"Expected succeeded execution, got {execution.execution_status}.")
    if len(execution.call_executions) != 1:
        failures.append(
            f"Expected one ToolCallExecution, got {len(execution.call_executions)}."
        )

    call_execution = execution.call_executions[0]
    if call_execution.exit_code != 0:
        failures.append(f"Expected exit code 0, got {call_execution.exit_code}.")
    if "--no_add_H" not in call_execution.command:
        failures.append("The Agent command did not translate add_H=False to --no_add_H.")

    expected_artifact_types = {
        "wt_cleaned_structure_cif",
        "wt_cleaned_structure_pdb",
        "wt_cleaned_sequence_fasta",
        "mut_cleaned_structure_cif",
        "mut_cleaned_structure_pdb",
        "mut_cleaned_sequence_fasta",
        "mut_clean_report_json",
        "log_file",
        "agent_stdout_log",
        "agent_stderr_log",
    }
    actual_artifact_types = {
        artifact.artifact_type for artifact in call_execution.artifacts
    }
    missing_artifact_types = sorted(expected_artifact_types - actual_artifact_types)
    if missing_artifact_types:
        failures.append(f"Missing tracked artifact types: {missing_artifact_types}.")

    cleaned_files = state["cleaned_files"]
    tracked_paths = {
        "wt_cif": cleaned_files.wt_cleaned_cif_file,
        "wt_pdb": cleaned_files.wt_cleaned_pdb_file,
        "wt_fasta": cleaned_files.wt_cleaned_fasta_file,
        "mut_cif": cleaned_files.mut_cleaned_cif_file,
        "mut_pdb": cleaned_files.mut_cleaned_pdb_file,
        "mut_fasta": cleaned_files.mut_cleaned_fasta_file,
        "report": cleaned_files.mut_clean_report_file,
    }
    for label, path_value in tracked_paths.items():
        if path_value is None or not Path(path_value).is_file():
            failures.append(f"CleanedFiles did not track an existing {label} file.")
    cleaned_substitution = cleaned_files.cleaned_amino_acid_substitution
    if not cleaned_substitution:
        failures.append("CleanedFiles did not retain cleaned_amino_acid_substitution.")

    if failures:
        raise AssertionError("\n".join(failures))
    return {
        "command": call_execution.command,
        "artifact_types": sorted(actual_artifact_types),
        "cleaned_amino_acid_substitution": cleaned_substitution,
        "tracked_paths": tracked_paths,
    }


def assert_failed_state(state: AgentState) -> dict[str, Any]:
    """验证底层退出码、错误日志和状态保护能够完整传回 Agent。"""

    execution = state["current_tool_execution"]
    failures: list[str] = []
    if execution.execution_status != "failed":
        failures.append(f"Expected failed execution, got {execution.execution_status}.")
    if execution.failed_step_index != 1:
        failures.append(
            f"Expected failed_step_index=1, got {execution.failed_step_index}."
        )
    if len(execution.call_executions) != 1:
        failures.append(
            f"Expected one failed ToolCallExecution, got {len(execution.call_executions)}."
        )

    call_execution = execution.call_executions[0]
    if call_execution.exit_code != 1:
        failures.append(f"Expected tool exit code 1, got {call_execution.exit_code}.")
    if "--no_add_H" in call_execution.command:
        failures.append("The Agent command incorrectly disabled add_H=True.")
    if "Mutation position not found in wild-type structure" not in (
        call_execution.stdout_excerpt
        + call_execution.stderr_excerpt
        + call_execution.log_excerpt
    ):
        failures.append("The mut-clean residue validation error was not retained.")
    if any(
        getattr(state["cleaned_files"], field_name) is not None
        for field_name in CleanedFiles.model_fields
    ):
        failures.append("A failed mut-clean call must not update CleanedFiles.")

    if failures:
        raise AssertionError("\n".join(failures))
    return {
        "command": call_execution.command,
        "exit_code": call_execution.exit_code,
        "error_message": call_execution.error_message,
        "log_excerpt": call_execution.log_excerpt,
    }


def run_case(
    case_id: str,
    run_dir: Path,
    wt_input: Path,
    mut_input: Path,
) -> dict[str, Any]:
    """运行一个真实 Agent 集成案例并返回可序列化结果。"""

    successful_case = case_id == "successful_execution"
    mutation = "S70G" if successful_case else "A25V"
    state = build_approved_state(
        run_dir / case_id,
        wt_input,
        mut_input,
        mutation,
        add_h=not successful_case,
    )
    failures: list[str] = []
    details: dict[str, Any] = {}
    try:
        tool_calling_node(state)
        details = (
            assert_success_state(state)
            if successful_case
            else assert_failed_state(state)
        )
    except Exception as error:
        failures.append(f"{type(error).__name__}: {error}")

    execution = state.get("current_tool_execution")
    return {
        "case_id": case_id,
        "status": "passed" if not failures else "failed",
        "mutation": mutation,
        "failures": failures,
        "details": details,
        "tool_execution": (
            execution.model_dump(mode="json") if execution is not None else None
        ),
    }


def main() -> int:
    """运行成功和失败路径，保存完整 ToolExecution 结果。"""

    args = build_parser().parse_args()
    try:
        installed_version = version("enzywizard-mut-clean")
    except PackageNotFoundError:
        print("[ERROR] enzywizard-mut-clean is not installed.", file=sys.stderr)
        return 2
    if installed_version != EXPECTED_PACKAGE_VERSION:
        print(
            "[ERROR] Expected enzywizard-mut-clean "
            f"{EXPECTED_PACKAGE_VERSION}, found {installed_version}.",
            file=sys.stderr,
        )
        return 2

    wt_input = Path(args.wt_input).expanduser().resolve()
    mut_input = Path(args.mut_input).expanduser().resolve()
    for input_path in (wt_input, mut_input):
        if not input_path.is_file():
            print(f"[ERROR] Test input not found: {input_path}", file=sys.stderr)
            return 2

    run_dir = create_run_directory(Path(args.workdir))
    selected_case_ids = args.case_ids or ["successful_execution", "failed_execution"]
    results: list[dict[str, Any]] = []
    for case_id in selected_case_ids:
        result = run_case(case_id, run_dir, wt_input, mut_input)
        results.append(result)
        print(f"[{'PASS' if result['status'] == 'passed' else 'FAIL'}] {case_id}")
        for failure in result["failures"]:
            print(f"  - {failure}")
        if args.verbose and result["tool_execution"] is not None:
            print(json.dumps(result["tool_execution"], ensure_ascii=False, indent=2))

    passed_count = sum(result["status"] == "passed" for result in results)
    failed_count = len(results) - passed_count
    payload = {
        "tested_package_version": installed_version,
        "wt_input": str(wt_input),
        "mut_input": str(mut_input),
        "run_dir": str(run_dir),
        "summary": {
            "total": len(results),
            "passed": passed_count,
            "failed": failed_count,
        },
        "results": results,
    }
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else run_dir / "results.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nSummary: total={len(results)}, passed={passed_count}, failed={failed_count}")
    print(f"Run directory: {run_dir}")
    print(f"Full results: {output_path}")
    return 1 if failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
