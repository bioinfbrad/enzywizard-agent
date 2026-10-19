"""Orchestrate the layered EnzyWizard-Agent evaluation project.

The default run covers deterministic contracts and all live LLM semantic/planning
cases without executing scientific tools. Pass --execute_tools only for the slower
25-case real-tool regression.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import uuid

import yaml

from evaluation_utils import aggregate_usage


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANUAL_TESTS = Path(__file__).resolve().parent
ROUTING_CASES = MANUAL_TESTS / "request_routing_cases.json"
PLANNING_CASES = MANUAL_TESTS / "planning_cases.json"
WORKFLOW_CASES = MANUAL_TESTS / "full_workflow_cases.json"
TOOL_SPECS = (
    PROJECT_ROOT
    / "src/enzywizard_agent/knowledge_base/enzywizard_tool_specs/enzywizard_tools.yaml"
)
DEFAULT_BASE_URL = (
    "https://ws-gwd0ta1p7z9yx2r9.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
)
SUITE_ORDER = ["contracts", "routing", "planning", "workflows"]
SUITE_LABELS = {
    "contract_request_routing": "请求路由与历史查询契约",
    "contract_planning_compilation": "Planning 需求编译契约",
    "contract_confirmation": "用户确认与恢复契约",
    "contract_tool_calling": "工具调用契约",
    "contract_runtime": "模型兼容、预算、隐私与运行时契约",
    "contract_analysis_and_cross_turn": "分析与跨轮状态契约",
    "live_request_routing_history_analysis": "请求路由、历史查询与分析语义",
    "live_planning": "计划生成语义",
    "router_planning_confirmation_tool_workflows": "完整工作流",
    "analysis_real_workflow_regression": "真实报告分析与跨轮回归",
}
ANALYSIS_REQUIRED_WORKFLOW_CASES = {
    "single_substrate_docking_workflow",
    "mutant_substrate_docking_workflow",
}
ANALYSIS_REQUIRED_REPORT_TYPES = {
    "enzywizard_clean",
    "enzywizard_mut_clean",
    "enzywizard_aaprops",
    "enzywizard_hydrocluster",
    "enzywizard_energy",
    "enzywizard_flexibility",
    "enzywizard_disorder",
    "enzywizard_conservation",
    "enzywizard_embedding",
    "enzywizard_pocket",
    "enzywizard_substrate",
    "enzywizard_dock",
    "enzywizard_interaction",
    "enzywizard_integrate",
    "enzywizard_mut_integrate",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the complete layered EnzyWizard-Agent evaluation suite."
    )
    parser.add_argument(
        "--api_key",
        default=os.environ.get("ENZYWIZARD_API_KEY"),
        help="LLM API key. Defaults to ENZYWIZARD_API_KEY.",
    )
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default="qwen3.7-plus")
    parser.add_argument(
        "--reasoning_effort",
        default=None,
        help=(
            "Optional OpenAI-compatible reasoning effort passed to every live "
            "model suite, for example 'none' to disable Qwen thinking."
        ),
    )
    parser.add_argument(
        "--workdir",
        default="~/enzywizard-agent-comprehensive-tests",
    )
    parser.add_argument(
        "--suites",
        nargs="*",
        choices=SUITE_ORDER,
        default=SUITE_ORDER,
        help="Suites to run in fixed dependency order.",
    )
    parser.add_argument(
        "--execute_tools",
        action="store_true",
        help="Execute real EnzyWizard commands in the 25 workflow cases.",
    )
    parser.add_argument(
        "--repeat_semantic",
        type=int,
        default=1,
        help="Repeat the live routing and Planning cases to measure stability.",
    )
    parser.add_argument(
        "--analysis_run_dir",
        default=None,
        help=(
            "Optional existing successful 25-workflow run directory. When set, "
            "the deterministic Analysis/report regression is included."
        ),
    )
    parser.add_argument("--stop_on_suite_failure", action="store_true")
    parser.add_argument("--show_details", action="store_true")
    return parser


def create_run_directory(parent: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = parent.expanduser().resolve() / f"run_{timestamp}_{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def load_case_count(path: Path) -> int:
    """读取测试案例数量，并在大规模运行前拒绝损坏的案例文件。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"Test case file has no cases: {path}")
    return len(cases)


def validate_workflow_tool_coverage() -> tuple[int, int, list[str]]:
    """确认真实工作流预期计划覆盖当前注册的每一个 EnzyWizard 命令。"""

    tool_payload = yaml.safe_load(TOOL_SPECS.read_text(encoding="utf-8"))
    tool_specs = tool_payload.get("tools") if isinstance(tool_payload, dict) else None
    if not isinstance(tool_specs, dict) or not tool_specs:
        raise ValueError(f"Tool specification file is invalid: {TOOL_SPECS}")
    registered_tools = set(tool_specs)

    workflow_payload = json.loads(WORKFLOW_CASES.read_text(encoding="utf-8"))
    workflow_cases = workflow_payload.get("cases", [])
    covered_tools: set[str] = set()
    for case in workflow_cases:
        for action in case.get("confirmation_actions", []):
            expected_plan = action.get("expected_plan", {})
            covered_tools.update(expected_plan.get("required_tools", []))
            covered_tools.update(expected_plan.get("exact_tools", []))
    missing_tools = sorted(registered_tools - covered_tools)
    return len(registered_tools), len(registered_tools & covered_tools), missing_tools


def analysis_fixture_blockers(workflow_root: Path) -> list[str]:
    """Check only the workflow outputs used by the Analysis regression suite."""

    results_path = workflow_root / "results.json"
    if not results_path.is_file():
        return [f"workflow results are missing: {results_path}"]
    try:
        payload = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [f"workflow results cannot be read: {exc}"]

    passed_cases = {
        item.get("case_id")
        for item in payload.get("results", [])
        if item.get("status") == "passed" and item.get("tool_execution") is not None
    }
    missing_cases = sorted(ANALYSIS_REQUIRED_WORKFLOW_CASES - passed_cases)

    observed_report_types: set[str] = set()
    for path in workflow_root.rglob("*.json"):
        if path == results_path or "input_reports" in path.parts:
            continue
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(report, dict) and isinstance(report.get("report_type"), str):
            observed_report_types.add(report["report_type"])
    missing_reports = sorted(ANALYSIS_REQUIRED_REPORT_TYPES - observed_report_types)

    blockers: list[str] = []
    if missing_cases:
        blockers.append(f"required workflow cases did not pass: {missing_cases}")
    if missing_reports:
        blockers.append(f"required report types are missing: {missing_reports}")
    return blockers


def _suite_payload_path(name: str, run_dir: Path) -> Path | None:
    if name == "live_request_routing_history_analysis":
        matches = sorted((run_dir / "routing").glob("run_*/results.json"))
        return matches[-1] if matches else None
    if name == "live_planning":
        return run_dir / "planning_results.json"
    if name == "router_planning_confirmation_tool_workflows":
        return run_dir / "workflows" / "results.json"
    return None


def attach_suite_metrics(
    suite_result: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    """Attach child case counts, token usage, and semantic failure stages."""

    path = _suite_payload_path(suite_result["suite"], run_dir)
    if path is None or not path.is_file():
        return suite_result
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        suite_result["metrics_error"] = f"Could not read child results: {exc}"
        return suite_result

    summary = payload.get("summary", {})
    if suite_result["suite"] == "router_planning_confirmation_tool_workflows":
        total = int(summary.get("completed", 0))
        passed = int(summary.get("passed", 0)) + int(summary.get("planned", 0))
    else:
        total = int(summary.get("total", 0))
        passed = int(summary.get("passed", 0))
    suite_result["case_summary"] = {
        "total": total,
        "passed": passed,
        "failed": max(0, total - passed),
    }
    suite_result["usage"] = payload.get("usage")
    suite_result["child_results_file"] = str(path)

    stage_counts: Counter[str] = Counter()
    for item in payload.get("results", []):
        failed = item.get("passed") is False or item.get("status") == "failed"
        if not failed:
            continue
        diagnostics = item.get("terminal_diagnostics") or {}
        stage = diagnostics.get("failure_stage")
        if not stage:
            if suite_result["suite"] == "live_request_routing_history_analysis":
                stage = "routing_history_analysis_semantics"
            elif suite_result["suite"] == "live_planning":
                stage = "planning_semantics"
            else:
                stage = "workflow_assertion"
        stage_counts[str(stage)] += 1
    suite_result["failure_stages"] = dict(sorted(stage_counts.items()))
    return suite_result


def test_command(script_name: str, *arguments: str) -> list[str]:
    return [sys.executable, "-u", str(MANUAL_TESTS / script_name), *arguments]


def build_suite_commands(
    args: argparse.Namespace,
    run_dir: Path,
) -> list[tuple[str, list[str]]]:
    commands: list[tuple[str, list[str]]] = []
    requested = set(args.suites)
    if "contracts" in requested:
        commands.extend(
            [
                (
                    "contract_request_routing",
                    test_command("run_request_routing_tests.py"),
                ),
                (
                    "contract_planning_compilation",
                    test_command("run_planning_contract_tests.py"),
                ),
                (
                    "contract_confirmation",
                    test_command("run_user_confirmation_interrupt.py"),
                ),
                (
                    "contract_tool_calling",
                    test_command("run_tool_calling.py"),
                ),
                (
                    "contract_runtime",
                    test_command("run_runtime_contract_tests.py"),
                ),
            ]
        )
        if args.analysis_run_dir:
            commands.append(
                (
                    "contract_analysis_and_cross_turn",
                    test_command(
                        "run_analysis_module_tests.py",
                        "--run_dir",
                        str(Path(args.analysis_run_dir).expanduser().resolve()),
                    ),
                )
            )
    if "routing" in requested:
        routing_args = [
            "--base_url",
            args.base_url,
            "--model",
            args.model,
            "--workdir",
            str(run_dir / "routing"),
            "--repeat",
            str(args.repeat_semantic),
            "--include_analysis",
        ]
        if args.reasoning_effort is not None:
            routing_args.extend(["--reasoning_effort", args.reasoning_effort])
        if args.show_details:
            routing_args.append("--show_details")
        commands.append(
            (
                "live_request_routing_history_analysis",
                test_command("run_request_routing_batch.py", *routing_args),
            )
        )
    if "planning" in requested:
        planning_args = [
            "--base_url",
            args.base_url,
            "--model",
            args.model,
            "--workdir",
            str(run_dir / "planning"),
            "--output",
            str(run_dir / "planning_results.json"),
            "--repeat",
            str(args.repeat_semantic),
        ]
        if args.reasoning_effort is not None:
            planning_args.extend(["--reasoning_effort", args.reasoning_effort])
        commands.append(
            (
                "live_planning",
                test_command("run_planning_batch.py", *planning_args),
            )
        )
    if "workflows" in requested:
        workflow_results = run_dir / "workflows" / "results.json"
        workflow_args = [
            "--base_url",
            args.base_url,
            "--model",
            args.model,
            "--workdir",
            str(run_dir / "workflows"),
            "--output",
            str(workflow_results),
        ]
        if args.reasoning_effort is not None:
            workflow_args.extend(["--reasoning_effort", args.reasoning_effort])
        if args.execute_tools:
            workflow_args.append("--execute")
        if args.show_details:
            workflow_args.append("--show_plans")
        commands.append(
            (
                "router_planning_confirmation_tool_workflows",
                test_command("run_full_workflow_tests.py", *workflow_args),
            )
        )
        if args.execute_tools:
            commands.append(
                (
                    "analysis_real_workflow_regression",
                    test_command(
                        "run_analysis_module_tests.py",
                        "--run_dir",
                        str(run_dir / "workflows"),
                    ),
                )
            )
    return commands


def run_command(
    name: str,
    command: list[str],
    run_dir: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    """Stream one child suite while retaining a complete local log."""

    started_at = time.monotonic()
    log_path = run_dir / "suite_logs" / f"{name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    label = SUITE_LABELS.get(name, name)
    print(f"\n[测试套件] 开始：{label}", flush=True)
    print(f"[详细日志] {log_path}", flush=True)
    # 契约脚本会输出用于调试的结构化明细；总控终端仅显示其最终状态，
    # 完整内容仍写入独立日志，避免一次性测试时淹没案例进度。
    stream_child_output = not (
        name.startswith("contract_")
        or name == "analysis_real_workflow_regression"
    )
    launch_error = ""
    process: subprocess.Popen[str] | None = None
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                if stream_child_output:
                    print(line, end="", flush=True)
                log_file.write(line)
                log_file.flush()
            return_code = process.wait()
    except KeyboardInterrupt:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        print(f"\n[用户终止] 已停止当前套件：{label}", flush=True)
        raise
    except OSError as exc:
        return_code = 127
        launch_error = f"{type(exc).__name__}: {exc}"
        log_path.write_text(
            f"Failed to launch suite {name}: {launch_error}\n",
            encoding="utf-8",
        )
        print(f"[套件错误] {launch_error}", flush=True)
    duration = round(time.monotonic() - started_at, 3)
    status = "passed" if return_code == 0 else "failed"
    print(
        f"[套件结果] {'通过' if status == 'passed' else '失败'}，"
        f"用时 {duration:.1f} 秒：{label}",
        flush=True,
    )
    return {
        "suite": name,
        "status": status,
        "return_code": return_code,
        "duration_seconds": duration,
        "log_file": str(log_path),
        "launch_error": launch_error,
    }


def save_results(
    path: Path,
    args: argparse.Namespace,
    run_dir: Path,
    results: list[dict[str, Any]],
) -> None:
    passed = sum(result["status"] == "passed" for result in results)
    failed = sum(result["status"] == "failed" for result in results)
    blocked = sum(result["status"] == "blocked" for result in results)
    case_summaries = [
        result["case_summary"]
        for result in results
        if result.get("case_summary") is not None
    ]
    case_total = sum(item["total"] for item in case_summaries)
    case_passed = sum(item["passed"] for item in case_summaries)
    failure_stages: Counter[str] = Counter()
    for result in results:
        failure_stages.update(result.get("failure_stages", {}))
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "base_url": args.base_url,
        "run_dir": str(run_dir),
        "execute_tools": args.execute_tools,
        "repeat_semantic": args.repeat_semantic,
        "summary": {
            "total": len(results),
            "passed": passed,
            "failed": failed,
            "blocked": blocked,
        },
        "case_summary": {
            "total": case_total,
            "passed": case_passed,
            "failed": case_total - case_passed,
            "pass_rate": round(case_passed / case_total, 6) if case_total else None,
        },
        "usage": aggregate_usage(result.get("usage") for result in results),
        "failure_stages": dict(sorted(failure_stages.items())),
        "total_suite_duration_seconds": round(
            sum(result.get("duration_seconds", 0.0) for result in results),
            3,
        ),
        "suites": results,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.suites:
        parser.error("--suites must contain at least one suite")
    if args.repeat_semantic < 1:
        parser.error("--repeat_semantic must be greater than or equal to 1")
    if args.execute_tools and "workflows" not in args.suites:
        parser.error("--execute_tools requires the workflows suite")
    live_suites = {"routing", "planning", "workflows"}.intersection(args.suites)
    if live_suites and not args.api_key:
        parser.error(
            "--api_key or ENZYWIZARD_API_KEY is required for live model suites"
        )

    try:
        routing_case_count = load_case_count(ROUTING_CASES)
        planning_case_count = load_case_count(PLANNING_CASES)
        workflow_case_count = load_case_count(WORKFLOW_CASES)
        registered_count, covered_count, missing_tools = (
            validate_workflow_tool_coverage()
        )
    except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        parser.error(f"test preflight failed: {exc}")
    if missing_tools:
        parser.error(
            "real workflow cases do not cover registered tools: "
            f"{missing_tools}"
        )

    run_dir = create_run_directory(Path(args.workdir))
    results_path = run_dir / "results.json"
    commands = build_suite_commands(args, run_dir)
    environment = os.environ.copy()
    if args.api_key:
        environment["ENZYWIZARD_API_KEY"] = str(args.api_key).strip()
    source_path = str((PROJECT_ROOT / "src").resolve())
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_path}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else source_path
    )

    print("[测试项目] EnzyWizard-Agent 全面测试", flush=True)
    print(f"[模型] {args.model}", flush=True)
    print(
        f"[推理强度] {args.reasoning_effort or '模型默认值'}",
        flush=True,
    )
    print(f"[运行目录] {run_dir}", flush=True)
    print(f"[测试套件数] {len(commands)}", flush=True)
    print(
        "[测试规模] "
        f"路由语义={routing_case_count} x {args.repeat_semantic}，"
        f"Planning={planning_case_count} x {args.repeat_semantic}，"
        f"工作流={workflow_case_count}",
        flush=True,
    )
    print(
        f"[工具覆盖] {covered_count}/{registered_count} 个注册命令",
        flush=True,
    )
    print(f"[真实工具] {'执行' if args.execute_tools else '不执行'}", flush=True)
    print("[失败策略] 记录当前失败并继续后续测试", flush=True)

    results: list[dict[str, Any]] = []
    try:
        for name, command in commands:
            blockers = (
                analysis_fixture_blockers(run_dir / "workflows")
                if name == "analysis_real_workflow_regression"
                else []
            )
            if blockers:
                log_path = run_dir / "suite_logs" / f"{name}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                reason = (
                    "Skipped because the Analysis regression's actual fixture "
                    f"dependencies are incomplete: {'; '.join(blockers)}"
                )
                log_path.write_text(reason + "\n", encoding="utf-8")
                result = {
                    "suite": name,
                    "status": "blocked",
                    "return_code": None,
                    "duration_seconds": 0.0,
                    "log_file": str(log_path),
                    "launch_error": reason,
                }
                print(
                    "\n[套件跳过] 真实报告分析所需的具体工作流或报告不完整："
                    + "; ".join(blockers),
                    flush=True,
                )
            else:
                result = run_command(name, command, run_dir, environment)
            result = attach_suite_metrics(result, run_dir)
            results.append(result)
            save_results(results_path, args, run_dir, results)
            if result["status"] != "passed" and args.stop_on_suite_failure:
                break
    except KeyboardInterrupt:
        save_results(results_path, args, run_dir, results)
        print(f"[已保存结果] {results_path}", flush=True)
        return 130

    passed = sum(result["status"] == "passed" for result in results)
    failed = sum(result["status"] == "failed" for result in results)
    blocked = sum(result["status"] == "blocked" for result in results)
    case_summaries = [
        result["case_summary"]
        for result in results
        if result.get("case_summary") is not None
    ]
    case_total = sum(item["total"] for item in case_summaries)
    case_passed = sum(item["passed"] for item in case_summaries)
    usage = aggregate_usage(result.get("usage") for result in results)
    print(
        f"\n[全部完成] 套件={len(results)}，通过={passed}，失败={failed}，"
        f"阻塞={blocked}",
        flush=True,
    )
    if case_total:
        print(
            f"[案例汇总] 总数={case_total}，通过={case_passed}，"
            f"失败={case_total - case_passed}，"
            f"通过率={case_passed / case_total:.1%}",
            flush=True,
        )
    if usage["llm_call_count"]:
        print(
            "[Token 汇总] "
            f"LLM调用={usage['llm_call_count']}，"
            f"输入={usage['input_tokens']}，输出={usage['output_tokens']}，"
            f"推理={usage['reasoning_tokens']}",
            flush=True,
        )
    print(f"[完整结果] {results_path}", flush=True)
    return 1 if failed or blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
