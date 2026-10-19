"""Batch integration test for the complete Planning node using a real LLM API."""

from __future__ import annotations

import argparse
import io
import json
import os
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any

from enzywizard_agent.conf import AgentConfig
from enzywizard_agent.data_models.agent_inputs import AgentInputs
from enzywizard_agent.data_models.artifact import CleanedFiles
from enzywizard_agent.nodes.planning import planning_node
from enzywizard_agent.utils.IO_utils import create_session_id, normalize_path

from evaluation_utils import aggregate_usage, runtime_usage_payload


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES_PATH = Path(__file__).with_name("planning_cases.json")


def build_parser() -> argparse.ArgumentParser:
    """Create command-line arguments for batch Planning tests."""

    parser = argparse.ArgumentParser(
        description="Run complete EnzyWizard-Agent Planning test cases."
    )
    parser.add_argument(
        "--api_key",
        default=os.environ.get("ENZYWIZARD_API_KEY"),
        help="LLM API key. Defaults to ENZYWIZARD_API_KEY.",
    )
    parser.add_argument("--base_url", required=True)
    parser.add_argument("--model", default="qwen3.7-plus")
    parser.add_argument(
        "--reasoning_effort",
        default=None,
        help="Optional OpenAI-compatible reasoning effort, such as 'none'.",
    )
    parser.add_argument("--workdir", default="/tmp/enzywizard-agent-planning-batch")
    parser.add_argument("--cases", default=str(DEFAULT_CASES_PATH))
    parser.add_argument(
        "--output",
        default="/tmp/enzywizard-agent-planning-batch-results.json",
    )
    parser.add_argument(
        "--case_ids",
        nargs="*",
        help="Run only the listed case IDs. By default all cases are run.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat every selected case to observe LLM stability.",
    )
    parser.add_argument(
        "--stop_on_failure",
        action="store_true",
        help="Stop after the first failed case.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the complete PlanningResult JSON for every case.",
    )
    return parser


def resolve_case_path(value: str | None) -> str | None:
    """Resolve a case input path relative to the project root."""

    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return normalize_path(str(path))


def load_cases(cases_path: Path) -> list[dict[str, Any]]:
    """Load and minimally validate the JSON test case list."""

    payload = json.loads(cases_path.read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(cases, list) or not cases:
        raise ValueError("The case file must contain a non-empty 'cases' list.")

    case_ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Each Planning test case must be a JSON object.")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("Each Planning test case requires a non-empty case_id.")
        if case_id in case_ids:
            raise ValueError(f"Duplicate Planning test case ID: {case_id}.")
        if not isinstance(case.get("message"), str) or not case["message"].strip():
            raise ValueError(f"Case {case_id} requires a non-empty message.")
        case_ids.add(case_id)
    return cases


def build_case_state(
    case: dict[str, Any],
    api_key: str,
    base_url: str,
    model: str,
    reasoning_effort: str | None,
    workdir: Path,
) -> dict[str, Any]:
    """Build the first-turn Agent state for one Planning test case."""

    inputs = case.get("inputs", {})
    cleaned_values = {
        field_name: None for field_name in CleanedFiles.model_fields
    }
    for field_name, value in case.get("cleaned_files", {}).items():
        if field_name not in cleaned_values:
            raise ValueError(
                f"Case {case['case_id']} contains unknown CleanedFiles field "
                f"{field_name!r}."
            )
        cleaned_values[field_name] = (
            resolve_case_path(value)
            if field_name.endswith(("_file", "_dir"))
            else value
        )

    return {
        "config": AgentConfig(
            api_key=api_key,
            base_url=base_url,
            model=model,
            reasoning_effort=reasoning_effort,
            workdir=workdir.resolve(),
            max_turns=3,
        ),
        "session_id": create_session_id(),
        "current_turn_id": 1,
        "current_plan_id": 0,
        "current_user_message": case["message"].strip(),
        "agent_inputs": AgentInputs(
            structure_file=resolve_case_path(inputs.get("structure_file")),
            msa_file=resolve_case_path(inputs.get("msa_file")),
            mut_structure_file=resolve_case_path(inputs.get("mut_structure_file")),
            mut_msa_file=resolve_case_path(inputs.get("mut_msa_file")),
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
        "user_messages": [],
        "user_message_summary": case.get("user_message_summary", ""),
        "planning_results": [],
        "planning_summary": "",
        "user_confirmations": [],
        "user_confirmation_summary": "",
        "cleaned_files": CleanedFiles(**cleaned_values),
        "current_plan_artifacts": [],
        "artifacts": [],
        "artifact_summary": "",
    }


def collect_plan_details(planning_result: Any) -> tuple[list[str], list[str]]:
    """Return selected tool names and ordered tool-call labels."""

    selected_tools = (
        list(planning_result.tool_selection.selected_tools)
        if planning_result.tool_selection is not None
        else []
    )
    call_labels = (
        [f"{call.tool_name}:{call.side}" for call in planning_result.tool_plan.calls]
        if planning_result.tool_plan is not None
        else []
    )
    return selected_tools, call_labels


def check_expected_parameters(
    planning_result: Any,
    parameter_expectations: list[dict[str, Any]],
) -> list[str]:
    """Check final ToolCall argument values requested by a case."""

    failures: list[str] = []
    calls = planning_result.tool_plan.calls if planning_result.tool_plan else []
    for expectation in parameter_expectations:
        matches = [
            call
            for call in calls
            if call.tool_name == expectation["tool_name"]
            and (
                expectation.get("side") is None
                or call.side == expectation["side"]
            )
        ]
        if len(matches) != 1:
            failures.append(
                "Expected exactly one call for "
                f"{expectation['tool_name']}:{expectation.get('side', '*')}, "
                f"found {len(matches)}."
            )
            continue
        parameter_name = expectation["parameter_name"]
        actual_value = matches[0].arguments.get(parameter_name)
        if actual_value != expectation["value"]:
            failures.append(
                f"Expected {expectation['tool_name']}.{parameter_name}="
                f"{expectation['value']!r}, got {actual_value!r}."
            )
    return failures


def evaluate_case(case: dict[str, Any], final_state: dict[str, Any]) -> list[str]:
    """Compare one Planning result with its declarative expectations."""

    expected = case.get("expected", {})
    planning_result = final_state["current_planning_result"]
    selected_tools, call_labels = collect_plan_details(planning_result)
    failures: list[str] = []

    scalar_expectations = {
        "route": final_state.get("route"),
        "purpose_type": planning_result.purpose_type,
        "planning_flag": planning_result.planning_flag,
    }
    for field_name, actual_value in scalar_expectations.items():
        if field_name in expected and actual_value != expected[field_name]:
            failures.append(
                f"Expected {field_name}={expected[field_name]!r}, got {actual_value!r}."
            )

    required_tools = set(expected.get("required_tools", []))
    missing_tools = required_tools - set(selected_tools)
    if missing_tools:
        failures.append(f"Missing required tools: {sorted(missing_tools)}.")

    forbidden_tools = set(expected.get("forbidden_tools", []))
    present_forbidden_tools = forbidden_tools & set(selected_tools)
    if present_forbidden_tools:
        failures.append(
            f"Unexpected forbidden tools: {sorted(present_forbidden_tools)}."
        )

    if "exact_selected_tools" in expected:
        expected_tools = expected["exact_selected_tools"]
        if selected_tools != expected_tools:
            failures.append(
                f"Expected selected_tools={expected_tools!r}, got {selected_tools!r}."
            )

    if "exact_call_sequence" in expected:
        expected_calls = expected["exact_call_sequence"]
        if call_labels != expected_calls:
            failures.append(
                f"Expected call sequence={expected_calls!r}, got {call_labels!r}."
            )

    error_text = " ".join(planning_result.error_messages)
    if "error_contains" in expected and expected["error_contains"] not in error_text:
        failures.append(
            f"Expected error text containing {expected['error_contains']!r}, "
            f"got {error_text!r}."
        )

    failures.extend(
        check_expected_parameters(
            planning_result,
            expected.get("parameter_values", []),
        )
    )
    return failures


def run_case(
    case: dict[str, Any],
    repetition: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Execute and evaluate one complete Planning test case."""

    case_id = case["case_id"]
    case_workdir = Path(args.workdir).expanduser() / case_id / f"run_{repetition:03d}"
    state = build_case_state(
        case,
        args.api_key,
        args.base_url,
        args.model,
        args.reasoning_effort,
        case_workdir,
    )
    captured_log = io.StringIO()
    try:
        with redirect_stdout(captured_log):
            final_state = planning_node(state)
        failures = evaluate_case(case, final_state)
        planning_result = final_state["current_planning_result"]
        selected_tools, call_labels = collect_plan_details(planning_result)
        return {
            "case_id": case_id,
            "repetition": repetition,
            "passed": not failures,
            "failures": failures,
            "route": final_state.get("route"),
            "purpose_type": planning_result.purpose_type,
            "planning_flag": planning_result.planning_flag,
            "selected_tools": selected_tools,
            "call_sequence": call_labels,
            "planning_result": planning_result.model_dump(mode="json"),
            "planning_log": captured_log.getvalue().splitlines(),
            "usage": runtime_usage_payload(final_state.get("runtime_session")),
        }
    except Exception as error:
        return {
            "case_id": case_id,
            "repetition": repetition,
            "passed": False,
            "failures": [f"Unhandled test exception: {error}"],
            "planning_log": captured_log.getvalue().splitlines(),
            "usage": runtime_usage_payload(state.get("runtime_session")),
        }


def main() -> None:
    """Run selected cases, print a concise summary, and save full JSON results."""

    parser = build_parser()
    args = parser.parse_args()
    if not args.api_key:
        parser.error("--api_key or ENZYWIZARD_API_KEY is required")
    if args.repeat < 1:
        parser.error("--repeat must be greater than or equal to 1")

    cases_path = Path(args.cases).expanduser().resolve()
    cases = load_cases(cases_path)
    if args.case_ids:
        requested_ids = set(args.case_ids)
        known_ids = {case["case_id"] for case in cases}
        unknown_ids = requested_ids - known_ids
        if unknown_ids:
            parser.error(f"Unknown case IDs: {', '.join(sorted(unknown_ids))}")
        cases = [case for case in cases if case["case_id"] in requested_ids]

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def save_current_results(results: list[dict[str, Any]]) -> None:
        passed = sum(result["passed"] for result in results)
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "cases_file": str(cases_path),
            "summary": {
                "total": len(results),
                "passed": passed,
                "failed": len(results) - passed,
            },
            "usage": aggregate_usage(result.get("usage") for result in results),
            "results": results,
        }
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print("[运行模式] Planning 批量语义测试", flush=True)
    print(f"[模型] {args.model}", flush=True)
    print(f"[推理强度] {args.reasoning_effort or '模型默认值'}", flush=True)
    print(f"[案例数量] {len(cases)} x {args.repeat}", flush=True)
    print(f"[运行目录] {Path(args.workdir).expanduser().resolve()}", flush=True)
    print(f"[结果文件] {output_path}", flush=True)
    print("[失败策略] 单个案例失败后记录错误并继续", flush=True)

    results: list[dict[str, Any]] = []
    should_stop = False
    total = len(cases) * args.repeat
    for repetition in range(1, args.repeat + 1):
        for case in cases:
            index = len(results) + 1
            print(
                f"\n[案例 {index}/{total}] 开始：{case['case_id']} "
                f"（第 {repetition} 次）",
                flush=True,
            )
            started_at = time.monotonic()
            result = run_case(case, repetition, args)
            duration = time.monotonic() - started_at
            result["duration_seconds"] = round(duration, 3)
            results.append(result)
            save_current_results(results)
            status = "通过" if result["passed"] else "失败"
            print(f"[案例结果] {status}，用时 {duration:.1f} 秒", flush=True)
            if "planning_result" in result:
                print(
                    f"  [规划] 类型={result['purpose_type']}，"
                    f"状态={result['planning_flag']}，路由={result['route']}",
                    flush=True,
                )
                print(
                    "  [调用顺序] " + " -> ".join(result["call_sequence"]),
                    flush=True,
                )
                error_messages = result["planning_result"].get("error_messages", [])
                if error_messages:
                    print(
                        "  [规划错误] " + "; ".join(error_messages),
                        flush=True,
                    )
            for failure in result["failures"]:
                print(f"  [失败原因] {failure}", flush=True)
            if args.verbose and "planning_result" in result:
                print("  [完整规划结果]")
                print(
                    json.dumps(
                        result["planning_result"],
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            failed = sum(not item["passed"] for item in results)
            print(
                f"[当前进度] 已完成 {len(results)}/{total}，失败 {failed}",
                flush=True,
            )
            if not result["passed"] and args.stop_on_failure:
                should_stop = True
                break
        if should_stop:
            break

    passed_count = sum(result["passed"] for result in results)
    failed_count = len(results) - passed_count
    save_current_results(results)
    print(
        f"\n[全部完成] 总数={len(results)}，通过={passed_count}，"
        f"失败={failed_count}"
    )
    print(f"[完整结果] {output_path}")

    if failed_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
