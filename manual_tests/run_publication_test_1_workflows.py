"""Publication Test 1: scientific-goal translation and dynamic workflows."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import secrets
import shutil
import time
from typing import Any

from enzywizard_agent.services.request_routing_service import (
    plan_history_query,
    route_user_request,
)
from enzywizard_agent.nodes.planning import planning_node
from enzywizard_agent.nodes.tool_calling import tool_calling_node
from enzywizard_agent.data_models.tool_execution import (
    ToolCallExecution,
    ToolExecution,
)
from enzywizard_agent.services.user_confirmation_service import (
    record_user_confirmation,
)
from enzywizard_agent.utils.tool_plan_utils import load_enzywizard_tool_specs

from evaluation_utils import runtime_usage_payload
from publication_benchmark_utils import (
    BENCHMARK_ROOT,
    DEFAULT_BASE_URL,
    DEFAULT_REPEAT,
    TEST_1_CASE_COUNT,
    PlanAlternative,
    PROJECT_ROOT,
    WorkflowBenchmarkCase,
    aggregate_result_usage,
    atomic_write_json,
    benchmark_metadata,
    check_benchmark_model_contract,
    create_run_dir,
    load_case_file,
    mean,
    metric_statistics,
    metric_statistics_by_subtype,
    select_profile_cases,
    summarize_repeated_results,
    write_flat_csv,
)
from run_full_workflow_tests import build_initial_state, run_case as run_workflow_case
from run_request_routing_batch import build_state as build_history_state


CASES_PATH = BENCHMARK_ROOT / "test_1_workflow_cases.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api_key", default=os.environ.get("ENZYWIZARD_API_KEY"))
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reasoning_effort", default=None)
    parser.add_argument(
        "--ontology_mode", choices=("off", "retrieve", "control"), default="retrieve"
    )
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT)
    parser.add_argument("--case_ids", nargs="*")
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument("--sample_per_subtype", type=int, default=4)
    parser.add_argument(
        "--quick_case_count",
        type=int,
        help="Exact total number of subtype-balanced cases in quick mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Sampling seed. Omit it for a new random quick sample each run.",
    )
    parser.add_argument("--workdir", default="~/enzywizard-publication-test-1")
    parser.add_argument("--output")
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument(
        "--execute_representative",
        action="store_true",
        help="Execute only cases marked as representative; intended for debugging.",
    )
    execution.add_argument(
        "--execute_all",
        action="store_true",
        help=(
            "Execute every case whose expected terminal state includes a tool "
            "workflow; use this mode for the formal end-to-end benchmark."
        ),
    )
    parser.add_argument("--show_plans", action="store_true")
    parser.add_argument("--stop_on_failure", action="store_true")
    return parser


def _order_is_valid(actual: list[str], constraints: list[tuple[str, str]]) -> bool:
    for before, after in constraints:
        if before not in actual or after not in actual:
            return False
        if actual.index(before) >= actual.index(after):
            return False
    return True


def _parameters_are_valid(
    calls: list[dict[str, Any]], alternative: PlanAlternative
) -> bool:
    for expected in alternative.parameter_expectations:
        matches = [
            call
            for call in calls
            if call.get("tool_name") == expected.tool_name
            and (expected.side is None or call.get("side") == expected.side)
        ]
        if not matches:
            return False
        values = [call.get("arguments", {}).get(expected.parameter_name) for call in matches]
        if not any(value in expected.accepted_values for value in values):
            return False
    return True


def _plan_contract_is_valid(calls: list[dict[str, Any]]) -> bool:
    """Check ordering and dependency references for every generated call."""

    if not calls:
        return False
    expected_steps = list(range(1, len(calls) + 1))
    if [int(call.get("step_index", 0)) for call in calls] != expected_steps:
        return False
    available_steps = set(expected_steps)
    calls_by_step = {int(call["step_index"]): call for call in calls}
    specs = load_enzywizard_tool_specs()
    for call in calls:
        step_index = int(call.get("step_index", 0))
        if not isinstance(call.get("arguments"), dict):
            return False
        spec = specs.get(str(call.get("tool_name")))
        if spec is None or call.get("command_name") != spec.get("command_name"):
            return False
        required_arguments = {
            str(item.get("name"))
            for item in spec.get("required_arguments", [])
            if isinstance(item, dict) and item.get("name")
        }
        if not required_arguments.issubset(call["arguments"]):
            return False
        for value in call.get("arguments", {}).values():
            if not isinstance(value, dict) or "source_step_index" not in value:
                continue
            source_step = int(value["source_step_index"])
            if source_step not in available_steps or source_step >= step_index:
                return False
            produced_types = {
                str(item.get("artifact_type"))
                for item in calls_by_step[source_step].get("output_artifacts") or []
            }
            if str(value.get("artifact_type")) not in produced_types:
                return False
        for rule in call.get("staging_rules") or []:
            source_step = rule.get("source_step_index")
            if source_step is not None and (
                int(source_step) not in available_steps or int(source_step) >= step_index
            ):
                return False
    return True


def _format_plan_alternatives(alternatives: list[PlanAlternative]) -> str:
    formatted: list[str] = []
    for alternative in alternatives:
        tools = " -> ".join(alternative.expected_tools)
        details: list[str] = []
        if alternative.forbidden_tools:
            details.append(f"exclude={','.join(alternative.forbidden_tools)}")
        if alternative.parameter_expectations:
            parameters = ",".join(
                f"{item.tool_name}.{item.parameter_name}={item.accepted_values}"
                for item in alternative.parameter_expectations
            )
            details.append(f"parameters={parameters}")
        suffix = f" ({'; '.join(details)})" if details else ""
        formatted.append(f"{alternative.name}: {tools}{suffix}")
    return " | ".join(formatted) if formatted else "none"


def _print_case(case: WorkflowBenchmarkCase, repetition: int, repeat: int) -> None:
    print(
        f"\n[{case.case_id}] repetition {repetition}/{repeat} "
        f"subtype={case.subtype}",
        flush=True,
    )
    print(f"  Description: {case.description}", flush=True)
    print(f"  User request: {case.message}", flush=True)
    if case.inputs:
        inputs = ", ".join(f"{key}={value}" for key, value in case.inputs.items())
        print(f"  Inputs: {inputs}", flush=True)
    else:
        print("  Inputs: none", flush=True)

    if case.subtype == "workflow":
        for index, action in enumerate(case.confirmation_actions, start=1):
            print(
                f"  Expected step {index}: decision={action.decision}; "
                f"plans={_format_plan_alternatives(action.acceptable_plans)}",
                flush=True,
            )
            if action.revision_request:
                print(f"    Revision request: {action.revision_request}", flush=True)
        if case.execute_representative:
            print("  Representative execution: enabled when requested", flush=True)
        return

    expected = case.history_expectation
    if expected is None:
        return
    print(
        "  Expected history: "
        f"request_type={expected.request_type}; "
        f"actions={expected.actions}; operations={expected.required_operations}; "
        f"turns={expected.resolved_turn_ids}; "
        f"artifacts={expected.required_artifact_types}; "
        f"reuse={expected.reuse_policy}; recomputation={expected.recomputation}",
        flush=True,
    )
    if expected.acceptable_plans:
        print(
            f"  Expected new workflow: "
            f"{_format_plan_alternatives(expected.acceptable_plans)}",
            flush=True,
        )


def score_plan(
    snapshot: dict[str, Any] | None, alternatives: list[PlanAlternative]
) -> dict[str, Any]:
    if snapshot is None:
        return {
            "tool_selection_accuracy": 0.0,
            "parameter_correctness_validity": 0.0,
            "workflow_order_validity": 0.0,
            "plan_correct": False,
            "matched_alternative": None,
        }
    tools = list(snapshot.get("selected_tools") or [])
    calls = list(snapshot.get("calls") or [])
    sequence = [str(item.get("tool_name")) for item in calls]
    candidates: list[dict[str, Any]] = []
    for alternative in alternatives:
        forbidden_absent = not (set(tools) & set(alternative.forbidden_tools))
        tool_selection_correct = bool(
            set(tools) == set(alternative.expected_tools) and forbidden_absent
        )
        order_correct = bool(
            _plan_contract_is_valid(calls)
            and _order_is_valid(sequence, alternative.order_constraints)
        )
        parameters_correct = bool(
            _plan_contract_is_valid(calls)
            and _parameters_are_valid(calls, alternative)
        )
        candidates.append(
            {
                "tool_selection_accuracy": float(tool_selection_correct),
                "parameter_correctness_validity": float(parameters_correct),
                "workflow_order_validity": float(order_correct),
                "plan_correct": bool(
                    tool_selection_correct and order_correct and parameters_correct
                ),
                "matched_alternative": alternative.name,
            }
        )
    return max(
        candidates,
        key=lambda item: (
            bool(item["plan_correct"]),
            float(item["tool_selection_accuracy"]),
            float(item["workflow_order_validity"]),
        ),
    )


def _workflow_payload(case: WorkflowBenchmarkCase) -> dict[str, Any]:
    def legacy_expectation(action: Any) -> dict[str, Any]:
        alternative = action.acceptable_plans[0]
        return {
            "required_tools": alternative.expected_tools,
            "allowed_tools": alternative.expected_tools,
            "forbidden_tools": alternative.forbidden_tools,
            "order_constraints": alternative.order_constraints,
            "parameter_values": [
                {
                    "tool_name": item.tool_name,
                    "side": item.side,
                    "parameter_name": item.parameter_name,
                    "value": item.accepted_values[0],
                }
                for item in alternative.parameter_expectations
                if len(item.accepted_values) == 1
            ],
        }

    return {
        "case_id": case.case_id,
        "description": case.description,
        "tags": case.tags,
        "message": case.message,
        "inputs": case.inputs,
        "confirmation_actions": [
            {
                "decision": action.decision,
                "revision_request": action.revision_request,
                "expected_plan": legacy_expectation(action),
            }
            for action in case.confirmation_actions
        ],
        "expected_execution_status": "succeeded",
    }


def _execution_mode(args: argparse.Namespace) -> str:
    if args.execute_all:
        return "execute_all"
    if args.execute_representative:
        return "execute_representative"
    return "plan_only"


def _should_execute_workflow(
    case: WorkflowBenchmarkCase, args: argparse.Namespace
) -> bool:
    return bool(
        args.execute_all
        or (args.execute_representative and case.execute_representative)
    )


def _required_artifact_completeness(
    plan_snapshot: dict[str, Any] | None,
    execution: dict[str, Any] | None,
) -> float | None:
    """Return the fraction of declared required outputs registered after execution."""

    if plan_snapshot is None or execution is None:
        return None
    expected: set[tuple[int, str]] = set()
    for call in plan_snapshot.get("calls") or []:
        step_index = int(call.get("step_index", 0))
        for output in call.get("output_artifacts") or []:
            if output.get("required"):
                expected.add((step_index, str(output.get("artifact_type"))))
    actual = {
        (int(call.get("step_index", 0)), str(artifact.get("artifact_type")))
        for call in execution.get("call_executions") or []
        for artifact in call.get("artifacts") or []
    }
    if not expected:
        return 1.0
    return round(len(expected & actual) / len(expected), 6)


def _execution_matches_plan(
    plan_snapshot: dict[str, Any] | None,
    execution: dict[str, Any] | None,
) -> bool | None:
    if plan_snapshot is None or execution is None:
        return None
    expected = [
        (int(call.get("step_index", 0)), call.get("tool_name"), call.get("side"))
        for call in plan_snapshot.get("calls") or []
    ]
    actual = [
        (int(call.get("step_index", 0)), call.get("tool_name"), call.get("side"))
        for call in execution.get("call_executions") or []
    ]
    return expected == actual


def _materialize_executable_history_artifacts(state: dict[str, Any]) -> None:
    """Replace routing-only placeholders with valid frozen files for execution."""

    source_by_type = {
        "cleaned_structure_cif": (
            PROJECT_ROOT.parent
            / "enzywizard-dock/examples/input/cleaned_3GP6.cif"
        ),
        "clean_report_json": (
            PROJECT_ROOT.parent
            / "enzywizard-clean/examples/output/clean_report_3GP6.json"
        ),
        "substrate_report_json": (
            PROJECT_ROOT.parent
            / "enzywizard-substrate/examples/output/substrate_report_glucose_fructose.json"
        ),
        "docked_substrate_sdf": (
            PROJECT_ROOT.parent
            / "enzywizard-dock/examples/output/docked_glucose.sdf"
        ),
        "docked_complex_structure": (
            PROJECT_ROOT.parent
            / "enzywizard-dock/examples/output/docked_cleaned_3GP6_glucose_fructose.pdb"
        ),
        "dock_report_json": (
            PROJECT_ROOT.parent
            / "enzywizard-dock/examples/output/dock_report_cleaned_3GP6_glucose_fructose.json"
        ),
    }
    for artifact in state.get("artifacts", []):
        source = source_by_type.get(artifact.artifact_type)
        if source is None:
            continue
        destination = Path(artifact.path).with_name(source.name)
        if source.is_file():
            shutil.copyfile(source, destination)
            artifact.path = str(destination.resolve())

    turn_three = [
        artifact
        for artifact in state.get("artifacts", [])
        if artifact.turn_id == 3 and Path(artifact.path).is_file()
    ]
    clean_structure = next(
        artifact.path
        for artifact in turn_three
        if artifact.artifact_type == "cleaned_structure_cif"
    )
    substrate_artifacts = [
        artifact
        for artifact in turn_three
        if artifact.source_tool == "enzywizard-substrate"
    ]
    dock_artifacts = [
        artifact
        for artifact in turn_three
        if artifact.source_tool == "enzywizard-dock"
    ]
    audit_dir = Path(state["config"].workdir) / "synthetic_history" / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = audit_dir / "stdout.txt"
    stderr_path = audit_dir / "stderr.txt"
    stdout_path.write_text("synthetic successful history\n", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    state["tool_executions"] = [
        ToolExecution(
            turn_id=3,
            plan_id=1,
            execution_status="succeeded",
            call_executions=[
                ToolCallExecution(
                    turn_id=3,
                    plan_id=1,
                    step_index=1,
                    tool_name="enzywizard-substrate",
                    side="single",
                    execution_status="succeeded",
                    command=["enzywizard-substrate"],
                    resolved_arguments={"substrate_names": "glucose"},
                    exit_code=0,
                    duration_seconds=0.01,
                    stdout_file=str(stdout_path),
                    stderr_file=str(stderr_path),
                    log_files=[],
                    stdout_excerpt="synthetic successful history",
                    stderr_excerpt="",
                    log_excerpt="",
                    artifacts=substrate_artifacts,
                    error_message=None,
                ),
                ToolCallExecution(
                    turn_id=3,
                    plan_id=1,
                    step_index=2,
                    tool_name="enzywizard-dock",
                    side="single",
                    execution_status="succeeded",
                    command=["enzywizard-dock"],
                    resolved_arguments={
                        "input_path": clean_structure,
                        "substrate_names": "glucose",
                    },
                    exit_code=0,
                    duration_seconds=0.01,
                    stdout_file=str(stdout_path),
                    stderr_file=str(stderr_path),
                    log_files=[],
                    stdout_excerpt="synthetic successful history",
                    stderr_excerpt="",
                    log_excerpt="",
                    artifacts=dock_artifacts,
                    error_message=None,
                ),
            ],
            failed_step_index=None,
            error_messages=[],
            summary="Synthetic successful substrate and docking history.",
        )
    ]


def run_workflow(
    case: WorkflowBenchmarkCase, repetition: int, args: argparse.Namespace, run_dir: Path
) -> dict[str, Any]:
    execute_tools = _should_execute_workflow(case, args)
    raw = run_workflow_case(
        _workflow_payload(case),
        args.api_key,
        args.base_url,
        args.model,
        args.reasoning_effort,
        args.ontology_mode,
        "safe",
        1,
        run_dir / f"repeat_{repetition:03d}",
        execute_tools,
        args.show_plans,
        False,
    )
    scores = [
        score_plan(
            raw.get("plan_snapshots", [])[index]
            if index < len(raw.get("plan_snapshots", []))
            else None,
            action.acceptable_plans,
        )
        for index, action in enumerate(case.confirmation_actions)
    ]
    metrics = {
        key: float(bool(scores) and all(item[key] == 1.0 for item in scores))
        for key in (
            "tool_selection_accuracy",
            "parameter_correctness_validity",
            "workflow_order_validity",
        )
    }
    execution = raw.get("tool_execution")
    execution_status_succeeded = (
        execution.get("execution_status") == "succeeded" if execution else None
    )
    execution_matches_plan = _execution_matches_plan(
        raw.get("plan_snapshots", [])[-1]
        if raw.get("plan_snapshots")
        else None,
        execution,
    )
    artifact_completeness = _required_artifact_completeness(
        raw.get("plan_snapshots", [])[-1]
        if raw.get("plan_snapshots")
        else None,
        execution,
    )
    metrics["required_artifact_completeness"] = (
        artifact_completeness
        if artifact_completeness is not None
        else (0.0 if execute_tools else None)
    )
    plan_correct = bool(scores) and all(item["plan_correct"] for item in scores)
    metrics["end_to_end_task_success"] = (
        plan_correct
        and bool(execution_status_succeeded)
        and bool(execution_matches_plan)
        and metrics["required_artifact_completeness"] == 1.0
        if execute_tools
        else None
    )
    passed = plan_correct
    if execute_tools:
        passed = passed and bool(metrics["end_to_end_task_success"])
    return {
        "case_id": case.case_id,
        "repetition": repetition,
        "subtype": case.subtype,
        "fair_relevance": case.fair_relevance,
        "execution_required": True,
        "execution_requested": execute_tools,
        "execution_completed": execution is not None,
        "task_success": passed if execute_tools else None,
        "passed": passed,
        "metrics": metrics,
        "execution_checks": {
            "status_succeeded": execution_status_succeeded,
            "matches_approved_plan": execution_matches_plan,
        },
        "action_scores": scores,
        "failures": raw.get("failures", []),
        "usage": raw.get("usage"),
        "trace": raw.get("trace"),
        "raw_result": raw,
    }


def run_routing_case(
    case: WorkflowBenchmarkCase, repetition: int, args: argparse.Namespace, run_dir: Path
) -> dict[str, Any]:
    started = time.monotonic()
    case_dir = run_dir / f"repeat_{repetition:03d}" / case.case_id
    case_dir.mkdir(parents=True, exist_ok=False)
    if case.subtype == "history":
        state = build_history_state(
            {"message": case.message},
            args.api_key,
            args.base_url,
            args.model,
            args.reasoning_effort,
            args.ontology_mode,
            case_dir,
        )
        _materialize_executable_history_artifacts(state)
    else:
        synthetic = {
            "case_id": case.case_id,
            "message": case.message,
            "inputs": case.inputs,
            "confirmation_actions": [
                {
                    "decision": "approved",
                    "revision_request": None,
                    "expected_plan": {},
                }
            ],
        }
        state = build_initial_state(
            synthetic,
            args.api_key,
            args.base_url,
            args.model,
            args.reasoning_effort,
            args.ontology_mode,
            "safe",
            1,
            case_dir,
        )
    failures: list[str] = []
    routing = None
    history = None
    planning = None
    execution = None
    plan_score = None
    execution_required = bool(
        case.history_expectation is not None
        and case.history_expectation.requires_tool_execution
    )
    execution_requested = bool(args.execute_all and execution_required)
    history_contract_correct: bool | None = None
    try:
        routing = route_user_request(state)
        if routing.routing_status != "succeeded" or routing.output is None:
            failures.append(routing.error_message or "routing failed")
        else:
            expected = case.history_expectation
            assert expected is not None
            output = routing.output
            actions = [item.action_type for item in output.actions]
            if output.request_type != expected.request_type:
                failures.append("unexpected request_type")
            if set(actions) != set(expected.actions) or len(actions) != len(set(actions)):
                failures.append(f"unexpected actions: {actions}")
            if output.reuse_policy != expected.reuse_policy:
                failures.append("unexpected reuse_policy")
            if output.requires_tool_execution != expected.requires_tool_execution:
                failures.append("unexpected requires_tool_execution")
            if output.requests_recomputation != expected.recomputation:
                failures.append("unexpected recomputation flag")
            if output.requires_history_artifacts:
                state["current_request_routing"] = routing
                history = plan_history_query(state)
                if history.query_status != "succeeded" or history.query is None:
                    failures.append(history.error_message or history.summary)
                else:
                    query = history.query
                    if not set(expected.required_operations).issubset(query.operations):
                        failures.append("missing history operation")
                    if expected.resolved_turn_ids is not None and set(
                        history.resolved_turn_ids
                    ) != set(expected.resolved_turn_ids):
                        failures.append("incorrect historical turn selection")
                    selected_types = {item.artifact_type for item in history.selected_artifacts}
                    if not set(expected.required_artifact_types).issubset(selected_types):
                        failures.append("missing required historical artifact")
                    state["current_history_query"] = history
                    state["current_history_artifacts"] = list(
                        history.selected_artifacts
                    )
                    state.setdefault("history_query_results", []).append(history)

            history_contract_correct = not failures
            if execution_required and not failures:
                state["current_request_routing"] = routing
                state = planning_node(state)
                planning = state.get("current_planning_result")
                if planning is None or planning.tool_plan is None:
                    failures.append("history request did not produce a tool plan")
                else:
                    plan_snapshot = {
                        "selected_tools": list(
                            dict.fromkeys(
                                call.tool_name for call in planning.tool_plan.calls
                            )
                        ),
                        "calls": [
                            call.model_dump(mode="json")
                            for call in planning.tool_plan.calls
                        ],
                    }
                    plan_score = score_plan(
                        plan_snapshot,
                        expected.acceptable_plans,
                    )
                    if not plan_score["plan_correct"]:
                        failures.append(
                            "historical request produced an incorrect downstream plan"
                        )
                    if execution_requested and not failures:
                        state = record_user_confirmation(
                            state,
                            {
                                "turn_id": planning.turn_id,
                                "plan_id": planning.plan_id,
                                "decision": "approved",
                                "revision_request": None,
                            },
                        )
                        state = tool_calling_node(state)
                        execution_model = state.get("current_tool_execution")
                        execution = (
                            execution_model.model_dump(mode="json")
                            if execution_model is not None
                            else None
                        )
                        if (
                            execution is None
                            or execution.get("execution_status") != "succeeded"
                        ):
                            failures.append(
                                "historical downstream workflow execution failed"
                            )
                        elif not _execution_matches_plan(
                            plan_snapshot,
                            execution,
                        ):
                            failures.append(
                                "historical execution did not match the approved plan"
                            )
                        elif _required_artifact_completeness(
                            plan_snapshot,
                            execution,
                        ) != 1.0:
                            failures.append(
                                "historical downstream workflow missed required outputs"
                            )
    except Exception as exc:
        failures.append(f"{type(exc).__name__}: {exc}")
    metrics = {
        "tool_selection_accuracy": 0.0,
        "parameter_correctness_validity": 0.0,
        "workflow_order_validity": 0.0,
        "required_artifact_completeness": (
            _required_artifact_completeness(plan_snapshot, execution)
            if execution is not None and plan_score is not None
            else (0.0 if execution_requested else None)
        ),
        "end_to_end_task_success": (
            bool(
                plan_score
                and plan_score["plan_correct"]
                and history_contract_correct is True
                and execution is not None
                and execution.get("execution_status") == "succeeded"
                and _execution_matches_plan(plan_snapshot, execution) is True
                and _required_artifact_completeness(plan_snapshot, execution) == 1.0
            )
            if execution_requested
            else None
        ),
    }
    if plan_score is not None:
        metrics.update(
            {
                key: plan_score[key]
                for key in (
                    "tool_selection_accuracy",
                    "parameter_correctness_validity",
                    "workflow_order_validity",
                )
            }
        )
    task_success = None
    if execution_required and execution_requested:
        task_success = not failures
    elif not execution_required:
        task_success = not failures
    return {
        "case_id": case.case_id,
        "repetition": repetition,
        "subtype": case.subtype,
        "fair_relevance": case.fair_relevance,
        "execution_required": execution_required,
        "execution_requested": execution_requested,
        "execution_completed": execution is not None,
        "task_success": task_success,
        "passed": not failures,
        "metrics": metrics,
        "failures": failures,
        "duration_seconds": round(time.monotonic() - started, 3),
        "usage": runtime_usage_payload(state.get("runtime_session")),
        "routing_result": routing.model_dump(mode="json") if routing else None,
        "history_result": history.model_dump(mode="json") if history else None,
        "planning_result": planning.model_dump(mode="json") if planning else None,
        "tool_execution": execution,
    }


def build_summary(
    results: list[dict[str, Any]], execution_mode: str = "plan_only"
) -> dict[str, Any]:
    summary = summarize_repeated_results(results)
    summary["metrics"] = metric_statistics(results)
    summary["metrics_by_subtype"] = metric_statistics_by_subtype(results)
    required = [item for item in results if item.get("execution_required")]
    requested = [item for item in required if item.get("execution_requested")]
    attempted = [item for item in requested if item.get("execution_completed")]
    task_results = [
        bool(item["task_success"])
        for item in results
        if isinstance(item.get("task_success"), bool)
    ]
    by_case: dict[str, list[bool]] = {}
    for item in results:
        if isinstance(item.get("task_success"), bool):
            by_case.setdefault(str(item["case_id"]), []).append(
                bool(item["task_success"])
            )
    stable = sum(bool(values) and all(values) for values in by_case.values())
    execution_results = [
        bool((item.get("metrics") or {}).get("end_to_end_task_success"))
        for item in attempted
    ]
    summary.update(
        {
            "execution_mode": execution_mode,
            "execution_required_runs": len(required),
            "execution_requested_runs": len(requested),
            "execution_request_coverage_rate": (
                round(len(requested) / len(required), 6) if required else None
            ),
            "execution_attempted_runs": len(attempted),
            "execution_coverage_rate": (
                round(len(attempted) / len(required), 6) if required else None
            ),
            "successful_execution_runs": sum(execution_results),
            "execution_success_rate": mean(execution_results),
            "objective_task_evaluated_runs": len(task_results),
            "objective_task_success_rate": mean(task_results),
            "stable_objective_task_cases": stable,
            "stable_objective_task_rate": (
                round(stable / len(by_case), 6) if by_case else None
            ),
        }
    )
    return summary


def main() -> int:
    args = build_parser().parse_args()
    if not args.api_key:
        raise SystemExit("--api_key or ENZYWIZARD_API_KEY is required")
    if args.repeat < 1:
        raise SystemExit("--repeat must be positive")
    resolved_seed = args.seed if args.seed is not None else secrets.randbits(32)
    all_cases = load_case_file(
        CASES_PATH, WorkflowBenchmarkCase, TEST_1_CASE_COUNT
    )
    cases = select_profile_cases(
        all_cases,
        case_ids=args.case_ids,
        profile=args.profile,
        sample_per_subtype=args.sample_per_subtype,
        seed=resolved_seed,
        quick_case_count=args.quick_case_count,
    )
    run_dir = create_run_dir(args.workdir, "test_1")
    output = Path(args.output).expanduser().resolve() if args.output else run_dir / "results.json"
    try:
        capability, capability_usage = check_benchmark_model_contract(
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            ontology_mode=args.ontology_mode,
            workdir=run_dir / "_model_capability_check",
        )
    except Exception as exc:
        print(f"[model contract failed] {exc}", flush=True)
        return 2
    results: list[dict[str, Any]] = []
    execution_mode = _execution_mode(args)
    print(
        f"[Test 1] cases={len(cases)} repeat={args.repeat} model={args.model} "
        f"profile={args.profile} mode={execution_mode} seed={resolved_seed}"
    )
    for case in cases:
        for repetition in range(1, args.repeat + 1):
            _print_case(case, repetition, args.repeat)
            if case.subtype == "workflow":
                result = run_workflow(case, repetition, args, run_dir)
            else:
                result = run_routing_case(case, repetition, args, run_dir)
            results.append(result)
            payload = {
                "metadata": {
                    **benchmark_metadata(
                        test_id="test_1",
                        cases_file=CASES_PATH,
                        model=args.model,
                        base_url=args.base_url,
                        reasoning_effort=args.reasoning_effort,
                        ontology_mode=args.ontology_mode,
                        repeat=args.repeat,
                    ),
                    "execution_mode": execution_mode,
                    "profile": args.profile,
                    "sample_per_subtype": args.sample_per_subtype,
                    "quick_case_count": args.quick_case_count,
                    "seed": resolved_seed,
                    "selected_case_ids": [item.case_id for item in cases],
                },
                "summary": build_summary(results, execution_mode),
                "model_capability": capability,
                "usage": aggregate_result_usage(
                    [{"usage": capability_usage}, *results]
                ),
                "results": results,
            }
            atomic_write_json(output, payload)
            if args.stop_on_failure and not result["passed"]:
                write_flat_csv(output.with_suffix(".csv"), results)
                return 1
    write_flat_csv(
        output.with_suffix(".csv"),
        [
            {
                "case_id": item["case_id"],
                "repetition": item["repetition"],
                "subtype": item["subtype"],
                "passed": item["passed"],
                "execution_required": item["execution_required"],
                "execution_requested": item["execution_requested"],
                "execution_completed": item["execution_completed"],
                "task_success": item["task_success"],
                **(item.get("metrics") or {}),
            }
            for item in results
        ],
    )
    print(f"[complete] {output}")
    return 0 if all(item["passed"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
