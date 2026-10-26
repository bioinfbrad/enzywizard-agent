"""Publication Test 3: failure diagnosis, bounded recovery, and traceability."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import secrets
import time
from typing import Any

from enzywizard_agent.conf import AgentConfig
from enzywizard_agent.data_models.planning import PlanningResult, ToolCall, ToolPlan
from enzywizard_agent.data_models.runtime import RuntimeSession
from enzywizard_agent.data_models.tool_execution import ToolCallExecution
from enzywizard_agent.data_models.user_confirmation import UserConfirmation
from enzywizard_agent.services import tool_calling_service
from enzywizard_agent.services.analysis_service import analyze_agent_state
from enzywizard_agent.services.request_routing_service import route_user_request
from enzywizard_agent.services.runtime_service import utc_now
from enzywizard_agent.services.tool_recovery_service import (
    diagnose_tool_failure,
)
from enzywizard_agent.nodes.planning import planning_node

from evaluation_utils import runtime_usage_payload
from publication_benchmark_utils import (
    BENCHMARK_ROOT,
    DEFAULT_BASE_URL,
    DEFAULT_REPEAT,
    TEST_3_CASE_COUNT,
    RecoveryBenchmarkCase,
    aggregate_result_usage,
    atomic_write_json,
    benchmark_metadata,
    check_benchmark_model_contract,
    create_run_dir,
    load_case_file,
    metric_statistics,
    metric_statistics_by_subtype,
    select_profile_cases,
    summarize_repeated_results,
    write_flat_csv,
)
from run_full_workflow_tests import build_initial_state


CASES_PATH = BENCHMARK_ROOT / "test_3_recovery_cases.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api_key", default=os.environ.get("ENZYWIZARD_API_KEY"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL)
    parser.add_argument("--reasoning_effort", default=None)
    parser.add_argument(
        "--ontology_mode", choices=("off", "retrieve", "control"), default="retrieve"
    )
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT)
    parser.add_argument("--case_ids", nargs="*")
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument("--sample_per_subtype", type=int, default=2)
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
    parser.add_argument("--workdir", default="~/enzywizard-publication-test-3")
    parser.add_argument("--output")
    parser.add_argument("--stop_on_failure", action="store_true")
    return parser


def _call_result(
    case: RecoveryBenchmarkCase,
    case_dir: Path,
    *,
    attempt_index: int,
    succeeded: bool,
) -> ToolCallExecution:
    return ToolCallExecution(
        turn_id=1,
        plan_id=1,
        step_index=1,
        tool_name=case.tool_name,
        side="single",
        attempt_index=attempt_index,
        execution_status="succeeded" if succeeded else "failed",
        command=[case.tool_name] if case.command_present or succeeded else [],
        resolved_arguments={},
        exit_code=0 if succeeded else 1,
        duration_seconds=0.01,
        stdout_file=str(case_dir / f"stdout-{attempt_index}.txt"),
        stderr_file=str(case_dir / f"stderr-{attempt_index}.txt"),
        log_files=[],
        stdout_excerpt="",
        stderr_excerpt="" if succeeded else case.error_message,
        log_excerpt="",
        artifacts=[],
        error_message=None if succeeded else case.error_message,
    )


def _state(
    case: RecoveryBenchmarkCase,
    case_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    call = ToolCall(
        turn_id=1,
        plan_id=1,
        step_index=1,
        tool_name=case.tool_name,
        command_name=case.tool_name,
        side="single",
        tool_execution_dir=str(case_dir / "tool"),
        arguments={},
        output_artifacts=[],
        staging_rules=[],
        call_reason="Controlled publication benchmark fault injection.",
    )
    plan = ToolPlan(
        turn_id=1,
        plan_id=1,
        purpose_type="single_protein_analysis",
        summary="Controlled recovery test.",
        plan_reason="Exercise deterministic recovery policy.",
        calls=[call],
    )
    planning = PlanningResult(
        turn_id=1,
        plan_id=1,
        planning_route="tool_plan_generated",
        planning_flag="tool_plan_generated",
        needs_tool=True,
        summary=plan.summary,
        purpose_type="single_protein_analysis",
        purpose_reason="Controlled benchmark.",
        tool_selection=None,
        parameter_extraction=None,
        tool_plan=plan,
        error_messages=[],
    )
    return {
        "config": AgentConfig(
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            ontology_mode=args.ontology_mode,
            tool_recovery_mode="safe",
            failure_diagnosis_mode="hybrid",
            max_tool_recovery_attempts=1,
            check_model_capabilities=False,
            workdir=case_dir,
        ),
        "session_id": f"recovery-{case.case_id}",
        "current_turn_id": 1,
        "current_plan_id": 1,
        "current_planning_result": planning,
        "current_user_confirmation": UserConfirmation(
            turn_id=1, plan_id=1, decision="approved", revision_request=None
        ),
        "runtime_session": RuntimeSession(
            started_at=utc_now(), started_monotonic=time.monotonic()
        ),
        "artifacts": [],
        "current_plan_artifacts": [],
        "current_user_message": case.message or (
            f"Run the planned {case.tool_name} workflow."
        ),
    }


def _trace_fraction(checks: list[bool]) -> float:
    return round(sum(checks) / len(checks), 6) if checks else 0.0


def _run_preflight_case(
    case: RecoveryBenchmarkCase,
    repetition: int,
    args: argparse.Namespace,
    case_dir: Path,
) -> dict[str, Any]:
    synthetic = {
        "case_id": case.case_id,
        "message": case.message,
        "inputs": case.inputs,
        "confirmation_actions": [
            {"decision": "approved", "revision_request": None, "expected_plan": {}}
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
    routing = route_user_request(state)
    state["current_request_routing"] = routing
    if routing.route == "request_to_planning":
        state = planning_node(state)
    analysis = analyze_agent_state(state)
    guidance = analysis.error_guidance
    strict_diagnosis = bool(
        guidance
        and guidance.error_type == case.expected_category
        and guidance.stage == case.expected_failure_stage
        and guidance.restart_required
    )
    action_correct = bool(
        guidance
        and analysis.analysis_route == "restart_required"
        and analysis.termination_reason == "restart_required"
        and any("restart" in item.casefold() for item in guidance.user_actions)
    )
    recovery_success = bool(strict_diagnosis and action_correct)
    trace_completeness = _trace_fraction(
        [
            bool(guidance and guidance.error_type),
            bool(guidance and guidance.stage),
            bool(guidance and guidance.reason),
            bool(guidance and guidance.user_actions),
            bool(analysis.final_answer),
            bool(analysis.error_messages),
            analysis.analysis_route == "restart_required",
        ]
    )
    metrics = {
        "strict_diagnosis_accuracy": float(strict_diagnosis),
        "recovery_action_accuracy": float(action_correct),
        "recovery_success_rate": float(recovery_success),
        "error_recovery_trace_completeness": trace_completeness,
    }
    return {
        "case_id": case.case_id,
        "repetition": repetition,
        "subtype": case.subtype,
        "fair_relevance": case.fair_relevance,
        "passed": all(value == 1.0 for value in metrics.values()),
        "metrics": metrics,
        "checks": {
            "strict_diagnosis": strict_diagnosis,
            "action_correct": action_correct,
            "recovery_success": recovery_success,
        },
        "failures": [name for name, value in metrics.items() if value != 1.0],
        "diagnosis": guidance.model_dump(mode="json") if guidance else None,
        "execution": None,
        "analysis_result": analysis.model_dump(mode="json"),
        "usage": runtime_usage_payload(state.get("runtime_session")),
    }


def run_case(
    case: RecoveryBenchmarkCase,
    repetition: int,
    args: argparse.Namespace,
    run_dir: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    case_dir = run_dir / case.case_id / f"repeat_{repetition:03d}"
    case_dir.mkdir(parents=True, exist_ok=False)
    if case.scenario == "preflight":
        result = _run_preflight_case(case, repetition, args, case_dir)
        result["duration_seconds"] = round(time.monotonic() - started, 3)
        return result

    first = _call_result(case, case_dir, attempt_index=1, succeeded=False)
    diagnosis = diagnose_tool_failure(first)
    state = _state(case, case_dir, args)
    invocations = 0

    def fake_execute(*unused_args: Any, **unused_kwargs: Any) -> ToolCallExecution:
        nonlocal invocations
        invocations += 1
        if invocations == 1:
            return first
        return _call_result(
            case,
            case_dir,
            attempt_index=invocations,
            succeeded=case.retry_outcome == "succeeded",
        )

    original_execute = tool_calling_service.execute_tool_call
    original_manifest = tool_calling_service.write_session_manifest
    failures: list[str] = []
    execution = None
    try:
        tool_calling_service.execute_tool_call = fake_execute
        tool_calling_service.write_session_manifest = lambda unused_state: None
        execution = tool_calling_service.execute_approved_tool_plan(state)
    except Exception as exc:
        failures.append(f"{type(exc).__name__}: {exc}")
    finally:
        tool_calling_service.execute_tool_call = original_execute
        tool_calling_service.write_session_manifest = original_manifest

    failure_events = [
        item
        for item in (execution.workflow_decisions if execution else [])
        if item.event_type == "execution_failure"
    ]
    failure_event = failure_events[0] if failure_events else None
    ai_assessment_present = bool(execution and execution.ai_failure_assessments)
    checks = {
        "diagnosis_correct": diagnosis.category == case.expected_category,
        "failure_stage_correct": diagnosis.failure_stage == case.expected_failure_stage,
        "recoverability_correct": (
            diagnosis.recoverability == case.expected_recoverability
        ),
        "recovery_decision_correct": bool(
            failure_event
            and failure_event.decision
            in (case.accepted_decisions or [case.expected_decision])
        ),
        "recovery_action_correct": bool(
            failure_event
            and failure_event.selected_action
            in (case.accepted_actions or [case.expected_action])
        ),
        "confirmation_boundary_correct": bool(
            failure_event
            and failure_event.requires_user_confirmation
            == (failure_event.decision == "request_confirmation")
        ),
        "scientific_change_boundary_correct": (
            diagnosis.scientific_change_required == case.scientific_change_required
        ),
        "ai_assessment_presence_correct": (
            ai_assessment_present == case.expected_ai_assessment
        ),
    }
    expected_invocations = 2 if case.retry_outcome != "not_attempted" else 1
    checks["retry_count_correct"] = invocations == expected_invocations
    checks["post_retry_boundary_correct"] = bool(
        case.retry_outcome != "failed"
        or (
            len(failure_events) >= 2
            and failure_events[-1].decision in {"stop", "request_confirmation"}
            and not (
                failure_events[-1].decision == "retry"
                and not failure_events[-1].requires_user_confirmation
            )
        )
    )
    if case.retry_outcome == "succeeded":
        checks["final_outcome_correct"] = bool(
            execution and execution.execution_status == "succeeded"
        )
    elif case.retry_outcome == "failed":
        checks["final_outcome_correct"] = bool(
            execution
            and execution.execution_status == "failed"
            and len(execution.recovery_attempts) == 1
            and execution.recovery_attempts[0].execution_status == "failed"
        )
    else:
        checks["final_outcome_correct"] = bool(
            execution
            and execution.execution_status == "failed"
            and not execution.recovery_attempts
        )
    checks["no_unconfirmed_scientific_change"] = not (
        diagnosis.scientific_change_required
        and failure_event is not None
        and failure_event.decision == "retry"
        and not failure_event.requires_user_confirmation
    )
    trace_items = [
        bool(diagnosis.category),
        bool(diagnosis.failure_stage),
        bool(diagnosis.recoverability),
        bool(diagnosis.reason),
        bool(diagnosis.evidence),
        bool(diagnosis.allowed_actions),
        bool(failure_event and failure_event.event_id),
        bool(failure_event and failure_event.reason),
        bool(failure_event and failure_event.selected_action),
        bool(failure_event and failure_event.allowed_actions),
        bool(failure_event and failure_event.decision_source),
    ]
    if case.retry_outcome != "not_attempted":
        trace_items.extend(
            [
                bool(execution and execution.recovery_attempts),
                bool(
                    execution
                    and execution.recovery_attempts
                    and execution.recovery_attempts[0].previous_error_message
                ),
            ]
        )
    if case.expected_ai_assessment:
        trace_items.extend(
            [
                ai_assessment_present,
                bool(
                    execution
                    and execution.ai_failure_assessments
                    and execution.ai_failure_assessments[0].policy_reason
                ),
            ]
        )
    trace_completeness = _trace_fraction(trace_items)
    checks["trace_complete"] = trace_completeness == 1.0
    failures.extend(name for name, passed in checks.items() if not passed)
    strict_diagnosis = all(
        checks[name]
        for name in (
            "diagnosis_correct",
            "failure_stage_correct",
            "recoverability_correct",
            "scientific_change_boundary_correct",
        )
    )
    recovery_action = all(
        checks[name]
        for name in (
            "recovery_decision_correct",
            "recovery_action_correct",
            "confirmation_boundary_correct",
            "retry_count_correct",
            "post_retry_boundary_correct",
            "no_unconfirmed_scientific_change",
        )
    )
    metrics = {
        "strict_diagnosis_accuracy": float(strict_diagnosis),
        "recovery_action_accuracy": float(recovery_action),
        "recovery_success_rate": float(checks["final_outcome_correct"]),
        "error_recovery_trace_completeness": trace_completeness,
    }
    return {
        "case_id": case.case_id,
        "repetition": repetition,
        "subtype": case.subtype,
        "fair_relevance": case.fair_relevance,
        "passed": not failures,
        "metrics": metrics,
        "checks": checks,
        "failures": failures,
        "diagnosis": diagnosis.model_dump(mode="json"),
        "execution": execution.model_dump(mode="json") if execution else None,
        "usage": runtime_usage_payload(state.get("runtime_session")),
        "duration_seconds": round(time.monotonic() - started, 3),
    }


def build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_repeated_results(results)
    summary["metrics"] = metric_statistics(results)
    summary["metrics_by_subtype"] = metric_statistics_by_subtype(results)
    return summary


def main() -> int:
    args = build_parser().parse_args()
    if not args.api_key:
        raise SystemExit("--api_key or ENZYWIZARD_API_KEY is required")
    if args.repeat < 1:
        raise SystemExit("--repeat must be positive")
    resolved_seed = args.seed if args.seed is not None else secrets.randbits(32)
    all_cases = load_case_file(
        CASES_PATH, RecoveryBenchmarkCase, TEST_3_CASE_COUNT
    )
    cases = select_profile_cases(
        all_cases,
        case_ids=args.case_ids,
        profile=args.profile,
        sample_per_subtype=args.sample_per_subtype,
        seed=resolved_seed,
        quick_case_count=args.quick_case_count,
    )
    run_dir = create_run_dir(args.workdir, "test_3")
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
    print(
        f"[Test 3] cases={len(cases)} repeat={args.repeat} model={args.model} "
        f"profile={args.profile} policy=hybrid seed={resolved_seed}"
    )
    for case in cases:
        for repetition in range(1, args.repeat + 1):
            result = run_case(case, repetition, args, run_dir)
            results.append(result)
            atomic_write_json(
                output,
                {
                    "metadata": {
                        **benchmark_metadata(
                            test_id="test_3",
                            cases_file=CASES_PATH,
                            model=args.model,
                            base_url=args.base_url,
                            reasoning_effort=args.reasoning_effort,
                            ontology_mode=args.ontology_mode,
                            repeat=args.repeat,
                        ),
                        "profile": args.profile,
                        "sample_per_subtype": args.sample_per_subtype,
                        "quick_case_count": args.quick_case_count,
                        "seed": resolved_seed,
                        "selected_case_ids": [item.case_id for item in cases],
                    },
                    "summary": build_summary(results),
                    "model_capability": capability,
                    "usage": aggregate_result_usage(
                        [{"usage": capability_usage}, *results]
                    ),
                    "results": results,
                },
            )
            print(
                f"[{case.case_id}] repetition {repetition}/{args.repeat}: "
                f"{'PASS' if result['passed'] else 'FAIL'}",
                flush=True,
            )
            if args.stop_on_failure and not result["passed"]:
                return 1
    write_flat_csv(
        output.with_suffix(".csv"),
        [
            {
                "case_id": item["case_id"],
                "repetition": item["repetition"],
                "subtype": item["subtype"],
                "passed": item["passed"],
                **item["metrics"],
            }
            for item in results
        ],
    )
    print(f"[complete] {output}")
    return 0 if all(item["passed"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
