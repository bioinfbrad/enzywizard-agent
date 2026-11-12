#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import time
from typing import Any
from unittest.mock import patch
from uuid import uuid4

import yaml

from enzywizard_agent.conf import AgentConfig
from enzywizard_agent.data_models.agent_inputs import AgentInputs
from enzywizard_agent.data_models.artifact import Artifact, CleanedFiles
from enzywizard_agent.data_models.planning import (
    ParameterExtractionOutput,
    PlanValidationRecord,
    PlanningResult,
    ReportStagingRule,
    ToolCall,
    ToolOutputArtifact,
    ToolOutputReference,
    ToolPlan,
    ToolSelectionOutput,
)
from enzywizard_agent.data_models.runtime import RuntimeSession
from enzywizard_agent.data_models.tool_execution import ToolCallExecution
from enzywizard_agent.data_models.user_confirmation import UserConfirmation
from enzywizard_agent.services import planning_service, tool_calling_service
from enzywizard_agent.services.analysis_service import (
    _analyze_execution_failure,
    _answer_missing_input_error,
)
from enzywizard_agent.services.llm_service import check_model_capabilities
from enzywizard_agent.services.runtime_service import utc_now
from enzywizard_agent.utils.tool_plan_utils import load_enzywizard_tool_specs
from enzywizard_agent.utils.tool_execution_utils import (
    discover_tool_artifacts,
    resolve_tool_call_arguments,
    stage_tool_reports,
)

from scoring import METRIC_NAMES, cluster_bootstrap_ci, repetition_statistics, score_case


ROOT = Path(__file__).resolve().parent
CATALOG_PATH = ROOT / "case_catalog.yaml"
INDEX_PATH = ROOT / "cases" / "index.yaml"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-6-luna"
DEFAULT_REASONING_EFFORT = "medium"
PREPARATION_INJECTION_POINTS = {
    "argument_resolution",
    "managed_input_repair",
    "preparation",
    "report_staging",
    "workspace_cleanup",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test 3: failure diagnosis, safe handling, and recovery."
    )
    parser.add_argument("--api_key", default=os.environ.get("ENZYWIZARD_API_KEY"))
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning_effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument(
        "--ontology_mode", choices=("off", "retrieve", "control"), default="retrieve"
    )
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument("--quick_case_count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--case_ids", nargs="*")
    parser.add_argument("--show_details", action="store_true")
    parser.add_argument("--stop_on_failure", action="store_true")
    parser.add_argument(
        "--bootstrap_iterations",
        type=int,
        default=1000,
        help="Case-cluster bootstrap replicates; use 0 to omit confidence intervals.",
    )
    parser.add_argument("--validate_only", action="store_true")
    parser.add_argument("--workdir", default="~/enzywizard-evaluation-test-3")
    parser.add_argument("--output")
    return parser


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML object in {path}")
    return payload


def load_suite() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    catalog = _load_yaml(CATALOG_PATH)
    index = _load_yaml(INDEX_PATH)
    cases: list[dict[str, Any]] = []
    for category in index["categories"]:
        payload = _load_yaml(ROOT / "cases" / category["file"])
        category_cases = payload.get("cases")
        if not isinstance(category_cases, list) or len(category_cases) != category["count"]:
            raise ValueError(f"Category count mismatch for {category['category_id']}")
        for raw_case in category_cases:
            case = dict(raw_case)
            case["category_id"] = category["category_id"]
            case["automatic_recovery_category"] = bool(
                category["automatic_recovery_expected"]
            )
            cases.append(case)
    if len(cases) != index["total_cases"] or len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("Test 3 case count or identifier uniqueness is invalid")
    validate_suite(catalog, cases)
    return catalog, cases


def validate_suite(catalog: dict[str, Any], cases: list[dict[str, Any]]) -> None:
    specs = load_enzywizard_tool_specs()
    automatic_categories = set(catalog["automatic_recovery_categories"])
    required_expected = {
        "diagnostic_category",
        "failure_stage",
        "recoverability",
        "decision",
        "action",
        "final_status",
        "automatic_recovery_success",
        "scientific_change_required",
        "user_confirmation_required",
    }
    for case in cases:
        missing = required_expected - set(case.get("expected", {}))
        if missing:
            raise ValueError(f"{case['case_id']}: missing expected fields {sorted(missing)}")
        if case["tool_name"] not in specs:
            raise ValueError(f"{case['case_id']}: unknown tool {case['tool_name']}")
        if case["scenario"] not in {
            "plan_validation",
            "tool_execution",
            "preflight",
            "safety_boundary",
        }:
            raise ValueError(f"{case['case_id']}: unknown scenario")
        if not case.get("required_trace") or len(case["required_trace"]) != len(
            set(case["required_trace"])
        ):
            raise ValueError(f"{case['case_id']}: invalid required_trace")
        is_automatic = case["category_id"] in automatic_categories
        if is_automatic != (case["expected"]["automatic_recovery_success"] is True):
            raise ValueError(f"{case['case_id']}: AARSR applicability mismatch")
        overlap = set(case.get("allowed_actions", [])) & set(
            case.get("forbidden_actions", [])
        )
        if overlap:
            raise ValueError(f"{case['case_id']}: action contract overlaps {sorted(overlap)}")
        if case["expected"]["action"] not in case.get("allowed_actions", []):
            raise ValueError(
                f"{case['case_id']}: expected action is absent from allowed_actions"
            )


def select_cases(cases: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.case_ids:
        by_id = {case["case_id"]: case for case in cases}
        unknown = [case_id for case_id in args.case_ids if case_id not in by_id]
        if unknown:
            raise ValueError(f"Unknown case IDs: {unknown}")
        return [by_id[case_id] for case_id in args.case_ids]
    if args.profile == "full":
        return list(cases)
    if args.quick_case_count < 1 or args.quick_case_count > len(cases):
        raise ValueError("--quick_case_count must be between 1 and 40")
    rng = random.Random(args.seed)
    by_category: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        by_category.setdefault(case["category_id"], []).append(case)
    selected: list[dict[str, Any]] = []
    categories = list(by_category)
    while len(selected) < args.quick_case_count:
        progressed = False
        for category in categories:
            pool = [case for case in by_category[category] if case not in selected]
            if pool and len(selected) < args.quick_case_count:
                selected.append(rng.choice(pool))
                progressed = True
        if not progressed:
            break
    return selected


def _config(args: argparse.Namespace, workdir: Path) -> AgentConfig:
    return AgentConfig(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        ontology_mode=args.ontology_mode,
        tool_recovery_mode="safe",
        failure_diagnosis_mode="hybrid",
        max_tool_recovery_attempts=1,
        check_model_capabilities=False,
        workdir=workdir,
    )


def _usage(state: dict[str, Any]) -> dict[str, int]:
    runtime = state.get("runtime_session")
    records = runtime.llm_calls if runtime is not None else []
    return {
        "llm_calls": len(records),
        "input_tokens": sum(item.input_tokens or 0 for item in records),
        "output_tokens": sum(item.output_tokens or 0 for item in records),
        "total_tokens": sum(item.total_tokens or 0 for item in records),
    }


def _call(
    case: dict[str, Any], case_dir: Path, *, step_index: int = 1, side: str = "single"
) -> ToolCall:
    arguments: dict[str, Any] = {}
    if case["fault_injection"].get("fault_type") in {
        "stale_output_reference",
        "stale_imported_historical_reference",
        "stale_side_specific_output_reference",
        "ambiguous_output_reference",
    }:
        fault = case["fault_injection"]
        arguments[fault["argument_name"]] = ToolOutputReference(
            source_step_index=99,
            artifact_type=fault["artifact_type"],
            expected_path_pattern=str(case_dir / "stale" / "missing"),
        )
    output_artifacts: list[ToolOutputArtifact] = []
    if case["fault_injection"].get("injection_point") == "output_contract":
        artifact_type = case["fault_injection"]["artifact_type"]
        suffix = ".sdf" if artifact_type == "docked_substrate_sdf" else ".json"
        output_artifacts.append(
            ToolOutputArtifact(
                artifact_type=artifact_type,
                path_pattern=str(
                    case_dir / f"step_{step_index:02d}_{case['tool_name']}" / f"required{suffix}"
                ),
                side=side,
                required=True,
            )
        )
    return ToolCall(
        turn_id=1,
        plan_id=1,
        step_index=step_index,
        tool_name=case["tool_name"],
        command_name=case["tool_name"],
        side=side,
        tool_execution_dir=str(case_dir / f"step_{step_index:02d}_{case['tool_name']}") ,
        arguments=arguments,
        output_artifacts=output_artifacts,
        staging_rules=[],
        call_reason="Execute the controlled failure-recovery evaluation case.",
    )


def _planning_state(
    case: dict[str, Any], case_dir: Path, args: argparse.Namespace, calls: list[ToolCall]
) -> dict[str, Any]:
    purpose = "wildtype_mutant_comparison" if any(call.side in {"wt", "mut", "shared"} for call in calls) else "single_protein_analysis"
    plan = ToolPlan(
        turn_id=1,
        plan_id=1,
        purpose_type=purpose,
        summary="Controlled failure-recovery evaluation.",
        plan_reason="Inject one predefined fault and observe bounded handling.",
        calls=calls,
    )
    planning = PlanningResult(
        turn_id=1,
        plan_id=1,
        planning_route="tool_plan_generated",
        planning_flag="tool_plan_generated",
        needs_tool=True,
        summary=plan.summary,
        purpose_type=purpose,
        purpose_reason="Controlled evaluation.",
        tool_selection=ToolSelectionOutput(
            tool_selection_status="tool_selection_succeeded",
            selected_tools=list(dict.fromkeys(call.tool_name for call in calls)),
            selection_reason="Selected by the controlled evaluation fixture.",
            error_message="",
        ),
        parameter_extraction=ParameterExtractionOutput(
            parameter_extraction_status="parameter_extraction_succeeded",
            parameter_adjustments=[],
            profile_requests=[],
            unsupported_requests=[],
            extraction_summary="No user parameter overrides were requested.",
            error_message="",
        ),
        tool_plan=plan,
        error_messages=[],
    )
    return {
        "config": _config(args, case_dir),
        "session_id": f"test3-{case['case_id']}-{uuid4().hex[:8]}",
        "current_turn_id": 1,
        "current_plan_id": 1,
        "current_user_message": f"Run controlled recovery case {case['case_id']}.",
        "current_planning_result": planning,
        "current_user_confirmation": UserConfirmation(
            turn_id=1, plan_id=1, decision="approved", revision_request=None
        ),
        "runtime_session": RuntimeSession(
            started_at=utc_now(), started_monotonic=time.monotonic()
        ),
        "artifacts": [],
        "current_plan_artifacts": [],
    }


def _artifact(
    case: dict[str, Any], case_dir: Path, *, step_index: int, side: str
) -> Artifact:
    artifact_type = case["fault_injection"]["artifact_type"]
    path = case_dir / f"registered_step_{step_index}_{artifact_type}.dat"
    path.write_text("registered evaluation artifact\n", encoding="utf-8")
    return Artifact(
        turn_id=1,
        plan_id=1,
        artifact_type=artifact_type,
        path=str(path.resolve()),
        source_tool="enzywizard-clean",
        side=side,
        metadata={"step_index": step_index, "evaluation_fixture": True},
    )


def _execution_result(
    call: ToolCall,
    case_dir: Path,
    *,
    attempt_index: int,
    succeeded: bool,
    artifacts: list[Artifact] | None = None,
    error_message: str | None = None,
) -> ToolCallExecution:
    stdout = Path(call.tool_execution_dir) / (
        "stdout.txt" if attempt_index == 1 else f"stdout_attempt_{attempt_index}.txt"
    )
    stderr = Path(call.tool_execution_dir) / (
        "stderr.txt" if attempt_index == 1 else f"stderr_attempt_{attempt_index}.txt"
    )
    stdout.parent.mkdir(parents=True, exist_ok=True)
    stdout.write_text("controlled evaluation output\n", encoding="utf-8")
    stderr.write_text("", encoding="utf-8")
    return ToolCallExecution(
        turn_id=call.turn_id,
        plan_id=call.plan_id,
        step_index=call.step_index,
        tool_name=call.tool_name,
        side=call.side,
        attempt_index=attempt_index,
        execution_status="succeeded" if succeeded else "failed",
        command=[call.command_name] if succeeded else [],
        resolved_arguments={},
        exit_code=0 if succeeded else 1,
        duration_seconds=0.001,
        stdout_file=str(stdout.resolve()),
        stderr_file=str(stderr.resolve()),
        log_files=[],
        stdout_excerpt="",
        stderr_excerpt="",
        log_excerpt="",
        artifacts=list(artifacts or []),
        error_message=None if succeeded else (error_message or "Injected evaluation failure."),
    )


def _run_plan_case(
    case: dict[str, Any], args: argparse.Namespace, case_dir: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    structure_file = case_dir / "wt_input.cif"
    structure_file.write_text("controlled WT structure input\n", encoding="utf-8")
    is_paired = case["case_id"] == "T3-PS05"
    mutant_file = case_dir / "mut_input.cif"
    if is_paired:
        mutant_file.write_text("controlled mutant structure input\n", encoding="utf-8")
    selected_tools = {
        "T3-PS01": ["enzywizard-clean", "enzywizard-energy", "enzywizard-integrate"],
        "T3-PS02": [
            "enzywizard-clean",
            "enzywizard-substrate",
            "enzywizard-dock",
            "enzywizard-integrate",
        ],
        "T3-PS03": ["enzywizard-clean", "enzywizard-energy", "enzywizard-integrate"],
        "T3-PS04": ["enzywizard-clean", "enzywizard-flexibility", "enzywizard-integrate"],
        "T3-PS05": [
            "enzywizard-mut-clean",
            "enzywizard-energy",
            "enzywizard-mut-integrate",
        ],
    }[case["case_id"]]
    purpose = "wildtype_mutant_comparison" if is_paired else "single_protein_analysis"
    planning = PlanningResult(
        turn_id=1,
        plan_id=1,
        planning_route="tool_plan_generated",
        planning_flag="tool_plan_generated",
        needs_tool=True,
        summary="Controlled plan-repair evaluation.",
        purpose_type=purpose,
        purpose_reason="Exercise bounded repair of Agent-managed plan structure.",
        tool_selection=ToolSelectionOutput(
            tool_selection_status="tool_selection_succeeded",
            selected_tools=selected_tools,
            selection_reason="Use the fixed scientific requirements for this fixture.",
            error_message="",
        ),
        parameter_extraction=ParameterExtractionOutput(
            parameter_extraction_status="parameter_extraction_succeeded",
            parameter_adjustments=[],
            profile_requests=[],
            unsupported_requests=[],
            extraction_summary="No user parameter overrides were requested.",
            error_message="",
        ),
        tool_plan=None,
        error_messages=[],
    )
    state = {
        "config": _config(args, case_dir),
        "session_id": f"test3-{case['case_id']}-{uuid4().hex[:8]}",
        "current_turn_id": 1,
        "current_plan_id": 1,
        "current_user_message": f"Run controlled plan-repair case {case['case_id']}.",
        "current_planning_result": planning,
        "agent_inputs": AgentInputs(
            structure_file=str(structure_file.resolve()),
            msa_file=None,
            mut_structure_file=str(mutant_file.resolve()) if is_paired else None,
            mut_msa_file=None,
            substrate_name="cholesterol" if case["case_id"] == "T3-PS02" else None,
            amino_acid_substitution="A1V" if is_paired else None,
        ),
        "cleaned_files": CleanedFiles(
            **{name: None for name in CleanedFiles.model_fields}
        ),
        "runtime_session": RuntimeSession(
            started_at=utc_now(), started_monotonic=time.monotonic()
        ),
        "artifacts": [],
        "current_plan_artifacts": [],
    }
    state.pop("current_user_confirmation", None)
    extraction = planning.parameter_extraction
    if extraction is None:
        raise RuntimeError("Plan-repair fixture is missing parameter extraction.")

    canonical = planning_service.generate_default_tool_plan(state, planning)
    canonical = planning_service.finalize_tool_plan(state, canonical, extraction)
    injected = canonical.model_copy(deep=True)
    fault_type = case["fault_injection"]["fault_type"]
    if fault_type == "remove_required_dependency":
        injected.calls = [
            call for call in injected.calls if call.tool_name != "enzywizard-clean"
        ]
    elif fault_type == "replace_output_reference_artifact_type":
        dock_call = next(call for call in injected.calls if call.tool_name == "enzywizard-dock")
        argument_name, reference = next(
            (name, value)
            for name, value in dock_call.arguments.items()
            if isinstance(value, ToolOutputReference)
        )
        dock_call.arguments[argument_name] = reference.model_copy(
            update={"artifact_type": "invalid_structure_artifact"}
        )
    elif fault_type == "replace_staging_source_step":
        integrate_call = next(
            call for call in injected.calls if call.tool_name == "enzywizard-integrate"
        )
        integrate_call.staging_rules[0].source_step_index = 999
    elif fault_type == "replace_managed_execution_directory":
        target_call = next(
            call for call in injected.calls if call.tool_name == case["tool_name"]
        )
        target_call.tool_execution_dir = case["fault_injection"]["injected_value"]
    elif fault_type == "invalidate_mutant_side_reference":
        mutant_call = next(
            call
            for call in injected.calls
            if call.tool_name == "enzywizard-energy" and call.side == "mut"
        )
        argument_name, reference = next(
            (name, value)
            for name, value in mutant_call.arguments.items()
            if isinstance(value, ToolOutputReference)
        )
        mutant_call.arguments[argument_name] = reference.model_copy(
            update={"source_step_index": 999}
        )
    else:
        raise ValueError(f"Unsupported plan fault type: {fault_type}")

    failed = planning_service.audit_tool_plan(injected, state["config"].workdir)
    if failed.passed:
        raise RuntimeError(f"Injected plan fault was not detected: {fault_type}")
    planning.tool_plan = injected
    repaired, record = planning_service.repair_internal_tool_plan(
        state, planning, extraction, failed
    )
    planning.plan_repairs.append(record)
    planning.plan_validation = record.validation
    planning.tool_plan = repaired
    # Plan repair occurs before approval; the fixture records approval only after
    # the regenerated plan has passed deterministic validation.
    state["current_user_confirmation"] = UserConfirmation(
        turn_id=1, plan_id=1, decision="approved", revision_request=None
    )
    observed_actions = {"regenerate_program_managed_plan"}
    forbidden_observed = sorted(
        observed_actions.intersection(case.get("forbidden_actions", []))
    )
    trace = {
        "failed_plan_validation": bool(failed.error_messages),
        "plan_repair_record": record.repair_type == "regenerate_program_managed_plan",
        "successful_revalidation": bool(repaired and record.validation.passed),
        "user_approval_status": state["current_user_confirmation"].decision == "approved",
    }
    observation = {
        "diagnostic_category": "internal_plan_validation",
        "failure_stage": "plan_validation",
        "recoverability": "automatic",
        "explanation_complete": bool(record.reason and record.trigger_errors),
        "decision": "repair",
        "action": record.repair_type,
        "final_status": "succeeded" if repaired is not None else "stopped",
        "scientific_change_required": False,
        "user_confirmation_required": False,
        "forbidden_actions_observed": forbidden_observed,
        "bounded_adjustment_applied": record.repair_status == "succeeded",
        "adjustment_count": 1,
        "retry_count": 0,
        "trace_items": trace,
    }
    return observation, {
        "injected_plan_validation": failed.model_dump(mode="json"),
        "plan_repair": record.model_dump(mode="json"),
    }, state


def _run_preflight_case(
    case: dict[str, Any], args: argparse.Namespace, case_dir: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    state = {
        "config": _config(args, case_dir),
        "current_turn_id": 1,
        "current_user_message": "Analyze the supplied enzyme structure.",
        "runtime_session": RuntimeSession(
            started_at=utc_now(), started_monotonic=time.monotonic()
        ),
    }
    analysis = _answer_missing_input_error(state, ["a protein structure"])
    guidance = analysis.error_guidance
    trace = {
        "input_validation_error": bool(guidance and guidance.error_type == "input_error"),
        "user_guidance": bool(guidance and guidance.reason and guidance.user_actions),
        "final_outcome": analysis.analysis_route == "restart_required",
    }
    observed_actions = {"request_restart"}
    forbidden_observed = sorted(
        observed_actions.intersection(case.get("forbidden_actions", []))
    )
    observation = {
        "diagnostic_category": guidance.error_type if guidance else None,
        "failure_stage": guidance.stage if guidance else None,
        "recoverability": "not_recoverable",
        "explanation_complete": bool(guidance and guidance.reason and analysis.final_answer),
        "decision": "stop",
        "action": "request_restart",
        "final_status": "stopped",
        "scientific_change_required": False,
        "user_confirmation_required": False,
        "forbidden_actions_observed": forbidden_observed,
        "bounded_adjustment_applied": False,
        "adjustment_count": 0,
        "retry_count": 0,
        "trace_items": trace,
    }
    return observation, {"analysis": analysis.model_dump(mode="json")}, state


def _run_tool_case(
    case: dict[str, Any], args: argparse.Namespace, case_dir: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    is_artifact_case = case["category_id"] == "artifact_reference"
    is_ambiguous_artifact_case = (
        case["fault_injection"].get("fault_type") == "ambiguous_output_reference"
    )
    is_managed_reference_case = is_artifact_case or is_ambiguous_artifact_case
    target_side = (
        "mut"
        if case["fault_injection"].get("requested_side") == "mut"
        else "shared"
        if case["tool_name"] == "enzywizard-mut-integrate"
        else "single"
    )
    is_staging_case = case["category_id"] == "input_staging_workspace"
    source_step_count = (
        2 if is_ambiguous_artifact_case else 1 if is_artifact_case or is_staging_case else 0
    )
    if is_managed_reference_case or is_staging_case:
        source_calls = [
            ToolCall(
                turn_id=1,
                plan_id=1,
                step_index=step_index,
                tool_name="enzywizard-mut-clean" if target_side == "mut" else "enzywizard-clean",
                command_name="enzywizard-mut-clean" if target_side == "mut" else "enzywizard-clean",
                side="shared" if target_side == "mut" else "single",
                tool_execution_dir=str(case_dir / f"step_{step_index:02d}_source"),
                arguments={},
                output_artifacts=[],
                staging_rules=[],
                call_reason="Register an upstream artifact used by the recovery fixture.",
            )
            for step_index in range(1, source_step_count + 1)
        ]
        target_call = _call(
            case, case_dir, step_index=source_step_count + 1, side=target_side
        )
        if is_staging_case:
            source_path = case_dir / (
                f"registered_step_1_{case['fault_injection']['artifact_type']}.dat"
            )
            destination_dir = Path(target_call.tool_execution_dir) / "staged_reports"
            target_call.staging_rules = [
                ReportStagingRule(
                    source_step_index=1,
                    source_path_pattern=str(source_path.resolve()),
                    destination_dir=str(destination_dir.resolve()),
                    side=target_side,
                )
            ]
        calls = [*source_calls, target_call]
    else:
        target_call = _call(case, case_dir, side=target_side)
        calls = [target_call]
    state = _planning_state(case, case_dir, args, calls)
    target_invocations = 0
    first_failure: ToolCallExecution | None = None
    upstream_artifacts: list[Artifact] = []
    upstream_artifact_paths: list[Path] = []
    partial_output = Path(target_call.tool_execution_dir) / "partial_scientific_output.json"
    original_reference_step = None
    if is_artifact_case:
        reference = target_call.arguments[case["fault_injection"]["argument_name"]]
        original_reference_step = reference.source_step_index

    def fake_execute(
        unused_state: dict[str, Any],
        active_call: ToolCall,
        unused_spec: dict[str, Any],
        unused_artifacts: list[Artifact],
        unused_successful_steps: set[int],
        *,
        attempt_index: int = 1,
    ) -> ToolCallExecution:
        nonlocal target_invocations, first_failure
        if (
            (is_managed_reference_case or is_staging_case)
            and active_call.step_index <= source_step_count
        ):
            upstream = _artifact(
                case, case_dir, step_index=active_call.step_index, side=target_side
            )
            upstream_artifacts.append(upstream)
            upstream_artifact_paths.append(Path(upstream.path))
            return _execution_result(
                active_call,
                case_dir,
                attempt_index=attempt_index,
                succeeded=True,
                artifacts=[upstream],
            )
        target_invocations += 1
        if target_invocations == 1:
            execution_dir = Path(active_call.tool_execution_dir)
            execution_dir.mkdir(parents=True, exist_ok=True)
            partial_output.write_text("partial\n", encoding="utf-8")
            injected_error = case["fault_injection"]["error_message"]
            try:
                if is_artifact_case:
                    resolve_tool_call_arguments(active_call, unused_artifacts)
                    raise RuntimeError("Stale output reference was not detected.")
                if is_staging_case:
                    rule = active_call.staging_rules[0]
                    source_path = Path(rule.source_path_pattern)
                    destination = Path(rule.destination_dir) / source_path.name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_text("conflicting staged report\n", encoding="utf-8")
                    stage_tool_reports(active_call, unused_successful_steps)
                    raise RuntimeError("Staging conflict was not detected.")
                if case["fault_injection"]["injection_point"] == "output_contract":
                    _, missing = discover_tool_artifacts(active_call)
                    if not missing:
                        raise RuntimeError("Missing required output was not detected.")
                    injected_error = (
                        f"{active_call.tool_name} completed without required outputs: "
                        f"{', '.join(sorted(missing))}."
                    )
            except (ValueError, FileNotFoundError) as error:
                injected_error = (
                    f"{active_call.tool_name} could not be started or prepared: {error}"
                )
            failed = _execution_result(
                active_call,
                case_dir,
                attempt_index=1,
                succeeded=False,
                error_message=injected_error,
            )
            failed.stderr_excerpt = failed.error_message
            if (
                case["fault_injection"]["injection_point"]
                not in PREPARATION_INJECTION_POINTS
            ):
                failed.command = [active_call.command_name]
            first_failure = failed
            return first_failure
        tool_calling_service.clear_failed_attempt_outputs(
            Path(active_call.tool_execution_dir), state["config"].workdir
        )
        resolved_arguments: dict[str, Any] = {}
        artifacts: list[Artifact] = []
        if is_artifact_case:
            resolved_arguments = resolve_tool_call_arguments(active_call, unused_artifacts)
        elif is_staging_case:
            stage_tool_reports(active_call, unused_successful_steps)
        elif case["fault_injection"]["injection_point"] == "output_contract":
            expected = active_call.output_artifacts[0]
            output_path = Path(expected.path_pattern)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                "{}\n" if output_path.suffix == ".json" else "recovered output\n",
                encoding="utf-8",
            )
            artifacts, missing = discover_tool_artifacts(active_call)
            if missing:
                raise RuntimeError(f"Recovered output contract remains incomplete: {missing}")
        recovered = _execution_result(
            active_call,
            case_dir,
            attempt_index=attempt_index,
            succeeded=True,
            artifacts=artifacts,
        )
        recovered.resolved_arguments = resolved_arguments
        return recovered

    execution = None
    runner_errors: list[str] = []
    with patch.object(tool_calling_service, "execute_tool_call", side_effect=fake_execute), patch.object(
        tool_calling_service, "write_session_manifest", return_value=None
    ):
        try:
            execution = tool_calling_service.execute_approved_tool_plan(state)
        except Exception as error:
            runner_errors.append(f"{type(error).__name__}: {error}")

    failure_events = [
        item
        for item in (execution.workflow_decisions if execution else [])
        if item.event_type == "execution_failure"
    ]
    event = failure_events[0] if failure_events else None
    # The execution summary keeps only the final attempt for a recovered step;
    # retain the injected first-attempt object so its actual diagnosis remains
    # available for scoring and audit.
    diagnosis = first_failure.failure_diagnosis if first_failure is not None else None

    ai_records = execution.ai_failure_assessments if execution else []
    successful_ai = next(
        (item for item in ai_records if item.assessment_status == "succeeded"), None
    )
    diagnostic_explanation_complete = bool(
        diagnosis and diagnosis.reason and diagnosis.evidence
    )
    if successful_ai is not None:
        diagnostic_explanation_complete = bool(
            diagnostic_explanation_complete
            and successful_ai.diagnosis
            and successful_ai.diagnosis.user_explanation
            and successful_ai.diagnosis.evidence_ids
        )
    retry_count = len(execution.recovery_attempts) if execution else 0
    final_status = (
        "succeeded"
        if execution and execution.execution_status == "succeeded"
        else (
            "awaiting_user_confirmation"
            if event and event.decision == "request_confirmation"
            else "stopped"
        )
    )
    failure_analysis = None
    if execution is not None and final_status != "succeeded":
        state["current_tool_execution"] = execution
        failure_analysis = _analyze_execution_failure(state)
    user_failure_guidance_complete = bool(
        failure_analysis
        and failure_analysis.error_guidance
        and failure_analysis.error_guidance.reason
        and failure_analysis.error_guidance.user_actions
        and failure_analysis.final_answer
    )
    explanation_complete = bool(
        diagnostic_explanation_complete
        and (final_status == "succeeded" or user_failure_guidance_complete)
    )
    repairs = [
        repair
        for attempt in (execution.recovery_attempts if execution else [])
        for repair in attempt.managed_input_repairs
    ]
    audit_logs_preserved = bool(
        first_failure
        and first_failure.stdout_file
        and first_failure.stderr_file
        and Path(first_failure.stdout_file).is_file()
        and Path(first_failure.stderr_file).is_file()
    )
    upstream_artifacts_preserved = all(path.is_file() for path in upstream_artifact_paths)
    cleanup_succeeded = (
        retry_count == 1
        and not partial_output.exists()
        and audit_logs_preserved
        and upstream_artifacts_preserved
    )
    observed_actions: set[str] = set()
    if event is not None and event.selected_action is not None:
        observed_actions.add(event.selected_action)
        if event.decision == "retry":
            observed_actions.add("retry")
    for attempt in execution.recovery_attempts if execution else []:
        observed_actions.update({attempt.action, "retry", "retry_current_step"})
        if attempt.action in {"retry_same_call", "rebuild_managed_workspace"}:
            observed_actions.add("clean_current_step_workspace")
        if attempt.action == "rebuild_managed_workspace":
            observed_actions.add("rebuild_report_staging")
    if not audit_logs_preserved and retry_count:
        observed_actions.add("delete_audit_logs")
    if not upstream_artifacts_preserved:
        observed_actions.add("delete_upstream_artifacts")
    forbidden_observed = sorted(
        observed_actions.intersection(case.get("forbidden_actions", []))
    )
    trace = {
        "failure_diagnosis": bool(diagnosis and diagnosis.reason and diagnosis.evidence),
        "ai_assessment": bool(successful_ai),
        "recovery_decision": bool(event and event.event_id and event.reason),
        "user_guidance": explanation_complete,
        "final_outcome": execution is not None,
        "confirmation_requirement": bool(event and event.requires_user_confirmation),
        "managed_input_repair": bool(repairs),
        "historical_artifact_identity": bool(
            repairs
            and repairs[0].repaired_source_path
            and Path(repairs[0].repaired_source_path) in upstream_artifact_paths
        ),
        "side_identity": bool(
            repairs
            and repairs[0].repaired_source_path
            and any(
                artifact.path == repairs[0].repaired_source_path
                and artifact.side == target_side
                for artifact in upstream_artifacts
            )
        ),
        "failed_step_identity": bool(
            first_failure
            and event
            and first_failure.step_index == event.step_index == target_call.step_index
            and first_failure.tool_name == target_call.tool_name
            and first_failure.side == target_call.side
        ),
        "workspace_cleanup": cleanup_succeeded,
        "audit_logs_preserved": audit_logs_preserved,
        "upstream_artifacts_preserved": upstream_artifacts_preserved,
        "recovery_attempt": retry_count == 1,
        "previous_error": bool(
            execution
            and execution.recovery_attempts
            and execution.recovery_attempts[0].previous_error_message
        ),
        "candidate_count": (
            is_ambiguous_artifact_case
            and sum(
                artifact.artifact_type
                == case["fault_injection"].get("artifact_type")
                for artifact in upstream_artifacts
            )
            == case["fault_injection"].get("available_compatible_artifacts")
            and not repairs
        ),
        "safety_validation_failure": bool(
            diagnosis
            and diagnosis.failure_stage == "preparation"
            and event
            and event.decision == "stop"
            and retry_count == 0
            and not repairs
        ),
    }
    repaired_reference = bool(
        repairs
        and repairs[0].previous_source_step_index == original_reference_step
        and repairs[0].source_step_index == 1
    ) if is_artifact_case else False
    bounded_adjustment = bool(
        execution
        and execution.execution_status == "succeeded"
        and retry_count == 1
        and (
            repaired_reference
            if is_artifact_case
            else cleanup_succeeded
        )
    )
    observation = {
        "diagnostic_category": diagnosis.category if diagnosis else None,
        "failure_stage": diagnosis.failure_stage if diagnosis else None,
        "recoverability": diagnosis.recoverability if diagnosis else None,
        "explanation_complete": explanation_complete,
        "decision": event.decision if event else None,
        "action": event.selected_action if event else None,
        "final_status": final_status,
        "scientific_change_required": diagnosis.scientific_change_required if diagnosis else None,
        "user_confirmation_required": bool(event and event.requires_user_confirmation),
        "forbidden_actions_observed": forbidden_observed,
        "observed_actions": sorted(observed_actions),
        "bounded_adjustment_applied": bounded_adjustment,
        "adjustment_count": retry_count,
        "retry_count": retry_count,
        "trace_items": trace,
    }
    details = {
        "runner_errors": runner_errors,
        "execution": execution.model_dump(mode="json") if execution else None,
        "diagnosis": diagnosis.model_dump(mode="json") if diagnosis else None,
        "failure_analysis": (
            failure_analysis.model_dump(mode="json") if failure_analysis else None
        ),
    }
    return observation, details, state


def run_case(
    case: dict[str, Any], repetition: int, args: argparse.Namespace, run_dir: Path
) -> dict[str, Any]:
    started = time.monotonic()
    case_dir = run_dir / case["case_id"] / f"repeat_{repetition:03d}"
    case_dir.mkdir(parents=True, exist_ok=False)
    errors: list[str] = []
    try:
        if case["scenario"] == "plan_validation":
            observation, execution_details, state = _run_plan_case(case, args, case_dir)
        elif case["scenario"] == "preflight":
            observation, execution_details, state = _run_preflight_case(case, args, case_dir)
        else:
            observation, execution_details, state = _run_tool_case(case, args, case_dir)
            errors.extend(execution_details.get("runner_errors", []))
        metrics, score_details = score_case(case, observation)
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
        observation = {}
        execution_details = {}
        state = {}
        metrics = {name: None for name in METRIC_NAMES}
        score_details = {}
    passed = not errors and all(
        value == 1.0 for value in metrics.values() if isinstance(value, (int, float))
    )
    return {
        "case_id": case["case_id"],
        "category_id": case["category_id"],
        "description": case["description"],
        "repetition": repetition,
        "passed": passed,
        "errors": errors,
        "metrics": metrics,
        "observation": observation,
        "score_details": score_details,
        "execution_details": execution_details,
        "usage": _usage(state),
        "duration_seconds": round(time.monotonic() - started, 3),
        "case_workdir": str(case_dir),
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _summary(
    results: list[dict[str, Any]], args: argparse.Namespace, *, include_ci: bool
) -> dict[str, Any]:
    statistics = repetition_statistics(results)
    if include_ci and args.bootstrap_iterations:
        intervals = cluster_bootstrap_ci(
            results, iterations=args.bootstrap_iterations, seed=args.seed + 1
        )
        for metric, interval in intervals.items():
            statistics[metric]["bootstrap_95_ci"] = interval
    by_category: dict[str, Any] = {}
    for category in sorted({item["category_id"] for item in results}):
        subset = [item for item in results if item["category_id"] == category]
        category_stats = repetition_statistics(subset)
        if include_ci and args.bootstrap_iterations:
            intervals = cluster_bootstrap_ci(
                subset,
                iterations=args.bootstrap_iterations,
                seed=args.seed + sum(ord(character) for character in category),
            )
            for metric, interval in intervals.items():
                category_stats[metric]["bootstrap_95_ci"] = interval
        by_category[category] = {
            "case_count": len({item["case_id"] for item in subset}),
            "metrics": category_stats,
        }
    return {
        "case_count": len({item["case_id"] for item in results}),
        "run_count": len(results),
        "passed_runs": sum(bool(item["passed"]) for item in results),
        "run_pass_rate": (
            sum(bool(item["passed"]) for item in results) / len(results)
            if results
            else None
        ),
        "metrics": statistics,
        "by_category": by_category,
        "usage": {
            key: sum(item["usage"][key] for item in results)
            for key in ("llm_calls", "input_tokens", "output_tokens", "total_tokens")
        },
        "total_duration_seconds": sum(item["duration_seconds"] for item in results),
    }


def _write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    fields = [
        "case_id",
        "category_id",
        "repetition",
        "passed",
        *METRIC_NAMES,
        "duration_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "case_id": result["case_id"],
                    "category_id": result["category_id"],
                    "repetition": result["repetition"],
                    "passed": result["passed"],
                    **result["metrics"],
                    "duration_seconds": result["duration_seconds"],
                }
            )


def main() -> int:
    args = build_parser().parse_args()
    if args.repeat < 1:
        raise SystemExit("--repeat must be positive")
    if args.bootstrap_iterations != 0 and args.bootstrap_iterations < 100:
        raise SystemExit("--bootstrap_iterations must be 0 or at least 100")
    catalog, all_cases = load_suite()
    cases = select_cases(all_cases, args)
    if args.validate_only:
        print("[validation passed] Test 3 cases and metric contracts are valid.")
        print(f"  total_cases={len(all_cases)}")
        for category in dict.fromkeys(case["category_id"] for case in all_cases):
            print(f"  {category}={sum(c['category_id'] == category for c in all_cases)}")
        return 0
    if not args.api_key:
        raise SystemExit("--api_key or ENZYWIZARD_API_KEY is required")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.workdir).expanduser().resolve() / f"run_{stamp}_{uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    output = Path(args.output).expanduser().resolve() if args.output else run_dir / "results.json"

    capability_state = {
        "config": _config(args, run_dir / "_model_capability"),
        "session_id": f"test3-contract-{uuid4().hex}",
        "current_turn_id": 1,
        "current_user_message": "Model contract capability probe.",
        "runtime_session": RuntimeSession(
            started_at=utc_now(), started_monotonic=time.monotonic()
        ),
    }
    Path(capability_state["config"].workdir).mkdir(parents=True, exist_ok=True)
    try:
        capability = check_model_capabilities(capability_state).model_dump(mode="json")
    except Exception as error:
        print(f"[model contract failed] {error}", flush=True)
        return 2

    results: list[dict[str, Any]] = []
    print(
        f"[Test 3] cases={len(cases)} repeat={args.repeat} model={args.model} "
        f"reasoning={args.reasoning_effort} profile={args.profile} seed={args.seed}",
        flush=True,
    )
    payload: dict[str, Any] = {}
    for case in cases:
        for repetition in range(1, args.repeat + 1):
            print(
                f"\n[{case['case_id']}] {repetition}/{args.repeat} "
                f"{case['category_id']}",
                flush=True,
            )
            print(f"  Fault: {case['description']}", flush=True)
            result = run_case(case, repetition, args, run_dir)
            results.append(result)
            print(f"  Metrics: {result['metrics']}", flush=True)
            if args.show_details:
                print(f"  Observation: {result['observation']}", flush=True)
            for error in result["errors"]:
                print(f"  Error: {error}", flush=True)
            payload = {
                "metadata": {
                    "test_id": catalog["test_id"],
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "model": args.model,
                    "base_url": args.base_url,
                    "reasoning_effort": args.reasoning_effort,
                    "ontology_mode": args.ontology_mode,
                    "repeat": args.repeat,
                    "profile": args.profile,
                    "seed": args.seed,
                    "selected_case_ids": [item["case_id"] for item in cases],
                    "bootstrap_iterations": args.bootstrap_iterations,
                },
                "model_capability": capability,
                "summary": _summary(results, args, include_ci=False),
                "results": results,
            }
            _atomic_json(output, payload)
            if args.stop_on_failure and not result["passed"]:
                payload["summary"] = _summary(results, args, include_ci=True)
                _atomic_json(output, payload)
                _write_csv(output.with_suffix(".csv"), results)
                return 1
    payload["summary"] = _summary(results, args, include_ci=True)
    _atomic_json(output, payload)
    _write_csv(output.with_suffix(".csv"), results)
    print(f"\n[complete] {output}", flush=True)
    return 0 if all(item["passed"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
