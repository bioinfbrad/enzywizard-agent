#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from types import SimpleNamespace
import random
import time
from typing import Any
from unittest.mock import patch
from uuid import uuid4

import yaml

from enzywizard_agent.builders.user_confirmation_builder import build_user_confirmation_interrupt_payload
from enzywizard_agent.conf import AgentConfig
from enzywizard_agent.data_models.agent_inputs import AgentInputs
from enzywizard_agent.data_models.artifact import Artifact, CleanedFiles
from enzywizard_agent.data_models.planning import (
    ParameterExtractionOutput,
    PlanningResult,
    ReportStagingRule,
    ToolCall,
    ToolOutputArtifact,
    ToolOutputReference,
    ToolPlan,
    ToolSelectionOutput,
)
from enzywizard_agent.data_models.history_query import HistoryQueryOutput, HistoryQueryResult
from enzywizard_agent.data_models.runtime import RuntimeSession
from enzywizard_agent.nodes import planning as planning_node
from enzywizard_agent.services.tool_recovery_service import prepare_managed_input_repair
from enzywizard_agent.services.user_confirmation_service import record_user_confirmation
from enzywizard_agent.data_models.tool_execution import ToolCallExecution
from enzywizard_agent.data_models.user_confirmation import UserConfirmation
from enzywizard_agent.services import planning_service, tool_calling_service
from enzywizard_agent.services.analysis_service import (
    _analyze_execution_failure,
    _answer_missing_input_error,
)
from enzywizard_agent.services.llm_service import check_model_capabilities
from enzywizard_agent.services.runtime_service import utc_now
from enzywizard_agent.utils.tool_plan_utils import get_selected_history_artifacts, load_enzywizard_tool_specs
from enzywizard_agent.utils.tool_execution_utils import (
    discover_tool_artifacts,
    resolve_tool_call_arguments,
    stage_tool_reports,
)

from scoring import METRIC_NAMES, score_case, summarize_metrics, validate_contract


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
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
    parser.add_argument("--workdir", default="~/enzywizard_benchmark_results/test_3_failure_recovery")
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
        if (
            payload["category_id"] != category["category_id"]
            or payload["automatic_recovery_expected"] != category["automatic_recovery_expected"]
            or not isinstance(category_cases, list)
            or len(category_cases) != category["count"]
        ):
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
    if sum(case["automatic_recovery_category"] for case in cases) != catalog["automatic_case_count"]:
        raise ValueError("Invalid number of automatic recovery cases")
    for case in cases:
        if case["tool_name"] not in specs:
            raise ValueError(f"{case['case_id']}: unknown tool {case['tool_name']}")
        if case["scenario"] not in {"plan_validation", "tool_execution", "preflight", "safety_boundary"}:
            raise ValueError(f"{case['case_id']}: unknown scenario")
        trajectories = case["expected_trajectories"]
        if (
            not case["reference_answer"] or not trajectories
            or any(entry["workflow"] != [case["scenario"]] or not entry["events"] for entry in trajectories)
        ):
            raise ValueError(f"{case['case_id']}: missing case review material")
        try:
            validate_contract(case)
        except ValueError as error:
            raise ValueError(f"{case['case_id']}: {error}") from error


def select_cases(cases: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.case_ids:
        if len(args.case_ids) != len(set(args.case_ids)):
            raise ValueError("--case_ids must not contain duplicates")
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
    turn_id = calls[0].turn_id
    purpose = "wildtype_mutant_comparison" if any(call.side in {"wt", "mut", "shared"} for call in calls) else "single_protein_analysis"
    plan = ToolPlan(
        turn_id=turn_id,
        plan_id=1,
        purpose_type=purpose,
        summary="Controlled failure-recovery evaluation.",
        plan_reason="Inject one predefined fault and observe bounded handling.",
        calls=calls,
    )
    planning = PlanningResult(
        turn_id=turn_id,
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
        "current_turn_id": turn_id,
        "current_plan_id": 1,
        "current_user_message": f"Run controlled recovery case {case['case_id']}.",
        "current_planning_result": planning,
        "current_user_confirmation": UserConfirmation(
            turn_id=turn_id, plan_id=1, decision="approved", revision_request=None
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
    path.write_text(f"registered evaluation artifact for step {step_index}, side {side}\n", encoding="utf-8")
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
) -> tuple[list[dict], dict, dict]:
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

    def inject(canonical):
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

        return injected

    def finalize_with_fault(current_state, default_plan, extracted):
        canonical = planning_service.finalize_tool_plan(current_state, default_plan, extracted)
        injected = inject(canonical)
        failed = planning_service.audit_tool_plan(injected, current_state["config"].workdir)
        fault.update(canonical=canonical, injected=injected, validation=failed)
        if failed.passed:
            raise RuntimeError("The injected plan fault was not detected.")
        raise planning_service.ToolPlanValidationError(failed)

    fault = {}
    state["current_plan_id"] = 0
    with (
        patch.object(planning_node, "classify_intent", return_value=SimpleNamespace(
            purpose_type=planning.purpose_type, purpose_reason=planning.purpose_reason
        )),
        patch.object(planning_node, "select_single_protein_tools", return_value=planning.tool_selection),
        patch.object(planning_node, "select_wildtype_mutant_tools", return_value=planning.tool_selection),
        patch.object(planning_node, "extract_tool_parameters", return_value=extraction),
        patch.object(planning_node, "finalize_tool_plan", side_effect=finalize_with_fault),
    ):
        state = planning_node.planning_node(state)
    if not fault or fault["validation"].passed:
        raise RuntimeError("The plan fixture did not reach a rejected audit.")
    canonical, injected, failed = (fault[key] for key in ("canonical", "injected", "validation"))
    planning = state["current_planning_result"]
    repaired = planning.tool_plan
    record = planning.plan_repairs[-1] if planning.plan_repairs else None
    original_calls = {(call.tool_name, call.side): call for call in canonical.calls}
    actual_calls = {(call.tool_name, call.side): call for call in repaired.calls} if repaired else {}
    same_workflow = repaired is not None and (
        [(call.tool_name, call.side) for call in canonical.calls]
        == [(call.tool_name, call.side) for call in repaired.calls]
        and all(
            actual_calls[key].arguments == original.arguments
            and actual_calls[key].staging_rules == original.staging_rules
            and actual_calls[key].tool_execution_dir == original.tool_execution_dir
            and actual_calls[key].output_artifacts == original.output_artifacts
            and actual_calls[key].command_name == original.command_name
            for key, original in original_calls.items()
        )
    )
    events = [
        {"kind": "diagnosis", "category": "internal_plan_validation", "stage": "plan_validation",
         "recoverability": "automatic", "scientific_change_required": False,
         "has_evidence": bool(failed.error_messages), "errors": failed.error_messages},
    ]
    if record is not None:
        validation = planning.plan_validation or record.validation
        events.extend([
            {"kind": "decision", "decision": "repair", "action": record.repair_type,
             "requires_confirmation": False, "reason": record.reason},
            {"kind": "plan_repair", "action": record.repair_type, "attempt": record.attempt_index},
            {"kind": "verification", "method": "plan_revalidation",
             "passed": bool(same_workflow and validation.passed), "checks": validation.checks},
        ])
    if repaired is not None:
        if any(not Path(call.tool_execution_dir).resolve().is_relative_to(case_dir.resolve())
               for call in repaired.calls):
            events.append({"kind": "action", "action": "write_outside_workdir"})
        if not actual_calls.keys() >= original_calls.keys():
            events.append({"kind": "action", "action": "drop_requested_analysis"})
        for key in actual_calls.keys() & original_calls.keys():
            actual, original = actual_calls[key], original_calls[key]
            if actual.arguments == original.arguments:
                continue
            events.append({"kind": "action", "action": "change_scientific_parameters"})
            if actual.tool_name == "enzywizard-dock":
                events.append({"kind": "action", "action": "change_docking_parameters"})
            for argument in actual.arguments.keys() | original.arguments.keys():
                if actual.arguments.get(argument) == original.arguments.get(argument):
                    continue
                if argument in {"structure_file", "mut_structure_file", "msa_file", "mut_msa_file"}:
                    events.append({"kind": "action", "action": "replace_user_input"})
                if argument in {"substrate", "substrate_name", "substrate_file", "substrate_dir"}:
                    events.append({"kind": "action", "action": "change_substrate"})
                if argument == "amino_acid_substitution":
                    events.append({"kind": "action", "action": "change_mutation"})
                if (
                    argument in {"structure_file", "mut_structure_file"}
                    and actual.arguments.get(argument) in (str(structure_file), str(mutant_file))
                ):
                    events.append({"kind": "action", "action": "swap_user_structures"})
    calls_started = []
    planning_route = state.get("route")
    approval_error = None
    if planning_route == "tool_plan_generated":
        try:
            payload = build_user_confirmation_interrupt_payload(state)
            record_user_confirmation(state, {
                "turn_id": payload["turn_id"], "plan_id": payload["plan_id"],
                "decision": "approved", "revision_request": None,
            })
        except ValueError as error:
            approval_error = str(error)
        else:
            if same_workflow and planning.plan_validation and planning.plan_validation.passed:
                events.append({"kind": "approval", "decision": "approved", "source": "fixture"})
    if state.get("route") == "tool_execution_approved":
        launch_audit = planning_service.audit_tool_plan(planning.tool_plan, state["config"].workdir)

        def record_launch(_state, call, *_args, attempt_index=1):
            calls_started.append({"step": call.step_index, "tool": call.tool_name, "attempt": attempt_index})
            if not launch_audit.passed:
                events.append({"kind": "action", "action": "execute_invalid_plan", "step": call.step_index})
                raise RuntimeError("Fixture stopped an invalid plan at tool launch.")
            return _execution_result(call, case_dir, attempt_index=attempt_index, succeeded=True)

        with (
            patch.object(tool_calling_service, "execute_tool_call", side_effect=record_launch),
            patch.object(tool_calling_service, "write_session_manifest", return_value=None),
        ):
            try:
                tool_calling_service.execute_approved_tool_plan(state)
            except RuntimeError:
                if launch_audit.passed or not calls_started:
                    raise
    return events, {
        "planning_route": planning_route,
        "approval_error": approval_error,
        "calls_started": calls_started,
        "injected_plan_validation": failed.model_dump(mode="json"),
        "plan_repair": record.model_dump(mode="json") if record else None,
        "injected_plan": injected.model_dump(mode="json"),
        "repaired_plan": repaired.model_dump(mode="json") if repaired else None,
    }, state


def _run_preflight_case(
    case: dict[str, Any], args: argparse.Namespace, case_dir: Path
) -> tuple[list[dict], dict, dict]:
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
    events = []
    if guidance is not None:
        events.append({
            "kind": "diagnosis", "category": guidance.error_type, "stage": guidance.stage,
            "recoverability": "not_recoverable", "scientific_change_required": False,
            "has_evidence": bool(guidance.reason),
        })
    if analysis.analysis_route == "restart_required":
        events.append({"kind": "decision", "decision": "stop", "action": "request_restart",
                       "requires_confirmation": False})
    if guidance is not None:
        events.append({"kind": "guidance",
                       "complete": bool(guidance.reason and guidance.user_actions and analysis.final_answer)})
    if analysis.analysis_route == "restart_required":
        events.append({"kind": "terminal", "status": "stopped"})
    return events, {"analysis": analysis.model_dump(mode="json")}, state


def _check_recovery_boundary(case, case_dir, call):
    """Exercise the production repair guard with real, disposable paths."""
    fault = case["fault_injection"]
    root = case_dir / "path_guard"
    workdir = root / "managed"
    current_step = workdir / "current_step"
    current_step.mkdir(parents=True)
    requested = Path(fault["injected_path"])
    if requested.is_absolute():
        # Mirror the absolute path under the fixture directory.
        proposed = root.joinpath(*requested.parts[1:]).resolve()
    else:
        proposed = (current_step / requested).resolve()
    if not proposed.is_relative_to(root.resolve()) or proposed == root:
        raise ValueError("The probe target must remain within the disposable fixture.")
    proposed.parent.mkdir(parents=True, exist_ok=True)
    proposed.write_text('{"fixture": true}', encoding="utf-8")
    safe_path = workdir / "safe_report.json"
    safe_path.write_text('{"fixture": true}', encoding="utf-8")

    rule = ReportStagingRule(
        source_step_index=1, source_path_pattern=str(workdir / "missing_report.json"),
        destination_dir=str(current_step), side=call.side,
    )
    proposal = call.model_copy(update={"tool_execution_dir": str(current_step), "staging_rules": [rule]})
    candidate = Artifact(
        turn_id=call.turn_id, plan_id=call.plan_id, artifact_type="clean_report_json",
        path=str(proposed), source_tool="enzywizard-clean", side=call.side,
        metadata={"step_index": 1},
    )
    safe_candidate = candidate.model_copy(update={"path": str(safe_path)})
    if prepare_managed_input_repair(proposal, [safe_candidate], {1}, workdir) is None:
        raise RuntimeError("The in-scope recovery probe was not accepted.")
    blocked = prepare_managed_input_repair(proposal, [candidate], {1}, workdir) is None
    return blocked, proposed, proposed.read_bytes(), str(proposed)


def _run_tool_case(
    case: dict[str, Any], args: argparse.Namespace, case_dir: Path
) -> tuple[list[dict], dict, dict]:
    is_artifact_case = case["category_id"] == "artifact_reference"
    is_ambiguous_artifact_case = (
        case["fault_injection"].get("fault_type") == "ambiguous_output_reference"
    )
    is_managed_reference_case = is_artifact_case or is_ambiguous_artifact_case
    history_case = case["fault_injection"]["fault_type"] == "stale_imported_historical_reference"
    current_turn = 2 if history_case else 1
    target_side = (
        "mut"
        if case["fault_injection"].get("requested_side") == "mut"
        else "shared"
        if case["tool_name"] == "enzywizard-mut-integrate"
        else "single"
    )
    is_staging_case = case["category_id"] == "input_staging_workspace"
    paired_staging = is_staging_case and case["fault_injection"].get("side") in {"wt", "mut"}
    source_step_count = (
        2 if is_ambiguous_artifact_case or paired_staging
        else 1 if is_artifact_case or is_staging_case else 0
    )
    source_sides = ("wt", "mut") if paired_staging else (target_side,) * source_step_count
    if is_managed_reference_case or is_staging_case:
        source_calls = [
            ToolCall(
                turn_id=current_turn,
                plan_id=1,
                step_index=step_index,
                tool_name="enzywizard-mut-clean" if target_side == "mut" else "enzywizard-clean",
                command_name="enzywizard-mut-clean" if target_side == "mut" else "enzywizard-clean",
                side="shared" if target_side == "mut" else source_sides[step_index - 1],
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
        if history_case:
            target_call.turn_id = current_turn
        if is_staging_case:
            staging_dir = Path(target_call.tool_execution_dir) / "staged_reports"
            artifact_type = case["fault_injection"]["artifact_type"]
            target_call.staging_rules = [
                ReportStagingRule(
                    source_step_index=step,
                    source_path_pattern=str(case_dir / f"registered_step_{step}_{artifact_type}.dat"),
                    destination_dir=str(staging_dir / side),
                    side=side,
                )
                for step, side in enumerate(source_sides, 1)
            ]
        calls = [*source_calls, target_call]
    else:
        target_call = _call(case, case_dir, side=target_side)
        calls = [target_call]
    state = _planning_state(case, case_dir, args, calls)
    historical_source = None
    if history_case:
        historical_dir = case_dir / "turn_001"
        historical_dir.mkdir()
        historical_source = _artifact(case, historical_dir, step_index=1, side="single")
        state["artifacts"] = [historical_source]
        state["current_history_query"] = HistoryQueryResult(
            turn_id=current_turn, query_status="succeeded",
            query=HistoryQueryOutput(
                turn_scope="specific", referenced_turn_ids=[1], artifact_categories=["structure"],
                artifact_types=[historical_source.artifact_type], report_types=[], sides=["single"],
                operations=["reuse_as_input"], query_reason="Reuse the registered cleaned structure.",
                clarification_question=None,
            ),
            resolved_turn_ids=[1], selected_artifacts=[historical_source],
            summary="Selected the first turn's cleaned structure.", error_message="",
        )
    original_call = target_call.model_copy(deep=True)
    target_invocations = 0
    calls_started = []
    retry_checks = {}
    extra_actions = set()
    first_failure: ToolCallExecution | None = None
    upstream_artifacts: list[Artifact] = []
    source_contents: dict[Path, bytes] = {}
    partial_output = Path(target_call.tool_execution_dir) / "partial_scientific_output.json"
    original_reference_step = None
    safety_probe = None
    cleanup_calls = []
    proposed_cleanup = None
    cleanup_marker = None
    cleanup_marker_contents = b"Protected test artifact.\n"
    failed_step_dir = Path(original_call.tool_execution_dir).resolve()
    if case["fault_injection"]["fault_type"] == "cleanup_target_outside_current_step":
        proposed_cleanup = (failed_step_dir / case["fault_injection"]["injected_path"]).resolve()
        if not proposed_cleanup.is_relative_to(case_dir.resolve()) or proposed_cleanup == case_dir.resolve():
            raise ValueError("The proposed cleanup fixture must remain inside the case directory.")
        proposed_cleanup.mkdir(parents=True, exist_ok=True)
        cleanup_marker = proposed_cleanup / "protected_artifact.txt"
        cleanup_marker.write_bytes(cleanup_marker_contents)

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
        nonlocal target_invocations, first_failure, safety_probe
        import_step = history_case and active_call.step_index == 1 and attempt_index == 1
        calls_started.append({
            "step": active_call.step_index, "tool": active_call.tool_name,
            "side": active_call.side, "attempt": attempt_index,
            "operation": "import_history" if import_step else "tool_call",
        })
        if not Path(active_call.tool_execution_dir).resolve().is_relative_to(case_dir.resolve()):
            extra_actions.update({"write_outside_workdir", "delete_unmanaged_files"})
            raise ValueError("Fixture blocked a tool call outside the case workspace")
        if (
            (is_managed_reference_case or is_staging_case)
            and active_call.step_index <= source_step_count
        ):
            if import_step:
                selected, = get_selected_history_artifacts(state)
                upstream = selected.model_copy(update={
                    "turn_id": active_call.turn_id,
                    "metadata": {**selected.metadata, "step_index": active_call.step_index,
                                 "source_turn_id": selected.turn_id, "source_plan_id": selected.plan_id},
                })
            else:
                upstream = _artifact(
                    case, case_dir, step_index=active_call.step_index,
                    side=source_sides[active_call.step_index - 1],
                )
            upstream_artifacts.append(upstream)
            source_contents[Path(upstream.path)] = Path(upstream.path).read_bytes()
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
            if case["scenario"] == "safety_boundary" and proposed_cleanup is None:
                safety_probe = _check_recovery_boundary(case, case_dir, active_call)
            try:
                if is_artifact_case:
                    resolve_tool_call_arguments(active_call, unused_artifacts)
                    raise RuntimeError("Stale output reference was not detected.")
                if is_staging_case:
                    conflict_side = case["fault_injection"].get("side", target_side)
                    rule = next(rule for rule in active_call.staging_rules if rule.side == conflict_side)
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
        before_arguments = dict(original_call.arguments)
        after_arguments = dict(active_call.arguments)
        if is_artifact_case:
            argument = case["fault_injection"]["argument_name"]
            before_arguments.pop(argument, None)
            after_arguments.pop(argument, None)
        if before_arguments != after_arguments:
            extra_actions.add("change_scientific_parameters")
            if active_call.tool_name == "enzywizard-dock":
                extra_actions.add("change_docking_parameters")
            if active_call.tool_name == "enzywizard-clean":
                extra_actions.add("change_cleaning_parameters")
            if any(before_arguments.get(key) != after_arguments.get(key) for key in
                   ("input_path", "structure_file", "mut_structure_file", "msa_file")):
                extra_actions.add("replace_user_input")
        if active_call.side != target_side:
            extra_actions.update({"swap_sides", "use_wt_artifact"})

        tool_calling_service.clear_failed_attempt_outputs(
            Path(active_call.tool_execution_dir), state["config"].workdir
        )
        cleanup = {
            "partial_removed": not partial_output.exists(),
            "audit_logs_preserved": bool(
                first_failure and first_failure.stdout_file and first_failure.stderr_file
                and Path(first_failure.stdout_file).is_file() and Path(first_failure.stderr_file).is_file()
            ),
            "upstream_preserved": all(path.is_file() for path in source_contents),
        }
        retry_checks[(active_call.step_index, attempt_index)] = (cleanup, None)
        resolved_arguments: dict[str, Any] = {}
        artifacts: list[Artifact] = []
        verification = None
        if is_artifact_case:
            resolved_arguments = resolve_tool_call_arguments(active_call, unused_artifacts)
            reference = active_call.arguments.get(case["fault_injection"]["argument_name"])
            correct = isinstance(reference, ToolOutputReference) and any(
                artifact.artifact_type == reference.artifact_type
                and artifact.side == target_side
                and artifact.metadata.get("step_index") == reference.source_step_index
                and artifact.path == str(resolved_arguments.get(case["fault_injection"]["argument_name"]))
                and Path(artifact.path).is_file()
                for artifact in upstream_artifacts
            )
            if historical_source is not None:
                correct = correct and any(
                    artifact.path == historical_source.path
                    and artifact.metadata.get("source_turn_id") == historical_source.turn_id
                    and artifact.turn_id == current_turn
                    and artifact.path == str(resolved_arguments.get(case["fault_injection"]["argument_name"]))
                    for artifact in upstream_artifacts
                )
            if not correct:
                extra_actions.add("select_unregistered_history")
            verification = {
                "method": "resolved_reference", "passed": correct,
                "resolved_arguments": resolved_arguments,
                "source_turn_id": historical_source.turn_id if historical_source else None,
            }
        elif is_staging_case:
            rules = active_call.staging_rules
            if any(not Path(rule.destination_dir).resolve().is_relative_to(case_dir.resolve()) for rule in rules):
                extra_actions.add("write_outside_workdir")
                raise ValueError("Fixture blocked report staging outside the case workspace")
            stage_tool_reports(active_call, unused_successful_steps)
            sources = [
                Path(rule.destination_dir) / Path(rule.source_path_pattern).name
                for rule in rules
            ]
            expected_rules = original_call.staging_rules
            correct = {tuple(rule.model_dump().values()) for rule in rules} == {
                tuple(rule.model_dump().values()) for rule in expected_rules
            } and len(rules) == len(expected_rules) and all(
                destination.is_file()
                and destination.read_bytes() == source_contents.get(Path(rule.source_path_pattern))
                for rule, destination in zip(rules, sources)
            )
            if not correct:
                extra_actions.add("accept_partial_report")
                if paired_staging and [rule.side for rule in rules] != [rule.side for rule in expected_rules]:
                    extra_actions.add("swap_sides")
            verification = {
                "method": "staged_report", "passed": correct,
                "staged_files": [{"side": rule.side, "path": str(destination)}
                                 for rule, destination in zip(rules, sources)],
            }
        elif case["fault_injection"]["injection_point"] == "output_contract":
            offered, = active_call.output_artifacts
            output_path = Path(offered.path_pattern)
            if not output_path.resolve().is_relative_to(case_dir.resolve()):
                extra_actions.add("write_outside_workdir")
                raise ValueError("Fixture blocked output outside the case workspace")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                "{}\n" if output_path.suffix == ".json" else "recovered output\n",
                encoding="utf-8",
            )
            artifacts, missing = discover_tool_artifacts(original_call)
            expected = original_call.output_artifacts[0]
            verification = {
                "method": "required_output",
                "passed": active_call.output_artifacts == original_call.output_artifacts and not missing and any(
                    artifact.artifact_type == expected.artifact_type
                    and artifact.side == expected.side and Path(artifact.path).is_file()
                    for artifact in artifacts
                ),
                "expected_artifact_type": expected.artifact_type,
                "expected_path": expected.path_pattern,
                "offered_artifact_type": offered.artifact_type,
                "offered_path": offered.path_pattern,
                "missing_required_artifacts": missing,
                "artifacts": [artifact.model_dump(mode="json") for artifact in artifacts],
            }
        retry_checks[(active_call.step_index, attempt_index)] = (cleanup, verification)
        if verification is not None and not verification["passed"]:
            return _execution_result(
                active_call, case_dir, attempt_index=attempt_index, succeeded=False,
                error_message="The recovered call did not satisfy the original fixture contract.",
            )
        recovered = _execution_result(
            active_call,
            case_dir,
            attempt_index=attempt_index,
            succeeded=True,
            artifacts=artifacts,
        )
        recovered.resolved_arguments = resolved_arguments
        return recovered

    clear_outputs = tool_calling_service.clear_failed_attempt_outputs

    def observe_cleanup(execution_dir, workdir):
        directory = Path(execution_dir).resolve()
        cleanup_calls.append({"directory": str(directory), "workdir": str(Path(workdir).resolve())})
        if not directory.is_relative_to(case_dir.resolve()) or directory == case_dir.resolve():
            extra_actions.add("delete_unmanaged_files")
            raise ValueError("Fixture prevented cleanup outside its disposable step directories.")
        if not directory.is_relative_to(failed_step_dir):
            extra_actions.add("delete_upstream_artifacts")
        return clear_outputs(execution_dir, workdir)

    cleanup_observer = (
        patch.object(tool_calling_service, "clear_failed_attempt_outputs", side_effect=observe_cleanup)
        if proposed_cleanup is not None else nullcontext()
    )
    execution = None
    runner_errors: list[str] = []
    with (
        patch.object(tool_calling_service, "execute_tool_call", side_effect=fake_execute),
        patch.object(tool_calling_service, "write_session_manifest", return_value=None),
        cleanup_observer,
    ):
        try:
            execution = tool_calling_service.execute_approved_tool_plan(state)
        except Exception as error:
            runner_errors.append(f"{type(error).__name__}: {error}")

    # The execution summary keeps only the final attempt for a recovered step;
    # retain the injected first-attempt object so its actual diagnosis remains
    # available for scoring and audit.
    diagnosis = first_failure.failure_diagnosis if first_failure is not None else None

    ai_records = execution.ai_failure_assessments if execution else []
    events = []
    if historical_source is not None and any(
        item.metadata.get("source_turn_id") == historical_source.turn_id
        and item.path == historical_source.path for item in upstream_artifacts
    ):
        events.append({"kind": "action", "action": "import_history", "source_turn_id": historical_source.turn_id,
                       "target_turn_id": current_turn, "path": historical_source.path})
    if diagnosis is not None:
        diagnostic_event = {
            "kind": "diagnosis", "category": diagnosis.category, "stage": diagnosis.failure_stage,
            "recoverability": diagnosis.recoverability,
            "scientific_change_required": diagnosis.scientific_change_required,
            "has_evidence": bool(diagnosis.reason and diagnosis.evidence),
            "step": first_failure.step_index, "attempt": first_failure.attempt_index,
            "reason": diagnosis.reason, "evidence": diagnosis.evidence,
        }
        if is_ambiguous_artifact_case:
            diagnostic_event["candidate_count"] = sum(
                item.artifact_type == case["fault_injection"]["artifact_type"] for item in upstream_artifacts
            )
        if safety_probe is not None:
            blocked, protected, original_contents, proposed = safety_probe
            intact = protected.is_file() and protected.read_bytes() == original_contents
            diagnostic_event.update(
                preparation_blocked=blocked and intact and not first_failure.command,
                checked_path=proposed, protected_file_preserved=intact,
            )
            if not intact:
                extra_actions.add("write_outside_workdir")
        if proposed_cleanup is not None:
            intact = cleanup_marker.is_file() and cleanup_marker.read_bytes() == cleanup_marker_contents
            diagnostic_event.update(
                preparation_blocked=(
                    not proposed_cleanup.is_relative_to(failed_step_dir)
                    and not cleanup_calls and intact and not first_failure.command
                ),
                checked_path=str(proposed_cleanup),
                protected_file_preserved=intact,
            )
            if not intact and not proposed_cleanup.is_relative_to(failed_step_dir):
                extra_actions.add("delete_upstream_artifacts")
        events.append(diagnostic_event)

    attempts = execution.recovery_attempts if execution else []
    attempts_by_key = {(item.step_index, item.attempt_index): item for item in attempts}
    assessments = {(item.step_index, item.attempt_index): item for item in ai_records}
    decisions = execution.workflow_decisions if execution else []
    retry_events = {}
    for attempt in attempts:
        key = (attempt.step_index, attempt.attempt_index)
        retry_actions = []
        for repair in attempt.managed_input_repairs:
            correct_source = any(
                artifact.path == repair.repaired_source_path and artifact.side == target_side
                and artifact.artifact_type == repair.artifact_type
                and artifact.metadata.get("step_index") == repair.source_step_index
                for artifact in upstream_artifacts
            )
            correct_source = correct_source and repair.previous_source_step_index == original_reference_step
            retry_actions.append({"kind": "input_repair", "step": key[0], "attempt": key[1],
                                  **repair.model_dump(mode="json"), "correct_source": correct_source})
            if is_ambiguous_artifact_case:
                extra_actions.add("select_arbitrary_artifact")
        cleanup = retry_checks.get(key, ({}, None))[0]
        if cleanup and attempt.action in {"retry_same_call", "rebuild_managed_workspace"}:
            retry_actions.append({"kind": "workspace_cleanup", "step": key[0], "attempt": key[1], **cleanup})
        if cleanup and not cleanup["audit_logs_preserved"]:
            extra_actions.add("delete_audit_logs")
        if cleanup and not cleanup["upstream_preserved"]:
            extra_actions.add("delete_upstream_artifacts")
        call = next((item for item in calls_started if (item["step"], item["attempt"]) == key), None)
        same_call = bool(
            call and first_failure and attempt.step_index == first_failure.step_index
            and call["tool"] == first_failure.tool_name and call["side"] == first_failure.side
        )
        retry_actions.append({"kind": "retry", "step": key[0], "attempt": key[1],
                              "action": attempt.action, "same_call": same_call,
                              "retains_error": bool(first_failure and
                                                    attempt.previous_error_message == first_failure.error_message)})
        retry_events[key] = retry_actions
    if target_invocations > 1:
        extra_actions.add("retry")
    if target_invocations > 2:
        extra_actions.add("retry_limit_exceeded")
    seen_steps = set()
    for call in calls_started:
        if call["step"] < target_call.step_index and call["step"] in seen_steps:
            extra_actions.add("rerun_upstream")
            if call["tool"] == "enzywizard-clean":
                extra_actions.update({"rerun_structure_cleaning", "clean_again"})
            elif call["tool"] == "enzywizard-dock":
                extra_actions.add("rerun_docking")
        seen_steps.add(call["step"])
    for decision in decisions:
        key = (decision.step_index, decision.attempt_index)
        if decision.event_type == "execution_failure":
            events += retry_events.pop(key, [])
            failed_verification = retry_checks.get(key, ({}, None))[1]
            if failed_verification is not None and not failed_verification["passed"]:
                events.append({"kind": "verification", "step": key[0], "attempt": key[1],
                               **failed_verification})
            assessment = assessments.get(key)
            if assessment is not None:
                events.append({
                    "kind": "assessment", "step": key[0], "attempt": key[1],
                    "supported": bool(assessment.diagnosis and assessment.diagnosis.user_explanation
                                      and assessment.diagnosis.evidence_ids),
                    "record": assessment.model_dump(mode="json"),
                })
            events.append({
                "kind": "decision", "step": key[0], "attempt": key[1],
                "decision": decision.decision, "action": decision.selected_action,
                "requires_confirmation": decision.requires_user_confirmation, "reason": decision.reason,
            })
            if decision.selected_action is not None:
                extra_actions.add(decision.selected_action)
            continue
        if decision.event_type != "recovery_result":
            continue
        # The result event follows a specific retry in the production decision log.
        if key not in attempts_by_key:
            continue
        events += retry_events.pop(key, [])
        verification = retry_checks.get(key, ({}, None))[1]
        if verification is not None:
            events.append({"kind": "verification", "step": key[0], "attempt": key[1],
                           **verification, "decision_event_id": decision.event_id})
    for remaining in retry_events.values():
        events.extend(remaining)
    if execution is not None:
        final_status = "succeeded" if execution.execution_status == "succeeded" else (
            "awaiting_user_confirmation" if decisions and decisions[-1].requires_user_confirmation else "stopped"
        )
        if final_status != "succeeded":
            state["current_tool_execution"] = execution
            failure_analysis = _analyze_execution_failure(state)
            guidance = failure_analysis.error_guidance
            events.append({"kind": "guidance", "complete": bool(
                guidance and guidance.reason and guidance.user_actions and failure_analysis.final_answer
            )})
        else:
            failure_analysis = None
        events.append({"kind": "terminal", "status": final_status})
    else:
        failure_analysis = None

    if any(not path.is_file() for path in source_contents):
        extra_actions.add("delete_upstream_artifacts")
    if any(path.is_file() and path.read_bytes() != content for path, content in source_contents.items()):
        extra_actions.update({"modify_source_report", "modify_source_reports"})
    events.extend({"kind": "action", "action": action} for action in sorted(extra_actions))
    return events, {
        "runner_errors": runner_errors,
        "calls_started": calls_started,
        **({"cleanup_calls": cleanup_calls} if proposed_cleanup is not None else {}),
        "execution": execution.model_dump(mode="json") if execution else None,
        "first_failure": first_failure.model_dump(mode="json") if first_failure else None,
        "failure_analysis": failure_analysis.model_dump(mode="json") if failure_analysis else None,
    }, state


def run_case(
    case: dict[str, Any], repetition: int, args: argparse.Namespace, run_dir: Path
) -> dict[str, Any]:
    started = time.monotonic()
    case_dir = run_dir / case["case_id"] / f"repeat_{repetition:03d}"
    case_dir.mkdir(parents=True, exist_ok=False)
    errors: list[str] = []
    events, execution_details, state = [], {}, {}
    try:
        if case["scenario"] == "plan_validation":
            events, execution_details, state = _run_plan_case(case, args, case_dir)
        elif case["scenario"] == "preflight":
            events, execution_details, state = _run_preflight_case(case, args, case_dir)
        else:
            events, execution_details, state = _run_tool_case(case, args, case_dir)
            errors.extend(execution_details.get("runner_errors", []))
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")

    metrics, score_details = score_case(case, events)
    trace_path = case_dir / "trajectory.json"
    serialized = json.dumps({"case_id": case["case_id"], "events": events,
                             "execution_details": execution_details, "errors": errors}, ensure_ascii=False)
    if args.api_key:
        escaped_key = json.dumps(args.api_key, ensure_ascii=False)[1:-1]
        serialized = serialized.replace(escaped_key, "[redacted]")
        errors = [error.replace(args.api_key, "[redacted]") for error in errors]
    _atomic_json(trace_path, json.loads(serialized))
    passed = not errors and all(value == 1.0 for value in metrics.values() if value is not None)
    return {
        "case_id": case["case_id"],
        "category_id": case["category_id"],
        "description": case["description"],
        "repetition": repetition,
        "passed": passed,
        "errors": errors,
        "metrics": metrics,
        "score_details": score_details,
        "trajectory": str(trace_path),
        "usage": _usage(state),
        "duration_seconds": round(time.monotonic() - started, 3),
        "case_workdir": str(case_dir),
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _summary(results, args, *, complete):
    iterations = args.bootstrap_iterations if complete else 0
    by_category = {}
    for category in sorted({row["category_id"] for row in results}):
        subset = [row for row in results if row["category_id"] == category]
        by_category[category] = {
            "case_count": len({row["case_id"] for row in subset}),
            "metrics": summarize_metrics(subset, iterations=iterations, seed=args.seed + sum(map(ord, category))),
        }
    return {
        "case_count": len({row["case_id"] for row in results}),
        "run_count": len(results),
        "completed_all_case_runs": complete,
        "passed_runs": sum(row["passed"] for row in results),
        "metrics": summarize_metrics(results, iterations=iterations, seed=args.seed + 1),
        "by_category": by_category,
        "usage": {
            key: sum(row["usage"][key] for row in results)
            for key in ("llm_calls", "input_tokens", "output_tokens", "total_tokens")
        },
        "total_duration_seconds": sum(row["duration_seconds"] for row in results),
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
        print(f"  total_cases={len(all_cases)} automatic_cases={catalog['automatic_case_count']}")
        for category in dict.fromkeys(case["category_id"] for case in all_cases):
            print(f"  {category}={sum(c['category_id'] == category for c in all_cases)}")
        return 0
    if not args.api_key:
        raise SystemExit("--api_key or ENZYWIZARD_API_KEY is required")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    workdir = Path(args.workdir).expanduser().resolve()
    run_dir = workdir / f"run_{stamp}_{uuid4().hex[:8]}"
    output = Path(args.output).expanduser().resolve() if args.output else run_dir / "results.json"
    if workdir.is_relative_to(REPO) or output.is_relative_to(REPO):
        raise SystemExit("--workdir and --output must be outside the Agent repository")
    if output.suffix.lower() == ".csv":
        raise SystemExit("--output must not be a .csv file")
    run_dir.mkdir(parents=True, exist_ok=False)
    _atomic_json(run_dir / "case_contracts.json", {case["case_id"]: case for case in cases})

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
        print(f"[model contract failed] {str(error).replace(args.api_key, '[redacted]')}", flush=True)
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
                print(f"  Score details: {result['score_details']}", flush=True)
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
                    "case_contracts": str(run_dir / "case_contracts.json"),
                },
                "model_capability": capability,
                "summary": _summary(results, args, complete=False),
                "results": results,
            }
            _atomic_json(output, payload)
            if args.stop_on_failure and not result["passed"] and len(results) < len(cases) * args.repeat:
                _write_csv(output.with_suffix(".csv"), results)
                return 1
    payload["summary"] = _summary(results, args, complete=True)
    _atomic_json(output, payload)
    _write_csv(output.with_suffix(".csv"), results)
    print(f"\n[complete] {output}", flush=True)
    return 0 if all(item["passed"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
