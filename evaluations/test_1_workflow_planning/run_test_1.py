#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any, get_args
from uuid import uuid4

from langgraph.graph import END, StateGraph
from langgraph.types import Command
import yaml

from enzywizard_agent.conf import AgentConfig
from enzywizard_agent.data_models.agent_inputs import AgentInputs
from enzywizard_agent.data_models.artifact import Artifact, CleanedFiles
from enzywizard_agent.data_models.planning import (
    ParameterExtractionOutput,
    PlanningResult,
    ToolCall,
    ToolPlan,
)
from enzywizard_agent.data_models.runtime import RuntimeSession
from enzywizard_agent.data_models.request_routing import RequestActionType, RequestType
from enzywizard_agent.data_models.tool_execution import ToolCallExecution, ToolExecution
from enzywizard_agent.data_models.user_confirmation import UserConfirmation
from enzywizard_agent.data_models.user_message import UserMessage
from enzywizard_agent.nodes.history_query import history_query_node
from enzywizard_agent.nodes.planning import planning_node
from enzywizard_agent.nodes.request_routing import request_routing_node
from enzywizard_agent.nodes.tool_calling import tool_calling_node
from enzywizard_agent.nodes.user_confirmation_interrupt import (
    user_confirmation_interrupt_node,
)
from enzywizard_agent.services.llm_service import check_model_capabilities
from enzywizard_agent.services.runtime_service import utc_now
from enzywizard_agent.services.user_confirmation_service import (
    record_user_confirmation,
)
from enzywizard_agent.state import AgentState
from enzywizard_agent.utils.CLI_utils import get_single_interrupt_payload
from enzywizard_agent.utils.checkpoint_utils import create_in_memory_checkpointer
from enzywizard_agent.utils.IO_utils import create_session_id
from enzywizard_agent.utils.tool_plan_utils import load_enzywizard_tool_specs

from scoring import (
    PROVENANCE_REQUIREMENTS,
    cluster_bootstrap_ci,
    flatten_required_artifacts,
    repetition_statistics,
    score_e2e,
    score_pcvr,
    score_ppcr,
    score_rac,
    score_tsa,
    score_wovr,
)


ROOT = Path(__file__).resolve().parent
CATALOG_PATH = ROOT / "case_catalog.yaml"
INDEX_PATH = ROOT / "cases" / "index.yaml"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-6-luna"
DEFAULT_REASONING_EFFORT = "medium"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test 1: scientific-goal translation and dynamic workflows."
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
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--case_ids", nargs="*")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--show_plans", action="store_true")
    parser.add_argument("--stop_on_failure", action="store_true")
    parser.add_argument(
        "--bootstrap_iterations",
        type=int,
        default=1000,
        help="Case-cluster bootstrap replicates; use 0 to omit confidence intervals.",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Validate cases and resources without calling an LLM.",
    )
    parser.add_argument(
        "--workdir", default="~/enzywizard-evaluation-test-1"
    )
    parser.add_argument("--output")
    return parser


def _route(state: AgentState) -> str:
    route = state.get("route")
    if not route:
        raise ValueError("Agent state did not provide a route")
    return route


def build_test_graph():
    graph = StateGraph(AgentState)
    graph.add_node("request_routing", request_routing_node)
    graph.add_node("history_query", history_query_node)
    graph.add_node("planning", planning_node)
    graph.add_node("confirmation", user_confirmation_interrupt_node)
    graph.add_node("tool_calling", tool_calling_node)
    graph.set_entry_point("request_routing")
    graph.add_conditional_edges(
        "request_routing",
        _route,
        {
            "request_to_planning": "planning",
            "request_to_history_query": "history_query",
            "request_to_analysis": END,
        },
    )
    graph.add_conditional_edges(
        "history_query",
        _route,
        {"history_to_planning": "planning", "history_to_analysis": END},
    )
    graph.add_conditional_edges(
        "planning",
        _route,
        {"tool_plan_generated": "confirmation", "tool_unused": END},
    )
    graph.add_conditional_edges(
        "confirmation",
        _route,
        {
            "tool_execution_approved": "tool_calling",
            "tool_plan_revision_requested": "request_routing",
            "tool_execution_rejected": END,
        },
    )
    graph.add_edge("tool_calling", END)
    return graph.compile(checkpointer=create_in_memory_checkpointer())


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML object in {path}")
    return value


def load_suite() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    catalog = _load_yaml(CATALOG_PATH)
    index = _load_yaml(INDEX_PATH)
    cases: list[dict[str, Any]] = []
    for category in index["categories"]:
        payload = _load_yaml(ROOT / "cases" / category["file"])
        category_cases = payload.get("cases")
        if len(category_cases) != category["count"]:
            raise ValueError(f"Category count mismatch for {category['category_id']}")
        for raw in category_cases:
            case = dict(raw)
            case["category_id"] = category["category_id"]
            cases.append(case)
    if len(cases) != index["total_cases"] or len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("Test 1 case count or identifier uniqueness is invalid")
    validate_suite(catalog, cases)
    return catalog, cases


def _plan_names(case: dict[str, Any]) -> list[str]:
    if case["category_id"] == "boundary_non_tool":
        return []
    if case["category_id"] == "plan_revision":
        return [case["expected_initial_plan"], case["expected_final_plan"]]
    return list(case.get("acceptable_plans") or [case["expected_plan"]])


def validate_suite(catalog: dict[str, Any], cases: list[dict[str, Any]]) -> None:
    profiles = catalog["plan_profiles"]
    input_profiles = catalog["input_profiles"]
    artifact_groups = catalog["artifact_groups"]
    provenance = catalog["provenance_profiles"]
    specs = load_enzywizard_tool_specs()
    valid_actions = set(get_args(RequestActionType))
    valid_request_types = set(get_args(RequestType))
    for profile_name, profile in profiles.items():
        expected_ids = {
            f"{item['tool']}:{item['side']}"
            for item in profile.get("expected_calls", [])
        }
        if not expected_ids:
            raise ValueError(f"Plan profile {profile_name} has no expected calls")
        unknown_tools = {
            item["tool"]
            for item in profile.get("expected_calls", [])
            if item["tool"] not in specs
        }
        if unknown_tools:
            raise ValueError(
                f"Plan profile {profile_name} has unknown tools: {sorted(unknown_tools)}"
            )
        forbidden_tools = set(profile.get("forbidden_tools", []))
        unknown_forbidden = forbidden_tools - set(specs)
        expected_tools = {
            item["tool"] for item in profile.get("expected_calls", [])
        }
        if unknown_forbidden or forbidden_tools.intersection(expected_tools):
            raise ValueError(
                f"Plan profile {profile_name} has invalid forbidden tools"
            )
        for before, after in profile.get("order_constraints", []):
            if before not in expected_ids or after not in expected_ids:
                raise ValueError(
                    f"Plan profile {profile_name} has an invalid order constraint"
                )
        for group in profile.get("required_artifact_groups", []):
            if group not in artifact_groups or not artifact_groups[group]:
                raise ValueError(
                    f"Plan profile {profile_name} has an invalid artifact group {group}"
                )
    for profile_name, requirements in provenance.items():
        unknown = set(requirements) - PROVENANCE_REQUIREMENTS
        if unknown or len(requirements) != len(set(requirements)):
            raise ValueError(
                f"Provenance profile {profile_name} is invalid: {sorted(unknown)}"
            )
    for case in cases:
        action_sequences = case.get("accepted_action_sequences") or [
            case.get("expected_actions", [])
        ]
        for actions in action_sequences:
            unknown_actions = set(actions) - valid_actions
            if unknown_actions or len(actions) != len(set(actions)):
                raise ValueError(
                    f"{case['case_id']}: invalid routing actions "
                    f"{sorted(unknown_actions)}"
                )
        if (
            case["category_id"] == "boundary_non_tool"
            and case.get("expected_request_type") not in valid_request_types
        ):
            raise ValueError(f"{case['case_id']}: invalid expected request type")
        if case["category_id"] != "boundary_non_tool":
            if case.get("input_profile") not in input_profiles:
                raise ValueError(f"{case['case_id']}: unknown input profile")
            if case.get("provenance_profile") not in provenance:
                raise ValueError(f"{case['case_id']}: unknown provenance profile")
        for plan_name in _plan_names(case):
            if plan_name not in profiles:
                raise ValueError(f"{case['case_id']}: unknown plan {plan_name}")
            for group in profiles[plan_name].get("required_artifact_groups", []):
                if group not in artifact_groups:
                    raise ValueError(f"{case['case_id']}: unknown artifact group {group}")
        plan_calls = {
            (item["tool"], item["side"])
            for plan_name in _plan_names(case)
            for item in profiles[plan_name].get("expected_calls", [])
        }
        forbidden_tools = set(case.get("expected_forbidden_tools", []))
        unknown_forbidden = forbidden_tools - set(specs)
        selected_tools = {tool for tool, _ in plan_calls}
        if unknown_forbidden or forbidden_tools.intersection(selected_tools):
            raise ValueError(f"{case['case_id']}: invalid forbidden tool contract")
        for expectation in case.get("parameter_expectations", []):
            key = (expectation["tool"], expectation.get("side", "single"))
            if key not in plan_calls or not expectation.get("accepted_values"):
                raise ValueError(
                    f"{case['case_id']}: invalid parameter expectation for {key}"
                )
            spec = specs[expectation["tool"]]
            parameter_names = {
                item["name"]
                for section in ("required_arguments", "optional_arguments")
                for item in spec.get(section, [])
            }
            if expectation["parameter"] not in parameter_names:
                raise ValueError(
                    f"{case['case_id']}: unknown parameter {expectation['parameter']}"
                )
        for value in catalog.get("input_profiles", {}).get(case.get("input_profile"), {}).values():
            if isinstance(value, str) and value.startswith("resources/") and not (ROOT / value).is_file():
                raise ValueError(f"{case['case_id']}: missing input resource {value}")
        history = case.get("history") or {}
        historical_types = {
            artifact["artifact_type"] for artifact in history.get("artifacts", [])
        }
        expected_tool_sets = [
            {
                item["tool"]
                for item in profiles[plan_name].get("expected_calls", [])
            }
            for plan_name in _plan_names(case)
        ]
        expects_reused_clean = any(
            "enzywizard-clean" not in tools for tools in expected_tool_sets
        ) and bool(
            historical_types.intersection(
                {"cleaned_structure_cif", "cleaned_structure_pdb"}
            )
        )
        if expects_reused_clean and not {
            "cleaned_sequence_fasta",
            "clean_report_json",
        }.issubset(historical_types):
            raise ValueError(
                f"{case['case_id']}: reusable clean fixture is incomplete"
            )
        expects_reused_substrate = any(
            "enzywizard-dock" in tools and "enzywizard-substrate" not in tools
            for tools in expected_tool_sets
        ) and "substrate_sdf" in historical_types
        if expects_reused_substrate and not {
            "substrate_report_json",
            "substrate_sdf",
        }.issubset(historical_types):
            raise ValueError(
                f"{case['case_id']}: reusable substrate fixture is incomplete"
            )
        expects_reused_docking = any(
            "enzywizard-interaction" in tools and "enzywizard-dock" not in tools
            for tools in expected_tool_sets
        ) and "docked_substrate_sdf" in historical_types
        if expects_reused_docking and not {
            "cleaned_structure_cif",
            "cleaned_sequence_fasta",
            "clean_report_json",
            "substrate_report_json",
            "dock_report_json",
            "docked_substrate_sdf",
        }.issubset(historical_types):
            raise ValueError(
                f"{case['case_id']}: reusable docking fixture is incomplete"
            )
        for artifact in history.get("artifacts", []):
            paths = artifact.get("artifact_paths") or [artifact.get("artifact_path")]
            for value in paths:
                if value and not (ROOT / value).is_file():
                    raise ValueError(f"{case['case_id']}: missing history resource {value}")


def select_cases(
    cases: list[dict[str, Any]], args: argparse.Namespace
) -> list[dict[str, Any]]:
    if args.case_ids:
        by_id = {case["case_id"]: case for case in cases}
        unknown = sorted(set(args.case_ids) - set(by_id))
        if unknown:
            raise ValueError(f"Unknown case IDs: {unknown}")
        return [by_id[item] for item in args.case_ids]
    if args.profile == "full":
        return cases
    if args.quick_case_count < 1:
        raise ValueError("--quick_case_count must be positive")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        grouped.setdefault(case["category_id"], []).append(case)
    rng = random.Random(args.seed)
    for values in grouped.values():
        rng.shuffle(values)
    selected: list[dict[str, Any]] = []
    offsets = {key: 0 for key in grouped}
    target = min(args.quick_case_count, len(cases))
    while len(selected) < target:
        for category in sorted(grouped):
            if len(selected) >= target:
                break
            offset = offsets[category]
            if offset < len(grouped[category]):
                selected.append(grouped[category][offset])
                offsets[category] += 1
    selected_ids = {case["case_id"] for case in selected}
    return [case for case in cases if case["case_id"] in selected_ids]


def _resolve_inputs(catalog: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    profile_name = case.get("input_profile")
    values = dict(catalog["input_profiles"][profile_name]) if profile_name else {}
    for name, value in list(values.items()):
        if isinstance(value, str) and value.startswith("resources/"):
            values[name] = str((ROOT / value).resolve())
    return values


def _build_agent_inputs(values: dict[str, Any]) -> AgentInputs:
    """Fill omitted optional inputs explicitly for the strict state model."""
    return AgentInputs(
        **{field: values.get(field) for field in AgentInputs.model_fields}
    )


def _empty_cleaned_files() -> CleanedFiles:
    return CleanedFiles(**{name: None for name in CleanedFiles.model_fields})


def _source_tool(artifact_type: str, path: Path) -> str:
    if artifact_type.startswith(("cleaned_", "clean_report")):
        return "enzywizard-clean"
    if artifact_type.startswith("substrate_"):
        return "enzywizard-substrate"
    if artifact_type.startswith(("dock_", "docked_")):
        return "enzywizard-dock"
    if artifact_type == "log_file":
        parent = path.parent.name.casefold()
        return "enzywizard-dock" if "dock" in parent else "enzywizard-energy"
    prefix = artifact_type.split("_report", 1)[0]
    return f"enzywizard-{prefix.replace('_', '-')}"


def _fixture_substrate_name(source: Path, artifact_type: str) -> str | None:
    report = source
    if artifact_type == "docked_substrate_sdf":
        report = source.with_name("dock_report.json")
    if artifact_type not in {
        "substrate_report_json",
        "dock_report_json",
        "docked_substrate_sdf",
    } or not report.is_file():
        return None
    payload = json.loads(report.read_text(encoding="utf-8"))
    if payload.get("report_type") == "enzywizard_substrate":
        substrates = payload.get("substrates") or []
        return substrates[0].get("substrate_name") if substrates else None
    docking = payload.get("enzyme_substrate_docking_result") or {}
    return docking.get("docked_substrate_names")


def _fixture_stage_name(
    case: dict[str, Any], artifact_type: str, source: Path
) -> str:
    """Restore the canonical filename emitted by the originating tool."""

    protein_name = f"{case['enzyme_id']}-assembly1"
    fixed_names = {
        "cleaned_structure_cif": f"cleaned_{protein_name}.cif",
        "cleaned_structure_pdb": f"cleaned_{protein_name}.pdb",
        "cleaned_sequence_fasta": f"cleaned_{protein_name}.fasta",
        "clean_report_json": f"clean_report_{protein_name}.json",
        "energy_report_json": f"energy_report_cleaned_{protein_name}.json",
    }
    if artifact_type in fixed_names:
        return fixed_names[artifact_type]
    substrate_name = _fixture_substrate_name(source, artifact_type)
    if substrate_name:
        substrate_slug = substrate_name.replace(" ", "_")
        if artifact_type == "substrate_report_json":
            return f"substrate_report_{substrate_slug}.json"
        if artifact_type == "dock_report_json":
            return f"dock_report_cleaned_{protein_name}_{substrate_slug}.json"
        if artifact_type == "docked_substrate_sdf":
            return f"docked_{substrate_slug}.sdf"
    return source.name


def _seed_artifacts(case: dict[str, Any], case_dir: Path) -> list[Artifact]:
    """Stage immutable history fixtures inside the isolated case workdir."""

    history = case.get("history") or {}
    output: list[Artifact] = []
    for item in history.get("artifacts", []):
        paths = item.get("artifact_paths") or [item.get("artifact_path")]
        for value in paths:
            source = (ROOT / value).resolve()
            source_tool = _source_tool(item["artifact_type"], source)
            fixture_dir = (
                case_dir
                / "history_fixture"
                / f"turn_{int(history['turn_id']):03d}"
                / source_tool
            )
            fixture_dir.mkdir(parents=True, exist_ok=True)
            path = fixture_dir / _fixture_stage_name(
                case, item["artifact_type"], source
            )
            shutil.copy2(source, path)
            output.append(
                Artifact(
                    turn_id=int(history["turn_id"]),
                    plan_id=1,
                    artifact_type=item["artifact_type"],
                    path=str(path),
                    source_tool=source_tool,
                    side=item["side"],
                    metadata={"fixture": True, "fixture_source": str(source)},
                )
            )
    return output


def _synthetic_executions(
    case: dict[str, Any], artifacts: list[Artifact], case_dir: Path, substrate: str | None
) -> list[ToolExecution]:
    if not artifacts:
        return []
    audit = case_dir / "history_audit"
    audit.mkdir(parents=True, exist_ok=True)
    stdout = audit / "stdout.txt"
    stderr = audit / "stderr.txt"
    stdout.write_text("fixed successful history fixture\n", encoding="utf-8")
    stderr.write_text("", encoding="utf-8")
    grouped: dict[tuple[str, str], list[Artifact]] = {}
    for artifact in artifacts:
        grouped.setdefault((artifact.source_tool, artifact.side), []).append(artifact)
    calls: list[ToolCallExecution] = []
    clean_path = next(
        (item.path for item in artifacts if "cleaned_structure" in item.artifact_type),
        str((ROOT / catalog_input_structure(case)).resolve()),
    )
    for index, ((tool, side), items) in enumerate(grouped.items(), start=1):
        arguments: dict[str, Any] = {}
        if tool == "enzywizard-substrate" and substrate:
            arguments["substrate_names"] = substrate
        if tool == "enzywizard-dock":
            arguments["input_path"] = clean_path
            if substrate:
                arguments["substrate_names"] = substrate
        calls.append(
            ToolCallExecution(
                turn_id=artifacts[0].turn_id,
                plan_id=1,
                step_index=index,
                tool_name=tool,
                side=side,
                execution_status="succeeded",
                command=[tool],
                resolved_arguments=arguments,
                exit_code=0,
                duration_seconds=0.0,
                stdout_file=str(stdout),
                stderr_file=str(stderr),
                log_files=[],
                stdout_excerpt="fixed successful history fixture",
                stderr_excerpt="",
                log_excerpt="",
                artifacts=items,
                error_message=None,
            )
        )
    return [
        ToolExecution(
            turn_id=artifacts[0].turn_id,
            plan_id=1,
            execution_status="succeeded",
            call_executions=calls,
            failed_step_index=None,
            error_messages=[],
            summary="Fixed successful history fixture.",
        )
    ]


def catalog_input_structure(case: dict[str, Any]) -> str:
    catalog = _load_yaml(CATALOG_PATH)
    return catalog["input_profiles"][case["input_profile"]]["structure_file"]


def _prior_plan(
    catalog: dict[str, Any], case: dict[str, Any], turn_id: int, case_dir: Path
) -> tuple[PlanningResult, UserConfirmation] | None:
    history = case.get("history") or {}
    prior = history.get("prior_plan")
    if not prior:
        return None
    profile = catalog["plan_profiles"][prior["plan_profile"]]
    parameter_values = {
        (item["tool"], item["side"], item["parameter"]): item["value"]
        for item in prior.get("parameters", [])
    }
    calls = []
    specs = load_enzywizard_tool_specs()
    for index, expected in enumerate(profile["expected_calls"], start=1):
        tool = expected["tool"]
        side = expected["side"]
        arguments = {
            name: value
            for (parameter_tool, parameter_side, name), value in parameter_values.items()
            if parameter_tool == tool and parameter_side == side
        }
        calls.append(
            ToolCall(
                turn_id=turn_id,
                plan_id=1,
                step_index=index,
                tool_name=tool,
                command_name=specs[tool]["command_name"],
                side=side,
                tool_execution_dir=str(case_dir / "prior_plan" / f"step_{index}"),
                arguments=arguments,
                output_artifacts=[],
                staging_rules=[],
                call_reason="Fixed approved historical plan fixture.",
            )
        )
    plan = ToolPlan(
        turn_id=turn_id,
        plan_id=1,
        purpose_type=(
            "wildtype_mutant_comparison"
            if any(item["side"] in {"wt", "mut", "shared"} and item["tool"].startswith("enzywizard-mut") for item in profile["expected_calls"])
            else "single_protein_analysis"
        ),
        summary="Fixed approved historical plan.",
        plan_reason="Fixture for historical plan reuse.",
        calls=calls,
    )
    extraction = ParameterExtractionOutput(
        parameter_extraction_status="parameter_extraction_succeeded",
        parameter_adjustments=[],
        profile_requests=[],
        unsupported_requests=[],
        extraction_summary="Fixed historical parameter record.",
        error_message="",
    )
    planning = PlanningResult(
        turn_id=turn_id,
        plan_id=1,
        planning_route="tool_plan_generated",
        planning_flag="tool_plan_generated",
        needs_tool=True,
        summary=plan.summary,
        purpose_type=plan.purpose_type,
        purpose_reason="Fixed historical workflow.",
        workflow_requirements=None,
        tool_selection=None,
        parameter_extraction=extraction,
        tool_plan=plan,
        plan_validation=None,
        error_messages=[],
    )
    return planning, UserConfirmation(
        turn_id=turn_id,
        plan_id=1,
        decision="approved",
        revision_request=None,
    )


def build_state(
    catalog: dict[str, Any], case: dict[str, Any], args: argparse.Namespace, case_dir: Path
) -> AgentState:
    inputs = _resolve_inputs(catalog, case)
    artifacts = _seed_artifacts(case, case_dir)
    history_turn = int((case.get("history") or {}).get("turn_id", 0))
    current_turn = history_turn + 1 if history_turn else 1
    prior = _prior_plan(catalog, case, history_turn, case_dir) if history_turn else None
    prior_plans = [prior[0]] if prior else []
    confirmations = [prior[1]] if prior else []
    executions = _synthetic_executions(
        case, artifacts, case_dir, inputs.get("substrate_name")
    )
    previous_messages = [
        UserMessage(turn_id=turn, content=f"Fixed prior request for turn {turn}.")
        for turn in range(1, current_turn)
    ]
    message = case.get("initial_request") or case.get("request")
    return {
        "config": AgentConfig(
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            ontology_mode=args.ontology_mode,
            workdir=case_dir,
            max_turns=max(3, current_turn + 2),
        ),
        "session_id": create_session_id(),
        "runtime_session": RuntimeSession(
            started_at=utc_now(), started_monotonic=time.monotonic()
        ),
        "agent_inputs": _build_agent_inputs(inputs),
        "current_turn_id": current_turn,
        "current_plan_id": 0,
        "current_user_message": message,
        "user_messages": previous_messages,
        "user_message_summary": "\n".join(
            f"Turn {item.turn_id}: {item.content}" for item in previous_messages
        ),
        "request_routing_results": [],
        "request_routing_summary": "",
        "history_query_results": [],
        "history_query_summary": "",
        "current_history_artifacts": [],
        "planning_results": prior_plans,
        "planning_summary": "Fixed approved historical plan." if prior else "",
        "user_confirmations": confirmations,
        "user_confirmation_summary": "Fixed prior approval." if prior else "",
        "tool_executions": executions,
        "tool_calling_summary": "Fixed successful history fixture." if executions else "",
        "analysis_results": [],
        "analysis_summary": "",
        "cleaned_files": _empty_cleaned_files(),
        "current_plan_artifacts": [],
        "artifacts": artifacts,
        "artifact_summary": "Fixed historical artifacts." if artifacts else "",
    }


def _snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "turn_id": payload.get("turn_id"),
        "plan_id": payload.get("plan_id"),
        "plan_reason": payload.get("plan_reason"),
        "calls": list(payload.get("calls") or []),
    }


def _alternatives(
    catalog: dict[str, Any], names: list[str], case: dict[str, Any]
) -> list[dict[str, Any]]:
    output = []
    for name in names:
        profile = dict(catalog["plan_profiles"][name])
        profile["name"] = name
        profile["forbidden_tools"] = sorted(
            set(profile.get("forbidden_tools", []))
            | set(case.get("expected_forbidden_tools", []))
        )
        output.append(profile)
    return output


def _score_plan_stage(
    calls: list[dict[str, Any]], alternatives: list[dict[str, Any]], case: dict[str, Any]
) -> dict[str, Any]:
    candidates = []
    for index, profile in enumerate(alternatives):
        tsa, matched = score_tsa(calls, [profile])
        pcvr, parameter_checks = score_pcvr(
            calls, profile, case.get("parameter_expectations", [])
        )
        wovr, order_checks = score_wovr(calls, profile)
        candidates.append(
            {
                "profile_index": index,
                "profile_name": profile["name"],
                "TSA": tsa,
                "PCVR": pcvr,
                "WOVR": wovr,
                "parameter_checks": parameter_checks,
                "order_checks": order_checks,
                "matched": matched is not None,
            }
        )
    return max(candidates, key=lambda item: (item["TSA"], item["PCVR"], item["WOVR"]))


def _runtime_usage(state: dict[str, Any]) -> dict[str, Any]:
    runtime = state.get("runtime_session")
    calls = list(runtime.llm_calls) if runtime else []
    return {
        "llm_calls": len(calls),
        "input_tokens": sum(item.input_tokens or 0 for item in calls),
        "output_tokens": sum(item.output_tokens or 0 for item in calls),
        "total_tokens": sum(item.total_tokens or 0 for item in calls),
    }


def run_case(
    catalog: dict[str, Any],
    case: dict[str, Any],
    repetition: int,
    args: argparse.Namespace,
    run_dir: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    case_dir = run_dir / case["case_id"] / f"repeat_{repetition:03d}"
    case_dir.mkdir(parents=True, exist_ok=False)
    state = build_state(catalog, case, args, case_dir)
    graph = build_test_graph()
    graph_config = {"configurable": {"thread_id": state["session_id"]}}
    result: dict[str, Any] = {}
    snapshots: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        result = graph.invoke(state, config=graph_config)
        if case["category_id"] != "boundary_non_tool":
            stage_count = 2 if case["category_id"] == "plan_revision" else 1
            for stage in range(stage_count):
                if not result.get("__interrupt__"):
                    routing = result.get("current_request_routing")
                    routed = routing.output if routing is not None else None
                    history = result.get("current_history_query")
                    errors.append(
                        "Missing confirmation interrupt for plan stage "
                        f"{stage + 1}; terminal_route={result.get('route')!r}, "
                        f"request_type={getattr(routed, 'request_type', None)!r}, "
                        "requires_tool_execution="
                        f"{getattr(routed, 'requires_tool_execution', None)!r}, "
                        f"history_status={getattr(history, 'query_status', None)!r}"
                    )
                    break
                payload = get_single_interrupt_payload(result)
                snapshots.append(_snapshot(payload))
                if args.show_plans:
                    sequence = " -> ".join(
                        f"{item['tool_name']}:{item['side']}" for item in payload["calls"]
                    )
                    print(f"    plan {payload['plan_id']}: {sequence}", flush=True)
                if stage + 1 < stage_count:
                    decision = "revision_requested"
                    revision = case["revision_request"]
                else:
                    if not args.execute:
                        result = record_user_confirmation(
                            result,
                            {
                                "turn_id": payload["turn_id"],
                                "plan_id": payload["plan_id"],
                                "decision": "approved",
                                "revision_request": None,
                            },
                        )
                        break
                    decision = "approved"
                    revision = None
                result = graph.invoke(
                    Command(
                        resume={
                            "turn_id": payload["turn_id"],
                            "plan_id": payload["plan_id"],
                            "decision": decision,
                            "revision_request": revision,
                        }
                    ),
                    config=graph_config,
                )
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")

    final_state = result or state
    if case["category_id"] == "boundary_non_tool":
        routing = final_state.get("current_request_routing")
        actual_calls: list[dict[str, Any]] = []
        if final_state.get("__interrupt__"):
            actual_calls = list(
                get_single_interrupt_payload(final_state).get("calls") or []
            )
        expected_type = case["expected_request_type"]
        route_ok = bool(
            routing
            and routing.output
            and routing.output.request_type == expected_type
            and not routing.output.requires_tool_execution
        )
        # The metric contract for these eight cases is only an empty-call TSA.
        # Request-type agreement is retained as a separate diagnostic.
        tsa, _ = score_tsa(actual_calls, [{"expected_calls": []}])
        metrics = {"TSA": tsa, "PCVR": None, "WOVR": None, "RAC": None, "E2E_SR": None, "PPCR": None}
        details = {"routing_correct": route_ok}
    else:
        if not snapshots:
            metrics = {name: 0.0 if name in {"TSA", "PCVR", "WOVR", "PPCR"} else None for name in ("TSA", "PCVR", "WOVR", "RAC", "E2E_SR", "PPCR")}
            details = {}
        else:
            if case["category_id"] == "plan_revision":
                stage_names = [[case["expected_initial_plan"]], [case["expected_final_plan"]]]
            else:
                stage_names = [list(case.get("acceptable_plans") or [case["expected_plan"]])]
            stage_scores = [
                _score_plan_stage(
                    snapshots[index]["calls"],
                    _alternatives(catalog, names, case),
                    case,
                )
                for index, names in enumerate(stage_names)
                if index < len(snapshots)
            ]
            tsa = float(len(stage_scores) == len(stage_names) and all(item["TSA"] == 1.0 for item in stage_scores))
            parameter_checks = [
                check
                for stage_score in stage_scores
                for check in stage_score["parameter_checks"]
            ]
            pcvr = (
                sum(float(check["correct"]) for check in parameter_checks)
                / len(parameter_checks)
                if len(stage_scores) == len(stage_names) and parameter_checks
                else 0.0
            )
            wovr = float(len(stage_scores) == len(stage_names) and all(item["WOVR"] == 1.0 for item in stage_scores))
            final_score = stage_scores[-1]
            final_profile = _alternatives(catalog, stage_names[-1], case)[final_score["profile_index"]]
            execution_model = final_state.get("current_tool_execution")
            execution = execution_model.model_dump(mode="json") if execution_model is not None else None
            if args.execute:
                required = flatten_required_artifacts(final_profile, catalog["artifact_groups"])
                rac, artifact_checks = score_rac(execution, required)
                e2e = score_e2e(tsa, pcvr, wovr, execution, rac)
            else:
                rac, e2e, artifact_checks = None, None, []
            current_turn = int(final_state.get("current_turn_id", state["current_turn_id"]))
            planning_results = [
                item for item in final_state.get("planning_results", []) if item.turn_id == current_turn
            ]
            confirmations = [
                item for item in final_state.get("user_confirmations", []) if item.turn_id == current_turn
            ]
            history_result = final_state.get("current_history_query") or state.get("current_history_query")
            provenance_requirements = catalog["provenance_profiles"][case["provenance_profile"]]
            ppcr, provenance_checks = score_ppcr(
                provenance_requirements,
                state=final_state,
                planning_results=planning_results,
                confirmations=confirmations,
                history_result=history_result,
                revision_request=case.get("revision_request"),
            )
            metrics = {"TSA": tsa, "PCVR": pcvr, "WOVR": wovr, "RAC": rac, "E2E_SR": e2e, "PPCR": ppcr}
            details = {"plan_stages": stage_scores, "artifact_checks": artifact_checks, "provenance_checks": provenance_checks}
    passed = not errors and all(
        value == 1.0 for value in metrics.values() if isinstance(value, (int, float))
    )
    return {
        "case_id": case["case_id"],
        "category_id": case["category_id"],
        "description": case["description"],
        "repetition": repetition,
        "execute": bool(args.execute),
        "passed": passed,
        "errors": errors,
        "metrics": metrics,
        "details": details,
        "plans": snapshots,
        "execution": (
            final_state["current_tool_execution"].model_dump(mode="json")
            if final_state.get("current_tool_execution") is not None
            else None
        ),
        "usage": _runtime_usage(final_state),
        "duration_seconds": round(time.monotonic() - started, 3),
        "case_workdir": str(case_dir),
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _summary(
    results: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    include_confidence_intervals: bool,
) -> dict[str, Any]:
    stats = repetition_statistics(results)
    if include_confidence_intervals and args.bootstrap_iterations:
        ci = cluster_bootstrap_ci(
            results, iterations=args.bootstrap_iterations, seed=args.seed + 1
        )
        for metric, interval in ci.items():
            stats[metric]["bootstrap_95_ci"] = interval
    by_category: dict[str, Any] = {}
    for category in sorted({item["category_id"] for item in results}):
        subset = [item for item in results if item["category_id"] == category]
        category_stats = repetition_statistics(subset)
        if include_confidence_intervals and args.bootstrap_iterations:
            category_ci = cluster_bootstrap_ci(
                subset,
                iterations=args.bootstrap_iterations,
                seed=args.seed + sum(ord(character) for character in category),
            )
            for metric, interval in category_ci.items():
                category_stats[metric]["bootstrap_95_ci"] = interval
        by_category[category] = {
            "case_count": len({item["case_id"] for item in subset}),
            "metrics": category_stats,
        }
    usage = {
        "llm_calls": sum(item["usage"]["llm_calls"] for item in results),
        "input_tokens": sum(item["usage"]["input_tokens"] for item in results),
        "output_tokens": sum(item["usage"]["output_tokens"] for item in results),
        "total_tokens": sum(item["usage"]["total_tokens"] for item in results),
    }
    return {
        "case_count": len({item["case_id"] for item in results}),
        "run_count": len(results),
        "passed_runs": sum(bool(item["passed"]) for item in results),
        "run_pass_rate": sum(bool(item["passed"]) for item in results) / len(results) if results else None,
        "metrics": stats,
        "by_category": by_category,
        "usage": usage,
        "total_duration_seconds": sum(item["duration_seconds"] for item in results),
    }


def _write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    fields = ["case_id", "category_id", "repetition", "passed", "TSA", "PCVR", "WOVR", "RAC", "E2E_SR", "PPCR", "duration_seconds"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in results:
            writer.writerow({
                "case_id": item["case_id"],
                "category_id": item["category_id"],
                "repetition": item["repetition"],
                "passed": item["passed"],
                **item["metrics"],
                "duration_seconds": item["duration_seconds"],
            })


def main() -> int:
    args = build_parser().parse_args()
    if args.repeat < 1:
        raise SystemExit("--repeat must be positive")
    if args.bootstrap_iterations != 0 and args.bootstrap_iterations < 100:
        raise SystemExit("--bootstrap_iterations must be 0 or at least 100")
    catalog, all_cases = load_suite()
    cases = select_cases(all_cases, args)
    if args.validate_only:
        counts: dict[str, int] = {}
        for case in all_cases:
            category = case["category_id"]
            counts[category] = counts.get(category, 0) + 1
        print("[validation passed] Test 1 cases and resource references are valid.")
        print(f"  total_cases={len(all_cases)}")
        for category, count in counts.items():
            print(f"  {category}={count}")
        return 0
    if not args.api_key:
        raise SystemExit("--api_key or ENZYWIZARD_API_KEY is required")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.workdir).expanduser().resolve() / f"run_{stamp}_{uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    output = Path(args.output).expanduser().resolve() if args.output else run_dir / "results.json"

    capability_state: AgentState = {
        "config": AgentConfig(
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            ontology_mode="off",
            workdir=run_dir / "_model_capability",
        ),
        "session_id": f"test1-contract-{uuid4().hex}",
        "current_turn_id": 1,
        "current_user_message": "Model contract capability probe.",
        "runtime_session": RuntimeSession(started_at=utc_now(), started_monotonic=time.monotonic()),
    }
    Path(capability_state["config"].workdir).mkdir(parents=True, exist_ok=True)
    try:
        capability = check_model_capabilities(capability_state).model_dump(mode="json")
    except Exception as error:
        print(f"[model contract failed] {error}", flush=True)
        return 2

    results: list[dict[str, Any]] = []
    print(
        f"[Test 1] cases={len(cases)} repeat={args.repeat} model={args.model} "
        f"reasoning={args.reasoning_effort} profile={args.profile} execute={args.execute} seed={args.seed}",
        flush=True,
    )
    for case in cases:
        for repetition in range(1, args.repeat + 1):
            request = case.get("initial_request") or case.get("request")
            print(f"\n[{case['case_id']}] {repetition}/{args.repeat} {case['category_id']}", flush=True)
            print(f"  Request: {request}", flush=True)
            if case.get("revision_request"):
                print(f"  Revision: {case['revision_request']}", flush=True)
            result = run_case(catalog, case, repetition, args, run_dir)
            results.append(result)
            print(f"  Metrics: {result['metrics']}", flush=True)
            if result["errors"]:
                for error in result["errors"]:
                    print(f"  Error: {error}", flush=True)
            payload = {
                "metadata": {
                    "test_id": "test_1_workflow_planning",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "model": args.model,
                    "base_url": args.base_url,
                    "reasoning_effort": args.reasoning_effort,
                    "ontology_mode": args.ontology_mode,
                    "repeat": args.repeat,
                    "profile": args.profile,
                    "execute": args.execute,
                    "seed": args.seed,
                    "selected_case_ids": [item["case_id"] for item in cases],
                    "bootstrap_iterations": args.bootstrap_iterations,
                },
                "model_capability": capability,
                "summary": _summary(
                    results, args, include_confidence_intervals=False
                ),
                "results": results,
            }
            _atomic_json(output, payload)
            if args.stop_on_failure and not result["passed"]:
                payload["summary"] = _summary(
                    results, args, include_confidence_intervals=True
                )
                _atomic_json(output, payload)
                _write_csv(output.with_suffix(".csv"), results)
                return 1
    payload["summary"] = _summary(
        results, args, include_confidence_intervals=True
    )
    _atomic_json(output, payload)
    _write_csv(output.with_suffix(".csv"), results)
    print(f"\n[complete] {output}", flush=True)
    return 0 if all(item["passed"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
