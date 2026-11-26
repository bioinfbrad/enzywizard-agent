"""Standalone inputs, session fixtures, and Agent execution wiring for Test 1."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Any

from langgraph.graph import END, StateGraph
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
from enzywizard_agent.services.runtime_service import utc_now
from enzywizard_agent.services.user_confirmation_service import (
    record_user_confirmation,
)
from enzywizard_agent.state import AgentState
from enzywizard_agent.utils.CLI_utils import get_single_interrupt_payload
from enzywizard_agent.utils.checkpoint_utils import create_in_memory_checkpointer
from enzywizard_agent.utils.IO_utils import create_session_id
from enzywizard_agent.utils.tool_plan_utils import load_enzywizard_tool_specs

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
CATALOG_PATH = ROOT / "case_catalog.yaml"
INDEX_PATH = ROOT / "cases" / "index.yaml"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-6-luna"
DEFAULT_REASONING_EFFORT = "medium"


def resource_path(value: str) -> Path:
    root = (ROOT / "resources").resolve()
    path = (ROOT / value).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Invalid benchmark resource: {value}")
    return path


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
        "--workdir", default="~/enzywizard-benchmark-results/test_1_workflow_planning"
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


def load_cases() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    catalog = _load_yaml(CATALOG_PATH)
    index = _load_yaml(INDEX_PATH)
    cases = []
    for category in index["categories"]:
        cases_file = INDEX_PATH.parent / category["file"]
        payload = _load_yaml(cases_file)
        if payload["category_id"] != category["category_id"] or len(payload["cases"]) != category["count"]:
            raise ValueError(f"Invalid category or case count in {cases_file}")
        for entry in payload["cases"]:
            case = {**entry, "category_id": category["category_id"]}
            _resolve_inputs(catalog, case)
            history = case.get("history") or {}
            if history.get("prior_plan"):
                catalog["prior_plan_profiles"][history["prior_plan"]["plan_profile"]]
            for artifact in history.get("artifacts", []):
                for path in artifact.get("artifact_paths") or [artifact["artifact_path"]]:
                    resource_path(path)
            cases.append(case)
    ids = {case["case_id"] for case in cases}
    if len(ids) != len(cases) or len(cases) != index["total_cases"] or len(cases) != 60:
        raise ValueError("The benchmark must contain 60 distinct workflow cases.")
    return catalog, cases


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
            values[name] = str(resource_path(value))
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
            source = resource_path(value)
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
    artifacts: list[Artifact], case_dir: Path, substrate: str | None, structure: str
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
        structure,
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


def _prior_plan(
    catalog: dict[str, Any], case: dict[str, Any], turn_id: int, case_dir: Path
) -> tuple[PlanningResult, UserConfirmation] | None:
    history = case.get("history") or {}
    prior = history.get("prior_plan")
    if not prior:
        return None
    profile = catalog["prior_plan_profiles"][prior["plan_profile"]]
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
        artifacts, case_dir, inputs.get("substrate_name"), inputs["structure_file"]
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


def _runtime_usage(state: dict[str, Any]) -> dict[str, Any]:
    runtime = state.get("runtime_session")
    calls = list(runtime.llm_calls) if runtime else []
    return {
        "llm_calls": len(calls),
        "input_tokens": sum(item.input_tokens or 0 for item in calls),
        "output_tokens": sum(item.output_tokens or 0 for item in calls),
        "total_tokens": sum(item.total_tokens or 0 for item in calls),
    }


def _write_planning_diagnostics(
    path: Path,
    result: dict[str, Any],
    snapshots: list[dict[str, Any]],
    state: AgentState,
    args: argparse.Namespace,
) -> None:
    runtime = state.get("runtime_session")
    entry = {
        "case_id": result["case_id"],
        "repetition": result["repetition"],
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "passed": result["passed"],
        "errors": result["errors"],
        "snapshots": snapshots,
        "llm_calls": [
            call.model_dump(mode="json") for call in runtime.llm_calls
        ] if runtime is not None else [],
        "runtime_errors": [
            error.model_dump(mode="json") for error in runtime.errors
        ] if runtime is not None else [],
    }
    line = json.dumps(entry, ensure_ascii=False)
    if args.api_key:
        key = json.dumps(args.api_key, ensure_ascii=False)[1:-1]
        line = line.replace(key, "[redacted]")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def call_key(call: dict[str, Any]) -> tuple[str, str]:
    return str(call.get("tool_name") or call.get("tool")), str(call.get("side"))


def value_type_is_valid(value: Any, type_name: str | None) -> bool:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="python")
    if isinstance(value, dict) and "source_step_index" in value:
        return True
    if type_name == "bool":
        return isinstance(value, bool)
    if type_name == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "float":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "str":
        return isinstance(value, str) and bool(value.strip())
    if type_name in {"list", "list[str]"}:
        return isinstance(value, list)
    return value is not None


def reference_is_valid(
    value: Any, current_index: int, outputs_by_step: dict[int, set[str]]
) -> bool:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="python")
    if isinstance(value, dict):
        if "source_step_index" in value:
            source = value.get("source_step_index")
            artifact_type = value.get("artifact_type")
            return (
                isinstance(source, int)
                and source < current_index
                and artifact_type in outputs_by_step.get(source, set())
            )
        return all(
            reference_is_valid(nested, current_index, outputs_by_step)
            for nested in value.values()
        )
    if isinstance(value, (list, tuple)):
        return all(
            reference_is_valid(nested, current_index, outputs_by_step)
            for nested in value
        )
    return True


def referenced_step_indexes(value: Any) -> set[int]:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="python")
    if isinstance(value, dict):
        output: set[int] = set()
        if isinstance(value.get("source_step_index"), int):
            output.add(int(value["source_step_index"]))
        for nested in value.values():
            output.update(referenced_step_indexes(nested))
        return output
    if isinstance(value, (list, tuple)):
        output: set[int] = set()
        for nested in value:
            output.update(referenced_step_indexes(nested))
        return output
    return set()


def artifact_file_is_valid(path_value: str, artifact_type: str) -> bool:
    path = Path(path_value)
    if not path.is_file() or path.stat().st_size == 0:
        return False
    suffix = path.suffix.casefold()
    try:
        if artifact_type.endswith("_json") or suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
        elif suffix in {".fasta", ".fa", ".faa"}:
            return path.read_text(encoding="utf-8", errors="ignore").lstrip().startswith(">")
        elif suffix == ".sdf":
            return "$$$$" in path.read_text(encoding="utf-8", errors="ignore")
        elif suffix in {".cif", ".mmcif"}:
            return "data_" in path.read_text(encoding="utf-8", errors="ignore")[:4096]
        elif suffix == ".pdb":
            text = path.read_text(encoding="utf-8", errors="ignore")
            return "ATOM" in text or "HETATM" in text
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return True


