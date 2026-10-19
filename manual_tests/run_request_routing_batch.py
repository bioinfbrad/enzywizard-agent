"""Run live Request Router and History Query semantic evaluations.

This suite calls the configured LLM but never executes an EnzyWizard command. Each
case receives the same controlled three-turn artifact history so history selectors,
mixed actions, and reuse requests can be checked independently from scientific tool
runtime and from Planning.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time
from typing import Any
import uuid

from enzywizard_agent.conf import AgentConfig
from enzywizard_agent.data_models.agent_inputs import AgentInputs
from enzywizard_agent.data_models.artifact import Artifact, CleanedFiles
from enzywizard_agent.data_models.planning import PlanningResult, ToolCall, ToolPlan
from enzywizard_agent.data_models.user_confirmation import UserConfirmation
from enzywizard_agent.data_models.user_message import UserMessage
from enzywizard_agent.services.request_routing_service import (
    plan_history_query,
    route_user_request,
)
from enzywizard_agent.services.analysis_service import analyze_agent_state
from enzywizard_agent.services.llm_service import check_model_capabilities
from enzywizard_agent.state import AgentState
from enzywizard_agent.utils.IO_utils import create_session_id

from evaluation_utils import aggregate_usage, runtime_usage_payload


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES_PATH = Path(__file__).with_name("request_routing_cases.json")
DEFAULT_BASE_URL = (
    "https://ws-gwd0ta1p7z9yx2r9.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate EnzyWizard-Agent request routing and structured history "
            "selection with a live LLM."
        )
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
        help="Optional OpenAI-compatible reasoning effort, such as 'none'.",
    )
    parser.add_argument(
        "--workdir",
        default="~/enzywizard-agent-request-routing-tests",
    )
    parser.add_argument("--cases", default=str(DEFAULT_CASES_PATH))
    parser.add_argument(
        "--output",
        default=None,
        help="Optional result JSON path. Defaults to results.json in the run directory.",
    )
    parser.add_argument("--case_ids", nargs="*")
    parser.add_argument(
        "--tags",
        nargs="*",
        help="Run cases matching at least one tag.",
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--stop_on_failure", action="store_true")
    parser.add_argument(
        "--include_analysis",
        action="store_true",
        help=(
            "Also generate and validate final Analysis answers for requests that "
            "do not require tool execution."
        ),
    )
    parser.add_argument(
        "--show_details",
        action="store_true",
        help="Show routing reasons and structured history selectors.",
    )
    return parser


def load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(cases, list) or not cases:
        raise ValueError("The case file must contain a non-empty 'cases' list.")

    known_ids: set[str] = set()
    required_expected_fields = {
        "request_type",
        "actions",
        "route",
        "tools",
        "history",
        "history_artifacts",
        "reuse_policy",
        "recomputation",
    }
    for case in cases:
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("Every routing case requires a non-empty case_id.")
        if case_id in known_ids:
            raise ValueError(f"Duplicate routing case ID: {case_id}.")
        if not isinstance(case.get("message"), str) or not case["message"].strip():
            raise ValueError(f"Case {case_id} requires a non-empty message.")
        tags = case.get("tags", [])
        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            raise ValueError(f"Case {case_id} contains invalid tags.")
        expected = case.get("expected")
        if not isinstance(expected, dict):
            raise ValueError(f"Case {case_id} requires an expected object.")
        missing = sorted(required_expected_fields - set(expected))
        if missing:
            raise ValueError(f"Case {case_id} is missing expectations: {missing}.")
        if expected["history_artifacts"] and "query" not in expected:
            raise ValueError(
                f"Case {case_id} requires history query expectations."
            )
        known_ids.add(case_id)
    return cases


def select_cases(
    cases: list[dict[str, Any]],
    case_ids: list[str] | None,
    tags: list[str] | None,
) -> list[dict[str, Any]]:
    selected = cases
    if case_ids:
        by_id = {case["case_id"]: case for case in cases}
        unknown = sorted(set(case_ids) - set(by_id))
        if unknown:
            raise ValueError(f"Unknown case IDs: {unknown}.")
        selected = [by_id[case_id] for case_id in case_ids]
    if tags:
        requested = {tag.strip() for tag in tags if tag.strip()}
        known = {tag for case in cases for tag in case.get("tags", [])}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"Unknown tags: {unknown}.")
        selected = [
            case
            for case in selected
            if requested.intersection(case.get("tags", []))
        ]
    if not selected:
        raise ValueError("No routing cases matched the requested filters.")
    return selected


def create_run_directory(parent: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = parent.expanduser().resolve() / f"run_{timestamp}_{uuid.uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def empty_cleaned_files() -> CleanedFiles:
    return CleanedFiles(
        **{field_name: None for field_name in CleanedFiles.model_fields}
    )


def write_artifact(
    root: Path,
    *,
    turn_id: int,
    artifact_type: str,
    source_tool: str,
    side: str,
    report_type: str | None = None,
    suffix: str = ".dat",
    serial: int = 1,
) -> Artifact:
    """Create a minimal real file and register it in the synthetic history."""

    directory = root / f"turn_{turn_id:03d}" / source_tool.replace("enzywizard-", "")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{artifact_type}_{side}_{serial}{suffix}"
    if report_type is not None:
        payload = {
            "report_type": report_type,
            "test_metric": float(turn_id * 10 + serial),
            "test_side": side,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        path.write_text(
            f"synthetic {artifact_type} for turn {turn_id} side {side}\n",
            encoding="utf-8",
        )
    return Artifact(
        turn_id=turn_id,
        plan_id=1,
        artifact_type=artifact_type,
        path=str(path.resolve()),
        source_tool=source_tool,
        side=side,
        metadata={"synthetic_history": True},
    )


def seed_history(root: Path) -> list[Artifact]:
    """Build three completed turns covering every history-selection dimension."""

    specs: list[dict[str, Any]] = [
        # Turn 1: single-protein property workflow.
        {"turn_id": 1, "artifact_type": "cleaned_structure_cif", "source_tool": "enzywizard-clean", "side": "single", "suffix": ".cif"},
        {"turn_id": 1, "artifact_type": "clean_report_json", "source_tool": "enzywizard-clean", "side": "single", "report_type": "enzywizard_clean", "suffix": ".json"},
        {"turn_id": 1, "artifact_type": "energy_report_json", "source_tool": "enzywizard-energy", "side": "single", "report_type": "enzywizard_energy", "suffix": ".json"},
        {"turn_id": 1, "artifact_type": "pocket_report_json", "source_tool": "enzywizard-pocket", "side": "single", "report_type": "enzywizard_pocket", "suffix": ".json"},
        {"turn_id": 1, "artifact_type": "integrate_report_json", "source_tool": "enzywizard-integrate", "side": "single", "report_type": "enzywizard_integrate", "suffix": ".json"},
        {"turn_id": 1, "artifact_type": "integrate_nodes_json", "source_tool": "enzywizard-integrate", "side": "single", "suffix": ".json"},
        {"turn_id": 1, "artifact_type": "integrate_edges_json", "source_tool": "enzywizard-integrate", "side": "single", "suffix": ".json"},
        {"turn_id": 1, "artifact_type": "log_file", "source_tool": "enzywizard-energy", "side": "single", "suffix": ".txt"},
        # Turn 2: paired WT/MUT comparison.
        {"turn_id": 2, "artifact_type": "wt_cleaned_structure_cif", "source_tool": "enzywizard-mut-clean", "side": "wt", "suffix": ".cif"},
        {"turn_id": 2, "artifact_type": "mut_cleaned_structure_cif", "source_tool": "enzywizard-mut-clean", "side": "mut", "suffix": ".cif"},
        {"turn_id": 2, "artifact_type": "mut_clean_report_json", "source_tool": "enzywizard-mut-clean", "side": "shared", "report_type": "enzywizard_mut_clean", "suffix": ".json"},
        {"turn_id": 2, "artifact_type": "energy_report_json", "source_tool": "enzywizard-energy", "side": "wt", "report_type": "enzywizard_energy", "suffix": ".json", "serial": 1},
        {"turn_id": 2, "artifact_type": "energy_report_json", "source_tool": "enzywizard-energy", "side": "mut", "report_type": "enzywizard_energy", "suffix": ".json", "serial": 2},
        {"turn_id": 2, "artifact_type": "flexibility_report_json", "source_tool": "enzywizard-flexibility", "side": "wt", "report_type": "enzywizard_flexibility", "suffix": ".json", "serial": 1},
        {"turn_id": 2, "artifact_type": "flexibility_report_json", "source_tool": "enzywizard-flexibility", "side": "mut", "report_type": "enzywizard_flexibility", "suffix": ".json", "serial": 2},
        {"turn_id": 2, "artifact_type": "pocket_report_json", "source_tool": "enzywizard-pocket", "side": "wt", "report_type": "enzywizard_pocket", "suffix": ".json", "serial": 1},
        {"turn_id": 2, "artifact_type": "pocket_report_json", "source_tool": "enzywizard-pocket", "side": "mut", "report_type": "enzywizard_pocket", "suffix": ".json", "serial": 2},
        {"turn_id": 2, "artifact_type": "mut_integrate_report_json", "source_tool": "enzywizard-mut-integrate", "side": "shared", "report_type": "enzywizard_mut_integrate", "suffix": ".json"},
        {"turn_id": 2, "artifact_type": "wt_integrate_nodes_json", "source_tool": "enzywizard-mut-integrate", "side": "wt", "suffix": ".json"},
        {"turn_id": 2, "artifact_type": "wt_integrate_edges_json", "source_tool": "enzywizard-mut-integrate", "side": "wt", "suffix": ".json"},
        {"turn_id": 2, "artifact_type": "mut_integrate_nodes_json", "source_tool": "enzywizard-mut-integrate", "side": "mut", "suffix": ".json"},
        {"turn_id": 2, "artifact_type": "mut_integrate_edges_json", "source_tool": "enzywizard-mut-integrate", "side": "mut", "suffix": ".json"},
        {"turn_id": 2, "artifact_type": "log_file", "source_tool": "enzywizard-mut-integrate", "side": "shared", "suffix": ".txt"},
        # Turn 3: substrate, docking, interaction, and integrated graph.
        {"turn_id": 3, "artifact_type": "cleaned_structure_cif", "source_tool": "enzywizard-clean", "side": "single", "suffix": ".cif"},
        {"turn_id": 3, "artifact_type": "substrate_report_json", "source_tool": "enzywizard-substrate", "side": "single", "report_type": "enzywizard_substrate", "suffix": ".json"},
        {"turn_id": 3, "artifact_type": "substrate_sdf", "source_tool": "enzywizard-substrate", "side": "single", "suffix": ".sdf", "serial": 1},
        {"turn_id": 3, "artifact_type": "substrate_sdf", "source_tool": "enzywizard-substrate", "side": "single", "suffix": ".sdf", "serial": 2},
        {"turn_id": 3, "artifact_type": "dock_report_json", "source_tool": "enzywizard-dock", "side": "single", "report_type": "enzywizard_dock", "suffix": ".json"},
        {"turn_id": 3, "artifact_type": "docked_substrate_sdf", "source_tool": "enzywizard-dock", "side": "single", "suffix": ".sdf"},
        {"turn_id": 3, "artifact_type": "docked_complex_structure", "source_tool": "enzywizard-dock", "side": "single", "suffix": ".pdb"},
        {"turn_id": 3, "artifact_type": "interaction_report_json", "source_tool": "enzywizard-interaction", "side": "single", "report_type": "enzywizard_interaction", "suffix": ".json"},
        {"turn_id": 3, "artifact_type": "integrate_report_json", "source_tool": "enzywizard-integrate", "side": "single", "report_type": "enzywizard_integrate", "suffix": ".json"},
        {"turn_id": 3, "artifact_type": "integrate_nodes_json", "source_tool": "enzywizard-integrate", "side": "single", "suffix": ".json"},
        {"turn_id": 3, "artifact_type": "integrate_edges_json", "source_tool": "enzywizard-integrate", "side": "single", "suffix": ".json"},
        {"turn_id": 3, "artifact_type": "log_file", "source_tool": "enzywizard-dock", "side": "single", "suffix": ".txt"},
    ]
    return [write_artifact(root, **spec) for spec in specs]


def prior_plan(root: Path) -> tuple[PlanningResult, UserConfirmation]:
    """Provide Router context for requests that refer to the previous plan."""

    tools = [
        ("enzywizard-clean", "single", {"pH": 7.0, "add_H": True}),
        ("enzywizard-substrate", "single", {"substrate_names": "glucose"}),
        ("enzywizard-dock", "single", {"substrate_names": "glucose", "exhaustiveness": 8}),
        ("enzywizard-interaction", "single", {"substrate_names": "docked_glucose"}),
        ("enzywizard-integrate", "single", {"strict": False}),
    ]
    calls = [
        ToolCall(
            turn_id=3,
            plan_id=1,
            step_index=index,
            tool_name=tool_name,
            command_name=tool_name,
            side=side,
            tool_execution_dir=str(root / f"prior_plan_{index:02d}"),
            arguments=arguments,
            output_artifacts=[],
            staging_rules=[],
            call_reason="Synthetic approved plan context.",
        )
        for index, (tool_name, side, arguments) in enumerate(tools, start=1)
    ]
    plan = ToolPlan(
        turn_id=3,
        plan_id=1,
        purpose_type="single_protein_analysis",
        summary="Approved substrate docking and interaction workflow.",
        plan_reason="Synthetic history for semantic routing tests.",
        calls=calls,
    )
    planning = PlanningResult(
        turn_id=3,
        plan_id=1,
        planning_route="tool_plan_generated",
        planning_flag="tool_plan_generated",
        needs_tool=True,
        summary=plan.summary,
        purpose_type="single_protein_analysis",
        purpose_reason="Synthetic prior workflow.",
        tool_selection=None,
        parameter_extraction=None,
        tool_plan=plan,
        error_messages=[],
    )
    confirmation = UserConfirmation(
        turn_id=3,
        plan_id=1,
        decision="approved",
        revision_request=None,
    )
    return planning, confirmation


def build_state(
    case: dict[str, Any],
    api_key: str,
    base_url: str,
    model: str,
    reasoning_effort: str | None,
    case_dir: Path,
) -> AgentState:
    history_root = case_dir / "synthetic_history"
    artifacts = seed_history(history_root)
    planning, confirmation = prior_plan(history_root)
    return {
        "config": AgentConfig(
            api_key=api_key,
            base_url=base_url,
            model=model,
            reasoning_effort=reasoning_effort,
            workdir=case_dir,
            max_turns=6,
        ),
        "session_id": create_session_id(),
        "agent_inputs": AgentInputs(
            structure_file=str((PROJECT_ROOT / "test_inputs/1ZG4_WT.pdb").resolve()),
            msa_file=str((PROJECT_ROOT / "test_inputs/jhmm_1ZG4_WT.sto").resolve()),
            mut_structure_file=str((PROJECT_ROOT / "test_inputs/1ZG6_S70G.pdb").resolve()),
            mut_msa_file=str((PROJECT_ROOT / "test_inputs/jhmm_1ZG6_S70G.sto").resolve()),
            substrate_name="glucose",
            amino_acid_substitution="S70G",
        ),
        "current_turn_id": 4,
        "current_plan_id": 0,
        "current_user_message": case["message"].strip(),
        "user_messages": [
            UserMessage(turn_id=1, content="Analyze energy and binding pockets."),
            UserMessage(turn_id=2, content="Compare WT and mutant energy, flexibility, and pockets."),
            UserMessage(turn_id=3, content="Dock glucose and analyze protein-substrate interactions."),
        ],
        "user_message_summary": (
            "Turn 1: Analyze energy and binding pockets.\n"
            "Turn 2: Compare WT and mutant energy, flexibility, and pockets.\n"
            "Turn 3: Dock glucose and analyze protein-substrate interactions."
        ),
        "request_routing_results": [],
        "request_routing_summary": "",
        "history_query_results": [],
        "history_query_summary": "",
        "current_history_artifacts": [],
        "planning_results": [planning],
        "planning_summary": "Turn 3 plan 1 was approved.",
        "user_confirmations": [confirmation],
        "user_confirmation_summary": "Turn 3 plan 1: approved.",
        "tool_executions": [],
        "tool_calling_summary": "",
        "analysis_results": [],
        "analysis_summary": "",
        "cleaned_files": empty_cleaned_files(),
        "current_plan_artifacts": [],
        "artifacts": artifacts,
        "artifact_summary": "Synthetic artifacts are available from turns 1, 2, and 3.",
    }


def evaluate_routing(
    expected: dict[str, Any],
    result: Any,
) -> list[str]:
    failures: list[str] = []
    if result.routing_status != "succeeded" or result.output is None:
        return [result.error_message or "Request routing did not succeed."]
    output = result.output
    actual_actions = [action.action_type for action in output.actions]
    expected_actions = expected["actions"]
    checks = {
        "request_type": (output.request_type, expected["request_type"]),
        "route": (result.route, expected["route"]),
        "requires_tool_execution": (output.requires_tool_execution, expected["tools"]),
        "references_history": (output.references_history, expected["history"]),
        "requires_history_artifacts": (
            output.requires_history_artifacts,
            expected["history_artifacts"],
        ),
        "reuse_policy": (output.reuse_policy, expected["reuse_policy"]),
        "requests_recomputation": (
            output.requests_recomputation,
            expected["recomputation"],
        ),
    }
    for label, (actual, wanted) in checks.items():
        if actual != wanted:
            failures.append(f"Expected {label}={wanted!r}, got {actual!r}.")
    if len(actual_actions) != len(set(actual_actions)):
        failures.append(f"Router returned duplicate actions: {actual_actions!r}.")
    if set(actual_actions) != set(expected_actions):
        failures.append(
            f"Expected actions={sorted(expected_actions)!r}, "
            f"got {sorted(actual_actions)!r}."
        )
    if expected["request_type"] == "clarification" and not output.clarification_question:
        failures.append("A clarification route must provide a question.")
    return failures


def evaluate_history(
    expected: dict[str, Any],
    result: Any,
) -> list[str]:
    # Router 与预期不一致却意外进入 History 时，路由断言已经记录根因；
    # 此处不能再因测试用例没有 history query 预期而抛出 KeyError。
    query_expected = expected.get("query")
    if not isinstance(query_expected, dict):
        return []

    failures: list[str] = []
    if result.query_status != "succeeded" or result.query is None:
        return [
            result.error_message
            or f"History query status was {result.query_status!r}: {result.summary}"
        ]
    query = result.query
    required_operations = set(query_expected.get("required_operations", []))
    if not required_operations.issubset(set(query.operations)):
        failures.append(
            f"Missing history operations: "
            f"{sorted(required_operations - set(query.operations))}."
        )
    wanted_turn_ids = query_expected.get("resolved_turn_ids")
    if wanted_turn_ids is not None and result.resolved_turn_ids != wanted_turn_ids:
        failures.append(
            f"Expected resolved turns={wanted_turn_ids!r}, "
            f"got {result.resolved_turn_ids!r}."
        )
    selected_types = {artifact.artifact_type for artifact in result.selected_artifacts}
    required_types = set(query_expected.get("required_artifact_types", []))
    if not required_types.issubset(selected_types):
        failures.append(
            f"Missing selected artifact types: "
            f"{sorted(required_types - selected_types)}."
        )
    selected_sides = {artifact.side for artifact in result.selected_artifacts}
    required_sides = set(query_expected.get("required_sides", []))
    if not required_sides.issubset(selected_sides):
        failures.append(
            f"Missing selected sides: {sorted(required_sides - selected_sides)}."
        )
    return failures


def expected_analysis_mode(request_type: str, actions: list[str]) -> str:
    if request_type == "history_analysis":
        return "scientific_result"
    if request_type == "artifact_query":
        return "artifact_query"
    if request_type == "software_help":
        return "software_help"
    if request_type == "general_scientific_question":
        return "general_scientific_question"
    if request_type == "conversation":
        return "conversation"
    if request_type == "out_of_scope":
        return "out_of_scope"
    if request_type == "clarification":
        return "clarification"
    if request_type == "exit":
        return "exit"
    if "analyze_history" in actions:
        return "scientific_result"
    if "list_artifacts" in actions:
        return "artifact_query"
    raise ValueError(f"No Analysis mode expectation for {request_type!r}.")


def evaluate_analysis(expected: dict[str, Any], result: Any) -> list[str]:
    failures: list[str] = []
    wanted_mode = expected_analysis_mode(
        expected["request_type"],
        expected["actions"],
    )
    if result.response_mode != wanted_mode:
        failures.append(
            f"Expected Analysis response_mode={wanted_mode!r}, "
            f"got {result.response_mode!r}."
        )
    wanted_route = "exit_requested" if wanted_mode == "exit" else "answer_generated"
    if result.analysis_route != wanted_route:
        failures.append(
            f"Expected Analysis route={wanted_route!r}, "
            f"got {result.analysis_route!r}."
        )
    if result.analysis_status == "failed":
        failures.append(
            f"Analysis failed: {result.error_messages or [result.summary]}."
        )
    if not result.final_answer.strip():
        failures.append("Analysis returned an empty final answer.")
    if re.search(r"[\u4e00-\u9fff]", result.final_answer):
        failures.append("Analysis returned Chinese text in an English-only CLI test.")
    return failures


def run_case(
    case: dict[str, Any],
    repetition: int,
    args: argparse.Namespace,
    run_dir: Path,
) -> dict[str, Any]:
    started_at = time.monotonic()
    case_dir = run_dir / case["case_id"] / f"run_{repetition:03d}"
    case_dir.mkdir(parents=True, exist_ok=False)
    state = build_state(
        case,
        str(args.api_key).strip(),
        str(args.base_url).strip(),
        str(args.model).strip(),
        args.reasoning_effort,
        case_dir,
    )
    failures: list[str] = []
    routing = None
    history = None
    analysis = None
    try:
        routing = route_user_request(state)
        failures.extend(evaluate_routing(case["expected"], routing))
        state["current_request_routing"] = routing
        if (
            routing.routing_status == "succeeded"
            and routing.output is not None
            and routing.output.requires_history_artifacts
        ):
            history = plan_history_query(state)
            failures.extend(evaluate_history(case["expected"], history))
            state["current_history_query"] = history
            state["current_history_artifacts"] = list(history.selected_artifacts)
        if (
            args.include_analysis
            and not case["expected"]["tools"]
            and routing.routing_status == "succeeded"
            and routing.output is not None
            and not routing.output.requires_tool_execution
        ):
            analysis = analyze_agent_state(state)
            failures.extend(evaluate_analysis(case["expected"], analysis))
    except Exception as exc:
        failures.append(f"Unhandled test exception: {type(exc).__name__}: {exc}")

    return {
        "case_id": case["case_id"],
        "description": case.get("description", ""),
        "tags": case.get("tags", []),
        "repetition": repetition,
        "passed": not failures,
        "failures": failures,
        "duration_seconds": round(time.monotonic() - started_at, 3),
        "routing_result": (
            routing.model_dump(mode="json") if routing is not None else None
        ),
        "history_query_result": (
            history.model_dump(mode="json") if history is not None else None
        ),
        "analysis_result": (
            analysis.model_dump(mode="json") if analysis is not None else None
        ),
        "usage": runtime_usage_payload(state.get("runtime_session")),
    }


def result_payload(
    args: argparse.Namespace,
    cases_path: Path,
    run_dir: Path,
    selected_count: int,
    results: list[dict[str, Any]],
    capability_usage: dict[str, Any] | None,
) -> dict[str, Any]:
    passed = sum(result["passed"] for result in results)
    usage = aggregate_usage(
        [capability_usage, *(result.get("usage") for result in results)]
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "base_url": args.base_url,
        "cases_file": str(cases_path),
        "run_dir": str(run_dir),
        "selected_case_count": selected_count,
        "repeat": args.repeat,
        "include_analysis": args.include_analysis,
        "summary": {
            "total": len(results),
            "passed": passed,
            "failed": len(results) - passed,
        },
        "usage": usage,
        "results": results,
    }


def save_results(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def compact_failure_message(message: str, max_chars: int = 500) -> str:
    """终端只显示单行根因，完整异常仍保存在结果 JSON。"""

    compact = " ".join(str(message).split())
    if len(compact) > max_chars:
        return f"{compact[: max_chars - 3]}..."
    return compact


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.api_key:
        parser.error("--api_key or ENZYWIZARD_API_KEY is required")
    if args.repeat < 1:
        parser.error("--repeat must be greater than or equal to 1")

    cases_path = Path(args.cases).expanduser().resolve()
    try:
        cases = select_cases(
            load_cases(cases_path),
            args.case_ids,
            args.tags,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    run_dir = create_run_directory(Path(args.workdir))
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else run_dir / "results.json"
    )
    print("[运行模式] 请求路由、历史查询与分析语义测试", flush=True)
    print(f"[模型] {args.model}", flush=True)
    print(f"[推理强度] {args.reasoning_effort or '模型默认值'}", flush=True)
    print(f"[案例数量] {len(cases)} x {args.repeat}", flush=True)
    print(f"[运行目录] {run_dir}", flush=True)
    print(f"[结果文件] {output_path}", flush=True)
    print("[失败策略] 单个案例失败后记录错误并继续", flush=True)

    capability_dir = run_dir / "_model_capability_check"
    capability_usage: dict[str, Any] | None = None
    try:
        capability_state = build_state(
            cases[0],
            str(args.api_key).strip(),
            str(args.base_url).strip(),
            str(args.model).strip(),
            args.reasoning_effort,
            capability_dir,
        )
        capability = check_model_capabilities(capability_state)
        capability_usage = runtime_usage_payload(
            capability_state.get("runtime_session")
        )
        print(
            "[模型契约] "
            f"{capability.status}，strict_json_schema="
            f"{capability.strict_structured_output}",
            flush=True,
        )
    except Exception as error:
        print(f"[模型契约] 失败：{compact_failure_message(str(error))}", flush=True)
        print("[终止] 模型未通过严格 JSON Schema 能力探测。", flush=True)
        return 2

    results: list[dict[str, Any]] = []
    stop = False
    total = len(cases) * args.repeat
    for repetition in range(1, args.repeat + 1):
        for case in cases:
            index = len(results) + 1
            print(
                f"\n[案例 {index}/{total}] 开始：{case['case_id']} "
                f"（第 {repetition} 次）",
                flush=True,
            )
            result = run_case(case, repetition, args, run_dir)
            results.append(result)
            save_results(
                output_path,
                result_payload(
                    args,
                    cases_path,
                    run_dir,
                    len(cases),
                    results,
                    capability_usage,
                ),
            )

            routing = result.get("routing_result") or {}
            output = routing.get("output") or {}
            if result["passed"]:
                print(
                    f"[案例结果] 通过，用时 {result['duration_seconds']:.1f} 秒；"
                    f"类型={output.get('request_type')}，路由={routing.get('route')}",
                    flush=True,
                )
            else:
                print(
                    f"[案例结果] 失败，用时 {result['duration_seconds']:.1f} 秒",
                    flush=True,
                )
                for failure in result["failures"]:
                    print(
                        f"  [失败原因] {compact_failure_message(failure)}",
                        flush=True,
                    )
            if args.show_details:
                print(f"  [动作] {output.get('actions')}", flush=True)
                print(f"  [理由] {output.get('routing_reason')}", flush=True)
                history = result.get("history_query_result") or {}
                if history:
                    print(f"  [历史查询状态] {history.get('query_status')}", flush=True)
                    print(f"  [解析轮次] {history.get('resolved_turn_ids')}", flush=True)
                    print(f"  [选择条件] {history.get('query')}", flush=True)
            failed = sum(not item["passed"] for item in results)
            print(
                f"[当前进度] 已完成 {len(results)}/{total}，失败 {failed}",
                flush=True,
            )
            if not result["passed"] and args.stop_on_failure:
                stop = True
                break
        if stop:
            break

    passed = sum(result["passed"] for result in results)
    failed = len(results) - passed
    print(
        f"\n[全部完成] 总数={len(results)}，通过={passed}，失败={failed}",
        flush=True,
    )
    print(f"[完整结果] {output_path}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
