"""使用既有真实工作流输出回归测试 Analysis 和多轮状态逻辑。

本脚本不执行 EnzyWizard 科学工具，也不访问模型接口。它读取指定运行目录中的
正式 JSON 报告，并强制 Analysis 走确定性降级分支，适合快速重复运行。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from langgraph.types import Command

from enzywizard_agent.conf import AgentConfig
from enzywizard_agent.data_models.agent_inputs import AgentInputs
from enzywizard_agent.data_models.analysis import ReportSummary
from enzywizard_agent.data_models.artifact import Artifact, CleanedFiles
from enzywizard_agent.data_models.history_query import (
    HistoryQueryOutput,
    HistoryQueryResult,
    HistoryTurnSelectionDraft,
)
from enzywizard_agent.data_models.request_routing import (
    RequestedAction,
    RequestRoutingOutput,
    RequestRoutingResult,
)
from enzywizard_agent.data_models.user_message import UserMessage
from enzywizard_agent.data_models.tool_execution import (
    ToolCallExecution,
    ToolExecution,
)
from enzywizard_agent.data_models.user_confirmation import UserConfirmation
from enzywizard_agent.graph import build_graph
from enzywizard_agent.nodes.analysis import analysis_node
import enzywizard_agent.services.analysis_service as analysis_service
import enzywizard_agent.nodes.history_query as history_query_node_module
import enzywizard_agent.nodes.request_routing as request_routing_node_module
from enzywizard_agent.services.request_routing_service import _resolve_turn_selection
from enzywizard_agent.services.user_input_service import record_next_user_input
from enzywizard_agent.state import AgentState
from enzywizard_agent.utils.analysis_utils import (
    build_report_evidence,
    summarize_report_artifact,
)
from enzywizard_agent.utils.checkpoint_utils import create_in_memory_checkpointer
from enzywizard_agent.utils.tool_plan_utils import (
    build_single_protein_tool_plan,
    build_wildtype_mutant_tool_plan,
    find_reusable_docking_output,
)


REPORT_TYPE_TO_ARTIFACT = {
    "enzywizard_clean": "clean_report_json",
    "enzywizard_mut_clean": "mut_clean_report_json",
    "enzywizard_aaprops": "aaprops_report_json",
    "enzywizard_hydrocluster": "hydrocluster_report_json",
    "enzywizard_energy": "energy_report_json",
    "enzywizard_flexibility": "flexibility_report_json",
    "enzywizard_disorder": "disorder_report_json",
    "enzywizard_conservation": "conservation_report_json",
    "enzywizard_embedding": "embedding_report_json",
    "enzywizard_pocket": "pocket_report_json",
    "enzywizard_substrate": "substrate_report_json",
    "enzywizard_dock": "dock_report_json",
    "enzywizard_interaction": "interaction_report_json",
    "enzywizard_integrate": "integrate_report_json",
    "enzywizard_mut_integrate": "mut_integrate_report_json",
}


def require_passed_workflow_case(
    results_payload: dict,
    case_id: str,
) -> dict:
    """返回一个已通过且具有执行记录的真实工作流案例。"""

    case = next(
        (
            item
            for item in results_payload.get("results", [])
            if item["case_id"] == case_id
        ),
        None,
    )
    if case is None:
        raise AssertionError(f"Required workflow case is missing: {case_id}")
    if case.get("status") != "passed" or case.get("tool_execution") is None:
        raise AssertionError(
            f"Required workflow case did not pass and cannot be used as an Analysis "
            f"fixture: {case_id}."
        )
    return case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regression tests for Analysis, history queries, and turn state."
    )
    parser.add_argument(
        "--run_dir",
        default=(
            "/home/wmk/enzywizard-agent-full-workflow-tests/"
            "run_20261015T111644Z_aff99307"
        ),
        help="Run directory containing outputs from the 25 real workflows.",
    )
    return parser.parse_args()


def infer_side(path: Path, report_type: str) -> str:
    text = str(path).casefold()
    if report_type in {"enzywizard_mut_clean", "enzywizard_mut_integrate"}:
        return "shared"
    if any(
        token in text
        for token in ("_wt/", "/wt/", "energy_wt", "pocket_wt", "flexibility_wt")
    ):
        return "wt"
    if any(
        token in text
        for token in ("_mut/", "/mut/", "energy_mut", "pocket_mut", "flexibility_mut")
    ):
        return "mut"
    return "single"


def collect_representative_reports(run_dir: Path) -> list[Artifact]:
    """每个 report_type/side 只保留一个真实文件，避免测试重复 staging 副本。"""

    selected: dict[tuple[str, str], Artifact] = {}
    for path in run_dir.rglob("*.json"):
        if "input_reports" in path.parts or path.name == "results.json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        report_type = payload.get("report_type")
        if report_type not in REPORT_TYPE_TO_ARTIFACT:
            continue
        side = infer_side(path, report_type)
        key = (report_type, side)
        selected.setdefault(
            key,
            Artifact(
                turn_id=1,
                plan_id=1,
                artifact_type=REPORT_TYPE_TO_ARTIFACT[report_type],
                path=str(path.resolve()),
                source_tool=report_type.replace("_", "-"),
                side=side,
                metadata={"test_source": "real_workflow_output"},
            ),
        )
    return list(selected.values())


def empty_cleaned_files() -> CleanedFiles:
    return CleanedFiles(
        single_clean_output_dir=None,
        single_cleaned_structure_file=None,
        single_cleaned_cif_file=None,
        single_cleaned_pdb_file=None,
        single_cleaned_fasta_file=None,
        single_clean_report_file=None,
        mut_clean_output_dir=None,
        wt_cleaned_structure_file=None,
        wt_cleaned_cif_file=None,
        wt_cleaned_pdb_file=None,
        wt_cleaned_fasta_file=None,
        mut_cleaned_structure_file=None,
        mut_cleaned_cif_file=None,
        mut_cleaned_pdb_file=None,
        mut_cleaned_fasta_file=None,
        mut_clean_report_file=None,
        cleaned_amino_acid_substitution=None,
    )


def routing_result(
    turn_id: int,
    request_type: str,
    actions: list[str],
    *,
    requires_tools: bool = False,
    references_history: bool = False,
    reuse_policy: str = "none",
) -> RequestRoutingResult:
    output = RequestRoutingOutput(
        request_type=request_type,
        actions=[
            RequestedAction(action_type=action, target="regression-test target")
            for action in actions
        ],
        requires_tool_execution=requires_tools,
        references_history=references_history,
        requires_history_artifacts=references_history,
        reuse_policy=reuse_policy,
        requests_recomputation=False,
        clarification_question=None,
        routing_reason="Structured regression-test route.",
    )
    route = (
        "request_to_history_query"
        if references_history
        else "request_to_planning"
        if requires_tools
        else "request_to_analysis"
    )
    return RequestRoutingResult(
        turn_id=turn_id,
        routing_status="succeeded",
        route=route,
        output=output,
        summary=f"Regression route: {request_type}.",
        error_message="",
    )


def history_result(
    turn_id: int,
    artifacts: list[Artifact],
    operations: list[str],
    *,
    categories: list[str] | None = None,
) -> HistoryQueryResult:
    query = HistoryQueryOutput(
        turn_scope="specific",
        referenced_turn_ids=[1],
        artifact_categories=categories or ["all"],
        artifact_types=[],
        report_types=[],
        sides=[],
        operations=operations,
        query_reason="Structured regression-test history query.",
        clarification_question=None,
    )
    return HistoryQueryResult(
        turn_id=turn_id,
        query_status="succeeded" if artifacts else "not_found",
        query=query,
        resolved_turn_ids=[1],
        selected_artifacts=artifacts,
        summary=(
            f"Selected {len(artifacts)} test artifacts."
            if artifacts
            else "No test artifact matched the structured query."
        ),
        error_message="",
    )


def build_state(run_dir: Path, artifacts: list[Artifact]) -> AgentState:
    routing = routing_result(
        2,
        "history_analysis",
        ["analyze_history"],
        references_history=True,
    )
    history = history_result(2, artifacts, ["explain"])
    return {
        "config": AgentConfig(
            api_key="not-used",
            base_url="https://example.invalid/v1",
            model="not-used",
            workdir=run_dir,
            max_turns=3,
        ),
        "session_id": "analysis-test",
        "agent_inputs": AgentInputs(
            structure_file=None,
            msa_file=None,
            mut_structure_file=None,
            mut_msa_file=None,
            substrate_name=None,
            amino_acid_substitution=None,
        ),
        "current_turn_id": 2,
        "current_plan_id": 0,
        "current_user_message": "Explain the energy results from the previous turn.",
        "user_messages": [UserMessage(turn_id=1, content="Run energy analysis.")],
        "user_message_summary": "Turn 1: Run energy analysis.",
        "current_request_routing": routing,
        "request_routing_results": [routing],
        "request_routing_summary": routing.summary,
        "current_history_query": history,
        "history_query_results": [history],
        "history_query_summary": history.summary,
        "current_history_artifacts": artifacts,
        "planning_results": [],
        "planning_summary": "",
        "user_confirmations": [],
        "user_confirmation_summary": "",
        "tool_executions": [],
        "tool_calling_summary": "",
        "analysis_results": [],
        "analysis_summary": "",
        "cleaned_files": empty_cleaned_files(),
        "current_plan_artifacts": [],
        "artifacts": artifacts,
        "artifact_summary": "",
    }


def fail_if_model_called(_state: AgentState):
    raise RuntimeError("model disabled by analysis regression test")


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        print(f"[FAIL] Real workflow directory does not exist: {run_dir}")
        return 1

    artifacts = collect_representative_reports(run_dir)
    observed_types = {
        json.loads(Path(item.path).read_text(encoding="utf-8"))["report_type"]
        for item in artifacts
    }
    missing_types = sorted(set(REPORT_TYPE_TO_ARTIFACT) - observed_types)
    if missing_types:
        print(f"[FAIL] Missing report types: {missing_types}")
        return 1
    print(f"[PASS] Found all {len(REPORT_TYPE_TO_ARTIFACT)} report types.")

    summaries = [summarize_report_artifact(item) for item in artifacts]
    for summary in summaries:
        if len(summary.key_metrics) > 60:
            raise AssertionError(
                f"{summary.report_type} exceeded the metric compression limit"
            )
        if any(isinstance(value, (dict, list)) for value in summary.key_metrics.values()):
            raise AssertionError(
                f"{summary.report_type} still contains an uncompressed object or array"
            )
    print(f"[PASS] Parsed and compressed {len(summaries)} representative reports.")

    energy_summaries = [
        item
        for item in summaries
        if item.report_type == "enzywizard_energy" and item.side in {"wt", "mut"}
    ]
    evidence, findings = build_report_evidence(energy_summaries)
    if not any(item.inference_level == "calculated" for item in evidence):
        raise AssertionError("No deterministic WT/MUT difference evidence was generated")
    if not any(item.finding_kind == "comparison" for item in findings):
        raise AssertionError("No WT/MUT comparison finding was generated")
    print("[PASS] WT/MUT metrics produced direct and MUT-WT calculated evidence.")

    # 同一 scope 内完全同名的独立报告和整合报告指标不一致时，必须产生
    # 程序计算的 warning；不同字段名不参与猜测式比较。
    consistency_summaries = [
        ReportSummary(
            turn_id=1,
            plan_id=1,
            artifact_type="energy_report_json",
            report_type="enzywizard_energy",
            source_tool="enzywizard-energy",
            side="single",
            path="/tmp/energy.json",
            key_metrics={"total_potential_energy": -100.0},
            highlights=["total_potential_energy=-100.0"],
            warnings=[],
        ),
        ReportSummary(
            turn_id=1,
            plan_id=1,
            artifact_type="integrate_report_json",
            report_type="enzywizard_integrate",
            source_tool="enzywizard-integrate",
            side="single",
            path="/tmp/integrate.json",
            key_metrics={"total_potential_energy": -90.0},
            highlights=["total_potential_energy=-90.0"],
            warnings=[],
        ),
    ]
    consistency_evidence, consistency_findings = build_report_evidence(
        consistency_summaries
    )
    if not any(item.evidence_id.startswith("consistency-") for item in consistency_evidence):
        raise AssertionError("Cross-report inconsistency evidence was not generated")
    if not any(item.finding_kind == "warning" for item in consistency_findings):
        raise AssertionError("Cross-report inconsistency warning was not generated")
    print("[PASS] Cross-report metric inconsistencies produce traceable warnings.")

    specific_selection = HistoryTurnSelectionDraft(
        action_id=0,
        turn_scope="specific",
        turn_ids=[1],
    )
    previous_selection = specific_selection.model_copy(
        update={"turn_scope": "previous", "turn_ids": []}
    )
    if _resolve_turn_selection(specific_selection, [1])[0] != [1]:
        raise AssertionError("Structured explicit turn resolution failed")
    if _resolve_turn_selection(previous_selection, [1])[0] != [1]:
        raise AssertionError("Structured previous-turn resolution failed")
    print("[PASS] Structured explicit and relative history selectors were resolved.")

    state = build_state(run_dir, artifacts)
    original_create_llm = analysis_service._create_llm
    analysis_service._create_llm = fail_if_model_called
    try:
        result = analysis_service.analyze_agent_state(state)
    finally:
        analysis_service._create_llm = original_create_llm
    if result.response_mode != "scientific_result" or not result.report_summaries:
        raise AssertionError("Historical report analysis did not return scientific_result")
    if result.analysis_status != "partial":
        raise AssertionError(
            "A disabled model should return a partial deterministic result"
        )
    print("[PASS] Historical reports return a deterministic summary without a model.")

    state["current_user_message"] = "Which energy report files were generated last turn?"
    energy_artifacts = [
        artifact
        for artifact in artifacts
        if artifact.artifact_type == "energy_report_json"
    ]
    artifact_routing = routing_result(
        2,
        "artifact_query",
        ["list_artifacts"],
        references_history=True,
    )
    state["current_request_routing"] = artifact_routing
    state["current_history_query"] = history_result(
        2,
        energy_artifacts,
        ["list_artifacts"],
        categories=["energy"],
    )
    artifact_result = analysis_service.analyze_agent_state(state)
    if artifact_result.response_mode != "artifact_query":
        raise AssertionError("The file query did not enter artifact_query mode")
    if not artifact_result.referenced_artifacts:
        raise AssertionError("The file query returned no artifact references")
    print("[PASS] Cross-turn file queries return paths and existence status.")

    state["current_user_message"] = "exit"
    state["current_request_routing"] = routing_result(
        2,
        "exit",
        ["exit_session"],
    )
    state.pop("current_history_query", None)
    exit_result = analysis_service.analyze_agent_state(state)
    if exit_result.analysis_route != "exit_requested":
        raise AssertionError("The exit request did not produce exit_requested")
    print("[PASS] An explicit exit request terminates the graph.")

    state["current_user_message"] = "The previous turn is complete."
    state["current_analysis_result"] = result
    state["current_plan_artifacts"] = artifacts[:2]
    state["current_plan_id"] = 4
    record_next_user_input(
        state,
        {"turn_id": 3, "message": "Show the files in the next turn."},
    )
    if state["current_turn_id"] != 3 or state["current_plan_id"] != 0:
        raise AssertionError("The next turn ID or plan_id was reset incorrectly")
    if state["current_plan_artifacts"]:
        raise AssertionError("The next turn retained prior current_plan_artifacts")
    if "current_analysis_result" in state:
        raise AssertionError("The next turn retained the prior current_analysis_result")
    if not state["artifacts"]:
        raise AssertionError("Session artifacts must not be cleared between turns")
    print("[PASS] Turn reset preserves fixed inputs, history, and session artifacts.")

    results_payload = json.loads(
        (run_dir / "results.json").read_text(encoding="utf-8")
    )
    docking_case = require_passed_workflow_case(
        results_payload,
        "single_substrate_docking_workflow",
    )
    docking_execution = ToolExecution.model_validate(docking_case["tool_execution"])
    docking_cleaned_files = CleanedFiles.model_validate(docking_case["cleaned_files"])
    docking_state = build_state(run_dir, [])
    docking_state["current_user_message"] = (
        "Reuse the previous output and only rerun protein-substrate interactions."
    )
    docking_state["agent_inputs"] = AgentInputs(
        structure_file=None,
        msa_file=None,
        mut_structure_file=None,
        mut_msa_file=None,
        substrate_name="glucose",
        amino_acid_substitution=None,
    )
    docking_state["cleaned_files"] = docking_cleaned_files
    docking_state["tool_executions"] = [docking_execution]
    docking_artifacts = [
        artifact
        for call in docking_execution.call_executions
        for artifact in call.artifacts
    ]
    docking_state["artifacts"] = docking_artifacts
    docking_state["current_request_routing"] = routing_result(
        2,
        "mixed",
        ["reuse_historical_output", "run_tool_workflow"],
        requires_tools=True,
        references_history=True,
        reuse_policy="required",
    )
    docking_state["current_history_query"] = history_result(
        2,
        docking_artifacts,
        ["reuse_as_input"],
        categories=["dock"],
    )
    reusable = find_reusable_docking_output(
        docking_state,
        "single",
        docking_cleaned_files.single_cleaned_cif_file,
    )
    if reusable is None:
        raise AssertionError("A compatible historical Dock output was not detected")
    reuse_plan = build_single_protein_tool_plan(
        docking_state,
        turn_id=2,
        plan_id=1,
        selected_tool_names=[
            "enzywizard-substrate",
            "enzywizard-dock",
            "enzywizard-interaction",
            "enzywizard-integrate",
        ],
        plan_reason="Regression test for explicit historical output reuse.",
    )
    reuse_tools = [call.tool_name for call in reuse_plan.calls]
    if reuse_tools != ["enzywizard-interaction", "enzywizard-integrate"]:
        raise AssertionError(
            f"The historical Dock reuse plan contains extra steps: {reuse_tools}"
        )
    interaction_arguments = reuse_plan.calls[0].arguments
    if interaction_arguments.get("substrate_names") != "docked_glucose":
        raise AssertionError("The historical Dock SDF stem was not passed to Interaction")
    if not any(
        rule.source_step_index is None
        and "dock_report" in rule.source_path_pattern
        for rule in reuse_plan.calls[-1].staging_rules
    ):
        raise AssertionError("Integrate did not stage the historical Dock report")

    substrate_artifacts = [
        artifact
        for artifact in docking_artifacts
        if artifact.artifact_type in {"substrate_report_json", "substrate_sdf"}
    ]
    docking_state["current_history_query"] = history_result(
        2,
        substrate_artifacts,
        ["reuse_as_input"],
        categories=["substrate"],
    )
    substrate_reuse_plan = build_single_protein_tool_plan(
        docking_state,
        turn_id=2,
        plan_id=2,
        selected_tool_names=[
            "enzywizard-substrate",
            "enzywizard-dock",
            "enzywizard-integrate",
        ],
        plan_reason="Regression test for historical substrate reuse.",
    )
    substrate_reuse_tools = [
        call.tool_name for call in substrate_reuse_plan.calls
    ]
    if substrate_reuse_tools != ["enzywizard-dock", "enzywizard-integrate"]:
        raise AssertionError(
            "Historical substrate reuse did not remove substrate preparation: "
            f"{substrate_reuse_tools}"
        )
    if not any(
        rule.source_step_index is None
        and "substrate_report" in rule.source_path_pattern
        for rule in substrate_reuse_plan.calls[-1].staging_rules
    ):
        raise AssertionError("Integrate did not stage the historical substrate report")

    docking_state["agent_inputs"].substrate_name = "fructose"
    if find_reusable_docking_output(
        docking_state,
        "single",
        docking_cleaned_files.single_cleaned_cif_file,
    ) is not None:
        raise AssertionError("A different substrate must not reuse historical Dock output")

    mutant_docking_case = require_passed_workflow_case(
        results_payload,
        "mutant_substrate_docking_workflow",
    )
    mutant_execution = ToolExecution.model_validate(
        mutant_docking_case["tool_execution"]
    )
    mutant_cleaned_files = CleanedFiles.model_validate(
        mutant_docking_case["cleaned_files"]
    )
    mutant_artifacts = [
        artifact
        for call in mutant_execution.call_executions
        for artifact in call.artifacts
    ]
    mutant_substrate_artifacts = [
        artifact
        for artifact in mutant_artifacts
        if artifact.artifact_type in {"substrate_report_json", "substrate_sdf"}
    ]
    mutant_reuse_state = build_state(run_dir, mutant_artifacts)
    mutant_reuse_state["agent_inputs"] = AgentInputs(
        structure_file=None,
        msa_file=None,
        mut_structure_file=None,
        mut_msa_file=None,
        substrate_name="glucose",
        amino_acid_substitution=None,
    )
    mutant_reuse_state["cleaned_files"] = mutant_cleaned_files
    mutant_reuse_state["tool_executions"] = [mutant_execution]
    mutant_reuse_state["current_request_routing"] = routing_result(
        2,
        "mixed",
        ["reuse_historical_output", "run_tool_workflow"],
        requires_tools=True,
        references_history=True,
        reuse_policy="required",
    )
    mutant_reuse_state["current_history_query"] = history_result(
        2,
        mutant_substrate_artifacts,
        ["reuse_as_input"],
        categories=["substrate"],
    )
    paired_substrate_plan = build_wildtype_mutant_tool_plan(
        mutant_reuse_state,
        turn_id=2,
        plan_id=1,
        selected_tool_names=[
            "enzywizard-substrate",
            "enzywizard-dock",
            "enzywizard-mut-integrate",
        ],
        plan_reason="Regression test for paired historical substrate reuse.",
    )
    paired_reuse_tools = [call.tool_name for call in paired_substrate_plan.calls]
    if paired_reuse_tools != [
        "enzywizard-dock",
        "enzywizard-dock",
        "enzywizard-mut-integrate",
    ]:
        raise AssertionError(
            f"Paired historical substrate reuse contains extra steps: {paired_reuse_tools}"
        )
    substrate_staging = [
        rule
        for rule in paired_substrate_plan.calls[-1].staging_rules
        if rule.source_step_index is None
        and "substrate_report" in rule.source_path_pattern
    ]
    if {rule.side for rule in substrate_staging} != {"wt", "mut"}:
        raise AssertionError(
            "Paired integration did not stage the historical substrate report "
            "for both sides"
        )
    print(
        "[PASS] Substrate and Dock reuse validate compatibility and report staging."
    )

    rejection_state = build_state(run_dir, [])
    rejection_state["current_turn_id"] = 3
    rejection_state["current_user_message"] = "Run energy analysis."
    rejection_state["current_user_confirmation"] = UserConfirmation(
        turn_id=3,
        plan_id=1,
        decision="rejected",
        revision_request=None,
    )
    rejection_state["current_request_routing"] = routing_result(
        3,
        "tool_workflow",
        ["run_tool_workflow"],
        requires_tools=True,
    )
    analysis_node(rejection_state)
    rejected_analysis = rejection_state["current_analysis_result"]
    if rejected_analysis.response_mode != "plan_rejected":
        raise AssertionError("Plan rejection did not produce a plan_rejected response")
    if (
        rejected_analysis.analysis_route != "exit_requested"
        or rejected_analysis.termination_reason != "max_turns"
    ):
        raise AssertionError("Final-turn rejection did not terminate at max_turns")
    print("[PASS] Final-turn plan rejection returns an answer and ends at max_turns.")

    failure_state = build_state(run_dir, [])
    failed_call = ToolCallExecution(
        turn_id=2,
        plan_id=1,
        step_index=1,
        tool_name="enzywizard-pocket",
        side="single",
        execution_status="failed",
        command=["enzywizard-pocket"],
        resolved_arguments={},
        exit_code=0,
        duration_seconds=0.1,
        stdout_file=str(run_dir / "stdout.txt"),
        stderr_file=str(run_dir / "stderr.txt"),
        log_files=[],
        stdout_excerpt="",
        stderr_excerpt="",
        log_excerpt="",
        artifacts=[],
        error_message=(
            "enzywizard-pocket completed without required outputs: "
            "pocket_report_json."
        ),
    )
    failure_state["current_tool_execution"] = ToolExecution(
        turn_id=2,
        plan_id=1,
        execution_status="failed",
        call_executions=[failed_call],
        failed_step_index=1,
        error_messages=[failed_call.error_message],
        summary="Tool execution stopped at step 1.",
    )
    failure_state["current_request_routing"] = routing_result(
        2,
        "tool_workflow",
        ["run_tool_workflow"],
        requires_tools=True,
    )
    failure_state.pop("current_history_query", None)
    failure_result = analysis_service.analyze_agent_state(failure_state)
    if failure_result.response_mode != "tool_execution_failure":
        raise AssertionError("Tool failure did not enter tool_execution_failure mode")
    if not any(item.source_kind == "knowledge_base" for item in failure_result.evidence):
        raise AssertionError("The explicit error did not match a local failure rule")
    print("[PASS] Tool failure retains evidence and matches local failure guidance.")

    def fake_route_user_request(state: AgentState) -> RequestRoutingResult:
        if state.get("current_turn_id") == 3:
            return routing_result(3, "exit", ["exit_session"])
        return routing_result(
            2,
            "artifact_query",
            ["list_artifacts"],
            references_history=True,
        )

    def fake_plan_history_query(state: AgentState) -> HistoryQueryResult:
        selected = [
            artifact
            for artifact in state.get("artifacts", [])
            if artifact.artifact_type == "energy_report_json"
        ]
        return history_result(
            state.get("current_turn_id", 2),
            selected,
            ["list_artifacts"],
            categories=["energy"],
        )

    original_router = request_routing_node_module.route_user_request
    original_history_planner = history_query_node_module.plan_history_query
    request_routing_node_module.route_user_request = fake_route_user_request
    history_query_node_module.plan_history_query = fake_plan_history_query
    try:
        graph = build_graph(checkpointer=create_in_memory_checkpointer())
        print("[PASS] The full graph compiles with the restricted InMemorySaver.")

        # 图执行结构化历史文件查询后停在 user_input interrupt；恢复后 Router
        # 将下一轮识别为 exit，验证完整的跨轮恢复路径。
        graph_state = build_state(run_dir, artifacts)
        graph_state["current_user_message"] = (
            "Which energy report files were generated last turn?"
        )
        graph_config = {"configurable": {"thread_id": graph_state["session_id"]}}
        interrupted = graph.invoke(graph_state, config=graph_config)
        interrupts = interrupted.get("__interrupt__", [])
        if len(interrupts) != 1 or interrupts[0].value.get("interrupt_type") != "user_input":
            raise AssertionError(
                "The graph did not enter user_input interrupt after the history answer"
            )
        completed = graph.invoke(
            Command(
                resume={
                    "turn_id": interrupts[0].value["next_turn_id"],
                    "message": "exit",
                }
            ),
            config=graph_config,
        )
        if completed.get("__interrupt__"):
            raise AssertionError("The graph must not interrupt again after exit")
        final_analysis = completed.get("current_analysis_result")
        if final_analysis is None or final_analysis.analysis_route != "exit_requested":
            raise AssertionError("The resumed graph did not terminate correctly")
        if completed.get("current_turn_id") != 3:
            raise AssertionError("The resumed graph has an incorrect turn ID")
        print("[PASS] The graph completed history query, resume, and exit flow.")
    finally:
        request_routing_node_module.route_user_request = original_router
        history_query_node_module.plan_history_query = original_history_planner
    print("\n[ALL PASSED] Analysis and cross-turn state regression checks completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
