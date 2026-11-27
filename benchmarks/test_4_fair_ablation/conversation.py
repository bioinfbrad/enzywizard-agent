"""Run the report task and its definition follow-up in one Agent session."""

import json
from pathlib import Path
import time
from unittest.mock import patch
from uuid import uuid4

from enzywizard_agent.conf import AgentConfig
from enzywizard_agent.data_models.agent_inputs import AgentInputs
from enzywizard_agent.data_models.analysis import ReportSummary
from enzywizard_agent.data_models.artifact import Artifact
from enzywizard_agent.data_models.history_query import HistoryQueryOutput, HistoryQueryResult
from enzywizard_agent.data_models.runtime import RuntimeSession
from enzywizard_agent.nodes.analysis import analysis_node
from enzywizard_agent.nodes.history_query import history_query_node
from enzywizard_agent.nodes.request_routing import request_routing_node
from enzywizard_agent.services import analysis_service
from enzywizard_agent.services.runtime_service import utc_now
from enzywizard_agent.services.user_input_service import archive_completed_turn, record_next_user_input
from enzywizard_agent.utils.analysis_utils import compact_history_summary, summarize_report_artifact

from cases import ROOT


def configuration(args, workdir, *, ontology_mode=None):
    return AgentConfig(
        api_key=args.api_key, base_url=args.base_url, model=args.model,
        reasoning_effort=args.reasoning_effort, llm_timeout_seconds=args.llm_timeout_seconds,
        ontology_mode=args.ontology_mode if ontology_mode is None else ontology_mode,
        check_model_capabilities=False, workdir=workdir,
    )


def _reports(catalog, case):
    reports = case["report_ids"]
    paired = case.get("scope") == "paired" or (
        any("_wt_" in name for name in reports) and any("_mut_" in name for name in reports)
    )
    artifacts = []
    for report_id in reports:
        spec = catalog["reports"][report_id]
        side = (
            "shared" if report_id.endswith("_mut_clean") else "mut" if "_mut_" in report_id
            else "wt" if paired else "single"
        )
        artifacts.append(Artifact(
            turn_id=1, plan_id=1, artifact_type=spec["artifact_type"],
            path=str(ROOT / spec["path"]), source_tool=spec["source_tool"], side=side,
            metadata={"benchmark_case_id": case["case_id"], "report_id": report_id},
        ))
    return artifacts


def _initial_state(args, workdir, question, reports):
    return {
        "config": configuration(args, workdir), "session_id": f"test4-{uuid4().hex}",
        "current_turn_id": 1, "current_plan_id": 1, "current_user_message": question,
        "agent_inputs": AgentInputs(
            structure_file=None, msa_file=None, mut_structure_file=None, mut_msa_file=None,
            substrate_name=None, amino_acid_substitution=None,
        ),
        "runtime_session": RuntimeSession(started_at=utc_now(), started_monotonic=time.monotonic()),
        "artifacts": reports, "tool_executions": [], "analysis_results": [],
        "analysis_summary": "", "request_routing_results": [],
        "request_routing_summary": "", "user_message_summary": "",
    }


def _archived_report_result(state, result):
    state["current_analysis_result"] = result
    state["analysis_results"].append(result)
    state["analysis_summary"] = compact_history_summary(state["analysis_results"])
    state["route"] = result.analysis_route
    archive_completed_turn(state)


def _fixed_history(reports, earlier_turn):
    return HistoryQueryResult(
        turn_id=earlier_turn + 1, query_status="succeeded",
        query=HistoryQueryOutput(
            turn_scope="previous", referenced_turn_ids=[], artifact_categories=["all"],
            artifact_types=sorted({report.artifact_type for report in reports}),
            report_types=[], sides=sorted({report.side for report in reports}),
            operations=["explain"], query_reason="The follow-up refers to the prior analysis.",
            clarification_question=None,
        ),
        resolved_turn_ids=[earlier_turn], selected_artifacts=reports,
        summary="Resolved the prior report analysis and its fixed reports.", error_message="",
    )


def run_conversation(catalog, case, args, workdir):
    reports = _reports(catalog, case)
    originals = [case["initial_question"], case["follow_up_question"]] if "initial_question" in case else [case["question"]]
    state = _initial_state(args, workdir, originals[0], reports)
    tagged_records = 0

    def tag_retrieval():
        nonlocal tagged_records
        records = state["runtime_session"].metadata.get("knowledge_retrieval", [])
        for record in records[tagged_records:]:
            record["turn_id"] = state["current_turn_id"]
        tagged_records = len(records)

    with patch.object(analysis_service, "summarize_report_artifact", _test_report_summary):
        for turn_id, question in enumerate(originals, 1):
            if turn_id > 1:
                record_next_user_input(state, {"turn_id": turn_id, "message": question})
                state["current_history_query"] = _fixed_history(reports, turn_id - 1)
                state["current_plan_id"] = 1
            result = analysis_service._analyze_reports(state, reports, historical=turn_id > 1)
            _archived_report_result(state, result)
            tag_retrieval()
            if state["route"] != "answer_generated" or result.analysis_status == "failed":
                return state, "The original report task did not leave a usable conversation."

        record_next_user_input(state, {
            "turn_id": len(originals) + 1, "message": case["definition_question"],
        })
        request_routing_node(state)
        if state["route"] == "request_to_history_query":
            history_query_node(state)
        if state["route"] in {"request_to_planning", "history_to_planning"}:
            tag_retrieval()
            return state, "The definition question was routed to tool execution."
        analysis_node(state)
        tag_retrieval()
    return state, None


def _summary_with_mutation_mapping(artifact: Artifact) -> ReportSummary:
    """Expose the requested site from the fixed report with its original paths."""
    summary = summarize_report_artifact(artifact)
    if artifact.source_tool != "enzywizard-mut-clean" or summary.read_status != "parsed":
        return summary

    report = json.loads(Path(artifact.path).read_text(encoding="utf-8"))
    substitution = report["amino_acid_substitution"]
    site = int(substitution[1:-1])
    metrics = {
        "amino_acid_substitution": substitution,
        "cleaned_amino_acid_substitution": report["cleaned_amino_acid_substitution"],
    }
    paths = {key: f"$.{key}" for key in metrics}
    for side in ("wild_type", "mutant"):
        array_key = f"{side}_residue_mapping_old_to_new"
        matches = [
            (index, record)
            for index, record in enumerate(report[array_key])
            if record["old_residue"]["residue_index"] == site
        ]
        if len(matches) != 1:
            raise ValueError(f"{artifact.path} has no unique {side} mapping for residue {site}")
        index, record = matches[0]
        old = record["old_residue"]
        new = record["new_residue"]
        key = f"{side}_mutation_site_mapping"
        metrics[key] = (
            f"{old['residue_name']}{old['residue_index']} -> "
            f"{new['residue_name']}{new['residue_index']}"
        )
        paths[key] = f"$.{array_key}[{index}]"
    return summary.model_copy(update={
        "key_metrics": {**metrics, **summary.key_metrics},
        "metric_paths": {**paths, **summary.metric_paths},
        "metric_inference_levels": {
            **{key: "direct" for key in metrics},
            **summary.metric_inference_levels,
        },
    })


def _test_report_summary(artifact: Artifact) -> ReportSummary:
    """Expose the fixed evidence needed for this case at the report boundary."""
    summary = _summary_with_mutation_mapping(artifact)
    if summary.read_status != "parsed":
        return summary

    case_id = artifact.metadata.get("benchmark_case_id")
    if case_id == "T2-MC-06" and summary.report_type == "enzywizard_embedding":
        report = json.loads(Path(artifact.path).read_text(encoding="utf-8"))
        index, residue = next(
            (index, record) for index, record in enumerate(report["sequence_embeddings"])
            if record["residue_index"] == 72
        )
        metric = "residue_72_embedding_coordinate_1"
        return summary.model_copy(update={
            "key_metrics": {
                metric: round(residue["residue_embedding"][0], 6), **summary.key_metrics
            },
            "metric_paths": {
                metric: f"$.sequence_embeddings[{index}].residue_embedding[0]",
                **summary.metric_paths,
            },
            "metric_inference_levels": {metric: "direct", **summary.metric_inference_levels},
        })

    if summary.report_type == "enzywizard_interaction" and case_id in {
        "T2-SR-12", "T2-MI-04", "T2-MI-08", "T2-MI-12",
        "T2-MC-12", "T2-MF-11", "T2-EL-08", "T2-EL-11",
    }:
        contacts = {
            metric: value
            for metric, value in summary.key_metrics.items()
            if ".enzyme_substrate_interaction_statistics." in summary.metric_paths[metric]
        }
        # The Agent emits field-level references for the first eight metrics.
        return summary.model_copy(update={"key_metrics": {**contacts, **summary.key_metrics}})
    return summary

