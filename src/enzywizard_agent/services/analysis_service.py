from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
import yaml

from ..builders.analysis_context_builder import (
    build_general_answer_context,
    build_scientific_analysis_context,
)
from ..data_models.analysis import (
    AnalysisArtifactReference,
    AnalysisClaimCitation,
    AnalysisErrorGuidance,
    AnalysisErrorType,
    AnalysisEvidence,
    AnalysisFinding,
    AnalysisNarrativeOutput,
    AnalysisResult,
    NarrativeClaimCitation,
    ReportSummary,
)
from ..data_models.artifact import Artifact
from ..prompts.analysis_prompt import (
    GENERAL_ANSWER_SYSTEM_PROMPT,
    SCIENTIFIC_RESULT_SYSTEM_PROMPT,
)
from ..state import AgentState
from ..utils.CLI_utils import _humanize_field_path
from ..utils.analysis_utils import (
    artifact_to_reference,
    build_report_evidence,
    select_report_artifacts,
    summarize_report_artifact,
    unique_wt_mut_report_pairs,
)
from .llm_service import create_chat_model_for_state, invoke_structured, invoke_text
from ..utils.terminology_utils import (
    CLAIM_EVIDENCE_APPENDIX_MARKER,
    TERMINOLOGY_APPENDIX_MARKER,
    annotate_terminology,
    claim_evidence_number,
    order_answer_markers,
    scientific_answer_body,
    terminology_number,
)
from .knowledge_retrieval_service import (
    fair_terms_sha256,
    load_fair_terms,
    retrieve_failure_hints,
)


def _create_llm(state: AgentState):
    return create_chat_model_for_state(state)


def _empty_result(
    state: AgentState,
    *,
    response_mode: str,
    summary: str,
    final_answer: str,
    analysis_status: str = "succeeded",
    analysis_route: str = "answer_generated",
    termination_reason: str | None = None,
    report_summaries: list[ReportSummary] | None = None,
    findings: list[AnalysisFinding] | None = None,
    evidence: list[AnalysisEvidence] | None = None,
    referenced_artifacts: list[AnalysisArtifactReference] | None = None,
    limitations: list[str] | None = None,
    error_messages: list[str] | None = None,
    used_language_model: bool = False,
    error_guidance: AnalysisErrorGuidance | None = None,
    claim_citations: list[AnalysisClaimCitation] | None = None,
) -> AnalysisResult:
    """集中填充 AnalysisResult，避免各响应分支遗漏状态字段。"""

    return AnalysisResult(
        turn_id=state.get("current_turn_id", 1),
        response_mode=response_mode,
        analysis_status=analysis_status,
        analysis_route=analysis_route,
        termination_reason=termination_reason,
        summary=summary,
        final_answer=final_answer,
        report_summaries=report_summaries or [],
        findings=findings or [],
        evidence=evidence or [],
        referenced_artifacts=referenced_artifacts or [],
        limitations=limitations or [],
        error_messages=error_messages or [],
        used_language_model=used_language_model,
        error_guidance=error_guidance,
        claim_citations=claim_citations,
    )


_INPUT_OPTION_BY_DESCRIPTION = {
    "a protein structure": "--structure_file",
    "a wild-type protein structure": "--structure_file",
    "a mutant protein structure": "--mut_structure_file",
    "a multiple sequence alignment": "--msa_file",
    "a wild-type multiple sequence alignment": "--msa_file",
    "a mutant multiple sequence alignment": "--mut_msa_file",
    "a substrate name or smiles": "--substrate_name",
    "an amino-acid substitution": "--amino_acid_substitution",
}


def _format_error_answer(
    guidance: AnalysisErrorGuidance,
    *,
    detail: str | None = None,
) -> str:
    """Render the structured error contract in a readable, stable form."""

    session_status = (
        "cannot continue; restart is required"
        if guidance.restart_required
        else "can continue after the action below"
    )
    lines = [
        f"Error type: `{guidance.error_type}`",
        f"Stage: `{guidance.stage}`",
        f"Current session: {session_status}.",
        f"Reason: {guidance.reason}",
    ]
    if detail and detail.strip() and detail.strip() != guidance.reason:
        lines.extend(["", "Details:", detail.strip()])
    lines.extend(["", "Required action:"])
    lines.extend(f"- {action}" for action in guidance.user_actions)
    if guidance.evidence_paths:
        lines.extend(["", "Diagnostic evidence:"])
        lines.extend(f"- {path}" for path in guidance.evidence_paths)
    return "\n".join(lines)


def _missing_input_guidance(missing_inputs: list[str]) -> AnalysisErrorGuidance:
    """Convert immutable startup-input gaps into explicit restart instructions."""

    normalized = list(dict.fromkeys(item.strip() for item in missing_inputs if item.strip()))
    if not normalized:
        normalized = ["a required workflow input"]
    options = list(
        dict.fromkeys(
            _INPUT_OPTION_BY_DESCRIPTION[item.casefold()]
            for item in normalized
            if item.casefold() in _INPUT_OPTION_BY_DESCRIPTION
        )
    )
    reason = "Missing required startup input: " + ", ".join(normalized) + "."
    if options:
        option_text = ", ".join(f"`{option} <value>`" for option in options)
        correction = (
            f"Restart EnzyWizard-Agent with {option_text} and the original request."
        )
    else:
        correction = (
            "Correct the required startup inputs, then restart EnzyWizard-Agent "
            "with the original request."
        )
    return AnalysisErrorGuidance(
        error_type="input_error",
        stage="workflow_input_validation",
        reason=reason,
        can_continue_current_session=False,
        restart_required=True,
        user_actions=[
            "Exit the current session; later chat messages cannot replace startup files or structured inputs.",
            correction,
        ],
    )


def _continuable_error_guidance(
    *,
    error_type: AnalysisErrorType,
    stage: str,
    reason: str,
    user_actions: list[str],
    evidence_paths: list[str] | None = None,
) -> AnalysisErrorGuidance:
    return AnalysisErrorGuidance(
        error_type=error_type,
        stage=stage,
        reason=reason,
        can_continue_current_session=True,
        restart_required=False,
        user_actions=user_actions,
        evidence_paths=evidence_paths or [],
    )


def _load_local_tool_help() -> str:
    """从现有工具 YAML 生成简短能力清单，不引入联网依赖。"""

    spec_path = (
        Path(__file__).resolve().parents[1]
        / "knowledge_base"
        / "enzywizard_tool_specs"
        / "enzywizard_tools.yaml"
    )
    try:
        payload = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
        tools = payload.get("tools", {}) if isinstance(payload, dict) else {}
        lines = []
        for name, spec in tools.items():
            description = str(spec.get("description", "")).strip().replace("\n", " ")
            lines.append(f"{name}: {description[:240]}")
        return "\n".join(lines)
    except (OSError, UnicodeError, yaml.YAMLError, AttributeError):
        return "EnzyWizard-Agent supports its locally registered protein-analysis tools."


def _unwrap_general_answer_text(text: str) -> str:
    """Recover user-facing text from common response wrappers without changing meaning."""

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(payload, dict):
        for key in (
            "final_answer",
            "answer",
            "response",
            "response_text",
            "content",
            "text",
        ):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    if isinstance(payload, list):
        chunks: list[str] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            for key in ("text", "content", "answer", "response"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    chunks.append(value.strip())
                    break
        if chunks:
            return "\n".join(chunks)
    return text


def _load_report_cautions(report_types: set[str]) -> list[str]:
    spec_path = (
        Path(__file__).resolve().parents[1]
        / "knowledge_base"
        / "analysis_report_specs.yaml"
    )
    try:
        payload = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
        reports = payload.get("reports", {}) if isinstance(payload, dict) else {}
        return [
            str(reports[report_type]["caution"])
            for report_type in report_types
            if report_type in reports and reports[report_type].get("caution")
        ]
    except (OSError, UnicodeError, yaml.YAMLError, AttributeError, TypeError):
        return []


def _analysis_retrieval_context(summaries: list[ReportSummary]) -> str:
    """Describe actual report semantics for bounded FAIR-term retrieval."""

    spec_path = (
        Path(__file__).resolve().parents[1]
        / "knowledge_base"
        / "analysis_report_specs.yaml"
    )
    terms: list[str] = []
    try:
        payload = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
        reports = payload.get("reports", {}) if isinstance(payload, dict) else {}
        for summary in summaries:
            spec = reports.get(summary.report_type, {})
            configured = spec.get("fair_retrieval_terms", [])
            if isinstance(configured, list):
                terms.extend(
                    str(item).strip()
                    for item in configured
                    if str(item).strip()
                )
    except (OSError, UnicodeError, yaml.YAMLError, AttributeError, TypeError):
        pass

    sides = {summary.side for summary in summaries}
    if {"wt", "mut"}.issubset(sides):
        terms.extend(["wild-type", "amino acid substitution", "WT/MUT comparison"])
    return "; ".join(dict.fromkeys(terms))


def _match_failure_hints(error_text: str) -> list[dict[str, str]]:
    """用本地规则匹配明确错误片段，不进行联网搜索或宽泛原因猜测。"""

    spec_path = (
        Path(__file__).resolve().parents[1]
        / "knowledge_base"
        / "analysis_failure_hints.yaml"
    )
    try:
        payload = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
        rules = payload.get("rules", []) if isinstance(payload, dict) else []
        valid_rules = [rule for rule in rules if isinstance(rule, dict)]
        return retrieve_failure_hints(error_text, valid_rules)
    except (OSError, UnicodeError, yaml.YAMLError, AttributeError, TypeError):
        return []


def _scientific_terminology(
    state: AgentState, answer: str
) -> tuple[str, list[AnalysisEvidence], str]:
    """Render definitions for terms actually used, not the prompt's candidate terms."""
    config = state.get("config")
    if config is None or config.ontology_mode != "retrieve":
        return answer, [], ""
    marked_answer, entries = annotate_terminology(answer, load_fair_terms())
    if not entries:
        return answer, [], ""

    evidence: list[AnalysisEvidence] = []
    lines: list[str] = []
    for number, entry in enumerate(entries, 1):
        term, uri, description = entry["term"], entry["uri"], entry["description"]
        evidence.append(
            AnalysisEvidence(
                evidence_id=f"fair-term-{number}",
                source_kind="knowledge_base",
                inference_level="direct",
                turn_id=state.get("current_turn_id"),
                plan_id=state.get("current_plan_id"),
                artifact_type=None,
                source_tool=None,
                side=None,
                path=None,
                field_path="description",
                value_summary=description,
                source_uri=uri,
                source_label=term,
                source_version=fair_terms_sha256(),
                metadata={
                    "role": "terminology_definition",
                    "stage": "analysis.scientific_narrative",
                    "source_documents": entry.get("source_documents", []),
                    "related_tools": entry.get("related_tools", []),
                },
            )
        )
        lines.append(f"- {terminology_number(number)} {term}: {description} ({uri})")
    return marked_answer, evidence, (
        TERMINOLOGY_APPENDIX_MARKER
        + "(definitions only, not computational-result evidence):\n"
        + "\n".join(lines)
    )


def _analyze_planning_failure(
    state: AgentState,
    override_error: str | None = None,
) -> AnalysisResult:
    planning = state.get("current_planning_result")
    errors = (
        [override_error]
        if override_error is not None
        else list(planning.error_messages)
        if planning is not None
        else ["Planning failed without a recorded result."]
    )
    if not errors:
        errors = ["Planning failed without a recorded error message."]
    reason = "; ".join(errors)
    guidance = _continuable_error_guidance(
        error_type="planning_error",
        stage="workflow_planning",
        reason=reason,
        user_actions=[
            "Revise or restate the requested workflow in the next message.",
            "No tools were executed, so no computed result needs to be discarded.",
        ],
    )
    answer = _format_error_answer(guidance)
    evidence = [
        AnalysisEvidence(
            evidence_id="planning-failure",
            source_kind="planning",
            inference_level="direct",
            turn_id=state.get("current_turn_id", 1),
            plan_id=getattr(planning, "plan_id", None),
            artifact_type=None,
            source_tool=None,
            side=None,
            path=None,
            field_path="error_messages",
            value_summary=reason,
        )
    ]
    findings = [
        AnalysisFinding(
            title="Planning failure",
            statement=reason,
            finding_kind="failure",
            inference_level="direct",
            evidence_ids=["planning-failure"],
        )
    ]
    return _empty_result(
        state,
        response_mode="planning_failure",
        analysis_status="failed",
        summary="Planning failed before tool execution.",
        final_answer=answer,
        findings=findings,
        evidence=evidence,
        error_messages=errors,
        error_guidance=guidance,
    )


def _analyze_execution_failure(state: AgentState) -> AnalysisResult:
    execution = state.get("current_tool_execution")
    if execution is None:
        guidance = AnalysisErrorGuidance(
            error_type="analysis_error",
            stage="tool_execution_analysis",
            reason="The execution failure record is unavailable.",
            can_continue_current_session=False,
            restart_required=True,
            user_actions=[
                "Exit the current session.",
                "Inspect the session manifest and restart the original request.",
            ],
        )
        return _empty_result(
            state,
            response_mode="tool_execution_failure",
            analysis_status="failed",
            analysis_route="restart_required",
            termination_reason="restart_required",
            summary="Tool execution failed without an execution record.",
            final_answer=_format_error_answer(guidance),
            error_messages=["Missing current_tool_execution."],
            error_guidance=guidance,
        )

    failed_call = next(
        (item for item in execution.call_executions if item.execution_status == "failed"),
        None,
    )
    errors = list(execution.error_messages)
    detail = "; ".join(errors)
    if failed_call is not None and failed_call.log_excerpt:
        # 执行层已从日志尾部截取有限文本；Analysis 再限制展示长度。
        detail += f"\n\nLog tail:\n{failed_call.log_excerpt[-1500:]}"

    failure_hints = _match_failure_hints(detail)
    tool_name = failed_call.tool_name if failed_call is not None else "unknown"
    diagnosis = failed_call.failure_diagnosis if failed_call is not None else None
    ai_assessment = next(
        (
            item
            for item in reversed(execution.ai_failure_assessments)
            if failed_call is not None
            and item.step_index == failed_call.step_index
        ),
        None,
    )
    ai_diagnosis = (
        ai_assessment.diagnosis
        if ai_assessment is not None
        and ai_assessment.assessment_status == "succeeded"
        else None
    )
    evidence_paths = []
    if failed_call is not None:
        evidence_paths = list(
            dict.fromkeys(
                path
                for path in [
                    failed_call.stderr_file,
                    failed_call.stdout_file,
                    *failed_call.log_files,
                ]
                if path
            )
        )
    agent_inputs = state.get("agent_inputs")
    startup_file_paths = {
        path
        for path in (
            getattr(agent_inputs, "structure_file", None),
            getattr(agent_inputs, "msa_file", None),
            getattr(agent_inputs, "mut_structure_file", None),
            getattr(agent_inputs, "mut_msa_file", None),
        )
        if path
    }
    failed_call_text = ""
    if failed_call is not None:
        failed_call_text = json.dumps(
            {
                "command": failed_call.command,
                "resolved_arguments": failed_call.resolved_arguments,
            },
            default=str,
        )
    immutable_startup_file_failed = bool(
        diagnosis is not None
        and diagnosis.category in {"path_or_file", "invalid_input"}
        and any(path in failed_call_text for path in startup_file_paths)
    )
    restart_required = bool(
        diagnosis is None
        or diagnosis.category == "missing_dependency"
        or immutable_startup_file_failed
        or (
            ai_assessment is not None
            and ai_assessment.approved_action == "request_restart"
        )
    )
    if restart_required:
        actions = [
            "Exit the current session; the stopped plan cannot be resumed in place.",
            "Correct the reported startup file or software dependency problem and restart the original request.",
        ]
    elif (
        ai_assessment is not None
        and ai_assessment.approved_action == "request_parameter_confirmation"
    ):
        actions = [
            "Review the AI interpretation, deterministic diagnosis, and logs below.",
            "In the next message, explicitly approve a scientifically meaningful parameter change or request a different workflow; no parameter was changed automatically.",
        ]
    elif (
        ai_assessment is not None
        and ai_assessment.approved_action == "request_user_input"
    ):
        actions = [
            "Review the AI interpretation and its supporting evidence below.",
            "Provide the requested clarification in the next message, or restart if a startup file must be replaced.",
        ]
    elif ai_assessment is not None and ai_assessment.approved_action == "stop":
        actions = [
            "Review the diagnosis and logs below; no safe automatic recovery was authorized.",
            "Request a different workflow in the next message or exit the session.",
        ]
    elif diagnosis is not None and diagnosis.recoverability == "confirmation_required":
        actions = [
            "Review the deterministic diagnosis and logs below.",
            "In the next message, explicitly approve a scientifically meaningful parameter change or request a different workflow.",
        ]
    else:
        actions = [
            "Review the diagnosis and logs below.",
            "Retry the workflow in the next message if the transient condition has cleared.",
        ]
    guidance = AnalysisErrorGuidance(
        error_type="execution_error",
        stage=(
            f"tool_execution.{diagnosis.failure_stage}"
            if diagnosis is not None
            else "tool_execution"
        ),
        reason=(
            ai_diagnosis.user_explanation
            if ai_diagnosis is not None
            else diagnosis.reason
            if diagnosis is not None
            else (detail or "Tool execution failed without a deterministic diagnosis.")
        ),
        can_continue_current_session=not restart_required,
        restart_required=restart_required,
        user_actions=actions,
        evidence_paths=evidence_paths,
    )
    diagnostic_sections = [detail] if detail else []
    if diagnosis is not None:
        diagnostic_sections.append(
            "Deterministic diagnosis:\n"
            f"Category: `{diagnosis.category}`; "
            f"stage=`{diagnosis.failure_stage}`, "
            f"recoverability=`{diagnosis.recoverability}`. "
            f"{diagnosis.reason}\n"
            "Allowed actions: " + ", ".join(diagnosis.allowed_actions) + "."
        )
    if ai_diagnosis is not None and ai_assessment is not None:
        diagnostic_sections.append(
            "AI-assisted interpretation (advisory):\n"
            f"{ai_diagnosis.user_explanation}\n"
            "Likely causes: " + "; ".join(ai_diagnosis.likely_causes) + "\n"
            f"Recommended action: {ai_diagnosis.recommended_action}; "
            f"confidence={ai_diagnosis.confidence:.2f}."
        )
        diagnostic_sections.append(
            "Recovery policy decision:\n"
            f"Decision: {ai_assessment.policy_decision}; "
            f"approved action: {ai_assessment.approved_action}.\n"
            f"Reason: {ai_assessment.policy_reason}"
        )
    elif ai_assessment is not None and ai_assessment.assessment_status == "failed":
        diagnostic_sections.append(
            "AI-assisted interpretation:\n"
            "Unavailable; the deterministic diagnosis and policy were retained. "
            f"Reason: {ai_assessment.error_message}"
        )
    if execution.recovery_attempts:
        succeeded = sum(
            item.execution_status == "succeeded"
            for item in execution.recovery_attempts
        )
        diagnostic_sections.append(
            f"Bounded recovery attempts: {len(execution.recovery_attempts)}; "
            f"successful: {succeeded}."
        )
    if failure_hints:
        hint_lines = ["Local failure guidance:"]
        for hint in failure_hints:
            hint_lines.append(
                f"- Possible cause: {hint['cause']} "
                f"Action: {hint['action']}"
            )
        diagnostic_sections.append("\n".join(hint_lines))
    answer = _format_error_answer(
        guidance,
        detail="\n\n".join(diagnostic_sections),
    )

    evidence = [
        AnalysisEvidence(
            evidence_id="execution-failure",
            source_kind="tool_execution",
            inference_level="direct",
            turn_id=execution.turn_id,
            plan_id=execution.plan_id,
            artifact_type=None,
            source_tool=tool_name,
            side=failed_call.side if failed_call is not None else None,
            path=failed_call.stderr_file if failed_call is not None else None,
            field_path="error_messages",
            value_summary="; ".join(errors),
            metadata=(
                {
                    "failure_category": diagnosis.category,
                    "failure_stage": diagnosis.failure_stage,
                    "recoverability": diagnosis.recoverability,
                    "allowed_actions": diagnosis.allowed_actions,
                }
                if diagnosis is not None
                else {}
            ),
        )
    ]
    findings = [
        AnalysisFinding(
            title=f"Tool failure: {tool_name}",
            statement="; ".join(errors),
            finding_kind="failure",
            inference_level="direct",
            evidence_ids=["execution-failure"],
        )
    ]
    if ai_diagnosis is not None and ai_assessment is not None:
        evidence.append(
            AnalysisEvidence(
                evidence_id="ai-failure-assessment",
                source_kind="tool_execution",
                inference_level="inferred",
                turn_id=execution.turn_id,
                plan_id=execution.plan_id,
                artifact_type=None,
                source_tool=tool_name,
                side=failed_call.side if failed_call is not None else None,
                path=None,
                field_path="ai_failure_assessments",
                value_summary=ai_diagnosis.user_explanation,
                source_evidence_ids=["execution-failure"],
                metadata={
                    "evidence_ids": ai_diagnosis.evidence_ids,
                    "recommended_action": ai_diagnosis.recommended_action,
                    "confidence": ai_diagnosis.confidence,
                    "policy_decision": ai_assessment.policy_decision,
                    "approved_action": ai_assessment.approved_action,
                    "policy_reason": ai_assessment.policy_reason,
                },
            )
        )
        findings.append(
            AnalysisFinding(
                title="AI-assisted failure interpretation",
                statement=ai_diagnosis.user_explanation,
                finding_kind="warning",
                inference_level="inferred",
                evidence_ids=["execution-failure", "ai-failure-assessment"],
            )
        )
    for index, hint in enumerate(failure_hints, start=1):
        evidence_id = f"failure-hint-{index}"
        evidence.append(
            AnalysisEvidence(
                evidence_id=evidence_id,
                source_kind="knowledge_base",
                inference_level="inferred",
                turn_id=execution.turn_id,
                plan_id=execution.plan_id,
                artifact_type=None,
                source_tool=tool_name,
                side=failed_call.side if failed_call is not None else None,
                path=None,
                field_path=None,
                value_summary=f"{hint['cause']} {hint['action']}",
            )
        )
        findings.append(
            AnalysisFinding(
                title="Local failure guidance",
                statement=f"{hint['cause']} {hint['action']}",
                finding_kind="warning",
                inference_level="inferred",
                evidence_ids=[evidence_id],
            )
        )
    references = [
        artifact_to_reference(artifact)
        for artifact in state.get("current_plan_artifacts", [])
    ]
    return _empty_result(
        state,
        response_mode="tool_execution_failure",
        analysis_status="failed",
        analysis_route=("restart_required" if restart_required else "answer_generated"),
        termination_reason=("restart_required" if restart_required else None),
        summary=execution.summary,
        final_answer=answer,
        findings=findings,
        evidence=evidence,
        referenced_artifacts=references,
        error_messages=errors,
        error_guidance=guidance,
    )


def _analyze_rejected_plan(state: AgentState) -> AnalysisResult:
    confirmation = state.get("current_user_confirmation")
    answer = "You rejected the current plan. No tools were executed in this turn."
    plan_id = confirmation.plan_id if confirmation is not None else None
    evidence = [
        AnalysisEvidence(
            evidence_id="plan-rejected",
            source_kind="planning",
            inference_level="direct",
            turn_id=state.get("current_turn_id", 1),
            plan_id=plan_id,
            artifact_type=None,
            source_tool=None,
            side=None,
            path=None,
            field_path="decision",
            value_summary="rejected",
        )
    ]
    findings = [
        AnalysisFinding(
            title="Plan rejected",
            statement="No tool call was executed.",
            finding_kind="information",
            inference_level="direct",
            evidence_ids=["plan-rejected"],
        )
    ]
    return _empty_result(
        state,
        response_mode="plan_rejected",
        summary="The user rejected the current tool plan.",
        final_answer=answer,
        findings=findings,
        evidence=evidence,
    )


def _deterministic_scientific_answer(
    state: AgentState,
    findings: list[AnalysisFinding],
    evidence: list[AnalysisEvidence],
    historical: bool,
    *,
    inline_evidence: bool = True,
) -> str:
    selected = findings[:12]
    heading = (
        "Historical report analysis:"
        if historical
        else "Analysis of the available reports:"
    )
    lines = [heading]
    by_id = {item.evidence_id: item for item in evidence}
    for finding in selected:
        labels = [
            _evidence_display_label(by_id[evidence_id])
            for evidence_id in finding.evidence_ids
            if evidence_id in by_id
        ]
        citation = f" [{'; '.join(labels)}]" if labels and inline_evidence else ""
        lines.append(f"- {finding.title}: {finding.statement}{citation}")
    if len(findings) > len(selected):
        lines.append(
            f"- {len(findings) - len(selected)} additional findings are retained "
            "in the structured analysis record."
        )
    return "\n".join(lines)


_INTERNAL_EVIDENCE_ID_SEARCH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_-])"
    r"(?:prior-turn-\d+-[A-Za-z0-9_-]+|"
    r"report-\d+(?:-metric-\d+)?|"
    r"(?:comparison|consistency|recovery|fair-term|artifact)-\d+|"
    r"planning-failure|execution-failure|plan-rejected)"
    r"(?![A-Za-z0-9_-])"
)
_BRACKETED_REFERENCE_PATTERN = re.compile(r"\[([^\[\]]+)\]")


def _evidence_display_label(item: AnalysisEvidence) -> str:
    """Derive a stable user-facing source description from structured fields."""

    turn = f" from turn {item.turn_id}" if item.turn_id is not None else ""
    field = f", field {item.field_path}" if item.field_path else ""
    tool = item.source_tool or "recorded source"
    side = {
        "single": "single-protein",
        "wt": "WT",
        "mut": "mutant",
        "shared": "WT/MUT",
    }.get(item.side, "")

    if item.source_kind == "report":
        if item.metadata.get("role") == "report_metric":
            metric_kind = (
                "calculated metric"
                if item.inference_level == "calculated"
                else "metric"
            )
            return f"Evidence: {metric_kind} from {tool} report{turn}{field}"
        if item.inference_level == "calculated":
            return f"Evidence: {side or 'report'} comparison calculated from {tool} reports{turn}{field}"
        prefix = f"{side} " if side else ""
        return f"Evidence: {prefix}{tool} report{turn}{field}"
    if item.source_kind == "artifact":
        artifact = item.artifact_type or "artifact"
        return f"Evidence: {artifact} produced by {tool}{turn}{field}"
    if item.source_kind == "tool_execution":
        return f"Execution evidence: {tool} execution record{turn}{field}"
    if item.source_kind == "planning":
        return f"Planning evidence: recorded workflow decision{turn}{field}"
    if item.source_kind == "agent_input":
        return f"Input evidence: recorded Agent input{turn}{field}"
    if item.metadata.get("role") == "terminology_definition":
        label = item.source_label or "recorded term"
        return f"Terminology source: {label} definition"
    return f"Guidance source: local knowledge base{turn}{field}"


def _replace_internal_evidence_references(
    answer: str,
    evidence: list[AnalysisEvidence],
) -> str:
    """Replace valid internal IDs and suppress unknown internal-looking IDs."""

    labels = {
        item.evidence_id: _evidence_display_label(item)
        for item in evidence
    }

    def replace(match: re.Match[str]) -> str:
        content = match.group(1)
        if _INTERNAL_EVIDENCE_ID_SEARCH_PATTERN.search(content) is None:
            return match.group(0)

        def replace_id(id_match: re.Match[str]) -> str:
            return labels.get(id_match.group(0), "")

        rendered_content = _INTERNAL_EVIDENCE_ID_SEARCH_PATTERN.sub(
            replace_id,
            content,
        )
        rendered_content = re.sub(r"\s*[,;]\s*", "; ", rendered_content)
        rendered_content = re.sub(r"\s{2,}", " ", rendered_content)
        rendered_content = rendered_content.strip(" ;,")
        return f"[{rendered_content}]" if rendered_content else ""

    rendered = _BRACKETED_REFERENCE_PATTERN.sub(replace, answer)
    for evidence_id, label in sorted(
        labels.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        rendered = re.sub(
            rf"(?<![A-Za-z0-9_-]){re.escape(evidence_id)}(?![A-Za-z0-9_-])",
            f"[{label}]",
            rendered,
        )
    return _INTERNAL_EVIDENCE_ID_SEARCH_PATTERN.sub("", rendered)


_CLAIM_NUMBER = re.compile(
    r"(?<![\w.])[-+\u2212]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+\u2212]?\d+)?(?![\w]|\.\d)"
)
_WT = re.compile(r"\b(?:WT|wild[- ]type)\b", re.IGNORECASE)
_MUT = re.compile(r"\b(?:MUT|mutant)\b", re.IGNORECASE)


def _claim_conflicts_with_source(text: str, source: AnalysisEvidence) -> bool:
    """Reject only obvious conflicts; an existing ID does not imply entailment."""
    wt, mut = bool(_WT.search(text)), bool(_MUT.search(text))
    if (wt and not mut and source.side == "mut") or (mut and not wt and source.side == "wt"):
        return True
    if source.metadata.get("role") != "report_metric" or "%" in text:
        return False
    values = list(_CLAIM_NUMBER.finditer(text))
    raw = (source.value_summary or "").partition("=")[2].strip()
    if len(values) != 1 or not _CLAIM_NUMBER.fullmatch(raw):
        return False
    if text[values[0].end():].strip(" .!,;?") and not str(
        source.metadata.get("metric_name", "")
    ).endswith("_count"):
        return False
    stated = values[0].group().replace("\u2212", "-")
    if "e" in stated.lower():
        return False
    actual, value = float(raw.replace("\u2212", "-")), float(stated)
    if not (math.isfinite(actual) and math.isfinite(value)):
        return False
    precision = len(stated.partition(".")[2])
    return not math.isclose(round(actual, precision), value, rel_tol=0, abs_tol=1e-12)


def _readable_evidence_source(item: AnalysisEvidence, *, include_plan: bool = False) -> str:
    sides = {"wt": "wild type", "mut": "mutant", "shared": "WT and mutant"}
    report = (
        Path(item.path).name
        if item.path else (item.source_tool or item.source_kind.replace("_", " "))
    )
    details = ", ".join(filter(None, (
        f"turn {item.turn_id}" if item.turn_id is not None else "",
        f"plan {item.plan_id}" if include_plan and item.plan_id is not None else "",
        sides.get(item.side, ""),
    )))
    return f"{report} ({details})" if details else report


_EVIDENCE_FIELD_SHORT_NAMES = {"residue_root_mean_square_fluctuation": "RMSF"}
_EVIDENCE_ARROW = " \u2192 "


def _readable_evidence_field(path: str | None) -> tuple[str, str]:
    """Return a field description and a short result label."""
    if not path or path == "$":
        return "", ""
    array_path, selection, leaf_path = path.partition("[*].")
    field_key = (leaf_path if selection else path).rsplit(".", 1)[-1]
    subject = _EVIDENCE_FIELD_SHORT_NAMES.get(field_key) or _humanize_field_path(field_key)
    if selection and "[*]" not in leaf_path:
        collection = _humanize_field_path(array_path).replace(" > ", " / ")
        leaf = _EVIDENCE_FIELD_SHORT_NAMES.get(leaf_path) or _humanize_field_path(leaf_path)
        if " / " not in collection and " > " not in leaf:
            return f"{leaf} values from all entries in {collection}", subject
    return _humanize_field_path(path).replace(" > ", " / "), subject


def _readable_evidence_chain(item: AnalysisEvidence, by_id: dict[str, AnalysisEvidence]) -> str:
    # Preserve source ordering; comparison input fields and values use the same order.
    parents = [by_id.get(eid) for eid in item.source_evidence_ids]
    trace = item.derivation
    fields = item.metadata.get("source_field_paths", [])
    field, subject = _readable_evidence_field(item.field_path)
    parts: list[str] = []
    if fields and len(fields) == len(parents):
        inputs = []
        for index, (parent, path) in enumerate(zip(parents, fields, strict=True)):
            label, source_subject = _readable_evidence_field(path)
            include_plan = parent is not None and any(
                other is not None and other.turn_id == parent.turn_id
                and other.plan_id != parent.plan_id for other in parents
            )
            location = (
                _readable_evidence_source(parent, include_plan=include_plan)
                if parent is not None else "Unresolved report source"
            )
            if trace and index < len(trace.source_values):
                value = trace.source_values[index]
                steps = trace.source_steps[index] if index < len(trace.source_steps) else []
                if steps:
                    result_label = f"{steps[-1]}: {source_subject}" if source_subject else steps[-1]
                    segments = [location, label, *steps[:-1], f"{result_label} = {value}"]
                    chain = _EVIDENCE_ARROW.join(part for part in segments if part)
                    inputs.append(f"({chain})")
                    continue
                label = (
                    f"Value used in comparison = {value} (from {label})"
                    if "[*]" in path
                    else f"{label or 'Source value'} = {value}"
                )
            inputs.append(f"{location}: {label}" if label else location)
        parts.append(" + ".join(inputs))
    else:
        parts.append(_readable_evidence_source(item))
        if field:
            parts.append(field)

    if item.inference_level == "inferred":
        sources = " + ".join(_readable_evidence_source(parent) for parent in parents if parent)
        return _EVIDENCE_ARROW.join([
            sources or parts[0], f"Interpretation: {item.value_summary}",
        ])

    if trace is not None:
        result = " ".join(trace.result_value.split())
        if trace.steps:
            parts.extend(trace.steps[:-1])
            label = trace.result_label or (
                f"{trace.steps[-1]}: {subject}" if subject else trace.steps[-1]
            )
            parts.append(f"{label} = {result}")
        elif item.inference_level == "direct" and field:
            parts[-1] += f" = {result}"
        else:
            label = "Calculated result" if item.inference_level == "calculated" else "Result"
            parts.append(f"{label} = {result}")
    elif item.metadata.get("role") == "report_metric" and "=" in item.value_summary:
        _, _, result = item.value_summary.partition("=")
        if item.inference_level == "direct" and field:
            parts[-1] += f" = {result}"
        else:
            parts.append(f"Calculated result = {result}")
    else:
        label = "Calculated result" if item.inference_level == "calculated" else "Recorded result"
        result = " ".join(item.value_summary.replace("_", " ").split())
        parts.append(f"{label}: {result}")
    return _EVIDENCE_ARROW.join(parts)


def _find_claim_span(answer: str, claim: str, cursor: int) -> int:
    if not claim:
        return -1
    left = r"(?<!\w)" if claim[0].isalnum() else ""
    right = r"(?!\w)" if claim[-1].isalnum() else ""
    if claim[-1].isdigit():
        right = r"(?!\w|\.\d)"
    if len(claim) > 1 and claim.endswith(".") and claim[-2].isdigit():
        right = r"(?!\d)"
    match = re.compile(left + re.escape(claim) + right).search(answer, cursor)
    return match.start() if match else -1


def _render_claim_citations(
    answer: str,
    drafts: list[NarrativeClaimCitation],
    evidence: list[AnalysisEvidence],
    *,
    supplied_evidence_ids: set[str] | None = None,
) -> tuple[str, str, list[AnalysisClaimCitation], list[str]]:
    by_id = {item.evidence_id: item for item in evidence if item.source_kind != "knowledge_base"}
    chunks: list[str] = []
    lines: list[str] = []
    citations: list[AnalysisClaimCitation] = []
    warnings: list[str] = []
    cursor = 0
    for draft in drafts:
        start = _find_claim_span(answer, draft.text, cursor)
        if start < 0:
            warnings.append("A proposed citation could not be located in the answer.")
            continue
        end = start + len(draft.text)
        ids = list(dict.fromkeys(
            eid for eid in draft.evidence_ids
            if eid in by_id and (supplied_evidence_ids is None or eid in supplied_evidence_ids)
        ))
        if len(ids) == 1 and _claim_conflicts_with_source(draft.text, by_id[ids[0]]):
            ids = []
        if len(set(draft.evidence_ids)) != len(ids):
            warnings.append("One or more proposed citation sources were invalid or inconsistent.")
        number = len(citations) + 1
        citations.append(AnalysisClaimCitation(number=number, text=draft.text, evidence_ids=ids))
        marker_at = end - 1 if answer[end - 1] in ".!?;," else end
        chunks.extend((answer[cursor:marker_at], claim_evidence_number(number), answer[marker_at:end]))
        cursor = end
        sources = "; ".join(_readable_evidence_chain(by_id[eid], by_id) for eid in ids)
        if not ids:
            sources = "No validated computational source was linked"
            warnings.append("One or more conclusions have no validated computational source.")
        lines.append(f"[{number}] {sources}")
    chunks.append(answer[cursor:])
    appendix = CLAIM_EVIDENCE_APPENDIX_MARKER + "\n".join(lines) if lines else ""
    return "".join(chunks), appendix, citations, list(dict.fromkeys(warnings))


def scientific_review_answer(result: AnalysisResult) -> str:
    """Expose the real citation IDs and sources to reviewers, not display numbers."""
    answer = scientific_answer_body(result.final_answer, keep_claim_markers=True) or ""
    by_id = {item.evidence_id: item for item in result.evidence}
    for citation in result.claim_citations or []:
        labels = [
            f"{_evidence_display_label(by_id[eid])} (ID: {eid})"
            for eid in citation.evidence_ids if eid in by_id
        ]
        answer = answer.replace(
            claim_evidence_number(citation.number),
            f" [{'; '.join(labels)}]" if labels else "",
            1,
        )
    return answer


def _successful_recovery_evidence(
    state: AgentState,
) -> tuple[list[AnalysisEvidence], list[AnalysisFinding]]:
    execution = state.get("current_tool_execution")
    if execution is None or execution.execution_status != "succeeded":
        return [], []
    evidence: list[AnalysisEvidence] = []
    findings: list[AnalysisFinding] = []
    recovery_evidence_by_step: dict[int, str] = {}
    for index, attempt in enumerate(execution.recovery_attempts, start=1):
        evidence_id = f"recovery-{index}"
        recovery_evidence_by_step[attempt.step_index] = evidence_id
        succeeded = attempt.execution_status == "succeeded"
        statement = (
            f"Step {attempt.step_index} recovery attempt {attempt.attempt_index} "
            f"{'succeeded' if succeeded else 'failed'} using {attempt.action}."
        )
        evidence.append(
            AnalysisEvidence(
                evidence_id=evidence_id,
                source_kind="tool_execution",
                inference_level="direct",
                turn_id=execution.turn_id,
                plan_id=execution.plan_id,
                artifact_type=None,
                source_tool=attempt.tool_name,
                side=None,
                path=attempt.stderr_file,
                field_path="recovery_attempts",
                value_summary=statement,
                metadata={
                    "trigger_category": attempt.trigger_category,
                    "action": attempt.action,
                    "execution_status": attempt.execution_status,
                    "managed_input_repairs": [
                        item.model_dump(mode="json")
                        for item in attempt.managed_input_repairs
                    ],
                },
            )
        )
        findings.append(
            AnalysisFinding(
                title=(
                    "Recovered execution failure"
                    if succeeded
                    else "Unsuccessful recovery attempt"
                ),
                statement=statement,
                finding_kind="information",
                inference_level="direct",
                evidence_ids=[evidence_id],
            )
        )
    for index, assessment in enumerate(
        execution.ai_failure_assessments,
        start=1,
    ):
        if (
            assessment.assessment_status != "succeeded"
            or assessment.diagnosis is None
            or assessment.approved_action != "retry_same_call"
        ):
            continue
        recovery_evidence_id = recovery_evidence_by_step.get(assessment.step_index)
        if recovery_evidence_id is None:
            continue
        evidence_id = f"ai-recovery-assessment-{index}"
        statement = (
            f"AI advisory diagnosis for step {assessment.step_index}: "
            f"{assessment.diagnosis.user_explanation}"
        )
        evidence.append(
            AnalysisEvidence(
                evidence_id=evidence_id,
                source_kind="tool_execution",
                inference_level="inferred",
                turn_id=execution.turn_id,
                plan_id=execution.plan_id,
                artifact_type=None,
                source_tool=assessment.tool_name,
                side=None,
                path=None,
                field_path="ai_failure_assessments",
                value_summary=statement,
                source_evidence_ids=[recovery_evidence_id],
                metadata={
                    "evidence_ids": assessment.diagnosis.evidence_ids,
                    "recommended_action": assessment.diagnosis.recommended_action,
                    "confidence": assessment.diagnosis.confidence,
                    "policy_decision": assessment.policy_decision,
                    "approved_action": assessment.approved_action,
                    "policy_reason": assessment.policy_reason,
                },
            )
        )
        findings.append(
            AnalysisFinding(
                title="AI-assisted recovery decision",
                statement=(
                    f"The model proposed an identical retry for step "
                    f"{assessment.step_index}; deterministic policy authorized it "
                    "and the retry result is recorded separately."
                ),
                finding_kind="information",
                inference_level="inferred",
                evidence_ids=[evidence_id, recovery_evidence_id],
            )
        )
    return evidence, findings


def _confirmed_staged_report_pairs(
    state: AgentState,
) -> tuple[list[Artifact], list[tuple[str, str]]]:
    """Use actual paired integration inputs to recognize approved historical reuse."""
    planning = state.get("current_planning_result")
    plan = planning.tool_plan if planning is not None else None
    execution = state.get("current_tool_execution")
    if plan is None or execution is None or execution.execution_status != "succeeded":
        return [], []
    if (plan.turn_id, plan.plan_id) != (execution.turn_id, execution.plan_id):
        return [], []
    succeeded = {
        call.step_index: call for call in execution.call_executions
        if call.execution_status == "succeeded"
    }
    tracked_by_path = {artifact.path: artifact for artifact in state.get("artifacts", [])}
    reused: dict[str, Artifact] = {}
    pairs: list[tuple[str, str]] = []
    for call in plan.calls:
        if call.tool_name != "enzywizard-mut-integrate" or call.step_index not in succeeded:
            continue
        staged: dict[str, Artifact] = {}
        old_paths: set[str] = set()
        for rule in call.staging_rules:
            if rule.side not in {"wt", "mut"}:
                continue
            if rule.source_step_index is None:
                artifact = tracked_by_path.get(rule.source_path_pattern)
                if artifact is not None and artifact.side == rule.side:
                    staged[artifact.path] = artifact
                    old_paths.add(artifact.path)
            elif (source := succeeded.get(rule.source_step_index)) is not None:
                for artifact in source.artifacts:
                    if artifact.side == rule.side and Path(artifact.path).match(rule.source_path_pattern):
                        staged[artifact.path] = artifact
        for pair in unique_wt_mut_report_pairs(staged.values()):
            if old_paths.intersection(pair):
                pairs.append(pair)
                reused.update((path, staged[path]) for path in pair if path in old_paths)
    return list(reused.values()), pairs


def _analyze_reports(
    state: AgentState,
    artifacts: list[Any],
    *,
    historical: bool,
) -> AnalysisResult:
    history = state.get("current_history_query")
    comparison_report_pairs = list(history.comparison_report_pairs) if (
        history is not None and history.query_status == "succeeded"
        and history.turn_id == state.get("current_turn_id")
    ) else []
    if not historical:
        reused, staged_pairs = _confirmed_staged_report_pairs(state)
        artifacts = _deduplicate_artifacts([*artifacts, *reused])
        comparison_report_pairs.extend(staged_pairs)
    report_artifacts = select_report_artifacts(artifacts)
    summaries = [summarize_report_artifact(artifact) for artifact in report_artifacts]
    readable = [summary for summary in summaries if summary.read_status == "parsed"]
    read_labels = {
        "missing": "file missing",
        "unreadable": "could not be read or parsed",
        "too_large": "exceeds the report size limit",
    }
    unavailable = [
        f"{Path(summary.path).name} (turn {summary.turn_id}, plan {summary.plan_id}, "
        f"{summary.side}): {read_labels[summary.read_status]}."
        for summary in summaries if summary.read_status != "parsed"
    ]
    comparison_requested = bool(
        history is not None
        and history.query is not None
        and "compare" in history.query.operations
    )
    evidence, findings = build_report_evidence(
        readable,
        enable_general_comparison=comparison_requested,
        comparison_report_pairs=comparison_report_pairs,
    )
    recovery_evidence, recovery_findings = _successful_recovery_evidence(state)
    evidence.extend(recovery_evidence)
    findings.extend(recovery_findings)
    references = [artifact_to_reference(artifact) for artifact in artifacts]
    limitations = [warning for summary in summaries for warning in summary.warnings]
    limitations.extend(unavailable)
    limitations.extend(_load_report_cautions({item.report_type for item in readable}))
    limitations.extend(
        finding.statement
        for finding in findings
        if finding.title.startswith("Cross-report inconsistency:")
    )
    fallback_answer = _deterministic_scientific_answer(
        state,
        findings,
        evidence,
        historical,
        inline_evidence=False,
    )
    fallback_summary = (
        f"Analyzed {len(readable)} historical reports."
        if historical
        else f"Executed the approved plan and analyzed {len(readable)} reports."
    )
    report_notice = ""
    if unavailable:
        lead = (
            "The requested comparison is incomplete."
            if comparison_requested else "Some selected reports could not be analyzed."
        )
        report_notice = lead + "\n" + "\n".join(f"- {detail}" for detail in unavailable) + "\n\n"

    if not readable:
        fallback_answer = (
            "Relevant artifacts were found, but no formal JSON report could be "
            "parsed. Inspect the artifact paths and execution logs."
        )
        return _empty_result(
            state,
            response_mode="scientific_result",
            analysis_status="partial",
            summary=fallback_summary,
            final_answer=report_notice + fallback_answer,
            report_summaries=summaries,
            referenced_artifacts=references,
            limitations=list(dict.fromkeys(limitations + ["No parseable report artifact was available."])),
        )

    try:
        context = build_scientific_analysis_context(state, summaries, evidence, findings)
        supplied_evidence_ids = {
            item["evidence_id"] for item in json.loads(context)["deterministic_evidence"]
        }
        output = invoke_structured(
            state,
            stage="analysis.scientific_narrative",
            schema=AnalysisNarrativeOutput,
            llm=_create_llm(state),
            retrieval_context=_analysis_retrieval_context(readable),
            messages=[
                SystemMessage(content=SCIENTIFIC_RESULT_SYSTEM_PROMPT),
                HumanMessage(content=context),
            ],
        )
        answer, evidence_appendix, citations, citation_warnings = _render_claim_citations(
            output.final_answer, output.claim_citations, evidence,
            supplied_evidence_ids=supplied_evidence_ids,
        )
        answer, term_evidence, term_appendix = _scientific_terminology(
            state, _replace_internal_evidence_references(answer, [])
        )
        return _empty_result(
            state,
            response_mode="scientific_result",
            analysis_status="partial" if unavailable else "succeeded",
            summary=output.summary,
            final_answer=(
                report_notice
                + order_answer_markers(answer)
                + evidence_appendix
                + term_appendix
            ),
            report_summaries=summaries,
            findings=findings,
            evidence=[*evidence, *term_evidence],
            referenced_artifacts=references,
            limitations=list(dict.fromkeys(limitations + output.limitations + citation_warnings)),
            used_language_model=True,
            claim_citations=citations,
        )
    except Exception as exc:
        # 报告解析和确定性比较仍有效，因此模型叙述失败只降级为 partial。
        drafts = [
            NarrativeClaimCitation(
                text=f"{finding.title}: {finding.statement}", evidence_ids=finding.evidence_ids
            )
            for finding in findings[:12]
        ]
        answer, evidence_appendix, citations, citation_warnings = _render_claim_citations(
            fallback_answer, drafts, evidence
        )
        answer, term_evidence, term_appendix = _scientific_terminology(state, answer)
        return _empty_result(
            state,
            response_mode="scientific_result",
            analysis_status="partial",
            summary=fallback_summary,
            final_answer=report_notice + order_answer_markers(answer) + evidence_appendix + term_appendix,
            report_summaries=summaries,
            findings=findings,
            evidence=[*evidence, *term_evidence],
            referenced_artifacts=references,
            limitations=list(
                dict.fromkeys(
                    limitations
                    + citation_warnings
                    + [
                        "Language-model synthesis was unavailable; a deterministic "
                        "summary was returned."
                    ]
                )
            ),
            error_messages=[f"Analysis narrative generation failed: {exc}"],
            claim_citations=citations,
        )


def _answer_artifact_query(
    state: AgentState,
    artifacts: list[Artifact],
) -> AnalysisResult:
    """列出 History Query Planner 已选择并由程序验证的真实产物。"""

    references = [artifact_to_reference(artifact) for artifact in artifacts]
    if references:
        lines = ["Found these artifacts:"]
        for reference in references[:60]:
            status = "exists" if reference.exists else "missing"
            lines.append(
                f"- Turn {reference.turn_id}, {reference.source_tool} "
                f"[{reference.side}], {reference.artifact_type} ({status}): "
                f"{reference.path}"
            )
        if len(references) > 60:
            lines.append(
                f"- Additional {len(references) - 60} artifacts are retained in "
                "the structured result."
            )
    else:
        lines = ["No artifact matched the requested turn and type."]

    evidence = [
        AnalysisEvidence(
            evidence_id=f"artifact-{index}",
            source_kind="artifact",
            inference_level="direct",
            turn_id=reference.turn_id,
            plan_id=reference.plan_id,
            artifact_type=reference.artifact_type,
            source_tool=reference.source_tool,
            side=reference.side,
            path=reference.path,
            field_path=None,
            value_summary="exists" if reference.exists else "missing",
        )
        for index, reference in enumerate(references, start=1)
    ]
    return _empty_result(
        state,
        response_mode="artifact_query",
        summary=f"Matched {len(references)} artifacts through the structured history query.",
        final_answer="\n".join(lines),
        evidence=evidence,
        referenced_artifacts=references,
    )


def _answer_general_request(
    state: AgentState,
    *,
    only_non_tool_actions: bool = False,
) -> AnalysisResult:
    routing = state.get("current_request_routing")
    request_type = (
        routing.output.request_type
        if routing is not None and routing.output is not None
        else "conversation"
    )
    response_mode = {
        "general_scientific_question": "general_scientific_question",
        "software_help": "software_help",
        "conversation": "conversation",
        "out_of_scope": "out_of_scope",
    }.get(request_type, "conversation")
    answer_scope = (
        "Answer only the routed software-help, general-question, conversation, or "
        "out-of-scope actions. Do not repeat historical or tool-workflow results."
        if only_non_tool_actions
        else "Answer the complete routed non-tool request."
    )
    context = build_general_answer_context(
        state,
        _load_local_tool_help(),
        answer_scope=answer_scope,
        # Software help and scope responses must describe current capabilities,
        # not replay scientific findings or plans from earlier turns. Mixed
        # requests already retain their scientific context in the primary result.
        include_prior_context=(
            not only_non_tool_actions
            and response_mode not in {"software_help", "out_of_scope"}
        ),
    )
    try:
        final_answer = _unwrap_general_answer_text(
            invoke_text(
                state,
                stage="analysis.general_answer",
                llm=_create_llm(state),
                messages=[
                    SystemMessage(content=GENERAL_ANSWER_SYSTEM_PROMPT),
                    HumanMessage(content=context),
                ],
            )
        )
        summary = {
            "general_scientific_question": "Answered a conceptual scientific question.",
            "software_help": "Answered an EnzyWizard usage question.",
            "conversation": "Continued the conversation.",
            "out_of_scope": "Declined an out-of-scope request.",
        }.get(response_mode, "Answered the non-tool request.")
        return _empty_result(
            state,
            response_mode=response_mode,
            summary=summary,
            final_answer=final_answer,
            used_language_model=True,
        )
    except Exception as exc:
        reason = f"General answer generation failed: {exc}"
        guidance = _continuable_error_guidance(
            error_type="analysis_error",
            stage="general_answer_generation",
            reason=reason,
            user_actions=[
                "Retry the question in the next message.",
                "No scientific tool was executed during this failed answer generation.",
            ],
        )
        return _empty_result(
            state,
            response_mode=response_mode,
            analysis_status="failed",
            summary="General answer generation failed.",
            final_answer=_format_error_answer(guidance),
            error_messages=[reason],
            error_guidance=guidance,
        )


def _answer_routing_failure(state: AgentState) -> AnalysisResult:
    routing = state.get("current_request_routing")
    error = (
        routing.error_message
        if routing is not None
        else "Request routing result is missing."
    )
    guidance = _continuable_error_guidance(
        error_type="routing_error",
        stage="request_routing",
        reason=error,
        user_actions=[
            "Retry or restate the requested action more explicitly in the next message.",
            "No tool was executed for this request.",
        ],
    )
    return _empty_result(
        state,
        response_mode="request_routing_failure",
        analysis_status="failed",
        summary="The request could not be routed safely.",
        final_answer=_format_error_answer(guidance),
        error_messages=[error],
        error_guidance=guidance,
    )


def _answer_clarification(state: AgentState, question: str) -> AnalysisResult:
    return _empty_result(
        state,
        response_mode="clarification",
        summary="Clarification is required before continuing.",
        final_answer=question,
    )


def _answer_missing_input_error(
    state: AgentState,
    missing_inputs: list[str],
) -> AnalysisResult:
    """Terminate safely when immutable CLI inputs required by a workflow are absent."""

    guidance = _missing_input_guidance(missing_inputs)
    return _empty_result(
        state,
        response_mode="input_error",
        analysis_status="failed",
        analysis_route="restart_required",
        termination_reason="restart_required",
        summary="Required workflow input is missing.",
        final_answer=_format_error_answer(guidance),
        error_messages=[guidance.reason],
        error_guidance=guidance,
    )


def _answer_history_issue(state: AgentState) -> AnalysisResult:
    history = state.get("current_history_query")
    if history is None:
        detail = "History query result is missing."
        status = "failed"
    else:
        detail = history.error_message or history.summary
        status = "failed" if history.query_status == "failed" else "partial"
    guidance = _continuable_error_guidance(
        error_type="history_error",
        stage="history_resolution",
        reason=detail,
        user_actions=[
            "Specify the intended earlier turn or artifact more explicitly in the next message.",
            "Alternatively, request a new computation without historical reuse.",
        ],
    )
    return _empty_result(
        state,
        response_mode="history_query_failure",
        analysis_status=status,
        summary="The requested historical artifacts could not be resolved.",
        final_answer=_format_error_answer(guidance),
        error_messages=[detail] if status == "failed" else [],
        error_guidance=guidance,
    )


def _append_answer_section(answer: str, title: str, content: str) -> str:
    """Insert mixed-request prose before both generated appendices."""
    offsets = (
        answer.find(marker)
        for marker in (CLAIM_EVIDENCE_APPENDIX_MARKER, TERMINOLOGY_APPENDIX_MARKER)
    )
    end = min((offset for offset in offsets if offset >= 0), default=len(answer))
    return f"{answer[:end]}\n\n{title}:\n{content}{answer[end:]}"


def _merge_artifact_listing(
    result: AnalysisResult,
    state: AgentState,
    artifacts: list[Artifact],
) -> AnalysisResult:
    """把显式文件查询附加到科学回答，支持 mixed 请求。"""

    listing = _answer_artifact_query(state, artifacts)
    existing_paths = {item.path for item in result.referenced_artifacts}
    references = list(result.referenced_artifacts) + [
        item for item in listing.referenced_artifacts if item.path not in existing_paths
    ]
    existing_ids = {item.evidence_id for item in result.evidence}
    evidence = list(result.evidence)
    for index, item in enumerate(listing.evidence, start=1):
        evidence_id = item.evidence_id
        if evidence_id in existing_ids:
            evidence_id = f"history-{evidence_id}-{index}"
            item = item.model_copy(update={"evidence_id": evidence_id})
        existing_ids.add(evidence_id)
        evidence.append(item)
    return result.model_copy(
        update={
            "summary": f"{result.summary} {listing.summary}",
            "final_answer": _append_answer_section(
                result.final_answer, "Artifacts", listing.final_answer
            ),
            "referenced_artifacts": references,
            "evidence": evidence,
        }
    )


def _merge_general_answer(
    result: AnalysisResult,
    state: AgentState,
) -> AnalysisResult:
    """把 mixed 请求中的独立非工具动作附加到已有结果回答。"""

    general = _answer_general_request(state, only_non_tool_actions=True)
    routing = state.get("current_request_routing")
    routed_actions = (
        routing.output.actions
        if routing is not None and routing.output is not None
        else []
    )
    action_types = {
        item.action_type
        for item in routed_actions
    }
    if "answer_software_help" in action_types:
        section_label = "Software help"
    elif "answer_general_question" in action_types:
        section_label = "Scientific answer"
    elif "decline_out_of_scope" in action_types:
        section_label = "Scope response"
    else:
        section_label = "Additional response"
    return result.model_copy(
        update={
            "summary": f"{result.summary} {general.summary}",
            "final_answer": _append_answer_section(
                result.final_answer, section_label, general.final_answer
            ),
            "limitations": list(
                dict.fromkeys(result.limitations + general.limitations)
            ),
            "error_messages": result.error_messages + general.error_messages,
            "analysis_status": (
                "partial"
                if general.analysis_status == "failed"
                and result.analysis_status == "succeeded"
                else result.analysis_status
            ),
            "used_language_model": (
                result.used_language_model or general.used_language_model
            ),
        }
    )


def _deduplicate_artifacts(artifacts: list[Artifact]) -> list[Artifact]:
    selected: list[Artifact] = []
    seen: set[str] = set()
    for artifact in artifacts:
        if artifact.path in seen:
            continue
        seen.add(artifact.path)
        selected.append(artifact)
    return selected


def analyze_agent_state(state: AgentState) -> AnalysisResult:
    """根据当前轮状态选择分析对象并生成最终回答。"""

    routing = state.get("current_request_routing")
    if routing is None or routing.routing_status == "failed" or routing.output is None:
        return _answer_routing_failure(state)

    routed = routing.output
    action_types = {item.action_type for item in routed.actions}
    general_action_types = {
        "answer_software_help",
        "answer_general_question",
        "continue_conversation",
        "decline_out_of_scope",
    }
    has_general_action = bool(action_types & general_action_types)
    if routed.request_type == "exit":
        return _empty_result(
            state,
            response_mode="exit",
            analysis_route="exit_requested",
            termination_reason="user_exit",
            summary="The user requested to exit.",
            final_answer="The session has ended.",
        )

    if routed.request_type == "clarification":
        if routed.routing_reason == "Required workflow input is unavailable.":
            return _answer_missing_input_error(state, ["a protein structure"])
        return _answer_clarification(
            state,
            routed.clarification_question
            or "Please clarify the requested action.",
        )

    history = state.get("current_history_query")
    if routed.requires_history_artifacts:
        if history is None or history.query_status == "failed":
            return _answer_history_issue(state)
        if history.query_status == "not_found" and not (
            routed.requires_tool_execution
            and routed.reuse_policy == "allowed"
            and not action_types.intersection({"analyze_history", "list_artifacts"})
        ):
            return _answer_history_issue(state)
        if history.query_status == "clarification_required":
            return _answer_clarification(
                state,
                history.query.clarification_question
                if history.query is not None
                else "Please clarify which earlier result should be used.",
            )

    planning = state.get("current_planning_result")
    if planning is not None and planning.turn_id == state.get("current_turn_id"):
        if planning.planning_flag == "clarification_required":
            missing_inputs = (
                list(planning.tool_selection.missing_inputs)
                if planning.tool_selection is not None
                else []
            )
            return _answer_missing_input_error(
                state,
                missing_inputs,
            )
        if planning.planning_flag in {
            "planning_failed",
            "tool_selection_failed",
            "parameter_extraction_failed",
        }:
            return _analyze_planning_failure(state)
        if routed.requires_tool_execution and planning.planning_flag == "tool_unused":
            return _analyze_planning_failure(
                state,
                "Request Router required tool execution, but Planning classified "
                "the request as tool-unused. No tool was executed.",
            )

    confirmation = state.get("current_user_confirmation")
    if (
        confirmation is not None
        and confirmation.turn_id == state.get("current_turn_id")
        and confirmation.decision == "rejected"
    ):
        return _analyze_rejected_plan(state)

    execution = state.get("current_tool_execution")
    if execution is not None and execution.turn_id == state.get("current_turn_id"):
        if execution.execution_status == "failed":
            return _analyze_execution_failure(state)
        report_artifacts = list(state.get("current_plan_artifacts", []))
        if "analyze_history" in action_types and history is not None:
            report_artifacts.extend(history.selected_artifacts)
        result = _analyze_reports(
            state,
            _deduplicate_artifacts(report_artifacts),
            historical=False,
        )
        if history is not None and history.query_status == "not_found":
            result.limitations.append(
                "No compatible historical artifact was found; the new workflow "
                "ran without optional historical reuse."
            )
        if "list_artifacts" in action_types and history is not None:
            result = _merge_artifact_listing(
                result,
                state,
                history.selected_artifacts,
            )
        if has_general_action:
            result = _merge_general_answer(result, state)
        return result

    if history is not None and history.query_status == "succeeded":
        operations = set(history.query.operations) if history.query is not None else set()
        if operations & {"summarize", "explain", "compare"}:
            result = _analyze_reports(
                state,
                history.selected_artifacts,
                historical=True,
            )
            if "list_artifacts" in operations or "list_artifacts" in action_types:
                result = _merge_artifact_listing(
                    result,
                    state,
                    history.selected_artifacts,
                )
            if has_general_action:
                result = _merge_general_answer(result, state)
            return result
        if "list_artifacts" in operations or "list_artifacts" in action_types:
            result = _answer_artifact_query(state, history.selected_artifacts)
            if has_general_action:
                result = _merge_general_answer(result, state)
            return result

    return _answer_general_request(state)
