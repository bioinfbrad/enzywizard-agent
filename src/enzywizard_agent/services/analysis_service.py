from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
import yaml

from ..builders.analysis_context_builder import (
    build_general_answer_context,
    build_scientific_analysis_context,
)
from ..data_models.analysis import (
    AnalysisArtifactReference,
    AnalysisEvidence,
    AnalysisFinding,
    AnalysisNarrativeOutput,
    AnalysisResult,
    ReportSummary,
)
from ..data_models.artifact import Artifact
from ..prompts.analysis_prompt import (
    GENERAL_ANSWER_SYSTEM_PROMPT,
    SCIENTIFIC_RESULT_SYSTEM_PROMPT,
)
from ..state import AgentState
from ..utils.analysis_utils import (
    artifact_to_reference,
    build_report_evidence,
    select_report_artifacts,
    summarize_report_artifact,
)
from .llm_service import create_chat_model_for_state, invoke_structured, invoke_text


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
        normalized = error_text.casefold()
        return [
            {
                "cause": str(rule["cause"]),
                "action": str(rule["action"]),
            }
            for rule in rules
            if isinstance(rule, dict)
            and str(rule.get("contains", "")).casefold() in normalized
            and rule.get("contains")
            and rule.get("cause")
            and rule.get("action")
        ]
    except (OSError, UnicodeError, yaml.YAMLError, AttributeError, TypeError):
        return []


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
    answer = f"Planning failed and no tools were executed. Reason: {reason}"
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
    )


def _analyze_execution_failure(state: AgentState) -> AnalysisResult:
    execution = state.get("current_tool_execution")
    if execution is None:
        return _empty_result(
            state,
            response_mode="tool_execution_failure",
            analysis_status="failed",
            summary="Tool execution failed without an execution record.",
            final_answer="Tool execution failed without an available execution record.",
            error_messages=["Missing current_tool_execution."],
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
    answer = (
        f"Tool execution stopped at `{tool_name}`; later steps were not run."
        f"\n\n{detail}"
    )
    if failure_hints:
        answer += "\n\nLocal failure guidance:"
        for hint in failure_hints:
            answer += (
                f"\n- Possible cause: {hint['cause']} "
                f"Action: {hint['action']}"
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
        summary=execution.summary,
        final_answer=answer,
        findings=findings,
        evidence=evidence,
        referenced_artifacts=references,
        error_messages=errors,
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
    historical: bool,
) -> str:
    selected = findings[:12]
    heading = (
        "Historical report analysis:"
        if historical
        else "All tools completed successfully. Key results:"
    )
    lines = [heading]
    lines.extend(f"- {finding.title}: {finding.statement}" for finding in selected)
    if len(findings) > len(selected):
        lines.append(
            f"- {len(findings) - len(selected)} additional findings are retained "
            "in the structured analysis record."
        )
    return "\n".join(lines)


def _analyze_reports(
    state: AgentState,
    artifacts: list[Any],
    *,
    historical: bool,
) -> AnalysisResult:
    report_artifacts = select_report_artifacts(artifacts)
    summaries = [summarize_report_artifact(artifact) for artifact in report_artifacts]
    evidence, findings = build_report_evidence(summaries)
    references = [artifact_to_reference(artifact) for artifact in artifacts]
    limitations = [warning for summary in summaries for warning in summary.warnings]
    limitations.extend(_load_report_cautions({item.report_type for item in summaries}))
    limitations.extend(
        finding.statement
        for finding in findings
        if finding.title.startswith("Cross-report inconsistency:")
    )
    fallback_answer = _deterministic_scientific_answer(state, findings, historical)
    fallback_summary = (
        f"Analyzed {len(summaries)} historical reports."
        if historical
        else f"Executed the approved plan and analyzed {len(summaries)} reports."
    )

    if not summaries:
        fallback_answer = (
            "Relevant artifacts were found, but no formal JSON report could be "
            "parsed. Inspect the artifact paths and execution logs."
        )
        return _empty_result(
            state,
            response_mode="scientific_result",
            analysis_status="partial",
            summary=fallback_summary,
            final_answer=fallback_answer,
            referenced_artifacts=references,
            limitations=["No parseable report artifact was available."],
        )

    try:
        context = build_scientific_analysis_context(state, summaries, evidence, findings)
        output = invoke_structured(
            state,
            stage="analysis.scientific_narrative",
            schema=AnalysisNarrativeOutput,
            llm=_create_llm(state),
            messages=[
                SystemMessage(content=SCIENTIFIC_RESULT_SYSTEM_PROMPT),
                HumanMessage(content=context),
            ],
        )
        return _empty_result(
            state,
            response_mode="scientific_result",
            summary=output.summary,
            final_answer=output.final_answer,
            report_summaries=summaries,
            findings=findings,
            evidence=evidence,
            referenced_artifacts=references,
            limitations=list(dict.fromkeys(limitations + output.limitations)),
            used_language_model=True,
        )
    except Exception as exc:
        # 报告解析和确定性比较仍有效，因此模型叙述失败只降级为 partial。
        return _empty_result(
            state,
            response_mode="scientific_result",
            analysis_status="partial",
            summary=fallback_summary,
            final_answer=fallback_answer,
            report_summaries=summaries,
            findings=findings,
            evidence=evidence,
            referenced_artifacts=references,
            limitations=list(
                dict.fromkeys(
                    limitations
                    + [
                        "Language-model synthesis was unavailable; a deterministic "
                        "summary was returned."
                    ]
                )
            ),
            error_messages=[f"Analysis narrative generation failed: {exc}"],
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
        answer = (
            "No EnzyWizard tool was required, but answer generation failed. "
            "You can retry the question."
        )
        return _empty_result(
            state,
            response_mode=response_mode,
            analysis_status="failed",
            summary="General answer generation failed.",
            final_answer=answer,
            error_messages=[f"General answer generation failed: {exc}"],
        )


def _answer_routing_failure(state: AgentState) -> AnalysisResult:
    routing = state.get("current_request_routing")
    error = (
        routing.error_message
        if routing is not None
        else "Request routing result is missing."
    )
    return _empty_result(
        state,
        response_mode="request_routing_failure",
        analysis_status="failed",
        summary="The request could not be routed safely.",
        final_answer=(
            "The request could not be interpreted safely, so no tool was run. "
            "Please retry or make the requested action more explicit."
        ),
        error_messages=[error],
    )


def _answer_clarification(state: AgentState, question: str) -> AnalysisResult:
    return _empty_result(
        state,
        response_mode="clarification",
        summary="Clarification is required before continuing.",
        final_answer=question,
    )


def _answer_history_issue(state: AgentState) -> AnalysisResult:
    history = state.get("current_history_query")
    if history is None:
        detail = "History query result is missing."
        status = "failed"
    else:
        detail = history.error_message or history.summary
        status = "failed" if history.query_status == "failed" else "partial"
    return _empty_result(
        state,
        response_mode="history_query_failure",
        analysis_status=status,
        summary="The requested historical artifacts could not be resolved.",
        final_answer=detail,
        error_messages=[detail] if status == "failed" else [],
    )


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
            "final_answer": f"{result.final_answer}\n\n{listing.final_answer}",
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
    return result.model_copy(
        update={
            "summary": f"{result.summary} {general.summary}",
            "final_answer": f"{result.final_answer}\n\n{general.final_answer}",
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
