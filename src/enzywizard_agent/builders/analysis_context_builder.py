from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..data_models.analysis import AnalysisEvidence, AnalysisFinding, ReportSummary
from ..state import AgentState
from ..utils.terminology_utils import scientific_answer_body
from .request_routing_context_builder import build_prior_plan_catalog


MAX_PRIOR_ANALYSES = 3
MAX_PRIOR_ANSWER_CHARS = 6000
MAX_PRIOR_FINDINGS = 16
MAX_PRIOR_EVIDENCE = 32


def _clip_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def _bounded_evidence(
    evidence: list[AnalysisEvidence], limit: int
) -> list[AnalysisEvidence]:
    """Keep report roots and calculations before optional field details."""

    primary = [
        item
        for item in evidence
        if item.metadata.get("role") != "report_metric"
    ]
    metrics = [
        item
        for item in evidence
        if item.metadata.get("role") == "report_metric"
    ]
    return (primary + metrics)[:limit]


def _explicit_prior_analysis_payload(state: AgentState) -> list[dict[str, Any]]:
    """Return only analyses from turns explicitly resolved by history query."""

    history = state.get("current_history_query")
    if history is None or history.query is None:
        return []
    if not set(history.query.operations).intersection(
        {"summarize", "explain", "compare"}
    ):
        return []

    current_turn = int(state.get("current_turn_id", 0))
    resolved_turns = set(history.resolved_turn_ids)
    selected = [
        result
        for result in state.get("analysis_results", [])
        if result.turn_id in resolved_turns
        and result.turn_id < current_turn
        and result.response_mode == "scientific_result"
    ][-MAX_PRIOR_ANALYSES:]

    payload: list[dict[str, Any]] = []
    for result in selected:
        report_evidence = [
            item for item in result.evidence
            if item.metadata.get("role") != "terminology_definition"
        ]
        evidence_id_map = {
            item.evidence_id: f"prior-turn-{result.turn_id}-{item.evidence_id}"
            for item in report_evidence
        }
        evidence = []
        for item in _bounded_evidence(report_evidence, MAX_PRIOR_EVIDENCE):
            dumped = item.model_dump(mode="json", exclude={"path", "derivation"})
            dumped["evidence_id"] = evidence_id_map[item.evidence_id]
            dumped["source_evidence_ids"] = [
                evidence_id_map.get(source_id, source_id)
                for source_id in item.source_evidence_ids
            ]
            dumped["artifact_name"] = Path(item.path).name if item.path else None
            evidence.append(dumped)
        findings = []
        for item in result.findings[:MAX_PRIOR_FINDINGS]:
            dumped = item.model_dump(mode="json")
            dumped["evidence_ids"] = [
                evidence_id_map.get(evidence_id, evidence_id)
                for evidence_id in item.evidence_ids
            ]
            findings.append(dumped)
        payload.append(
            {
                "turn_id": result.turn_id,
                "summary": _clip_text(result.summary, 1200),
                "final_answer": _clip_text(
                    scientific_answer_body(result.final_answer) or "",
                    MAX_PRIOR_ANSWER_CHARS,
                ),
                "findings": findings,
                "evidence": evidence,
                "limitations": result.limitations[:12],
            }
        )
    return payload


def build_scientific_analysis_context(
    state: AgentState,
    report_summaries: list[ReportSummary],
    evidence: list[AnalysisEvidence],
    findings: list[AnalysisFinding],
) -> str:
    """构造有界科学分析上下文，不包含原始大数组或完整图数据。"""

    # 模型上下文进一步收紧到每份报告前 24 个有序指标。完整摘要和本地路径仍
    # 保存在 AnalysisResult 中，模型不需要重复接收这些审计数据。
    compact_reports = [
        {
            "turn_id": item.turn_id,
            "plan_id": item.plan_id,
            "artifact_type": item.artifact_type,
            "report_type": item.report_type,
            "source_tool": item.source_tool,
            "side": item.side,
            "artifact_name": Path(item.path).name,
            "key_metrics": dict(list(item.key_metrics.items())[:24]),
            "highlights": item.highlights,
            "warnings": item.warnings,
            **({"read_status": item.read_status} if item.read_status != "parsed" else {}),
        }
        for item in report_summaries
    ]
    compact_evidence = [
        item.model_dump(mode="json", exclude={"path", "derivation"})
        for item in _bounded_evidence(evidence, 80)
    ]
    payload: dict[str, Any] = {
        "user_request": state.get("current_user_message", ""),
        "turn_id": state.get("current_turn_id"),
        "agent_inputs": state["agent_inputs"].model_dump(mode="json"),
        "report_summaries": compact_reports,
        "deterministic_evidence": compact_evidence,
        "deterministic_findings": [
            item.model_dump(mode="json") for item in findings[:80]
        ],
        "request_routing": _routing_payload(state),
        "history_query": _history_query_payload(state),
        "prior_approved_plans": build_prior_plan_catalog(state),
    }
    prior_analyses = _explicit_prior_analysis_payload(state)
    if prior_analyses:
        payload["explicitly_referenced_prior_analyses"] = prior_analyses
        payload["prior_analysis_instruction"] = (
            "Use the referenced prior analysis only to answer the requested follow-up "
            "or revision. Recheck every retained or revised scientific statement "
            "against the supplied current report summaries and deterministic evidence."
        )
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_general_answer_context(
    state: AgentState,
    tool_help: str,
    *,
    answer_scope: str = "Answer the routed non-tool request.",
    include_prior_context: bool = True,
) -> str:
    """为无需运行工具的回答提供会话摘要和本地能力说明。"""

    payload: dict[str, Any] = {
        "user_request": state.get("current_user_message", ""),
        "turn_id": state.get("current_turn_id"),
        "local_tool_help": tool_help,
        "request_routing": _routing_payload(state),
        "answer_scope": answer_scope,
    }
    if include_prior_context:
        payload.update(
            {
                "prior_user_messages": state.get("user_message_summary", ""),
                "prior_analysis_summary": state.get("analysis_summary", ""),
                "prior_approved_plans": build_prior_plan_catalog(state),
            }
        )
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _routing_payload(state: AgentState) -> dict[str, Any] | None:
    """只把 Router 的语义结论传给回答模型，不暴露运行时对象。"""

    result = state.get("current_request_routing")
    if result is None or result.output is None:
        return None
    return result.output.model_dump(mode="json")


def _history_query_payload(state: AgentState) -> dict[str, Any] | None:
    """提供历史逻辑查询及已解析轮次，不重复传入文件路径。"""

    result = state.get("current_history_query")
    if result is None or result.query is None:
        return None
    return {
        "query": result.query.model_dump(mode="json"),
        "resolved_turn_ids": result.resolved_turn_ids,
        "selected_artifact_count": len(result.selected_artifacts),
    }
