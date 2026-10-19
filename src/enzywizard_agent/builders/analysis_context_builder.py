from __future__ import annotations

import json
from typing import Any

from ..data_models.analysis import AnalysisEvidence, AnalysisFinding, ReportSummary
from ..state import AgentState
from .request_routing_context_builder import build_prior_plan_catalog


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
            "key_metrics": dict(list(item.key_metrics.items())[:24]),
            "highlights": item.highlights,
            "warnings": item.warnings,
        }
        for item in report_summaries
    ]
    compact_evidence = [
        {
            key: value
            for key, value in item.model_dump(mode="json").items()
            if key != "path"
        }
        for item in evidence[:80]
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
        "prior_analysis_summary": state.get("analysis_summary", ""),
        "request_routing": _routing_payload(state),
        "history_query": _history_query_payload(state),
        "prior_approved_plans": build_prior_plan_catalog(state),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_general_answer_context(
    state: AgentState,
    tool_help: str,
    *,
    answer_scope: str = "Answer the routed non-tool request.",
) -> str:
    """为无需运行工具的回答提供会话摘要和本地能力说明。"""

    payload = {
        "user_request": state.get("current_user_message", ""),
        "turn_id": state.get("current_turn_id"),
        "prior_user_messages": state.get("user_message_summary", ""),
        "prior_analysis_summary": state.get("analysis_summary", ""),
        "local_tool_help": tool_help,
        "request_routing": _routing_payload(state),
        "prior_approved_plans": build_prior_plan_catalog(state),
        "answer_scope": answer_scope,
    }
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
