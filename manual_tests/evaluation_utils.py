"""Shared result helpers for live-model evaluation runners."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def agent_trace_payload(state: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return machine-evaluable planning, recovery, and evidence trace counts."""

    if not state:
        return {
            "validated_plan_count": 0,
            "recovery_attempt_count": 0,
            "workflow_decision_count": 0,
            "finding_count": 0,
            "supported_finding_count": 0,
            "evidence_count": 0,
            "claim_evidence_coverage": None,
            "ontology_evidence_count": 0,
            "evidence_with_source_uri": 0,
        }

    planning_results = list(state.get("planning_results", []))
    executions = list(state.get("tool_executions", []))
    analyses = list(state.get("analysis_results", []))
    current_analysis = state.get("current_analysis_result")
    if current_analysis is not None and all(
        getattr(item, "turn_id", None) != getattr(current_analysis, "turn_id", None)
        for item in analyses
    ):
        analyses.append(current_analysis)

    findings = [
        finding
        for analysis in analyses
        for finding in getattr(analysis, "findings", [])
    ]
    evidence = [
        item
        for analysis in analyses
        for item in getattr(analysis, "evidence", [])
    ]
    evidence_ids = {getattr(item, "evidence_id", "") for item in evidence}
    supported = sum(
        bool(getattr(finding, "evidence_ids", []))
        and all(
            evidence_id in evidence_ids
            for evidence_id in getattr(finding, "evidence_ids", [])
        )
        for finding in findings
    )
    return {
        "validated_plan_count": sum(
            bool(getattr(item, "plan_validation", None))
            and bool(getattr(item.plan_validation, "passed", False))
            for item in planning_results
        ),
        "recovery_attempt_count": sum(
            len(getattr(item, "recovery_attempts", [])) for item in executions
        ),
        "workflow_decision_count": sum(
            len(getattr(item, "workflow_decisions", [])) for item in executions
        ),
        "finding_count": len(findings),
        "supported_finding_count": supported,
        "evidence_count": len(evidence),
        "claim_evidence_coverage": (
            round(supported / len(findings), 6) if findings else None
        ),
        "ontology_evidence_count": sum(
            getattr(item, "source_kind", None) == "knowledge_base"
            and getattr(item, "metadata", {}).get("role")
            == "terminology_definition"
            for item in evidence
        ),
        "evidence_with_source_uri": sum(
            bool(getattr(item, "source_uri", None)) for item in evidence
        ),
    }


def aggregate_agent_traces(
    records: Iterable[Mapping[str, Any] | None],
) -> dict[str, Any]:
    """Aggregate per-case trace counters and recompute evidence coverage."""

    available = [record for record in records if record is not None]
    summed_fields = (
        "validated_plan_count",
        "recovery_attempt_count",
        "workflow_decision_count",
        "finding_count",
        "supported_finding_count",
        "evidence_count",
        "ontology_evidence_count",
        "evidence_with_source_uri",
    )
    totals = {
        field: sum(int(record.get(field, 0) or 0) for record in available)
        for field in summed_fields
    }
    finding_count = totals["finding_count"]
    totals["claim_evidence_coverage"] = (
        round(totals["supported_finding_count"] / finding_count, 6)
        if finding_count
        else None
    )
    return totals


def runtime_usage_payload(runtime: Any) -> dict[str, Any]:
    """Return a JSON-safe, compact view of one RuntimeSession."""

    if runtime is None:
        calls: list[Any] = []
        errors: list[Any] = []
        metadata: Mapping[str, Any] = {}
    elif isinstance(runtime, Mapping):
        calls = list(runtime.get("llm_calls", []))
        errors = list(runtime.get("errors", []))
        raw_metadata = runtime.get("metadata", {})
        metadata = raw_metadata if isinstance(raw_metadata, Mapping) else {}
    else:
        calls = list(getattr(runtime, "llm_calls", []))
        errors = list(getattr(runtime, "errors", []))
        raw_metadata = getattr(runtime, "metadata", {})
        metadata = raw_metadata if isinstance(raw_metadata, Mapping) else {}

    raw_retrievals = metadata.get("knowledge_retrieval", [])
    retrieval_items = raw_retrievals if isinstance(raw_retrievals, list) else []
    retrievals = [
        dict(item)
        for item in retrieval_items
        if isinstance(item, Mapping)
    ]

    def value(item: Any, name: str, default: Any = None) -> Any:
        if isinstance(item, Mapping):
            return item.get(name, default)
        return getattr(item, name, default)

    failed_stages = [
        str(value(call, "stage", "unknown"))
        for call in calls
        if value(call, "status") != "succeeded"
    ]
    stages: dict[str, dict[str, Any]] = {}
    for call in calls:
        stage = str(value(call, "stage", "unknown"))
        stage_usage = stages.setdefault(
            stage,
            {
                "call_count": 0,
                "failed_calls": 0,
                "duration_seconds": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
            },
        )
        stage_usage["call_count"] += 1
        stage_usage["failed_calls"] += value(call, "status") != "succeeded"
        stage_usage["duration_seconds"] += (
            value(call, "duration_seconds", 0.0) or 0.0
        )
        stage_usage["input_tokens"] += value(call, "input_tokens", 0) or 0
        stage_usage["output_tokens"] += value(call, "output_tokens", 0) or 0
        stage_usage["reasoning_tokens"] += value(call, "reasoning_tokens", 0) or 0
    for stage_usage in stages.values():
        stage_usage["duration_seconds"] = round(
            stage_usage["duration_seconds"],
            6,
        )
    return {
        "llm_call_count": len(calls),
        "input_tokens": sum(value(call, "input_tokens", 0) or 0 for call in calls),
        "output_tokens": sum(value(call, "output_tokens", 0) or 0 for call in calls),
        "reasoning_tokens": sum(
            value(call, "reasoning_tokens", 0) or 0 for call in calls
        ),
        "exact_usage_calls": sum(
            value(call, "usage_quality") == "exact" for call in calls
        ),
        "failed_llm_calls": len(failed_stages),
        "failed_stages": failed_stages,
        "stages": dict(sorted(stages.items())),
        "runtime_errors": [
            {
                "stage": value(error, "stage", "unknown"),
                "category": value(error, "category", "internal"),
                "message": value(error, "message", ""),
                "recoverable": bool(value(error, "recoverable", False)),
            }
            for error in errors
        ],
        "knowledge_retrieval": {
            "event_count": len(retrievals),
            "matched_event_count": sum(
                bool(item.get("injected_term_count")) for item in retrievals
            ),
            "injected_term_count": sum(
                int(item.get("injected_term_count", 0) or 0)
                for item in retrievals
            ),
            "estimated_input_tokens": sum(
                int(item.get("estimated_input_tokens", 0) or 0)
                for item in retrievals
            ),
            "events": retrievals,
        },
    }


def aggregate_usage(records: Iterable[Mapping[str, Any] | None]) -> dict[str, Any]:
    """Aggregate compact usage records without assuming provider token support."""

    available = [record for record in records if record is not None]
    failed_stages: list[str] = []
    runtime_errors: list[dict[str, Any]] = []
    stages: dict[str, dict[str, Any]] = {}
    retrieval_event_count = 0
    retrieval_matched_event_count = 0
    retrieval_injected_term_count = 0
    retrieval_estimated_input_tokens = 0
    for record in available:
        failed_stages.extend(str(item) for item in record.get("failed_stages", []))
        runtime_errors.extend(
            dict(item) for item in record.get("runtime_errors", [])
        )
        retrieval = record.get("knowledge_retrieval", {})
        if isinstance(retrieval, Mapping):
            retrieval_event_count += int(retrieval.get("event_count", 0) or 0)
            retrieval_matched_event_count += int(
                retrieval.get("matched_event_count", 0) or 0
            )
            retrieval_injected_term_count += int(
                retrieval.get("injected_term_count", 0) or 0
            )
            retrieval_estimated_input_tokens += int(
                retrieval.get("estimated_input_tokens", 0) or 0
            )
        for stage, values in record.get("stages", {}).items():
            stage_usage = stages.setdefault(
                stage,
                {
                    "call_count": 0,
                    "failed_calls": 0,
                    "duration_seconds": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "reasoning_tokens": 0,
                },
            )
            for name in (
                "call_count",
                "failed_calls",
                "duration_seconds",
                "input_tokens",
                "output_tokens",
                "reasoning_tokens",
            ):
                stage_usage[name] += values.get(name, 0)
    for stage_usage in stages.values():
        stage_usage["duration_seconds"] = round(
            stage_usage["duration_seconds"],
            6,
        )
    return {
        "record_count": len(available),
        "llm_call_count": sum(
            record.get("llm_call_count", 0) for record in available
        ),
        "input_tokens": sum(record.get("input_tokens", 0) for record in available),
        "output_tokens": sum(record.get("output_tokens", 0) for record in available),
        "reasoning_tokens": sum(
            record.get("reasoning_tokens", 0) for record in available
        ),
        "exact_usage_calls": sum(
            record.get("exact_usage_calls", 0) for record in available
        ),
        "failed_llm_calls": sum(
            record.get("failed_llm_calls", 0) for record in available
        ),
        "failed_stages": failed_stages,
        "stages": dict(sorted(stages.items())),
        "runtime_errors": runtime_errors,
        "knowledge_retrieval": {
            "event_count": retrieval_event_count,
            "matched_event_count": retrieval_matched_event_count,
            "injected_term_count": retrieval_injected_term_count,
            "estimated_input_tokens": retrieval_estimated_input_tokens,
        },
    }
