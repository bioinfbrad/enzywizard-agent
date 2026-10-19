"""Shared result helpers for live-model evaluation runners."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def runtime_usage_payload(runtime: Any) -> dict[str, Any]:
    """Return a JSON-safe, compact view of one RuntimeSession."""

    if runtime is None:
        calls: list[Any] = []
        errors: list[Any] = []
    elif isinstance(runtime, Mapping):
        calls = list(runtime.get("llm_calls", []))
        errors = list(runtime.get("errors", []))
    else:
        calls = list(getattr(runtime, "llm_calls", []))
        errors = list(getattr(runtime, "errors", []))

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
    }


def aggregate_usage(records: Iterable[Mapping[str, Any] | None]) -> dict[str, Any]:
    """Aggregate compact usage records without assuming provider token support."""

    available = [record for record in records if record is not None]
    failed_stages: list[str] = []
    runtime_errors: list[dict[str, Any]] = []
    stages: dict[str, dict[str, Any]] = {}
    for record in available:
        failed_stages.extend(str(item) for item in record.get("failed_stages", []))
        runtime_errors.extend(
            dict(item) for item in record.get("runtime_errors", [])
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
    }
