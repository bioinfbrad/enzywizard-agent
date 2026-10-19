from __future__ import annotations

from datetime import datetime, timezone
import time
from typing import Any, Mapping

from ..data_models.runtime import (
    AgentErrorRecord,
    AgentRuntimeError,
    ErrorCategory,
    RuntimeBudgetError,
    RuntimeSession,
    SessionTimeoutError,
)
from ..state import AgentState
from ..utils.security_utils import redact_text


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_runtime_session(state: AgentState) -> RuntimeSession:
    runtime = state.get("runtime_session")
    if runtime is None:
        runtime = RuntimeSession(
            started_at=utc_now(),
            started_monotonic=time.monotonic(),
        )
        state["runtime_session"] = runtime
    elif isinstance(runtime, Mapping):
        # 兼容旧 checkpoint 或受限反序列化器产生的普通映射，同时让后续代码
        # 始终只处理经过 Pydantic 校验的 RuntimeSession。
        runtime = RuntimeSession.model_validate(runtime)
        state["runtime_session"] = runtime
    elif not isinstance(runtime, RuntimeSession):
        raise TypeError(
            "runtime_session must be a RuntimeSession, mapping, or None, "
            f"got {type(runtime).__name__}."
        )
    return runtime


def classify_exception(error: BaseException) -> ErrorCategory:
    category = getattr(error, "category", None)
    if category in {
        "user_input", "model", "tool", "file", "timeout", "budget", "internal"
    }:
        return category
    if isinstance(error, (TimeoutError,)) or "timeout" in type(error).__name__.casefold():
        return "timeout"
    if isinstance(error, (FileNotFoundError, PermissionError, OSError)):
        return "file"
    name = type(error).__name__.casefold()
    if any(marker in name for marker in ("openai", "api", "rate", "model")):
        return "model"
    return "internal"


def record_runtime_error(
    state: AgentState,
    *,
    stage: str,
    error: BaseException | str,
    category: ErrorCategory | None = None,
    recoverable: bool = False,
) -> AgentErrorRecord:
    runtime = ensure_runtime_session(state)
    config = state.get("config")
    message = redact_text(
        error,
        api_key=config.api_key if config is not None else None,
    )
    record = AgentErrorRecord(
        timestamp=utc_now(),
        stage=stage,
        category=category or (
            classify_exception(error) if isinstance(error, BaseException) else "internal"
        ),
        message=message,
        recoverable=recoverable,
    )
    runtime.errors.append(record)
    return record


def elapsed_session_seconds(state: AgentState) -> float:
    return max(0.0, time.monotonic() - ensure_runtime_session(state).started_monotonic)


def remaining_session_seconds(state: AgentState) -> float | None:
    config = state.get("config")
    if config is None:
        return None
    timeout = config.session_timeout_seconds
    if timeout is None:
        return None
    return timeout - elapsed_session_seconds(state)


def enforce_session_deadline(state: AgentState, stage: str) -> None:
    remaining = remaining_session_seconds(state)
    if remaining is not None and remaining <= 0:
        runtime = ensure_runtime_session(state)
        runtime.status = "timed_out"
        runtime.stop_reason = "session_timeout"
        error = SessionTimeoutError(
            f"Session time budget was exhausted before stage {stage}."
        )
        record_runtime_error(state, stage=stage, error=error, category="timeout")
        raise error


def enforce_llm_budget(state: AgentState, stage: str, estimated_input_tokens: int) -> None:
    enforce_session_deadline(state, stage)
    config = state.get("config")
    if config is None:
        return
    runtime = ensure_runtime_session(state)
    if config.max_llm_calls is not None and len(runtime.llm_calls) >= config.max_llm_calls:
        runtime.status = "budget_exceeded"
        runtime.stop_reason = "max_llm_calls"
        error = RuntimeBudgetError(
            f"LLM call budget of {config.max_llm_calls} was exhausted before stage {stage}."
        )
        record_runtime_error(state, stage=stage, error=error, category="budget")
        raise error
    if (
        config.max_input_tokens is not None
        and runtime.input_tokens + estimated_input_tokens > config.max_input_tokens
    ):
        runtime.status = "budget_exceeded"
        runtime.stop_reason = "max_input_tokens"
        error = RuntimeBudgetError(
            f"Input token budget of {config.max_input_tokens} would be exceeded at stage {stage}."
        )
        record_runtime_error(state, stage=stage, error=error, category="budget")
        raise error
    if (
        config.max_output_tokens is not None
        and runtime.output_tokens >= config.max_output_tokens
    ):
        runtime.status = "budget_exceeded"
        runtime.stop_reason = "max_output_tokens"
        error = RuntimeBudgetError(
            f"Output token budget of {config.max_output_tokens} was exhausted before stage {stage}."
        )
        record_runtime_error(state, stage=stage, error=error, category="budget")
        raise error


def is_runtime_limit_error(error: BaseException) -> bool:
    return isinstance(error, (RuntimeBudgetError, SessionTimeoutError))
