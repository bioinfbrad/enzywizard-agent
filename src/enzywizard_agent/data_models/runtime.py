from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


ErrorCategory = Literal[
    "user_input",
    "model",
    "tool",
    "file",
    "timeout",
    "budget",
    "internal",
]

RuntimeStatus = Literal[
    "running",
    "completed",
    "failed",
    "cancelled",
    "timed_out",
    "budget_exceeded",
]


class AgentRuntimeError(RuntimeError):
    """带稳定错误类别的 Agent 基础异常。"""

    category: ErrorCategory = "internal"


class UserInputError(AgentRuntimeError):
    category: ErrorCategory = "user_input"


class ModelContractError(AgentRuntimeError):
    category: ErrorCategory = "model"


class RuntimeBudgetError(AgentRuntimeError):
    category: ErrorCategory = "budget"


class SessionTimeoutError(AgentRuntimeError):
    category: ErrorCategory = "timeout"


class ToolTimeoutError(AgentRuntimeError):
    category: ErrorCategory = "timeout"


class AgentErrorRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    timestamp: str
    stage: str
    category: ErrorCategory
    message: str
    recoverable: bool


class LLMCallRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    stage: str
    started_at: str
    duration_seconds: float
    status: Literal["succeeded", "failed"]
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    reasoning_tokens: int | None
    usage_quality: Literal["exact", "estimated", "unavailable"]
    error_category: ErrorCategory | None = None


class ModelCapabilityResult(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    status: Literal["passed", "failed", "skipped"]
    chat_completions: bool
    system_messages: bool
    strict_structured_output: bool
    token_usage_available: bool
    checked_at: str
    detail: str
    cached: bool = False


class RuntimeSession(BaseModel):
    """会话内共享的资源用量和错误台账，可随图状态完成 checkpoint 恢复。"""

    model_config = ConfigDict(extra="forbid")

    started_at: str
    started_monotonic: float
    status: RuntimeStatus = "running"
    llm_calls: list[LLMCallRecord] = Field(default_factory=list)
    errors: list[AgentErrorRecord] = Field(default_factory=list)
    model_capability: ModelCapabilityResult | None = None
    stop_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def input_tokens(self) -> int:
        return sum(item.input_tokens or 0 for item in self.llm_calls)

    @property
    def output_tokens(self) -> int:
        return sum(item.output_tokens or 0 for item in self.llm_calls)
