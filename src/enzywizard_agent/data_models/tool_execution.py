from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .artifact import Artifact
from .planning import ToolCallSide


# =============================================================================
# Enum types
# =============================================================================

ToolCallExecutionStatus = Literal[
    # 当前工具调用及其必需产物均已成功完成。
    "succeeded",
    # 命令构造、准备、执行或产物校验中的任一步失败。
    "failed",
]

ToolExecutionStatus = Literal[
    # ToolPlan 中的全部调用均成功完成。
    "succeeded",
    # 执行在首个失败调用处停止，后续调用不再运行。
    "failed",
]

ToolFailureCategory = Literal[
    "transient_external",
    "timeout",
    "path_or_file",
    "missing_output",
    "invalid_output",
    "resource_exhaustion",
    "invalid_input",
    "missing_dependency",
    "scientific_configuration",
    "unknown",
]

ToolFailureStage = Literal[
    "preparation",
    "subprocess",
    "output_contract",
    "output_validation",
]

FailureRecoverability = Literal[
    "automatic",
    "confirmation_required",
    "not_recoverable",
]

RecoveryAction = Literal[
    "retry_same_call",
    "locate_artifact",
    "regenerate_upstream",
    "adjust_runtime_resources",
    "change_scientific_parameters",
    "request_user_input",
    "stop",
]

WorkflowDecisionAction = Literal[
    "continue",
    "retry",
    "request_confirmation",
    "stop",
]

AIFailureAction = Literal[
    "retry_same_call",
    "request_user_input",
    "request_restart",
    "request_parameter_confirmation",
    "stop",
]

AIFailureEvidenceId = Literal[
    "error_message",
    "stderr_excerpt",
    "stdout_excerpt",
    "log_excerpt",
    "deterministic_diagnosis",
    "workflow_context",
]


# 已解析参数不再包含 ToolOutputReference，只保留可直接传给 CLI 的值。
ResolvedToolArgumentValue = str | int | float | bool | list[str]


# =============================================================================
# Output schemas
# =============================================================================


class ToolFailureDiagnosis(BaseModel):
    """Deterministic diagnosis used to constrain recovery decisions."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    category: ToolFailureCategory
    failure_stage: ToolFailureStage
    recoverability: FailureRecoverability
    reason: str
    evidence: list[str]
    allowed_actions: list[RecoveryAction]
    scientific_change_required: bool = False

    @model_validator(mode="after")
    def validate_diagnosis(self) -> "ToolFailureDiagnosis":
        if not self.reason or not self.evidence or not self.allowed_actions:
            raise ValueError(
                "failure diagnosis requires a reason, evidence, and allowed actions"
            )
        if self.recoverability == "automatic" and "retry_same_call" not in self.allowed_actions:
            raise ValueError("automatic recovery requires retry_same_call")
        if self.scientific_change_required and self.recoverability != "confirmation_required":
            raise ValueError(
                "scientific changes must require explicit user confirmation"
            )
        return self


class AIFailureDiagnosis(BaseModel):
    """Advisory LLM interpretation constrained to auditable evidence and actions."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    likely_causes: list[str] = Field(min_length=1, max_length=3)
    evidence_ids: list[AIFailureEvidenceId] = Field(min_length=1, max_length=6)
    user_explanation: str
    recommended_action: AIFailureAction
    confidence: float = Field(ge=0.0, le=1.0)
    scientific_change_required: bool

    @model_validator(mode="after")
    def validate_ai_diagnosis(self) -> "AIFailureDiagnosis":
        if not self.user_explanation or any(not item for item in self.likely_causes):
            raise ValueError("AI failure diagnosis requires causes and an explanation")
        if (
            self.recommended_action == "retry_same_call"
            and self.scientific_change_required
        ):
            raise ValueError("same-call retry cannot require a scientific change")
        if (
            self.recommended_action == "request_parameter_confirmation"
            and not self.scientific_change_required
        ):
            raise ValueError(
                "parameter confirmation requires a declared scientific change"
            )
        return self


class AIFailureAssessmentRecord(BaseModel):
    """LLM diagnosis plus the deterministic policy decision applied to it."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    step_index: int
    attempt_index: int
    tool_name: str
    assessment_status: Literal["succeeded", "failed"]
    diagnosis: AIFailureDiagnosis | None
    policy_decision: WorkflowDecisionAction
    approved_action: AIFailureAction | None
    policy_reason: str
    error_message: str | None = None

    @model_validator(mode="after")
    def validate_assessment(self) -> "AIFailureAssessmentRecord":
        if self.step_index < 1 or self.attempt_index < 1 or not self.tool_name:
            raise ValueError("AI failure assessment requires valid call identifiers")
        if not self.policy_reason:
            raise ValueError("AI failure assessment requires a policy reason")
        if self.assessment_status == "succeeded":
            if (
                self.diagnosis is None
                or self.approved_action is None
                or self.error_message is not None
            ):
                raise ValueError(
                    "successful AI assessment requires a diagnosis, action, and no error"
                )
        elif (
            self.diagnosis is not None
            or self.approved_action is not None
            or not self.error_message
        ):
            raise ValueError(
                "failed AI assessment requires only an error"
            )
        if self.policy_decision == "retry" and self.approved_action != "retry_same_call":
            raise ValueError("retry policy decision requires retry_same_call")
        return self


class RecoveryAttemptRecord(BaseModel):
    """One bounded recovery action and its observed result."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    step_index: int
    attempt_index: int
    tool_name: str
    action: RecoveryAction
    trigger_category: ToolFailureCategory
    execution_status: ToolCallExecutionStatus
    previous_error_message: str
    result_error_message: str | None
    stdout_file: str | None
    stderr_file: str | None
    duration_seconds: float

    @model_validator(mode="after")
    def validate_attempt(self) -> "RecoveryAttemptRecord":
        if self.step_index < 1 or self.attempt_index < 2:
            raise ValueError("recovery attempts require valid step and attempt indexes")
        if self.duration_seconds < 0:
            raise ValueError("recovery duration must not be negative")
        if not self.tool_name or not self.previous_error_message:
            raise ValueError("recovery attempts require the triggering error")
        if self.execution_status == "failed" and not self.result_error_message:
            raise ValueError("failed recovery attempts require a result error")
        if self.execution_status == "succeeded" and self.result_error_message is not None:
            raise ValueError("successful recovery attempts cannot contain a result error")
        return self


class WorkflowDecisionRecord(BaseModel):
    """Auditable event-triggered decision made during plan execution."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    event_id: str
    event_type: Literal["call_succeeded", "execution_failure", "recovery_result"]
    step_index: int
    attempt_index: int
    decision: WorkflowDecisionAction
    selected_action: RecoveryAction | None = None
    reason: str
    allowed_actions: list[RecoveryAction]
    requires_user_confirmation: bool
    decision_source: Literal[
        "deterministic_policy",
        "llm_reviewed_policy",
    ] = "deterministic_policy"

    @model_validator(mode="after")
    def validate_decision(self) -> "WorkflowDecisionRecord":
        if not self.event_id or not self.reason:
            raise ValueError("workflow decisions require an ID and reason")
        if self.step_index < 1 or self.attempt_index < 1:
            raise ValueError("workflow decision indexes must be positive")
        if self.decision == "request_confirmation" and not self.requires_user_confirmation:
            raise ValueError("confirmation decisions must require user confirmation")
        if self.event_type == "execution_failure" and self.selected_action is None:
            raise ValueError("failure decisions must record the selected action")
        return self


class ToolCallExecution(BaseModel):
    """记录 ToolPlan 中一次具体调用的真实执行结果。"""

    # 这些标识与原 ToolCall 一一对应，便于从整份计划定位失败步骤。
    turn_id: int
    plan_id: int
    step_index: int
    tool_name: str
    side: ToolCallSide
    attempt_index: int = 1

    # command 保存实际提交给 subprocess 的参数列表；resolved_arguments 保存
    # 延迟引用解析后的值，便于审计执行时究竟使用了哪些输入。
    execution_status: ToolCallExecutionStatus
    command: list[str]
    resolved_arguments: dict[str, ResolvedToolArgumentValue]
    exit_code: int | None
    duration_seconds: float

    # stdout/stderr 始终保存到调用目录；工具自身可能产生一个或多个 log.txt。
    stdout_file: str
    stderr_file: str
    log_files: list[str]

    # 状态中只保存有限文本，完整内容保留在上面的文件中供 Analysis 按需读取。
    stdout_excerpt: str
    stderr_excerpt: str
    log_excerpt: str

    # artifacts 同时包含工具产物和 Agent 保存的 stdout/stderr 文件。
    artifacts: list[Artifact]
    error_message: str | None
    failure_diagnosis: ToolFailureDiagnosis | None = None

    @model_validator(mode="after")
    def validate_execution_result(self) -> "ToolCallExecution":
        """保证成功和失败结果使用一致的字段组合。"""

        if self.step_index < 1:
            raise ValueError("step_index must be greater than or equal to 1")
        if self.attempt_index < 1:
            raise ValueError("attempt_index must be greater than or equal to 1")
        if self.duration_seconds < 0:
            raise ValueError("duration_seconds must not be negative")
        if self.execution_status == "succeeded" and self.error_message is not None:
            raise ValueError("successful tool calls require error_message=None")
        if self.execution_status == "succeeded" and not self.command:
            raise ValueError("successful tool calls require a non-empty command")
        if self.execution_status == "failed" and not self.error_message:
            raise ValueError("failed tool calls require a non-empty error_message")
        if self.execution_status == "succeeded" and self.failure_diagnosis is not None:
            raise ValueError("successful tool calls cannot contain a failure diagnosis")
        return self


# =============================================================================
# State data models
# =============================================================================


class ToolExecution(BaseModel):
    """汇总一次已批准 ToolPlan 的完整执行记录。

    失败时保留此前成功调用及失败调用的全部记录，但不包含尚未运行的调用。
    """

    turn_id: int
    plan_id: int
    execution_status: ToolExecutionStatus
    call_executions: list[ToolCallExecution]
    failed_step_index: int | None
    error_messages: list[str]
    summary: str
    recovery_attempts: list[RecoveryAttemptRecord] = Field(default_factory=list)
    workflow_decisions: list[WorkflowDecisionRecord] = Field(default_factory=list)
    ai_failure_assessments: list[AIFailureAssessmentRecord] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_plan_execution(self) -> "ToolExecution":
        """校验计划级状态与各调用结果之间的一致性。"""

        step_indexes = [result.step_index for result in self.call_executions]
        if step_indexes != list(range(1, len(step_indexes) + 1)):
            raise ValueError("call_executions must use consecutive step indexes from 1")
        if any(result.turn_id != self.turn_id for result in self.call_executions):
            raise ValueError("all call executions must use the ToolExecution turn_id")
        if any(result.plan_id != self.plan_id for result in self.call_executions):
            raise ValueError("all call executions must use the ToolExecution plan_id")

        failed_results = [
            result
            for result in self.call_executions
            if result.execution_status == "failed"
        ]
        if self.execution_status == "succeeded":
            if not self.call_executions:
                raise ValueError("successful ToolExecution requires call executions")
            if failed_results or self.failed_step_index is not None or self.error_messages:
                raise ValueError("successful ToolExecution contains failure fields")
        else:
            if not self.error_messages:
                raise ValueError("failed ToolExecution requires error_messages")
            if len(failed_results) != 1:
                raise ValueError("failed ToolExecution requires exactly one failed call")
            if self.failed_step_index != failed_results[0].step_index:
                raise ValueError("failed_step_index does not match the failed call")
        if not self.summary:
            raise ValueError("ToolExecution summary must not be empty")
        if any(item.step_index > len(self.call_executions) for item in self.recovery_attempts):
            raise ValueError("recovery attempt references an unexecuted plan step")
        if any(
            item.step_index > len(self.call_executions)
            for item in self.ai_failure_assessments
        ):
            raise ValueError("AI assessment references an unexecuted plan step")
        return self
