from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, model_validator

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


# 已解析参数不再包含 ToolOutputReference，只保留可直接传给 CLI 的值。
ResolvedToolArgumentValue = str | int | float | bool | list[str]


# =============================================================================
# Output schemas
# =============================================================================


class ToolCallExecution(BaseModel):
    """记录 ToolPlan 中一次具体调用的真实执行结果。"""

    # 这些标识与原 ToolCall 一一对应，便于从整份计划定位失败步骤。
    turn_id: int
    plan_id: int
    step_index: int
    tool_name: str
    side: ToolCallSide

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

    @model_validator(mode="after")
    def validate_execution_result(self) -> "ToolCallExecution":
        """保证成功和失败结果使用一致的字段组合。"""

        if self.step_index < 1:
            raise ValueError("step_index must be greater than or equal to 1")
        if self.duration_seconds < 0:
            raise ValueError("duration_seconds must not be negative")
        if self.execution_status == "succeeded" and self.error_message is not None:
            raise ValueError("successful tool calls require error_message=None")
        if self.execution_status == "succeeded" and not self.command:
            raise ValueError("successful tool calls require a non-empty command")
        if self.execution_status == "failed" and not self.error_message:
            raise ValueError("failed tool calls require a non-empty error_message")
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
        return self
