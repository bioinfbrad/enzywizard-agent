from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

# =============================================================================
# Enum types
# =============================================================================

ToolPlanPurposeType = Literal[
    # ToolPlan 只对应两种需要执行 EnzyWizard 工具的工作流。
    "single_protein_analysis",
    "wildtype_mutant_comparison",
]

PlanningRoute = Literal[
    # 已生成完整计划，下一步进入用户确认。
    "tool_plan_generated",
    # 当前请求无需工具，下一步直接进入 Analysis。
    "tool_unused",
]

PlanningFlag = Literal[
    # Planning 内部进度和失败状态，不直接等同于图路由。
    "started",
    "intent_recognized",
    "tool_selection_succeeded",
    "tool_selection_failed",
    "tool_plan_generated",
    "tool_unused",
    "planning_failed",
]

PlanningPurposeType = Literal[
    # 意图识别允许的固定输出枚举。
    "single_protein_analysis",
    "wildtype_mutant_comparison",
    "other",
]

ToolSelectionStatus = Literal[
    # 工具选择阶段的结构化成功或失败状态。
    "tool_selection_succeeded",
    "tool_selection_failed",
]


# =============================================================================
# Output schemas
# =============================================================================

class IntentClassificationOutput(BaseModel):
    """LLM 意图识别步骤的结构化输出模型。"""

    # 拒绝模型输出的额外字段，并自动清理字符串首尾空白。
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    # 固定工作流类型及模型给出的简短判定原因。
    purpose_type: PlanningPurposeType
    purpose_reason: str

    @field_validator("purpose_type", mode="before")
    @classmethod
    def normalize_purpose_type(cls, value: object) -> object:
        """在 Literal 校验前统一意图值的大小写和分隔符格式。"""

        # 只规范化字符串；非字符串交给 Pydantic 按原类型报告错误。
        if isinstance(value, str):
            return re.sub(r"[-\s]+", "_", value.strip().casefold())
        return value

    @field_validator("purpose_reason")
    @classmethod
    def validate_purpose_reason(cls, value: str) -> str:
        """阻止模型返回空的意图判定原因。"""

        if not value:
            raise ValueError("purpose_reason must not be empty")
        return value


class ToolSelectionOutput(BaseModel):
    """LLM 工具选择步骤以及程序合并后的统一输出结构。"""

    # 这里只校验字段和基础类型，跨字段业务一致性由 planning_utils 处理。
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    # 状态由 LLM 返回；空 selected_tools 是否表示“仅准备”由程序结合准备工具判断。
    tool_selection_status: ToolSelectionStatus

    # 成功时 selected_tools 保存最终工具名，失败时必须为空。
    selected_tools: list[str]

    # 成功原因与失败原因分开保存，最终只能有一个字段非空。
    selection_reason: str
    error_message: str

    @field_validator("selected_tools", mode="before")
    @classmethod
    def normalize_selected_tool_names(cls, value: object) -> object:
        """在候选枚举校验前统一工具名的大小写和分隔符。"""

        if isinstance(value, list):
            return [
                re.sub(r"[_\s]+", "-", item.strip().casefold())
                if isinstance(item, str)
                else item
                for item in value
            ]
        return value


# =============================================================================
# State data models
# =============================================================================

class ToolCall(BaseModel):
    """描述完整 ToolPlan 中一次具体的命令行工具调用。"""

    # 调用所属轮次、计划以及计划内执行顺序。
    turn_id: int
    plan_id: int
    step_index: int

    # 逻辑工具名和实际 CLI 入口名称。
    tool_name: str
    command_name: str

    # 当前调用独占的输出目录，后续参数规划阶段生成。
    tool_execution_dir: str

    # 已完成类型转换的命令参数和预计生成的产物类型或路径模式。
    arguments: dict[str, str | int | float | bool | list[str]]
    expected_outputs: list[str]

    # 解释当前计划为什么需要这次调用。
    call_reason: str


class ToolPlan(BaseModel):
    """保存一个 Planning 结果中的完整、有序工具调用计划。"""

    turn_id: int
    plan_id: int
    purpose_type: ToolPlanPurposeType

    # summary 可以在不需要简短摘要时为 None；plan_reason 必须说明总体目的。
    summary: str | None
    plan_reason: str

    # calls 按 step_index 表示计划执行顺序。
    calls: list[ToolCall]


class PlanningResult(BaseModel):
    """汇总一次 Planning 节点从意图到完整工具计划的全部结果。"""

    # 当前轮次、该轮中的计划编号，以及图路由和 Planning 内部进度标识。
    turn_id: int
    plan_id: int
    planning_route: PlanningRoute
    planning_flag: PlanningFlag

    # 是否需要 EnzyWizard 工具，以及可选的整体摘要。
    needs_tool: bool
    summary: str | None

    # 意图识别结果；初始化或识别失败时 purpose_type 可以为 None。
    purpose_type: PlanningPurposeType | None
    purpose_reason: str

    # 工具选择和完整 ToolPlan 按 Planning 进度逐步写入。
    tool_selection: ToolSelectionOutput | None
    tool_plan: ToolPlan | None

    # 保存本次 Planning 过程中产生的所有错误。
    error_messages: list[str]
