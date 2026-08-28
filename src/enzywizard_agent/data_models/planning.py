from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

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
    "default_tool_plan_generated",
    "parameter_extraction_succeeded",
    "parameter_extraction_failed",
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

ParameterExtractionStatus = Literal[
    # 参数提取模型成功理解用户要求，允许程序继续校验和应用。
    "parameter_extraction_succeeded",
    # 模型无法可靠提取参数要求，Planning 应停止并交给 Analysis。
    "parameter_extraction_failed",
]

ParameterProfile = Literal[
    # fast/accurate 只表达用户偏好，实际数值由程序中的工具级规则决定。
    "fast",
    "accurate",
]

UnsupportedRequestType = Literal[
    # 输入和输出路径均由 Agent 状态与工作目录规则管理。
    "input_path_override",
    "output_path_override",
    # 其他不允许由用户或 LLM 修改的程序管理参数。
    "program_managed_parameter",
]

ToolCallSide = Literal[
    # single 用于单蛋白调用；wt/mut 是配对工作流的两侧调用。
    "single",
    "wt",
    "mut",
    # shared 用于 mut-clean、共享底物、mut-batch 和 mut-integrate 等调用。
    "shared",
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


class ParameterAdjustment(BaseModel):
    """LLM 从用户文本中提取的一项明确参数修改。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    # step_index 精确定位 ToolPlan 中的一次调用，避免 WT/MUT 同名工具混淆。
    step_index: int
    parameter_name: str
    parameter_value: str | int | float | bool
    adjustment_reason: str

    @model_validator(mode="after")
    def validate_adjustment(self) -> "ParameterAdjustment":
        """确保修改可以定位且具有可审计原因。"""

        if self.step_index < 1:
            raise ValueError("step_index must be greater than or equal to 1")
        if not self.parameter_name:
            raise ValueError("parameter_name must not be empty")
        if not self.adjustment_reason:
            raise ValueError("adjustment_reason must not be empty")
        return self


class ParameterProfileRequest(BaseModel):
    """保存用户针对一个或多个调用提出的速度或精度偏好。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    step_indexes: list[int]
    profile: ParameterProfile
    profile_reason: str

    @model_validator(mode="after")
    def validate_profile_request(self) -> "ParameterProfileRequest":
        """确保 profile 目标非空、有效且不重复。"""

        if not self.step_indexes:
            raise ValueError("step_indexes must not be empty")
        if any(step_index < 1 for step_index in self.step_indexes):
            raise ValueError("step_indexes must be greater than or equal to 1")
        if len(self.step_indexes) != len(set(self.step_indexes)):
            raise ValueError("step_indexes must not contain duplicates")
        if not self.profile_reason:
            raise ValueError("profile_reason must not be empty")
        return self


class UnsupportedParameterRequest(BaseModel):
    """记录 LLM 识别到的程序管理参数修改要求。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    # 无法归属某个调用时允许 step_index 和 tool_name 为 None，但参数名和原始值
    # 必须保留，供程序产生稳定错误并让 Analysis 向用户解释。
    step_index: int | None
    tool_name: str | None
    parameter_name: str
    requested_value: str
    request_type: UnsupportedRequestType

    @model_validator(mode="after")
    def validate_unsupported_request(self) -> "UnsupportedParameterRequest":
        """确保被拒绝要求保留足够的定位和原始值信息。"""

        if self.step_index is not None and self.step_index < 1:
            raise ValueError("step_index must be greater than or equal to 1")
        if not self.parameter_name:
            raise ValueError("parameter_name must not be empty")
        if not self.requested_value:
            raise ValueError("requested_value must not be empty")
        return self


class ParameterExtractionOutput(BaseModel):
    """参数提取 LLM 的统一结构化输出。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    parameter_extraction_status: ParameterExtractionStatus
    parameter_adjustments: list[ParameterAdjustment]
    profile_requests: list[ParameterProfileRequest]
    unsupported_requests: list[UnsupportedParameterRequest]
    extraction_summary: str
    error_message: str

    @model_validator(mode="after")
    def validate_status_fields(self) -> "ParameterExtractionOutput":
        """校验成功和失败状态对应的字段组合。"""

        request_lists = (
            self.parameter_adjustments,
            self.profile_requests,
            self.unsupported_requests,
        )
        if self.parameter_extraction_status == "parameter_extraction_succeeded":
            if not self.extraction_summary:
                raise ValueError("successful extraction requires extraction_summary")
            if self.error_message:
                raise ValueError("successful extraction requires an empty error_message")
        else:
            if any(request_lists):
                raise ValueError("failed extraction requires all request lists to be empty")
            if self.extraction_summary:
                raise ValueError("failed extraction requires an empty extraction_summary")
            if not self.error_message:
                raise ValueError("failed extraction requires error_message")
        return self


# =============================================================================
# State data models
# =============================================================================

class ToolOutputReference(BaseModel):
    """引用当前计划中尚未执行的上游工具产物。"""

    # source_step_index 指向上游 ToolCall；artifact_type 用于执行后从 artifacts
    # 中解析真实路径。expected_path_pattern 只用于展示和执行前检查。
    source_step_index: int
    artifact_type: str
    expected_path_pattern: str
    json_field: str | None


class ExpectedToolOutput(BaseModel):
    """描述一次工具调用预计产生的一类文件。"""

    artifact_type: str
    path_pattern: str
    side: ToolCallSide


class ReportStagingRule(BaseModel):
    """描述 integrate/mut-integrate 执行前的报告复制规则。"""

    # source_path_pattern 可以指向当前调用目录下的 *.json，也可以是复用的
    # clean report 绝对路径；执行器只复制匹配文件，不复制日志或结构文件。
    source_step_index: int | None
    source_path_pattern: str
    destination_dir: str
    side: ToolCallSide


ToolArgumentValue = str | int | float | bool | list[str] | ToolOutputReference

class ToolCall(BaseModel):
    """描述完整 ToolPlan 中一次具体的命令行工具调用。"""

    # 调用所属轮次、计划以及计划内执行顺序。
    turn_id: int
    plan_id: int
    step_index: int

    # 逻辑工具名和实际 CLI 入口名称。
    tool_name: str
    command_name: str
    side: ToolCallSide

    # 当前调用独占的输出目录，后续参数规划阶段生成。
    tool_execution_dir: str

    # 参数可以是立即可用的标量，也可以引用尚未执行的上游产物。执行器在真正
    # 调用 CLI 前解析 ToolOutputReference。
    arguments: dict[str, ToolArgumentValue]

    # expected_outputs 用于执行后扫描产物；staging_rules 仅供整合工具使用。
    expected_outputs: list[ExpectedToolOutput]
    staging_rules: list[ReportStagingRule]

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

    @model_validator(mode="after")
    def validate_calls(self) -> "ToolPlan":
        """保证调用顺序、所属 ID 和目录在计划内一致。"""

        if not self.calls:
            raise ValueError("ToolPlan must contain at least one ToolCall")
        expected_step_indexes = list(range(1, len(self.calls) + 1))
        if [call.step_index for call in self.calls] != expected_step_indexes:
            raise ValueError("ToolCall step_index values must be consecutive from 1")
        if any(call.turn_id != self.turn_id for call in self.calls):
            raise ValueError("all ToolCalls must use the ToolPlan turn_id")
        if any(call.plan_id != self.plan_id for call in self.calls):
            raise ValueError("all ToolCalls must use the ToolPlan plan_id")
        execution_dirs = [call.tool_execution_dir for call in self.calls]
        if len(execution_dirs) != len(set(execution_dirs)):
            raise ValueError("ToolCall tool_execution_dir values must be unique")
        return self


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
    parameter_extraction: ParameterExtractionOutput | None
    tool_plan: ToolPlan | None

    # 保存本次 Planning 过程中产生的所有错误。
    error_messages: list[str]
