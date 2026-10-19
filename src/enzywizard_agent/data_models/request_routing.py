from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


RequestType = Literal[
    "tool_workflow",
    "history_analysis",
    "artifact_query",
    "mixed",
    "software_help",
    "general_scientific_question",
    "conversation",
    "out_of_scope",
    "clarification",
    "exit",
]

RequestActionType = Literal[
    "run_tool_workflow",
    "analyze_history",
    "list_artifacts",
    "reuse_historical_output",
    "reference_historical_context",
    "answer_software_help",
    "answer_general_question",
    "continue_conversation",
    "decline_out_of_scope",
    "clarify_request",
    "exit_session",
]

RoutingActionKind = Literal[
    "run_workflow",
    "rerun_previous_workflow",
    "recompute_workflow",
    "summarize_history",
    "explain_history",
    "compare_history",
    "list_history_files",
    "reuse_history_output_required",
    "reuse_history_output_optional",
    "reference_previous_plan",
    "software_help",
    "science_question",
    "conversation",
    "out_of_scope",
    "clarify",
    "exit",
]

RequestedHistoryOperation = Literal[
    "summarize",
    "explain",
    "compare",
    "list_artifacts",
    "reuse_as_input",
]

ReusePolicy = Literal[
    "none",
    "allowed",
    "required",
]

RequestRoutingStatus = Literal[
    "succeeded",
    "failed",
]

RequestRoute = Literal[
    "request_to_planning",
    "request_to_history_query",
    "request_to_analysis",
]


class RequestedAction(BaseModel):
    """保存 Router 从当前用户消息中识别出的一个独立动作。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action_type: RequestActionType
    target: str
    # 历史动作的具体操作只在 Router 识别一次，History Planner 不再重新猜测
    # summarize / explain / compare / list / reuse 之间的区别。
    history_operation: RequestedHistoryOperation | None = None

    @field_validator("target")
    @classmethod
    def validate_target(cls, value: str) -> str:
        if not value:
            raise ValueError("action target must not be empty")
        return value


class RoutingDraftAction(BaseModel):
    """A minimal semantic action identified by the routing model."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    kind: RoutingActionKind = Field(
        description="One self-contained user-visible action."
    )
    target: str = Field(
        description="Concise target copied or faithfully paraphrased from the request."
    )

    @field_validator("target")
    @classmethod
    def validate_target(cls, value: str) -> str:
        if not value:
            raise ValueError("action target must not be empty")
        return value


class RequestRoutingDraft(BaseModel):
    """The minimal model-facing request-routing contract."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    actions: list[RoutingDraftAction]

    @model_validator(mode="after")
    def validate_draft(self) -> "RequestRoutingDraft":
        if not self.actions:
            raise ValueError("actions must contain at least one requested action")
        return self


class RequestRoutingOutput(BaseModel):
    """Request Router LLM 的严格结构化输出。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    request_type: RequestType
    actions: list[RequestedAction]
    requires_tool_execution: bool
    references_history: bool
    requires_history_artifacts: bool
    reuse_policy: ReusePolicy
    requests_recomputation: bool
    clarification_question: str | None
    routing_reason: str

    @model_validator(mode="after")
    def validate_routing_output(self) -> "RequestRoutingOutput":
        if not self.actions:
            raise ValueError("actions must contain at least one requested action")
        if not self.routing_reason:
            raise ValueError("routing_reason must not be empty")

        action_types = {item.action_type for item in self.actions}
        history_actions = {
            "analyze_history",
            "list_artifacts",
            "reuse_historical_output",
        }
        all_history_actions = history_actions | {"reference_historical_context"}
        if self.requires_tool_execution != ("run_tool_workflow" in action_types):
            raise ValueError(
                "requires_tool_execution must match the run_tool_workflow action"
            )
        if self.references_history != bool(action_types & all_history_actions):
            raise ValueError(
                "references_history must match history-related actions"
            )
        if self.requires_history_artifacts != bool(action_types & history_actions):
            raise ValueError(
                "requires_history_artifacts must match artifact-related history actions"
            )
        if self.requires_history_artifacts and not self.references_history:
            raise ValueError(
                "history artifact resolution requires a history reference"
            )
        if self.requests_recomputation and not self.requires_tool_execution:
            raise ValueError("recomputation requests require tool execution")
        if self.request_type == "exit":
            if action_types != {"exit_session"}:
                raise ValueError("exit requests must contain only exit_session")
            if (
                self.requires_tool_execution
                or self.references_history
                or self.requires_history_artifacts
            ):
                raise ValueError("exit requests cannot require tools or history")
        if self.request_type == "clarification":
            if action_types != {"clarify_request"}:
                raise ValueError(
                    "clarification requests must contain only clarify_request"
                )
            if not self.clarification_question:
                raise ValueError(
                    "clarification requests require clarification_question"
                )
            if self.requires_tool_execution:
                raise ValueError("clarification requests cannot execute tools")
            if self.references_history or self.requires_history_artifacts:
                raise ValueError(
                    "clarification requests cannot resolve history before clarification"
                )
        elif self.clarification_question is not None:
            raise ValueError(
                "clarification_question is only valid for clarification requests"
            )

        if self.request_type == "tool_workflow" and not self.requires_tool_execution:
            raise ValueError("tool_workflow requires tool execution")
        if self.request_type in {"history_analysis", "artifact_query"}:
            if not self.requires_history_artifacts:
                raise ValueError(
                    f"{self.request_type} requires history artifact resolution"
                )
            if self.requires_tool_execution:
                raise ValueError(
                    f"{self.request_type} cannot require tool execution"
                )
        if self.request_type == "mixed" and len(self.actions) < 2:
            raise ValueError("mixed requests require at least two actions")
        required_action_by_type = {
            "tool_workflow": "run_tool_workflow",
            "history_analysis": "analyze_history",
            "artifact_query": "list_artifacts",
            "software_help": "answer_software_help",
            "general_scientific_question": "answer_general_question",
            "conversation": "continue_conversation",
            "out_of_scope": "decline_out_of_scope",
            "clarification": "clarify_request",
            "exit": "exit_session",
        }
        required_action = required_action_by_type.get(self.request_type)
        if required_action is not None and required_action not in action_types:
            raise ValueError(
                f"{self.request_type} requires the {required_action} action"
            )
        if self.reuse_policy != "none":
            if not self.references_history or not self.requires_tool_execution:
                raise ValueError(
                    "historical output reuse requires history and tool execution"
                )
            if not self.requires_history_artifacts:
                raise ValueError(
                    "historical output reuse requires history artifact resolution"
                )
            if "reuse_historical_output" not in action_types:
                raise ValueError(
                    "reuse_policy requires a reuse_historical_output action"
                )
        elif "reuse_historical_output" in action_types:
            raise ValueError(
                "reuse_historical_output requires an allowed or required policy"
            )
        return self


class RequestRoutingResult(BaseModel):
    """保存一次 Router 调用结果以及程序计算出的正式图路由。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    turn_id: int
    routing_status: RequestRoutingStatus
    route: RequestRoute
    output: RequestRoutingOutput | None
    summary: str
    error_message: str

    @model_validator(mode="after")
    def validate_result(self) -> "RequestRoutingResult":
        if self.turn_id < 1:
            raise ValueError("turn_id must be greater than or equal to 1")
        if not self.summary:
            raise ValueError("summary must not be empty")
        if self.routing_status == "succeeded":
            if self.output is None or self.error_message:
                raise ValueError(
                    "successful routing requires output and no error_message"
                )
        elif self.output is not None or not self.error_message:
            raise ValueError(
                "failed routing requires error_message and output=None"
            )
        return self
