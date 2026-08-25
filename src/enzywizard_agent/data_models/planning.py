from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from .knowledge_query import KnowledgeQuery


# =============================================================================
# Enum types
# =============================================================================

ToolPlanPurposeType = Literal[
    "single_protein_analysis",
    "wildtype_mutant_comparison",
]

PlanningRoute = Literal[
    "tool_plan_generated",
    "tool_unused",
]

PlanningFlag = Literal[
    "started",
    "intent_recognized",
    "tool_selection_succeeded",
    "tool_selection_failed",
    "tool_plan_generated",
    "tool_unused",
    "planning_failed",
]

PlanningPurposeType = Literal[
    "single_protein_analysis",
    "wildtype_mutant_comparison",
    "other",
]

ToolSelectionStatus = Literal[
    "tool_selection_succeeded",
    "tool_selection_failed",
]


# =============================================================================
# Output schemas
# =============================================================================

class IntentClassificationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    purpose_type: PlanningPurposeType
    purpose_reason: str

    @field_validator("purpose_type", mode="before")
    @classmethod
    def normalize_purpose_type(cls, value: object) -> object:
        if isinstance(value, str):
            return re.sub(r"[-\s]+", "_", value.strip().casefold())
        return value

    @field_validator("purpose_reason")
    @classmethod
    def validate_purpose_reason(cls, value: str) -> str:
        if not value:
            raise ValueError("purpose_reason must not be empty")
        return value


class ToolSelectionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    tool_selection_status: ToolSelectionStatus
    selected_tools: list[str]
    selection_reason: str
    error_message: str


# =============================================================================
# State data models
# =============================================================================

class ToolCall(BaseModel):
    turn_id: int
    plan_id: int
    step_index: int
    tool_name: str
    command_name: str
    tool_execution_dir: str
    arguments: dict[str, str | int | float | bool | list[str]]
    expected_outputs: list[str]
    call_reason: str


class ToolPlan(BaseModel):
    turn_id: int
    plan_id: int
    purpose_type: ToolPlanPurposeType
    summary: str | None
    plan_reason: str
    calls: list[ToolCall]


class PlanningResult(BaseModel):
    turn_id: int
    planning_route: PlanningRoute
    planning_flag: PlanningFlag
    needs_tool: bool
    summary: str | None
    purpose_type: PlanningPurposeType | None
    purpose_reason: str
    tool_selection: ToolSelectionOutput | None
    tool_plan: ToolPlan | None
    error_messages: list[str]
    knowledge_queries: list[KnowledgeQuery]
