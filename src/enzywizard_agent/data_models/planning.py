from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

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
    "tool_plan_generated",
    "tool_unused",
    "planning_failed",
]

PlanningPurposeType = Literal[
    "single_protein_analysis",
    "wildtype_mutant_comparison",
    "direct_answer",
    "help_request",
    "exit_request",
    "other",
]


# =============================================================================
# Output schemas
# =============================================================================

class IntentClassificationOutput(BaseModel):
    purpose_type: PlanningPurposeType
    purpose_reason: str


# =============================================================================
# State data models
# =============================================================================

class ToolCall(BaseModel):
    turn: int
    step_index: int
    tool_name: str
    command_name: str
    arguments: dict[str, str | int | float | bool | list[str]]
    expected_outputs: list[str]
    call_reason: str


class ToolPlan(BaseModel):
    turn: int
    purpose_type: ToolPlanPurposeType
    summary: str | None
    plan_reason: str
    calls: list[ToolCall]


class PlanningResult(BaseModel):
    turn: int
    planning_route: PlanningRoute
    planning_flag: PlanningFlag
    needs_tool: bool
    summary: str | None
    purpose_type: PlanningPurposeType
    purpose_reason: str
    tool_plan: ToolPlan | None
    error_messages: list[str]
    knowledge_queries: list[KnowledgeQuery]
