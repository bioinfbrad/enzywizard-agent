from __future__ import annotations

from typing import Dict, List, TypedDict

from .conf import AgentConfig
from .data_models import (
    AgentMessage,
    AnalysisResult,
    Artifact,
    CleanedFiles,
    InputFiles,
    PlanningResult,
    ToolCall,
    UserConfirmation,
)


class AgentState(TypedDict, total=False):
    config: AgentConfig
    input_files: InputFiles
    turn_count: int
    route: str

    current_user_message: str
    user_messages: List[AgentMessage]
    user_message_summary: str

    current_planning_result: PlanningResult
    planning_results: List[PlanningResult]
    planning_summary: str

    current_user_confirmation: UserConfirmation
    user_confirmations: List[UserConfirmation]
    user_confirmation_summary: str

    current_tool_call: ToolCall
    tool_calls: List[ToolCall]
    tool_calling_summary: str

    current_analysis_result: AnalysisResult
    analysis_results: List[AnalysisResult]
    analysis_summary: str

    cleaned_files: CleanedFiles
    current_artifact: Artifact
    artifacts: List[Artifact]
    artifact_summary: str
