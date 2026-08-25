from __future__ import annotations

from typing import List, TypedDict

from .conf import AgentConfig
from .data_models.analysis import AnalysisResult
from .data_models.agent_inputs import AgentInputs
from .data_models.artifact import Artifact, CleanedFiles
from .data_models.planning import PlanningResult
from .data_models.tool_execution import ToolExecution
from .data_models.user_confirmation import UserConfirmation
from .data_models.user_message import UserMessage


class AgentState(TypedDict, total=False):
    config: AgentConfig
    session_id: str
    agent_inputs: AgentInputs
    current_turn_id: int
    current_plan_id: int
    route: str

    current_user_message: str
    user_messages: List[UserMessage]
    user_message_summary: str

    current_planning_result: PlanningResult
    planning_results: List[PlanningResult]
    planning_summary: str

    current_user_confirmation: UserConfirmation
    user_confirmations: List[UserConfirmation]
    user_confirmation_summary: str

    current_tool_execution: ToolExecution
    tool_executions: List[ToolExecution]
    tool_calling_summary: str

    current_analysis_result: AnalysisResult
    analysis_results: List[AnalysisResult]
    analysis_summary: str

    cleaned_files: CleanedFiles
    current_plan_artifacts: List[Artifact]
    artifacts: List[Artifact]
    artifact_summary: str
