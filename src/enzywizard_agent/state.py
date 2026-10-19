from __future__ import annotations

from typing import List, TypedDict

from .conf import AgentConfig
from .data_models.analysis import AnalysisResult
from .data_models.agent_inputs import AgentInputs
from .data_models.artifact import Artifact, CleanedFiles
from .data_models.planning import PlanningResult
from .data_models.history_query import HistoryQueryResult
from .data_models.request_routing import RequestRoutingResult
from .data_models.runtime import RuntimeSession
from .data_models.tool_execution import ToolExecution
from .data_models.user_confirmation import UserConfirmation
from .data_models.user_message import UserMessage


class AgentState(TypedDict, total=False):
    """LangGraph 节点之间共享的完整状态定义。

    total=False 表示构建中间状态时可以只提供当前阶段已有的字段；各节点仍需在
    读取必需字段前确认它已经由入口或上游节点写入。
    """

    # 会话级配置、标识、当前输入和图路由值。
    config: AgentConfig
    session_id: str
    agent_inputs: AgentInputs
    current_turn_id: int
    current_plan_id: int
    route: str

    # 运行时台账记录模型调用、资源用量和分类错误，并随图中断 checkpoint 恢复。
    runtime_session: RuntimeSession

    # 用户消息：当前消息单独保存，历史消息和摘要用于后续上下文压缩。
    current_user_message: str
    user_messages: List[UserMessage]
    user_message_summary: str

    # Request Router：保存本轮结构化动作、历史引用意图和路由历史。
    current_request_routing: RequestRoutingResult
    request_routing_results: List[RequestRoutingResult]
    request_routing_summary: str

    # History Query Planner：保存逻辑查询及程序解析出的真实历史产物。
    current_history_query: HistoryQueryResult
    history_query_results: List[HistoryQueryResult]
    history_query_summary: str
    current_history_artifacts: List[Artifact]

    # Planning 阶段：当前结果、历史结果以及历史摘要。
    current_planning_result: PlanningResult
    planning_results: List[PlanningResult]
    planning_summary: str

    # 用户确认阶段：保存本轮确认结果、历史确认记录和摘要。
    current_user_confirmation: UserConfirmation
    user_confirmations: List[UserConfirmation]
    user_confirmation_summary: str

    # 工具执行阶段：一次 ToolExecution 对应一个完整计划的执行记录。
    current_tool_execution: ToolExecution
    tool_executions: List[ToolExecution]
    tool_calling_summary: str

    # Analysis 阶段：当前回答结果、历史分析结果和摘要。
    current_analysis_result: AnalysisResult
    analysis_results: List[AnalysisResult]
    analysis_summary: str

    # 文件追踪：CleanedFiles 可跨轮复用，普通 artifacts 保存所有计划产物。
    cleaned_files: CleanedFiles
    current_plan_artifacts: List[Artifact]
    artifacts: List[Artifact]
    artifact_summary: str
