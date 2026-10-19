from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


# =============================================================================
# Checkpoint serialization configuration
# =============================================================================

# AgentState 中保存的是项目自己的 Pydantic 对象。显式列出允许反序列化的类型，
# 避免 JsonPlusSerializer 使用默认的“允许任意 Python 类型”兼容模式。
CHECKPOINT_ALLOWED_MSGPACK_MODULES = [
    ("enzywizard_agent.conf", "AgentConfig"),
    ("enzywizard_agent.data_models.agent_inputs", "AgentInputs"),
    ("enzywizard_agent.data_models.analysis", "AnalysisArtifactReference"),
    ("enzywizard_agent.data_models.analysis", "AnalysisEvidence"),
    ("enzywizard_agent.data_models.analysis", "AnalysisFinding"),
    ("enzywizard_agent.data_models.analysis", "AnalysisResult"),
    ("enzywizard_agent.data_models.analysis", "ReportSummary"),
    ("enzywizard_agent.data_models.artifact", "Artifact"),
    ("enzywizard_agent.data_models.artifact", "CleanedFiles"),
    ("enzywizard_agent.data_models.history_query", "HistoryQueryOutput"),
    ("enzywizard_agent.data_models.history_query", "HistoryQueryResult"),
    ("enzywizard_agent.data_models.planning", "PlanningResult"),
    ("enzywizard_agent.data_models.request_routing", "RequestedAction"),
    ("enzywizard_agent.data_models.request_routing", "RequestRoutingOutput"),
    ("enzywizard_agent.data_models.request_routing", "RequestRoutingResult"),
    ("enzywizard_agent.data_models.runtime", "AgentErrorRecord"),
    ("enzywizard_agent.data_models.runtime", "LLMCallRecord"),
    ("enzywizard_agent.data_models.runtime", "ModelCapabilityResult"),
    ("enzywizard_agent.data_models.runtime", "RuntimeSession"),
    ("enzywizard_agent.data_models.tool_execution", "ToolCallExecution"),
    ("enzywizard_agent.data_models.tool_execution", "ToolExecution"),
    ("enzywizard_agent.data_models.user_confirmation", "UserConfirmation"),
    ("enzywizard_agent.data_models.user_input", "UserTurnInput"),
    ("enzywizard_agent.data_models.user_message", "UserMessage"),
]


# =============================================================================
# Checkpoint construction
# =============================================================================

def create_in_memory_checkpointer() -> InMemorySaver:
    """创建供一次 CLI 会话使用的安全内存 checkpointer。"""

    serializer = JsonPlusSerializer(
        allowed_msgpack_modules=CHECKPOINT_ALLOWED_MSGPACK_MODULES,
    )
    return InMemorySaver(serde=serializer)
