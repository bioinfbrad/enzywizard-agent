from __future__ import annotations

from ..state import AgentState


def tool_calling_node(state: AgentState) -> AgentState:
    """Tool Calling 节点占位函数。

    后续将按已批准 ToolPlan 执行命令、捕获结果、更新 CleanedFiles 和 artifacts，
    并把完整 ToolExecution 写回状态。当前尚未实现。
    """

    pass
