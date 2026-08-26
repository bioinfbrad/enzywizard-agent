from __future__ import annotations

from ..state import AgentState


def user_confirmation_interrupt_node(state: AgentState) -> AgentState:
    """等待用户批准、修改或拒绝工具计划的中断节点占位函数。

    后续会使用 LangGraph interrupt 暂停执行，并根据恢复输入写入 UserConfirmation
    和对应 route。当前尚未实现。
    """

    pass
