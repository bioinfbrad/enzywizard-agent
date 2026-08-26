from __future__ import annotations

from ..state import AgentState


def user_input_interrupt_node(state: AgentState) -> AgentState:
    """等待下一轮用户消息的中断节点占位函数。

    后续会更新轮次、当前消息和当前轮输入，再将 route 引回 Planning。当前尚未
    实现。
    """

    pass
