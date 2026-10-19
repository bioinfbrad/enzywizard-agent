from __future__ import annotations

from langgraph.types import interrupt

from ..builders.user_input_builder import build_user_input_interrupt_payload
from ..services.user_input_service import record_next_user_input
from ..state import AgentState


def user_input_interrupt_node(state: AgentState) -> AgentState:
    """暂停图并在恢复后登记下一轮自然语言消息。"""

    # 与计划确认节点相同：interrupt 前不修改状态，确保 checkpoint 重放幂等。
    payload = build_user_input_interrupt_payload(state)
    resume_value = interrupt(payload)
    return record_next_user_input(state, resume_value)
