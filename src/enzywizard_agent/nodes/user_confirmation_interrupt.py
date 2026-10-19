from __future__ import annotations

from langgraph.types import interrupt

from ..builders.user_confirmation_builder import (
    build_user_confirmation_interrupt_payload,
)
from ..services.user_confirmation_service import record_user_confirmation
from ..state import AgentState


def user_confirmation_interrupt_node(state: AgentState) -> AgentState:
    """暂停状态图，等待用户批准、修改或拒绝当前工具计划。

    节点第一次执行时，interrupt 会把计划 payload 返回给外部调用方并暂停图。
    外部使用同一个 thread_id 和 Command(resume=...) 恢复后，interrupt 返回恢复
    数据，节点再校验并写入 UserConfirmation 和对应 route。
    """

    # 中断之前只构造纯数据，不修改状态，确保恢复时节点重新执行不会重复写历史。
    interrupt_payload = build_user_confirmation_interrupt_payload(state)
    resume_value = interrupt(interrupt_payload)

    # 所有状态变更都发生在 interrupt 成功恢复之后。
    return record_user_confirmation(state, resume_value)
