from __future__ import annotations

from typing import Any

from ..data_models.user_confirmation import UserConfirmation
from ..state import AgentState


CONFIRMATION_ROUTES = {
    "approved": "tool_execution_approved",
    "revision_requested": "tool_plan_revision_requested",
    "rejected": "tool_execution_rejected",
}


def record_user_confirmation(
    state: AgentState,
    resume_value: Any,
) -> AgentState:
    """校验中断恢复输入，并把用户决定写入状态和下一步路由。"""

    planning_result = state.get("current_planning_result")
    if planning_result is None:
        raise ValueError("Missing current_planning_result before user confirmation.")
    if planning_result.tool_plan is None:
        raise ValueError("Cannot confirm a PlanningResult without a tool_plan.")

    # model_validate 同时执行字段类型、额外字段和跨字段业务校验。
    confirmation = UserConfirmation.model_validate(resume_value)

    # 中断页面可能长期停留；恢复时必须再次核对当前轮次和计划，避免旧确认
    # 被错误应用到用户后来生成的新计划。
    if confirmation.turn_id != planning_result.turn_id:
        raise ValueError(
            "Confirmation turn_id does not match the current PlanningResult."
        )
    if confirmation.plan_id != planning_result.plan_id:
        raise ValueError(
            "Confirmation plan_id does not match the current PlanningResult."
        )

    # current 字段供紧随其后的节点读取；历史列表用于后续摘要和审计。
    state["current_user_confirmation"] = confirmation
    state.setdefault("user_confirmations", []).append(confirmation)
    state["user_confirmation_summary"] = "\n".join(
        f"Turn {item.turn_id}, plan {item.plan_id}: {item.decision}"
        + (f" ({item.revision_request})" if item.revision_request else "")
        for item in state["user_confirmations"][-12:]
    )
    state["route"] = CONFIRMATION_ROUTES[confirmation.decision]
    return state
