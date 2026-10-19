from __future__ import annotations

from typing import Any

from ..state import AgentState


def build_user_confirmation_interrupt_payload(
    state: AgentState,
) -> dict[str, Any]:
    """构造发送给网页或 API 的工具计划确认中断内容。"""

    planning_result = state.get("current_planning_result")
    if planning_result is None:
        raise ValueError("Missing current_planning_result before user confirmation.")

    # 只有已经完成全部 Planning 步骤的结果才允许进入用户确认。
    if planning_result.planning_flag != "tool_plan_generated":
        raise ValueError(
            "User confirmation requires planning_flag='tool_plan_generated'."
        )
    if planning_result.planning_route != "tool_plan_generated":
        raise ValueError(
            "User confirmation requires planning_route='tool_plan_generated'."
        )
    if planning_result.tool_plan is None:
        raise ValueError("User confirmation requires a completed tool_plan.")

    tool_plan = planning_result.tool_plan
    if tool_plan.turn_id != planning_result.turn_id:
        raise ValueError("ToolPlan turn_id does not match PlanningResult turn_id.")
    if tool_plan.plan_id != planning_result.plan_id:
        raise ValueError("ToolPlan plan_id does not match PlanningResult plan_id.")

    # payload 只包含计划审阅所需信息，不传递 AgentConfig、API key 或完整状态。
    return {
        "interrupt_type": "tool_plan_confirmation",
        "turn_id": planning_result.turn_id,
        "plan_id": planning_result.plan_id,
        "purpose_type": planning_result.purpose_type,
        "summary": planning_result.summary,
        "plan_reason": tool_plan.plan_reason,
        "calls": [
            call.model_dump(mode="json")
            for call in tool_plan.calls
        ],
        # 外部调用方必须从下面三个固定值中选择，并原样带回 turn_id/plan_id。
        "allowed_decisions": [
            "approved",
            "revision_requested",
            "rejected",
        ],
        "revision_request_required_for": "revision_requested",
    }
