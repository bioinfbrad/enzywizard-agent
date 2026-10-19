from __future__ import annotations

from typing import Any

from ..data_models.user_input import UserTurnInput
from ..data_models.user_message import UserMessage
from ..state import AgentState
from ..utils.analysis_utils import compact_history_summary


TRANSIENT_TURN_FIELDS = (
    "current_request_routing",
    "current_history_query",
    "current_planning_result",
    "current_user_confirmation",
    "current_tool_execution",
    "current_analysis_result",
)


def _archive_current_user_message(state: AgentState) -> None:
    """按 turn_id 幂等归档消息，防止中断恢复重放造成重复历史。"""

    turn_id = state.get("current_turn_id", 1)
    messages = state.setdefault("user_messages", [])
    if any(item.turn_id == turn_id for item in messages):
        return
    messages.append(
        UserMessage(
            turn_id=turn_id,
            content=state.get("current_user_message", ""),
        )
    )
    state["user_message_summary"] = "\n".join(
        f"Turn {item.turn_id}: {item.content}" for item in messages[-12:]
    )


def archive_completed_turn(state: AgentState) -> AgentState:
    """供 Analysis 和拒绝分支统一归档已结束轮次。"""

    _archive_current_user_message(state)
    state["planning_summary"] = compact_history_summary(
        state.get("planning_results", [])
    )
    state["tool_calling_summary"] = compact_history_summary(
        state.get("tool_executions", [])
    )
    state["artifact_summary"] = "\n".join(
        f"Turn {artifact.turn_id}, plan {artifact.plan_id}: "
        f"{artifact.artifact_type} -> {artifact.path}"
        for artifact in state.get("artifacts", [])[-30:]
    )
    return state


def record_next_user_input(state: AgentState, resume_value: Any) -> AgentState:
    """校验下一轮消息并重置仅属于上一轮的临时状态。"""

    next_input = UserTurnInput.model_validate(resume_value)
    current_turn_id = state.get("current_turn_id", 1)
    expected_turn_id = current_turn_id + 1
    if next_input.turn_id != expected_turn_id:
        raise ValueError(
            f"User input turn_id must be {expected_turn_id}, got {next_input.turn_id}."
        )
    if next_input.turn_id > state["config"].max_turns:
        raise ValueError("User input exceeds the configured max_turns.")

    archive_completed_turn(state)

    # AgentInputs、CleanedFiles、artifacts 和各阶段历史故意保留；下面字段只能服务
    # 于刚完成的一轮，若不清理会被 Analysis 误认为新一轮的执行结果。
    for field_name in TRANSIENT_TURN_FIELDS:
        state.pop(field_name, None)
    state["current_plan_artifacts"] = []
    state["current_history_artifacts"] = []
    state["current_plan_id"] = 0
    state["current_turn_id"] = next_input.turn_id
    state["current_user_message"] = next_input.message
    state["route"] = "new_user_input_received"
    return state
