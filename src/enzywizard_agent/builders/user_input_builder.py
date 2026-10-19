from __future__ import annotations

from typing import Any

from ..state import AgentState


def build_user_input_interrupt_payload(state: AgentState) -> dict[str, Any]:
    """构造下一轮输入中断；payload 不包含 API key 或完整内部状态。"""

    current_turn_id = state.get("current_turn_id", 1)
    config = state["config"]
    if current_turn_id >= config.max_turns:
        raise ValueError("Cannot request another user input after max_turns.")

    analysis = state.get("current_analysis_result")
    if analysis is not None and analysis.turn_id == current_turn_id:
        final_answer = analysis.final_answer
        response_mode = analysis.response_mode
        analysis_status = analysis.analysis_status
    else:
        # 计划被用户拒绝时不会经过 Analysis，但仍应给出明确的轮次结束信息。
        final_answer = "The current plan was rejected. No tools were executed."
        response_mode = "plan_rejected"
        analysis_status = "not_run"

    return {
        "interrupt_type": "user_input",
        "completed_turn_id": current_turn_id,
        "next_turn_id": current_turn_id + 1,
        "max_turns": config.max_turns,
        "final_answer": final_answer,
        "response_mode": response_mode,
        "analysis_status": analysis_status,
        "available_artifact_turns": sorted(
            {artifact.turn_id for artifact in state.get("artifacts", [])}
        ),
    }
