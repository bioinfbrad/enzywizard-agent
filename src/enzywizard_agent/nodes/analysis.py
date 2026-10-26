from __future__ import annotations

from ..data_models.analysis import AnalysisErrorGuidance, AnalysisResult
from ..services.analysis_service import analyze_agent_state
from ..services.user_input_service import archive_completed_turn
from ..state import AgentState
from ..utils.analysis_utils import compact_history_summary


def analysis_node(state: AgentState) -> AgentState:
    """分析当前轮结果、生成回答并决定继续对话或退出。"""

    try:
        result = analyze_agent_state(state)
    except Exception as exc:
        # 未预料的 Analysis 程序错误仍转换成合法状态，避免图直接丢失 checkpoint。
        guidance = AnalysisErrorGuidance(
            error_type="analysis_error",
            stage="analysis_node",
            reason=f"Unexpected analysis error: {exc}",
            can_continue_current_session=False,
            restart_required=True,
            user_actions=[
                "Exit the current session.",
                "Inspect the session manifest, then restart the original request.",
            ],
        )
        result = AnalysisResult(
            turn_id=state.get("current_turn_id", 1),
            response_mode="tool_execution_failure",
            analysis_status="failed",
            analysis_route="restart_required",
            termination_reason="restart_required",
            summary="Analysis failed because of an internal error.",
            final_answer=(
                "Error type: `analysis_error`\n"
                "Stage: `analysis_node`\n"
                "Current session: cannot continue; restart is required.\n"
                f"Reason: Unexpected analysis error: {exc}\n\n"
                "Required action:\n"
                "- Exit the current session.\n"
                "- Inspect the session manifest, then restart the original request."
            ),
            report_summaries=[],
            findings=[],
            evidence=[],
            referenced_artifacts=[],
            limitations=[],
            error_messages=[f"Unexpected analysis error: {exc}"],
            used_language_model=False,
            error_guidance=guidance,
        )

    state["current_analysis_result"] = result
    state.setdefault("analysis_results", []).append(result)
    state["analysis_summary"] = compact_history_summary(
        state["analysis_results"]
    )
    archive_completed_turn(state)

    # 资源边界一旦触发，本轮仍尽可能返回已有证据和部分结果，但不得再进入
    # 下一轮输入。这样不会把预算耗尽误报成普通完成，也不会丢失已有产物。
    runtime = state.get("runtime_session")
    if runtime is not None and runtime.status in {"timed_out", "budget_exceeded"}:
        result.analysis_route = "exit_requested"
        result.termination_reason = (
            "session_timeout"
            if runtime.status == "timed_out"
            else "budget_exceeded"
        )
        reason = runtime.stop_reason or runtime.status
        result.limitations.append(
            f"The session ended because the configured runtime limit was reached: {reason}."
        )

    # max_turns 是会话级硬上限。最后一轮仍返回正常答案，但不再请求下一轮输入。
    config = state["config"]
    if (
        result.analysis_route == "answer_generated"
        and state.get("current_turn_id", 1) >= config.max_turns
    ):
        result.analysis_route = "exit_requested"
        result.termination_reason = "max_turns"
        result.limitations.append(
            f"The configured maximum of {config.max_turns} turns was reached."
        )

    state["route"] = result.analysis_route
    return state
