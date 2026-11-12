from __future__ import annotations

from ..data_models.analysis import AnalysisErrorGuidance, AnalysisResult
from ..services.analysis_service import analyze_agent_state
from ..services.user_input_service import archive_completed_turn
from ..state import AgentState
from ..utils.analysis_utils import compact_history_summary


def _runtime_limit_action(state: AgentState, stop_reason: str) -> str:
    """Return an actionable restart instruction for one configured limit."""

    config = state["config"]
    if stop_reason == "max_llm_calls":
        current = config.max_llm_calls
        suggestion = max(128, (current or 0) * 2)
        return (
            "Start a new session with a higher LLM-call limit, for example "
            f"`--max_llm_calls {suggestion}`."
        )
    if stop_reason == "max_input_tokens":
        current = config.max_input_tokens
        suggestion = max(1_000_000, (current or 0) * 2)
        return (
            "Start a new session with a higher input-token limit, for example "
            f"`--max_input_tokens {suggestion}`."
        )
    if stop_reason == "max_output_tokens":
        current = config.max_output_tokens
        suggestion = max(256_000, (current or 0) * 2)
        return (
            "Start a new session with a higher output-token limit, for example "
            f"`--max_output_tokens {suggestion}`."
        )
    return (
        "Start a new session with a longer session limit using "
        "`--session_timeout_seconds <seconds>`."
    )


def _apply_runtime_limit(state: AgentState, result: AnalysisResult) -> None:
    """Make the displayed response agree with a terminal runtime state."""

    runtime = state.get("runtime_session")
    if runtime is None or runtime.status not in {"timed_out", "budget_exceeded"}:
        return

    stop_reason = runtime.stop_reason or runtime.status
    termination_reason = (
        "session_timeout" if runtime.status == "timed_out" else "budget_exceeded"
    )
    reason = next(
        (
            item.message
            for item in reversed(runtime.errors)
            if item.category in {"timeout", "budget"}
        ),
        f"The configured runtime limit was reached: {stop_reason}.",
    )
    action = _runtime_limit_action(state, stop_reason)
    notice = (
        f"Limit: `{stop_reason}`\n"
        "Current session: cannot continue because this process has ended.\n"
        f"Reason: {reason}\n\n"
        "Required action:\n"
        f"- {action}\n"
        "- Existing reports and artifacts remain available in the session work directory."
    )

    if result.analysis_status == "failed":
        # A limit reached before the requested stage produced no usable answer;
        # replace misleading stage-specific text such as "routing failure".
        result.response_mode = "runtime_limit"
        result.summary = "The session ended at a configured runtime limit."
        result.final_answer = notice
    else:
        # Preserve a completed or partial scientific answer when usage was found
        # to exceed a limit only after the LLM response was recorded.
        result.analysis_status = "partial"
        result.final_answer = f"{result.final_answer}\n\nSession limit:\n{notice}"

    result.analysis_route = "exit_requested"
    result.termination_reason = termination_reason
    result.error_guidance = None
    result.error_messages = list(dict.fromkeys(result.error_messages + [reason]))
    result.limitations = list(
        dict.fromkeys(
            result.limitations
            + [f"The session ended at the configured runtime limit: {stop_reason}."]
        )
    )


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

    # 资源边界一旦触发，本轮仍尽可能返回已有证据和部分结果，但不得再进入
    # 下一轮输入。这样不会把预算耗尽误报成普通完成，也不会丢失已有产物。
    _apply_runtime_limit(state, result)

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

    # Archive only after all terminal-state normalization so summaries and the
    # manifest cannot retain a pre-normalization route or contradictory guidance.
    state["current_analysis_result"] = result
    state.setdefault("analysis_results", []).append(result)
    state["analysis_summary"] = compact_history_summary(
        state["analysis_results"]
    )
    archive_completed_turn(state)

    state["route"] = result.analysis_route
    return state
