from __future__ import annotations

import json
from typing import Any


_MESSAGE_TYPE_LABELS = {
    "scientific_result": "Scientific result",
    "general_scientific_question": "General scientific answer",
    "planning_failure": "Workflow planning failure",
    "plan_rejected": "Plan rejected",
    "tool_execution_failure": "Tool execution failure",
    "artifact_query": "Artifact query",
    "software_help": "Software help",
    "conversation": "Conversation",
    "out_of_scope": "Out-of-scope response",
    "clarification": "Clarification request",
    "input_error": "Required input error",
    "request_routing_failure": "Request routing failure",
    "history_query_failure": "History query failure",
    "exit": "Session exit",
}

_BODY_SECTION_LABELS = {
    "scientific_result": "Scientific analysis",
    "general_scientific_question": "Scientific answer",
    "planning_failure": "Planning error report",
    "plan_rejected": "Plan decision",
    "tool_execution_failure": "Execution failure report",
    "artifact_query": "Artifacts",
    "software_help": "Software help",
    "conversation": "Response",
    "out_of_scope": "Scope response",
    "clarification": "Clarification required",
    "input_error": "Input error report",
    "request_routing_failure": "Routing error report",
    "history_query_failure": "History error report",
    "exit": "Session status",
}


def _humanize_identifier(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    return text.replace("_", " ").capitalize()


def _message_type_label(response_mode: Any) -> str:
    return _MESSAGE_TYPE_LABELS.get(
        str(response_mode or ""),
        _humanize_identifier(response_mode, "Agent response"),
    )


def _body_section_label(response_mode: Any) -> str:
    return _BODY_SECTION_LABELS.get(
        str(response_mode or ""),
        "Response",
    )


def _session_state_label(
    *,
    analysis_route: Any,
    termination_reason: Any,
    awaiting_input: bool,
) -> str:
    if awaiting_input:
        return "Waiting for the next user message"
    if analysis_route == "restart_required":
        return "Ended; restart required"
    if analysis_route == "exit_requested":
        reason = _humanize_identifier(termination_reason, "completed")
        return f"Ended ({reason})"
    return "Completed"


def _display_message_envelope(
    *,
    response_mode: Any,
    analysis_status: Any,
    analysis_route: Any,
    termination_reason: Any,
    final_answer: str,
    awaiting_input: bool,
) -> None:
    """Render one answer with stable user-facing type and section labels."""

    print("\n=== EnzyWizard-Agent ===")
    print(f"Message type: {_message_type_label(response_mode)}")
    print(
        "Status: "
        + _humanize_identifier(analysis_status, "Not available")
    )
    print(
        "Session: "
        + _session_state_label(
            analysis_route=analysis_route,
            termination_reason=termination_reason,
            awaiting_input=awaiting_input,
        )
    )
    print(f"\n{_body_section_label(response_mode)}:")
    print(final_answer)


def get_single_interrupt_payload(result: dict[str, Any]) -> dict[str, Any]:
    """从 LangGraph 执行结果中读取唯一的中断 payload。"""

    graph_interrupts = result.get("__interrupt__")
    if not graph_interrupts:
        raise ValueError("Agent graph result does not contain an interrupt.")
    if len(graph_interrupts) != 1:
        raise ValueError("CLI mode supports exactly one active interrupt at a time.")

    payload = graph_interrupts[0].value
    if not isinstance(payload, dict):
        raise ValueError("Agent interrupt payload must be a dictionary.")
    return payload


def display_tool_plan_confirmation(payload: dict[str, Any]) -> None:
    """以便于终端审阅的格式展示当前 ToolPlan。"""

    print("\n=== Tool Plan Confirmation ===")
    print(f"Turn ID: {payload['turn_id']}")
    print(f"Plan ID: {payload['plan_id']}")
    print(f"Purpose: {payload['purpose_type']}")
    if payload.get("summary"):
        print(f"Summary: {payload['summary']}")
    print(f"Reason: {payload['plan_reason']}")

    # 每个步骤都展示实际参数和预期产物，使用户确认的是完整可执行计划。
    for call in payload.get("calls", []):
        print(
            f"\nStep {call['step_index']}: {call['tool_name']} "
            f"[{call['side']}]"
        )
        print(f"Reason: {call['call_reason']}")
        print("Arguments:")
        print(json.dumps(call["arguments"], ensure_ascii=False, indent=2))
        print("Expected outputs:")
        for artifact in call["output_artifacts"]:
            requirement = "required" if artifact["required"] else "optional"
            print(
                f"- {artifact['artifact_type']} ({requirement}): "
                f"{artifact['path_pattern']}"
            )


def read_tool_plan_confirmation(payload: dict[str, Any]) -> dict[str, Any]:
    """读取一次合法的终端确认，并生成 interrupt 恢复数据。"""

    while True:
        choice = input(
            "\nChoose [a]pprove, [m]odify, or [r]eject: "
        ).strip().casefold()

        if choice in {"a", "approve", "approved"}:
            decision = "approved"
            revision_request = None
            break
        if choice in {"m", "modify", "revision", "revision_requested"}:
            revision_request = input("Enter the plan revision request: ").strip()
            if not revision_request:
                print("[ERROR] A revision request cannot be empty.")
                continue
            decision = "revision_requested"
            break
        if choice in {"r", "reject", "rejected"}:
            decision = "rejected"
            revision_request = None
            break

        print("[ERROR] Enter a, m, or r.")

    # turn_id 和 plan_id 原样取自中断 payload，节点恢复时会再次与状态核对。
    return {
        "turn_id": payload["turn_id"],
        "plan_id": payload["plan_id"],
        "decision": decision,
        "revision_request": revision_request,
    }


def display_user_input_request(payload: dict[str, Any]) -> None:
    """展示上一轮回答和下一轮编号。"""

    _display_message_envelope(
        response_mode=payload.get("response_mode"),
        analysis_status=payload.get("analysis_status"),
        analysis_route=payload.get("analysis_route", "answer_generated"),
        termination_reason=payload.get("termination_reason"),
        final_answer=str(payload.get("final_answer", "")),
        awaiting_input=True,
    )
    print(
        f"\n[Turn {payload['next_turn_id']}/{payload['max_turns']}] "
        "Enter the next message."
    )


def read_next_user_input(payload: dict[str, Any]) -> dict[str, Any]:
    """读取非空下一轮消息；完整自然语言请求统一交给图内 Router 处理。"""

    while True:
        message = input("> ").strip()
        if message:
            return {
                "turn_id": payload["next_turn_id"],
                "message": message,
            }
        print("[ERROR] The next message cannot be empty.")


def display_final_analysis_result(result: dict[str, Any]) -> None:
    """图直接结束时展示最后一轮 Analysis 回答。"""

    analysis = result.get("current_analysis_result")
    if analysis is None:
        return
    final_answer = getattr(analysis, "final_answer", None)
    if final_answer:
        _display_message_envelope(
            response_mode=getattr(analysis, "response_mode", None),
            analysis_status=getattr(analysis, "analysis_status", None),
            analysis_route=getattr(analysis, "analysis_route", None),
            termination_reason=getattr(analysis, "termination_reason", None),
            final_answer=final_answer,
            awaiting_input=False,
        )
