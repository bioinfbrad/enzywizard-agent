from __future__ import annotations

import json
from typing import Any


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

    print("\n=== EnzyWizard-Agent ===")
    print(payload.get("final_answer", ""))
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
        print("\n=== EnzyWizard-Agent ===")
        print(final_answer)
