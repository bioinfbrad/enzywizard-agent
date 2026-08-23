from __future__ import annotations

from ..state import AgentState

#####这里代码怪怪的，逻辑不统一
def build_planning_context(state: AgentState) -> str:
    sections: list[str] = []

    current_user_message = state.get("current_user_message")
    if current_user_message:
        sections.append(f"Current user request:\n{current_user_message.strip()}")

    agent_inputs = state.get("agent_inputs")
    if agent_inputs is not None:
        input_lines = []
        for field_name, value in agent_inputs.model_dump().items():
            if value is not None:
                input_lines.append(f"- {field_name}: {value}")
        if input_lines:
            input_context = "\n".join(input_lines)
            sections.append(f"Provided agent inputs:\n{input_context}")

    user_message_summary = state.get("user_message_summary")
    if user_message_summary:
        sections.append(f"Previous user request summary:\n{user_message_summary.strip()}")

    user_confirmation = state.get("current_user_confirmation")
    if user_confirmation is not None and user_confirmation.revision_request:
        sections.append(f"User revision request:\n{user_confirmation.revision_request.strip()}")

    return "\n\n".join(sections)
