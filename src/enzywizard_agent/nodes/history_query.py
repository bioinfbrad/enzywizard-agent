from __future__ import annotations

from ..services.request_routing_service import plan_history_query
from ..state import AgentState


def history_query_node(state: AgentState) -> AgentState:
    """解析历史语义选择器，并从会话 artifact registry 选择真实产物。"""

    result = plan_history_query(state)
    state["current_history_query"] = result
    state["current_history_artifacts"] = list(result.selected_artifacts)
    state.setdefault("history_query_results", []).append(result)
    state["history_query_summary"] = "\n".join(
        f"Turn {item.turn_id}: {item.summary}"
        for item in state["history_query_results"][-12:]
    )

    routing = state.get("current_request_routing")
    requires_tools = bool(
        routing is not None
        and routing.output is not None
        and routing.output.requires_tool_execution
    )
    reuse_required = bool(
        routing is not None
        and routing.output is not None
        and routing.output.reuse_policy == "required"
    )
    action_types = {
        item.action_type
        for item in (
            routing.output.actions
            if routing is not None and routing.output is not None
            else []
        )
    }
    optional_reuse_only = (
        routing is not None
        and routing.output is not None
        and routing.output.reuse_policy == "allowed"
        and not action_types.intersection({"analyze_history", "list_artifacts"})
    )
    can_continue_to_planning = requires_tools and (
        result.query_status == "succeeded"
        or (
            result.query_status == "not_found"
            and not reuse_required
            and optional_reuse_only
        )
    )
    state["route"] = (
        "history_to_planning"
        if can_continue_to_planning
        else "history_to_analysis"
    )
    return state
