from __future__ import annotations

from langgraph.graph import END, StateGraph

from .nodes.analysis import analysis_node
from .nodes.planning import planning_node
from .nodes.tool_calling import tool_calling_node
from .nodes.user_confirmation_interrupt import user_confirmation_interrupt_node
from .nodes.user_input_interrupt import user_input_interrupt_node
from .state import AgentState


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("planning", planning_node)
    graph.add_node("user_confirmation_interrupt", user_confirmation_interrupt_node)
    graph.add_node("tool_calling", tool_calling_node)
    graph.add_node("analysis", analysis_node)
    graph.add_node("user_input_interrupt", user_input_interrupt_node)

    graph.set_entry_point("planning")

    graph.add_conditional_edges(
        "planning",
        route_from_state,
        {
            "tool_plan_generated": "user_confirmation_interrupt",
            "tool_unused": "analysis",
        },
    )
    graph.add_conditional_edges(
        "user_confirmation_interrupt",
        route_from_state,
        {
            "tool_execution_approved": "tool_calling",
            "tool_plan_revision_requested": "planning",
            "tool_execution_rejected": "user_input_interrupt",
        },
    )
    graph.add_edge("tool_calling", "analysis")
    graph.add_conditional_edges(
        "analysis",
        route_from_state,
        {
            "answer_generated": "user_input_interrupt",
            "exit_requested": END,
        },
    )
    graph.add_edge("user_input_interrupt", "planning")

    return graph.compile()


def route_from_state(state: AgentState) -> str:
    route = state.get("route")
    if route is None or not route.strip():
        raise ValueError("Missing route in agent state.")
    return route
