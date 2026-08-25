from __future__ import annotations

from ..data_models.planning import PlanningResult
from ..services.planning_service import (
    classify_intent,
    select_single_protein_tools,
    select_wildtype_mutant_tools,
)
from ..state import AgentState
from ..utils.planning_utils import purpose_needs_tool


def planning_node(state: AgentState) -> AgentState:
    planning_result = PlanningResult(
        turn_id=state.get("current_turn_id", 1),
        planning_route="tool_unused",
        planning_flag="started",
        needs_tool=False,
        summary=None,
        purpose_type=None,
        purpose_reason="",
        tool_selection=None,
        tool_plan=None,
        error_messages=[],
        knowledge_queries=[],
    )

    try:
        intent = classify_intent(state)
    except Exception as e:
        error_message = f"Intent recognition failed: {e}"
        planning_result.planning_flag = "planning_failed"
        planning_result.error_messages.append(error_message)
        state["current_planning_result"] = planning_result
        state.setdefault("planning_results", []).append(planning_result)
        state["route"] = planning_result.planning_route
        print(f"[ERROR] {error_message}")
        return state

    planning_result.purpose_type = intent.purpose_type
    planning_result.purpose_reason = intent.purpose_reason
    planning_result.needs_tool = purpose_needs_tool(intent.purpose_type)
    planning_result.planning_flag = "intent_recognized"

    print(f"[INFO] Planning intent classified: purpose_type={intent.purpose_type}")
    print(f"[INFO] Planning intent reason: {intent.purpose_reason}")
    print(f"[INFO] Planning needs tool: {planning_result.needs_tool}")

    if planning_result.purpose_type == "single_protein_analysis":
        tool_selection = select_single_protein_tools(state, planning_result)
        planning_result.tool_selection = tool_selection
        planning_result.planning_flag = tool_selection.tool_selection_status
        print(
            "[INFO] Tool selection result: "
            f"status={tool_selection.tool_selection_status}, "
            f"selected_tools={tool_selection.selected_tools}"
        )
        if tool_selection.selection_reason:
            print(f"[INFO] Tool selection reason: {tool_selection.selection_reason}")
        if tool_selection.error_message:
            planning_result.error_messages.append(tool_selection.error_message)
            print(f"[INFO] Tool selection error: {tool_selection.error_message}")

    elif planning_result.purpose_type == "wildtype_mutant_comparison":
        tool_selection = select_wildtype_mutant_tools(state, planning_result)
        planning_result.tool_selection = tool_selection
        planning_result.planning_flag = tool_selection.tool_selection_status
        print(
            "[INFO] Tool selection result: "
            f"status={tool_selection.tool_selection_status}, "
            f"selected_tools={tool_selection.selected_tools}"
        )
        if tool_selection.selection_reason:
            print(f"[INFO] Tool selection reason: {tool_selection.selection_reason}")
        if tool_selection.error_message:
            planning_result.error_messages.append(tool_selection.error_message)
            print(f"[INFO] Tool selection error: {tool_selection.error_message}")

    else:
        planning_result.planning_flag = "tool_unused"

    if planning_result.planning_flag in ("tool_selection_failed", "tool_unused"):
        state["current_planning_result"] = planning_result
        state.setdefault("planning_results", []).append(planning_result)
        state["route"] = planning_result.planning_route

    return state
