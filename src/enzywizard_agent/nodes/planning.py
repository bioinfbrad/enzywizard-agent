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
    """执行 Planning 节点当前已实现的意图识别和工具选择步骤。

    当前尚未生成 ToolPlan。只有无需工具或工具选择失败的分支会在这里完成状态
    回填；工具选择成功后将由后续开发继续补充排序、参数和计划。
    """

    # 先创建一个完整的中间 PlanningResult，后续步骤在同一对象上逐项填写。
    planning_result = PlanningResult(
        turn_id=state.get("current_turn_id", 1),
        plan_id=state.get("current_plan_id", 1),
        planning_route="tool_unused",
        planning_flag="started",
        needs_tool=False,
        summary=None,
        purpose_type=None,
        purpose_reason="",
        tool_selection=None,
        tool_plan=None,
        error_messages=[],
    )

    # 意图识别涉及外部模型调用，任何异常都转换为 planning_failed。
    try:
        intent = classify_intent(state)
    except Exception as e:
        error_message = f"Intent recognition failed: {e}"

        # 记录失败状态和具体错误，供未来 Analysis 节点解释。
        planning_result.planning_flag = "planning_failed"
        planning_result.error_messages.append(error_message)

        # 识别失败已经结束本次 Planning，因此写入当前结果和历史列表。
        state["current_planning_result"] = planning_result
        state.setdefault("planning_results", []).append(planning_result)

        # 默认 planning_route 为 tool_unused，使失败结果后续进入 Analysis。
        state["route"] = planning_result.planning_route
        print(f"[ERROR] {error_message}")
        return state

    # 意图识别成功后，把结构化输出写入正在构建的 PlanningResult。
    planning_result.purpose_type = intent.purpose_type
    planning_result.purpose_reason = intent.purpose_reason
    planning_result.needs_tool = purpose_needs_tool(intent.purpose_type)
    planning_result.planning_flag = "intent_recognized"

    print(f"[INFO] Planning intent classified: purpose_type={intent.purpose_type}")
    print(f"[INFO] Planning intent reason: {intent.purpose_reason}")
    print(f"[INFO] Planning needs tool: {planning_result.needs_tool}")

    # 单蛋白与 WT/MUT 使用不同 Prompt 和候选规格，但返回统一 ToolSelectionOutput。
    if planning_result.purpose_type == "single_protein_analysis":
        tool_selection = select_single_protein_tools(state, planning_result)
        planning_result.tool_selection = tool_selection
        planning_result.planning_flag = tool_selection.tool_selection_status

        # 日志同时展示最终合并后的工具列表和选择原因。
        print(
            "[INFO] Tool selection result: "
            f"status={tool_selection.tool_selection_status}, "
            f"selected_tools={tool_selection.selected_tools}"
        )
        if tool_selection.selection_reason:
            print(f"[INFO] Tool selection reason: {tool_selection.selection_reason}")
        if tool_selection.error_message:
            # 工具选择失败原因同时进入 PlanningResult 的统一错误列表。
            planning_result.error_messages.append(tool_selection.error_message)
            print(f"[INFO] Tool selection error: {tool_selection.error_message}")

        # 工具选择失败表示本次 Planning 已终止；保存结果后立即交给 Analysis。
        if planning_result.planning_flag == "tool_selection_failed":
            state["current_planning_result"] = planning_result
            state.setdefault("planning_results", []).append(planning_result)
            state["route"] = planning_result.planning_route
            return state

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

        # WT/MUT 工具选择失败同样结束本次 Planning，不能继续生成 ToolPlan。
        if planning_result.planning_flag == "tool_selection_failed":
            state["current_planning_result"] = planning_result
            state.setdefault("planning_results", []).append(planning_result)
            state["route"] = planning_result.planning_route
            return state

    else:
        # purpose=other 不调用 EnzyWizard 工具，本次 Planning 在此结束。
        planning_result.planning_flag = "tool_unused"
        state["current_planning_result"] = planning_result
        state.setdefault("planning_results", []).append(planning_result)
        state["route"] = planning_result.planning_route
        return state

    # 工具选择成功后，后续逻辑将在这里继续生成完整 ToolPlan。
    return state
