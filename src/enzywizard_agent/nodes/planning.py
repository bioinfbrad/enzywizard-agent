from __future__ import annotations

from ..data_models.planning import PlanningResult
from ..services.planning_service import (
    classify_intent,
    extract_tool_parameters,
    finalize_tool_plan,
    generate_default_tool_plan,
    select_single_protein_tools,
    select_wildtype_mutant_tools,
)
from ..state import AgentState
from ..utils.planning_utils import purpose_needs_tool


def planning_node(state: AgentState) -> AgentState:
    """执行从意图识别到最终 ToolPlan 的完整 Planning 节点。"""

    # 每次进入 Planning 都表示一次新的计划尝试；同一轮用户要求修改计划时
    # turn_id 不变，但 plan_id 递增，确保目录和产物不会覆盖上一版计划。
    # turn_id 在这一轮完整完成后再加
    plan_id = state.get("current_plan_id", 0) + 1
    state["current_plan_id"] = plan_id

    # 先创建一个完整的中间 PlanningResult，后续步骤在同一对象上逐项填写。
    planning_result = PlanningResult(
        turn_id=state.get("current_turn_id", 1),
        plan_id=plan_id,
        planning_route="tool_unused",
        planning_flag="started",
        needs_tool=False,
        summary=None,
        purpose_type=None,
        purpose_reason="",
        tool_selection=None,
        parameter_extraction=None,
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

    # 工具选择成功后，程序先根据固定依赖、路径和默认参数生成一份默认计划。
    try:
        default_tool_plan = generate_default_tool_plan(state, planning_result)
        planning_result.planning_flag = "default_tool_plan_generated"
        print(
            "[INFO] Default tool plan generated: "
            f"calls={len(default_tool_plan.calls)}"
        )
    except Exception as e:
        error_message = f"Default tool plan generation failed: {e}"
        planning_result.planning_flag = "planning_failed"
        planning_result.error_messages.append(error_message)
        state["current_planning_result"] = planning_result
        state.setdefault("planning_results", []).append(planning_result)
        state["route"] = planning_result.planning_route
        print(f"[ERROR] {error_message}")
        return state

    # 参数提取只读取用户明确要求，不允许模型重选工具、调整顺序或生成路径。
    try:
        parameter_extraction = extract_tool_parameters(
            state,
            planning_result,
            default_tool_plan,
        )
        planning_result.parameter_extraction = parameter_extraction
        planning_result.planning_flag = parameter_extraction.parameter_extraction_status
        print(
            "[INFO] Parameter extraction result: "
            f"status={parameter_extraction.parameter_extraction_status}, "
            f"adjustments={len(parameter_extraction.parameter_adjustments)}, "
            f"profiles={len(parameter_extraction.profile_requests)}, "
            f"unsupported={len(parameter_extraction.unsupported_requests)}"
        )
    except Exception as e:
        error_message = f"Parameter extraction failed: {e}"
        planning_result.planning_flag = "parameter_extraction_failed"
        planning_result.error_messages.append(error_message)
        state["current_planning_result"] = planning_result
        state.setdefault("planning_results", []).append(planning_result)
        state["route"] = planning_result.planning_route
        print(f"[ERROR] {error_message}")
        return state

    # 模型主动报告提取失败时不继续应用参数，保存其原始错误交给 Analysis。
    if parameter_extraction.parameter_extraction_status == "parameter_extraction_failed":
        error_message = (
            parameter_extraction.error_message
            or "Parameter extraction failed without an error message."
        )
        planning_result.error_messages.append(error_message)
        state["current_planning_result"] = planning_result
        state.setdefault("planning_results", []).append(planning_result)
        state["route"] = planning_result.planning_route
        print(f"[INFO] Parameter extraction error: {error_message}")
        return state

    # 程序在最后一步应用 profile 和明确参数，拒绝路径覆盖，并重新校验所有调用。
    try:
        final_tool_plan = finalize_tool_plan(
            state,
            default_tool_plan,
            parameter_extraction,
        )
    except Exception as e:
        error_message = f"Tool plan parameter validation failed: {e}"
        planning_result.planning_flag = "planning_failed"
        planning_result.error_messages.append(error_message)
        state["current_planning_result"] = planning_result
        state.setdefault("planning_results", []).append(planning_result)
        state["route"] = planning_result.planning_route
        print(f"[ERROR] {error_message}")
        return state

    # 完整计划成功后才写入状态并切换到用户确认路由。Planning 阶段只保存路径，
    # 不创建 ToolCall 目录，也不复制 integrate 输入报告。
    planning_result.tool_plan = final_tool_plan
    planning_result.summary = final_tool_plan.summary
    planning_result.planning_flag = "tool_plan_generated"
    planning_result.planning_route = "tool_plan_generated"
    state["current_planning_result"] = planning_result
    state.setdefault("planning_results", []).append(planning_result)
    state["route"] = planning_result.planning_route
    print(
        "[INFO] Final tool plan generated: "
        f"plan_id={planning_result.plan_id}, calls={len(final_tool_plan.calls)}"
    )
    return state
