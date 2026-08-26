from __future__ import annotations

from langgraph.graph import END, StateGraph

from .nodes.analysis import analysis_node
from .nodes.planning import planning_node
from .nodes.tool_calling import tool_calling_node
from .nodes.user_confirmation_interrupt import user_confirmation_interrupt_node
from .nodes.user_input_interrupt import user_input_interrupt_node
from .state import AgentState


def build_graph():
    """按照既定模块流向创建并编译 EnzyWizard-Agent 状态图。"""

    # StateGraph 使用 AgentState 作为所有节点共享的数据结构。
    graph = StateGraph(AgentState)

    # 节点名称同时也是条件边映射中的目标标识。
    graph.add_node("planning", planning_node)
    graph.add_node("user_confirmation_interrupt", user_confirmation_interrupt_node)
    graph.add_node("tool_calling", tool_calling_node)
    graph.add_node("analysis", analysis_node)
    graph.add_node("user_input_interrupt", user_input_interrupt_node)

    # 每次新会话或下一轮用户输入都从 Planning 开始处理。
    graph.set_entry_point("planning")

    # Planning 有工具计划时等待用户确认；无需工具时直接交给 Analysis。
    graph.add_conditional_edges(
        "planning",
        route_from_state,
        {
            "tool_plan_generated": "user_confirmation_interrupt",
            "tool_unused": "analysis",
        },
    )

    # 用户可以批准计划、要求重新规划，或者拒绝执行并进入下一轮输入。
    graph.add_conditional_edges(
        "user_confirmation_interrupt",
        route_from_state,
        {
            "tool_execution_approved": "tool_calling",
            "tool_plan_revision_requested": "planning",
            "tool_execution_rejected": "user_input_interrupt",
        },
    )

    # 工具执行结果无论成功或失败都由 Analysis 解释并生成面向用户的回答。
    graph.add_edge("tool_calling", "analysis")

    # Analysis 通常进入下一轮输入；只有明确退出请求才结束图执行。
    graph.add_conditional_edges(
        "analysis",
        route_from_state,
        {
            "answer_generated": "user_input_interrupt",
            "exit_requested": END,
        },
    )

    # 收到下一轮用户输入后重新进入 Planning，形成多轮会话循环。
    graph.add_edge("user_input_interrupt", "planning")

    # compile 返回可调用的 LangGraph 应用；当前服务尚未正式 invoke。
    return graph.compile()


def route_from_state(state: AgentState) -> str:
    """读取上游节点写入的 route，并把它交给 LangGraph 条件边匹配。"""

    route = state.get("route")

    # 缺少路由意味着上游节点没有完成其状态契约，应立即暴露程序错误。
    if route is None or not route.strip():
        raise ValueError("Missing route in agent state.")
    return route
