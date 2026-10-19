from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

from .nodes.analysis import analysis_node
from .nodes.history_query import history_query_node
from .nodes.planning import planning_node
from .nodes.request_routing import request_routing_node
from .nodes.tool_calling import tool_calling_node
from .nodes.user_confirmation_interrupt import user_confirmation_interrupt_node
from .nodes.user_input_interrupt import user_input_interrupt_node
from .state import AgentState


def build_graph(checkpointer: BaseCheckpointSaver | None = None):

    # StateGraph 使用 AgentState 作为所有节点共享的数据结构。
    # StateGraph(AgentState)并没有真正建我们定义的图状态对象，只是让图状态机知道这个图类中有哪些状态变量，
    # 后续还要编译图，编译图就是一个可以用的agent了，运行它时传入我们初始化的图状态对象
    graph = StateGraph(AgentState)

    # 节点名称同时也是条件边映射中的目标标识。
    graph.add_node("request_routing", request_routing_node)
    graph.add_node("history_query", history_query_node)
    graph.add_node("planning", planning_node)
    graph.add_node("user_confirmation_interrupt", user_confirmation_interrupt_node)
    graph.add_node("tool_calling", tool_calling_node)
    graph.add_node("analysis", analysis_node)
    graph.add_node("user_input_interrupt", user_input_interrupt_node)

    # 每轮先由专用 Router 解析完整请求，Planning 只接收需要执行工具的请求。
    graph.set_entry_point("request_routing")

    graph.add_conditional_edges(
        "request_routing",
        route_from_state,
        {
            "request_to_planning": "planning",
            "request_to_history_query": "history_query",
            "request_to_analysis": "analysis",
        },
    )

    graph.add_conditional_edges(
        "history_query",
        route_from_state,
        {
            "history_to_planning": "planning",
            "history_to_analysis": "analysis",
        },
    )

    # Planning 有工具计划时等待用户确认；无需工具时直接交给 Analysis。
    graph.add_conditional_edges(
        "planning",
        route_from_state,
        {
            "tool_plan_generated": "user_confirmation_interrupt",
            "tool_unused": "analysis",
        },
    )

    # 用户可以批准计划、要求重新规划，或者拒绝执行。拒绝也进入 Analysis，
    # 由其生成明确回答并统一处理最后一轮的 max_turns 终止逻辑。
    graph.add_conditional_edges(
        "user_confirmation_interrupt",
        route_from_state,
        {
            "tool_execution_approved": "tool_calling",
            "tool_plan_revision_requested": "request_routing",
            "tool_execution_rejected": "analysis",
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

    # 收到下一轮用户输入后重新进入 Router，形成多轮会话循环。
    graph.add_edge("user_input_interrupt", "request_routing")

    # compile 返回可调用的 LangGraph 应用。传入 checkpointer 后，调用方必须在
    # invoke/config 中提供稳定 thread_id，才能在中断后恢复同一会话。
    return graph.compile(checkpointer=checkpointer)


def route_from_state(state: AgentState) -> str:
    """读取上游节点写入的 route，并把它交给 LangGraph 条件边匹配。"""

    route = state.get("route")

    # 缺少路由意味着上游节点没有完成其状态契约，应立即暴露程序错误。
    if route is None or not route.strip():
        raise ValueError("Missing route in agent state.")
    return route
