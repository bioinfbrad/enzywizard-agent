from __future__ import annotations

from ..services.request_routing_service import route_user_request
from ..state import AgentState


def request_routing_node(state: AgentState) -> AgentState:
    """使用结构化 LLM 输出决定本轮进入 Planning、历史解析或 Analysis。"""

    # 同一轮计划修改会再次进入 Router；先清除上一次路由留下的临时历史选择，
    # 防止新修改已取消历史复用时 Planning 仍读取旧产物。
    state.pop("current_history_query", None)
    state["current_history_artifacts"] = []
    result = route_user_request(state)
    state["current_request_routing"] = result
    state.setdefault("request_routing_results", []).append(result)
    state["request_routing_summary"] = "\n".join(
        f"Turn {item.turn_id}: {item.summary}"
        for item in state["request_routing_results"][-12:]
    )
    state["route"] = result.route
    return state
