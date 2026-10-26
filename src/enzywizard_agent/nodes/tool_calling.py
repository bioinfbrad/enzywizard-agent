from __future__ import annotations

from ..services.tool_calling_service import execute_approved_tool_plan
from ..services.manifest_service import write_session_manifest
from ..state import AgentState
from ..utils.tool_execution_utils import (
    artifacts_for_global_tracking,
    update_cleaned_files_from_executions,
)


def tool_calling_node(state: AgentState) -> AgentState:
    """执行当前已批准 ToolPlan，并把真实结果写回 AgentState。

    节点不决定后续路由。Graph 已将 Tool Calling 的成功和失败统一连接到
    Analysis，由 Analysis 根据 current_tool_execution 生成用户可读回答。
    """

    tool_execution = execute_approved_tool_plan(state)

    # 当前计划产物包括成功步骤、失败步骤的部分产物以及完整 stdout/stderr。
    current_plan_artifacts = [
        artifact
        for call_execution in tool_execution.call_executions
        for artifact in call_execution.artifacts
    ]
    state["current_tool_execution"] = tool_execution
    state.setdefault("tool_executions", []).append(tool_execution)
    state["current_plan_artifacts"] = current_plan_artifacts
    known_paths = {artifact.path for artifact in state.setdefault("artifacts", [])}
    state["artifacts"].extend(
        artifact
        for call_execution in tool_execution.call_executions
        for artifact in artifacts_for_global_tracking(call_execution)
        if artifact.path not in known_paths
    )

    # 只有成功的 clean/mut-clean 调用会更新跨轮可复用的 CleanedFiles。
    state["cleaned_files"] = update_cleaned_files_from_executions(
        state["cleaned_files"],
        tool_execution.call_executions,
    )

    if tool_execution.execution_status == "succeeded":
        print(f"[INFO] {tool_execution.summary}")
    else:
        print(f"[ERROR] {tool_execution.summary}")
    if state.get("config") is not None:
        write_session_manifest(state)
    return state
