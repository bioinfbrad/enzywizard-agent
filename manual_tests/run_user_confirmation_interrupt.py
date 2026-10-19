"""不调用 LLM，手工验证用户确认中断及其三个恢复分支。"""

from __future__ import annotations

import json
import time
from uuid import uuid4

from langgraph.graph import END, StateGraph
from pydantic import ValidationError

from enzywizard_agent.data_models.planning import (
    PlanningResult,
    ToolCall,
    ToolPlan,
)
from enzywizard_agent.data_models.runtime import (
    AgentErrorRecord,
    LLMCallRecord,
    ModelCapabilityResult,
    RuntimeSession,
)
from enzywizard_agent.nodes.user_confirmation_interrupt import (
    user_confirmation_interrupt_node,
)
from enzywizard_agent.services.agent_service import (
    execute_agent_graph,
    resume_agent_graph,
)
from enzywizard_agent.services.user_confirmation_service import (
    record_user_confirmation,
)
from enzywizard_agent.state import AgentState
from enzywizard_agent.utils.checkpoint_utils import create_in_memory_checkpointer


def build_test_state() -> AgentState:
    """构造已完成 Planning、正在等待确认的最小状态。"""

    tool_call = ToolCall(
        turn_id=1,
        plan_id=1,
        step_index=1,
        tool_name="enzywizard-pocket",
        command_name="enzywizard-pocket",
        side="single",
        tool_execution_dir="/tmp/enzywizard-agent-confirmation/01_pocket",
        arguments={
            "input_path": "/tmp/input.cif",
            "output_dir": "/tmp/enzywizard-agent-confirmation/01_pocket",
        },
        output_artifacts=[],
        staging_rules=[],
        call_reason="Identify binding pockets in the prepared structure.",
    )
    tool_plan = ToolPlan(
        turn_id=1,
        plan_id=1,
        purpose_type="single_protein_analysis",
        summary="Run binding-pocket analysis.",
        plan_reason="The user requested binding-pocket analysis.",
        calls=[tool_call],
    )
    planning_result = PlanningResult(
        turn_id=1,
        plan_id=1,
        planning_route="tool_plan_generated",
        planning_flag="tool_plan_generated",
        needs_tool=True,
        summary=tool_plan.summary,
        purpose_type="single_protein_analysis",
        purpose_reason="The request requires single-protein analysis.",
        tool_selection=None,
        parameter_extraction=None,
        tool_plan=tool_plan,
        error_messages=[],
    )
    return {
        "runtime_session": RuntimeSession(
            started_at="2026-10-16T00:00:00+00:00",
            started_monotonic=time.monotonic(),
            llm_calls=[
                LLMCallRecord(
                    stage="confirmation.contract",
                    started_at="2026-10-16T00:00:00+00:00",
                    duration_seconds=0.1,
                    status="succeeded",
                    input_tokens=10,
                    output_tokens=2,
                    total_tokens=12,
                    reasoning_tokens=0,
                    usage_quality="exact",
                )
            ],
            errors=[
                AgentErrorRecord(
                    timestamp="2026-10-16T00:00:00+00:00",
                    stage="confirmation.contract",
                    category="model",
                    message="Recoverable contract error.",
                    recoverable=True,
                )
            ],
            model_capability=ModelCapabilityResult(
                status="passed",
                chat_completions=True,
                system_messages=True,
                strict_structured_output=True,
                token_usage_available=True,
                checked_at="2026-10-16T00:00:00+00:00",
                detail="Contract check passed.",
            ),
        ),
        "current_turn_id": 1,
        "current_plan_id": 1,
        "current_planning_result": planning_result,
        "user_confirmations": [],
        "route": "tool_plan_generated",
    }


def build_confirmation_graph():
    """构造只包含确认节点的测试图，隔离其他正式流程节点。"""

    graph = StateGraph(AgentState)
    graph.add_node("user_confirmation_interrupt", user_confirmation_interrupt_node)
    graph.set_entry_point("user_confirmation_interrupt")
    graph.add_edge("user_confirmation_interrupt", END)

    return graph.compile(checkpointer=create_in_memory_checkpointer())


def run_confirmation_case(
    decision: str,
    revision_request: str | None,
    expected_route: str,
) -> dict[str, object]:
    """执行一次中断和恢复，并断言最终确认记录与路由。"""

    graph = build_confirmation_graph()
    config = {"configurable": {"thread_id": str(uuid4())}}

    test_state = build_test_state()
    test_state["session_id"] = config["configurable"]["thread_id"]
    interrupted = execute_agent_graph(graph, test_state)
    if "__interrupt__" not in interrupted:
        raise AssertionError("Graph did not pause at user confirmation.")
    interrupted_runtime = interrupted.get("runtime_session")
    if not isinstance(interrupted_runtime, RuntimeSession):
        raise AssertionError("RuntimeSession did not survive interrupt checkpointing.")
    if not isinstance(interrupted_runtime.llm_calls[0], LLMCallRecord):
        raise AssertionError("Nested LLMCallRecord was not restored safely.")
    if not isinstance(interrupted_runtime.errors[0], AgentErrorRecord):
        raise AssertionError("Nested AgentErrorRecord was not restored safely.")
    if not isinstance(
        interrupted_runtime.model_capability,
        ModelCapabilityResult,
    ):
        raise AssertionError("Nested ModelCapabilityResult was not restored safely.")

    resumed = resume_agent_graph(
        graph,
        config["configurable"]["thread_id"],
        {
            "turn_id": 1,
            "plan_id": 1,
            "decision": decision,
            "revision_request": revision_request,
        },
    )
    if resumed["route"] != expected_route:
        raise AssertionError(
            f"Expected route={expected_route!r}, got {resumed['route']!r}."
        )
    if len(resumed["user_confirmations"]) != 1:
        raise AssertionError("Confirmation history must contain exactly one record.")
    resumed_runtime = resumed.get("runtime_session")
    if not isinstance(resumed_runtime, RuntimeSession):
        raise AssertionError("RuntimeSession did not survive graph resume.")
    if resumed_runtime.input_tokens != 10 or resumed_runtime.output_tokens != 2:
        raise AssertionError("Runtime token accounting changed during graph resume.")

    return {
        "decision": decision,
        "route": resumed["route"],
        "confirmation": resumed["current_user_confirmation"].model_dump(mode="json"),
    }


def validate_invalid_confirmation_inputs() -> None:
    """确认过期计划和缺失修改意见的恢复输入都会被拒绝。"""

    stale_plan_state = build_test_state()
    try:
        record_user_confirmation(
            stale_plan_state,
            {
                "turn_id": 1,
                "plan_id": 2,
                "decision": "approved",
                "revision_request": None,
            },
        )
    except ValueError:
        pass
    else:
        raise AssertionError("A stale plan_id must not be accepted.")

    missing_revision_state = build_test_state()
    try:
        record_user_confirmation(
            missing_revision_state,
            {
                "turn_id": 1,
                "plan_id": 1,
                "decision": "revision_requested",
                "revision_request": None,
            },
        )
    except ValidationError:
        pass
    else:
        raise AssertionError(
            "revision_requested without revision_request must not be accepted."
        )


def validate_invalid_graph_resume() -> None:
    """确认不存在或未处于中断状态的会话不能接收恢复数据。"""

    graph = build_confirmation_graph()
    try:
        resume_agent_graph(
            graph,
            str(uuid4()),
            {
                "turn_id": 1,
                "plan_id": 1,
                "decision": "approved",
                "revision_request": None,
            },
        )
    except ValueError:
        pass
    else:
        raise AssertionError("A session without an active interrupt must not resume.")


def main() -> None:
    """验证批准、修改和拒绝三个合法恢复分支。"""

    validate_invalid_confirmation_inputs()
    validate_invalid_graph_resume()
    results = [
        run_confirmation_case(
            decision="approved",
            revision_request=None,
            expected_route="tool_execution_approved",
        ),
        run_confirmation_case(
            decision="revision_requested",
            revision_request="Remove pocket analysis and run energy analysis instead.",
            expected_route="tool_plan_revision_requested",
        ),
        run_confirmation_case(
            decision="rejected",
            revision_request=None,
            expected_route="tool_execution_rejected",
        ),
    ]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
