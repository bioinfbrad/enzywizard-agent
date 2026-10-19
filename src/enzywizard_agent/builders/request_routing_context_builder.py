from __future__ import annotations

import json

from ..state import AgentState


def build_prior_plan_catalog(state: AgentState) -> list[dict[str, object]]:
    """压缩已批准或已执行计划，只保留工具、侧别和非路径标量参数。"""

    approved_keys = {
        (item.turn_id, item.plan_id)
        for item in state.get("user_confirmations", [])
        if item.decision == "approved"
    }
    approved_keys.update(
        (item.turn_id, item.plan_id)
        for item in state.get("tool_executions", [])
    )
    catalog: list[dict[str, object]] = []
    for result in state.get("planning_results", []):
        plan = result.tool_plan
        if plan is None or (plan.turn_id, plan.plan_id) not in approved_keys:
            continue
        calls = []
        for call in plan.calls:
            parameters = {
                name: value
                for name, value in call.arguments.items()
                if isinstance(value, (str, int, float, bool))
                and not name.endswith(("_path", "_dir", "_msa", "_fasta"))
                and name not in {"input_path", "output_dir", "input_dir"}
            }
            calls.append(
                {
                    "tool_name": call.tool_name,
                    "side": call.side,
                    "parameters": parameters,
                }
            )
        catalog.append(
            {
                "turn_id": plan.turn_id,
                "plan_id": plan.plan_id,
                "purpose_type": plan.purpose_type,
                "calls": calls,
            }
        )
    return catalog[-8:]


def _active_pending_plan(state: AgentState) -> dict[str, object] | None:
    """只暴露正在修改的当前计划，避免把已完成计划误当作代词指代对象。"""

    planning = state.get("current_planning_result")
    confirmation = state.get("current_user_confirmation")
    if (
        planning is None
        or planning.tool_plan is None
        or confirmation is None
        or confirmation.decision != "revision_requested"
        or confirmation.turn_id != planning.turn_id
        or confirmation.plan_id != planning.plan_id
    ):
        return None
    return {
        "turn_id": planning.turn_id,
        "plan_id": planning.plan_id,
        "purpose_type": planning.tool_plan.purpose_type,
        "call_sequence": [
            f"{call.tool_name}:{call.side}" for call in planning.tool_plan.calls
        ],
    }


def build_request_routing_context(state: AgentState) -> str:
    """构造首层 Router 的精简会话上下文。"""

    confirmation = state.get("current_user_confirmation")
    agent_inputs = state.get("agent_inputs")
    current_turn_id = state.get("current_turn_id", 1)
    payload = {
        "current_turn_id": current_turn_id,
        "current_user_request": state.get("current_user_message", ""),
        "current_plan_revision": (
            confirmation.revision_request
            if confirmation is not None and confirmation.revision_request
            else None
        ),
        "current_input_availability": (
            {
                "scope": "current supplied inputs; not historical artifacts",
                "structure": bool(agent_inputs.structure_file),
                "msa": bool(agent_inputs.msa_file),
                "mut_structure": bool(agent_inputs.mut_structure_file),
                "mut_msa": bool(agent_inputs.mut_msa_file),
                "substrate": bool(agent_inputs.substrate_name),
                "amino_acid_substitution": bool(
                    agent_inputs.amino_acid_substitution
                ),
            }
            if agent_inputs is not None
            else {}
        ),
        "active_pending_plan": _active_pending_plan(state),
        "recent_user_requests": [
            item.model_dump(mode="json")
            for item in state.get("user_messages", [])[-8:]
        ],
        "completed_turn_ids": sorted(
            {
                artifact.turn_id
                for artifact in state.get("artifacts", [])
                if artifact.turn_id < current_turn_id
            }
        ),
        "prior_approved_plans": build_prior_plan_catalog(state),
    }
    return json.dumps(payload, indent=2)
