from __future__ import annotations

import json
from typing import Any

from ..data_models.planning import PlanningResult, ToolPlan
from ..state import AgentState
from ..utils.planning_utils import load_enzywizard_tool_specs
from ..utils.tool_plan_utils import (
    PARAMETER_PROFILE_VALUES,
    PARAMETER_VALUE_CONSTRAINTS,
    PROGRAM_MANAGED_OPTIONAL_PARAMETERS,
)
from .request_routing_context_builder import build_prior_plan_catalog


def _append_routing_context(sections: list[str], state: AgentState) -> None:
    """向 Planning 提供上游语义结论，避免其重新解释历史动作。"""

    routing = state.get("current_request_routing")
    if routing is not None and routing.output is not None:
        sections.append(
            "Request Router output:\n"
            + json.dumps(
                routing.output.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
            )
        )
    prior_plans = build_prior_plan_catalog(state)
    if prior_plans:
        sections.append(
            "Prior approved plan catalog:\n"
            + json.dumps(prior_plans, ensure_ascii=False, indent=2)
        )
    history = state.get("current_history_query")
    if history is not None and history.query is not None:
        sections.append(
            "Resolved history request:\n"
            + json.dumps(
                {
                    "query": history.query.model_dump(mode="json"),
                    "resolved_turn_ids": history.resolved_turn_ids,
                    "selected_artifact_types": sorted(
                        {item.artifact_type for item in history.selected_artifacts}
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )


def build_planning_context(state: AgentState) -> str:
    """构造意图识别所需的精简文本上下文。

    此阶段只提供当前请求、用户修改、历史摘要和非空输入，不加入工具规格，避免
    意图分类模型提前进行工具规划。
    """

    # 每个非空信息块先独立构造，最后使用空行拼接，便于模型区分语义区域。
    sections: list[str] = []

    # 当前用户请求是意图判断的主要依据。
    current_user_message = state.get("current_user_message")
    if current_user_message:
        sections.append(f"Current user request:\n{current_user_message.strip()}")

    _append_routing_context(sections, state)

    # 用户在确认中提出的修改意见代表最新要求，Prompt 会赋予它更高优先级。
    user_confirmation = state.get("current_user_confirmation")
    if user_confirmation is not None and user_confirmation.revision_request:
        sections.append(
            f"User revision request:\n{user_confirmation.revision_request.strip()}"
        )

    # 历史摘要提供多轮背景，但不重复传入全部历史消息。
    user_message_summary = state.get("user_message_summary")
    if user_message_summary:
        sections.append(f"Previous user request summary:\n{user_message_summary.strip()}")

    # 上轮回答摘要让“继续解释/再比较一下”等省略主语的请求仍有语义背景；它只
    # 是背景，当前消息和当前计划修改意见始终具有更高优先级。
    analysis_summary = state.get("analysis_summary")
    if analysis_summary:
        sections.append(f"Previous analysis summary:\n{analysis_summary.strip()}")

    # 只列出实际提供的输入字段；None 字段不会占用 Prompt 长度。
    agent_inputs = state.get("agent_inputs")
    if agent_inputs is not None:
        input_lines = []
        for field_name, value in agent_inputs.model_dump().items():
            if value is not None:
                input_lines.append(f"- {field_name}: {value}")
        if input_lines:
            input_context = "\n".join(input_lines)
            sections.append(f"Provided agent inputs:\n{input_context}")

    # 空行分隔的信息块会作为一条 HumanMessage 发送给意图识别模型。
    return "\n\n".join(sections)


def build_workflow_requirements_context(
    state: AgentState,
    planning_result: PlanningResult,
    preparation_tools: list[str],
) -> str:
    """构造科学需求提取上下文，不向模型暴露工具名选择任务。"""

    agent_inputs = state.get("agent_inputs")
    previous_requirements = next(
        (
            result.workflow_requirements
            for result in reversed(state.get("planning_results", []))
            if result.turn_id == planning_result.turn_id
            and result.workflow_requirements is not None
        ),
        None,
    )
    confirmation = state.get("current_user_confirmation")
    routing = state.get("current_request_routing")
    payload = {
        "purpose_type": planning_result.purpose_type,
        "current_user_request": state.get("current_user_message", ""),
        "latest_revision_request": (
            confirmation.revision_request
            if confirmation is not None and confirmation.revision_request
            else None
        ),
        "previous_proposed_requirements": (
            previous_requirements.model_dump(mode="json")
            if previous_requirements is not None
            else None
        ),
        "request_router_output": (
            routing.output.model_dump(mode="json")
            if routing is not None and routing.output is not None
            else None
        ),
        "prior_approved_plans": build_prior_plan_catalog(state),
        "program_selected_preparation_tools": preparation_tools,
        "provided_inputs": (
            {
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
        "allowed_analysis_types": [
            "aaprops",
            "hydrocluster",
            "energy",
            "flexibility",
            "disorder",
            "conservation",
            "embedding",
            "pocket",
            "substrate",
            "docking",
            "interaction",
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_parameter_extraction_context(
    state: AgentState,
    planning_result: PlanningResult,
    tool_plan: ToolPlan,
) -> str:
    """构造参数提取步骤使用的精简调用与参数上下文。

    此处只提供已经选定的调用，不再传入17个工具的完整规格。每个调用把可编辑
    参数和程序管理参数分开，避免模型把路径、输入文件或上游产物当作普通参数。
    """

    sections: list[str] = []

    # 最新修改意见具有最高优先级；没有修改意见时使用当前轮原始请求。
    user_confirmation = state.get("current_user_confirmation")
    if user_confirmation is not None and user_confirmation.revision_request:
        sections.append(
            f"User revision request:\n{user_confirmation.revision_request.strip()}"
        )

    current_user_message = state.get("current_user_message")
    if current_user_message:
        sections.append(f"Current user request:\n{current_user_message.strip()}")

    _append_routing_context(sections, state)

    # 历史摘要只能辅助消解“仍按之前参数”等引用，不允许它引入跨轮文件路径。
    user_message_summary = state.get("user_message_summary")
    if user_message_summary:
        sections.append(f"Previous user request summary:\n{user_message_summary.strip()}")

    tool_specs = load_enzywizard_tool_specs()
    planned_call_contexts: list[dict[str, Any]] = []
    for call in tool_plan.calls:
        tool_spec = tool_specs[call.tool_name]
        serialized_arguments = call.model_dump(mode="json")["arguments"]
        required_parameter_names = {
            argument["name"] for argument in tool_spec.get("required_arguments", [])
        }
        optional_specs = {
            argument["name"]: argument
            for argument in tool_spec.get("optional_arguments", [])
        }

        # 所有 required 参数和特定可选输入参数均由程序管理。参数值可能是上游
        # ToolOutputReference，model_dump 会把引用稳定序列化为 JSON 对象。
        program_managed_parameter_names = required_parameter_names | (
            set(optional_specs) & PROGRAM_MANAGED_OPTIONAL_PARAMETERS
        )
        program_managed_parameters = {
            parameter_name: serialized_arguments.get(parameter_name)
            for parameter_name in sorted(program_managed_parameter_names)
            if parameter_name in serialized_arguments
        }

        editable_parameters = []
        for parameter_name, parameter_spec in optional_specs.items():
            if parameter_name in PROGRAM_MANAGED_OPTIONAL_PARAMETERS:
                continue
            editable_parameters.append(
                {
                    "parameter_name": parameter_name,
                    "python_type": parameter_spec["python_type"],
                    "current_value": serialized_arguments.get(parameter_name),
                    "allowed_values": parameter_spec.get("allowed_values"),
                    # 精确范围来自工具 service 的现有校验，并由最终程序校验复核。
                    "validation_rule": PARAMETER_VALUE_CONSTRAINTS.get(
                        parameter_name,
                        "Use the value type and parameter description.",
                    ),
                    "description": parameter_spec["description"],
                }
            )

        supported_profiles = sorted(
            PARAMETER_PROFILE_VALUES.get(call.tool_name, {})
        )
        planned_call_contexts.append(
            {
                "step_index": call.step_index,
                "tool_name": call.tool_name,
                "side": call.side,
                "program_managed_parameters": program_managed_parameters,
                "editable_parameters": editable_parameters,
                "supported_profiles": supported_profiles,
            }
        )

    # default=str 只处理 Path 等展示类型；Pydantic 子模型已经由 model_dump 转换。
    serialized_calls = json.dumps(
        planned_call_contexts,
        ensure_ascii=False,
        indent=2,
        default=str,
    )
    sections.append(f"Planned calls and parameter policy:\n{serialized_calls}")

    planning_context = {
        "turn_id": planning_result.turn_id,
        "plan_id": planning_result.plan_id,
        "purpose_type": planning_result.purpose_type,
    }
    sections.append(
        "Current planning identifiers:\n"
        + json.dumps(planning_context, ensure_ascii=False, indent=2)
    )

    return "\n\n".join(sections)
