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


def build_tool_selection_context(
    state: AgentState,
    planning_result: PlanningResult,
    tool_specs: dict[str, dict[str, Any]],
    preparation_tools: list[str],
) -> str:
    """构造工具选择步骤的结构化文本上下文。

    preparation_tools 已由程序确定；tool_specs 已排除准备工具和整合工具，只
    包含允许 LLM 选择的单项分析工具与 batch。该函数负责把程序结论、用户
    要求和候选工具组织成模型容易读取的分区文本。
    """

    sections: list[str] = []

    agent_inputs = state.get("agent_inputs")
    cleaned_files = state.get("cleaned_files")

    # 根据 purpose 只生成当前工作流相关的输入可用性，避免单蛋白和突变字段混杂。
    if planning_result.purpose_type == "single_protein_analysis":
        input_availability = {
            # LLM 只能看到该结论，不能自行增删准备工具。
            "program_selected_preparation_tools": preparation_tools,
            # provided_inputs 描述用户实际提供且准备步骤不能代替的外部输入。
            "provided_inputs": {
                "structure": bool(
                    agent_inputs is not None and agent_inputs.structure_file
                ),
                "msa": bool(agent_inputs is not None and agent_inputs.msa_file),
                "substrate": bool(
                    agent_inputs is not None and agent_inputs.substrate_name
                ),
            },
            # 这里描述准备工具完成后，下游分析实际可以依赖的基础文件。
            "available_analysis_inputs": {
                "cleaned_structure": bool(
                    # 当前计划包含 clean，或状态中已有可信 cleaned structure，均视为可用。
                    "enzywizard-clean" in preparation_tools
                    or (
                        cleaned_files is not None
                        and (
                            cleaned_files.single_cleaned_structure_file
                            or cleaned_files.single_cleaned_cif_file
                            or cleaned_files.single_cleaned_pdb_file
                        )
                    )
                ),
                "clean_report": bool(
                    # clean report 与结构分别检查，因为 integrate 需要该报告。
                    "enzywizard-clean" in preparation_tools
                    or (
                        cleaned_files is not None
                        and cleaned_files.single_clean_report_file
                    )
                ),
            },
        }
    else:
        # WT/MUT 工作流需要同时描述两侧输入以及标准化突变信息。
        input_availability = {
            "program_selected_preparation_tools": preparation_tools,
            "provided_inputs": {
                "structure": bool(
                    agent_inputs is not None and agent_inputs.structure_file
                ),
                "msa": bool(agent_inputs is not None and agent_inputs.msa_file),
                "mut_structure": bool(
                    agent_inputs is not None and agent_inputs.mut_structure_file
                ),
                "mut_msa": bool(
                    agent_inputs is not None and agent_inputs.mut_msa_file
                ),
                "substrate": bool(
                    agent_inputs is not None and agent_inputs.substrate_name
                ),
                "amino_acid_substitution": bool(
                    agent_inputs is not None
                    and agent_inputs.amino_acid_substitution
                ),
            },
            "available_analysis_inputs": {
                # mut-clean 被计划后，两侧 cleaned structure 均会在执行时产生。
                "wt_cleaned_structure": bool(
                    "enzywizard-mut-clean" in preparation_tools
                    or (
                        cleaned_files is not None
                        and (
                            cleaned_files.wt_cleaned_structure_file
                            or cleaned_files.wt_cleaned_cif_file
                            or cleaned_files.wt_cleaned_pdb_file
                        )
                    )
                ),
                "mut_cleaned_structure": bool(
                    "enzywizard-mut-clean" in preparation_tools
                    or (
                        cleaned_files is not None
                        and (
                            cleaned_files.mut_cleaned_structure_file
                            or cleaned_files.mut_cleaned_cif_file
                            or cleaned_files.mut_cleaned_pdb_file
                        )
                    )
                ),
                "mut_clean_report": bool(
                    # mut-integrate 使用配对清理报告，而不是两个独立 clean report。
                    "enzywizard-mut-clean" in preparation_tools
                    or (
                        cleaned_files is not None
                        and cleaned_files.mut_clean_report_file
                    )
                ),
                "cleaned_amino_acid_substitution": bool(
                    # 下游突变工具使用 mut-clean 标准化后的替换信息。
                    "enzywizard-mut-clean" in preparation_tools
                    or (
                        cleaned_files is not None
                        and cleaned_files.cleaned_amino_acid_substitution
                    )
                ),
            },
        }

    # 先解释三个自定义字段，再提供程序计算结果，避免模型猜测字段含义。
    sections.append(
        "Program-resolved input state:\n"
        "- program_selected_preparation_tools: Preparation tools already added to "
        "the current plan by the program. They may not have run yet. Do not return "
        "these tools.\n"
        "- provided_inputs: External inputs supplied by the user. Preparation cannot "
        "create a missing MSA or substrate.\n"
        "- available_analysis_inputs: Inputs that will exist when the selected analysis "
        "tools start. A true value means the input will be available.\n"
        + json.dumps(input_availability, ensure_ascii=False, indent=2)
    )

    # 当前请求决定分析目标和所需分析工具。
    current_user_message = state.get("current_user_message")
    if current_user_message:
        sections.append(f"Current user request:\n{current_user_message.strip()}")

    # 若用户要求修改上一版计划，将修改意见作为最新约束传给工具选择模型。
    user_confirmation = state.get("current_user_confirmation")
    if user_confirmation is not None and user_confirmation.revision_request:
        sections.append(
            f"User revision request:\n{user_confirmation.revision_request.strip()}"
        )

    # 历史用户需求只通过摘要传入，普通历史 artifacts 不参与跨轮工具规划。
    user_message_summary = state.get("user_message_summary")
    if user_message_summary:
        sections.append(f"Previous user request summary:\n{user_message_summary.strip()}")

    # 保留输入的实际非空值，使模型能判断 MSA、底物等外部输入是否存在。
    if agent_inputs is not None:
        input_context = json.dumps(
            agent_inputs.model_dump(exclude_none=True),
            ensure_ascii=False,
            indent=2,
        )
        sections.append(f"Agent inputs:\n{input_context}")

    # 重申上一步意图识别结果，工具选择模型不再重新判断 purpose。
    planning_context = {
        "purpose_type": planning_result.purpose_type,
        "purpose_reason": planning_result.purpose_reason,
        "needs_tool": planning_result.needs_tool,
    }
    sections.append(
        "Current planning intent:\n"
        + json.dumps(planning_context, ensure_ascii=False, indent=2)
    )

    # 单独列出 selected_tools 的动态允许值，便于模型直接核对准确名称。
    allowed_tool_names = sorted(tool_specs)
    sections.append(
        "Allowed selected_tools values:\n"
        + json.dumps(allowed_tool_names, ensure_ascii=False, indent=2)
    )

    # 候选规格已按 purpose 筛选并删除程序管理的准备、整合工具，同时只保留
    # 当前 LLM 选择步骤需要的字段。
    tool_spec_context = json.dumps(tool_specs, ensure_ascii=False, indent=2)
    sections.append(f"Candidate EnzyWizard tool specifications:\n{tool_spec_context}")

    return "\n\n".join(sections)


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
