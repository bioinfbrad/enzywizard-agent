from __future__ import annotations

import re
from importlib.resources import files
from typing import Any

import yaml
from ..data_models.planning import PlanningPurposeType, ToolSelectionOutput
from ..state import AgentState


TOOL_SPEC_FIELD_NAMES = [
    # 这些字段足以支持工具选择；参数和输出细节留给后续参数规划读取。
    "command_name",
    "description",
    "capabilities",
    "use_when",
    "do_not_use_when",
    "input_file_requirements",
    "planning_rules",
]


def purpose_needs_tool(purpose_type: PlanningPurposeType) -> bool:
    """判断意图是否进入 EnzyWizard 工具工作流。"""

    return purpose_type in (
        "single_protein_analysis",
        "wildtype_mutant_comparison",
    )


def load_enzywizard_tool_specs() -> dict[str, dict[str, Any]]:
    """从包内 YAML 资源读取17个 EnzyWizard 工具的完整规格。"""

    # importlib.resources 可同时支持源码目录和安装后的 Python 包。
    spec_path = files("enzywizard_agent.knowledge_base.enzywizard_tool_specs").joinpath(
        "enzywizard_tools.yaml"
    )

    # YAML 顶层的 tools 映射以标准工具名为键。
    spec_data = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    return spec_data["tools"]


def load_tool_specs_for_purpose(
    purpose_type: PlanningPurposeType,
) -> dict[str, dict[str, Any]]:
    """按 purpose 筛选工具，并返回适合选择阶段使用的精简规格。"""

    tool_specs = load_enzywizard_tool_specs()
    selected_specs: dict[str, dict[str, Any]] = {}

    # allowed_in_purpose_types 可以包含一个或两个工作流类型。
    for tool_name, tool_spec in tool_specs.items():
        allowed_purpose_types = tool_spec.get("allowed_in_purpose_types", [])
        if purpose_type in allowed_purpose_types:
            selected_specs[tool_name] = select_tool_spec_fields(tool_spec)

    return selected_specs


def load_analysis_tool_specs_for_purpose(
    purpose_type: PlanningPurposeType,
) -> dict[str, dict[str, Any]]:
    """返回当前 purpose 允许编译的单项分析与 batch 工具规格。"""

    tool_specs = load_tool_specs_for_purpose(purpose_type)

    # 准备和整合均由程序按确定性规则加入。batch 保留在允许集中，
    # 仅当结构化需求明确为 complete_analysis 时由编译器选用。
    program_managed_tool_names = (
        {"enzywizard-clean", "enzywizard-integrate"}
        if purpose_type == "single_protein_analysis"
        else {"enzywizard-mut-clean", "enzywizard-mut-integrate"}
    )
    return {
        tool_name: tool_spec
        for tool_name, tool_spec in tool_specs.items()
        if tool_name not in program_managed_tool_names
    }


def select_tool_spec_fields(tool_spec: dict[str, Any]) -> dict[str, Any]:
    """从完整 YAML 条目投影出工具选择真正需要的字段。"""

    # 使用白名单可防止 required_arguments、output_files 等大字段进入当前 Prompt。
    return {
        field_name: tool_spec[field_name]
        for field_name in TOOL_SPEC_FIELD_NAMES
        if field_name in tool_spec
    }


def determine_preparation_tools(
    state: AgentState,
    purpose_type: PlanningPurposeType,
) -> tuple[list[str], str | None]:
    """用确定性规则判断当前工作流是否需要 clean 或 mut-clean。

    返回值为 ``(准备工具列表, 错误信息)``。成功时错误为 None；基础输入不足时
    工具列表为空并返回可直接写入 ToolSelectionOutput 的错误文本。
    """

    agent_inputs = state.get("agent_inputs")
    cleaned_files = state.get("cleaned_files")

    if purpose_type == "single_protein_analysis":
        # 完整单蛋白清理状态必须同时包含结构、FASTA 和 clean report，确保后续
        # embedding、conservation 和 integrate 都能直接复用。
        cleaned_structure_available = bool(
            cleaned_files is not None
            and (
                cleaned_files.single_cleaned_structure_file
                or cleaned_files.single_cleaned_cif_file
                or cleaned_files.single_cleaned_pdb_file
            )
        )
        clean_report_available = bool(
            cleaned_files is not None and cleaned_files.single_clean_report_file
        )
        cleaned_fasta_available = bool(
            cleaned_files is not None and cleaned_files.single_cleaned_fasta_file
        )

        # 完整清理结果可直接复用，不重复运行 clean。
        if (
            cleaned_structure_available
            and cleaned_fasta_available
            and clean_report_available
        ):
            return [], None

        # 清理结果不完整但当前有原始结构时，由程序强制加入 clean。
        if agent_inputs is not None and agent_inputs.structure_file:
            return ["enzywizard-clean"], None

        # 既没有完整清理结果也没有原始结构，任何下游结构分析都无法开始。
        return [], (
            "Single-protein analysis requires either a complete trusted clean result "
            "or a current structure_file that enzywizard-clean can process."
        )

    # WT/MUT 完整基础状态必须同时具有两侧结构、两侧 FASTA、配对报告和
    # 标准化替换，才能覆盖所有后续配对分析工具。
    paired_cleaned_inputs_available = bool(
        cleaned_files is not None
        and (
            cleaned_files.wt_cleaned_structure_file
            or cleaned_files.wt_cleaned_cif_file
            or cleaned_files.wt_cleaned_pdb_file
        )
        and (
            cleaned_files.mut_cleaned_structure_file
            or cleaned_files.mut_cleaned_cif_file
            or cleaned_files.mut_cleaned_pdb_file
        )
        and cleaned_files.mut_clean_report_file
        and cleaned_files.cleaned_amino_acid_substitution
        and cleaned_files.wt_cleaned_fasta_file
        and cleaned_files.mut_cleaned_fasta_file
    )
    if paired_cleaned_inputs_available:
        return [], None

    # mut-clean 只能在当前 WT、MUT 结构和替换信息都存在时生成完整基础状态。
    paired_raw_inputs_available = bool(
        agent_inputs is not None
        and agent_inputs.structure_file
        and agent_inputs.mut_structure_file
        and agent_inputs.amino_acid_substitution
    )
    if paired_raw_inputs_available:
        return ["enzywizard-mut-clean"], None

    # 缺少任一组完整输入时，提前失败而不是把无法补救的问题交给 LLM。
    return [], (
        "WT/MUT comparison requires either a complete trusted mut-clean result or "
        "current WT and MUT structure files plus an amino acid substitution that "
        "enzywizard-mut-clean can process."
    )


def compose_tool_selection_reason(
    llm_reason: str,
    system_reasons: list[str],
) -> str:
    """按来源拼接 LLM 原因和程序更新原因，不改写 LLM 原始文本。"""

    reason_parts = [f"[LLM] {llm_reason}"]
    for reason in system_reasons:
        formatted_reason = reason if reason.endswith(".") else f"{reason}."
        reason_parts.append(f"[System] {formatted_reason}")
    return " ".join(reason_parts)


def validate_and_update_tool_selection(
    tool_selection: ToolSelectionOutput,
    candidate_tool_names: set[str],
    preparation_tool_names: list[str],
    purpose_type: PlanningPurposeType,
) -> ToolSelectionOutput:
    """验证 LLM 工具选择，并按程序规则更新为最终工具集合。"""

    # 模型主动报告失败时不再尝试合并工具，只规范失败字段。
    if tool_selection.tool_selection_status == "tool_selection_failed":
        error_message = tool_selection.error_message or tool_selection.selection_reason
        return create_tool_selection_failure(error_message)

    # 成功且 LLM 工具列表为空时，程序将其解释为“仅执行准备工具”。如果程序
    # 没有选出准备工具，同样的空列表就是无效成功结果。
    only_preparation_needed = not tool_selection.selected_tools
    if only_preparation_needed and not preparation_tool_names:
        return create_tool_selection_failure(
            "Tool selection reported success without selecting analysis tools or "
            "having a program-selected preparation tool."
        )

    # 成功结果必须有原因且不能同时携带错误信息。
    if not tool_selection.selection_reason:
        return create_tool_selection_failure(
            "Tool selection reported success without a selection reason."
        )
    if tool_selection.error_message:
        return create_tool_selection_failure(tool_selection.error_message)

    # 准备工具必须排在分析工具之前，最终列表也保持这一基本顺序。
    selected_tool_names = preparation_tool_names + tool_selection.selected_tools

    # 将候选工具建立为“宽松格式 -> YAML 标准名称”的映射。
    normalized_candidates = {
        re.sub(r"[_\s]+", "-", tool_name.strip().casefold()): tool_name
        for tool_name in candidate_tool_names
    }
    # 允许模型在大小写、空格和下划线上有轻微格式偏差。
    normalized_tool_names = [
        re.sub(r"[_\s]+", "-", tool_name.strip().casefold())
        for tool_name in selected_tool_names
    ]
    # 规范化后仍不在允许集合中的名称属于模型幻觉或越权选择。
    invalid_tool_names = [
        tool_name
        for tool_name, normalized_name in zip(
            selected_tool_names,
            normalized_tool_names,
        )
        if normalized_name not in normalized_candidates
    ]
    if invalid_tool_names:
        invalid_names = ", ".join(invalid_tool_names)
        return ToolSelectionOutput(
            tool_selection_status="tool_selection_failed",
            selected_tools=[],
            selection_reason="",
            error_message=(
                "Tool selection returned tools outside the allowed candidate set: "
                f"{invalid_names}."
            ),
        )

    # 后续状态只保存 YAML 中的标准工具名，不保存模型原始拼写。
    canonical_tool_names = [
        normalized_candidates[normalized_name]
        for normalized_name in normalized_tool_names
    ]
    # 不同原始写法可能映射到同一工具，因此需要在规范化后再次查重。
    if len(canonical_tool_names) != len(set(canonical_tool_names)):
        return create_tool_selection_failure(
            "Tool selection returned duplicate tools after name normalization."
        )

    # 当前 purpose 唯一决定其整合工具和 batch 工具，避免两类工作流混用名称。
    integration_tool_name, batch_tool_name = (
        ("enzywizard-integrate", "enzywizard-batch")
        if purpose_type == "single_protein_analysis"
        else ("enzywizard-mut-integrate", "enzywizard-mut-batch")
    )

    # 每个成功结果都从 LLM 原因开始，并至少追加一条程序决策原因。准备工具
    # 是独立更新，可能与单项整合或直接 batch 同时出现。
    system_reasons: list[str] = []
    if preparation_tool_names:
        preparation_names = ", ".join(preparation_tool_names)
        if only_preparation_needed:
            system_reasons.append(
                f"Kept only {preparation_names} because the user requested structure "
                f"preparation without analysis."
            )
        elif purpose_type == "single_protein_analysis":
            system_reasons.append(
                f"Added {preparation_names} because no complete trusted "
                f"single-protein clean result was available for reuse."
            )
        else:
            system_reasons.append(
                f"Added {preparation_names} because no complete trusted paired "
                f"WT/MUT clean result was available for reuse."
            )

    # 准备、整合和 batch 都不属于单项分析工具。这里单独提取单项工具，供
    # 后续判断是否需要由程序补充整合步骤。
    selected_analysis_tool_names = set(canonical_tool_names) - set(
        preparation_tool_names
    )
    individual_tool_names = selected_analysis_tool_names - {
        integration_tool_name,
        batch_tool_name,
    }

    # 普通分析计划必须至少包含一个单项分析工具或唯一 batch。integrate 不能
    # 单独执行，因为它需要消费当前计划中单项工具生成的报告。
    if (
        not only_preparation_needed
        and not individual_tool_names
        and batch_tool_name not in selected_analysis_tool_names
    ):
        return create_tool_selection_failure(
            "Tool selection must include at least one individual analysis tool or "
            f"{batch_tool_name}."
        )

    # batch/mut-batch 已覆盖全部单项分析和整合，只允许与程序准备工具共存。
    for selected_batch_tool_name in ("enzywizard-batch", "enzywizard-mut-batch"):
        if selected_batch_tool_name in selected_analysis_tool_names:
            redundant_tool_names = selected_analysis_tool_names - {
                selected_batch_tool_name
            }
            if not redundant_tool_names:
                continue

            redundant_names = ", ".join(sorted(redundant_tool_names))
            return create_tool_selection_failure(
                f"{selected_batch_tool_name} already covers all individual analysis and "
                f"integration tools, so it cannot be combined with: {redundant_names}."
            )

    # 完整分析需求编译为 batch 时不再改工具集合，但仍记录
    # 程序确认原因，便于用户审核最终计划。
    if batch_tool_name in selected_analysis_tool_names:
        system_reasons.append(
            f"Kept {batch_tool_name} because the structured requirements specify a "
            f"complete analysis, and the batch workflow already includes analysis "
            f"and report integration."
        )

    # 任意数量的单项分析都必须生成整合图。整合工具不参与 LLM 选择，由程序
    # 在最终集合中统一补齐。batch 自带整合，因此不再追加独立整合工具。
    if (
        not only_preparation_needed
        and individual_tool_names
        and batch_tool_name not in selected_analysis_tool_names
        and integration_tool_name not in selected_analysis_tool_names
    ):
        individual_tool_count = len(individual_tool_names)
        individual_tool_label = (
            "individual analysis tool"
            if individual_tool_count == 1
            else "individual analysis tools"
        )
        canonical_tool_names.append(integration_tool_name)
        system_reasons.append(
            f"Added {integration_tool_name} because the selection contains "
            f"{individual_tool_count} {individual_tool_label}, and individual-tool "
            f"workflows require report integration."
        )

    # 所有校验通过后，返回字段完全一致的成功结果。
    return ToolSelectionOutput(
        tool_selection_status="tool_selection_succeeded",
        selected_tools=canonical_tool_names,
        selection_reason=compose_tool_selection_reason(
            tool_selection.selection_reason,
            system_reasons,
        ),
        error_message="",
    )


def create_tool_selection_failure(error_message: str) -> ToolSelectionOutput:
    """把任意失败文本转换成字段一致的 ToolSelectionOutput。"""

    # 防止调用方传入空字符串，确保失败结果始终具有可解释原因。
    message = str(error_message).strip() or "Tool selection failed without an error message."
    return ToolSelectionOutput(
        tool_selection_status="tool_selection_failed",
        selected_tools=[],
        selection_reason="",
        error_message=message,
    )
