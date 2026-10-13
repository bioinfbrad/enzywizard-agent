from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ..builders.planning_context_builder import (
    build_parameter_extraction_context,
    build_planning_context,
    build_tool_selection_context,
)
from ..data_models.planning import (
    IntentClassificationOutput,
    ParameterExtractionOutput,
    PlanningResult,
    ToolSelectionOutput,
    ToolPlan,
)
from ..prompts.planning_prompt import (
    INTENT_CLASSIFICATION_SYSTEM_PROMPT,
    PARAMETER_EXTRACTION_SYSTEM_PROMPT,
    SINGLE_PROTEIN_TOOL_SELECTION_SYSTEM_PROMPT,
    WILDTYPE_MUTANT_TOOL_SELECTION_SYSTEM_PROMPT,
)
from ..state import AgentState
from ..utils.planning_utils import (
    create_tool_selection_output_model,
    create_tool_selection_failure,
    determine_preparation_tools,
    load_analysis_tool_specs_for_purpose,
    validate_and_update_tool_selection,
)
from ..utils.tool_plan_utils import (
    apply_parameter_extraction,
    build_default_tool_plan,
)


def classify_intent(state: AgentState) -> IntentClassificationOutput:
    """调用一次 LLM，将当前用户请求识别为固定 Planning purpose。"""

    # 配置由入口创建；意图上下文由 builder 统一组织。
    config = state["config"]
    context = build_planning_context(state)

    # 温度为 0 以降低分类波动；禁用模型 thinking 以减少等待时间和无关输出。
    llm = ChatOpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        model=config.model,
        temperature=0,
        max_retries=0,
        extra_body={"enable_thinking": False},
    )

    # LangChain 根据 Pydantic 模型要求模型返回固定 JSON 字段和枚举值。
    structured_llm = llm.with_structured_output(IntentClassificationOutput)

    # SystemMessage 保存稳定规则，HumanMessage 只承载本轮动态上下文。
    return structured_llm.invoke(
        [
            SystemMessage(content=INTENT_CLASSIFICATION_SYSTEM_PROMPT),
            HumanMessage(content=context),
        ]
    )


def select_single_protein_tools(
    state: AgentState,
    planning_result: PlanningResult,
) -> ToolSelectionOutput:
    """为单蛋白工作流选择准备工具和剩余分析工具。"""

    purpose_type = "single_protein_analysis"

    # 交给 LLM 的规格已排除 enzywizard-clean 和 enzywizard-integrate，
    # 只保留单项分析工具与 enzywizard-batch。
    analysis_tool_specs = load_analysis_tool_specs_for_purpose(purpose_type)

    # clean 是否需要由程序根据 AgentInputs 和 CleanedFiles 确定。
    preparation_tools, preparation_error = determine_preparation_tools(
        state,
        purpose_type,
    )

    # 没有可信清理结果且没有原始结构时，LLM 无法补救，直接返回规范失败结果。
    if preparation_error is not None:
        return create_tool_selection_failure(preparation_error)

    # Builder 会告诉模型准备完成后的输入状态，并只附加当前允许的候选规格。
    context = build_tool_selection_context(
        state,
        planning_result,
        analysis_tool_specs,
        preparation_tools,
    )
    config = state["config"]

    # 工具选择同样采用确定性采样配置。
    llm = ChatOpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        model=config.model,
        temperature=0,
        max_retries=0,
        extra_body={"enable_thinking": False},
    )
    # 动态模型把本次候选名称写入 selected_tools 的 JSON Schema 枚举。
    # 用于根据当前允许选择的工具，动态创建一个更严格的 Pydantic 输出模型
    output_model = create_tool_selection_output_model(
        set(analysis_tool_specs),
        purpose_type,
    )
    structured_llm = llm.with_structured_output(output_model)

    try:
        # LLM 只返回单项分析工具或 batch；准备和整合均由程序加入。
        tool_selection = structured_llm.invoke(
            [
                SystemMessage(content=SINGLE_PROTEIN_TOOL_SELECTION_SYSTEM_PROMPT),
                HumanMessage(content=context),
            ]
        )

        # 允许集合由 LLM 候选加上程序实际选中的准备工具组成。
        # 验证 LLM 选择，并合并准备工具、整合工具或 batch 等程序更新。
        return validate_and_update_tool_selection(
            tool_selection,
            set(analysis_tool_specs) | set(preparation_tools),
            preparation_tools,
            purpose_type,
        )
    except Exception as e:
        # API、结构化解析等异常统一转换成 ToolSelectionOutput 失败结果。
        return create_tool_selection_failure(
            f"Single-protein tool selection failed: {e}"
        )


def select_wildtype_mutant_tools(
    state: AgentState,
    planning_result: PlanningResult,
) -> ToolSelectionOutput:
    """为 WT/MUT 比较工作流选择准备工具和剩余配对分析工具。"""

    purpose_type = "wildtype_mutant_comparison"

    # 交给 LLM 的规格排除 enzywizard-mut-clean 和 enzywizard-mut-integrate，
    # 只保留配对单项分析工具与 enzywizard-mut-batch。
    analysis_tool_specs = load_analysis_tool_specs_for_purpose(purpose_type)

    # 程序检查完整 mut-clean 状态，或检查能否从当前 WT/MUT 原始输入生成。
    preparation_tools, preparation_error = determine_preparation_tools(
        state,
        purpose_type,
    )
    if preparation_error is not None:
        return create_tool_selection_failure(preparation_error)

    # 动态上下文只描述突变比较相关输入和准备后的两侧可用状态。
    context = build_tool_selection_context(
        state,
        planning_result,
        analysis_tool_specs,
        preparation_tools,
    )
    config = state["config"]

    # 保持与单蛋白分支一致的模型参数和结构化输出方式。
    llm = ChatOpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        model=config.model,
        temperature=0,
        max_retries=0,
        extra_body={"enable_thinking": False},
    )
    # WT/MUT 使用自己的候选枚举，避免模型返回单蛋白工具或程序管理的整合工具。
    output_model = create_tool_selection_output_model(
        set(analysis_tool_specs),
        purpose_type,
    )
    structured_llm = llm.with_structured_output(output_model)

    try:
        # LLM 返回一组逻辑分析工具；后续 ToolPlan 会把蛋白侧工具展开为 WT/MUT
        # 两侧调用，并把共享底物准备保留为单次调用。
        tool_selection = structured_llm.invoke(
            [
                SystemMessage(content=WILDTYPE_MUTANT_TOOL_SELECTION_SYSTEM_PROMPT),
                HumanMessage(content=context),
            ]
        )

        # 合并程序决定的 mut-clean，并拒绝 LLM 越权选择未允许的准备工具。
        return validate_and_update_tool_selection(
            tool_selection,
            set(analysis_tool_specs) | set(preparation_tools),
            preparation_tools,
            purpose_type,
        )
    except Exception as e:
        # 保留突变分支名称，便于日志区分是哪类工具选择调用失败。
        return create_tool_selection_failure(
            f"Wild-type/mutant tool selection failed: {e}"
        )


def generate_default_tool_plan(
    state: AgentState,
    planning_result: PlanningResult,
) -> ToolPlan:
    """用确定性规则把工具选择展开为有序默认计划。"""

    # 具体顺序、路径、上下游引用和默认值集中由 tool_plan_utils 生成；服务层
    # 保持 Planning 节点与工具函数之间的稳定调用边界。
    tool_plan = build_default_tool_plan(state, planning_result)

    # 依赖闭包可能补充 substrate 或 dock。最终 ToolSelectionOutput 必须与实际
    # ToolPlan 保持一致，避免用户确认时看到两套不同工具集合。
    planned_tool_names = list(dict.fromkeys(call.tool_name for call in tool_plan.calls))
    tool_selection = planning_result.tool_selection
    if tool_selection is not None:
        added_tool_names = [
            tool_name
            for tool_name in planned_tool_names
            if tool_name not in tool_selection.selected_tools
        ]
        tool_selection.selected_tools = planned_tool_names
        if added_tool_names:
            added_names = ", ".join(added_tool_names)
            tool_selection.selection_reason += (
                f" [System] Added required workflow dependencies: {added_names}."
            )
    return tool_plan


def extract_tool_parameters(
    state: AgentState,
    planning_result: PlanningResult,
    default_tool_plan: ToolPlan,
) -> ParameterExtractionOutput:
    """调用一次 LLM，提取用户对默认 ToolPlan 的稀疏参数要求。"""

    config = state["config"]
    context = build_parameter_extraction_context(
        state,
        planning_result,
        default_tool_plan,
    )

    # 与前两次 Planning 调用保持一致：温度为0、禁用 thinking、不自动重试。
    llm = ChatOpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        model=config.model,
        temperature=0,
        max_retries=0,
        extra_body={"enable_thinking": False},
    )
    structured_llm = llm.with_structured_output(ParameterExtractionOutput)

    # SystemMessage 只包含固定参数政策，HumanMessage 提供当前计划的动态参数表。
    return structured_llm.invoke(
        [
            SystemMessage(content=PARAMETER_EXTRACTION_SYSTEM_PROMPT),
            HumanMessage(content=context),
        ]
    )


def finalize_tool_plan(
    state: AgentState,
    default_tool_plan: ToolPlan,
    parameter_extraction: ParameterExtractionOutput,
) -> ToolPlan:
    """应用并校验参数提取结果，返回可进入用户确认的最终计划。"""

    # apply_parameter_extraction 会拒绝路径覆盖、未知参数、非法类型、范围错误和
    # 参数依赖冲突；任何异常由 Planning 节点统一转换为 planning_failed。
    return apply_parameter_extraction(
        default_tool_plan,
        parameter_extraction,
        state["config"].workdir,
    )
