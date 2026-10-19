from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from ..builders.planning_context_builder import (
    build_parameter_extraction_context,
    build_planning_context,
    build_workflow_requirements_context,
)
from ..data_models.planning import (
    IntentClassificationOutput,
    ParameterAdjustment,
    ParameterExtractionOutput,
    PlanningResult,
    ToolSelectionOutput,
    ToolPlan,
    WorkflowRequirementsDraft,
    WorkflowRequirementsOutput,
)
from ..prompts.planning_prompt import (
    INTENT_CLASSIFICATION_SYSTEM_PROMPT,
    PARAMETER_EXTRACTION_SYSTEM_PROMPT,
    WORKFLOW_REQUIREMENTS_REVIEW_SYSTEM_PROMPT,
    WORKFLOW_REQUIREMENTS_SYSTEM_PROMPT,
)
from ..state import AgentState
from .llm_service import create_chat_model_for_state, invoke_structured
from .runtime_service import is_runtime_limit_error
from ..utils.planning_utils import (
    create_tool_selection_failure,
    determine_preparation_tools,
    load_analysis_tool_specs_for_purpose,
    validate_and_update_tool_selection,
)
from ..utils.tool_plan_utils import (
    apply_parameter_extraction,
    build_default_tool_plan,
    get_editable_parameter_specs,
)


ANALYSIS_TYPE_TO_TOOL = {
    "aaprops": "enzywizard-aaprops",
    "hydrocluster": "enzywizard-hydrocluster",
    "energy": "enzywizard-energy",
    "flexibility": "enzywizard-flexibility",
    "disorder": "enzywizard-disorder",
    "conservation": "enzywizard-conservation",
    "embedding": "enzywizard-embedding",
    "pocket": "enzywizard-pocket",
    "substrate": "enzywizard-substrate",
    "docking": "enzywizard-dock",
    "interaction": "enzywizard-interaction",
}
def _is_historical_rerun(state: AgentState) -> bool:
    routing = state.get("current_request_routing")
    if routing is None or routing.output is None:
        return False
    action_types = {action.action_type for action in routing.output.actions}
    return (
        routing.output.requests_recomputation
        and "reference_historical_context" in action_types
        and "run_tool_workflow" in action_types
    )


def get_historical_rerun_source(state: AgentState) -> PlanningResult | None:
    """Resolve a rerun to a previously approved or executed tool plan."""

    if not _is_historical_rerun(state):
        return None
    approved_keys = {
        (confirmation.turn_id, confirmation.plan_id)
        for confirmation in state.get("user_confirmations", [])
        if confirmation.decision == "approved"
    }
    approved_keys.update(
        (execution.turn_id, execution.plan_id)
        for execution in state.get("tool_executions", [])
    )
    candidates = [
        result
        for result in state.get("planning_results", [])
        if result.tool_plan is not None
        and (result.turn_id, result.plan_id) in approved_keys
    ]
    history = state.get("current_history_query")
    resolved_turn_ids = set(
        history.resolved_turn_ids
        if history is not None and history.query_status == "succeeded"
        else []
    )
    matching_turn_candidates = [
        result for result in candidates if result.turn_id in resolved_turn_ids
    ]
    if resolved_turn_ids:
        candidates = matching_turn_candidates
    if not candidates:
        return None
    return max(candidates, key=lambda result: (result.turn_id, result.plan_id))


def _requirements_from_historical_plan(
    source: PlanningResult,
) -> WorkflowRequirementsOutput:
    if (
        source.workflow_requirements is not None
        and source.workflow_requirements.workflow_requirements_status
        == "workflow_requirements_succeeded"
    ):
        return source.workflow_requirements.model_copy(
            update={
                "requirements_reason": (
                    f"Reused approved requirements from turn {source.turn_id}, "
                    f"plan {source.plan_id}."
                )
            }
        )

    assert source.tool_plan is not None
    tool_names = {call.tool_name for call in source.tool_plan.calls}
    if tool_names.intersection({"enzywizard-batch", "enzywizard-mut-batch"}):
        scope_mode = "complete_analysis"
        requested: list[str] = []
    else:
        requested = [
            analysis_type
            for analysis_type, tool_name in ANALYSIS_TYPE_TO_TOOL.items()
            if tool_name in tool_names
        ]
        scope_mode = "specific_analyses" if requested else "preparation_only"
    return WorkflowRequirementsOutput.model_validate(
        {
            "workflow_requirements_status": "workflow_requirements_succeeded",
            "scope_mode": scope_mode,
            "requested_analysis_types": requested,
            "excluded_analysis_types": [],
            "requirements_reason": (
                f"Derived approved requirements from turn {source.turn_id}, "
                f"plan {source.plan_id}."
            ),
            "error_message": "",
        }
    )


def apply_historical_rerun_settings(
    state: AgentState,
    default_tool_plan: ToolPlan,
) -> ToolPlan:
    """Apply prior editable values to a freshly generated plan."""

    source = get_historical_rerun_source(state)
    if source is None or source.tool_plan is None:
        if _is_historical_rerun(state):
            raise ValueError("No approved historical tool plan is available to rerun.")
        return default_tool_plan

    source_calls: dict[tuple[str, str], list] = {}
    for call in source.tool_plan.calls:
        source_calls.setdefault((call.tool_name, call.side), []).append(call)

    adjustments: list[ParameterAdjustment] = []
    used_indexes: dict[tuple[str, str], int] = {}
    for call in default_tool_plan.calls:
        key = (call.tool_name, call.side)
        source_index = used_indexes.get(key, 0)
        matching_calls = source_calls.get(key, [])
        if source_index >= len(matching_calls):
            continue
        source_call = matching_calls[source_index]
        used_indexes[key] = source_index + 1
        editable_specs = get_editable_parameter_specs(call.tool_name)
        for parameter_name in editable_specs:
            value = source_call.arguments.get(parameter_name)
            if not isinstance(value, (str, int, float, bool)):
                continue
            adjustments.append(
                ParameterAdjustment(
                    step_index=call.step_index,
                    parameter_name=parameter_name,
                    parameter_value=value,
                    adjustment_reason=(
                        f"Reuse turn {source.turn_id}, plan {source.plan_id} setting."
                    ),
                )
            )

    inherited = ParameterExtractionOutput(
        parameter_extraction_status="parameter_extraction_succeeded",
        parameter_adjustments=adjustments,
        profile_requests=[],
        unsupported_requests=[],
        extraction_summary=(
            f"Inherited editable settings from turn {source.turn_id}, "
            f"plan {source.plan_id}."
        ),
        error_message="",
    )
    return apply_parameter_extraction(
        default_tool_plan,
        inherited,
        state["config"].workdir,
    )


def _create_planning_llm(state: AgentState):
    """统一 Planning 模型配置，保证需求提取和审核采用相同边界。"""

    return create_chat_model_for_state(state)


def _normalize_workflow_requirements_draft(
    draft: WorkflowRequirementsDraft | WorkflowRequirementsOutput,
) -> WorkflowRequirementsOutput:
    """把可修正的 LLM 草稿编译为严格工作流需求对象。"""

    if draft.workflow_requirements_status == "workflow_requirements_failed":
        return WorkflowRequirementsOutput(
            workflow_requirements_status="workflow_requirements_failed",
            scope_mode=draft.scope_mode,
            requested_analysis_types=[],
            excluded_analysis_types=[],
            requirements_reason="",
            error_message=(
                draft.error_message
                or "The scientific workflow requirements could not be determined."
            ),
        )

    requested = list(dict.fromkeys(draft.requested_analysis_types))
    excluded = list(dict.fromkeys(draft.excluded_analysis_types))
    excluded_set = set(excluded)
    requested = [item for item in requested if item not in excluded_set]

    # complete 和 preparation 的具体分析列表没有独立语义；清空它们后由程序
    # 分别映射到 batch 或纯准备流程，避免可修正草稿在审核前被拒绝。
    if draft.scope_mode in {"complete_analysis", "preparation_only"}:
        requested = []

    if not draft.requirements_reason:
        return WorkflowRequirementsOutput(
            workflow_requirements_status="workflow_requirements_failed",
            scope_mode=draft.scope_mode,
            requested_analysis_types=[],
            excluded_analysis_types=[],
            requirements_reason="",
            error_message="The workflow requirements did not include a rationale.",
        )
    if draft.scope_mode == "specific_analyses" and not requested:
        return WorkflowRequirementsOutput(
            workflow_requirements_status="workflow_requirements_failed",
            scope_mode=draft.scope_mode,
            requested_analysis_types=[],
            excluded_analysis_types=[],
            requirements_reason="",
            error_message=(
                "A specific-analysis workflow did not identify any supported "
                "analysis type."
            ),
        )

    return WorkflowRequirementsOutput(
        workflow_requirements_status="workflow_requirements_succeeded",
        scope_mode=draft.scope_mode,
        requested_analysis_types=requested,
        excluded_analysis_types=excluded,
        requirements_reason=draft.requirements_reason,
        error_message="",
    )


def _extract_workflow_requirements(
    state: AgentState,
    planning_result: PlanningResult,
    preparation_tools: list[str],
) -> WorkflowRequirementsOutput:
    """先提取科学需求，再用独立模型审核最新修订和范围边界。"""

    rerun_source = get_historical_rerun_source(state)
    if rerun_source is not None:
        return _requirements_from_historical_plan(rerun_source)
    if _is_historical_rerun(state):
        return WorkflowRequirementsOutput(
            workflow_requirements_status="workflow_requirements_failed",
            scope_mode="preparation_only",
            requested_analysis_types=[],
            excluded_analysis_types=[],
            requirements_reason="",
            error_message="No approved historical tool plan is available to rerun.",
        )

    context = build_workflow_requirements_context(
        state,
        planning_result,
        preparation_tools,
    )
    draft = invoke_structured(
        state,
        stage="planning.workflow_requirements",
        schema=WorkflowRequirementsDraft,
        llm=_create_planning_llm(state),
        messages=[
            SystemMessage(content=WORKFLOW_REQUIREMENTS_SYSTEM_PROMPT),
            HumanMessage(content=context),
        ],
    )
    review_context = (
        "Original workflow context:\n"
        + context
        + "\n\nCandidate workflow requirements:\n"
        + draft.model_dump_json(indent=2)
    )
    try:
        reviewed = invoke_structured(
            state,
            stage="planning.workflow_requirements_review",
            schema=WorkflowRequirementsDraft,
            llm=_create_planning_llm(state),
            messages=[
                SystemMessage(content=WORKFLOW_REQUIREMENTS_REVIEW_SYSTEM_PROMPT),
                HumanMessage(content=review_context),
            ],
        )
    except Exception as error:
        if is_runtime_limit_error(error):
            raise
        # 审核服务异常不应抹掉已经满足基础类型契约的首轮草稿。
        reviewed = draft

    candidate_output = _normalize_workflow_requirements_draft(draft)
    reviewed_output = _normalize_workflow_requirements_draft(reviewed)
    if (
        candidate_output.workflow_requirements_status
        == "workflow_requirements_succeeded"
        and reviewed_output.workflow_requirements_status
        == "workflow_requirements_failed"
    ):
        # Reviewer 可以纠正科学范围，但不能把已经可编译的草稿降级为纯结构
        # 失败。此时保留首轮结果，后续依赖闭包和参数校验仍会执行严格检查。
        return candidate_output
    return reviewed_output


def _compile_workflow_requirements(
    state: AgentState,
    planning_result: PlanningResult,
    requirements: WorkflowRequirementsOutput,
    preparation_tools: list[str],
    candidate_tool_names: set[str],
) -> ToolSelectionOutput:
    """把科学需求确定性编译成工具集合，并校验不可由准备步骤生成的输入。"""

    planning_result.workflow_requirements = requirements
    if requirements.workflow_requirements_status == "workflow_requirements_failed":
        return create_tool_selection_failure(requirements.error_message)

    requested = set(requirements.requested_analysis_types)
    agent_inputs = state.get("agent_inputs")
    if "conservation" in requested:
        msa_available = bool(agent_inputs is not None and agent_inputs.msa_file)
        mut_msa_available = bool(
            agent_inputs is not None and agent_inputs.mut_msa_file
        )
        if not msa_available or (
            planning_result.purpose_type == "wildtype_mutant_comparison"
            and not mut_msa_available
        ):
            return create_tool_selection_failure(
                "Conservation analysis requires the supplied MSA inputs for every "
                "protein side in the workflow."
            )
    if requested.intersection({"substrate", "docking"}) and not (
        agent_inputs is not None and agent_inputs.substrate_name
    ):
        return create_tool_selection_failure(
            "Substrate preparation and docking require a supplied substrate name or "
            "SMILES input."
        )

    if requirements.scope_mode == "complete_analysis":
        batch_tool = (
            "enzywizard-batch"
            if planning_result.purpose_type == "single_protein_analysis"
            else "enzywizard-mut-batch"
        )
        selected_tools = [batch_tool]
    elif requirements.scope_mode == "preparation_only":
        selected_tools = []
    else:
        selected_tools = [
            ANALYSIS_TYPE_TO_TOOL[analysis_type]
            for analysis_type in requirements.requested_analysis_types
        ]

    raw_selection = ToolSelectionOutput(
        tool_selection_status="tool_selection_succeeded",
        selected_tools=selected_tools,
        selection_reason=requirements.requirements_reason,
        error_message="",
    )
    return validate_and_update_tool_selection(
        raw_selection,
        candidate_tool_names | set(preparation_tools),
        preparation_tools,
        planning_result.purpose_type,
    )


def classify_intent(state: AgentState) -> IntentClassificationOutput:
    """调用一次 LLM，将当前用户请求识别为固定 Planning purpose。"""

    rerun_source = get_historical_rerun_source(state)
    if rerun_source is not None and rerun_source.tool_plan is not None:
        return IntentClassificationOutput(
            purpose_type=rerun_source.tool_plan.purpose_type,
            purpose_reason=(
                f"Reuse approved workflow purpose from turn {rerun_source.turn_id}, "
                f"plan {rerun_source.plan_id}."
            ),
        )

    # 配置由入口创建；意图上下文由 builder 统一组织。
    context = build_planning_context(state)

    # SystemMessage 保存稳定规则，HumanMessage 只承载本轮动态上下文。
    return invoke_structured(
        state,
        stage="planning.intent_classification",
        schema=IntentClassificationOutput,
        llm=_create_planning_llm(state),
        messages=[
            SystemMessage(content=INTENT_CLASSIFICATION_SYSTEM_PROMPT),
            HumanMessage(content=context),
        ],
    )


def select_single_protein_tools(
    state: AgentState,
    planning_result: PlanningResult,
) -> ToolSelectionOutput:
    """为单蛋白工作流选择准备工具和剩余分析工具。"""

    purpose_type = "single_protein_analysis"

    # 允许编译的规格排除 clean 和 integrate；这两类步骤由程序管理。
    analysis_tool_specs = load_analysis_tool_specs_for_purpose(purpose_type)

    # clean 是否需要由程序根据 AgentInputs 和 CleanedFiles 确定。
    preparation_tools, preparation_error = determine_preparation_tools(
        state,
        purpose_type,
    )

    # 没有可信清理结果且没有原始结构时，LLM 无法补救，直接返回规范失败结果。
    if preparation_error is not None:
        return create_tool_selection_failure(preparation_error)

    try:
        requirements = _extract_workflow_requirements(
            state,
            planning_result,
            preparation_tools,
        )
        return _compile_workflow_requirements(
            state,
            planning_result,
            requirements,
            preparation_tools,
            set(analysis_tool_specs),
        )
    except Exception as e:
        # API、结构化解析或审核异常统一转换为可审计失败。
        return create_tool_selection_failure(
            f"Single-protein workflow requirement extraction failed: {e}"
        )


def select_wildtype_mutant_tools(
    state: AgentState,
    planning_result: PlanningResult,
) -> ToolSelectionOutput:
    """为 WT/MUT 比较工作流选择准备工具和剩余配对分析工具。"""

    purpose_type = "wildtype_mutant_comparison"

    # 允许编译的规格排除 mut-clean 和 mut-integrate；这两类步骤由程序管理。
    analysis_tool_specs = load_analysis_tool_specs_for_purpose(purpose_type)

    # 程序检查完整 mut-clean 状态，或检查能否从当前 WT/MUT 原始输入生成。
    preparation_tools, preparation_error = determine_preparation_tools(
        state,
        purpose_type,
    )
    if preparation_error is not None:
        return create_tool_selection_failure(preparation_error)

    try:
        requirements = _extract_workflow_requirements(
            state,
            planning_result,
            preparation_tools,
        )
        return _compile_workflow_requirements(
            state,
            planning_result,
            requirements,
            preparation_tools,
            set(analysis_tool_specs),
        )
    except Exception as e:
        # 保留突变分支名称，便于日志定位失败阶段。
        return create_tool_selection_failure(
            f"Wild-type/mutant workflow requirement extraction failed: {e}"
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

    context = build_parameter_extraction_context(
        state,
        planning_result,
        default_tool_plan,
    )

    # SystemMessage 只包含固定参数政策，HumanMessage 提供当前计划的动态参数表。
    return invoke_structured(
        state,
        stage="planning.parameter_extraction",
        schema=ParameterExtractionOutput,
        llm=_create_planning_llm(state),
        messages=[
            SystemMessage(content=PARAMETER_EXTRACTION_SYSTEM_PROMPT),
            HumanMessage(content=context),
        ],
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
