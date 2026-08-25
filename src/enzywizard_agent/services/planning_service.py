from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ..builders.planning_context_builder import (
    build_planning_context,
    build_tool_selection_context,
)
from ..data_models.planning import (
    IntentClassificationOutput,
    PlanningResult,
    ToolSelectionOutput,
)
from ..prompts.planning_prompt import (
    INTENT_CLASSIFICATION_SYSTEM_PROMPT,
    SINGLE_PROTEIN_TOOL_SELECTION_SYSTEM_PROMPT,
    WILDTYPE_MUTANT_TOOL_SELECTION_SYSTEM_PROMPT,
)
from ..state import AgentState
from ..utils.planning_utils import (
    create_tool_selection_failure,
    load_tool_specs_for_purpose,
    validate_selected_tools,
)


def classify_intent(state: AgentState) -> IntentClassificationOutput:
    config = state["config"]
    context = build_planning_context(state)

    llm = ChatOpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        model=config.model,
        temperature=0,
        max_retries=0,
        extra_body={"enable_thinking": False},
    )
    structured_llm = llm.with_structured_output(IntentClassificationOutput)

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
    tool_specs = load_tool_specs_for_purpose("single_protein_analysis")
    context = build_tool_selection_context(state, planning_result, tool_specs)
    config = state["config"]

    llm = ChatOpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        model=config.model,
        temperature=0,
        max_retries=0,
        extra_body={"enable_thinking": False},
    )
    structured_llm = llm.with_structured_output(ToolSelectionOutput)

    try:
        tool_selection = structured_llm.invoke(
            [
                SystemMessage(content=SINGLE_PROTEIN_TOOL_SELECTION_SYSTEM_PROMPT),
                HumanMessage(content=context),
            ]
        )
        return validate_selected_tools(tool_selection, set(tool_specs))
    except Exception as e:
        return create_tool_selection_failure(
            f"Single-protein tool selection failed: {e}"
        )


def select_wildtype_mutant_tools(
    state: AgentState,
    planning_result: PlanningResult,
) -> ToolSelectionOutput:
    tool_specs = load_tool_specs_for_purpose("wildtype_mutant_comparison")
    context = build_tool_selection_context(state, planning_result, tool_specs)
    config = state["config"]

    llm = ChatOpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        model=config.model,
        temperature=0,
        max_retries=0,
        extra_body={"enable_thinking": False},
    )
    structured_llm = llm.with_structured_output(ToolSelectionOutput)

    try:
        tool_selection = structured_llm.invoke(
            [
                SystemMessage(content=WILDTYPE_MUTANT_TOOL_SELECTION_SYSTEM_PROMPT),
                HumanMessage(content=context),
            ]
        )
        return validate_selected_tools(tool_selection, set(tool_specs))
    except Exception as e:
        return create_tool_selection_failure(
            f"Wild-type/mutant tool selection failed: {e}"
        )
