from __future__ import annotations

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from ..builders.planning_context_builder import build_planning_context
from ..data_models.planning import IntentClassificationOutput
from ..prompts.planning_prompt import INTENT_CLASSIFICATION_SYSTEM_PROMPT
from ..state import AgentState


def classify_intent(state: AgentState) -> IntentClassificationOutput:
    config = state["config"]
    context = build_planning_context(state)

    llm = ChatOpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        model=config.model,
        temperature=0,
        extra_body={"enable_thinking": False},
    )
    structured_llm = llm.with_structured_output(IntentClassificationOutput)

    return structured_llm.invoke(
        [
            SystemMessage(content=INTENT_CLASSIFICATION_SYSTEM_PROMPT),
            HumanMessage(content=context),
        ]
    )
