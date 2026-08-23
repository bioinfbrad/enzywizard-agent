from __future__ import annotations

from ..services.planning_service import classify_intent
from ..state import AgentState


def planning_node(state: AgentState) -> AgentState:
    intent = classify_intent(state)
    print(f"[INFO] Planning intent classified: purpose_type={intent.purpose_type}")
    print(f"[INFO] Planning intent reason: {intent.purpose_reason}")
    return state
