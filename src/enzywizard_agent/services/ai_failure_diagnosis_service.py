from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..conf import AgentConfig
from ..data_models.tool_execution import (
    AIFailureAction,
    AIFailureDiagnosis,
    ToolCallExecution,
    ToolFailureDiagnosis,
    WorkflowDecisionAction,
)
from ..prompts.recovery_prompt import TOOL_FAILURE_DIAGNOSIS_SYSTEM_PROMPT
from ..state import AgentState
from ..utils.security_utils import redact_text
from .llm_service import create_chat_model_for_state, invoke_structured


def _summarize_argument(name: str, value: Any) -> dict[str, Any]:
    """Retain diagnostic shape while avoiding unnecessary local path disclosure."""

    normalized_name = name.casefold()
    if isinstance(value, str) and any(
        marker in normalized_name
        for marker in ("path", "file", "dir", "pattern")
    ):
        path = Path(value).expanduser()
        return {
            "kind": "path",
            "suffix": path.suffix.casefold(),
            "exists": path.exists(),
        }
    if isinstance(value, list):
        return {
            "kind": "list",
            "length": len(value),
            "item_types": sorted({type(item).__name__ for item in value}),
        }
    if isinstance(value, str):
        return {
            "kind": "string",
            "length": len(value),
            "value": value[:160],
        }
    return {"kind": type(value).__name__, "value": value}


def build_ai_failure_context(
    state: AgentState,
    call: ToolCallExecution,
    diagnosis: ToolFailureDiagnosis,
    tool_spec: dict[str, Any],
    completed_tools: list[str],
) -> dict[str, Any]:
    """Build a bounded, auditable context for one advisory diagnosis call."""

    config = state.get("config")
    api_key = config.api_key if config is not None else None
    evidence_items = {
        "error_message": redact_text(call.error_message or "", api_key=api_key)[:1200],
        "stderr_excerpt": redact_text(call.stderr_excerpt, api_key=api_key)[-1600:],
        "stdout_excerpt": redact_text(call.stdout_excerpt, api_key=api_key)[-1000:],
        "log_excerpt": redact_text(call.log_excerpt, api_key=api_key)[-1600:],
        "deterministic_diagnosis": diagnosis.model_dump(mode="json"),
        "workflow_context": {
            "user_request": redact_text(
                state.get("current_user_message", ""), api_key=api_key
            )[:1000],
            "step_index": call.step_index,
            "attempt_index": call.attempt_index,
            "completed_tools": completed_tools,
            "available_artifact_types": sorted(
                {
                    artifact.artifact_type
                    for artifact in state.get("current_plan_artifacts", [])
                }
            ),
        },
    }
    return {
        "failed_call": {
            "tool_name": call.tool_name,
            "side": call.side,
            "exit_code": call.exit_code,
            "arguments": {
                name: _summarize_argument(name, value)
                for name, value in call.resolved_arguments.items()
            },
        },
        "tool_contract": {
            "description": str(tool_spec.get("description", ""))[:1000],
            "input_requirements": tool_spec.get("input_file_requirements", []),
            "argument_rules": tool_spec.get("argument_rules", []),
            "required_output_types": [
                item.get("artifact_type")
                for item in tool_spec.get("output_files", [])
                if isinstance(item, dict) and item.get("artifact_type")
            ],
        },
        "evidence_items": evidence_items,
        "safety_constraints": {
            "automatic_actions": ["retry_same_call"],
            "scientific_changes_require_confirmation": True,
            "startup_files_cannot_be_replaced_in_chat": True,
            "arbitrary_commands_are_forbidden": True,
        },
    }


def diagnose_failure_with_llm(
    state: AgentState,
    call: ToolCallExecution,
    diagnosis: ToolFailureDiagnosis,
    tool_spec: dict[str, Any],
    completed_tools: list[str],
) -> AIFailureDiagnosis:
    """Ask the configured model for an advisory explanation and recovery proposal."""

    context = build_ai_failure_context(
        state,
        call,
        diagnosis,
        tool_spec,
        completed_tools,
    )
    return invoke_structured(
        state,
        stage="tool_recovery.ai_diagnosis",
        schema=AIFailureDiagnosis,
        llm=create_chat_model_for_state(state),
        messages=[
            SystemMessage(content=TOOL_FAILURE_DIAGNOSIS_SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(context, ensure_ascii=False, indent=2)),
        ],
        retrieval_context=(
            f"{call.tool_name} {call.error_message or ''} "
            f"{diagnosis.category} {diagnosis.reason}"
        ),
    )


def review_ai_recovery_proposal(
    diagnosis: AIFailureDiagnosis,
    deterministic_diagnosis: ToolFailureDiagnosis,
    config: AgentConfig | None,
    completed_recovery_attempts: int,
) -> tuple[WorkflowDecisionAction, AIFailureAction, bool, str]:
    """Authorize only bounded actions; the model never directly controls execution."""

    action = diagnosis.recommended_action
    if action == "retry_same_call":
        if config is None or config.failure_diagnosis_mode != "hybrid":
            return "stop", "stop", False, "Hybrid AI recovery is disabled."
        if config.tool_recovery_mode != "safe":
            return "stop", "stop", False, "Automatic tool recovery is disabled."
        if completed_recovery_attempts >= config.max_tool_recovery_attempts:
            return "stop", "stop", False, "The recovery-attempt limit was reached."
        if diagnosis.confidence < 0.8:
            return "stop", "stop", False, "AI retry confidence is below 0.80."
        if diagnosis.scientific_change_required:
            return "stop", "stop", False, "Scientific changes cannot be automatic."
        if deterministic_diagnosis.category not in {"unknown", "invalid_output"}:
            return (
                "stop",
                "stop",
                False,
                "AI same-call retry is restricted to unknown or invalid-output failures.",
            )
        return (
            "retry",
            "retry_same_call",
            False,
            "A high-confidence same-call retry passed the deterministic safety policy.",
        )
    if action == "request_parameter_confirmation":
        return (
            "request_confirmation",
            action,
            True,
            "The proposed scientific change requires explicit user confirmation.",
        )
    if action == "request_user_input":
        return (
            "request_confirmation",
            action,
            True,
            "The proposed recovery requires additional user input.",
        )
    if action == "request_restart":
        return (
            "stop",
            action,
            False,
            "The required startup or environment change cannot be applied in this session.",
        )
    return "stop", "stop", False, "The AI diagnosis did not identify a safe recovery."
