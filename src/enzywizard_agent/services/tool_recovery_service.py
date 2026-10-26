from __future__ import annotations

from collections.abc import Sequence

from ..conf import AgentConfig
from ..data_models.tool_execution import (
    RecoveryAction,
    ToolCallExecution,
    ToolFailureDiagnosis,
    WorkflowDecisionAction,
)


_TRANSIENT_MARKERS = (
    "temporarily unavailable",
    "temporary failure",
    "connection reset",
    "connection aborted",
    "connection refused",
    "broken pipe",
    "service unavailable",
    "too many requests",
    "rate limit",
    "http 429",
    "http 502",
    "http 503",
    "resource temporarily unavailable",
)
_TIMEOUT_MARKERS = ("timed out", "timeout", "exceeded its")
_RESOURCE_MARKERS = (
    "cuda out of memory",
    "out of memory",
    "cannot allocate memory",
    "disk quota exceeded",
    "no space left on device",
)
_DEPENDENCY_MARKERS = (
    "executable command was not found",
    "command not found",
    "module not found",
    "modulenotfounderror",
)
_PATH_MARKERS = (
    "no such file or directory",
    "file not found",
    "path does not exist",
    "staging pattern matched no files",
    "could not resolve",
    "missing_structure",
)
_INVALID_INPUT_MARKERS = (
    "invalid input",
    "failed to parse",
    "malformed input",
    "unsupported file format",
    "empty input",
)
_SCIENTIFIC_CONFIGURATION_MARKERS = (
    "invalid parameter",
    "unsupported parameter",
    "invalid residue",
    "mutation does not match",
    "amino acid substitution",
)


def _matching_markers(text: str, markers: Sequence[str]) -> list[str]:
    return [marker for marker in markers if marker in text]


def _failure_stage(call: ToolCallExecution) -> str:
    message = (call.error_message or "").casefold()
    if "completed without required outputs" in message:
        return "output_contract"
    if "invalid required output content" in message:
        return "output_validation"
    if not call.command or "could not be started or prepared" in message:
        return "preparation"
    return "subprocess"


def diagnose_tool_failure(call: ToolCallExecution) -> ToolFailureDiagnosis:
    """Classify one failed call without asking an LLM or changing its inputs."""

    if call.execution_status != "failed":
        raise ValueError("only failed tool calls can be diagnosed")
    text = "\n".join(
        value
        for value in (
            call.error_message,
            call.stderr_excerpt,
            call.log_excerpt,
            call.stdout_excerpt,
        )
        if value
    ).casefold()
    stage = _failure_stage(call)

    checks = (
        (
            "timeout",
            _TIMEOUT_MARKERS,
            "confirmation_required",
            ["adjust_runtime_resources", "request_user_input", "stop"],
            False,
            "The call exceeded an execution time boundary.",
        ),
        (
            "resource_exhaustion",
            _RESOURCE_MARKERS,
            "confirmation_required",
            [
                "adjust_runtime_resources",
                "change_scientific_parameters",
                "request_user_input",
                "stop",
            ],
            True,
            "The execution environment reported exhausted compute or storage resources.",
        ),
        (
            "missing_dependency",
            _DEPENDENCY_MARKERS,
            "not_recoverable",
            ["request_user_input", "stop"],
            False,
            "A required executable or software dependency is unavailable.",
        ),
        (
            "missing_output",
            ("completed without required outputs",),
            "automatic",
            ["retry_same_call", "regenerate_upstream", "stop"],
            False,
            "The command returned successfully but did not satisfy its declared output contract.",
        ),
        (
            "invalid_output",
            ("invalid required output content",),
            "not_recoverable",
            ["regenerate_upstream", "request_user_input", "stop"],
            False,
            "A required output exists but failed deterministic content validation.",
        ),
        (
            "path_or_file",
            _PATH_MARKERS,
            "confirmation_required",
            ["locate_artifact", "regenerate_upstream", "request_user_input", "stop"],
            False,
            "A required path or upstream artifact could not be resolved.",
        ),
        (
            "invalid_input",
            _INVALID_INPUT_MARKERS,
            "not_recoverable",
            ["request_user_input", "stop"],
            False,
            "The tool rejected the supplied input or could not parse it.",
        ),
        (
            "scientific_configuration",
            _SCIENTIFIC_CONFIGURATION_MARKERS,
            "confirmation_required",
            ["change_scientific_parameters", "request_user_input", "stop"],
            True,
            "Recovery would require changing a scientific input or parameter.",
        ),
        (
            "transient_external",
            _TRANSIENT_MARKERS,
            "automatic",
            ["retry_same_call", "stop"],
            False,
            "The failure matches a transient external or transport condition.",
        ),
    )
    for category, markers, recoverability, actions, scientific, reason in checks:
        matches = _matching_markers(text, markers)
        if matches:
            return ToolFailureDiagnosis(
                category=category,
                failure_stage=stage,
                recoverability=recoverability,
                reason=reason,
                evidence=[f"matched marker: {marker}" for marker in matches[:3]],
                allowed_actions=actions,
                scientific_change_required=scientific,
            )

    return ToolFailureDiagnosis(
        category="unknown",
        failure_stage=stage,
        recoverability="not_recoverable",
        reason=(
            "The failure does not match a deterministic recovery rule; automatic "
            "changes would be unsafe."
        ),
        evidence=[(call.error_message or "Unknown tool execution error.")[:500]],
        allowed_actions=["request_user_input", "stop"],
        scientific_change_required=False,
    )


def select_recovery_decision(
    diagnosis: ToolFailureDiagnosis,
    config: AgentConfig | None,
    completed_recovery_attempts: int,
) -> tuple[WorkflowDecisionAction, RecoveryAction, bool, str]:
    """Select one action from a diagnosis; never invent parameter changes."""

    if diagnosis.recoverability == "automatic":
        enabled = config is not None and config.tool_recovery_mode == "safe"
        limit = config.max_tool_recovery_attempts if config is not None else 0
        if enabled and completed_recovery_attempts < limit:
            return (
                "retry",
                "retry_same_call",
                False,
                "A bounded same-call retry is permitted by the deterministic policy.",
            )
        reason = (
            "Safe automatic recovery is disabled."
            if not enabled
            else "The configured recovery-attempt limit has been reached."
        )
        return "stop", "stop", False, reason
    if diagnosis.recoverability == "confirmation_required":
        return (
            "request_confirmation",
            diagnosis.allowed_actions[0],
            True,
            "The available recovery may change resources, inputs, or scientific settings.",
        )
    return (
        "stop",
        "stop",
        False,
        "No deterministic low-risk recovery action is available.",
    )
