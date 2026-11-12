from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ..conf import AgentConfig
from ..data_models.artifact import Artifact
from ..data_models.planning import (
    ReportStagingRule,
    ToolCall,
    ToolOutputReference,
)
from ..data_models.tool_execution import (
    ManagedInputRepairRecord,
    RecoveryAction,
    ToolCallExecution,
    ToolFailureDiagnosis,
    WorkflowDecisionAction,
)
from ..utils.tool_execution_utils import expand_file_pattern


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
    "expected exactly one artifact_type",
    "missing_structure",
)
_MANAGED_WORKSPACE_MARKERS = (
    "staging would overwrite an existing report",
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
            _MANAGED_WORKSPACE_MARKERS,
            "automatic",
            ["rebuild_managed_workspace", "stop"],
            False,
            "Agent-managed step contents conflict with deterministic report staging.",
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


def _is_within(path: Path, parent: Path) -> bool:
    return path == parent or path.is_relative_to(parent)


def _artifact_matches_staging_side(
    artifact: Artifact,
    rule: ReportStagingRule,
) -> bool:
    if rule.side == "single":
        return artifact.side == "single"
    if rule.side in {"wt", "mut"}:
        return artifact.side in {rule.side, "shared"}
    return artifact.side == "shared"


def _artifact_matches_call_side(artifact: Artifact, tool_call: ToolCall) -> bool:
    if tool_call.side == "single":
        return artifact.side == "single"
    if tool_call.side in {"wt", "mut"}:
        return artifact.side in {tool_call.side, "shared"}
    return artifact.side == "shared"


def prepare_managed_input_repair(
    tool_call: ToolCall,
    available_artifacts: list[Artifact],
    successful_step_indexes: set[int],
    workdir: Path,
) -> tuple[ToolCall, list[ManagedInputRepairRecord]] | None:
    """Safely rebind stale report-staging patterns to registered outputs.

    This deliberately repairs only Agent-managed report staging. It never changes
    user inputs, scientific parameters, output declarations, or step dependencies.
    """

    resolved_workdir = workdir.resolve()
    execution_dir = Path(tool_call.tool_execution_dir).resolve()
    if not _is_within(execution_dir, resolved_workdir):
        return None

    repaired_rules: list[ReportStagingRule] = []
    repairs: list[ManagedInputRepairRecord] = []
    for index, rule in enumerate(tool_call.staging_rules):
        destination_dir = Path(rule.destination_dir).resolve()
        if not _is_within(destination_dir, execution_dir):
            return None
        if expand_file_pattern(rule.source_path_pattern):
            repaired_rules.append(rule)
            continue
        if (
            rule.source_step_index is None
            or rule.source_step_index not in successful_step_indexes
        ):
            return None

        candidates = [
            artifact
            for artifact in available_artifacts
            if artifact.turn_id == tool_call.turn_id
            and artifact.plan_id == tool_call.plan_id
            and artifact.metadata.get("step_index") == rule.source_step_index
            and artifact.artifact_type.endswith("_report_json")
            and _artifact_matches_staging_side(artifact, rule)
            and Path(artifact.path).is_file()
            and _is_within(Path(artifact.path).resolve(), resolved_workdir)
        ]
        if len(candidates) != 1:
            return None
        candidate = candidates[0]
        repaired_path = str(Path(candidate.path).resolve())
        repaired_rules.append(
            rule.model_copy(update={"source_path_pattern": repaired_path})
        )
        repairs.append(
            ManagedInputRepairRecord(
                repair_type="rebind_staging_source",
                staging_rule_index=index,
                source_step_index=rule.source_step_index,
                artifact_type=candidate.artifact_type,
                previous_source_path_pattern=rule.source_path_pattern,
                repaired_source_path=repaired_path,
            )
        )

    repaired_arguments = dict(tool_call.arguments)
    for argument_name, value in tool_call.arguments.items():
        if not isinstance(value, ToolOutputReference):
            continue
        current_matches = [
            artifact
            for artifact in available_artifacts
            if artifact.turn_id == tool_call.turn_id
            and artifact.plan_id == tool_call.plan_id
            and artifact.metadata.get("step_index") == value.source_step_index
            and artifact.artifact_type == value.artifact_type
            and _artifact_matches_call_side(artifact, tool_call)
            and Path(artifact.path).is_file()
            and _is_within(Path(artifact.path).resolve(), resolved_workdir)
        ]
        if len(current_matches) == 1:
            continue
        candidates = [
            artifact
            for artifact in available_artifacts
            if artifact.turn_id == tool_call.turn_id
            and artifact.plan_id == tool_call.plan_id
            and isinstance(artifact.metadata.get("step_index"), int)
            and artifact.metadata["step_index"] in successful_step_indexes
            and artifact.metadata["step_index"] < tool_call.step_index
            and artifact.artifact_type == value.artifact_type
            and _artifact_matches_call_side(artifact, tool_call)
            and Path(artifact.path).is_file()
            and _is_within(Path(artifact.path).resolve(), resolved_workdir)
        ]
        if len(candidates) != 1:
            return None
        candidate = candidates[0]
        repaired_source_step = int(candidate.metadata["step_index"])
        repaired_arguments[argument_name] = value.model_copy(
            update={"source_step_index": repaired_source_step}
        )
        repairs.append(
            ManagedInputRepairRecord(
                repair_type="rebind_output_reference",
                argument_name=argument_name,
                source_step_index=repaired_source_step,
                artifact_type=candidate.artifact_type,
                previous_source_step_index=value.source_step_index,
                repaired_source_path=str(Path(candidate.path).resolve()),
            )
        )

    if not repairs:
        return None
    return (
        tool_call.model_copy(
            update={
                "staging_rules": repaired_rules,
                "arguments": repaired_arguments,
            }
        ),
        repairs,
    )


def authorize_managed_input_repair(
    diagnosis: ToolFailureDiagnosis,
    repair: tuple[ToolCall, list[ManagedInputRepairRecord]] | None,
) -> ToolFailureDiagnosis:
    """Permit deterministic repair only for a verified preparation-path failure."""

    if (
        repair is None
        or diagnosis.category != "path_or_file"
        or diagnosis.failure_stage != "preparation"
        or diagnosis.scientific_change_required
        or not any(
            marker in item.casefold()
            for item in diagnosis.evidence
            for marker in (
                "staging pattern matched no files",
                "expected exactly one artifact_type",
            )
        )
    ):
        return diagnosis
    return ToolFailureDiagnosis.model_validate(
        {
            **diagnosis.model_dump(mode="python"),
            "recoverability": "automatic",
            "reason": (
                diagnosis.reason
                + " A unique registered upstream artifact can safely replace the "
                "stale Agent-managed input reference."
            ),
            "evidence": diagnosis.evidence
            + ["verified unique registered upstream artifact for managed input"],
            "allowed_actions": ["repair_managed_inputs", "stop"],
        }
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
            if "repair_managed_inputs" in diagnosis.allowed_actions:
                action: RecoveryAction = "repair_managed_inputs"
            elif "rebuild_managed_workspace" in diagnosis.allowed_actions:
                action = "rebuild_managed_workspace"
            else:
                action = "retry_same_call"
            return (
                "retry",
                action,
                False,
                (
                    "A bounded Agent-managed input repair and current-step retry is "
                    "permitted by the deterministic policy."
                    if action == "repair_managed_inputs"
                    else (
                        "A bounded cleanup and rebuild of the current Agent-managed "
                        "step workspace is permitted by the deterministic policy."
                        if action == "rebuild_managed_workspace"
                        else (
                            "A bounded same-call retry is permitted by the deterministic "
                            "policy."
                        )
                    )
                ),
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
