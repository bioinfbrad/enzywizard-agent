from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ..data_models.runtime import RuntimeStatus
from ..state import AgentState
from ..utils.planning_utils import load_enzywizard_tool_specs
from ..utils.security_utils import privacy_safe_file_record, redact_mapping, redact_url
from ..version import __version__
from .runtime_service import elapsed_session_seconds, ensure_runtime_session, utc_now


MANIFEST_SCHEMA_VERSION = "2"


def get_session_dir(state: AgentState) -> Path:
    return state["config"].workdir / f"session_{state['session_id']}"


@lru_cache(maxsize=4096)
def _sha256_file_cached(path_text: str, size: int, modified_ns: int) -> str | None:
    path = Path(path_text)
    if not path.is_file():
        return None
    digest = sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _sha256_file(path: Path) -> str | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return _sha256_file_cached(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


@lru_cache(maxsize=8)
def _fingerprint_source_tree(directory: Path) -> str:
    digest = sha256()
    for path in sorted(directory.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        try:
            digest.update(path.read_bytes())
        except OSError:
            digest.update(b"unavailable")
    return digest.hexdigest()


def _file_record(path_value: str, session_dir: Path) -> dict[str, Any]:
    path = Path(path_value).expanduser()
    record = privacy_safe_file_record(path, session_dir)
    if path.is_file():
        try:
            record["size_bytes"] = path.stat().st_size
        except OSError:
            record["size_bytes"] = None
    else:
        record["size_bytes"] = None
    record["sha256"] = _sha256_file(path)
    return record


def _input_records(state: AgentState, session_dir: Path) -> dict[str, Any]:
    inputs = state.get("agent_inputs")
    if inputs is None:
        return {}
    output: dict[str, Any] = {}
    for name in ("structure_file", "msa_file", "mut_structure_file", "mut_msa_file"):
        value = getattr(inputs, name)
        output[name] = _file_record(value, session_dir) if value else None
    for name in ("substrate_name", "amino_acid_substitution"):
        value = getattr(inputs, name)
        output[name] = (
            {
                "provided": True,
                "length": len(value),
                "sha256": sha256(value.encode("utf-8")).hexdigest(),
            }
            if value
            else None
        )
    return output


def _artifact_records(state: AgentState, session_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for artifact in state.get("artifacts", []):
        if artifact.path in seen:
            continue
        seen.add(artifact.path)
        record = _file_record(artifact.path, session_dir)
        record.update(
            {
                "turn_id": artifact.turn_id,
                "plan_id": artifact.plan_id,
                "artifact_type": artifact.artifact_type,
                "source_tool": artifact.source_tool,
                "side": artifact.side,
            }
        )
        records.append(record)
    return records


def _execution_records(state: AgentState, session_dir: Path) -> list[dict[str, Any]]:
    """Serialize execution decisions without copying logs or scientific payloads."""

    records: list[dict[str, Any]] = []
    executions = list(state.get("tool_executions", []))
    current = state.get("current_tool_execution")
    if current is not None and all(
        (item.turn_id, item.plan_id) != (current.turn_id, current.plan_id)
        for item in executions
    ):
        executions.append(current)
    for execution in executions:
        records.append(
            {
                "turn_id": execution.turn_id,
                "plan_id": execution.plan_id,
                "execution_status": execution.execution_status,
                "failed_step_index": execution.failed_step_index,
                "summary": execution.summary,
                "calls": [
                    {
                        "step_index": call.step_index,
                        "attempt_index": call.attempt_index,
                        "tool_name": call.tool_name,
                        "side": call.side,
                        "execution_status": call.execution_status,
                        "exit_code": call.exit_code,
                        "duration_seconds": round(call.duration_seconds, 6),
                        "stdout_file": _file_record(
                            call.stdout_file, session_dir
                        )["path"],
                        "stderr_file": _file_record(
                            call.stderr_file, session_dir
                        )["path"],
                        "failure_diagnosis": (
                            call.failure_diagnosis.model_dump(mode="json")
                            if call.failure_diagnosis is not None
                            else None
                        ),
                        "artifacts": [
                            {
                                "artifact_type": artifact.artifact_type,
                                "path": _file_record(artifact.path, session_dir)["path"],
                                "side": artifact.side,
                            }
                            for artifact in call.artifacts
                        ],
                    }
                    for call in execution.call_executions
                ],
                "recovery_attempts": [
                    {
                        **attempt.model_dump(
                            mode="json",
                            exclude={"stdout_file", "stderr_file"},
                        ),
                        "stdout_file": (
                            _file_record(attempt.stdout_file, session_dir)["path"]
                            if attempt.stdout_file
                            else None
                        ),
                        "stderr_file": (
                            _file_record(attempt.stderr_file, session_dir)["path"]
                            if attempt.stderr_file
                            else None
                        ),
                    }
                    for attempt in execution.recovery_attempts
                ],
                "workflow_decisions": [
                    item.model_dump(mode="json")
                    for item in execution.workflow_decisions
                ],
                "ai_failure_assessments": [
                    item.model_dump(mode="json")
                    for item in execution.ai_failure_assessments
                ],
            }
        )
    return records


def _history_records(state: AgentState, session_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for result in state.get("history_query_results", []):
        records.append(
            {
                "turn_id": result.turn_id,
                "query_status": result.query_status,
                "query": result.query.model_dump(mode="json") if result.query else None,
                "resolved_turn_ids": result.resolved_turn_ids,
                "selected_artifacts": [
                    {
                        "turn_id": artifact.turn_id,
                        "plan_id": artifact.plan_id,
                        "artifact_type": artifact.artifact_type,
                        "source_tool": artifact.source_tool,
                        "side": artifact.side,
                        "path": _file_record(artifact.path, session_dir)["path"],
                    }
                    for artifact in result.selected_artifacts
                ],
            }
        )
    return records


def _analysis_records(state: AgentState, session_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    analyses = list(state.get("analysis_results", []))
    current = state.get("current_analysis_result")
    if current is not None and all(item.turn_id != current.turn_id for item in analyses):
        analyses.append(current)
    for result in analyses:
        records.append(
            {
                "turn_id": result.turn_id,
                "response_mode": result.response_mode,
                "analysis_status": result.analysis_status,
                "analysis_route": result.analysis_route,
                "termination_reason": result.termination_reason,
                "used_language_model": result.used_language_model,
                "error_guidance": (
                    {
                        **result.error_guidance.model_dump(
                            mode="json",
                            exclude={"evidence_paths"},
                        ),
                        "evidence_paths": [
                            _file_record(path, session_dir)["path"]
                            for path in result.error_guidance.evidence_paths
                        ],
                    }
                    if result.error_guidance is not None
                    else None
                ),
                "findings": [
                    {
                        "title": finding.title,
                        "finding_kind": finding.finding_kind,
                        "inference_level": finding.inference_level,
                        "evidence_ids": finding.evidence_ids,
                        "statement_sha256": sha256(
                            finding.statement.encode("utf-8")
                        ).hexdigest(),
                    }
                    for finding in result.findings
                ],
                "evidence": [
                    {
                        "evidence_id": evidence.evidence_id,
                        "source_kind": evidence.source_kind,
                        "inference_level": evidence.inference_level,
                        "turn_id": evidence.turn_id,
                        "plan_id": evidence.plan_id,
                        "artifact_type": evidence.artifact_type,
                        "source_tool": evidence.source_tool,
                        "side": evidence.side,
                        "path": (
                            _file_record(evidence.path, session_dir)["path"]
                            if evidence.path
                            else None
                        ),
                        "field_path": evidence.field_path,
                        "source_uri": evidence.source_uri,
                        "source_label": evidence.source_label,
                        "source_version": evidence.source_version,
                        "metadata": evidence.metadata,
                    }
                    for evidence in result.evidence
                ],
            }
        )
    return records


def _safe_parameter_value(name: str, value: Any, session_dir: Path) -> Any:
    """保留可复现参数，同时压缩路径和可能敏感的科学标识。"""

    normalized_name = name.casefold()
    if isinstance(value, BaseModel):
        return _safe_parameter_value(
            name,
            value.model_dump(mode="python"),
            session_dir,
        )
    if isinstance(value, dict):
        return {
            str(key): _safe_parameter_value(str(key), item, session_dir)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _safe_parameter_value(name, item, session_dir)
            for item in value
        ]
    if isinstance(value, str) and any(
        marker in normalized_name
        for marker in ("path", "dir", "file", "pattern")
    ):
        return privacy_safe_file_record(value, session_dir)["path"]
    if isinstance(value, str) and any(
        marker in normalized_name
        for marker in ("substrate", "smiles", "amino_acid_substitution")
    ):
        return {
            "provided": True,
            "length": len(value),
            "sha256": sha256(value.encode("utf-8")).hexdigest(),
        }
    if isinstance(value, str) and len(value) > 512:
        return {
            "length": len(value),
            "sha256": sha256(value.encode("utf-8")).hexdigest(),
        }
    return value


def _plan_records(state: AgentState, session_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    planning_results = list(state.get("planning_results", []))
    current = state.get("current_planning_result")
    if current is not None:
        planning_results.append(current)
    for planning_result in planning_results:
        tool_plan = planning_result.tool_plan
        if tool_plan is None or (tool_plan.turn_id, tool_plan.plan_id) in seen:
            continue
        seen.add((tool_plan.turn_id, tool_plan.plan_id))
        records.append(
            {
                "turn_id": tool_plan.turn_id,
                "plan_id": tool_plan.plan_id,
                "purpose_type": tool_plan.purpose_type,
                "validation": (
                    planning_result.plan_validation.model_dump(mode="json")
                    if planning_result.plan_validation is not None
                    else None
                ),
                "calls": [
                    {
                        "step_index": call.step_index,
                        "tool_name": call.tool_name,
                        "side": call.side,
                        "arguments": {
                            name: _safe_parameter_value(name, value, session_dir)
                            for name, value in call.arguments.items()
                        },
                    }
                    for call in tool_plan.calls
                ],
            }
        )
    return records


@lru_cache(maxsize=1)
def _tool_versions() -> dict[str, str]:
    output: dict[str, str] = {}
    for tool_name in sorted(load_enzywizard_tool_specs()):
        try:
            output[tool_name] = version(tool_name)
        except PackageNotFoundError:
            output[tool_name] = "not-installed"
    return output


def build_session_manifest(state: AgentState) -> dict[str, Any]:
    runtime = ensure_runtime_session(state)
    config = state["config"]
    session_dir = get_session_dir(state)
    package_root = Path(__file__).resolve().parents[1]
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "agent_version": __version__,
        "session_id": state["session_id"],
        "status": runtime.status,
        "stop_reason": runtime.stop_reason,
        "started_at": runtime.started_at,
        "updated_at": utc_now(),
        "elapsed_seconds": round(elapsed_session_seconds(state), 6),
        "current_turn_id": state.get("current_turn_id"),
        "current_plan_id": state.get("current_plan_id"),
        "model": {
            "base_url": redact_url(config.base_url, api_key=config.api_key),
            "model": config.model,
            "temperature": config.temperature,
            "max_completion_tokens": config.max_completion_tokens,
            "reasoning_effort": config.reasoning_effort,
            "extra_body": redact_mapping(
                config.llm_extra_body,
                api_key=config.api_key,
            ),
            "timeout_seconds": config.llm_timeout_seconds,
            "max_retries": config.llm_max_retries,
            "capability_check_enabled": config.check_model_capabilities,
            "ontology_mode": config.ontology_mode,
            "tool_recovery_mode": config.tool_recovery_mode,
            "max_tool_recovery_attempts": config.max_tool_recovery_attempts,
            "failure_diagnosis_mode": config.failure_diagnosis_mode,
        },
        "budgets": {
            "max_turns": config.max_turns,
            "max_llm_calls": config.max_llm_calls,
            "max_input_tokens": config.max_input_tokens,
            "max_output_tokens": config.max_output_tokens,
            "session_timeout_seconds": config.session_timeout_seconds,
            "tool_timeout_seconds": config.tool_timeout_seconds,
        },
        "usage": {
            "llm_call_count": len(runtime.llm_calls),
            "input_tokens": runtime.input_tokens,
            "output_tokens": runtime.output_tokens,
            "calls": [item.model_dump(mode="json") for item in runtime.llm_calls],
        },
        "model_capability": (
            runtime.model_capability.model_dump(mode="json")
            if runtime.model_capability is not None
            else None
        ),
        "errors": [item.model_dump(mode="json") for item in runtime.errors],
        "runtime_metadata": redact_mapping(
            runtime.metadata,
            api_key=config.api_key,
        ),
        "versions": {
            "tools": _tool_versions(),
            "prompt_fingerprint": _fingerprint_source_tree(package_root / "prompts"),
            "schema_fingerprint": _fingerprint_source_tree(package_root / "data_models"),
            "fair_terms_sha256": _sha256_file(
                package_root / "knowledge_base" / "fair_terms.yaml"
            ),
        },
        "inputs": _input_records(state, session_dir),
        "tool_plans": _plan_records(state, session_dir),
        "history_queries": _history_records(state, session_dir),
        "tool_executions": _execution_records(state, session_dir),
        "analyses": _analysis_records(state, session_dir),
        "artifacts": _artifact_records(state, session_dir),
    }


def write_session_manifest(
    state: AgentState,
    *,
    status: RuntimeStatus | None = None,
    stop_reason: str | None = None,
) -> Path:
    """原子写入当前会话事实；该文件不包含可恢复图状态。"""

    runtime = ensure_runtime_session(state)
    if status is not None:
        runtime.status = status
    if stop_reason is not None:
        runtime.stop_reason = stop_reason
    session_dir = get_session_dir(state)
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / "session_manifest.json"
    temporary = session_dir / ".session_manifest.json.tmp"
    payload = build_session_manifest(state)
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path
