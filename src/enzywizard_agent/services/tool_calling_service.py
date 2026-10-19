from __future__ import annotations

import os
import signal
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from ..data_models.artifact import Artifact
from ..data_models.planning import ToolCall, ToolPlan
from ..data_models.tool_execution import (
    ResolvedToolArgumentValue,
    ToolCallExecution,
    ToolExecution,
)
from ..state import AgentState
from ..utils.planning_utils import load_enzywizard_tool_specs
from ..utils.tool_execution_utils import (
    STDERR_TAIL_LENGTH,
    STDOUT_HEAD_LENGTH,
    STDOUT_TAIL_LENGTH,
    build_tool_command,
    create_agent_log_artifacts,
    create_head_tail_excerpt,
    create_tail_excerpt,
    discover_tool_artifacts,
    read_combined_log_excerpt,
    resolve_tool_call_arguments,
    sanitize_error_message,
    stage_tool_reports,
    validate_tool_output_contents,
)
from ..utils.security_utils import redact_text
from .manifest_service import write_session_manifest
from .runtime_service import (
    enforce_session_deadline,
    record_runtime_error,
    remaining_session_seconds,
)
from ..data_models.runtime import ToolTimeoutError


def get_approved_tool_plan(state: AgentState) -> ToolPlan:
    """读取并再次核对用户刚刚批准的当前 ToolPlan。"""

    planning_result = state.get("current_planning_result")
    confirmation = state.get("current_user_confirmation")
    if planning_result is None or planning_result.tool_plan is None:
        raise ValueError("Tool Calling requires a current ToolPlan.")
    if confirmation is None or confirmation.decision != "approved":
        raise ValueError("Tool Calling requires an approved UserConfirmation.")
    if (
        confirmation.turn_id != planning_result.turn_id
        or confirmation.plan_id != planning_result.plan_id
    ):
        raise ValueError("Approved confirmation does not match the current ToolPlan.")
    return planning_result.tool_plan


def write_process_streams(
    execution_dir: Path,
    stdout_text: str,
    stderr_text: str,
    *,
    api_key: str | None = None,
) -> tuple[Path, Path]:
    """把 subprocess 的完整标准输出和错误输出保存到当前调用目录。"""

    stdout_file = execution_dir / "stdout.txt"
    stderr_file = execution_dir / "stderr.txt"
    stdout_file.write_text(redact_text(stdout_text, api_key=api_key), encoding="utf-8")
    stderr_file.write_text(redact_text(stderr_text, api_key=api_key), encoding="utf-8")
    return stdout_file, stderr_file


def create_failed_call_execution(
    tool_call: ToolCall,
    command: list[str],
    resolved_arguments: dict[str, ResolvedToolArgumentValue],
    exit_code: int | None,
    duration_seconds: float,
    stdout_text: str,
    stderr_text: str,
    stdout_file: Path,
    stderr_file: Path,
    artifacts: list[Artifact],
    error_message: str,
    *,
    api_key: str | None = None,
) -> ToolCallExecution:
    """将任一执行阶段错误转换为字段完整、可交给 Analysis 的失败记录。"""

    log_files = sorted(
        artifact.path
        for artifact in artifacts
        if artifact.artifact_type == "log_file"
    )
    return ToolCallExecution(
        turn_id=tool_call.turn_id,
        plan_id=tool_call.plan_id,
        step_index=tool_call.step_index,
        tool_name=tool_call.tool_name,
        side=tool_call.side,
        execution_status="failed",
        command=command,
        resolved_arguments=resolved_arguments,
        exit_code=exit_code,
        duration_seconds=duration_seconds,
        stdout_file=str(stdout_file.resolve()),
        stderr_file=str(stderr_file.resolve()),
        log_files=log_files,
        stdout_excerpt=create_head_tail_excerpt(
            stdout_text,
            STDOUT_HEAD_LENGTH,
            STDOUT_TAIL_LENGTH,
        ),
        stderr_excerpt=create_tail_excerpt(stderr_text, STDERR_TAIL_LENGTH),
        log_excerpt=redact_text(
            read_combined_log_excerpt(log_files),
            api_key=api_key,
        ),
        artifacts=artifacts,
        error_message=sanitize_error_message(error_message),
    )


def execute_tool_call(
    state: AgentState,
    tool_call: ToolCall,
    tool_spec: dict[str, Any],
    available_artifacts: list[Artifact],
    successful_step_indexes: set[int],
) -> ToolCallExecution:
    """准备、执行并验证一条 ToolCall，所有运行错误均返回失败记录。"""

    execution_dir = Path(tool_call.tool_execution_dir)
    execution_dir.mkdir(parents=True, exist_ok=True)
    stdout_text = ""
    stderr_text = ""
    command: list[str] = []
    resolved_arguments: dict[str, ResolvedToolArgumentValue] = {}
    exit_code: int | None = None
    artifacts: list[Artifact] = []
    started_at = time.monotonic()
    config = state.get("config")
    api_key = config.api_key if config is not None else None

    try:
        # 整合工具必须在命令启动前把本轮报告复制到计划指定的输入目录。
        stage_tool_reports(tool_call, successful_step_indexes)
        resolved_arguments = resolve_tool_call_arguments(
            tool_call,
            available_artifacts,
        )
        command = build_tool_command(tool_call, resolved_arguments, tool_spec)

        # 提前检查入口命令可以产生比 subprocess 的 FileNotFoundError 更明确的错误。
        if shutil.which(command[0]) is None:
            raise ValueError(f"Executable command was not found on PATH: {command[0]}.")

        # 每个工具使用独立进程组。这样超时或 Ctrl-C 时可同时终止工具及其子进程，
        # 避免外部科学程序留在后台继续占用 CPU/GPU。
        configured_timeout = config.tool_timeout_seconds if config is not None else None
        session_remaining = remaining_session_seconds(state)
        timeout_candidates = [
            value
            for value in (configured_timeout, session_remaining)
            if value is not None
        ]
        effective_timeout = min(timeout_candidates) if timeout_candidates else None
        if effective_timeout is not None:
            effective_timeout = max(0.001, effective_timeout)
        process = subprocess.Popen(
            command,
            cwd=execution_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
            start_new_session=True,
        )
        try:
            stdout_text, stderr_text = process.communicate(timeout=effective_timeout)
        except subprocess.TimeoutExpired:
            _terminate_process_group(process)
            stdout_text, stderr_text = process.communicate()
            runtime = state.get("runtime_session")
            if runtime is not None:
                runtime.status = "timed_out"
                runtime.stop_reason = "tool_timeout"
            raise ToolTimeoutError(
                f"{tool_call.tool_name} exceeded its {effective_timeout:.1f}-second timeout."
            )
        except KeyboardInterrupt:
            _terminate_process_group(process)
            stdout_text, stderr_text = process.communicate()
            stdout_file, stderr_file = write_process_streams(
                execution_dir,
                stdout_text or "",
                stderr_text or "",
                api_key=api_key,
            )
            artifacts = create_agent_log_artifacts(
                tool_call,
                stdout_file,
                stderr_file,
            )
            state.setdefault("artifacts", []).extend(artifacts)
            state["current_plan_artifacts"] = list(artifacts)
            record_runtime_error(
                state,
                stage=f"tool.{tool_call.tool_name}",
                error="Tool execution was cancelled by the user.",
                category="tool",
            )
            if state.get("config") is not None:
                write_session_manifest(
                    state,
                    status="cancelled",
                    stop_reason="user_interrupt",
                )
            raise
        exit_code = process.returncode
        stdout_text = redact_text(stdout_text or "", api_key=api_key)
        stderr_text = redact_text(stderr_text or "", api_key=api_key)
    except Exception as error:
        safe_error = redact_text(error, api_key=api_key)
        stdout_text = redact_text(stdout_text, api_key=api_key)
        stderr_text = f"{type(error).__name__}: {safe_error}"
        stdout_file, stderr_file = write_process_streams(
            execution_dir,
            stdout_text,
            stderr_text,
            api_key=api_key,
        )
        artifacts = create_agent_log_artifacts(
            tool_call,
            stdout_file,
            stderr_file,
        )
        return create_failed_call_execution(
            tool_call,
            command,
            resolved_arguments,
            exit_code,
            time.monotonic() - started_at,
            stdout_text,
            stderr_text,
            stdout_file,
            stderr_file,
            artifacts,
            f"{tool_call.tool_name} could not be started or prepared: {safe_error}",
            api_key=api_key,
        )

    stdout_file, stderr_file = write_process_streams(
        execution_dir,
        stdout_text,
        stderr_text,
        api_key=api_key,
    )
    discovered_artifacts, missing_required_types = discover_tool_artifacts(tool_call)
    artifacts = discovered_artifacts + create_agent_log_artifacts(
        tool_call,
        stdout_file,
        stderr_file,
    )

    if exit_code != 0:
        return create_failed_call_execution(
            tool_call,
            command,
            resolved_arguments,
            exit_code,
            time.monotonic() - started_at,
            stdout_text,
            stderr_text,
            stdout_file,
            stderr_file,
            artifacts,
            f"{tool_call.tool_name} exited with code {exit_code}.",
            api_key=api_key,
        )
    if missing_required_types:
        missing_labels = ", ".join(sorted(missing_required_types))
        return create_failed_call_execution(
            tool_call,
            command,
            resolved_arguments,
            exit_code,
            time.monotonic() - started_at,
            stdout_text,
            stderr_text,
            stdout_file,
            stderr_file,
            artifacts,
            f"{tool_call.tool_name} completed without required outputs: "
            f"{missing_labels}.",
            api_key=api_key,
        )

    try:
        validate_tool_output_contents(tool_call, artifacts)
    except ValueError as error:
        return create_failed_call_execution(
            tool_call,
            command,
            resolved_arguments,
            exit_code,
            time.monotonic() - started_at,
            stdout_text,
            stderr_text,
            stdout_file,
            stderr_file,
            artifacts,
            f"{tool_call.tool_name} produced invalid required output content: {error}",
            api_key=api_key,
        )

    log_files = sorted(
        artifact.path
        for artifact in artifacts
        if artifact.artifact_type == "log_file"
    )
    return ToolCallExecution(
        turn_id=tool_call.turn_id,
        plan_id=tool_call.plan_id,
        step_index=tool_call.step_index,
        tool_name=tool_call.tool_name,
        side=tool_call.side,
        execution_status="succeeded",
        command=command,
        resolved_arguments=resolved_arguments,
        exit_code=exit_code,
        duration_seconds=time.monotonic() - started_at,
        stdout_file=str(stdout_file.resolve()),
        stderr_file=str(stderr_file.resolve()),
        log_files=log_files,
        stdout_excerpt=create_head_tail_excerpt(
            stdout_text,
            STDOUT_HEAD_LENGTH,
            STDOUT_TAIL_LENGTH,
        ),
        stderr_excerpt=create_tail_excerpt(stderr_text, STDERR_TAIL_LENGTH),
        log_excerpt=redact_text(
            read_combined_log_excerpt(log_files),
            api_key=api_key,
        ),
        artifacts=artifacts,
        error_message=None,
    )


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    """先温和、再强制终止一个工具进程组。"""

    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def execute_approved_tool_plan(state: AgentState) -> ToolExecution:
    """串行执行已批准计划，并在首个失败调用处停止。"""

    tool_plan = get_approved_tool_plan(state)
    tool_specs = load_enzywizard_tool_specs()
    call_executions: list[ToolCallExecution] = []
    available_artifacts: list[Artifact] = []
    successful_step_indexes: set[int] = set()

    for tool_call in tool_plan.calls:
        enforce_session_deadline(state, f"tool.{tool_call.tool_name}")
        if tool_call.tool_name not in tool_specs:
            raise ValueError(
                f"ToolPlan contains an unknown tool: {tool_call.tool_name}."
            )
        call_execution = execute_tool_call(
            state,
            tool_call,
            tool_specs[tool_call.tool_name],
            available_artifacts,
            successful_step_indexes,
        )
        call_executions.append(call_execution)
        available_artifacts.extend(call_execution.artifacts)

        # 逐步登记已产生文件，使中断时的 manifest 能反映真实部分结果。
        state["current_plan_artifacts"] = list(available_artifacts)
        known_paths = {artifact.path for artifact in state.setdefault("artifacts", [])}
        state["artifacts"].extend(
            artifact
            for artifact in call_execution.artifacts
            if artifact.path not in known_paths
        )
        if call_execution.execution_status == "failed":
            error_message = (
                call_execution.error_message or "Unknown tool execution error."
            )
            runtime = state.get("runtime_session")
            category = (
                "timeout"
                if runtime is not None
                and runtime.status == "timed_out"
                and runtime.stop_reason == "tool_timeout"
                else "tool"
            )
            record_runtime_error(
                state,
                stage=f"tool.{tool_call.tool_name}",
                error=error_message,
                category=category,
            )
            if state.get("config") is not None:
                write_session_manifest(state)
            completed_count = len(successful_step_indexes)
            return ToolExecution(
                turn_id=tool_plan.turn_id,
                plan_id=tool_plan.plan_id,
                execution_status="failed",
                call_executions=call_executions,
                failed_step_index=tool_call.step_index,
                error_messages=[error_message],
                summary=(
                    f"Tool execution stopped at step {tool_call.step_index} "
                    f"({tool_call.tool_name}); {completed_count} of "
                    f"{len(tool_plan.calls)} calls completed successfully."
                ),
            )
        if state.get("config") is not None:
            write_session_manifest(state)
        successful_step_indexes.add(tool_call.step_index)

    return ToolExecution(
        turn_id=tool_plan.turn_id,
        plan_id=tool_plan.plan_id,
        execution_status="succeeded",
        call_executions=call_executions,
        failed_step_index=None,
        error_messages=[],
        summary=f"Executed all {len(tool_plan.calls)} tool calls successfully.",
    )
