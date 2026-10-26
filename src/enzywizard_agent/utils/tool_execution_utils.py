from __future__ import annotations

import glob
import json
import re
import shutil
from pathlib import Path
from typing import Any

from ..data_models.artifact import Artifact, ArtifactSide, CleanedFiles
from ..data_models.planning import ToolCall, ToolOutputReference
from ..data_models.tool_execution import (
    ResolvedToolArgumentValue,
    ToolCallExecution,
    ToolExecution,
)
from .security_utils import redact_text


# =============================================================================
# Text excerpt helpers
# =============================================================================

STDOUT_HEAD_LENGTH = 500
STDOUT_TAIL_LENGTH = 1500
STDERR_TAIL_LENGTH = 2000
LOG_TAIL_LENGTH = 2000
FAILURE_AUDIT_ARTIFACT_TYPES = frozenset(
    {"agent_stdout_log", "agent_stderr_log", "log_file"}
)


def artifacts_for_global_tracking(
    call_execution: ToolCallExecution,
) -> list[Artifact]:
    """Return artifacts that may persist in cross-turn global state.

    Failed scientific outputs remain attached to their ToolCallExecution for
    auditing, but only process/tool logs may enter the global artifact catalog.
    """

    if call_execution.execution_status == "succeeded":
        return list(call_execution.artifacts)
    return [
        artifact
        for artifact in call_execution.artifacts
        if artifact.artifact_type in FAILURE_AUDIT_ARTIFACT_TYPES
    ]


def is_safe_historical_artifact(
    artifact: Artifact,
    tool_executions: list[ToolExecution],
) -> bool:
    """Reject scientific outputs produced by failed calls in older checkpoints."""

    if artifact.artifact_type in FAILURE_AUDIT_ARTIFACT_TYPES:
        return True
    return not any(
        failed_artifact.path == artifact.path
        for execution in tool_executions
        for call_execution in execution.call_executions
        if call_execution.execution_status == "failed"
        for failed_artifact in call_execution.artifacts
    )


def is_agent_process_log(path: Path) -> bool:
    """Return whether a path is an Agent-owned stdout/stderr audit log."""

    name = path.name
    if name in {"stdout.txt", "stderr.txt"}:
        return True
    return (
        name.startswith("stdout_attempt_") or name.startswith("stderr_attempt_")
    ) and name.endswith(".txt")


def create_head_tail_excerpt(text: str, head_length: int, tail_length: int) -> str:
    """保留文本开头和结尾，并在中间内容被省略时加入明确标记。"""

    if len(text) <= head_length + tail_length:
        return text
    omitted_length = len(text) - head_length - tail_length
    return (
        text[:head_length]
        + f"\n... [{omitted_length} characters omitted] ...\n"
        + text[-tail_length:]
    )


def create_tail_excerpt(text: str, maximum_length: int) -> str:
    """从文本尾部保留有限字符，适合错误日志和 stderr。"""

    if len(text) <= maximum_length:
        return text
    omitted_length = len(text) - maximum_length
    return f"... [{omitted_length} characters omitted] ...\n" + text[-maximum_length:]


def read_combined_log_excerpt(log_paths: list[str]) -> str:
    """读取一个调用产生的全部工具日志，并只返回合并文本的尾部。"""

    sections: list[str] = []
    for log_path in log_paths:
        path = Path(log_path)
        try:
            log_text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as error:
            log_text = f"Unable to read log file: {error}"
        sections.append(f"=== {path} ===\n{log_text}")
    return create_tail_excerpt("\n\n".join(sections), LOG_TAIL_LENGTH)


# =============================================================================
# Upstream output reference resolution
# =============================================================================

def find_referenced_artifact(
    reference: ToolOutputReference,
    available_artifacts: list[Artifact],
) -> Artifact:
    """按上游步骤和产物类型定位唯一的真实文件。"""

    matching_artifacts = [
        artifact
        for artifact in available_artifacts
        if artifact.metadata.get("step_index") == reference.source_step_index
        and artifact.artifact_type == reference.artifact_type
    ]
    if len(matching_artifacts) != 1:
        raise ValueError(
            f"Expected exactly one artifact_type={reference.artifact_type!r} from "
            f"step {reference.source_step_index}, found {len(matching_artifacts)}."
        )
    return matching_artifacts[0]


def read_json_document(json_path: str) -> Any:
    """读取上游 JSON 产物，并把文件或格式错误转换为稳定异常。"""

    try:
        return json.loads(Path(json_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Unable to read referenced JSON output {json_path}: {error}"
        ) from error


def validate_json_reference_value(
    value: Any,
    value_label: str,
) -> ResolvedToolArgumentValue:
    """限制 JSON 引用结果为 CLI 参数能够直接表达的简单类型。"""

    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise ValueError(
        f"Referenced JSON value {value_label!r} does not contain a supported CLI value."
    )


def read_json_field(json_path: str, field_name: str) -> ResolvedToolArgumentValue:
    """从上游 JSON 产物读取顶层字段，并限制为 CLI 可接受的简单值。"""

    payload = read_json_document(json_path)

    if not isinstance(payload, dict) or field_name not in payload:
        raise ValueError(
            f"Referenced JSON output {json_path} does not contain field {field_name!r}."
        )
    return validate_json_reference_value(payload[field_name], field_name)


def read_json_path(
    json_path: str,
    value_path: list[str | int],
) -> ResolvedToolArgumentValue:
    """按字典键和列表下标逐层读取嵌套 JSON 值。"""

    value: Any = read_json_document(json_path)
    traversed: list[str] = []
    for segment in value_path:
        traversed.append(str(segment))
        if isinstance(segment, str):
            if not isinstance(value, dict) or segment not in value:
                raise ValueError(
                    f"Referenced JSON output {json_path} does not contain path "
                    f"{value_path!r}; failed at {traversed!r}."
                )
            value = value[segment]
            continue
        if not isinstance(value, list) or segment < 0 or segment >= len(value):
            raise ValueError(
                f"Referenced JSON output {json_path} does not contain path "
                f"{value_path!r}; failed at {traversed!r}."
            )
        value = value[segment]
    return validate_json_reference_value(value, repr(value_path))


def resolve_tool_output_reference(
    reference: ToolOutputReference,
    available_artifacts: list[Artifact],
) -> ResolvedToolArgumentValue:
    """把 ToolOutputReference 替换为真实路径或 JSON 字段值。"""

    artifact = find_referenced_artifact(reference, available_artifacts)
    if reference.json_field is not None:
        return read_json_field(artifact.path, reference.json_field)
    if reference.json_path is not None:
        return read_json_path(artifact.path, reference.json_path)
    if reference.path_selector == "stem":
        return Path(artifact.path).stem
    return artifact.path


def resolve_tool_call_arguments(
    tool_call: ToolCall,
    available_artifacts: list[Artifact],
) -> dict[str, ResolvedToolArgumentValue]:
    """解析一次调用中的全部延迟引用，得到可直接生成 CLI 的参数字典。"""

    resolved_arguments: dict[str, ResolvedToolArgumentValue] = {}
    for argument_name, argument_value in tool_call.arguments.items():
        if isinstance(argument_value, ToolOutputReference):
            resolved_arguments[argument_name] = resolve_tool_output_reference(
                argument_value,
                available_artifacts,
            )
        else:
            resolved_arguments[argument_name] = argument_value
    return resolved_arguments


# =============================================================================
# CLI command construction
# =============================================================================

def get_preferred_cli_flag(argument_spec: dict[str, Any]) -> str:
    """优先使用可读性更高的长参数名，若没有则使用规格中的最后一个参数名。"""

    cli_flags = argument_spec.get("cli_flags", [])
    if not cli_flags:
        raise ValueError(f"Argument {argument_spec.get('name')!r} has no CLI flags.")
    long_flags = [flag for flag in cli_flags if flag.startswith("--")]
    return long_flags[-1] if long_flags else cli_flags[-1]


def append_cli_argument(
    command: list[str],
    argument_spec: dict[str, Any],
    value: ResolvedToolArgumentValue,
) -> None:
    """按照 YAML 的 cli_action 把一个参数安全地追加到 subprocess 参数列表。"""

    cli_flag = get_preferred_cli_flag(argument_spec)
    cli_action = argument_spec.get("cli_action")

    # store_true/store_false 参数只通过“是否出现该开关”表达布尔值，不能再追加
    # 字符串 true/false，否则 argparse 会把它误认为多余参数。
    if cli_action == "store_true":
        if not isinstance(value, bool):
            raise ValueError(f"Argument {argument_spec['name']} must be boolean.")
        if value:
            command.append(cli_flag)
        return
    if cli_action == "store_false":
        if not isinstance(value, bool):
            raise ValueError(f"Argument {argument_spec['name']} must be boolean.")
        if not value:
            command.append(cli_flag)
        return
    if cli_action is not None:
        raise ValueError(
            f"Unsupported cli_action={cli_action!r} for argument {argument_spec['name']}."
        )

    command.append(cli_flag)
    if isinstance(value, list):
        if not value:
            raise ValueError(f"Argument {argument_spec['name']} cannot use an empty list.")
        command.extend(value)
    else:
        command.append(str(value))


def build_tool_command(
    tool_call: ToolCall,
    resolved_arguments: dict[str, ResolvedToolArgumentValue],
    tool_spec: dict[str, Any],
) -> list[str]:
    """根据完整 YAML 规格生成无需 shell 解释的安全命令参数列表。"""

    expected_command_name = tool_spec.get("command_name")
    if tool_call.command_name != expected_command_name:
        raise ValueError(
            f"ToolCall command_name={tool_call.command_name!r} does not match YAML "
            f"command_name={expected_command_name!r}."
        )

    required_specs = tool_spec.get("required_arguments", [])
    optional_specs = tool_spec.get("optional_arguments", [])
    argument_specs = {
        argument_spec["name"]: argument_spec
        for argument_spec in required_specs + optional_specs
    }

    unknown_arguments = sorted(set(resolved_arguments) - set(argument_specs))
    if unknown_arguments:
        raise ValueError(
            f"ToolCall contains arguments not declared by YAML: {unknown_arguments}."
        )
    missing_required_arguments = [
        argument_spec["name"]
        for argument_spec in required_specs
        if argument_spec["name"] not in resolved_arguments
    ]
    if missing_required_arguments:
        raise ValueError(
            f"ToolCall is missing required arguments: {missing_required_arguments}."
        )

    command = [tool_call.command_name]
    # 使用 YAML 中的稳定顺序输出参数，确保日志和测试结果可重复。
    for argument_spec in required_specs + optional_specs:
        argument_name = argument_spec["name"]
        if argument_name in resolved_arguments:
            append_cli_argument(
                command,
                argument_spec,
                resolved_arguments[argument_name],
            )
    return command


# =============================================================================
# Report staging and artifact discovery
# =============================================================================

def expand_file_pattern(path_pattern: str) -> list[Path]:
    """展开绝对 glob，仅返回去重后的真实文件，不把目录登记为产物。"""

    matching_paths = {
        Path(path).resolve()
        for path in glob.glob(path_pattern, recursive=True)
        if Path(path).is_file()
    }
    return sorted(matching_paths, key=str)


def stage_tool_reports(tool_call: ToolCall, successful_step_indexes: set[int]) -> None:
    """执行 integrate/mut-integrate 计划中声明的报告复制规则。"""

    for staging_rule in tool_call.staging_rules:
        if (
            staging_rule.source_step_index is not None
            and staging_rule.source_step_index not in successful_step_indexes
        ):
            raise ValueError(
                f"Staging source step {staging_rule.source_step_index} has not completed "
                "successfully."
            )

        source_paths = expand_file_pattern(staging_rule.source_path_pattern)
        if not source_paths:
            raise ValueError(
                f"Staging pattern matched no files: {staging_rule.source_path_pattern}."
            )

        destination_dir = Path(staging_rule.destination_dir)
        destination_dir.mkdir(parents=True, exist_ok=True)
        for source_path in source_paths:
            destination_path = destination_dir / source_path.name
            if destination_path.exists() and destination_path.resolve() != source_path:
                raise ValueError(
                    f"Staging would overwrite an existing report: {destination_path}."
                )
            if destination_path.resolve() != source_path:
                shutil.copy2(source_path, destination_path)


def create_artifact(
    tool_call: ToolCall,
    artifact_type: str,
    path: Path,
    side: ArtifactSide,
    required: bool,
    expected_path_pattern: str,
) -> Artifact:
    """把一个已确认存在的文件转换为统一 Artifact 记录。"""

    return Artifact(
        turn_id=tool_call.turn_id,
        plan_id=tool_call.plan_id,
        artifact_type=artifact_type,
        path=str(path.resolve()),
        source_tool=tool_call.tool_name,
        side=side,
        metadata={
            "step_index": tool_call.step_index,
            "command_name": tool_call.command_name,
            "required": required,
            "expected_path_pattern": expected_path_pattern,
        },
    )


def discover_tool_artifacts(tool_call: ToolCall) -> tuple[list[Artifact], list[str]]:
    """按计划声明发现实际输出，并返回缺失的必需产物类型。"""

    artifacts: list[Artifact] = []
    missing_required_types: list[str] = []
    reserved_paths: set[Path] = set()

    # 先匹配明确产物，再处理 optional_intermediate_outputs 的递归通配，避免
    # 同一个最终报告同时被登记为“明确产物”和“可选中间文件”。
    ordered_output_artifacts = sorted(
        tool_call.output_artifacts,
        key=lambda output: output.artifact_type == "optional_intermediate_outputs",
    )
    for expected_output in ordered_output_artifacts:
        matching_paths = expand_file_pattern(expected_output.path_pattern)
        if expected_output.artifact_type == "optional_intermediate_outputs":
            matching_paths = [
                path
                for path in matching_paths
                if path not in reserved_paths
                and not is_agent_process_log(path)
            ]
        else:
            reserved_paths.update(matching_paths)

        if expected_output.required and not matching_paths:
            missing_required_types.append(expected_output.artifact_type)
        for matching_path in matching_paths:
            artifacts.append(
                create_artifact(
                    tool_call,
                    expected_output.artifact_type,
                    matching_path,
                    expected_output.side,
                    expected_output.required,
                    expected_output.path_pattern,
                )
            )
    return artifacts, missing_required_types


def validate_tool_output_contents(
    tool_call: ToolCall,
    artifacts: list[Artifact],
) -> None:
    """校验仅凭文件存在性无法保证的工具级必需内容。"""

    if tool_call.tool_name != "enzywizard-mut-clean":
        return

    report_paths = [
        artifact.path
        for artifact in artifacts
        if artifact.artifact_type == "mut_clean_report_json"
    ]
    if len(report_paths) != 1:
        raise ValueError(
            "A successful enzywizard-mut-clean call requires exactly one "
            "mut_clean_report_json artifact."
        )
    cleaned_substitution = read_json_field(
        report_paths[0],
        "cleaned_amino_acid_substitution",
    )
    if not isinstance(cleaned_substitution, str) or not cleaned_substitution.strip():
        raise ValueError(
            "mut-clean report contains an invalid cleaned_amino_acid_substitution."
        )


def create_agent_log_artifacts(
    tool_call: ToolCall,
    stdout_file: Path,
    stderr_file: Path,
) -> list[Artifact]:
    """把 Agent 捕获的完整 stdout 和 stderr 文件加入统一产物追踪。"""

    return [
        create_artifact(
            tool_call,
            "agent_stdout_log",
            stdout_file,
            tool_call.side,
            False,
            str(stdout_file),
        ),
        create_artifact(
            tool_call,
            "agent_stderr_log",
            stderr_file,
            tool_call.side,
            False,
            str(stderr_file),
        ),
    ]


# =============================================================================
# CleanedFiles state update
# =============================================================================

def find_execution_artifact_path(
    call_execution: ToolCallExecution,
    artifact_type: str,
) -> str | None:
    """从一次成功调用中读取指定类型的唯一产物路径。"""

    paths = [
        artifact.path
        for artifact in call_execution.artifacts
        if artifact.artifact_type == artifact_type
    ]
    if not paths:
        return None
    if len(paths) != 1:
        raise ValueError(
            f"Expected at most one {artifact_type} artifact from step "
            f"{call_execution.step_index}, found {len(paths)}."
        )
    return paths[0]


def update_cleaned_files_from_executions(
    cleaned_files: CleanedFiles,
    call_executions: list[ToolCallExecution],
) -> CleanedFiles:
    """用成功 clean/mut-clean 的真实输出更新可跨轮复用的基础文件。"""

    updated_values = cleaned_files.model_dump()
    for call_execution in call_executions:
        if call_execution.execution_status != "succeeded":
            continue

        if call_execution.tool_name == "enzywizard-clean":
            cleaned_cif = find_execution_artifact_path(
                call_execution, "cleaned_structure_cif"
            )
            cleaned_pdb = find_execution_artifact_path(
                call_execution, "cleaned_structure_pdb"
            )
            updated_values.update(
                {
                    "single_clean_output_dir": str(
                        Path(call_execution.stdout_file).parent
                    ),
                    "single_cleaned_structure_file": cleaned_cif or cleaned_pdb,
                    "single_cleaned_cif_file": cleaned_cif,
                    "single_cleaned_pdb_file": cleaned_pdb,
                    "single_cleaned_fasta_file": find_execution_artifact_path(
                        call_execution, "cleaned_sequence_fasta"
                    ),
                    "single_clean_report_file": find_execution_artifact_path(
                        call_execution, "clean_report_json"
                    ),
                }
            )

        elif call_execution.tool_name == "enzywizard-mut-clean":
            wt_cleaned_cif = find_execution_artifact_path(
                call_execution, "wt_cleaned_structure_cif"
            )
            wt_cleaned_pdb = find_execution_artifact_path(
                call_execution, "wt_cleaned_structure_pdb"
            )
            mut_cleaned_cif = find_execution_artifact_path(
                call_execution, "mut_cleaned_structure_cif"
            )
            mut_cleaned_pdb = find_execution_artifact_path(
                call_execution, "mut_cleaned_structure_pdb"
            )
            mut_clean_report = find_execution_artifact_path(
                call_execution, "mut_clean_report_json"
            )
            cleaned_substitution = (
                read_json_field(mut_clean_report, "cleaned_amino_acid_substitution")
                if mut_clean_report is not None
                else None
            )
            if cleaned_substitution is not None and not isinstance(
                cleaned_substitution, str
            ):
                raise ValueError(
                    "cleaned_amino_acid_substitution in mut-clean report must be a string."
                )
            updated_values.update(
                {
                    "mut_clean_output_dir": str(
                        Path(call_execution.stdout_file).parent
                    ),
                    "wt_cleaned_structure_file": wt_cleaned_cif or wt_cleaned_pdb,
                    "wt_cleaned_cif_file": wt_cleaned_cif,
                    "wt_cleaned_pdb_file": wt_cleaned_pdb,
                    "wt_cleaned_fasta_file": find_execution_artifact_path(
                        call_execution, "wt_cleaned_sequence_fasta"
                    ),
                    "mut_cleaned_structure_file": mut_cleaned_cif or mut_cleaned_pdb,
                    "mut_cleaned_cif_file": mut_cleaned_cif,
                    "mut_cleaned_pdb_file": mut_cleaned_pdb,
                    "mut_cleaned_fasta_file": find_execution_artifact_path(
                        call_execution, "mut_cleaned_sequence_fasta"
                    ),
                    "mut_clean_report_file": mut_clean_report,
                    "cleaned_amino_acid_substitution": cleaned_substitution,
                }
            )

    return CleanedFiles.model_validate(updated_values)


def sanitize_error_message(message: str) -> str:
    """把多行底层异常压缩成适合 ToolExecution 汇总的一行文本。"""

    return re.sub(r"\s+", " ", redact_text(message)).strip()
