from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from ..data_models.artifact import Artifact
from ..data_models.planning import (
    ParameterExtractionOutput,
    ReportStagingRule,
    ToolCall,
    ToolCallSide,
    ToolOutputArtifact,
    ToolOutputPathSelector,
    ToolOutputReference,
    ToolPlan,
    PlanningResult,
)
from ..state import AgentState
from .planning_utils import load_enzywizard_tool_specs


# 单项工具采用固定顺序，LLM 返回顺序不会直接控制执行顺序。substrate 必须在
# dock 前执行，interaction 位于 dock 后，整合工具始终最后执行。
TOOL_EXECUTION_ORDER = [
    "enzywizard-clean",
    "enzywizard-mut-clean",
    "enzywizard-substrate",
    "enzywizard-aaprops",
    "enzywizard-hydrocluster",
    "enzywizard-energy",
    "enzywizard-flexibility",
    "enzywizard-disorder",
    "enzywizard-conservation",
    "enzywizard-embedding",
    "enzywizard-pocket",
    "enzywizard-dock",
    "enzywizard-interaction",
    "enzywizard-integrate",
    "enzywizard-batch",
    "enzywizard-mut-integrate",
    "enzywizard-mut-batch",
]


# 这些可选参数虽然出现在 CLI 中，但它们来自 AgentInputs、CleanedFiles 或上游
# 工具产物，不能由用户通过自然语言覆盖。
PROGRAM_MANAGED_OPTIONAL_PARAMETERS = {
    "input_msa",
    "wt_input_msa",
    "mut_input_msa",
    "substrate_names",
    "substrate_dir",
}


# fast/accurate 只调整计算量含义明确的参数。物理阈值、相互作用距离和科学定义
# 参数不在这里，防止模型把“更精确”机械解释为任意增大或减小数值。
PARAMETER_PROFILE_VALUES: dict[str, dict[str, dict[str, Any]]] = {
    "enzywizard-energy": {
        "fast": {"minimization_iteration": 25},
        "accurate": {"minimization_iteration": 500},
    },
    "enzywizard-flexibility": {
        "fast": {"n_modes": 10},
        "accurate": {"n_modes": 40},
    },
    "enzywizard-embedding": {
        "fast": {"model_name": "esm2_t6_8M_UR50D"},
        "accurate": {"model_name": "esm2_t30_150M_UR50D"},
    },
    "enzywizard-substrate": {
        "fast": {"max_synonyms": 10, "num_confs": 2},
        "accurate": {"max_synonyms": 30, "num_confs": 10},
    },
    "enzywizard-dock": {
        "fast": {
            "max_docking_attempt_num": 10,
            "early_stop": True,
            "exhaustiveness": 4,
        },
        "accurate": {
            "max_docking_attempt_num": 40,
            "early_stop": False,
            "exhaustiveness": 16,
        },
    },
    "enzywizard-batch": {
        "fast": {
            "energy_minimization_iteration": 25,
            "flexibility_n_modes": 10,
            "embedding_model_name": "esm2_t6_8M_UR50D",
            "substrate_max_synonyms": 10,
            "substrate_num_confs": 2,
            "dock_max_attempt_num": 10,
            "dock_early_stop": True,
            "dock_exhaustiveness": 4,
        },
        "accurate": {
            "energy_minimization_iteration": 500,
            "flexibility_n_modes": 40,
            "embedding_model_name": "esm2_t30_150M_UR50D",
            "substrate_max_synonyms": 30,
            "substrate_num_confs": 10,
            "dock_max_attempt_num": 40,
            "dock_early_stop": False,
            "dock_exhaustiveness": 16,
        },
    },
    "enzywizard-mut-batch": {
        "fast": {
            "energy_minimization_iteration": 25,
            "flexibility_n_modes": 10,
            "embedding_model_name": "esm2_t6_8M_UR50D",
            "substrate_max_synonyms": 10,
            "substrate_num_confs": 2,
            "dock_max_attempt_num": 10,
            "dock_early_stop": True,
            "dock_exhaustiveness": 4,
        },
        "accurate": {
            "energy_minimization_iteration": 500,
            "flexibility_n_modes": 40,
            "embedding_model_name": "esm2_t30_150M_UR50D",
            "substrate_max_synonyms": 30,
            "substrate_num_confs": 10,
            "dock_max_attempt_num": 40,
            "dock_early_stop": False,
            "dock_exhaustiveness": 16,
        },
    },
}


# 参数提取上下文和程序校验共用同一份约束说明。这里的边界来自17个工具当前
# service 层已有的输入校验，避免 Prompt、Planning 和实际 CLI 对合法值理解不一。
PARAMETER_VALUE_CONSTRAINTS: dict[str, str] = {
    "pH": "Must be between 0 and 14, inclusive.",
    "minimization_iteration": "Must be a positive integer.",
    "energy_minimization_iteration": "Must be a positive integer.",
    "cutoff": "Must be a positive number.",
    "hydrocluster_cutoff": "Must be a positive number.",
    "flexibility_cutoff": "Must be a positive number.",
    "n_modes": "Must be a positive integer.",
    "flexibility_n_modes": "Must be a positive integer.",
    "window_size": "Must be an odd integer from 3 to 50, inclusive.",
    "disorder_window_size": "Must be an odd integer from 3 to 50, inclusive.",
    "min_region_length": (
        "Must be an integer from 3 to 50 and no greater than window_size."
    ),
    "disorder_min_region_length": (
        "Must be an integer from 3 to 50 and no greater than disorder_window_size."
    ),
    "min_rad": "Must be at least 1.2 and smaller than max_rad.",
    "max_rad": "Must be greater than min_rad.",
    "min_volume": "Must be greater than 20.",
    "pocket_min_rad": "Must be at least 1.2 and smaller than pocket_max_rad.",
    "pocket_max_rad": "Must be greater than pocket_min_rad.",
    "pocket_min_volume": "Must be greater than 20.",
    "max_synonyms": "Must be an integer from 1 to 200, inclusive.",
    "substrate_max_synonyms": "Must be an integer from 1 to 200, inclusive.",
    "fp_radius": "Must be an integer from 1 to 5, inclusive.",
    "substrate_fp_radius": "Must be an integer from 1 to 5, inclusive.",
    "n_bits": "Must be an integer from 1 to 2048, inclusive.",
    "substrate_n_bits": "Must be an integer from 1 to 2048, inclusive.",
    "num_confs": "Must be an integer from 1 to 20, inclusive.",
    "substrate_num_confs": "Must be an integer from 1 to 20, inclusive.",
    "prune_rms": "Must be greater than 0 and no greater than 5.0.",
    "substrate_prune_rms": "Must be greater than 0 and no greater than 5.0.",
    "max_docking_attempt_num": "Must be an integer from 1 to 100, inclusive.",
    "dock_max_attempt_num": "Must be an integer from 1 to 100, inclusive.",
    "exhaustiveness": "Must be an integer from 1 to 64, inclusive.",
    "dock_exhaustiveness": "Must be an integer from 1 to 64, inclusive.",
    "cpu": "Must be a non-negative integer; 0 lets the docking engine decide.",
    "dock_cpu": "Must be a non-negative integer; 0 lets the docking engine decide.",
    "catalytic_residue": "Must be a positive cleaned residue index.",
    "dock_catalytic_residue": "Must be a positive cleaned residue index.",
    "hbond_bonded_h_min_distance": "Must be between 0.5 and 1.1, inclusive.",
    "hbond_bonded_h_max_distance": "Must be between 1.0 and 1.5, inclusive.",
    "hbond_da_max_distance": "Must be between 2.5 and 4.5, inclusive.",
    "hbond_ha_max_distance": "Must be between 1.5 and 3.0, inclusive.",
    "hbond_angle": "Must be between 60 and 180 degrees, inclusive.",
    "ionic_distance_cutoff": "Must be between 2.0 and 6.0, inclusive.",
    "vdw_mu": "Must be between 0.001 and 0.1, inclusive.",
    "ppstack_center_distance_cutoff": "Must be between 4.0 and 8.0, inclusive.",
    "pication_distance_cutoff": "Must be between 3.0 and 7.0, inclusive.",
    "pication_angle_cutoff": "Must be between 0 and 90 degrees, inclusive.",
    "ssbond_max_distance": "Must be between 1.8 and 3.0, inclusive.",
}


# (最小值, 最大值, 是否包含最小值, 是否包含最大值)。只列出存在明确边界的
# 数值参数；其余数值参数继续使用有限正数这一公共规则。
PARAMETER_NUMERIC_RANGES: dict[str, tuple[float | None, float | None, bool, bool]] = {
    "pH": (0.0, 14.0, True, True),
    "window_size": (3, 50, True, True),
    "disorder_window_size": (3, 50, True, True),
    "min_region_length": (3, 50, True, True),
    "disorder_min_region_length": (3, 50, True, True),
    "min_rad": (1.2, None, True, True),
    "min_volume": (20, None, False, True),
    "pocket_min_rad": (1.2, None, True, True),
    "pocket_min_volume": (20, None, False, True),
    "max_synonyms": (1, 200, True, True),
    "substrate_max_synonyms": (1, 200, True, True),
    "fp_radius": (1, 5, True, True),
    "substrate_fp_radius": (1, 5, True, True),
    "n_bits": (1, 2048, True, True),
    "substrate_n_bits": (1, 2048, True, True),
    "num_confs": (1, 20, True, True),
    "substrate_num_confs": (1, 20, True, True),
    "prune_rms": (0.0, 5.0, False, True),
    "substrate_prune_rms": (0.0, 5.0, False, True),
    "max_docking_attempt_num": (1, 100, True, True),
    "dock_max_attempt_num": (1, 100, True, True),
    "exhaustiveness": (1, 64, True, True),
    "dock_exhaustiveness": (1, 64, True, True),
    "cpu": (0, None, True, True),
    "dock_cpu": (0, None, True, True),
    "hbond_bonded_h_min_distance": (0.5, 1.1, True, True),
    "hbond_bonded_h_max_distance": (1.0, 1.5, True, True),
    "hbond_da_max_distance": (2.5, 4.5, True, True),
    "hbond_ha_max_distance": (1.5, 3.0, True, True),
    "hbond_angle": (60, 180, True, True),
    "ionic_distance_cutoff": (2.0, 6.0, True, True),
    "vdw_mu": (0.001, 0.1, True, True),
    "ppstack_center_distance_cutoff": (4.0, 8.0, True, True),
    "pication_distance_cutoff": (3.0, 7.0, True, True),
    "pication_angle_cutoff": (0, 90, True, True),
    "ssbond_max_distance": (1.8, 3.0, True, True),
}


def sanitize_enzywizard_filename(name: str) -> str:
    """按 EnzyWizard 系列工具的公共规则预测安全文件名。"""

    normalized_name = re.sub(r"[,;:=+\s]+", "_", str(name).strip())
    normalized_name = re.sub(r"[^\w\-.]", "_", normalized_name)
    normalized_name = re.sub(r"_+", "_", normalized_name).strip("_.")
    return normalized_name[:150]


def build_plan_root_dir(state: AgentState, turn_id: int, plan_id: int) -> Path:
    """根据稳定 ID 计算计划根目录，但不创建任何目录。"""

    config = state["config"]
    session_id = state["session_id"]
    return (
        config.workdir
        / f"session_{session_id}"
        / f"turn_{turn_id:03d}"
        / f"plan_{plan_id:03d}"
    ).resolve()


def build_tool_execution_dir(
    plan_root_dir: Path,
    step_index: int,
    tool_name: str,
    side: ToolCallSide,
) -> Path:
    """生成一次工具调用独占的计划目录路径。"""

    tool_label = tool_name.removeprefix("enzywizard-").replace("-", "_")
    side_suffix = f"_{side}" if side in ("wt", "mut") else ""
    return (plan_root_dir / f"{step_index:02d}_{tool_label}{side_suffix}").resolve()


def sort_selected_tool_names(selected_tool_names: list[str]) -> list[str]:
    """按固定工作流顺序排序，并拒绝未知或重复工具。"""

    if len(selected_tool_names) != len(set(selected_tool_names)):
        raise ValueError("Selected tools contain duplicate names before plan generation.")

    order_map = {tool_name: index for index, tool_name in enumerate(TOOL_EXECUTION_ORDER)}
    unknown_tool_names = [
        tool_name for tool_name in selected_tool_names if tool_name not in order_map
    ]
    if unknown_tool_names:
        raise ValueError(
            "Cannot order unknown selected tools: " + ", ".join(unknown_tool_names)
        )
    return sorted(selected_tool_names, key=order_map.__getitem__)


def add_required_tool_dependencies(
    selected_tool_names: list[str],
    substrate_available: bool,
    prepared_substrate_available: bool = False,
) -> tuple[list[str], list[str]]:
    """补齐 docking 和底物相互作用所需的确定性上游工具。"""

    completed_tool_names = list(selected_tool_names)
    added_tool_names: list[str] = []

    # dock 必须读取 substrate 生成的 SDF，因此选择 dock 时必须同时有底物输入
    # 和 substrate 调用。
    if "enzywizard-dock" in completed_tool_names:
        if not substrate_available:
            raise ValueError(
                "enzywizard-dock requires a current substrate_name or SMILES input."
            )
        if (
            "enzywizard-substrate" not in completed_tool_names
            and not prepared_substrate_available
        ):
            completed_tool_names.append("enzywizard-substrate")
            added_tool_names.append("enzywizard-substrate")

    # 同时选择 substrate 和 interaction 表示蛋白-底物相互作用模式；该模式必须
    # 先得到 docked SDF。只有 interaction 时仍保持蛋白内部相互作用模式。
    if (
        "enzywizard-substrate" in completed_tool_names
        and "enzywizard-interaction" in completed_tool_names
        and "enzywizard-dock" not in completed_tool_names
    ):
        completed_tool_names.append("enzywizard-dock")
        added_tool_names.append("enzywizard-dock")

    return sort_selected_tool_names(completed_tool_names), added_tool_names


def get_historical_reuse_policy(state: AgentState) -> str:
    """从 Router 结构化结果读取复用策略，不解析用户自然语言。"""

    routing = state.get("current_request_routing")
    if routing is None or routing.output is None:
        return "none"
    return routing.output.reuse_policy


def get_selected_history_artifacts(state: AgentState) -> list[Artifact]:
    """返回 History Query Planner 已验证的本轮历史产物选择。"""

    history = state.get("current_history_query")
    if history is None or history.query_status != "succeeded":
        return []
    return list(history.selected_artifacts)


def _selected_historical_path(
    state: AgentState,
    artifact_types: tuple[str, ...],
    side: ToolCallSide,
) -> str | None:
    """按类型优先级选择最新轮次中存在的历史文件。"""

    artifacts = [
        artifact
        for artifact in get_selected_history_artifacts(state)
        if artifact.side == side
        and artifact.artifact_type in artifact_types
        and Path(artifact.path).is_file()
    ]
    if not artifacts:
        return None
    latest_turn = max(artifact.turn_id for artifact in artifacts)
    latest = [artifact for artifact in artifacts if artifact.turn_id == latest_turn]
    for artifact_type in artifact_types:
        matches = [
            artifact.path
            for artifact in latest
            if artifact.artifact_type == artifact_type
        ]
        if len(matches) == 1:
            return matches[0]
    return None


def _selected_historical_reports(
    state: AgentState,
    allowed_sides: set[ToolCallSide],
) -> list[Artifact]:
    """提取用户明确选择且可供 integrate staging 的正式 JSON 报告。"""

    selected = [
        artifact
        for artifact in get_selected_history_artifacts(state)
        if artifact.side in allowed_sides
        and artifact.artifact_type.endswith("_report_json")
        and Path(artifact.path).is_file()
    ]
    grouped_paths: dict[tuple[ToolCallSide, str], set[str]] = {}
    for artifact in selected:
        grouped_paths.setdefault(
            (artifact.side, artifact.artifact_type),
            set(),
        ).add(artifact.path)
    ambiguous = [
        f"{side}:{artifact_type}"
        for (side, artifact_type), paths in grouped_paths.items()
        if len(paths) > 1
    ]
    if ambiguous:
        raise ValueError(
            "Historical report reuse requires one report per type and side; "
            "ambiguous selections: " + ", ".join(sorted(ambiguous))
        )
    return selected


def _artifact_path(
    artifacts: list[Artifact],
    artifact_types: tuple[str, ...],
    side: ToolCallSide,
) -> str | None:
    """按类型优先级从已验证产物集合选择唯一文件。"""

    for artifact_type in artifact_types:
        paths = sorted({
            artifact.path
            for artifact in artifacts
            if artifact.side == side
            and artifact.artifact_type == artifact_type
            and Path(artifact.path).is_file()
        })
        if len(paths) == 1:
            return paths[0]
    return None


def _same_resolved_path(left: Any, right: str) -> bool:
    if not isinstance(left, str):
        return False
    return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()


def find_reusable_docking_output(
    state: AgentState,
    side: ToolCallSide,
    cleaned_structure_path: str,
) -> dict[str, Any] | None:
    """查找与当前蛋白侧和底物完全兼容的历史 Dock 输出。

    只有成功调用、相同清理结构、相同原始底物且唯一最佳姿态 SDF 仍存在时才
    允许复用。该保守策略避免把另一侧、另一底物或残缺计划的文件接入 Interaction。
    """

    substrate_name = state["agent_inputs"].substrate_name
    if not substrate_name:
        return None

    current_turn_id = state.get("current_turn_id", 1)
    history = state.get("current_history_query")
    allowed_turn_ids = (
        set(history.resolved_turn_ids)
        if history is not None and history.query_status == "succeeded"
        else None
    )
    selected_paths = {
        artifact.path for artifact in get_selected_history_artifacts(state)
    }
    selected_clean_paths = {
        str(Path(artifact.path).expanduser().resolve())
        for artifact in get_selected_history_artifacts(state)
        if artifact.artifact_type
        in {
            "cleaned_structure_cif",
            "cleaned_structure_pdb",
            "wt_cleaned_structure_cif",
            "wt_cleaned_structure_pdb",
            "mut_cleaned_structure_cif",
            "mut_cleaned_structure_pdb",
        }
    }
    for execution in reversed(state.get("tool_executions", [])):
        if execution.turn_id >= current_turn_id:
            continue
        if allowed_turn_ids is not None and execution.turn_id not in allowed_turn_ids:
            continue
        dock_calls = [
            call
            for call in execution.call_executions
            if call.tool_name == "enzywizard-dock"
            and call.side == side
            and call.execution_status == "succeeded"
        ]
        if len(dock_calls) != 1:
            continue

        substrate_calls = [
            call
            for call in execution.call_executions
            if call.tool_name == "enzywizard-substrate"
            and call.execution_status == "succeeded"
            and call.resolved_arguments.get("substrate_names") == substrate_name
        ]
        if len(substrate_calls) != 1:
            continue

        dock_call = dock_calls[0]
        docked_sdf = [
            artifact
            for artifact in dock_call.artifacts
            if artifact.artifact_type == "docked_substrate_sdf"
            and Path(artifact.path).is_file()
        ]
        if len(docked_sdf) != 1:
            continue
        if selected_paths and docked_sdf[0].path not in selected_paths:
            continue

        dock_structure = dock_call.resolved_arguments.get("input_path")
        if not isinstance(dock_structure, str) or not Path(dock_structure).is_file():
            continue
        # 显式选择历史 Dock 时，该调用实际使用的 cleaned structure 才是后续
        # Interaction 的可信配套输入；未提供结构化选择时维持严格路径比较。
        if selected_clean_paths and str(
            Path(dock_structure).expanduser().resolve()
        ) not in selected_clean_paths:
            continue
        if not selected_paths and not _same_resolved_path(
            dock_structure,
            cleaned_structure_path,
        ):
            continue

        clean_source = next(
            (
                artifact
                for artifact in state.get("artifacts", [])
                if artifact.path == dock_structure
                and artifact.artifact_type
                in {
                    "cleaned_structure_cif",
                    "cleaned_structure_pdb",
                    "wt_cleaned_structure_cif",
                    "wt_cleaned_structure_pdb",
                    "mut_cleaned_structure_cif",
                    "mut_cleaned_structure_pdb",
                }
            ),
            None,
        )
        supporting_clean_artifacts = (
            [
                artifact
                for artifact in state.get("artifacts", [])
                if clean_source is not None
                and artifact.turn_id == clean_source.turn_id
                and artifact.plan_id == clean_source.plan_id
                and artifact.source_tool == clean_source.source_tool
                and Path(artifact.path).is_file()
            ]
            if clean_source is not None
            else []
        )

        report_artifacts: list[Artifact] = [
            artifact
            for call in (substrate_calls[0], dock_call)
            for artifact in call.artifacts
            if artifact.artifact_type in {
                "substrate_report_json",
                "dock_report_json",
            }
            and Path(artifact.path).is_file()
        ]
        return {
            "docked_sdf": docked_sdf[0],
            "report_artifacts": report_artifacts,
            "source_turn_id": execution.turn_id,
            "source_plan_id": execution.plan_id,
            "cleaned_structure_path": dock_structure,
            "supporting_clean_artifacts": supporting_clean_artifacts,
        }
    return None


def find_reusable_substrate_output(state: AgentState) -> dict[str, Any] | None:
    """查找与当前底物输入匹配且文件完整的历史 Substrate 输出。"""

    substrate_name = state["agent_inputs"].substrate_name
    if not substrate_name:
        return None
    history = state.get("current_history_query")
    allowed_turn_ids = (
        set(history.resolved_turn_ids)
        if history is not None and history.query_status == "succeeded"
        else None
    )
    selected_paths = {
        artifact.path for artifact in get_selected_history_artifacts(state)
    }
    for execution in reversed(state.get("tool_executions", [])):
        if execution.turn_id >= state.get("current_turn_id", 1):
            continue
        if allowed_turn_ids is not None and execution.turn_id not in allowed_turn_ids:
            continue
        calls = [
            call
            for call in execution.call_executions
            if call.tool_name == "enzywizard-substrate"
            and call.execution_status == "succeeded"
            and call.resolved_arguments.get("substrate_names") == substrate_name
        ]
        if len(calls) != 1:
            continue
        call = calls[0]
        sdf_artifacts = [
            artifact
            for artifact in call.artifacts
            if artifact.artifact_type == "substrate_sdf"
            and Path(artifact.path).is_file()
        ]
        reports = [
            artifact
            for artifact in call.artifacts
            if artifact.artifact_type == "substrate_report_json"
            and Path(artifact.path).is_file()
        ]
        if not sdf_artifacts or len(reports) != 1:
            continue
        if selected_paths and not any(
            artifact.path in selected_paths for artifact in sdf_artifacts
        ):
            continue
        try:
            payload = json.loads(Path(reports[0].path).read_text(encoding="utf-8"))
            resolved_name = payload["substrates"][0]["substrate_name"]
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, IndexError, TypeError):
            continue
        if not isinstance(resolved_name, str) or not resolved_name.strip():
            continue
        directories = {str(Path(artifact.path).parent) for artifact in sdf_artifacts}
        if len(directories) != 1:
            continue
        return {
            "substrate_name": resolved_name,
            "substrate_dir": directories.pop(),
            "report_artifact": reports[0],
            "source_turn_id": execution.turn_id,
            "source_plan_id": execution.plan_id,
        }
    return None


def get_default_optional_arguments(tool_spec: dict[str, Any]) -> dict[str, Any]:
    """读取 YAML 中非 None 的 CLI 默认值。"""

    return {
        argument["name"]: argument["default"]
        for argument in tool_spec.get("optional_arguments", [])
        if argument.get("default") is not None
    }


def build_output_artifacts(
    tool_name: str,
    execution_dir: Path,
    side: ToolCallSide,
    tool_spec: dict[str, Any],
) -> list[ToolOutputArtifact]:
    """把 YAML 声明的全部输出产物绑定到当前工具执行目录。"""

    output_artifacts: list[ToolOutputArtifact] = []
    for output_spec in tool_spec.get("output_files", []):
        artifact_type = output_spec["artifact_type"]
        filename_pattern = str(output_spec["filename_pattern"])

        # 花括号字段只能在工具运行后由真实蛋白名或底物名确定，计划阶段以 glob
        # 通配符表达；YAML 中已经写成标准 glob 的模式则直接保留。
        filename_pattern = re.sub(r"\{[^{}]+\}", "*", filename_pattern)

        # mut-batch 和 mut-integrate 在调用目录下使用两个输出子目录，因此采用
        # 递归模式。YAML 已明确提供递归 glob 时不能再次添加递归层。
        if filename_pattern.startswith("**/"):
            path_pattern = str(execution_dir / filename_pattern)
        elif tool_name in ("enzywizard-mut-batch", "enzywizard-mut-integrate"):
            path_pattern = str(execution_dir / "**" / filename_pattern)
        else:
            path_pattern = str(execution_dir / filename_pattern)

        # shared 调用也可能产生明确属于 WT 或 MUT 一侧的文件，artifact side
        # 必须按类型细分，后续执行器才能正确登记和暂存。
        output_side = side
        shared_report_types = {
            "mut_clean_report_json",
            "mut_integrate_report_json",
        }
        if artifact_type in shared_report_types:
            output_side = "shared"
        elif side == "shared" and artifact_type.startswith("wt_"):
            output_side = "wt"
        elif side == "shared" and artifact_type.startswith("mut_"):
            output_side = "mut"

        output_artifacts.append(
            ToolOutputArtifact(
                artifact_type=artifact_type,
                path_pattern=path_pattern,
                side=output_side,
                required=bool(output_spec.get("required", True)),
            )
        )
    return output_artifacts


def create_tool_output_reference(
    source_call: ToolCall,
    artifact_type: str,
    json_field: str | None = None,
    json_path: list[str | int] | None = None,
    path_selector: ToolOutputPathSelector = "path",
) -> ToolOutputReference:
    """从上游调用的预期产物中创建可延迟解析的参数引用。"""

    matching_artifacts = [
        artifact
        for artifact in source_call.output_artifacts
        if artifact.artifact_type == artifact_type
    ]
    if len(matching_artifacts) != 1:
        raise ValueError(
            f"Step {source_call.step_index} does not declare exactly one output "
            f"artifact of type {artifact_type}."
        )
    return ToolOutputReference(
        source_step_index=source_call.step_index,
        artifact_type=artifact_type,
        expected_path_pattern=matching_artifacts[0].path_pattern,
        json_field=json_field,
        json_path=json_path,
        path_selector=path_selector,
    )


def create_tool_call(
    state: AgentState,
    turn_id: int,
    plan_id: int,
    step_index: int,
    tool_name: str,
    side: ToolCallSide,
    arguments: dict[str, Any],
    staging_rules: list[ReportStagingRule],
    call_reason: str,
) -> ToolCall:
    """使用完整工具规格构造一个尚未执行的 ToolCall。"""

    tool_specs = load_enzywizard_tool_specs()
    tool_spec = tool_specs[tool_name]
    execution_dir = build_tool_execution_dir(
        build_plan_root_dir(state, turn_id, plan_id),
        step_index,
        tool_name,
        side,
    )

    return ToolCall(
        turn_id=turn_id,
        plan_id=plan_id,
        step_index=step_index,
        tool_name=tool_name,
        command_name=tool_spec["command_name"],
        side=side,
        tool_execution_dir=str(execution_dir),
        arguments=arguments,
        output_artifacts=build_output_artifacts(
            tool_name,
            execution_dir,
            side,
            tool_spec,
        ),
        staging_rules=staging_rules,
        call_reason=call_reason,
    )


def get_preferred_cleaned_structure_path(
    cleaned_cif_file: str | None,
    cleaned_structure_file: str | None,
    cleaned_pdb_file: str | None,
) -> str:
    """按 CIF、首选结构、PDB 的顺序选择可复用结构。"""

    selected_path = cleaned_cif_file or cleaned_structure_file or cleaned_pdb_file
    if not selected_path:
        raise ValueError("A trusted cleaned structure path is required.")
    return selected_path


def get_call_by_tool_and_side(
    calls: list[ToolCall],
    tool_name: str,
    side: ToolCallSide,
) -> ToolCall:
    """在已构建调用中定位唯一的工具和侧别组合。"""

    matching_calls = [
        call for call in calls if call.tool_name == tool_name and call.side == side
    ]
    if len(matching_calls) != 1:
        raise ValueError(
            f"Expected one {tool_name} call for side={side}, found "
            f"{len(matching_calls)}."
        )
    return matching_calls[0]


def build_single_protein_tool_plan(
    state: AgentState,
    turn_id: int,
    plan_id: int,
    selected_tool_names: list[str],
    plan_reason: str,
) -> ToolPlan:
    """构建单蛋白工作流的默认有序调用计划。"""

    agent_inputs = state["agent_inputs"]
    cleaned_files = state["cleaned_files"]
    tool_specs = load_enzywizard_tool_specs()
    reuse_policy = get_historical_reuse_policy(state)
    historical_reports = (
        _selected_historical_reports(state, {"single", "shared"})
        if reuse_policy != "none"
        else []
    )
    reused_historical_input = False
    reusable_docking: dict[str, Any] | None = None
    reusable_substrate: dict[str, Any] | None = None
    if (
        "enzywizard-interaction" in selected_tool_names
        and "enzywizard-clean" not in selected_tool_names
        and reuse_policy != "none"
    ):
        historical_structure = _selected_historical_path(
            state,
            ("cleaned_structure_cif", "cleaned_structure_pdb"),
            "single",
        ) or get_preferred_cleaned_structure_path(
            cleaned_files.single_cleaned_cif_file,
            cleaned_files.single_cleaned_structure_file,
            cleaned_files.single_cleaned_pdb_file,
        )
        reusable_docking = find_reusable_docking_output(
            state,
            "single",
            historical_structure,
        )
        if reusable_docking is not None:
            # 结构化复用策略生效时不重复准备底物和 Dock；Interaction 直接接入
            # 已通过结构、底物、侧别和执行状态验证的历史最佳姿态。
            selected_tool_names = [
                tool_name
                for tool_name in selected_tool_names
                if tool_name not in {"enzywizard-substrate", "enzywizard-dock"}
            ]
            reused_historical_input = True
        elif reuse_policy == "required" and any(
            artifact.artifact_type == "docked_substrate_sdf"
            for artifact in get_selected_history_artifacts(state)
        ):
            raise ValueError(
                "The selected historical docking output is not compatible with "
                "the current structure, substrate, side, or execution record."
            )
    if "enzywizard-dock" in selected_tool_names and reuse_policy != "none":
        reusable_substrate = find_reusable_substrate_output(state)
        if reusable_substrate is not None:
            selected_tool_names = [
                tool_name
                for tool_name in selected_tool_names
                if tool_name != "enzywizard-substrate"
            ]
            reused_historical_input = True
        elif reuse_policy == "required" and any(
            artifact.artifact_type == "substrate_sdf"
            for artifact in get_selected_history_artifacts(state)
        ):
            raise ValueError(
                "The selected historical substrate output is incomplete or does "
                "not match the current substrate input."
            )
    selected_tool_names, added_dependencies = add_required_tool_dependencies(
        selected_tool_names,
        bool(agent_inputs.substrate_name),
        prepared_substrate_available=reusable_substrate is not None,
    )
    calls: list[ToolCall] = []

    # 如果本轮运行 clean，下游结构、FASTA 和报告均保存为上游产物引用；否则
    # 直接使用 CleanedFiles 中已经执行并登记的绝对路径。
    if "enzywizard-clean" in selected_tool_names:
        if not agent_inputs.structure_file:
            raise ValueError("enzywizard-clean requires current structure_file input.")
        step_index = len(calls) + 1
        execution_dir = build_tool_execution_dir(
            build_plan_root_dir(state, turn_id, plan_id),
            step_index,
            "enzywizard-clean",
            "single",
        )
        clean_arguments = get_default_optional_arguments(tool_specs["enzywizard-clean"])
        clean_arguments.update(
            {
                "input_path": agent_inputs.structure_file,
                "output_dir": str(execution_dir),
            }
        )
        clean_call = create_tool_call(
            state,
            turn_id,
            plan_id,
            step_index,
            "enzywizard-clean",
            "single",
            clean_arguments,
            [],
            "Prepare a trusted cleaned structure, FASTA sequence, and clean report.",
        )

        # clean 的四个主要产物名称完全由输入文件 stem 决定，因此使用精确模式
        # 替换 YAML 中的通用模式。
        raw_name = Path(agent_inputs.structure_file).stem
        exact_clean_outputs = {
            "cleaned_structure_cif": sanitize_enzywizard_filename(
                f"cleaned_{raw_name}.cif"
            ),
            "cleaned_structure_pdb": sanitize_enzywizard_filename(
                f"cleaned_{raw_name}.pdb"
            ),
            "cleaned_sequence_fasta": sanitize_enzywizard_filename(
                f"cleaned_{raw_name}.fasta"
            ),
            "clean_report_json": sanitize_enzywizard_filename(
                f"clean_report_{raw_name}.json"
            ),
        }
        for output_artifact in clean_call.output_artifacts:
            exact_name = exact_clean_outputs.get(output_artifact.artifact_type)
            if exact_name is not None:
                output_artifact.path_pattern = str(execution_dir / exact_name)
        calls.append(clean_call)

        # 创建延迟引用，就是把上一级call输入进去，之后执行这个call时，用上一步call的实际输出文件,再将这步ToolOutputReference替换为真实路径
        cleaned_structure: Any = create_tool_output_reference(
            clean_call,
            "cleaned_structure_cif",
        )
        cleaned_fasta: Any = create_tool_output_reference(
            clean_call,
            "cleaned_sequence_fasta",
        )
        clean_report_path = create_tool_output_reference(
            clean_call,
            "clean_report_json",
        )
    else:
        reusable_clean_artifacts = (
            reusable_docking["supporting_clean_artifacts"]
            if reusable_docking is not None
            else []
        )
        selected_structure = (
            (
                reusable_docking["cleaned_structure_path"]
                if reusable_docking is not None
                else None
            )
            or _selected_historical_path(
                state,
                ("cleaned_structure_cif", "cleaned_structure_pdb"),
                "single",
            )
            if reuse_policy != "none"
            else None
        )
        cleaned_structure = selected_structure or get_preferred_cleaned_structure_path(
            cleaned_files.single_cleaned_cif_file,
            cleaned_files.single_cleaned_structure_file,
            cleaned_files.single_cleaned_pdb_file,
        )
        selected_fasta = (
            _artifact_path(
                reusable_clean_artifacts,
                ("cleaned_sequence_fasta",),
                "single",
            )
            or _selected_historical_path(
                state,
                ("cleaned_sequence_fasta",),
                "single",
            )
            if reuse_policy != "none"
            else None
        )
        selected_clean_report = (
            _artifact_path(
                reusable_clean_artifacts,
                ("clean_report_json",),
                "single",
            )
            or _selected_historical_path(
                state,
                ("clean_report_json",),
                "single",
            )
            if reuse_policy != "none"
            else None
        )
        if not (selected_fasta or cleaned_files.single_cleaned_fasta_file):
            raise ValueError("Trusted single-protein cleaned FASTA is missing.")
        if not (selected_clean_report or cleaned_files.single_clean_report_file):
            raise ValueError("Trusted single-protein clean report is missing.")
        cleaned_fasta = selected_fasta or cleaned_files.single_cleaned_fasta_file
        clean_report_path = (
            selected_clean_report or cleaned_files.single_clean_report_file
        )
        reused_historical_input = reused_historical_input or bool(
            selected_structure or selected_fasta or selected_clean_report
        )

    # batch 是单次完整工作流，不再展开单项工具或独立 integrate。
    if "enzywizard-batch" in selected_tool_names:
        if reuse_policy == "required" and not reused_historical_input:
            raise ValueError(
                "Historical output reuse was required, but no compatible selected "
                "historical input can be consumed by enzywizard-batch."
            )
        step_index = len(calls) + 1
        execution_dir = build_tool_execution_dir(
            build_plan_root_dir(state, turn_id, plan_id),
            step_index,
            "enzywizard-batch",
            "single",
        )
        arguments = get_default_optional_arguments(tool_specs["enzywizard-batch"])
        arguments.update(
            {
                "cleaned_input_path": cleaned_structure,
                "output_dir": str(execution_dir),
            }
        )
        if agent_inputs.msa_file:
            arguments["input_msa"] = agent_inputs.msa_file
        if agent_inputs.substrate_name:
            arguments["substrate_names"] = agent_inputs.substrate_name
        calls.append(
            create_tool_call(
                state,
                turn_id,
                plan_id,
                step_index,
                "enzywizard-batch",
                "single",
                arguments,
                [],
                "Run the complete single-protein analysis and integrated graph workflow.",
            )
        )
        return ToolPlan(
            turn_id=turn_id,
            plan_id=plan_id,
            purpose_type="single_protein_analysis",
            summary=f"Ordered single-protein plan with {len(calls)} tool calls.",
            plan_reason=plan_reason,
            calls=calls,
        )

    # 单项工具按固定顺序生成。clean 和 integrate 在各自专门分支处理。
    for tool_name in selected_tool_names:
        if tool_name in ("enzywizard-clean", "enzywizard-integrate"):
            continue

        step_index = len(calls) + 1
        execution_dir = build_tool_execution_dir(
            build_plan_root_dir(state, turn_id, plan_id),
            step_index,
            tool_name,
            "single",
        )
        arguments = get_default_optional_arguments(tool_specs[tool_name])

        if tool_name in (
            "enzywizard-aaprops",
            "enzywizard-hydrocluster",
            "enzywizard-energy",
            "enzywizard-flexibility",
            "enzywizard-disorder",
            "enzywizard-pocket",
        ):
            arguments.update(
                {"input_path": cleaned_structure, "output_dir": str(execution_dir)}
            )
        elif tool_name == "enzywizard-conservation":
            if not agent_inputs.msa_file:
                raise ValueError("enzywizard-conservation requires current msa_file.")
            arguments.update(
                {
                    "input_fasta": cleaned_fasta,
                    "input_msa": agent_inputs.msa_file,
                    "output_dir": str(execution_dir),
                }
            )
        elif tool_name == "enzywizard-embedding":
            arguments.update(
                {"input_fasta": cleaned_fasta, "output_dir": str(execution_dir)}
            )
        elif tool_name == "enzywizard-substrate":
            if not agent_inputs.substrate_name:
                raise ValueError("enzywizard-substrate requires current substrate_name.")
            arguments.update(
                {
                    "substrate_names": agent_inputs.substrate_name,
                    "output_dir": str(execution_dir),
                }
            )
        elif tool_name == "enzywizard-dock":
            substrate_calls = [
                call for call in calls if call.tool_name == "enzywizard-substrate"
            ]
            if substrate_calls:
                substrate_call = get_call_by_tool_and_side(
                    calls,
                    "enzywizard-substrate",
                    "single",
                )
                # Substrate 可能把直接输入的 SMILES 重命名为 smiles1。Dock 使用
                # 报告中的基础名称，自行匹配 name_1.sdf 等全部构象。
                resolved_substrate_name: Any = create_tool_output_reference(
                    substrate_call,
                    "substrate_report_json",
                    json_path=["substrates", 0, "substrate_name"],
                )
                substrate_dir = substrate_call.tool_execution_dir
            elif reusable_substrate is not None:
                resolved_substrate_name = reusable_substrate["substrate_name"]
                substrate_dir = reusable_substrate["substrate_dir"]
            else:
                raise ValueError(
                    "enzywizard-dock requires a current or reusable substrate output."
                )
            arguments.update(
                {
                    "input_path": cleaned_structure,
                    "substrate_names": resolved_substrate_name,
                    "substrate_dir": substrate_dir,
                    "output_dir": str(execution_dir),
                }
            )
        elif tool_name == "enzywizard-interaction":
            arguments.update(
                {"input_path": cleaned_structure, "output_dir": str(execution_dir)}
            )
            if any(call.tool_name == "enzywizard-dock" for call in calls):
                dock_call = get_call_by_tool_and_side(
                    calls,
                    "enzywizard-dock",
                    "single",
                )
                # Interaction 按参数值查找同名 SDF，因此直接使用 Dock 实际
                # 输出文件的 stem，例如 docked_glucose 或 docked_smiles1。
                arguments["substrate_names"] = create_tool_output_reference(
                    dock_call,
                    "docked_substrate_sdf",
                    path_selector="stem",
                )
                arguments["substrate_dir"] = dock_call.tool_execution_dir
            elif reusable_docking is not None:
                docked_sdf = reusable_docking["docked_sdf"]
                arguments["substrate_names"] = Path(docked_sdf.path).stem
                arguments["substrate_dir"] = str(Path(docked_sdf.path).parent)
        else:
            raise ValueError(f"Unsupported single-protein tool in plan: {tool_name}.")

        calls.append(
            create_tool_call(
                state,
                turn_id,
                plan_id,
                step_index,
                tool_name,
                "single",
                arguments,
                [],
                "Run the selected single-protein analysis with program-resolved inputs.",
            )
        )

    if "enzywizard-integrate" in selected_tool_names:
        step_index = len(calls) + 1
        execution_dir = build_tool_execution_dir(
            build_plan_root_dir(state, turn_id, plan_id),
            step_index,
            "enzywizard-integrate",
            "single",
        )
        staging_dir = execution_dir / "input_reports"
        staging_rules: list[ReportStagingRule] = []

        # 当前计划中的每个上游调用只复制 JSON 报告；复用旧 clean 时只复制状态
        # 中登记的 clean report，普通跨轮 artifacts 不参与本轮整合。
        for call in calls:
            staging_rules.append(
                ReportStagingRule(
                    source_step_index=call.step_index,
                    source_path_pattern=str(Path(call.tool_execution_dir) / "*.json"),
                    destination_dir=str(staging_dir),
                    side="single",
                )
            )
        if not any(call.tool_name == "enzywizard-clean" for call in calls):
            staging_rules.append(
                ReportStagingRule(
                    source_step_index=None,
                    source_path_pattern=str(clean_report_path),
                    destination_dir=str(staging_dir),
                    side="single",
                )
            )
        if reusable_docking is not None:
            for artifact in reusable_docking["report_artifacts"]:
                staging_rules.append(
                    ReportStagingRule(
                        source_step_index=None,
                        source_path_pattern=artifact.path,
                        destination_dir=str(staging_dir),
                        side="single",
                    )
                )
        if reusable_substrate is not None:
            artifact = reusable_substrate["report_artifact"]
            staging_rules.append(
                ReportStagingRule(
                    source_step_index=None,
                    source_path_pattern=artifact.path,
                    destination_dir=str(staging_dir),
                    side="single",
                )
            )
        staged_paths = {
            rule.source_path_pattern
            for rule in staging_rules
            if rule.source_step_index is None
        }
        current_report_types = {
            output.artifact_type
            for call in calls
            for output in call.output_artifacts
            if output.artifact_type.endswith("_report_json")
        }
        for artifact in historical_reports:
            if (
                artifact.path in staged_paths
                or artifact.artifact_type in current_report_types
            ):
                continue
            staging_rules.append(
                ReportStagingRule(
                    source_step_index=None,
                    source_path_pattern=artifact.path,
                    destination_dir=str(staging_dir),
                    side="single",
                )
            )
            staged_paths.add(artifact.path)
            reused_historical_input = True

        arguments = get_default_optional_arguments(tool_specs["enzywizard-integrate"])
        arguments.update(
            {"input_dir": str(staging_dir), "output_dir": str(execution_dir)}
        )
        calls.append(
            create_tool_call(
                state,
                turn_id,
                plan_id,
                step_index,
                "enzywizard-integrate",
                "single",
                arguments,
                staging_rules,
                "Integrate all JSON reports generated in the current plan.",
            )
        )

    if reuse_policy == "required" and not reused_historical_input:
        raise ValueError(
            "Historical output reuse was required, but the selected artifacts are "
            "not compatible inputs for the generated workflow."
        )

    dependency_note = (
        " Program added required dependencies: " + ", ".join(added_dependencies) + "."
        if added_dependencies
        else ""
    )
    reuse_note = (
        " Program reused a compatible historical docking output for interaction analysis."
        if reusable_docking is not None
        else ""
    )
    return ToolPlan(
        turn_id=turn_id,
        plan_id=plan_id,
        purpose_type="single_protein_analysis",
        summary=f"Ordered single-protein plan with {len(calls)} tool calls.",
        plan_reason=plan_reason + dependency_note + reuse_note,
        calls=calls,
    )


def build_wildtype_mutant_tool_plan(
    state: AgentState,
    turn_id: int,
    plan_id: int,
    selected_tool_names: list[str],
    plan_reason: str,
) -> ToolPlan:
    """构建 WT/MUT 比较工作流的默认有序调用计划。"""

    agent_inputs = state["agent_inputs"]
    cleaned_files = state["cleaned_files"]
    tool_specs = load_enzywizard_tool_specs()
    reuse_policy = get_historical_reuse_policy(state)
    historical_reports = (
        _selected_historical_reports(state, {"wt", "mut", "shared"})
        if reuse_policy != "none"
        else []
    )
    reused_historical_input = False
    reusable_docking_by_side: dict[str, dict[str, Any]] = {}
    reusable_substrate: dict[str, Any] | None = None
    if (
        "enzywizard-interaction" in selected_tool_names
        and "enzywizard-mut-clean" not in selected_tool_names
        and reuse_policy != "none"
    ):
        historical_structures = {
            "wt": _selected_historical_path(
                state,
                ("wt_cleaned_structure_cif", "wt_cleaned_structure_pdb"),
                "wt",
            ) or get_preferred_cleaned_structure_path(
                cleaned_files.wt_cleaned_cif_file,
                cleaned_files.wt_cleaned_structure_file,
                cleaned_files.wt_cleaned_pdb_file,
            ),
            "mut": _selected_historical_path(
                state,
                ("mut_cleaned_structure_cif", "mut_cleaned_structure_pdb"),
                "mut",
            ) or get_preferred_cleaned_structure_path(
                cleaned_files.mut_cleaned_cif_file,
                cleaned_files.mut_cleaned_structure_file,
                cleaned_files.mut_cleaned_pdb_file,
            ),
        }
        for side, structure_path in historical_structures.items():
            reusable = find_reusable_docking_output(
                state,
                side,
                structure_path,
            )
            if reusable is not None:
                reusable_docking_by_side[side] = reusable
        # 配对工作流必须两侧同时兼容，否则两侧都重新运行，避免比较来源不一致。
        if set(reusable_docking_by_side) == {"wt", "mut"}:
            selected_tool_names = [
                tool_name
                for tool_name in selected_tool_names
                if tool_name not in {"enzywizard-substrate", "enzywizard-dock"}
            ]
            reused_historical_input = True
        else:
            reusable_docking_by_side = {}
            if reuse_policy == "required" and any(
                artifact.artifact_type == "docked_substrate_sdf"
                for artifact in get_selected_history_artifacts(state)
            ):
                raise ValueError(
                    "Historical docking reuse for a paired workflow requires "
                    "compatible successful WT and MUT docking outputs."
                )
    if "enzywizard-dock" in selected_tool_names and reuse_policy != "none":
        reusable_substrate = find_reusable_substrate_output(state)
        if reusable_substrate is not None:
            selected_tool_names = [
                tool_name
                for tool_name in selected_tool_names
                if tool_name != "enzywizard-substrate"
            ]
            reused_historical_input = True
        elif reuse_policy == "required" and any(
            artifact.artifact_type == "substrate_sdf"
            for artifact in get_selected_history_artifacts(state)
        ):
            raise ValueError(
                "The selected historical substrate output is incomplete or does "
                "not match the current substrate input."
            )
    selected_tool_names, added_dependencies = add_required_tool_dependencies(
        selected_tool_names,
        bool(agent_inputs.substrate_name),
        prepared_substrate_available=reusable_substrate is not None,
    )
    calls: list[ToolCall] = []

    if "enzywizard-mut-clean" in selected_tool_names:
        if not (
            agent_inputs.structure_file
            and agent_inputs.mut_structure_file
            and agent_inputs.amino_acid_substitution
        ):
            raise ValueError(
                "enzywizard-mut-clean requires current WT structure, MUT structure, "
                "and amino_acid_substitution inputs."
            )
        step_index = len(calls) + 1
        execution_dir = build_tool_execution_dir(
            build_plan_root_dir(state, turn_id, plan_id),
            step_index,
            "enzywizard-mut-clean",
            "shared",
        )
        arguments = get_default_optional_arguments(tool_specs["enzywizard-mut-clean"])
        arguments.update(
            {
                "wt_input_path": agent_inputs.structure_file,
                "mut_input_path": agent_inputs.mut_structure_file,
                "mutation": agent_inputs.amino_acid_substitution,
                "output_dir": str(execution_dir),
            }
        )
        mut_clean_call = create_tool_call(
            state,
            turn_id,
            plan_id,
            step_index,
            "enzywizard-mut-clean",
            "shared",
            arguments,
            [],
            "Prepare aligned trusted WT/MUT structures and the mutation mapping report.",
        )

        wt_name = Path(agent_inputs.structure_file).stem
        mut_name = Path(agent_inputs.mut_structure_file).stem
        exact_outputs = {
            "wt_cleaned_structure_cif": sanitize_enzywizard_filename(
                f"cleaned_{wt_name}.cif"
            ),
            "wt_cleaned_structure_pdb": sanitize_enzywizard_filename(
                f"cleaned_{wt_name}.pdb"
            ),
            "wt_cleaned_sequence_fasta": sanitize_enzywizard_filename(
                f"cleaned_{wt_name}.fasta"
            ),
            "mut_cleaned_structure_cif": sanitize_enzywizard_filename(
                f"cleaned_{mut_name}.cif"
            ),
            "mut_cleaned_structure_pdb": sanitize_enzywizard_filename(
                f"cleaned_{mut_name}.pdb"
            ),
            "mut_cleaned_sequence_fasta": sanitize_enzywizard_filename(
                f"cleaned_{mut_name}.fasta"
            ),
            "mut_clean_report_json": sanitize_enzywizard_filename(
                f"mut_clean_report_{wt_name}_to_{mut_name}.json"
            ),
        }
        for output_artifact in mut_clean_call.output_artifacts:
            exact_name = exact_outputs.get(output_artifact.artifact_type)
            if exact_name is not None:
                output_artifact.path_pattern = str(execution_dir / exact_name)
        calls.append(mut_clean_call)

        wt_structure: Any = create_tool_output_reference(
            mut_clean_call,
            "wt_cleaned_structure_cif",
        )
        mut_structure: Any = create_tool_output_reference(
            mut_clean_call,
            "mut_cleaned_structure_cif",
        )
        wt_fasta: Any = create_tool_output_reference(
            mut_clean_call,
            "wt_cleaned_sequence_fasta",
        )
        mut_fasta: Any = create_tool_output_reference(
            mut_clean_call,
            "mut_cleaned_sequence_fasta",
        )
        mut_clean_report: Any = create_tool_output_reference(
            mut_clean_call,
            "mut_clean_report_json",
        )
        cleaned_mutation: Any = create_tool_output_reference(
            mut_clean_call,
            "mut_clean_report_json",
            json_field="cleaned_amino_acid_substitution",
        )
    else:
        reusable_wt_clean_artifacts = (
            reusable_docking_by_side["wt"]["supporting_clean_artifacts"]
            if "wt" in reusable_docking_by_side
            else []
        )
        reusable_mut_clean_artifacts = (
            reusable_docking_by_side["mut"]["supporting_clean_artifacts"]
            if "mut" in reusable_docking_by_side
            else []
        )
        selected_wt_structure = (
            (
                reusable_docking_by_side["wt"]["cleaned_structure_path"]
                if "wt" in reusable_docking_by_side
                else None
            )
            or _selected_historical_path(
                state,
                ("wt_cleaned_structure_cif", "wt_cleaned_structure_pdb"),
                "wt",
            )
            if reuse_policy != "none"
            else None
        )
        selected_mut_structure = (
            (
                reusable_docking_by_side["mut"]["cleaned_structure_path"]
                if "mut" in reusable_docking_by_side
                else None
            )
            or _selected_historical_path(
                state,
                ("mut_cleaned_structure_cif", "mut_cleaned_structure_pdb"),
                "mut",
            )
            if reuse_policy != "none"
            else None
        )
        wt_structure = selected_wt_structure or get_preferred_cleaned_structure_path(
            cleaned_files.wt_cleaned_cif_file,
            cleaned_files.wt_cleaned_structure_file,
            cleaned_files.wt_cleaned_pdb_file,
        )
        mut_structure = selected_mut_structure or get_preferred_cleaned_structure_path(
            cleaned_files.mut_cleaned_cif_file,
            cleaned_files.mut_cleaned_structure_file,
            cleaned_files.mut_cleaned_pdb_file,
        )
        selected_wt_fasta = (
            _artifact_path(
                reusable_wt_clean_artifacts,
                ("wt_cleaned_sequence_fasta",),
                "wt",
            )
            or _selected_historical_path(
                state,
                ("wt_cleaned_sequence_fasta",),
                "wt",
            )
            if reuse_policy != "none"
            else None
        )
        selected_mut_fasta = (
            _artifact_path(
                reusable_mut_clean_artifacts,
                ("mut_cleaned_sequence_fasta",),
                "mut",
            )
            or _selected_historical_path(
                state,
                ("mut_cleaned_sequence_fasta",),
                "mut",
            )
            if reuse_policy != "none"
            else None
        )
        selected_mut_clean_report = (
            _artifact_path(
                reusable_wt_clean_artifacts + reusable_mut_clean_artifacts,
                ("mut_clean_report_json",),
                "shared",
            )
            or _selected_historical_path(
                state,
                ("mut_clean_report_json",),
                "shared",
            )
            if reuse_policy != "none"
            else None
        )
        if not (selected_wt_fasta or cleaned_files.wt_cleaned_fasta_file):
            raise ValueError("Trusted WT cleaned FASTA is missing.")
        if not (selected_mut_fasta or cleaned_files.mut_cleaned_fasta_file):
            raise ValueError("Trusted MUT cleaned FASTA is missing.")
        if not (selected_mut_clean_report or cleaned_files.mut_clean_report_file):
            raise ValueError("Trusted mut-clean report is missing.")
        if not cleaned_files.cleaned_amino_acid_substitution:
            raise ValueError("Trusted cleaned amino acid substitution is missing.")
        wt_fasta = selected_wt_fasta or cleaned_files.wt_cleaned_fasta_file
        mut_fasta = selected_mut_fasta or cleaned_files.mut_cleaned_fasta_file
        mut_clean_report = (
            selected_mut_clean_report or cleaned_files.mut_clean_report_file
        )
        cleaned_mutation = cleaned_files.cleaned_amino_acid_substitution
        reused_historical_input = reused_historical_input or bool(
            selected_wt_structure
            or selected_mut_structure
            or selected_wt_fasta
            or selected_mut_fasta
            or selected_mut_clean_report
        )

    if "enzywizard-mut-batch" in selected_tool_names:
        if reuse_policy == "required" and not reused_historical_input:
            raise ValueError(
                "Historical output reuse was required, but no compatible selected "
                "historical input can be consumed by enzywizard-mut-batch."
            )
        step_index = len(calls) + 1
        execution_dir = build_tool_execution_dir(
            build_plan_root_dir(state, turn_id, plan_id),
            step_index,
            "enzywizard-mut-batch",
            "shared",
        )
        arguments = get_default_optional_arguments(tool_specs["enzywizard-mut-batch"])
        arguments.update(
            {
                "wt_cleaned_input_path": wt_structure,
                "mut_cleaned_input_path": mut_structure,
                "cleaned_amino_acid_substitution": cleaned_mutation,
                "wt_output_dir": str(execution_dir / "wt"),
                "mut_output_dir": str(execution_dir / "mut"),
            }
        )
        if agent_inputs.msa_file:
            arguments["wt_input_msa"] = agent_inputs.msa_file
        if agent_inputs.mut_msa_file:
            arguments["mut_input_msa"] = agent_inputs.mut_msa_file
        if agent_inputs.substrate_name:
            arguments["substrate_names"] = agent_inputs.substrate_name
        calls.append(
            create_tool_call(
                state,
                turn_id,
                plan_id,
                step_index,
                "enzywizard-mut-batch",
                "shared",
                arguments,
                [],
                "Run the complete paired WT/MUT analysis and integrated graph workflow.",
            )
        )
        return ToolPlan(
            turn_id=turn_id,
            plan_id=plan_id,
            purpose_type="wildtype_mutant_comparison",
            summary=f"Ordered WT/MUT plan with {len(calls)} tool calls.",
            plan_reason=plan_reason,
            calls=calls,
        )

    # substrate 是两侧共享输入，只生成一次；其余逻辑分析工具分别展开为 WT 和
    # MUT 两次调用，保证双方使用相同参数默认值。
    for tool_name in selected_tool_names:
        if tool_name in ("enzywizard-mut-clean", "enzywizard-mut-integrate"):
            continue

        if tool_name == "enzywizard-substrate":
            step_index = len(calls) + 1
            execution_dir = build_tool_execution_dir(
                build_plan_root_dir(state, turn_id, plan_id),
                step_index,
                tool_name,
                "shared",
            )
            arguments = get_default_optional_arguments(tool_specs[tool_name])
            arguments.update(
                {
                    "substrate_names": agent_inputs.substrate_name,
                    "output_dir": str(execution_dir),
                }
            )
            calls.append(
                create_tool_call(
                    state,
                    turn_id,
                    plan_id,
                    step_index,
                    tool_name,
                    "shared",
                    arguments,
                    [],
                    "Prepare the single substrate shared by WT and MUT analyses.",
                )
            )
            continue

        for side, structure_path, fasta_path, msa_path in (
            ("wt", wt_structure, wt_fasta, agent_inputs.msa_file),
            ("mut", mut_structure, mut_fasta, agent_inputs.mut_msa_file),
        ):
            step_index = len(calls) + 1
            execution_dir = build_tool_execution_dir(
                build_plan_root_dir(state, turn_id, plan_id),
                step_index,
                tool_name,
                side,
            )
            arguments = get_default_optional_arguments(tool_specs[tool_name])

            if tool_name in (
                "enzywizard-aaprops",
                "enzywizard-hydrocluster",
                "enzywizard-energy",
                "enzywizard-flexibility",
                "enzywizard-disorder",
                "enzywizard-pocket",
            ):
                arguments.update(
                    {"input_path": structure_path, "output_dir": str(execution_dir)}
                )
            elif tool_name == "enzywizard-conservation":
                if not msa_path:
                    raise ValueError(
                        f"enzywizard-conservation requires the current {side.upper()} MSA."
                    )
                arguments.update(
                    {
                        "input_fasta": fasta_path,
                        "input_msa": msa_path,
                        "output_dir": str(execution_dir),
                    }
                )
            elif tool_name == "enzywizard-embedding":
                arguments.update(
                    {"input_fasta": fasta_path, "output_dir": str(execution_dir)}
                )
            elif tool_name == "enzywizard-dock":
                substrate_calls = [
                    call
                    for call in calls
                    if call.tool_name == "enzywizard-substrate"
                ]
                if substrate_calls:
                    substrate_call = get_call_by_tool_and_side(
                        calls,
                        "enzywizard-substrate",
                        "shared",
                    )
                    # WT 和 MUT 共享 Substrate 输出。解析后的基础名称既支持普通
                    # 名称，也支持被工具规范化为 smiles1 的直接 SMILES 输入。
                    resolved_substrate_name: Any = create_tool_output_reference(
                        substrate_call,
                        "substrate_report_json",
                        json_path=["substrates", 0, "substrate_name"],
                    )
                    substrate_dir = substrate_call.tool_execution_dir
                elif reusable_substrate is not None:
                    resolved_substrate_name = reusable_substrate["substrate_name"]
                    substrate_dir = reusable_substrate["substrate_dir"]
                else:
                    raise ValueError(
                        "enzywizard-dock requires a current or reusable substrate output."
                    )
                arguments.update(
                    {
                        "input_path": structure_path,
                        "substrate_names": resolved_substrate_name,
                        "substrate_dir": substrate_dir,
                        "output_dir": str(execution_dir),
                    }
                )
            elif tool_name == "enzywizard-interaction":
                arguments.update(
                    {"input_path": structure_path, "output_dir": str(execution_dir)}
                )
                matching_dock_calls = [
                    call
                    for call in calls
                    if call.tool_name == "enzywizard-dock" and call.side == side
                ]
                if matching_dock_calls:
                    dock_call = matching_dock_calls[0]
                    # 每一侧只引用本侧 Dock 的最佳姿态 SDF，避免 WT/MUT 输出
                    # 目录交叉，并把实际 stem 直接交给 Interaction。
                    arguments["substrate_names"] = create_tool_output_reference(
                        dock_call,
                        "docked_substrate_sdf",
                        path_selector="stem",
                    )
                    arguments["substrate_dir"] = dock_call.tool_execution_dir
                elif side in reusable_docking_by_side:
                    docked_sdf = reusable_docking_by_side[side]["docked_sdf"]
                    arguments["substrate_names"] = Path(docked_sdf.path).stem
                    arguments["substrate_dir"] = str(Path(docked_sdf.path).parent)
            else:
                raise ValueError(f"Unsupported WT/MUT side tool in plan: {tool_name}.")

            calls.append(
                create_tool_call(
                    state,
                    turn_id,
                    plan_id,
                    step_index,
                    tool_name,
                    side,
                    arguments,
                    [],
                    f"Run the selected analysis on the {side.upper()} protein side.",
                )
            )

    if "enzywizard-mut-integrate" in selected_tool_names:
        step_index = len(calls) + 1
        execution_dir = build_tool_execution_dir(
            build_plan_root_dir(state, turn_id, plan_id),
            step_index,
            "enzywizard-mut-integrate",
            "shared",
        )
        wt_staging_dir = execution_dir / "wt_input_reports"
        mut_staging_dir = execution_dir / "mut_input_reports"
        staging_rules: list[ReportStagingRule] = []

        for call in calls:
            if call.tool_name == "enzywizard-mut-clean":
                continue
            destination_sides: tuple[ToolCallSide, ...]
            if call.side == "shared":
                # 共享 substrate 报告同时复制到两侧，确保图节点定义一致。
                destination_sides = ("wt", "mut")
            else:
                destination_sides = (call.side,)

            for destination_side in destination_sides:
                destination_dir = (
                    wt_staging_dir if destination_side == "wt" else mut_staging_dir
                )
                staging_rules.append(
                    ReportStagingRule(
                        source_step_index=call.step_index,
                        source_path_pattern=str(
                            Path(call.tool_execution_dir) / "*.json"
                        ),
                        destination_dir=str(destination_dir),
                        side=destination_side,
                    )
                )

        # 历史 substrate/dock 报告与本轮 interaction 报告一起进入配对整合；
        # source_step_index=None 表示来源已经在旧轮次成功完成。
        for side, reusable in reusable_docking_by_side.items():
            destination_dir = wt_staging_dir if side == "wt" else mut_staging_dir
            for artifact in reusable["report_artifacts"]:
                staging_rules.append(
                    ReportStagingRule(
                        source_step_index=None,
                        source_path_pattern=artifact.path,
                        destination_dir=str(destination_dir),
                        side=side,
                    )
                )

        if reusable_substrate is not None:
            artifact = reusable_substrate["report_artifact"]
            for destination_side, destination_dir in (
                ("wt", wt_staging_dir),
                ("mut", mut_staging_dir),
            ):
                staging_rules.append(
                    ReportStagingRule(
                        source_step_index=None,
                        source_path_pattern=artifact.path,
                        destination_dir=str(destination_dir),
                        side=destination_side,
                    )
                )

        staged_paths = {
            (rule.source_path_pattern, rule.destination_dir)
            for rule in staging_rules
            if rule.source_step_index is None
        }
        current_report_types_by_side: dict[ToolCallSide, set[str]] = {
            "wt": set(),
            "mut": set(),
        }
        for call in calls:
            if call.tool_name == "enzywizard-mut-clean":
                continue
            destination_sides = (
                ("wt", "mut") if call.side == "shared" else (call.side,)
            )
            for destination_side in destination_sides:
                current_report_types_by_side[destination_side].update(
                    output.artifact_type
                    for output in call.output_artifacts
                    if output.artifact_type.endswith("_report_json")
                )
        for artifact in historical_reports:
            if artifact.artifact_type == "mut_clean_report_json":
                continue
            destination_sides: tuple[ToolCallSide, ...]
            if artifact.side == "shared":
                destination_sides = ("wt", "mut")
            else:
                destination_sides = (artifact.side,)
            for destination_side in destination_sides:
                if (
                    artifact.artifact_type
                    in current_report_types_by_side[destination_side]
                ):
                    continue
                destination_dir = (
                    wt_staging_dir if destination_side == "wt" else mut_staging_dir
                )
                key = (artifact.path, str(destination_dir))
                if key in staged_paths:
                    continue
                staging_rules.append(
                    ReportStagingRule(
                        source_step_index=None,
                        source_path_pattern=artifact.path,
                        destination_dir=str(destination_dir),
                        side=destination_side,
                    )
                )
                staged_paths.add(key)
                reused_historical_input = True

        arguments = get_default_optional_arguments(
            tool_specs["enzywizard-mut-integrate"]
        )
        arguments.update(
            {
                "mut_clean_report_path": mut_clean_report,
                "wt_input_dir": str(wt_staging_dir),
                "mut_input_dir": str(mut_staging_dir),
                "wt_output_dir": str(execution_dir / "wt"),
                "mut_output_dir": str(execution_dir / "mut"),
            }
        )
        calls.append(
            create_tool_call(
                state,
                turn_id,
                plan_id,
                step_index,
                "enzywizard-mut-integrate",
                "shared",
                arguments,
                staging_rules,
                "Integrate matched WT and MUT reports generated in the current plan.",
            )
        )

    if reuse_policy == "required" and not reused_historical_input:
        raise ValueError(
            "Historical output reuse was required, but the selected artifacts are "
            "not compatible inputs for the generated paired workflow."
        )

    dependency_note = (
        " Program added required dependencies: " + ", ".join(added_dependencies) + "."
        if added_dependencies
        else ""
    )
    reuse_note = (
        " Program reused compatible historical WT and MUT docking outputs for "
        "interaction analysis."
        if reusable_docking_by_side
        else ""
    )
    return ToolPlan(
        turn_id=turn_id,
        plan_id=plan_id,
        purpose_type="wildtype_mutant_comparison",
        summary=f"Ordered WT/MUT plan with {len(calls)} tool calls.",
        plan_reason=plan_reason + dependency_note + reuse_note,
        calls=calls,
    )


def build_default_tool_plan(
    state: AgentState,
    planning_result: PlanningResult,
) -> ToolPlan:
    """根据已验证的工具选择生成完整默认 ToolPlan。"""

    tool_selection = planning_result.tool_selection
    if (
        tool_selection is None
        or tool_selection.tool_selection_status != "tool_selection_succeeded"
    ):
        raise ValueError("A successful tool selection is required to build ToolPlan.")
    if planning_result.purpose_type == "single_protein_analysis":
        return build_single_protein_tool_plan(
            state,
            planning_result.turn_id,
            planning_result.plan_id,
            tool_selection.selected_tools,
            tool_selection.selection_reason,
        )
    if planning_result.purpose_type == "wildtype_mutant_comparison":
        return build_wildtype_mutant_tool_plan(
            state,
            planning_result.turn_id,
            planning_result.plan_id,
            tool_selection.selected_tools,
            tool_selection.selection_reason,
        )
    raise ValueError("ToolPlan cannot be built for purpose_type=other.")


def get_editable_parameter_specs(tool_name: str) -> dict[str, dict[str, Any]]:
    """返回允许 LLM 修改的可选参数规格。"""

    tool_spec = load_enzywizard_tool_specs()[tool_name]
    return {
        argument["name"]: argument
        for argument in tool_spec.get("optional_arguments", [])
        if argument["name"] not in PROGRAM_MANAGED_OPTIONAL_PARAMETERS
    }


def convert_parameter_value(value: Any, parameter_spec: dict[str, Any]) -> Any:
    """按照 YAML python_type 转换 LLM 返回值，并处理枚举大小写。"""

    python_type = parameter_spec["python_type"]
    if python_type == "bool":
        if isinstance(value, bool):
            converted_value = value
        elif isinstance(value, str) and value.strip().casefold() in ("true", "false"):
            converted_value = value.strip().casefold() == "true"
        else:
            raise ValueError(f"Expected a boolean value, received {value!r}.")
    elif python_type == "int":
        if isinstance(value, bool):
            raise ValueError(f"Expected an integer value, received {value!r}.")
        converted_value = int(value)
        if isinstance(value, float) and not value.is_integer():
            raise ValueError(f"Expected an integer value, received {value!r}.")
    elif python_type == "float":
        if isinstance(value, bool):
            raise ValueError(f"Expected a numeric value, received {value!r}.")
        converted_value = float(value)
        if not math.isfinite(converted_value):
            raise ValueError("Numeric parameter values must be finite.")
    elif python_type == "str":
        converted_value = str(value).strip()
        if not converted_value:
            raise ValueError("String parameter values must not be empty.")
    else:
        raise ValueError(f"Unsupported parameter python_type: {python_type}.")

    allowed_values = parameter_spec.get("allowed_values")
    if allowed_values:
        normalized_allowed_values = {
            str(allowed_value).casefold(): allowed_value
            for allowed_value in allowed_values
        }
        normalized_value = str(converted_value).casefold()
        if normalized_value not in normalized_allowed_values:
            raise ValueError(
                f"Value {converted_value!r} is not in allowed values: {allowed_values}."
            )
        converted_value = normalized_allowed_values[normalized_value]
    return converted_value


def validate_parameter_value(parameter_name: str, value: Any) -> None:
    """执行与 EnzyWizard service 层一致的单参数数值边界校验。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return

    if not math.isfinite(float(value)):
        raise ValueError(f"{parameter_name} must be finite.")

    numeric_range = PARAMETER_NUMERIC_RANGES.get(parameter_name)
    if numeric_range is not None:
        minimum, maximum, minimum_inclusive, maximum_inclusive = numeric_range
        below_minimum = minimum is not None and (
            value < minimum if minimum_inclusive else value <= minimum
        )
        above_maximum = maximum is not None and (
            value > maximum if maximum_inclusive else value >= maximum
        )
        if below_minimum or above_maximum:
            rule = PARAMETER_VALUE_CONSTRAINTS[parameter_name]
            raise ValueError(f"{parameter_name} is invalid. {rule}")
        return

    # 没有独立范围表的计算参数沿用各工具普遍采用的有限正数要求。
    if value <= 0:
        raise ValueError(f"{parameter_name} must be greater than 0.")


def validate_call_parameter_relationships(call: ToolCall) -> None:
    """校验同一调用内的参数依赖、互斥和上下界关系。"""

    arguments = call.arguments
    minimum_maximum_pairs = (
        ("min_rad", "max_rad"),
        ("pocket_min_rad", "pocket_max_rad"),
        ("hbond_bonded_h_min_distance", "hbond_bonded_h_max_distance"),
    )
    for minimum_name, maximum_name in minimum_maximum_pairs:
        if minimum_name in arguments and maximum_name in arguments:
            if arguments[minimum_name] >= arguments[maximum_name]:
                raise ValueError(
                    f"{call.tool_name} requires {minimum_name} < {maximum_name}."
                )

    # disorder 的窗口必须为奇数，最短区域不能长于该窗口。batch 使用带前缀
    # 参数名，单项 disorder 使用无前缀参数名。
    disorder_parameter_pairs = (
        ("window_size", "min_region_length"),
        ("disorder_window_size", "disorder_min_region_length"),
    )
    for window_name, region_name in disorder_parameter_pairs:
        if window_name not in arguments:
            continue
        window_size = arguments[window_name]
        if not isinstance(window_size, int) or isinstance(window_size, bool):
            raise ValueError(f"{window_name} must be an integer.")
        if window_size % 2 == 0:
            raise ValueError(f"{window_name} must be odd.")
        if region_name in arguments and arguments[region_name] > window_size:
            raise ValueError(
                f"{call.tool_name} requires {region_name} <= {window_name}."
            )

    center_name = (
        "dock_catalytic_residue"
        if call.tool_name in ("enzywizard-batch", "enzywizard-mut-batch")
        else "catalytic_residue"
    )
    coordinate_name = (
        "dock_catalytic_site_coord"
        if call.tool_name in ("enzywizard-batch", "enzywizard-mut-batch")
        else "catalytic_site_coord"
    )
    box_name = (
        "dock_box_size"
        if call.tool_name in ("enzywizard-batch", "enzywizard-mut-batch")
        else "box_size"
    )
    center_present = center_name in arguments
    coordinate_present = coordinate_name in arguments
    box_present = box_name in arguments
    if center_present and coordinate_present:
        raise ValueError(
            f"{call.tool_name} cannot use both residue and coordinate docking centers."
        )
    if (center_present or coordinate_present) and not box_present:
        raise ValueError(
            f"{call.tool_name} requires {box_name} with a manual docking center."
        )
    if box_present and not (center_present or coordinate_present):
        raise ValueError(
            f"{call.tool_name} cannot use {box_name} without a manual docking center."
        )

    # 坐标和 box_size 均采用三个逗号分隔数值；box_size 三个维度必须为正。
    for parameter_name in (coordinate_name, box_name):
        if parameter_name not in arguments:
            continue
        try:
            vector = [
                float(item.strip())
                for item in str(arguments[parameter_name]).split(",")
            ]
        except ValueError as e:
            raise ValueError(
                f"{parameter_name} must contain three comma-separated numbers."
            ) from e
        if len(vector) != 3 or not all(math.isfinite(item) for item in vector):
            raise ValueError(
                f"{parameter_name} must contain three finite comma-separated numbers."
            )
        if parameter_name == box_name and any(item <= 0 for item in vector):
            raise ValueError(f"{box_name} dimensions must all be greater than 0.")


def validate_program_managed_paths(tool_plan: ToolPlan, workdir: Path) -> None:
    """确保所有 Agent 管理的写入和暂存目录都位于 workdir 内。"""

    resolved_workdir = workdir.resolve()
    managed_directory_names = {
        "output_dir",
        "wt_output_dir",
        "mut_output_dir",
        "input_dir",
        "wt_input_dir",
        "mut_input_dir",
        "substrate_dir",
    }
    for call in tool_plan.calls:
        execution_dir = Path(call.tool_execution_dir).resolve()
        if not execution_dir.is_relative_to(resolved_workdir):
            raise ValueError(
                f"Tool execution directory escapes workdir: {execution_dir}."
            )
        for parameter_name in managed_directory_names:
            parameter_value = call.arguments.get(parameter_name)
            if not isinstance(parameter_value, str):
                continue
            resolved_path = Path(parameter_value).resolve()
            if not resolved_path.is_relative_to(resolved_workdir):
                raise ValueError(
                    f"Program-managed {parameter_name} escapes workdir: {resolved_path}."
                )
        for staging_rule in call.staging_rules:
            destination_dir = Path(staging_rule.destination_dir).resolve()
            if not destination_dir.is_relative_to(resolved_workdir):
                raise ValueError(
                    f"Report staging destination escapes workdir: {destination_dir}."
                )


def apply_parameter_extraction(
    tool_plan: ToolPlan,
    parameter_extraction: ParameterExtractionOutput,
    workdir: Path,
) -> ToolPlan:
    """验证并应用 LLM 的稀疏参数修改，返回新的最终 ToolPlan。"""

    if parameter_extraction.parameter_extraction_status == "parameter_extraction_failed":
        raise ValueError(
            parameter_extraction.error_message
            or "Parameter extraction failed without an error message."
        )
    if parameter_extraction.error_message:
        raise ValueError(
            "Successful parameter extraction must have an empty error_message."
        )
    if parameter_extraction.unsupported_requests:
        unsupported_descriptions = []
        for request in parameter_extraction.unsupported_requests:
            tool_label = request.tool_name or "the requested tool"
            unsupported_descriptions.append(
                f"{tool_label}.{request.parameter_name}={request.requested_value!r} "
                f"({request.request_type})"
            )
        raise ValueError(
            "Program-managed parameter overrides are not allowed: "
            + "; ".join(unsupported_descriptions)
            + ". Configure the session workdir or explicit Agent input files instead."
        )

    final_plan = deepcopy(tool_plan)
    calls_by_step = {call.step_index: call for call in final_plan.calls}
    if len(calls_by_step) != len(final_plan.calls):
        raise ValueError("ToolPlan contains duplicate step indexes.")

    # profile 先应用，用户明确指定的具体值随后覆盖 profile 值。
    profile_by_step: dict[int, str] = {}
    for profile_request in parameter_extraction.profile_requests:
        if not profile_request.step_indexes:
            raise ValueError("A parameter profile request must target at least one step.")
        for step_index in profile_request.step_indexes:
            call = calls_by_step.get(step_index)
            if call is None:
                raise ValueError(f"Parameter profile references unknown step {step_index}.")
            previous_profile = profile_by_step.get(step_index)
            if previous_profile and previous_profile != profile_request.profile:
                raise ValueError(
                    f"Step {step_index} received conflicting fast and accurate profiles."
                )
            profile_by_step[step_index] = profile_request.profile
            profile_values = PARAMETER_PROFILE_VALUES.get(call.tool_name, {}).get(
                profile_request.profile,
                {},
            )
            editable_specs = get_editable_parameter_specs(call.tool_name)
            for parameter_name, parameter_value in profile_values.items():
                if parameter_name not in editable_specs:
                    raise ValueError(
                        f"Profile rule references non-editable parameter "
                        f"{call.tool_name}.{parameter_name}."
                    )
                call.arguments[parameter_name] = parameter_value

    adjusted_keys: set[tuple[int, str]] = set()
    for adjustment in parameter_extraction.parameter_adjustments:
        call = calls_by_step.get(adjustment.step_index)
        if call is None:
            raise ValueError(
                f"Parameter adjustment references unknown step {adjustment.step_index}."
            )
        adjustment_key = (adjustment.step_index, adjustment.parameter_name)
        if adjustment_key in adjusted_keys:
            raise ValueError(
                f"Duplicate parameter adjustment for step {adjustment.step_index}: "
                f"{adjustment.parameter_name}."
            )
        adjusted_keys.add(adjustment_key)

        editable_specs = get_editable_parameter_specs(call.tool_name)
        if adjustment.parameter_name not in editable_specs:
            raise ValueError(
                f"Parameter {call.tool_name}.{adjustment.parameter_name} is unknown or "
                f"program-managed and cannot be changed."
            )
        parameter_spec = editable_specs[adjustment.parameter_name]
        converted_value = convert_parameter_value(
            adjustment.parameter_value,
            parameter_spec,
        )
        validate_parameter_value(adjustment.parameter_name, converted_value)
        call.arguments[adjustment.parameter_name] = converted_value

    # 最终复核不能只覆盖 LLM 明确修改的值。默认值、profile 值和明确值都按
    # 同一份 YAML 类型/枚举与 service 范围规则验证，确保确认后的计划可执行。
    for call in final_plan.calls:
        editable_specs = get_editable_parameter_specs(call.tool_name)
        for parameter_name, parameter_spec in editable_specs.items():
            if parameter_name not in call.arguments:
                continue
            converted_value = convert_parameter_value(
                call.arguments[parameter_name],
                parameter_spec,
            )
            validate_parameter_value(parameter_name, converted_value)
            call.arguments[parameter_name] = converted_value
        validate_call_parameter_relationships(call)

    # strict integrate 只能用于包含所有11种单项分析的完整报告集合；部分整合使用
    # strict 会在执行时必然失败，因此在 Planning 阶段提前拒绝。
    selected_individual_tools = {
        call.tool_name
        for call in final_plan.calls
        if call.tool_name
        not in {
            "enzywizard-clean",
            "enzywizard-mut-clean",
            "enzywizard-integrate",
            "enzywizard-mut-integrate",
            "enzywizard-batch",
            "enzywizard-mut-batch",
        }
    }
    for call in final_plan.calls:
        if call.tool_name in (
            "enzywizard-integrate",
            "enzywizard-mut-integrate",
        ) and call.arguments.get("strict"):
            if len(selected_individual_tools) != 11:
                raise ValueError(
                    f"{call.tool_name} strict mode requires all 11 individual analysis "
                    "tool types in the current plan."
                )

    validate_program_managed_paths(final_plan, workdir)

    # 最终 ToolPlan 自身保留参数决策摘要；具体结构化请求仍完整保存在
    # PlanningResult.parameter_extraction 中，二者分别服务于展示和审计。
    if parameter_extraction.parameter_adjustments or parameter_extraction.profile_requests:
        final_plan.plan_reason += (
            f" [Parameters] {parameter_extraction.extraction_summary}"
        )
    return final_plan
