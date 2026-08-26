from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


# =============================================================================
# Enum types
# =============================================================================

ArtifactSide = Literal[
    # 单蛋白分析产物。
    "single",
    # WT 或 MUT 单侧产物。
    "wt",
    "mut",
    # 同时服务于两侧的共享输入或配对产物。
    "shared",
]


# =============================================================================
# Output schemas
# =============================================================================


# =============================================================================
# State data models
# =============================================================================


class CleanedFiles(BaseModel):
    """集中记录可跨轮复用的 clean 与 mut-clean 基础文件。

    这些字段与普通 artifacts 分开保存，因为后续工具计划始终以可信清理结果
    为基础。字段为 None 表示对应清理结果尚未生成或当前不可用。
    """

    # 单蛋白 clean 的输出目录、首选结构以及各标准格式文件。
    single_clean_output_dir: str | None
    single_cleaned_structure_file: str | None
    single_cleaned_cif_file: str | None
    single_cleaned_pdb_file: str | None
    single_cleaned_fasta_file: str | None

    # integrate 需要 clean report 作为残基映射和基础信息来源。
    single_clean_report_file: str | None

    # mut-clean 的公共输出目录。
    mut_clean_output_dir: str | None

    # WT 侧清理后的首选结构及标准格式文件。
    wt_cleaned_structure_file: str | None
    wt_cleaned_cif_file: str | None
    wt_cleaned_pdb_file: str | None
    wt_cleaned_fasta_file: str | None

    # MUT 侧清理后的首选结构及标准格式文件。
    mut_cleaned_structure_file: str | None
    mut_cleaned_cif_file: str | None
    mut_cleaned_pdb_file: str | None
    mut_cleaned_fasta_file: str | None

    # mut-integrate 和突变比较工具使用的配对清理报告与标准化替换。
    mut_clean_report_file: str | None
    cleaned_amino_acid_substitution: str | None


class Artifact(BaseModel):
    """描述一个由工具生成并纳入 Agent 追踪的文件产物。"""

    # 产物所属的会话轮次和计划编号，用于追溯生成上下文。
    turn_id: int
    plan_id: int

    # 逻辑产物类型，例如 pocket_report_json 或 cleaned_structure_cif。
    artifact_type: str

    # 文件在本地文件系统中的实际路径。
    path: str

    # 生成该文件的 EnzyWizard 工具名称。
    source_tool: str

    # 标记产物属于单蛋白、WT、MUT 或两侧共享。
    side: ArtifactSide

    # 保存工具参数、来源文件等暂不固定结构的附加追踪信息。
    metadata: dict[str, Any]
