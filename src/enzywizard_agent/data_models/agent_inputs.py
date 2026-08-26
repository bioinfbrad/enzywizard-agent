from __future__ import annotations

from pydantic import BaseModel


# =============================================================================
# Enum types
# =============================================================================


# =============================================================================
# Output schemas
# =============================================================================


# =============================================================================
# State data models
# =============================================================================


class AgentInputs(BaseModel):
    """保存当前蛋白系统可供 Planning 使用的标准化输入。

    所有路径在 agent_service 中转换为绝对路径；None 表示用户没有提供该输入。
    当前接口只支持一个单蛋白系统或一组 WT/MUT 系统。
    """

    # 单蛋白工作流的结构；在突变工作流中作为 WT 结构。
    structure_file: str | None

    # 单蛋白或 WT 侧的多序列比对文件。
    msa_file: str | None

    # 突变比较工作流中的 MUT 结构和 MUT 侧 MSA。
    mut_structure_file: str | None
    mut_msa_file: str | None

    # 单一底物名称或 SMILES，WT/MUT 比较时两侧共享。
    substrate_name: str | None

    # 当前只允许一个替换，例如 A123V。
    amino_acid_substitution: str | None
