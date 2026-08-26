from __future__ import annotations

import re


# 允许出现在单字母氨基酸替换表示中的20种标准氨基酸。
STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def validate_single_amino_acid_substitution(value: str) -> str | None:
    """校验单个 A123V 风格氨基酸替换，成功返回 None，失败返回错误文本。"""

    # 当前产品不支持多突变；分号明确表示多个替换项。
    if ";" in value:
        return "Multiple amino acid substitutions are not supported yet. Provide one substitution such as A123V."

    # 三组依次捕获原始残基、从1开始的正整数位置和目标残基。
    match = re.fullmatch(r"([A-Z])([1-9][0-9]*)([A-Z])", value)
    if match is None:
        return "Invalid amino_acid_substitution format. Use one amino acid substitution such as A123V."

    original_residue, _, mutant_residue = match.groups()

    # 格式正确后继续确认两端字符都属于标准氨基酸集合。
    if original_residue not in STANDARD_AMINO_ACIDS:
        return f"Invalid amino_acid_substitution original residue: {original_residue}."
    if mutant_residue not in STANDARD_AMINO_ACIDS:
        return f"Invalid amino_acid_substitution mutant residue: {mutant_residue}."
    # 原始和目标残基相同不构成有效替换。
    if original_residue == mutant_residue:
        return "Invalid amino_acid_substitution: original and mutant residues must be different."

    # 此处不验证结构对应位置的实际残基，该生物学一致性由用户负责。
    return None
