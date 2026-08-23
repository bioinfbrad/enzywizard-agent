from __future__ import annotations

import re


STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def validate_single_amino_acid_substitution(value: str) -> str | None:
    if ";" in value:
        return "Multiple amino acid substitutions are not supported yet. Provide one substitution such as A123V."

    match = re.fullmatch(r"([A-Z])([1-9][0-9]*)([A-Z])", value)
    if match is None:
        return "Invalid amino_acid_substitution format. Use one amino acid substitution such as A123V."

    original_residue, _, mutant_residue = match.groups()
    if original_residue not in STANDARD_AMINO_ACIDS:
        return f"Invalid amino_acid_substitution original residue: {original_residue}."
    if mutant_residue not in STANDARD_AMINO_ACIDS:
        return f"Invalid amino_acid_substitution mutant residue: {mutant_residue}."
    if original_residue == mutant_residue:
        return "Invalid amino_acid_substitution: original and mutant residues must be different."

    return None
