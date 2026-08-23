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
    structure_file: str | None
    msa_file: str | None
    mut_structure_file: str | None
    mut_msa_file: str | None
    substrate_name: str | None
    amino_acid_substitution: str | None
