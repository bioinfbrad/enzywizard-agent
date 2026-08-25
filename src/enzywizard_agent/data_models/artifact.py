from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


# =============================================================================
# Enum types
# =============================================================================

ArtifactSide = Literal[
    "single",
    "wt",
    "mut",
    "shared",
]


# =============================================================================
# Output schemas
# =============================================================================


# =============================================================================
# State data models
# =============================================================================


class CleanedFiles(BaseModel):
    single_clean_output_dir: str | None
    single_cleaned_structure_file: str | None
    single_cleaned_cif_file: str | None
    single_cleaned_pdb_file: str | None
    single_cleaned_fasta_file: str | None
    single_clean_report_file: str | None

    mut_clean_output_dir: str | None
    wt_cleaned_structure_file: str | None
    wt_cleaned_cif_file: str | None
    wt_cleaned_pdb_file: str | None
    wt_cleaned_fasta_file: str | None
    mut_cleaned_structure_file: str | None
    mut_cleaned_cif_file: str | None
    mut_cleaned_pdb_file: str | None
    mut_cleaned_fasta_file: str | None
    mut_clean_report_file: str | None
    cleaned_amino_acid_substitution: str | None


class Artifact(BaseModel):
    turn_id: int
    plan_id: int
    artifact_type: str
    path: str
    source_tool: str
    side: ArtifactSide
    metadata: dict[str, Any]
