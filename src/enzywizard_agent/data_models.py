from __future__ import annotations

from pydantic import BaseModel


class InputFiles(BaseModel):
    structure_file: str | None = None
    msa_file: str | None = None
    mut_structure_file: str | None = None
    mut_msa_file: str | None = None


class AgentMessage(BaseModel):
    pass


class PlanningResult(BaseModel):
    pass


class UserConfirmation(BaseModel):
    pass


class ToolCall(BaseModel):
    pass


class AnalysisResult(BaseModel):
    pass


class CleanedFiles(BaseModel):
    pass


class Artifact(BaseModel):
    pass
