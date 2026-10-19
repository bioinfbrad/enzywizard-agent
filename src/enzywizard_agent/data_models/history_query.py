from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .artifact import Artifact


HistoryTurnScope = Literal[
    "specific",
    "previous",
    "all_previous",
    "latest_matching",
]

HistoryOperation = Literal[
    "summarize",
    "explain",
    "compare",
    "list_artifacts",
    "reuse_as_input",
]

ArtifactCategory = Literal[
    "all",
    "clean",
    "aaprops",
    "hydrocluster",
    "energy",
    "flexibility",
    "disorder",
    "conservation",
    "embedding",
    "pocket",
    "substrate",
    "dock",
    "interaction",
    "integrate",
    "structure",
    "sequence",
    "graph",
    "log",
]

ReportType = Literal[
    "enzywizard_clean",
    "enzywizard_mut_clean",
    "enzywizard_aaprops",
    "enzywizard_hydrocluster",
    "enzywizard_energy",
    "enzywizard_flexibility",
    "enzywizard_disorder",
    "enzywizard_conservation",
    "enzywizard_embedding",
    "enzywizard_pocket",
    "enzywizard_substrate",
    "enzywizard_dock",
    "enzywizard_interaction",
    "enzywizard_integrate",
    "enzywizard_mut_integrate",
]

HistoryQueryStatus = Literal[
    "succeeded",
    "not_found",
    "clarification_required",
    "failed",
]


class HistoryTurnSelectionDraft(BaseModel):
    """Small model-facing contract for resolving one routed action to turns."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action_id: int = Field(ge=0)
    turn_scope: HistoryTurnScope
    turn_ids: list[int]

    @field_validator("turn_ids")
    @classmethod
    def validate_turn_ids(cls, value: list[int]) -> list[int]:
        if any(turn_id < 1 for turn_id in value):
            raise ValueError("turn IDs must be greater than or equal to 1")
        if len(value) != len(set(value)):
            raise ValueError("turn_ids must not contain duplicates")
        return value

    @model_validator(mode="after")
    def validate_selection(self) -> "HistoryTurnSelectionDraft":
        if self.turn_scope == "specific" and not self.turn_ids:
            raise ValueError("specific turn scope requires turn_ids")
        if self.turn_scope != "specific" and self.turn_ids:
            raise ValueError("only specific turn scope may contain turn_ids")
        return self


class HistoryArtifactSelectionDraft(BaseModel):
    """Small model-facing contract for selecting verified artifact catalog IDs."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    action_id: int = Field(ge=0)
    artifact_ids: list[str]

    @field_validator("artifact_ids")
    @classmethod
    def validate_artifact_ids(cls, value: list[str]) -> list[str]:
        if any(not artifact_id for artifact_id in value):
            raise ValueError("artifact_ids must not contain empty values")
        if len(value) != len(set(value)):
            raise ValueError("artifact_ids must not contain duplicates")
        return value


class HistoryQueryOutput(BaseModel):
    """Program-compiled, auditable history query used by downstream stages."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    turn_scope: HistoryTurnScope
    referenced_turn_ids: list[int]
    artifact_categories: list[ArtifactCategory]
    # 精确产物类型直接来自当前 artifact catalog；无法匹配时程序请求用户澄清。
    artifact_types: list[str]
    report_types: list[ReportType]
    sides: list[Literal["single", "wt", "mut", "shared"]]
    operations: list[HistoryOperation]
    query_reason: str
    clarification_question: str | None

    @field_validator(
        "referenced_turn_ids",
        "artifact_categories",
        "artifact_types",
        "report_types",
        "sides",
        "operations",
    )
    @classmethod
    def reject_duplicates(cls, value: list) -> list:
        if len(value) != len(set(value)):
            raise ValueError("history query lists cannot contain duplicates")
        return value

    @field_validator("artifact_types")
    @classmethod
    def validate_artifact_types(cls, value: list[str]) -> list[str]:
        if any(not item for item in value):
            raise ValueError("artifact_types must not contain empty values")
        return value

    @model_validator(mode="after")
    def validate_query(self) -> "HistoryQueryOutput":
        if not self.operations:
            raise ValueError("operations must not be empty")
        if not self.query_reason:
            raise ValueError("query_reason must not be empty")
        if self.turn_scope == "specific" and not self.referenced_turn_ids:
            raise ValueError("specific turn scope requires referenced_turn_ids")
        if "all" in self.artifact_categories and len(self.artifact_categories) > 1:
            raise ValueError("artifact category 'all' cannot be combined with others")
        return self


class HistoryQueryResult(BaseModel):
    """保存历史查询条件、真实匹配产物及可审计状态。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    turn_id: int
    query_status: HistoryQueryStatus
    query: HistoryQueryOutput | None
    resolved_turn_ids: list[int]
    selected_artifacts: list[Artifact]
    summary: str
    error_message: str

    @model_validator(mode="after")
    def validate_result(self) -> "HistoryQueryResult":
        if self.turn_id < 1:
            raise ValueError("turn_id must be greater than or equal to 1")
        if not self.summary:
            raise ValueError("summary must not be empty")
        if self.query_status == "succeeded":
            if self.query is None or not self.selected_artifacts or self.error_message:
                raise ValueError(
                    "successful history query requires query, artifacts, and no error"
                )
        elif self.query_status == "not_found":
            if self.query is None or self.selected_artifacts:
                raise ValueError(
                    "not_found history query requires query and no selected artifacts"
                )
        elif self.query_status == "clarification_required":
            if self.query is None or not self.query.clarification_question:
                raise ValueError(
                    "clarification_required needs a clarification question"
                )
        elif self.query is not None or not self.error_message:
            raise ValueError("failed history query requires error_message and query=None")
        return self
