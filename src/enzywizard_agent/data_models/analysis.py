from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# =============================================================================
# Enum types
# =============================================================================

AnalysisResponseMode = Literal[
    "scientific_result",
    "general_scientific_question",
    "planning_failure",
    "plan_rejected",
    "tool_execution_failure",
    "artifact_query",
    "software_help",
    "conversation",
    "out_of_scope",
    "clarification",
    "input_error",
    "request_routing_failure",
    "history_query_failure",
    "exit",
]

AnalysisStatus = Literal[
    "succeeded",
    "partial",
    "failed",
]

AnalysisRoute = Literal[
    "answer_generated",
    "exit_requested",
    "restart_required",
]

AnalysisTerminationReason = Literal[
    "user_exit",
    "max_turns",
    "session_timeout",
    "budget_exceeded",
    "restart_required",
]

AnalysisErrorType = Literal[
    "input_error",
    "routing_error",
    "planning_error",
    "history_error",
    "execution_error",
    "analysis_error",
]

EvidenceSourceKind = Literal[
    "report",
    "artifact",
    "tool_execution",
    "planning",
    "agent_input",
    "knowledge_base",
]

InferenceLevel = Literal[
    # 报告或执行记录中直接存在的信息。
    "direct",
    # Agent 使用确定性规则从直接证据计算得到的结果。
    "calculated",
    # 模型基于证据给出的科学解释，不应伪装成工具原始输出。
    "inferred",
]

FindingKind = Literal[
    "result",
    "comparison",
    "warning",
    "failure",
    "information",
]


# =============================================================================
# Output schemas
# =============================================================================


class AnalysisNarrativeOutput(BaseModel):
    """LLM 只负责组织语言时使用的最小结构化输出。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    summary: str
    final_answer: str
    limitations: list[str]

    @model_validator(mode="after")
    def validate_non_empty_text(self) -> "AnalysisNarrativeOutput":
        if not self.summary:
            raise ValueError("summary must not be empty")
        if not self.final_answer:
            raise ValueError("final_answer must not be empty")
        return self


# =============================================================================
# State data models
# =============================================================================


class AnalysisArtifactReference(BaseModel):
    """保存回答中实际使用或列出的文件产物。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    turn_id: int
    plan_id: int
    artifact_type: str
    source_tool: str
    side: Literal["single", "wt", "mut", "shared"]
    path: str
    exists: bool


class ReportSummary(BaseModel):
    """对一个 JSON 报告的受控压缩结果。

    原始报告可能包含大量逐残基数据或向量。状态中只保存用于回答的指标、摘要和
    警告，避免把完整报告重复放入 checkpoint 和模型上下文。
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    turn_id: int
    plan_id: int
    artifact_type: str
    report_type: str
    source_tool: str
    side: Literal["single", "wt", "mut", "shared"]
    path: str
    key_metrics: dict[str, Any]
    highlights: list[str]
    warnings: list[str]


class AnalysisEvidence(BaseModel):
    """一条可回溯到本地来源的分析证据。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    evidence_id: str
    source_kind: EvidenceSourceKind
    inference_level: InferenceLevel
    turn_id: int | None
    plan_id: int | None
    artifact_type: str | None
    source_tool: str | None
    side: Literal["single", "wt", "mut", "shared"] | None
    path: str | None
    field_path: str | None
    value_summary: str
    source_uri: str | None = None
    source_label: str | None = None
    source_version: str | None = None
    source_evidence_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("evidence_id", "value_summary")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        if not value:
            raise ValueError("evidence identifiers and summaries must not be empty")
        return value


class AnalysisFinding(BaseModel):
    """用户可见结论及其证据引用。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str
    statement: str
    finding_kind: FindingKind
    inference_level: InferenceLevel
    evidence_ids: list[str]

    @model_validator(mode="after")
    def validate_finding(self) -> "AnalysisFinding":
        if not self.title or not self.statement:
            raise ValueError("finding title and statement must not be empty")
        if not self.evidence_ids:
            raise ValueError("every finding must reference at least one evidence item")
        return self


class AnalysisErrorGuidance(BaseModel):
    """Stable user-facing classification and resolution for a failed turn."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    error_type: AnalysisErrorType
    stage: str
    reason: str
    can_continue_current_session: bool
    restart_required: bool
    user_actions: list[str]
    evidence_paths: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_resolution(self) -> "AnalysisErrorGuidance":
        if not self.stage or not self.reason or not self.user_actions:
            raise ValueError(
                "error guidance requires a stage, reason, and user actions"
            )
        if self.can_continue_current_session == self.restart_required:
            raise ValueError(
                "exactly one of can_continue_current_session and "
                "restart_required must be true"
            )
        return self


class AnalysisResult(BaseModel):
    """Analysis 节点写入状态的完整、可审计结果。"""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    turn_id: int
    response_mode: AnalysisResponseMode
    analysis_status: AnalysisStatus
    analysis_route: AnalysisRoute
    termination_reason: AnalysisTerminationReason | None
    summary: str
    final_answer: str
    report_summaries: list[ReportSummary]
    findings: list[AnalysisFinding]
    evidence: list[AnalysisEvidence]
    referenced_artifacts: list[AnalysisArtifactReference]
    limitations: list[str]
    error_messages: list[str]
    used_language_model: bool
    error_guidance: AnalysisErrorGuidance | None = None

    @model_validator(mode="after")
    def validate_analysis_result(self) -> "AnalysisResult":
        if self.turn_id < 1:
            raise ValueError("turn_id must be greater than or equal to 1")
        if not self.summary or not self.final_answer:
            raise ValueError("summary and final_answer must not be empty")
        if self.response_mode == "exit":
            if self.analysis_route != "exit_requested":
                raise ValueError("exit responses require analysis_route='exit_requested'")
            if self.termination_reason != "user_exit":
                raise ValueError("exit responses require termination_reason='user_exit'")
        if self.analysis_route == "exit_requested" and self.termination_reason is None:
            raise ValueError("exit_requested route requires termination_reason")
        if self.analysis_route == "answer_generated" and self.termination_reason is not None:
            raise ValueError("answer_generated route cannot have termination_reason")
        if self.analysis_route == "restart_required":
            if self.termination_reason != "restart_required":
                raise ValueError(
                    "restart_required route requires matching termination_reason"
                )
            if self.error_guidance is None or not self.error_guidance.restart_required:
                raise ValueError(
                    "restart_required route requires restart error guidance"
                )
        if (
            self.error_guidance is not None
            and self.error_guidance.restart_required
            and self.analysis_route != "restart_required"
        ):
            raise ValueError(
                "restart error guidance requires analysis_route='restart_required'"
            )
        if self.analysis_status == "failed" and not self.error_messages:
            raise ValueError("failed analysis requires error_messages")

        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence_id values must be unique")
        missing_ids = {
            evidence_id
            for finding in self.findings
            for evidence_id in finding.evidence_ids
            if evidence_id not in evidence_ids
        }
        if missing_ids:
            raise ValueError(
                f"findings reference unknown evidence IDs: {sorted(missing_ids)}"
            )
        evidence_by_id = {item.evidence_id: item for item in self.evidence}
        missing_source_ids = {
            source_id
            for item in self.evidence
            for source_id in item.source_evidence_ids
            if source_id not in evidence_by_id
        }
        if missing_source_ids:
            raise ValueError(
                "evidence references unknown source evidence IDs: "
                f"{sorted(missing_source_ids)}"
            )
        self_references = {
            item.evidence_id
            for item in self.evidence
            if item.evidence_id in item.source_evidence_ids
        }
        if self_references:
            raise ValueError(
                "evidence cannot reference itself as source evidence: "
                f"{sorted(self_references)}"
            )
        for finding in self.findings:
            if finding.finding_kind not in {"result", "comparison"}:
                continue
            sources = [evidence_by_id[item] for item in finding.evidence_ids]
            if sources and all(item.source_kind == "knowledge_base" for item in sources):
                raise ValueError(
                    "scientific results and comparisons cannot rely only on "
                    "knowledge-base terminology"
                )
        return self
