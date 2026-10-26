"""Shared contracts and reporting utilities for the publication benchmark."""

from __future__ import annotations

import csv
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import random
import subprocess
from statistics import stdev
import time
from typing import Any, Literal, TypeVar
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, model_validator
import yaml

from enzywizard_agent.conf import AgentConfig
from enzywizard_agent.data_models.runtime import RuntimeSession
from enzywizard_agent.services.llm_service import (
    check_model_capabilities,
    create_chat_model_for_state,
    invoke_structured,
)
from enzywizard_agent.services.runtime_service import utc_now

from evaluation_utils import aggregate_usage, runtime_usage_payload


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = Path(__file__).resolve().parent / "publication_benchmark"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_REPEAT = 3
TEST_1_CASE_COUNT = 72
TEST_2_CASE_COUNT = 48
TEST_3_CASE_COUNT = 40
BenchmarkCaseT = TypeVar("BenchmarkCaseT", bound=BaseModel)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ParameterExpectation(StrictModel):
    tool_name: str
    side: Literal["single", "wt", "mut", "shared"] | None = None
    parameter_name: str
    accepted_values: list[str | int | float | bool]


class PlanAlternative(StrictModel):
    name: str
    expected_tools: list[str]
    forbidden_tools: list[str] = Field(default_factory=list)
    order_constraints: list[tuple[str, str]] = Field(default_factory=list)
    parameter_expectations: list[ParameterExpectation] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_tools(self) -> "PlanAlternative":
        if not self.expected_tools:
            raise ValueError("a plan alternative requires expected_tools")
        if len(self.expected_tools) != len(set(self.expected_tools)):
            raise ValueError("expected_tools must be unique")
        if set(self.expected_tools) & set(self.forbidden_tools):
            raise ValueError("expected_tools and forbidden_tools must not overlap")
        return self


class ConfirmationAction(StrictModel):
    decision: Literal["approved", "revision_requested", "rejected"]
    revision_request: str | None = None
    acceptable_plans: list[PlanAlternative]

    @model_validator(mode="after")
    def validate_revision(self) -> "ConfirmationAction":
        if self.decision == "revision_requested" and not self.revision_request:
            raise ValueError("revision_requested requires revision_request")
        if self.decision != "revision_requested" and self.revision_request is not None:
            raise ValueError("only revision_requested may include revision_request")
        return self


class HistoryExpectation(StrictModel):
    request_type: Literal["history_analysis", "artifact_query", "mixed"]
    actions: list[str]
    required_operations: list[str]
    resolved_turn_ids: list[int] | None = None
    required_artifact_types: list[str] = Field(default_factory=list)
    required_sides: list[str] = Field(default_factory=list)
    reuse_policy: Literal["none", "allowed", "required"] = "none"
    requires_tool_execution: bool = False
    recomputation: bool = False
    acceptable_plans: list[PlanAlternative] = Field(default_factory=list)


class WorkflowBenchmarkCase(StrictModel):
    case_id: str
    description: str
    subtype: Literal["workflow", "history"]
    tags: list[str]
    fair_relevance: Literal["relevant", "neutral"]
    message: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    confirmation_actions: list[ConfirmationAction] = Field(default_factory=list)
    history_expectation: HistoryExpectation | None = None
    execute_representative: bool = False

    @model_validator(mode="after")
    def validate_subtype(self) -> "WorkflowBenchmarkCase":
        if self.subtype == "workflow" and not self.confirmation_actions:
            raise ValueError("workflow cases require confirmation_actions")
        if self.subtype == "history" and self.history_expectation is None:
            raise ValueError("history cases require history_expectation")
        if (
            self.subtype == "history"
            and self.history_expectation is not None
            and not self.history_expectation.requires_tool_execution
        ):
            raise ValueError("Test 1 history cases must continue to tool execution")
        return self


class AnalysisArtifactSpec(StrictModel):
    path: str
    artifact_type: str
    source_tool: str
    side: Literal["single", "wt", "mut", "shared"]


class AnalysisBenchmarkCase(StrictModel):
    case_id: str
    description: str
    subtype: Literal[
        "single",
        "wt_mut",
        "enzyme_substrate",
        "supporting",
        "conflicting",
        "incomplete",
        "historical",
    ]
    tags: list[str]
    fair_relevance: Literal["relevant", "neutral"]
    question: str
    artifacts: list[AnalysisArtifactSpec]
    required_report_types: list[str]
    required_facts: list[str]
    allowed_conclusions: list[str]
    forbidden_claims: list[str]
    required_limitations: list[str] = Field(default_factory=list)


class RecoveryBenchmarkCase(StrictModel):
    case_id: str
    description: str
    scenario: Literal["preflight", "tool_failure"] = "tool_failure"
    subtype: Literal[
        "transient",
        "timeout",
        "path_or_file",
        "missing_output",
        "invalid_output",
        "resource",
        "invalid_input",
        "scientific_configuration",
        "missing_dependency",
        "missing_input",
        "unknown",
    ]
    tags: list[str]
    fair_relevance: Literal["relevant", "neutral"]
    error_message: str
    message: str | None = None
    inputs: dict[str, Any] = Field(default_factory=dict)
    tool_name: str = "enzywizard-energy"
    command_present: bool = True
    retry_outcome: Literal["succeeded", "failed", "not_attempted"]
    expected_category: str
    expected_failure_stage: str
    expected_recoverability: str
    expected_decision: str
    expected_action: str
    expected_confirmation: bool
    accepted_decisions: list[str] = Field(default_factory=list)
    accepted_actions: list[str] = Field(default_factory=list)
    scientific_change_required: bool
    expected_ai_assessment: bool = False

    @model_validator(mode="after")
    def validate_scenario(self) -> "RecoveryBenchmarkCase":
        if self.scenario == "preflight" and not self.message:
            raise ValueError("preflight recovery cases require a user message")
        return self


class BlindJudgeOutput(StrictModel):
    factual_precision: float = Field(ge=0, le=1)
    key_fact_recall: float = Field(ge=0, le=1)
    integrated_interpretation_score: float = Field(ge=0, le=1)
    identified_required_facts: list[str]
    unsupported_claims: list[str]
    rationale: str


def load_case_file(path: Path, model: type[BenchmarkCaseT], expected_count: int) -> list[BenchmarkCaseT]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported benchmark schema in {path}")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError(f"Benchmark file has no cases list: {path}")
    catalog = payload.get("catalog") or {}
    expanded_cases = [_expand_catalog_references(item, catalog) for item in raw_cases]
    cases = [model.model_validate(item) for item in expanded_cases]
    if len(cases) != expected_count:
        raise ValueError(f"Expected {expected_count} cases in {path}, found {len(cases)}")
    identifiers = [item.case_id for item in cases]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"Duplicate case IDs in {path}")
    return cases


def _expand_catalog_references(value: Any, catalog: dict[str, Any]) -> Any:
    """Expand explicit ``$input``/``$plans`` references in benchmark YAML."""

    if isinstance(value, list):
        return [_expand_catalog_references(item, catalog) for item in value]
    if not isinstance(value, dict):
        return value
    result = deepcopy(value)
    input_key = result.pop("$input", None)
    if input_key is not None:
        inputs = catalog.get("inputs", {}).get(input_key)
        if inputs is None:
            raise ValueError(f"Unknown benchmark input catalog key: {input_key}")
        result["inputs"] = deepcopy(inputs)
    plans_key = result.pop("$plans", None)
    if plans_key is not None:
        plans = catalog.get("plans", {}).get(plans_key)
        if plans is None:
            raise ValueError(f"Unknown benchmark plan catalog key: {plans_key}")
        result["acceptable_plans"] = deepcopy(plans)
    artifacts_key = result.pop("$artifacts", None)
    if artifacts_key is not None:
        artifacts = catalog.get("artifacts", {}).get(artifacts_key)
        if artifacts is None:
            raise ValueError(
                f"Unknown benchmark artifact catalog key: {artifacts_key}"
            )
        result["artifacts"] = deepcopy(artifacts)
    return {
        key: _expand_catalog_references(item, catalog)
        for key, item in result.items()
    }


def select_cases(cases: list[BenchmarkCaseT], case_ids: list[str] | None) -> list[BenchmarkCaseT]:
    if not case_ids:
        return cases
    by_id = {item.case_id: item for item in cases}
    unknown = sorted(set(case_ids) - set(by_id))
    if unknown:
        raise ValueError(f"Unknown case IDs: {unknown}")
    return [by_id[item] for item in case_ids]


def select_profile_cases(
    cases: list[BenchmarkCaseT],
    *,
    case_ids: list[str] | None,
    profile: Literal["quick", "full"],
    sample_per_subtype: int,
    seed: int,
    quick_case_count: int | None = None,
) -> list[BenchmarkCaseT]:
    """Select an explicit list or a reproducible subtype-stratified profile."""

    if case_ids:
        return select_cases(cases, case_ids)
    if profile == "full":
        return cases
    if sample_per_subtype < 1:
        raise ValueError("sample_per_subtype must be positive")
    if quick_case_count is not None and quick_case_count < 1:
        raise ValueError("quick_case_count must be positive")
    grouped: dict[str, list[BenchmarkCaseT]] = {}
    for case in cases:
        grouped.setdefault(str(getattr(case, "subtype", "all")), []).append(case)
    rng = random.Random(seed)
    shuffled: dict[str, list[BenchmarkCaseT]] = {}
    # Consume the RNG in a stable subtype order so an existing seed keeps
    # selecting the same cases when the exact-count option is used.
    for subtype in sorted(grouped):
        candidates = list(grouped[subtype])
        rng.shuffle(candidates)
        shuffled[subtype] = candidates

    selected: list[BenchmarkCaseT] = []
    if quick_case_count is None:
        for subtype in sorted(shuffled):
            selected.extend(shuffled[subtype][:sample_per_subtype])
    else:
        target = min(quick_case_count, len(cases))
        offsets = {subtype: 0 for subtype in shuffled}
        while len(selected) < target:
            added = False
            for subtype in sorted(shuffled):
                offset = offsets[subtype]
                candidates = shuffled[subtype]
                if offset >= len(candidates):
                    continue
                selected.append(candidates[offset])
                offsets[subtype] = offset + 1
                added = True
                if len(selected) == target:
                    break
            if not added:
                break
    selected_ids = {item.case_id for item in selected}
    return [item for item in cases if item.case_id in selected_ids]


def create_run_dir(parent: str | Path, label: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(parent).expanduser().resolve() / f"{label}_{timestamp}_{uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def benchmark_metadata(
    *,
    test_id: str,
    cases_file: Path,
    model: str,
    base_url: str,
    reasoning_effort: str | None,
    ontology_mode: str,
    repeat: int,
) -> dict[str, Any]:
    return {
        "test_id": test_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cases_file": str(cases_file.resolve()),
        "cases_sha256": file_sha256(cases_file),
        "git_revision": git_revision(),
        "model": model,
        "base_url": base_url,
        "reasoning_effort": reasoning_effort,
        "ontology_mode": ontology_mode,
        "repeat": repeat,
    }


def check_benchmark_model_contract(
    *,
    api_key: str,
    base_url: str,
    model: str,
    reasoning_effort: str | None,
    ontology_mode: str,
    workdir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply the Agent's strict three-stage JSON Schema gate once per runner."""

    workdir.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "config": AgentConfig(
            api_key=api_key,
            base_url=base_url,
            model=model,
            reasoning_effort=reasoning_effort,
            # Capability probes validate transport and JSON Schema support, not
            # the experimental FAIR treatment.
            ontology_mode="off",
            check_model_capabilities=True,
            workdir=workdir,
        ),
        "session_id": f"benchmark-contract-{uuid4().hex}",
        "current_turn_id": 1,
        "current_user_message": "Model contract capability probe.",
        "runtime_session": RuntimeSession(
            started_at=utc_now(), started_monotonic=time.monotonic()
        ),
    }
    result = check_model_capabilities(state)
    return result.model_dump(mode="json"), runtime_usage_payload(
        state["runtime_session"]
    )


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_flat_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )


def mean(values: list[float | int | bool | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return round(sum(clean) / len(clean), 6) if clean else None


def metric_statistics(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Report mean and between-repetition SD for scalar benchmark metrics."""

    names = sorted(
        {name for item in results for name in (item.get("metrics") or {})}
    )
    summary: dict[str, dict[str, Any]] = {}
    for name in names:
        values_by_repetition: dict[int, list[float]] = {}
        for item in results:
            value = (item.get("metrics") or {}).get(name)
            if value is None or not isinstance(value, (int, float, bool)):
                continue
            repetition = int(item.get("repetition", 1))
            values_by_repetition.setdefault(repetition, []).append(float(value))
        repetition_means = [
            sum(values) / len(values)
            for _, values in sorted(values_by_repetition.items())
            if values
        ]
        observed_count = sum(len(values) for values in values_by_repetition.values())
        summary[name] = {
            "mean": (
                round(sum(repetition_means) / len(repetition_means), 6)
                if repetition_means
                else None
            ),
            "sd": (
                round(stdev(repetition_means), 6)
                if len(repetition_means) > 1
                else (0.0 if repetition_means else None)
            ),
            "n": observed_count,
            "repetition_n": len(repetition_means),
        }
    return summary


def metric_statistics_by_subtype(
    results: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(str(result.get("subtype", "unknown")), []).append(result)
    return {
        subtype: metric_statistics(items)
        for subtype, items in sorted(grouped.items())
    }


def summarize_repeated_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_case: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        by_case.setdefault(str(result["case_id"]), []).append(result)
    stable_cases = sum(
        bool(items) and all(bool(item.get("passed")) for item in items)
        for items in by_case.values()
    )
    return {
        "case_count": len(by_case),
        "run_count": len(results),
        "passed_runs": sum(bool(item.get("passed")) for item in results),
        "run_pass_rate": mean([bool(item.get("passed")) for item in results]),
        "stable_passed_cases": stable_cases,
        "three_run_stable_pass_rate": (
            round(stable_cases / len(by_case), 6) if by_case else None
        ),
    }


JUDGE_SYSTEM_PROMPT = """You are a blinded scientific evaluation judge.
Evaluate only the supplied candidate output against the supplied expert rubric and
structured report summaries and evidence. A claim is supported when it appears in either
structured source. The candidate output contains both the user-facing answer and a
separate list of stated limitations; evaluate both together. Required limitations are
semantic requirements, not exact phrases: count a clearly equivalent scientific boundary
as satisfied, but do not count merely omitting a prohibited claim. Do not infer the tested
model, experimental condition, or expected treatment. Different wording is acceptable.
Conservative methodological caveats in the limitations list, such as describing computed
values as model estimates rather than experimental measurements, are not scientific-result
claims and must not be counted as unsupported unless they contradict the supplied evidence.
Score factual precision as the fraction of candidate scientific claims supported by the
reports or explicitly marked as hypotheses. Score key-fact recall as the fraction of
expert-required facts that the answer covers. Score integrated interpretation for whether
the answer correctly relates report dimensions, preserves scientific boundaries, and
handles required limitations. Evidence-link structure is scored separately by code.
Return only the strict JSON Schema output."""


def run_blind_judge(
    *,
    api_key: str,
    base_url: str,
    model: str,
    reasoning_effort: str | None,
    workdir: Path,
    question: str,
    answer: str,
    limitations: list[str],
    required_facts: list[str],
    allowed_conclusions: list[str],
    forbidden_claims: list[str],
    required_limitations: list[str],
    structured_report_summaries: list[dict[str, Any]],
    structured_evidence: list[dict[str, Any]],
) -> tuple[BlindJudgeOutput, dict[str, Any]]:
    config = AgentConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        reasoning_effort=reasoning_effort,
        ontology_mode="off",
        check_model_capabilities=False,
        workdir=workdir,
    )
    state: dict[str, Any] = {
        "config": config,
        "session_id": f"judge-{uuid4().hex}",
        "current_turn_id": 1,
        "current_user_message": question,
        "runtime_session": RuntimeSession(
            started_at=utc_now(),
            started_monotonic=time.monotonic(),
        ),
    }
    payload = {
        "question": question,
        "candidate_output": {
            "answer": answer,
            "limitations": limitations,
        },
        "expert_rubric": {
            "required_facts": required_facts,
            "allowed_conclusions": allowed_conclusions,
            "forbidden_claims": forbidden_claims,
            "required_limitations": required_limitations,
        },
        "structured_report_summaries": structured_report_summaries,
        "structured_evidence": structured_evidence,
    }
    output = invoke_structured(
        state,
        stage="benchmark.blind_judge",
        schema=BlindJudgeOutput,
        llm=create_chat_model_for_state(state),
        messages=[
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ],
    )
    return output, runtime_usage_payload(state["runtime_session"])


def aggregate_result_usage(results: list[dict[str, Any]]) -> dict[str, Any]:
    records: list[dict[str, Any] | None] = []
    for item in results:
        records.append(item.get("usage"))
        records.append(item.get("judge_usage"))
    return aggregate_usage(records)
