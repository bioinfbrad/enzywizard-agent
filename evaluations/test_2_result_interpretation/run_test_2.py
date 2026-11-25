#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Literal
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field
import yaml

from enzywizard_agent.conf import AgentConfig
from enzywizard_agent.data_models.agent_inputs import AgentInputs
from enzywizard_agent.data_models.artifact import Artifact
from enzywizard_agent.data_models.history_query import (
    HistoryQueryOutput,
    HistoryQueryResult,
)
from enzywizard_agent.data_models.runtime import RuntimeSession
from enzywizard_agent.services.analysis_service import (
    _analyze_reports,
    analyze_agent_state,
)
from enzywizard_agent.services.llm_service import (
    check_model_capabilities,
    create_chat_model_for_state,
    invoke_structured,
)
from enzywizard_agent.services.request_routing_service import route_user_request
from enzywizard_agent.services.runtime_service import utc_now
from enzywizard_agent.state import AgentState
from enzywizard_agent.utils.terminology_utils import (
    scientific_answer_body as _scientific_answer_body,
)

from scoring import (
    METRIC_NAMES,
    cluster_bootstrap_ci,
    repetition_statistics,
    score_case,
    score_case_full_review,
    validate_full_review,
    validate_iis_review,
)


ROOT = Path(__file__).resolve().parent
CATALOG_PATH = ROOT / "case_catalog.yaml"
INDEX_PATH = ROOT / "cases" / "index.yaml"
GOLD_STANDARD_PATH = ROOT / "gold_report_facts.json"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-6-luna"
DEFAULT_REASONING_EFFORT = "medium"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class IISJudgment(StrictModel):
    dimension_id: str
    applicable: bool
    score: int | None = Field(default=None, ge=0, le=2)
    reason: str


class IISReviewOutput(StrictModel):
    interpretation_judgments: list[IISJudgment]
    overall_notes: str


class FactualJudgment(StrictModel):
    claim_id: str
    claim: str
    supported: bool
    reason: str


class KeyFactJudgment(StrictModel):
    fact_id: str
    covered: bool
    reason: str


class EvidenceAssociationJudgment(StrictModel):
    association_id: str
    claim: str
    evidence_reference: str
    supported: bool
    reason: str


class EvidenceCoverageJudgment(StrictModel):
    fact_id: str
    evidence_supported: bool
    reason: str


class FullReviewOutput(IISReviewOutput):
    factual_judgments: list[FactualJudgment]
    key_fact_judgments: list[KeyFactJudgment]
    evidence_association_judgments: list[EvidenceAssociationJudgment]
    evidence_coverage_judgments: list[EvidenceCoverageJudgment]


IIS_JUDGE_PROMPT = """You simulate the human IIS reviewer for Test 2.
Use only the supplied IIS review packet. Evaluate only the listed IIS dimensions using
the same rubric shown to a human: 0 means missing or substantially incorrect, 1 means
partly correct, and 2 means complete and correct. Preserve each dimension identifier
and its applicability. Use null for a non-applicable dimension.

Apply these rules consistently:
- Score relevance and scientific quality, not answer length. A concise answer and a
  detailed answer must receive the same score when both satisfy the request.
- The case-specific must_address entries are requirements. The available report
  metrics are an evidence pool, not a checklist: do not penalize omission of optional
  metrics unless the question or a must_address entry requires them.
- Award 2 for result integration when the answer selects the relevant major results
  and satisfies the case-specific requirements; exhaustive metric enumeration is not
  required.
- For inference boundaries, require only the listed required_limitations and the
  avoidance of forbidden claims or actual overclaiming. Do not demand unrelated
  generic caveats.
- In a follow-up, judge compliance with the follow-up instruction. Do not require all
  details from the initial answer when the user asks to shorten, focus, or restrict it;
  do require preservation when the user explicitly asks not to change supported facts.
- Do not infer a defect merely because an optional report field is absent.

Do not calculate or alter FP, KFR, or EA-F1. Return only the strict JSON Schema output."""


FULL_REVIEW_PROMPT = """You perform the full item-level semantic review for Test 2.
Use only the supplied review packet and follow these requirements:

If objective_metrics_applicable is false, this is a boundary/non-report case: return
empty factual_judgments, key_fact_judgments, evidence_association_judgments, and
evidence_coverage_judgments. Evaluate only the supplied IIS boundary dimensions.

1. FP: decompose the candidate answer into all independently verifiable atomic factual
claims. Exclude advice, headings, pure uncertainty statements, and subjective prose.
Return one factual_judgment per verifiable claim and decide whether the supplied reports
directly support it. Do not merge separate numerical or directional claims.
Exclude general terminology definitions, statements that explain a concept's meaning,
and general inference-boundary or limitation statements from FP unless they also assert
a case-specific result about the supplied reports. These semantic statements are
evaluated by IIS; they are not report-result facts merely because they appear in the
answer. If one sentence combines a general definition with a case-specific result,
extract and judge only the case-specific result as an FP claim.
Evidence-label boilerplate and terminology-source labels are citation metadata rather
than scientific claims unless the answer makes an additional factual assertion about
them.
2. KFR: return exactly one key_fact_judgment for every supplied KFR key-fact ID and
decide whether the candidate answer semantically covers it. Equivalent wording is
acceptable; object, side, direction, value, unit, and qualification must remain correct.
3. EA-F1 precision: identify every explicit conclusion-evidence association generated
in the candidate answer and return one evidence_association_judgment for each. Keep
separate links separate. It is supported only if the cited evidence is real, correctly
identified, and supports the linked claim at the stated report/side/field granularity.
Use one atomic conclusion plus one distinct evidence source/field as the association
unit. One conclusion citing two independent sources is two associations; two conclusions
using one source are also two associations. Count an identical repeated association only
once, including duplication between the narrative and an evidence-summary section. A
source listed without a linked conclusion is not an association.
Use candidate_evidence_registry to resolve the evidence actually attached to findings.
It is provenance context, not a gold label: independently verify whether each referenced
source supports the linked conclusion using the supplied report summaries and key facts.
4. EA-F1 recall: return exactly one evidence_coverage_judgment for every KFR key-fact
ID. Mark evidence_supported only when the candidate answer covers that fact and links it
to at least one correct, resolvable supporting source.
5. IIS: apply the listed 0/1/2 dimensions. Score scientific quality and relevance, not
length. Report metrics are an evidence pool, not a checklist. Require only the listed
case requirements and limitations; do not demand unrelated generic caveats.

Do not return final metric percentages. The program computes all four metrics from your
item-level judgments. Return only the strict JSON Schema output."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test 2: scientific result interpretation and evidence provenance."
    )
    parser.add_argument("--api_key", default=os.environ.get("ENZYWIZARD_API_KEY"))
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning_effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument(
        "--llm_timeout_seconds",
        type=float,
        default=180.0,
        help="Timeout for each tested-model request (default: 180 seconds).",
    )
    parser.add_argument(
        "--ontology_mode", choices=("off", "retrieve", "control"), default="retrieve"
    )
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument("--quick_case_count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--case_ids", nargs="*")
    parser.add_argument(
        "--iis_evaluator", choices=("human", "llm"), default="human"
    )
    parser.add_argument(
        "--reviewer",
        dest="iis_evaluator",
        choices=("human", "llm"),
        help="Alias for --iis_evaluator; selects the human or LLM reviewer.",
    )
    parser.add_argument(
        "--evaluation_mode",
        choices=("hybrid", "full_review"),
        default="hybrid",
        help=(
            "hybrid computes FP/KFR/EA-F1 automatically and reviews IIS; "
            "full_review reviews all four metrics item by item"
        ),
    )
    parser.add_argument(
        "--judge_api_key", default=os.environ.get("ENZYWIZARD_JUDGE_API_KEY")
    )
    parser.add_argument("--judge_model", default="gpt-6-luna")
    parser.add_argument("--judge_base_url", default=DEFAULT_BASE_URL)
    parser.add_argument("--judge_reasoning_effort")
    parser.add_argument(
        "--judge_timeout_seconds",
        type=float,
        default=180.0,
        help="Timeout for each judge request (default: 180 seconds).",
    )
    parser.add_argument("--show_details", action="store_true")
    parser.add_argument("--stop_on_failure", action="store_true")
    parser.add_argument(
        "--bootstrap_iterations",
        type=int,
        default=1000,
        help="Case-cluster bootstrap replicates; use 0 to omit confidence intervals.",
    )
    parser.add_argument("--validate_only", action="store_true")
    parser.add_argument(
        "--workdir", default="~/enzywizard-evaluation-test-2"
    )
    parser.add_argument("--output")
    return parser


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def load_suite() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    catalog = _read_yaml(CATALOG_PATH)
    gold_standard = json.loads(GOLD_STANDARD_PATH.read_text(encoding="utf-8"))
    if not isinstance(gold_standard, dict):
        raise ValueError("Test 2 gold standard must be a JSON object")
    catalog["_gold_standard"] = gold_standard
    index = _read_yaml(INDEX_PATH)
    if catalog.get("schema_version") != 2 or index.get("schema_version") != 1:
        raise ValueError("Unsupported Test 2 fixture schema")
    cases: list[dict[str, Any]] = []
    for filename in index.get("case_files", []):
        path = ROOT / "cases" / str(filename)
        payload = _read_yaml(path)
        if payload.get("schema_version") != 1:
            raise ValueError(f"Unsupported case schema: {path}")
        category = str(payload.get("category_id", ""))
        entries = payload.get("cases")
        if not category or not isinstance(entries, list):
            raise ValueError(f"Invalid category fixture: {path}")
        if len(entries) != int(payload.get("case_count", -1)):
            raise ValueError(f"Incorrect case_count in {path}")
        for entry in entries:
            item = dict(entry)
            item["category_id"] = category
            cases.append(item)
    _validate_suite(catalog, cases)
    return catalog, cases


def _case_key_facts(
    catalog: dict[str, Any], case: dict[str, Any]
) -> list[tuple[str, dict[str, Any]]]:
    """Select a compact, fixed set of question-relevant facts for one case."""
    if case["category_id"] == "boundary_unrelated_analysis":
        return []
    gold_reports = catalog["_gold_standard"]["reports"]
    priorities = catalog["key_fact_metric_priority"]
    analysis_report_ids = [
        report_id
        for report_id in case["report_ids"]
        if catalog["reports"][report_id]["source_tool"] != "enzywizard-mut-clean"
    ]
    has_mutation_report = any(
        catalog["reports"][report_id]["source_tool"] == "enzywizard-mut-clean"
        for report_id in case["report_ids"]
    )
    paired_comparison = (
        case.get("scope") == "paired"
        and len(analysis_report_ids) == 2
        and len(
            {
                gold_reports[report_id]["report_type"]
                for report_id in analysis_report_ids
            }
        )
        == 1
    )
    quota = (
        3
        if len(analysis_report_ids) == 1
        else 2
        if has_mutation_report or paired_comparison
        else 3
        if len(analysis_report_ids) == 2
        else 2
    )
    selected: list[tuple[str, dict[str, Any]]] = []
    for report_id in analysis_report_ids:
        report = gold_reports[report_id]
        by_metric = {item["metric_name"]: item for item in report["facts"]}
        preferred = [
            by_metric[name]
            for name in priorities[report["report_type"]]
            if name in by_metric
        ]
        if len(preferred) < quota:
            preferred.extend(
                item for item in report["facts"] if item not in preferred
            )
        selected.extend((report_id, item) for item in preferred[:quota])

    if paired_comparison:
        wt_report_id = next(
            report_id for report_id in analysis_report_ids if "_wt_" in report_id
        )
        mut_report_id = next(
            report_id for report_id in analysis_report_ids if "_mut_" in report_id
        )
        report_type = gold_reports[wt_report_id]["report_type"]
        comparison_metric = next(
            metric
            for metric in priorities[report_type]
            if any(
                fact["metric_name"] == metric
                for fact in gold_reports[wt_report_id]["facts"]
            )
            and any(
                fact["metric_name"] == metric
                for fact in gold_reports[mut_report_id]["facts"]
            )
        )
        wt_fact = next(
            fact
            for fact in gold_reports[wt_report_id]["facts"]
            if fact["metric_name"] == comparison_metric
        )
        mut_fact = next(
            fact
            for fact in gold_reports[mut_report_id]["facts"]
            if fact["metric_name"] == comparison_metric
        )
        wt_value = wt_fact["expected_value"]
        mut_value = mut_fact["expected_value"]
        if isinstance(wt_value, (int, float)) and isinstance(mut_value, (int, float)):
            relation = (
                "higher than"
                if mut_value > wt_value
                else "lower than"
                if mut_value < wt_value
                else "equal to"
            )
        else:
            relation = "equal to" if mut_value == wt_value else "different from"
        selected.append(
            (
                wt_report_id,
                {
                    "fact_id": f"{case['case_id']}:comparison:{comparison_metric}",
                    "metric_name": f"comparison::{comparison_metric}",
                    "field_path": f"$comparison.{comparison_metric}",
                    "expected_value": (
                        f"MUT {comparison_metric} ({mut_value}) is {relation} "
                        f"WT ({wt_value})"
                    ),
                    "inference_level": "calculated",
                    "numeric_abs_tolerance": 0.0,
                    "numeric_rel_tolerance": 0.0,
                    "accepted_aliases": [
                        f"WT MUT {comparison_metric} direction",
                        f"{comparison_metric} comparison",
                    ],
                    "supporting_report_ids": [wt_report_id, mut_report_id],
                    "derived_comparison_fact": True,
                },
            )
        )

    for report_id in case["report_ids"]:
        spec = catalog["reports"][report_id]
        if spec["source_tool"] != "enzywizard-mut-clean":
            continue
        mutation = catalog["enzymes"][case["enzyme_id"]]["mutation"]
        selected.append(
            (
                report_id,
                {
                    "fact_id": f"{report_id}:mutation_identity",
                    "metric_name": "amino_acid_substitution",
                    "field_path": "$.amino_acid_substitution",
                    "expected_value": mutation,
                    "inference_level": "direct",
                    "numeric_abs_tolerance": 0.0,
                    "numeric_rel_tolerance": 0.0,
                    "accepted_aliases": [
                        "mutation identity",
                        "amino acid substitution",
                    ],
                    "synthetic_case_fact": True,
                },
            )
        )
    return selected


def _validate_suite(catalog: dict[str, Any], cases: list[dict[str, Any]]) -> None:
    automatic_policy = catalog.get("automatic_evaluation")
    if not isinstance(automatic_policy, dict) or not all(
        automatic_policy.get(key)
        for key in (
            "factual_unit",
            "factual_support",
            "required_fact_policy",
            "evidence_attribution_policy",
            "narrative_policy",
        )
    ):
        raise ValueError("Test 2 requires a complete automatic_evaluation policy")
    identifiers = [str(item.get("case_id", "")) for item in cases]
    if any(not item for item in identifiers) or len(identifiers) != len(set(identifiers)):
        raise ValueError("Test 2 case IDs must be non-empty and unique")
    expected_categories = catalog.get("categories") or {}
    actual_counts: dict[str, int] = {}
    for case in cases:
        category = str(case["category_id"])
        actual_counts[category] = actual_counts.get(category, 0) + 1
        reports = case.get("report_ids")
        if not isinstance(reports, list):
            raise ValueError(f"{case['case_id']} report_ids must be a list")
        unknown = [item for item in reports if item not in catalog.get("reports", {})]
        if unknown:
            raise ValueError(f"{case['case_id']} has unknown reports: {unknown}")
        is_boundary = category == "boundary_unrelated_analysis"
        if is_boundary:
            if reports or not case.get("expected_behavior"):
                raise ValueError(
                    f"{case['case_id']} boundary cases require no reports and an expected_behavior"
                )
        elif not reports or not case.get("required_facts"):
            raise ValueError(
                f"{case['case_id']} requires reports and required_facts"
            )
        if category == "multi_turn_follow_up_revision" and not (
            case.get("initial_question") and case.get("follow_up_question")
        ):
            raise ValueError(f"{case['case_id']} requires both conversation turns")
        if category != "multi_turn_follow_up_revision" and not case.get("question"):
            raise ValueError(f"{case['case_id']} requires a question")
        for field in ("required_facts", "required_limitations", "forbidden_claims"):
            for requirement in case.get(field, []):
                text = str(requirement).strip()
                if not text or text.casefold().startswith(("and ", "or ", "but ")):
                    raise ValueError(
                        f"{case['case_id']} has a malformed {field} entry: {requirement!r}"
                    )
    for category, spec in expected_categories.items():
        expected = int(spec["case_count"])
        if actual_counts.get(category, 0) != expected:
            raise ValueError(
                f"{category} expected {expected} cases, found {actual_counts.get(category, 0)}"
            )
    if sum(actual_counts.values()) != 68:
        raise ValueError(f"Test 2 requires 68 cases, found {sum(actual_counts.values())}")
    for report_id, spec in catalog.get("reports", {}).items():
        path = (ROOT / spec["path"]).resolve()
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing report fixture {report_id}: {path}")
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON report {report_id}: {path}") from exc
    gold_reports = catalog.get("_gold_standard", {}).get("reports")
    if not isinstance(gold_reports, dict):
        raise ValueError("Test 2 gold standard requires a reports mapping")
    if set(gold_reports) != set(catalog.get("reports", {})):
        raise ValueError("Gold-standard report IDs differ from the report catalog")
    for report_id, gold in gold_reports.items():
        facts = gold.get("facts") if isinstance(gold, dict) else None
        if not isinstance(facts, list) or not facts:
            raise ValueError(f"Gold report {report_id} has no atomic facts")
        fact_ids = [str(item.get("fact_id", "")) for item in facts]
        if any(not item for item in fact_ids) or len(fact_ids) != len(set(fact_ids)):
            raise ValueError(f"Gold report {report_id} has invalid fact IDs")
    priorities = catalog.get("key_fact_metric_priority")
    if not isinstance(priorities, dict):
        raise ValueError("Test 2 requires key_fact_metric_priority")
    report_types = {item["report_type"] for item in gold_reports.values()}
    report_types.discard("enzywizard_mut_clean")
    if not report_types.issubset(priorities):
        raise ValueError(
            "Missing key-fact priorities for report types: "
            f"{sorted(report_types - set(priorities))}"
        )
    for case in cases:
        selected = _case_key_facts(catalog, case)
        if case["category_id"] == "boundary_unrelated_analysis":
            if selected:
                raise ValueError(f"{case['case_id']} boundary case has key facts")
            continue
        if not 3 <= len(selected) <= 6:
            raise ValueError(
                f"{case['case_id']} requires 3-6 key facts, found {len(selected)}"
            )
        fact_ids = [item[1]["fact_id"] for item in selected]
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError(f"{case['case_id']} has duplicate key facts")
        covered_reports = {report_id for report_id, _ in selected}
        required_reports = {
            report_id
            for report_id in case["report_ids"]
            if catalog["reports"][report_id]["source_tool"]
            != "enzywizard-mut-clean"
        }
        if not required_reports.issubset(covered_reports):
            raise ValueError(
                f"{case['case_id']} key facts do not cover all required reports"
            )


def select_cases(
    all_cases: list[dict[str, Any]], args: argparse.Namespace
) -> list[dict[str, Any]]:
    if args.case_ids:
        requested = list(dict.fromkeys(args.case_ids))
        by_id = {item["case_id"]: item for item in all_cases}
        missing = [item for item in requested if item not in by_id]
        if missing:
            raise ValueError(f"Unknown case IDs: {missing}")
        return [by_id[item] for item in requested]
    if args.profile == "full":
        return list(all_cases)
    if args.quick_case_count < 1 or args.quick_case_count > len(all_cases):
        raise ValueError("--quick_case_count must be between 1 and 68")
    rng = random.Random(args.seed)
    by_category: dict[str, list[dict[str, Any]]] = {}
    for item in all_cases:
        by_category.setdefault(item["category_id"], []).append(item)
    for values in by_category.values():
        rng.shuffle(values)
    categories = list(by_category)
    selected: list[dict[str, Any]] = []
    index = 0
    while len(selected) < args.quick_case_count:
        category = categories[index % len(categories)]
        values = by_category[category]
        if values:
            selected.append(values.pop())
        index += 1
    return selected


def _base_state(
    args: argparse.Namespace, case_dir: Path, message: str, turn_id: int = 1
) -> AgentState:
    return {
        "config": AgentConfig(
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            llm_timeout_seconds=args.llm_timeout_seconds,
            ontology_mode=args.ontology_mode,
            check_model_capabilities=False,
            workdir=case_dir,
        ),
        "session_id": f"test2-{uuid4().hex}",
        "current_turn_id": turn_id,
        "current_plan_id": 1,
        "current_user_message": message,
        "agent_inputs": AgentInputs(
            structure_file=None,
            msa_file=None,
            mut_structure_file=None,
            mut_msa_file=None,
            substrate_name=None,
            amino_acid_substitution=None,
        ),
        "runtime_session": RuntimeSession(
            started_at=utc_now(), started_monotonic=time.monotonic()
        ),
        "artifacts": [],
        "tool_executions": [],
        "analysis_results": [],
        "analysis_summary": "",
        "request_routing_results": [],
        "request_routing_summary": "",
        "user_message_summary": "",
    }


def _report_side(case: dict[str, Any], report_id: str) -> str:
    if report_id.endswith("_mut_clean"):
        return "shared"
    paired = case.get("scope") == "paired" or (
        any("_wt_" in item for item in case.get("report_ids", []))
        and any("_mut_" in item for item in case.get("report_ids", []))
    )
    if paired:
        return "mut" if "_mut_" in report_id else "wt"
    if "_mut_" in report_id:
        return "mut"
    return "single"


def _artifacts(
    catalog: dict[str, Any], case: dict[str, Any]
) -> list[Artifact]:
    output: list[Artifact] = []
    for report_id in case.get("report_ids", []):
        spec = catalog["reports"][report_id]
        output.append(
            Artifact(
                turn_id=1,
                plan_id=1,
                artifact_type=spec["artifact_type"],
                path=str((ROOT / spec["path"]).resolve()),
                source_tool=spec["source_tool"],
                side=_report_side(case, report_id),
                metadata={"benchmark_case_id": case["case_id"], "report_id": report_id},
            )
        )
    return output


def _history_result(artifacts: list[Artifact]) -> HistoryQueryResult:
    query = HistoryQueryOutput(
        turn_scope="previous",
        referenced_turn_ids=[],
        artifact_categories=["all"],
        artifact_types=sorted({item.artifact_type for item in artifacts}),
        report_types=[],
        sides=sorted({item.side for item in artifacts}),
        operations=["explain"],
        query_reason="The benchmark follow-up explicitly refers to the prior analysis.",
        clarification_question=None,
    )
    return HistoryQueryResult(
        turn_id=2,
        query_status="succeeded",
        query=query,
        resolved_turn_ids=[1],
        selected_artifacts=artifacts,
        summary="Resolved the fixed prior benchmark analysis and reports.",
        error_message="",
    )


def _run_analysis(
    catalog: dict[str, Any],
    case: dict[str, Any],
    args: argparse.Namespace,
    case_dir: Path,
) -> tuple[Any, Any | None, dict[str, Any]]:
    category = case["category_id"]
    if category == "boundary_unrelated_analysis":
        state = _base_state(args, case_dir, case["question"])
        routing = route_user_request(state)
        state["current_request_routing"] = routing
        state["request_routing_results"] = [routing]
        state["request_routing_summary"] = routing.summary
        result = analyze_agent_state(state)
        return result, None, state

    artifacts = _artifacts(catalog, case)
    if category == "multi_turn_follow_up_revision":
        state = _base_state(args, case_dir, case["initial_question"])
        state["artifacts"] = artifacts
        initial = _analyze_reports(state, artifacts, historical=False)
        state["analysis_results"] = [initial]
        state["analysis_summary"] = f"Turn 1: {initial.summary}"
        state["current_turn_id"] = 2
        state["current_user_message"] = case["follow_up_question"]
        state["current_history_query"] = _history_result(artifacts)
        result = _analyze_reports(state, artifacts, historical=True)
        return result, initial, state

    state = _base_state(args, case_dir, case["question"])
    state["artifacts"] = artifacts
    result = _analyze_reports(state, artifacts, historical=False)
    return result, None, state


def _runtime_usage(state: dict[str, Any] | None) -> dict[str, int]:
    runtime = state.get("runtime_session") if state else None
    calls = list(runtime.llm_calls) if runtime else []
    return {
        "llm_calls": len(calls),
        "input_tokens": sum(item.input_tokens or 0 for item in calls),
        "output_tokens": sum(item.output_tokens or 0 for item in calls),
        "total_tokens": sum(item.total_tokens or 0 for item in calls),
    }


def _iis_evaluator_state(
    args: argparse.Namespace, case_dir: Path
) -> AgentState:
    return {
        "config": AgentConfig(
            api_key=args.judge_api_key or args.api_key,
            base_url=args.judge_base_url,
            model=args.judge_model,
            reasoning_effort=(
                args.judge_reasoning_effort
                if args.judge_reasoning_effort is not None
                else args.reasoning_effort
            ),
            llm_timeout_seconds=args.judge_timeout_seconds,
            ontology_mode="off",
            check_model_capabilities=False,
            workdir=case_dir / "iis_evaluator",
        ),
        "session_id": f"test2-iis-{uuid4().hex}",
        "current_turn_id": 1,
        "current_user_message": "Score the fixed Test 2 IIS rubric.",
        "runtime_session": RuntimeSession(
            started_at=utc_now(), started_monotonic=time.monotonic()
        ),
    }


def _evidence_is_structurally_valid(
    evidence_id: str, evidence_by_id: dict[str, Any], seen: set[str] | None = None
) -> bool:
    item = evidence_by_id.get(evidence_id)
    if item is None:
        return False
    seen = set(seen or ())
    if evidence_id in seen:
        return False
    seen.add(evidence_id)
    if item.inference_level == "direct":
        return bool(
            item.source_tool
            and item.artifact_type
            and item.field_path
            and item.path
            and Path(item.path).is_file()
        )
    return bool(item.source_evidence_ids) and all(
        _evidence_is_structurally_valid(source_id, evidence_by_id, seen)
        for source_id in item.source_evidence_ids
    )


def _iis_dimensions(case: dict[str, Any]) -> list[dict[str, Any]]:
    if case["category_id"] == "boundary_unrelated_analysis":
        return [
            {
                "dimension_id": "boundary_recognition",
                "label": "Correctly recognizes the request boundary",
                "applicable": True,
            },
            {
                "dimension_id": "no_fabrication",
                "label": "Avoids inventing report results or evidence",
                "applicable": True,
            },
            {
                "dimension_id": "response_appropriateness",
                "label": "Provides an appropriate response for the request type",
                "applicable": True,
            },
        ]
    return [
        {
            "dimension_id": "result_integration",
            "label": (
                "Selects the relevant major results and satisfies the case-specific "
                "requirements without needing to enumerate optional metrics"
            ),
            "applicable": True,
        },
        {
            "dimension_id": "evidence_relationship",
            "label": "Correctly relates supporting, complementary, or conflicting evidence",
            "applicable": len(case.get("report_ids", [])) > 1,
        },
        {
            "dimension_id": "scientific_consistency",
            "label": "Provides a scientifically coherent interpretation consistent with reports",
            "applicable": True,
        },
        {
            "dimension_id": "inference_boundaries",
            "label": "Separates direct observations, interpretation, hypotheses, and limitations",
            "applicable": True,
        },
    ]


def _compact_report_summaries(
    result: Any, report_ids_by_path: dict[str, str]
) -> list[dict[str, Any]]:
    return [
        {
            "report_id": report_ids_by_path.get(
                str(Path(item.path).resolve()), f"REPORT-{index:02d}"
            ),
            "report_type": item.report_type,
            "source_tool": item.source_tool,
            "side": item.side,
            "artifact_name": Path(item.path).name,
            "key_metrics": dict(list(item.key_metrics.items())[:24]),
            "metric_paths": item.metric_paths,
            "highlights": item.highlights,
            "warnings": item.warnings,
        }
        for index, item in enumerate(result.report_summaries, start=1)
    ]


def _build_evaluation_packet(
    catalog: dict[str, Any],
    case: dict[str, Any],
    result: Any,
    initial: Any | None,
) -> dict[str, Any]:
    report_ids_by_path = {
        str((ROOT / spec["path"]).resolve()): report_id
        for report_id, spec in catalog["reports"].items()
    }
    evidence_by_id = {item.evidence_id: item for item in result.evidence}
    summary_by_report_id = {
        report_ids_by_path.get(str(Path(item.path).resolve())): item
        for item in result.report_summaries
        if report_ids_by_path.get(str(Path(item.path).resolve()))
    }
    metric_evidence = {
        (
            report_ids_by_path.get(str(Path(item.path).resolve())),
            item.metadata.get("metric_name"),
        ): item
        for item in result.evidence
        if item.path
        and item.metadata.get("role") == "report_metric"
        and report_ids_by_path.get(str(Path(item.path).resolve()))
    }
    gold_reports = catalog["_gold_standard"]["reports"]
    associations: list[dict[str, Any]] = []
    atomic_facts: list[dict[str, Any]] = []

    def values_match(actual: Any, gold: dict[str, Any]) -> bool:
        expected = gold["expected_value"]
        if (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and isinstance(expected, (int, float))
            and not isinstance(expected, bool)
        ):
            return math.isclose(
                float(actual),
                float(expected),
                rel_tol=float(gold.get("numeric_rel_tolerance") or 0.0),
                abs_tol=float(gold.get("numeric_abs_tolerance") or 0.0),
            )
        return str(actual).strip().casefold() == str(expected).strip().casefold()

    selected_key_facts = _case_key_facts(catalog, case)
    gold_by_metric = {
        (report_id, item["metric_name"]): item
        for report_id, item in selected_key_facts
    }
    selected_by_report: dict[str, list[dict[str, Any]]] = {}
    for report_id, item in selected_key_facts:
        selected_by_report.setdefault(report_id, []).append(item)
    supported_by_metric: dict[tuple[str, str], bool] = {}
    association_by_metric: dict[tuple[str, str], str] = {}

    def gold_for_metric(report_id: str, metric_name: str) -> dict[str, Any] | None:
        exact = gold_by_metric.get((report_id, metric_name))
        if exact is not None:
            return exact
        normalized = metric_name.replace("_", " ").strip().casefold()
        for candidate in selected_by_report.get(report_id, []):
            accepted = {
                candidate["metric_name"].replace("_", " ").strip().casefold(),
                *{
                    str(alias).strip().casefold()
                    for alias in candidate.get("accepted_aliases", [])
                },
            }
            if normalized in accepted:
                return candidate
        return None

    # Hybrid-mode proxy denominator: selected structured metrics emitted by the Agent.
    for report_id in case.get("report_ids", []):
        summary = summary_by_report_id.get(report_id)
        selected_metrics = {
            item["metric_name"] for item in selected_by_report.get(report_id, [])
        }
        actual_metrics = (
            [
                (name, value)
                for name, value in summary.key_metrics.items()
                if name in selected_metrics
            ]
            if summary
            else []
        )
        for metric_name, actual_value in actual_metrics:
            gold = gold_for_metric(report_id, metric_name)
            evidence = metric_evidence.get((report_id, metric_name))
            structural = bool(
                evidence is not None
                and _evidence_is_structurally_valid(
                    evidence.evidence_id, evidence_by_id
                )
            )
            field_matches = bool(
                gold is not None
                and evidence is not None
                and evidence.field_path == gold["field_path"]
            )
            expected_side = _report_side(case, report_id)
            side_matches = bool(
                evidence is not None and evidence.side == expected_side
            )
            source_matches = bool(
                evidence is not None
                and evidence.source_tool == gold_reports[report_id]["source_tool"]
                and evidence.artifact_type
                == gold_reports[report_id]["artifact_type"]
            )
            value_matches = gold is not None and values_match(actual_value, gold)
            supported = (
                structural
                and field_matches
                and side_matches
                and source_matches
                and value_matches
            )
            association_id = f"EA-{len(associations) + 1:03d}"
            key = (report_id, metric_name)
            supported_by_metric[key] = supported
            association_by_metric[key] = association_id
            if gold is not None:
                gold_key = (report_id, gold["metric_name"])
                supported_by_metric[gold_key] = supported
                association_by_metric[gold_key] = association_id
            associations.append(
                {
                    "association_id": association_id,
                    "fact_id": gold["fact_id"] if gold else f"unmatched:{report_id}:{metric_name}",
                    "report_id": report_id,
                    "metric_name": metric_name,
                    "expected_value": gold["expected_value"] if gold else None,
                    "actual_value": actual_value,
                    "accepted_aliases": gold.get("accepted_aliases", []) if gold else [],
                    "evidence_id": evidence.evidence_id if evidence else None,
                    "source_tool": evidence.source_tool if evidence else None,
                    "side": evidence.side if evidence else None,
                    "expected_side": expected_side,
                    "field_path": evidence.field_path if evidence else None,
                    "expected_field_path": gold["field_path"] if gold else None,
                    "structurally_valid": structural,
                    "field_matches": field_matches,
                    "side_matches": side_matches,
                    "source_matches": source_matches,
                    "value_matches": value_matches,
                    "automatically_supported": supported,
                }
            )
            fact = {
                "fact_id": gold["fact_id"] if gold else f"unmatched:{report_id}:{metric_name}",
                "report_id": report_id,
                "metric_name": metric_name,
                "expected_value": gold["expected_value"] if gold else None,
                "actual_value": actual_value,
                "field_path": gold["field_path"] if gold else None,
                "accepted_aliases": gold.get("accepted_aliases", []) if gold else [],
                "evidence_association_ids": [association_id],
                "automatically_supported": supported,
            }
            atomic_facts.append(fact)

    # Mutation identity is a case-critical fact present in the raw mut-clean report,
    # but it is not currently emitted as a summarized report metric.
    for report_id, gold in selected_key_facts:
        if not (
            gold.get("synthetic_case_fact")
            or gold.get("derived_comparison_fact")
        ):
            continue
        spec = catalog["reports"][report_id]
        if gold.get("derived_comparison_fact"):
            actual_value = gold["expected_value"]
            supported = True
            source_tool = "derived-report-comparison"
            side = "shared"
        else:
            payload = json.loads((ROOT / spec["path"]).read_text(encoding="utf-8"))
            actual_value = payload.get(gold["metric_name"])
            supported = values_match(actual_value, gold)
            source_tool = spec["source_tool"]
            side = "shared"
        association_id = f"EA-{len(associations) + 1:03d}"
        key = (report_id, gold["metric_name"])
        supported_by_metric[key] = supported
        association_by_metric[key] = association_id
        associations.append(
            {
                "association_id": association_id,
                "fact_id": gold["fact_id"],
                "report_id": report_id,
                "metric_name": gold["metric_name"],
                "expected_value": gold["expected_value"],
                "actual_value": actual_value,
                "accepted_aliases": gold["accepted_aliases"],
                "evidence_id": None,
                "source_tool": source_tool,
                "side": side,
                "expected_side": side,
                "field_path": gold["field_path"],
                "expected_field_path": gold["field_path"],
                "structurally_valid": True,
                "field_matches": True,
                "side_matches": True,
                "source_matches": True,
                "value_matches": supported,
                "automatically_supported": supported,
            }
        )
        atomic_facts.append(
            {
                "fact_id": gold["fact_id"],
                "report_id": report_id,
                "metric_name": gold["metric_name"],
                "expected_value": gold["expected_value"],
                "actual_value": actual_value,
                "field_path": gold["field_path"],
                "accepted_aliases": gold["accepted_aliases"],
                "evidence_association_ids": [association_id],
                "automatically_supported": supported,
            }
        )

    # KFR denominator: every predeclared gold fact assigned to this case.
    required_facts: list[dict[str, Any]] = []
    for (report_id, metric_name), gold in gold_by_metric.items():
        supported = supported_by_metric.get((report_id, metric_name), False)
        association_id = association_by_metric.get((report_id, metric_name))
        required_facts.append(
            {
                "fact_id": gold["fact_id"],
                "report_id": report_id,
                "metric_name": metric_name,
                "expected_value": gold["expected_value"],
                "field_path": gold["field_path"],
                "accepted_aliases": gold.get("accepted_aliases", []),
                "supporting_report_ids": gold.get(
                    "supporting_report_ids", [report_id]
                ),
                "evidence_association_ids": (
                    [association_id] if association_id else []
                ),
                "requirement_context": case.get("required_facts", []),
                "automatically_covered": supported,
                "evidence_supported": supported,
            }
        )
    evidence_by_id = {item.evidence_id: item for item in result.evidence}
    referenced_evidence_ids = {
        evidence_id
        for finding in result.findings
        for evidence_id in finding.evidence_ids
    }
    pending = list(referenced_evidence_ids)
    while pending:
        evidence_id = pending.pop()
        item = evidence_by_id.get(evidence_id)
        if item is None:
            continue
        for source_id in item.source_evidence_ids:
            if source_id not in referenced_evidence_ids:
                referenced_evidence_ids.add(source_id)
                pending.append(source_id)
    candidate_evidence_registry = []
    for evidence_id in sorted(referenced_evidence_ids):
        item = evidence_by_id.get(evidence_id)
        if item is None:
            continue
        report_id = (
            report_ids_by_path.get(str(Path(item.path).resolve()))
            if item.path
            else None
        )
        candidate_evidence_registry.append(
            {
                "evidence_id": item.evidence_id,
                "report_id": report_id,
                "source_kind": item.source_kind,
                "inference_level": item.inference_level,
                "source_tool": item.source_tool,
                "artifact_type": item.artifact_type,
                "side": item.side,
                "turn_id": item.turn_id,
                "field_path": item.field_path,
                "value_summary": item.value_summary,
                "source_evidence_ids": item.source_evidence_ids,
            }
        )

    question = case.get("follow_up_question") or case.get("question")
    return {
        "case_id": case["case_id"],
        "category_id": case["category_id"],
        "question": question,
        "initial_question": case.get("initial_question"),
        "initial_answer": (
            _scientific_answer_body(initial.final_answer)
            if initial is not None
            else None
        ),
        "candidate_answer": _scientific_answer_body(result.final_answer),
        "candidate_limitations": result.limitations,
        "expected_behavior": case.get("expected_behavior"),
        "required_limitations": case.get("required_limitations", []),
        "forbidden_claims": case.get("forbidden_claims", []),
        "iis_requirements": {
            "must_address": case.get("required_facts", []),
            "required_limitations": case.get("required_limitations", []),
            "forbidden_claims": case.get("forbidden_claims", []),
            "optional_metric_policy": (
                "Report metrics are available evidence, not an exhaustive narrative "
                "checklist. Omission is not penalized unless required by the question "
                "or must_address entries."
            ),
        },
        "report_summaries": _compact_report_summaries(result, report_ids_by_path),
        "atomic_facts": atomic_facts,
        "required_facts": required_facts,
        "iis_dimensions": _iis_dimensions(case),
        "evidence_associations": associations,
        "candidate_evidence_registry": candidate_evidence_registry,
    }


def _iis_review_packet(packet: dict[str, Any]) -> dict[str, Any]:
    """Return the narrative-review view without automatic-metric gold labels."""
    return {
        key: packet[key]
        for key in (
            "case_id",
            "category_id",
            "question",
            "initial_question",
            "initial_answer",
            "candidate_answer",
            "candidate_limitations",
            "expected_behavior",
            "iis_requirements",
            "report_summaries",
            "iis_dimensions",
        )
    }


def _full_review_packet(packet: dict[str, Any]) -> dict[str, Any]:
    """Return all information needed for item-level semantic metric review."""
    output = _iis_review_packet(packet)
    output["objective_metrics_applicable"] = (
        packet.get("category_id") != "boundary_unrelated_analysis"
    )
    output["kfr_key_facts"] = [
        {
            "fact_id": item["fact_id"],
            "report_id": item["report_id"],
            "metric_name": item["metric_name"],
            "expected_value": item["expected_value"],
            "field_path": item["field_path"],
            "accepted_aliases": item.get("accepted_aliases", []),
            "supporting_report_ids": item.get(
                "supporting_report_ids", [item["report_id"]]
            ),
        }
        for item in packet["required_facts"]
    ]
    output["candidate_evidence_registry"] = packet.get(
        "candidate_evidence_registry", []
    )
    return output


def _display_packet(packet: dict[str, Any]) -> None:
    packet = _iis_review_packet(packet)
    print("\n" + "=" * 78, flush=True)
    print(
        f"Human evaluation: {packet['case_id']} ({packet['category_id']})",
        flush=True,
    )
    if packet.get("initial_question"):
        print(f"\nInitial question:\n{packet['initial_question']}", flush=True)
        print(f"\nInitial answer:\n{packet['initial_answer']}", flush=True)
    print(f"\nQuestion:\n{packet['question']}", flush=True)
    print(f"\nCandidate answer:\n{packet['candidate_answer']}", flush=True)
    if packet["candidate_limitations"]:
        print("\nCandidate limitations:", flush=True)
        for item in packet["candidate_limitations"]:
            print(f"  - {item}", flush=True)
    if packet["report_summaries"]:
        print("\nReport summaries:", flush=True)
        for report in packet["report_summaries"]:
            metrics = json.dumps(
                report["key_metrics"], ensure_ascii=False, default=str
            )
            if len(metrics) > 2400:
                metrics = metrics[:2397] + "..."
            print(
                f"  [{report['report_id']}] {report['source_tool']} "
                f"side={report['side']} fields={metrics}",
                flush=True,
            )
    if packet.get("expected_behavior"):
        print(
            f"\nExpected boundary behavior:\n  - {packet['expected_behavior']}",
            flush=True,
        )
    requirements = packet["iis_requirements"]
    if requirements["must_address"]:
        print("\nCase-specific requirements:", flush=True)
        for item in requirements["must_address"]:
            print(f"  - {item}", flush=True)
    if requirements["required_limitations"]:
        print("\nRequired limitations:", flush=True)
        for item in requirements["required_limitations"]:
            print(f"  - {item}", flush=True)
    if requirements["forbidden_claims"]:
        print("\nForbidden claims:", flush=True)
        for item in requirements["forbidden_claims"]:
            print(f"  - {item}", flush=True)
    print(
        "\nReview policy: score relevance and scientific quality, not length; "
        "available metrics are not an exhaustive checklist.",
        flush=True,
    )


def human_review(
    packet: dict[str, Any], input_fn: Any = input, *, display: bool = True
) -> dict[str, Any]:
    if display:
        _display_packet(packet)
    dimensions = packet["iis_dimensions"]
    applicable = [item for item in dimensions if item["applicable"]]
    print(
        "\n[IIS human rubric] Score each applicable dimension: "
        "0=incorrect/missing, 1=partial, 2=complete and correct",
        flush=True,
    )
    for index, item in enumerate(applicable, start=1):
        print(f"  {index}. {item['label']}", flush=True)
    while True:
        raw = input_fn(
            f"Enter {len(applicable)} scores separated by spaces: "
        ).strip()
        try:
            scores = [int(item) for item in raw.split()]
            if len(scores) != len(applicable) or any(
                item not in {0, 1, 2} for item in scores
            ):
                raise ValueError
            break
        except ValueError:
            print("  Enter exactly one 0/1/2 score per applicable dimension.")
    score_by_id = {
        item["dimension_id"]: score for item, score in zip(applicable, scores)
    }
    interpretation = [
        {
            "dimension_id": item["dimension_id"],
            "applicable": item["applicable"],
            "score": score_by_id.get(item["dimension_id"]),
            "reason": "Human review",
        }
        for item in dimensions
    ]
    return {
        "evaluation_mode": "automatic_plus_human_iis",
        "interpretation_judgments": interpretation,
        "overall_notes": "Only IIS was scored by the human reviewer.",
    }


def _human_boolean(input_fn: Any, prompt: str) -> bool:
    while True:
        value = input_fn(prompt + " [y/n]: ").strip().casefold()
        if value in {"y", "yes", "1"}:
            return True
        if value in {"n", "no", "0"}:
            return False
        print("  Enter y or n.", flush=True)


def human_full_review(
    packet: dict[str, Any], input_fn: Any = input
) -> dict[str, Any]:
    """Collect auditable item-level human judgments for all four metrics."""
    _display_packet(packet)
    print(
        "\n[FP] Enter every independently verifiable atomic claim in the candidate "
        "answer. Press Enter on an empty line when finished.",
        flush=True,
    )
    factual: list[dict[str, Any]] = []
    while True:
        claim = input_fn("Atomic claim: ").strip()
        if not claim:
            break
        supported = _human_boolean(
            input_fn, "Directly supported by the supplied reports?"
        )
        factual.append(
            {
                "claim_id": f"HUMAN-CLAIM-{len(factual) + 1:03d}",
                "claim": claim,
                "supported": supported,
                "reason": "Human full review",
            }
        )

    print("\n[KFR] Review each predefined required fact.", flush=True)
    key_facts: list[dict[str, Any]] = []
    for item in packet["required_facts"]:
        source_ids = ", ".join(
            item.get("supporting_report_ids", [item["report_id"]])
        )
        label = (
            f"{item['report_id']} :: {item['metric_name']}="
            f"{item['expected_value']} ({item['field_path']}; sources={source_ids})"
        )
        print(f"  {label}", flush=True)
        key_facts.append(
            {
                "fact_id": item["fact_id"],
                "covered": _human_boolean(
                    input_fn, "Semantically covered by the candidate answer?"
                ),
                "reason": "Human full review",
            }
        )

    print(
        "\n[EA precision] Enter every explicit claim-evidence link in the candidate "
        "answer. Use one atomic claim plus one distinct source/field per link, deduplicate "
        "identical repeated links, and press Enter on an empty claim when finished.",
        flush=True,
    )
    evidence_associations: list[dict[str, Any]] = []
    while True:
        claim = input_fn("Linked claim: ").strip()
        if not claim:
            break
        evidence_reference = input_fn("Evidence reference: ").strip()
        while not evidence_reference:
            print("  Evidence reference cannot be empty.", flush=True)
            evidence_reference = input_fn("Evidence reference: ").strip()
        evidence_associations.append(
            {
                "association_id": f"HUMAN-EA-{len(evidence_associations) + 1:03d}",
                "claim": claim,
                "evidence_reference": evidence_reference,
                "supported": _human_boolean(
                    input_fn, "Does this resolvable evidence support the claim?"
                ),
                "reason": "Human full review",
            }
        )

    print("\n[EA recall] Check evidence coverage for each required fact.", flush=True)
    evidence_coverage: list[dict[str, Any]] = []
    for item in packet["required_facts"]:
        print(
            f"  {item['fact_id']}: {item['metric_name']}={item['expected_value']}",
            flush=True,
        )
        evidence_coverage.append(
            {
                "fact_id": item["fact_id"],
                "evidence_supported": _human_boolean(
                    input_fn,
                    "Covered and linked to correct, resolvable supporting evidence?",
                ),
                "reason": "Human full review",
            }
        )

    iis = human_review(packet, input_fn=input_fn, display=False)
    return {
        "evaluation_mode": "full_review_human",
        "factual_judgments": factual,
        "key_fact_judgments": key_facts,
        "evidence_association_judgments": evidence_associations,
        "evidence_coverage_judgments": evidence_coverage,
        "interpretation_judgments": iis["interpretation_judgments"],
        "overall_notes": "All four metrics received item-level human review.",
    }


def llm_iis_review(
    packet: dict[str, Any], evaluator_state: AgentState
) -> dict[str, Any]:
    review_packet = _iis_review_packet(packet)
    errors: list[str] = []
    for attempt in range(2):
        messages = [
            SystemMessage(content=IIS_JUDGE_PROMPT),
            HumanMessage(content=json.dumps(review_packet, ensure_ascii=False)),
        ]
        if errors:
            messages.append(
                HumanMessage(
                    content=(
                        "Your previous response violated the output contract: "
                        + "; ".join(errors)
                        + ". Return every required item exactly once."
                    )
                )
            )
        output = invoke_structured(
            evaluator_state,
            stage=f"evaluation.test2.iis_judge.attempt_{attempt + 1}",
            schema=IISReviewOutput,
            llm=create_chat_model_for_state(evaluator_state),
            messages=messages,
        )
        review = {
            "evaluation_mode": "automatic_plus_llm_iis",
            **output.model_dump(mode="json"),
        }
        errors = validate_iis_review(packet, review)
        if not errors:
            return review
    raise ValueError("LLM IIS reviewer violated contract: " + "; ".join(errors))


def llm_full_review(
    packet: dict[str, Any], evaluator_state: AgentState
) -> dict[str, Any]:
    review_packet = _full_review_packet(packet)
    errors: list[str] = []
    for attempt in range(2):
        messages = [
            SystemMessage(content=FULL_REVIEW_PROMPT),
            HumanMessage(content=json.dumps(review_packet, ensure_ascii=False)),
        ]
        if errors:
            messages.append(
                HumanMessage(
                    content=(
                        "Your previous response violated the output contract: "
                        + "; ".join(errors)
                        + ". Return every required item exactly once."
                    )
                )
            )
        output = invoke_structured(
            evaluator_state,
            stage=f"evaluation.test2.full_review_judge.attempt_{attempt + 1}",
            schema=FullReviewOutput,
            llm=create_chat_model_for_state(evaluator_state),
            messages=messages,
        )
        review = {
            "evaluation_mode": "full_review_llm",
            **output.model_dump(mode="json"),
        }
        errors = validate_full_review(packet, review)
        if not errors:
            return review
    raise ValueError("LLM full reviewer violated contract: " + "; ".join(errors))


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def run_case(
    catalog: dict[str, Any],
    case: dict[str, Any],
    repetition: int,
    args: argparse.Namespace,
    run_dir: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    case_dir = run_dir / case["case_id"] / f"repeat_{repetition:03d}"
    case_dir.mkdir(parents=True, exist_ok=False)
    result = None
    initial = None
    state: dict[str, Any] | None = None
    evaluator_state: dict[str, Any] | None = None
    packet: dict[str, Any] | None = None
    review: dict[str, Any] | None = None
    metrics = {name: None for name in METRIC_NAMES}
    details: dict[str, Any] = {}
    errors: list[str] = []
    warnings: list[str] = []
    try:
        result, initial, state = _run_analysis(catalog, case, args, case_dir)
        _atomic_json(
            case_dir / "analysis_result.json", result.model_dump(mode="json")
        )
        if initial is not None:
            _atomic_json(
                case_dir / "initial_analysis_result.json",
                initial.model_dump(mode="json"),
            )
        if result.analysis_status != "succeeded":
            errors.append(
                "Agent analysis did not succeed: "
                + "; ".join(result.error_messages or [result.summary])
            )
        if initial is not None and initial.analysis_status != "succeeded":
            errors.append(
                "Initial Agent analysis did not succeed: "
                + "; ".join(initial.error_messages or [initial.summary])
            )
        packet = _build_evaluation_packet(catalog, case, result, initial)
        _atomic_json(case_dir / "evaluation_packet.json", packet)
        if args.evaluation_mode == "full_review" and args.iis_evaluator == "human":
            review = human_full_review(packet)
        elif args.evaluation_mode == "full_review":
            evaluator_state = _iis_evaluator_state(args, case_dir)
            review = llm_full_review(packet, evaluator_state)
        elif args.iis_evaluator == "human":
            review = human_review(packet)
        else:
            evaluator_state = _iis_evaluator_state(args, case_dir)
            review = llm_iis_review(packet, evaluator_state)
        if args.evaluation_mode == "full_review":
            metrics, details = score_case_full_review(packet, review)
        else:
            metrics, details = score_case(packet, review)
        _atomic_json(case_dir / "review.json", review)
    except (EOFError, KeyboardInterrupt):
        raise
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    passed = not errors and review is not None
    return {
        "case_id": case["case_id"],
        "category_id": case["category_id"],
        "enzyme_id": case.get("enzyme_id"),
        "repetition": repetition,
        "evaluation_mode": (
            f"full_review_{args.iis_evaluator}"
            if args.evaluation_mode == "full_review"
            else f"automatic_plus_{args.iis_evaluator}_iis"
        ),
        "passed": passed,
        "errors": errors,
        "warnings": warnings,
        "metrics": metrics,
        "metric_details": details,
        "review": review,
        "analysis_result": result.model_dump(mode="json") if result else None,
        "evaluation_packet": packet,
        "usage": {
            "agent": _runtime_usage(state),
            "iis_evaluator": _runtime_usage(evaluator_state),
        },
        "duration_seconds": round(time.monotonic() - started, 3),
        "case_workdir": str(case_dir),
    }


def _summary(
    results: list[dict[str, Any]],
    args: argparse.Namespace,
    *,
    confidence_intervals: bool,
) -> dict[str, Any]:
    statistics = repetition_statistics(results)
    if confidence_intervals and args.bootstrap_iterations:
        intervals = cluster_bootstrap_ci(
            results,
            iterations=args.bootstrap_iterations,
            seed=args.seed + 1,
        )
        for metric, interval in intervals.items():
            statistics[metric]["bootstrap_95_ci"] = interval
    by_category: dict[str, Any] = {}
    for category in sorted({item["category_id"] for item in results}):
        subset = [item for item in results if item["category_id"] == category]
        category_stats = repetition_statistics(subset)
        if confidence_intervals and args.bootstrap_iterations:
            intervals = cluster_bootstrap_ci(
                subset,
                iterations=args.bootstrap_iterations,
                seed=args.seed + sum(ord(character) for character in category),
            )
            for metric, interval in intervals.items():
                category_stats[metric]["bootstrap_95_ci"] = interval
        by_category[category] = {
            "case_count": len({item["case_id"] for item in subset}),
            "metrics": category_stats,
        }
    usage = {
        group: {
            key: sum(item["usage"][group][key] for item in results)
            for key in ("llm_calls", "input_tokens", "output_tokens", "total_tokens")
        }
        for group in ("agent", "iis_evaluator")
    }
    return {
        "case_count": len({item["case_id"] for item in results}),
        "run_count": len(results),
        "completed_runs": sum(bool(item["passed"]) for item in results),
        "completion_rate": (
            sum(bool(item["passed"]) for item in results) / len(results)
            if results
            else None
        ),
        "metrics": statistics,
        "by_category": by_category,
        "usage": usage,
        "total_duration_seconds": sum(item["duration_seconds"] for item in results),
    }


def _write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    fields = [
        "case_id",
        "category_id",
        "enzyme_id",
        "repetition",
        "evaluation_mode",
        "passed",
        "FP",
        "KFR",
        "IIS",
        "EA_F1",
        "duration_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in results:
            writer.writerow(
                {
                    key: item.get(key)
                    for key in fields
                    if key not in METRIC_NAMES
                }
                | item["metrics"]
            )


def _metadata(
    args: argparse.Namespace,
    catalog: dict[str, Any],
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "test_id": "test_2_result_interpretation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "base_url": args.base_url,
        "reasoning_effort": args.reasoning_effort,
        "llm_timeout_seconds": args.llm_timeout_seconds,
        "ontology_mode": args.ontology_mode,
        "evaluation_mode": (
            f"full_review_all_metrics_by_{args.iis_evaluator}"
            if args.evaluation_mode == "full_review"
            else f"automatic_fp_kfr_ea_f1_plus_{args.iis_evaluator}_iis"
        ),
        "metric_evaluation_mode": args.evaluation_mode,
        "iis_evaluator": args.iis_evaluator,
        "judge_model": (
            args.judge_model
            if args.iis_evaluator == "llm"
            else None
        ),
        "judge_base_url": (
            args.judge_base_url
            if args.iis_evaluator == "llm"
            else None
        ),
        "judge_reasoning_effort": (
            args.judge_reasoning_effort or args.reasoning_effort
            if args.iis_evaluator == "llm"
            else None
        ),
        "judge_timeout_seconds": (
            args.judge_timeout_seconds if args.iis_evaluator == "llm" else None
        ),
        "automatic_evaluation_policy": catalog["automatic_evaluation"],
        "repeat": args.repeat,
        "profile": args.profile,
        "seed": args.seed,
        "selected_case_ids": [item["case_id"] for item in cases],
        "bootstrap_iterations": args.bootstrap_iterations,
    }


def main() -> int:
    args = build_parser().parse_args()
    if args.repeat < 1:
        raise SystemExit("--repeat must be positive")
    if args.llm_timeout_seconds <= 0 or args.judge_timeout_seconds <= 0:
        raise SystemExit("LLM timeout values must be positive")
    if args.bootstrap_iterations != 0 and args.bootstrap_iterations < 100:
        raise SystemExit("--bootstrap_iterations must be 0 or at least 100")
    catalog, all_cases = load_suite()
    cases = select_cases(all_cases, args)
    if args.validate_only:
        print("[validation passed] Test 2 cases and report fixtures are valid.")
        print(f"  total_cases={len(all_cases)}")
        for category, spec in catalog["categories"].items():
            print(f"  {category}={spec['case_count']}")
        return 0
    if not args.api_key:
        raise SystemExit("--api_key or ENZYWIZARD_API_KEY is required")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = (
        Path(args.workdir).expanduser().resolve()
        / f"run_{stamp}_{uuid4().hex[:8]}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else run_dir / "results.json"
    )

    capability_state: AgentState = {
        "config": AgentConfig(
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            ontology_mode="off",
            workdir=run_dir / "_model_capability",
        ),
        "session_id": f"test2-contract-{uuid4().hex}",
        "current_turn_id": 1,
        "current_user_message": "Model contract capability probe.",
        "runtime_session": RuntimeSession(
            started_at=utc_now(), started_monotonic=time.monotonic()
        ),
    }
    Path(capability_state["config"].workdir).mkdir(parents=True, exist_ok=True)
    try:
        capability = check_model_capabilities(capability_state).model_dump(mode="json")
    except Exception as exc:
        print(f"[model contract failed] {exc}", flush=True)
        return 2

    print(
        f"[Test 2] cases={len(cases)} repeat={args.repeat} model={args.model} "
        f"reasoning={args.reasoning_effort} profile={args.profile} "
        f"evaluation={args.evaluation_mode}+{args.iis_evaluator} seed={args.seed}",
        flush=True,
    )
    results: list[dict[str, Any]] = []
    payload: dict[str, Any] = {}
    try:
        for case in cases:
            for repetition in range(1, args.repeat + 1):
                question = case.get("follow_up_question") or case.get("question")
                print(
                    f"\n[{case['case_id']}] {repetition}/{args.repeat} "
                    f"{case['category_id']}",
                    flush=True,
                )
                print(f"  Question: {question}", flush=True)
                result = run_case(catalog, case, repetition, args, run_dir)
                results.append(result)
                print(f"  Metrics: {result['metrics']}", flush=True)
                for warning in result["warnings"]:
                    print(f"  Warning: {warning}", flush=True)
                for error in result["errors"]:
                    print(f"  Error: {error}", flush=True)
                payload = {
                    "metadata": _metadata(args, catalog, cases),
                    "model_capability": capability,
                    "summary": _summary(
                        results, args, confidence_intervals=False
                    ),
                    "results": results,
                }
                _atomic_json(output, payload)
                if args.stop_on_failure and not result["passed"]:
                    payload["summary"] = _summary(
                        results, args, confidence_intervals=True
                    )
                    _atomic_json(output, payload)
                    _write_csv(output.with_suffix(".csv"), results)
                    return 1
    except (EOFError, KeyboardInterrupt):
        print("\n[stopped] Existing completed cases were saved.", flush=True)
        if results:
            payload["summary"] = _summary(
                results, args, confidence_intervals=True
            )
            _atomic_json(output, payload)
            _write_csv(output.with_suffix(".csv"), results)
        return 130

    payload["summary"] = _summary(results, args, confidence_intervals=True)
    _atomic_json(output, payload)
    _write_csv(output.with_suffix(".csv"), results)
    print(f"\n[complete] {output}", flush=True)
    return 0 if all(item["passed"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
