"""Fixed Test 2 cases, Agent inputs, and judge materials."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time
from typing import Any
from unittest.mock import patch
from uuid import uuid4

import yaml

from enzywizard_agent.conf import AgentConfig
from enzywizard_agent.data_models.analysis import AnalysisResult, ReportSummary
from enzywizard_agent.data_models.agent_inputs import AgentInputs
from enzywizard_agent.data_models.artifact import Artifact
from enzywizard_agent.data_models.history_query import HistoryQueryOutput, HistoryQueryResult
from enzywizard_agent.data_models.runtime import RuntimeSession
from enzywizard_agent.services import analysis_service
from enzywizard_agent.services.analysis_service import (
    _analyze_reports,
    scientific_review_answer,
)
from enzywizard_agent.services.runtime_service import utc_now
from enzywizard_agent.state import AgentState
from enzywizard_agent.utils.analysis_utils import summarize_report_artifact
from enzywizard_agent.utils.terminology_utils import (
    CLAIM_EVIDENCE_APPENDIX_MARKER,
    TERMINOLOGY_APPENDIX_MARKER,
)


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
CATALOG_PATH = ROOT / "case_catalog.yaml"
INDEX_PATH = ROOT / "cases" / "index.yaml"
GOLD_STANDARD_PATH = ROOT / "gold_report_facts.json"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-6-luna"
DEFAULT_REASONING_EFFORT = "medium"


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
    """Resolve the fixed facts explicitly listed for this question."""
    case_id = case["case_id"]
    fact_ids = catalog["case_key_fact_overrides"][case_id]["fact_ids"]
    if not fact_ids:
        return []

    gold_reports = catalog["_gold_standard"]["reports"]
    report_ids = case["report_ids"]
    available = {
        fact["fact_id"]: (report_id, fact)
        for report_id in report_ids
        for fact in gold_reports[report_id]["facts"]
    }
    for report_id in report_ids:
        mutation_id = f"{report_id}:mutation_identity"
        if mutation_id in fact_ids and catalog["reports"][report_id]["source_tool"] == "enzywizard-mut-clean":
            available[mutation_id] = (
                report_id,
                {
                    "fact_id": mutation_id,
                    "metric_name": "amino_acid_substitution",
                    "field_path": "$.amino_acid_substitution",
                    "expected_value": catalog["enzymes"][case["enzyme_id"]]["mutation"],
                    "accepted_aliases": ["mutation identity", "amino acid substitution"],
                    "synthetic_case_fact": True,
                },
            )

    comparison_prefix = f"{case_id}:comparison:"
    for fact_id in fact_ids:
        if not fact_id.startswith(comparison_prefix):
            continue
        metric = fact_id.removeprefix(comparison_prefix)
        sources = {
            _report_side(case, report_id): (report_id, fact["expected_value"])
            for report_id in report_ids
            for fact in gold_reports[report_id]["facts"]
            if fact["metric_name"] == metric
        }
        if "wt" not in sources or "mut" not in sources:
            raise ValueError(f"{case_id} has no paired source for {metric}")
        wt_report_id, wt = sources["wt"]
        mut_report_id, mut = sources["mut"]
        if not isinstance(wt, (int, float)) or not isinstance(mut, (int, float)):
            raise ValueError(f"{case_id} requires numeric comparison values for {metric}")
        relation = "higher than" if mut > wt else "lower than" if mut < wt else "equal to"
        available[fact_id] = (
            wt_report_id,
            {
                "fact_id": fact_id,
                "metric_name": f"comparison::{metric}",
                "field_path": f"$comparison.{metric}",
                "expected_value": f"MUT {metric} ({mut}) is {relation} WT ({wt})",
                "accepted_aliases": [f"WT MUT {metric} direction", f"{metric} comparison"],
                "supporting_report_ids": [wt_report_id, mut_report_id],
                "derived_comparison_fact": True,
            },
        )

    missing = set(fact_ids) - available.keys()
    if missing:
        raise ValueError(f"{case_id} has unavailable key facts: {sorted(missing)}")
    return [available[fact_id] for fact_id in fact_ids]


def _validate_suite(catalog: dict[str, Any], cases: list[dict[str, Any]]) -> None:
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
        if not reports or not case.get("required_facts"):
            raise ValueError(f"{case['case_id']} requires reports and required_facts")
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
    if actual_counts.keys() != expected_categories.keys():
        raise ValueError("Case categories differ from the report-case catalog")
    for category, spec in expected_categories.items():
        expected = int(spec["case_count"])
        if actual_counts.get(category, 0) != expected:
            raise ValueError(
                f"{category} expected {expected} cases, found {actual_counts.get(category, 0)}"
            )
    if sum(actual_counts.values()) != 60:
        raise ValueError(f"Test 2 requires 60 cases, found {sum(actual_counts.values())}")
    for report_id, spec in catalog.get("reports", {}).items():
        path = (ROOT / spec["path"]).resolve()
        if not path.is_relative_to(ROOT / "resources") or not path.is_file() or path.stat().st_size == 0:
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
    overrides = catalog.get("case_key_fact_overrides", {})
    report_cases = {
        case["case_id"]: case for case in cases
    }
    if not isinstance(overrides, dict) or overrides.keys() != report_cases.keys():
        raise ValueError("Every report case requires an explicit scoring standard")
    for case_id, override in overrides.items():
        fact_ids = override.get("fact_ids") if isinstance(override, dict) else None
        if (
            not isinstance(fact_ids, list)
            or any(not isinstance(item, str) or not item.strip() for item in fact_ids)
            or len(fact_ids) != len(set(fact_ids))
            or len(fact_ids) > 6
        ):
            raise ValueError(f"{case_id} has an invalid key-fact override")
        requirements = override.get("requirements", [])
        if not isinstance(requirements, list) or not 1 <= len(fact_ids) + len(requirements) <= 6:
            raise ValueError(f"{case_id} has invalid semantic requirements")
        requirement_ids: set[str] = set()
        for requirement in requirements:
            if not isinstance(requirement, dict) or any(
                not isinstance(requirement.get(key), str) or not requirement[key].strip()
                for key in ("id", "acceptance", "evidence")
            ):
                raise ValueError(f"{case_id} has a malformed semantic requirement")
            requirement_id = requirement["id"]
            sources = requirement.get("report_ids")
            if (
                requirement_id in requirement_ids
                or not isinstance(sources, list)
                or not sources
                or any(source not in report_cases[case_id]["report_ids"] for source in sources)
            ):
                raise ValueError(f"{case_id} has invalid scope for requirement {requirement_id}")
            requirement_ids.add(requirement_id)
        _case_key_facts(catalog, report_cases[case_id])


def select_cases(
    all_cases: list[dict[str, Any]], args: argparse.Namespace
) -> list[dict[str, Any]]:
    if args.case_ids:
        requested = args.case_ids
        if len(requested) != len(set(requested)):
            raise ValueError("--case_ids must not contain duplicates")
        by_id = {item["case_id"]: item for item in all_cases}
        missing = [item for item in requested if item not in by_id]
        if missing:
            raise ValueError(f"Unknown case IDs: {missing}")
        return [by_id[item] for item in requested]
    if args.profile == "full":
        return list(all_cases)
    if args.quick_case_count < 1 or args.quick_case_count > len(all_cases):
        raise ValueError("--quick_case_count must be between 1 and 60")
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


def _summary_with_mutation_mapping(artifact: Artifact) -> ReportSummary:
    """Expose the requested site from the fixed report with its original paths."""
    summary = summarize_report_artifact(artifact)
    if artifact.source_tool != "enzywizard-mut-clean" or summary.read_status != "parsed":
        return summary

    report = json.loads(Path(artifact.path).read_text(encoding="utf-8"))
    substitution = report["amino_acid_substitution"]
    site = int(substitution[1:-1])
    metrics = {
        "amino_acid_substitution": substitution,
        "cleaned_amino_acid_substitution": report["cleaned_amino_acid_substitution"],
    }
    paths = {key: f"$.{key}" for key in metrics}
    for side in ("wild_type", "mutant"):
        array_key = f"{side}_residue_mapping_old_to_new"
        matches = [
            (index, record)
            for index, record in enumerate(report[array_key])
            if record["old_residue"]["residue_index"] == site
        ]
        if len(matches) != 1:
            raise ValueError(f"{artifact.path} has no unique {side} mapping for residue {site}")
        index, record = matches[0]
        old = record["old_residue"]
        new = record["new_residue"]
        key = f"{side}_mutation_site_mapping"
        metrics[key] = (
            f"{old['residue_name']}{old['residue_index']} -> "
            f"{new['residue_name']}{new['residue_index']}"
        )
        paths[key] = f"$.{array_key}[{index}]"
    return summary.model_copy(update={
        "key_metrics": {**metrics, **summary.key_metrics},
        "metric_paths": {**paths, **summary.metric_paths},
        "metric_inference_levels": {
            **{key: "direct" for key in metrics},
            **summary.metric_inference_levels,
        },
    })


def _test_report_summary(artifact: Artifact) -> ReportSummary:
    """Expose the fixed evidence needed for this case at the report boundary."""
    summary = _summary_with_mutation_mapping(artifact)
    if summary.read_status != "parsed":
        return summary

    case_id = artifact.metadata.get("benchmark_case_id")
    if case_id == "T2-MC-06" and summary.report_type == "enzywizard_embedding":
        report = json.loads(Path(artifact.path).read_text(encoding="utf-8"))
        index, residue = next(
            (index, record) for index, record in enumerate(report["sequence_embeddings"])
            if record["residue_index"] == 72
        )
        metric = "residue_72_embedding_coordinate_1"
        return summary.model_copy(update={
            "key_metrics": {
                metric: round(residue["residue_embedding"][0], 6), **summary.key_metrics
            },
            "metric_paths": {
                metric: f"$.sequence_embeddings[{index}].residue_embedding[0]",
                **summary.metric_paths,
            },
            "metric_inference_levels": {metric: "direct", **summary.metric_inference_levels},
        })

    if summary.report_type == "enzywizard_interaction" and case_id in {
        "T2-SR-12", "T2-MI-04", "T2-MI-08", "T2-MI-12",
        "T2-MC-12", "T2-MF-11", "T2-EL-08", "T2-EL-11",
    }:
        contacts = {
            metric: value
            for metric, value in summary.key_metrics.items()
            if ".enzyme_substrate_interaction_statistics." in summary.metric_paths[metric]
        }
        # The Agent emits field-level references for the first eight metrics.
        return summary.model_copy(update={"key_metrics": {**contacts, **summary.key_metrics}})
    return summary


def _run_analysis(
    catalog: dict[str, Any],
    case: dict[str, Any],
    args: argparse.Namespace,
    case_dir: Path,
) -> tuple[Any, Any | None, dict[str, Any]]:
    category = case["category_id"]
    artifacts = _artifacts(catalog, case)
    state = _base_state(args, case_dir, case.get("initial_question") or case["question"])
    state["artifacts"] = artifacts
    # This serial benchmark supplies the fixture fields through the Agent's
    # normal analysis and citation path.
    with patch.object(analysis_service, "summarize_report_artifact", _test_report_summary):
        result = _analyze_reports(state, artifacts, historical=False)
        if category == "multi_turn_follow_up_revision":
            initial = result
            state["analysis_results"] = [initial]
            state["analysis_summary"] = f"Turn 1: {initial.summary}"
            state["current_turn_id"] = 2
            state["current_user_message"] = case["follow_up_question"]
            state["current_history_query"] = _history_result(artifacts)
            result = _analyze_reports(state, artifacts, historical=True)
            return result, initial, state
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


def _iis_dimensions(case: dict[str, Any]) -> list[dict[str, Any]]:
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
            "label": "Stays within the evidence and states the required scientific limitations",
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


def _answer_for_review(result: AnalysisResult) -> str:
    """Keep resolved inline citations and the evidence section shown to the user."""

    answer = scientific_review_answer(result)
    visible = (result.final_answer or "").partition(TERMINOLOGY_APPENDIX_MARKER)[0]
    _, marker, evidence = visible.partition(CLAIM_EVIDENCE_APPENDIX_MARKER)
    return answer + marker + evidence.rstrip()


def _case_semantic_requirements(
    catalog: dict[str, Any], case: dict[str, Any]
) -> list[dict[str, Any]]:
    override = catalog.get("case_key_fact_overrides", {}).get(case["case_id"], {})
    return [
        {
            "fact_id": f"{case['case_id']}:requirement:{item['id']}",
            "kind": "semantic_requirement",
            "report_id": item["report_ids"][0],
            "supporting_report_ids": item["report_ids"],
            "metric_name": item["id"],
            "expected_value": item["acceptance"],
            "field_path": None,
            "evidence_requirement": item["evidence"],
        }
        for item in override.get("requirements", [])
    ]


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)
