"""Publication Test 2: multidimensional interpretation and provenance."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import secrets
from statistics import stdev
import time
from typing import Any

from enzywizard_agent.conf import AgentConfig
from enzywizard_agent.data_models.agent_inputs import AgentInputs
from enzywizard_agent.data_models.artifact import Artifact
from enzywizard_agent.data_models.runtime import RuntimeSession
from enzywizard_agent.services.analysis_service import _analyze_reports
from enzywizard_agent.services.runtime_service import utc_now

from evaluation_utils import runtime_usage_payload
from publication_benchmark_utils import (
    AnalysisBenchmarkCase,
    BENCHMARK_ROOT,
    DEFAULT_BASE_URL,
    DEFAULT_REPEAT,
    TEST_2_CASE_COUNT,
    aggregate_result_usage,
    atomic_write_json,
    benchmark_metadata,
    check_benchmark_model_contract,
    create_run_dir,
    load_case_file,
    metric_statistics,
    metric_statistics_by_subtype,
    resolve_project_path,
    run_blind_judge,
    select_profile_cases,
    summarize_repeated_results,
    write_flat_csv,
)


CASES_PATH = BENCHMARK_ROOT / "test_2_analysis_cases.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api_key", default=os.environ.get("ENZYWIZARD_API_KEY"))
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reasoning_effort", default=None)
    parser.add_argument(
        "--ontology_mode", choices=("off", "retrieve", "control"), default="retrieve"
    )
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT)
    parser.add_argument("--case_ids", nargs="*")
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument("--sample_per_subtype", type=int, default=2)
    parser.add_argument(
        "--quick_case_count",
        type=int,
        help="Exact total number of subtype-balanced cases in quick mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Sampling seed. Omit it for a new random quick sample each run.",
    )
    parser.add_argument("--workdir", default="~/enzywizard-publication-test-2")
    parser.add_argument("--output")
    parser.add_argument("--judge_model")
    parser.add_argument("--judge_base_url")
    parser.add_argument("--judge_reasoning_effort")
    parser.add_argument("--skip_llm_judge", action="store_true")
    parser.add_argument(
        "--show_details",
        action="store_true",
        help=(
            "Print each case's reports, generated analysis, limitations, evidence, "
            "and judge result."
        ),
    )
    parser.add_argument("--stop_on_failure", action="store_true")
    return parser


def _artifacts(case: AnalysisBenchmarkCase) -> list[Artifact]:
    return [
        Artifact(
            turn_id=1,
            plan_id=1,
            artifact_type=spec.artifact_type,
            path=str(resolve_project_path(spec.path)),
            source_tool=spec.source_tool,
            side=spec.side,
            metadata={"benchmark_case_id": case.case_id},
        )
        for spec in case.artifacts
    ]


def _automatic_evaluation(case: AnalysisBenchmarkCase, result: Any) -> dict[str, Any]:
    report_types = {item.report_type for item in result.report_summaries}
    required_reports = set(case.required_report_types)
    evidence_by_id = {item.evidence_id: item for item in result.evidence}
    all_references = [
        evidence_id
        for finding in result.findings
        for evidence_id in finding.evidence_ids
    ]
    valid_references = [item for item in all_references if item in evidence_by_id]
    substantive = [
        finding
        for finding in result.findings
        if finding.finding_kind in {"result", "comparison"}
    ]
    linked_substantive = [
        finding
        for finding in substantive
        if finding.evidence_ids
        and all(item in evidence_by_id for item in finding.evidence_ids)
    ]
    citation_precision = (
        len(valid_references) / len(all_references) if all_references else 0.0
    )
    evidence_recall = (
        len(linked_substantive) / len(substantive) if substantive else 0.0
    )
    evidence_f1 = (
        2 * citation_precision * evidence_recall / (citation_precision + evidence_recall)
        if citation_precision + evidence_recall
        else 0.0
    )
    report_evidence = [
        item for item in result.evidence if item.source_kind == "report"
    ]
    def has_complete_provenance(item: Any) -> bool:
        if not item.source_tool or not item.artifact_type or not item.field_path:
            return False
        if item.inference_level == "direct":
            return bool(item.path)
        if not item.source_evidence_ids:
            return False
        sources = [evidence_by_id.get(source_id) for source_id in item.source_evidence_ids]
        return all(
            source is not None
            and source.inference_level == "direct"
            and source.path
            and source.source_tool
            and source.artifact_type
            and source.field_path
            for source in sources
        )

    complete_report_evidence = [
        item for item in report_evidence if has_complete_provenance(item)
    ]
    fair_evidence = [
        item for item in result.evidence if item.source_kind == "knowledge_base"
    ]
    valid_fair_evidence = [
        item
        for item in fair_evidence
        if item.source_uri and item.source_label and item.source_version
    ]
    return {
        "required_report_coverage": (
            len(report_types & required_reports) / len(required_reports)
            if required_reports
            else 1.0
        ),
        "evidence_attribution_precision": citation_precision,
        "evidence_attribution_recall": evidence_recall,
        "evidence_attribution_f1": evidence_f1,
        "provenance_record_completeness": (
            len(complete_report_evidence) / len(report_evidence)
            if report_evidence
            else 0.0
        ),
        "fair_term_citation_accuracy": (
            len(valid_fair_evidence) / len(fair_evidence) if fair_evidence else None
        ),
        "fair_term_count": len(fair_evidence),
    }


def _print_case_details(case: AnalysisBenchmarkCase) -> None:
    print(f"  Description: {case.description}", flush=True)
    print(f"  Question: {case.question}", flush=True)
    print("  Reports:", flush=True)
    for spec in case.artifacts:
        print(
            "    - "
            f"{spec.source_tool} ({spec.side}, {spec.artifact_type}): "
            f"{Path(spec.path).name}",
            flush=True,
        )
    if case.required_limitations:
        print("  Expected limitations:", flush=True)
        for limitation in case.required_limitations:
            print(f"    - {limitation}", flush=True)


def _print_result_details(result: dict[str, Any]) -> None:
    analysis = result.get("analysis_result") or {}
    print(
        "  Analysis: "
        f"status={analysis.get('analysis_status', 'unavailable')}; "
        f"used_language_model={analysis.get('used_language_model', False)}",
        flush=True,
    )
    final_answer = analysis.get("final_answer")
    if final_answer:
        print("  Final answer:", flush=True)
        for line in str(final_answer).splitlines():
            print(f"    {line}", flush=True)
    print("  Limitations:", flush=True)
    for limitation in analysis.get("limitations") or ["None reported"]:
        print(f"    - {limitation}", flush=True)
    print("  Evidence:", flush=True)
    for evidence in analysis.get("evidence") or []:
        source = evidence.get("source_label") or evidence.get("source_tool") or "unknown"
        print(
            "    - "
            f"{evidence.get('evidence_id', 'unknown')}: {source}; "
            f"{evidence.get('value_summary', '')}",
            flush=True,
        )
    judge = result.get("judge_result")
    if judge:
        print(
            "  Judge: "
            f"factual_precision={judge.get('factual_precision')}; "
            f"key_fact_recall={judge.get('key_fact_recall')}; "
            "integrated_interpretation_score="
            f"{judge.get('integrated_interpretation_score')}",
            flush=True,
        )
        print(f"  Judge rationale: {judge.get('rationale', '')}", flush=True)
    if result.get("failures"):
        print("  Failures:", flush=True)
        for failure in result["failures"]:
            print(f"    - {failure}", flush=True)
    if result.get("evaluation_errors"):
        print("  Evaluation errors:", flush=True)
        for error in result["evaluation_errors"]:
            print(f"    - {error}", flush=True)


def run_case(
    case: AnalysisBenchmarkCase,
    repetition: int,
    args: argparse.Namespace,
    run_dir: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    case_dir = run_dir / case.case_id / f"repeat_{repetition:03d}"
    case_dir.mkdir(parents=True, exist_ok=False)
    state: dict[str, Any] = {
        "config": AgentConfig(
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            ontology_mode=args.ontology_mode,
            check_model_capabilities=False,
            workdir=case_dir,
        ),
        "session_id": f"publication-{case.case_id}-{repetition}",
        "current_turn_id": 1,
        "current_plan_id": 1,
        "current_user_message": case.question,
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
    }
    failures: list[str] = []
    evaluation_errors: list[str] = []
    result = None
    judge = None
    judge_usage = None
    metrics: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    try:
        artifacts = _artifacts(case)
        missing_paths = [item.path for item in artifacts if not Path(item.path).is_file()]
        if missing_paths:
            raise FileNotFoundError(f"Missing benchmark reports: {missing_paths}")
        state["artifacts"] = artifacts
        result = _analyze_reports(
            state, artifacts, historical=case.subtype == "historical"
        )
        diagnostics = _automatic_evaluation(case, result)
        metrics["evidence_attribution_f1"] = diagnostics[
            "evidence_attribution_f1"
        ]
        narrative_succeeded = (
            result.analysis_status == "succeeded" and result.used_language_model
        )
        if not narrative_succeeded:
            detail = "; ".join(result.error_messages) or result.summary
            failures.append(
                "scientific narrative generation did not complete successfully: "
                f"{detail}"
            )
        if not args.skip_llm_judge and narrative_succeeded:
            try:
                judge, judge_usage = run_blind_judge(
                    api_key=args.api_key,
                    base_url=args.judge_base_url or args.base_url,
                    model=args.judge_model or args.model,
                    reasoning_effort=(
                        args.judge_reasoning_effort
                        if args.judge_reasoning_effort is not None
                        else args.reasoning_effort
                    ),
                    workdir=case_dir / "judge",
                    question=case.question,
                    answer=result.final_answer,
                    limitations=result.limitations,
                    required_facts=case.required_facts,
                    allowed_conclusions=case.allowed_conclusions,
                    forbidden_claims=case.forbidden_claims,
                    required_limitations=case.required_limitations,
                    structured_report_summaries=[
                        {
                            "turn_id": item.turn_id,
                            "plan_id": item.plan_id,
                            "artifact_type": item.artifact_type,
                            "report_type": item.report_type,
                            "source_tool": item.source_tool,
                            "side": item.side,
                            "artifact_name": Path(item.path).name,
                            "key_metrics": dict(list(item.key_metrics.items())[:24]),
                            "highlights": item.highlights,
                            "warnings": item.warnings,
                        }
                        for item in result.report_summaries
                    ],
                    structured_evidence=[
                        item.model_dump(mode="json") for item in result.evidence
                    ],
                )
                metrics.update(
                    {
                        "factual_precision": judge.factual_precision,
                        "key_fact_recall": judge.key_fact_recall,
                        "integrated_interpretation_score": (
                            judge.integrated_interpretation_score
                        ),
                    }
                )
            except Exception as exc:
                evaluation_errors.append(
                    f"blind judge failed: {type(exc).__name__}: {exc}"
                )
        if diagnostics.get("required_report_coverage") != 1.0:
            failures.append("not all required reports were parsed")
        if diagnostics.get("evidence_attribution_precision") != 1.0:
            failures.append("one or more evidence references were invalid")
    except Exception as exc:
        failures.append(f"{type(exc).__name__}: {exc}")
    technical_completed = not failures
    return {
        "case_id": case.case_id,
        "repetition": repetition,
        "subtype": case.subtype,
        "fair_relevance": case.fair_relevance,
        # ``passed`` remains as a runner compatibility field. It now means that
        # analysis and objective provenance checks completed, not that a
        # subjective scientific-quality threshold was crossed.
        "passed": technical_completed,
        "technical_completed": technical_completed,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "failures": failures,
        "evaluation_completed": judge is not None,
        "evaluation_skipped": args.skip_llm_judge,
        "evaluation_errors": evaluation_errors,
        "analysis_result": result.model_dump(mode="json") if result else None,
        "judge_result": judge.model_dump(mode="json") if judge else None,
        "usage": runtime_usage_payload(state.get("runtime_session")),
        "judge_usage": judge_usage,
        "duration_seconds": round(time.monotonic() - started, 3),
    }


def _repeated_metric_variability(
    results: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Summarize within-case variation without comparing free-text wording."""

    metric_names = sorted(
        {
            name
            for item in results
            for name, value in (item.get("metrics") or {}).items()
            if isinstance(value, (int, float, bool))
        }
    )
    variability: dict[str, dict[str, Any]] = {}
    for name in metric_names:
        by_case: dict[str, list[float]] = {}
        for item in results:
            value = (item.get("metrics") or {}).get(name)
            if isinstance(value, (int, float, bool)):
                by_case.setdefault(str(item["case_id"]), []).append(float(value))
        standard_deviations = [
            stdev(values) for values in by_case.values() if len(values) > 1
        ]
        variability[name] = {
            "mean_within_case_sd": (
                round(sum(standard_deviations) / len(standard_deviations), 6)
                if standard_deviations
                else None
            ),
            "max_within_case_sd": (
                round(max(standard_deviations), 6)
                if standard_deviations
                else None
            ),
            "case_count": len(standard_deviations),
        }
    return variability


def build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    technical = summarize_repeated_results(results)
    summary = {
        "case_count": technical["case_count"],
        "run_count": technical["run_count"],
        "technically_completed_runs": technical["passed_runs"],
        "technical_completion_rate": technical["run_pass_rate"],
        "technically_stable_cases": technical["stable_passed_cases"],
        "three_run_technical_completion_rate": technical[
            "three_run_stable_pass_rate"
        ],
        "quality_evaluation_completed_runs": sum(
            bool(item.get("evaluation_completed")) for item in results
        ),
        "quality_evaluation_completion_rate": (
            round(
                sum(bool(item.get("evaluation_completed")) for item in results)
                / len(results),
                6,
            )
            if results
            else None
        ),
        "quality_metrics": metric_statistics(results),
        "quality_metrics_by_subtype": metric_statistics_by_subtype(results),
        "repeated_metric_variability": _repeated_metric_variability(results),
    }
    return summary


def main() -> int:
    args = build_parser().parse_args()
    if not args.api_key:
        raise SystemExit("--api_key or ENZYWIZARD_API_KEY is required")
    if args.repeat < 1:
        raise SystemExit("--repeat must be positive")
    resolved_seed = args.seed if args.seed is not None else secrets.randbits(32)
    all_cases = load_case_file(
        CASES_PATH, AnalysisBenchmarkCase, TEST_2_CASE_COUNT
    )
    cases = select_profile_cases(
        all_cases,
        case_ids=args.case_ids,
        profile=args.profile,
        sample_per_subtype=args.sample_per_subtype,
        seed=resolved_seed,
        quick_case_count=args.quick_case_count,
    )
    run_dir = create_run_dir(args.workdir, "test_2")
    output = Path(args.output).expanduser().resolve() if args.output else run_dir / "results.json"
    try:
        capability, capability_usage = check_benchmark_model_contract(
            api_key=args.api_key,
            base_url=args.base_url,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            ontology_mode=args.ontology_mode,
            workdir=run_dir / "_model_capability_check",
        )
    except Exception as exc:
        print(f"[model contract failed] {exc}", flush=True)
        return 2
    results: list[dict[str, Any]] = []
    print(
        f"[Test 2] cases={len(cases)} repeat={args.repeat} model={args.model} "
        f"profile={args.profile} seed={resolved_seed}"
    )
    for case in cases:
        if args.show_details:
            print(flush=True)
            print(f"[{case.case_id}]", flush=True)
            _print_case_details(case)
        for repetition in range(1, args.repeat + 1):
            print(f"[{case.case_id}] repetition {repetition}/{args.repeat}", flush=True)
            result = run_case(case, repetition, args, run_dir)
            results.append(result)
            if args.show_details:
                _print_result_details(result)
            atomic_write_json(
                output,
                {
                    "metadata": {
                        **benchmark_metadata(
                            test_id="test_2",
                            cases_file=CASES_PATH,
                            model=args.model,
                            base_url=args.base_url,
                            reasoning_effort=args.reasoning_effort,
                            ontology_mode=args.ontology_mode,
                            repeat=args.repeat,
                        ),
                        "profile": args.profile,
                        "sample_per_subtype": args.sample_per_subtype,
                        "quick_case_count": args.quick_case_count,
                        "seed": resolved_seed,
                        "selected_case_ids": [item.case_id for item in cases],
                        "judge_enabled": not args.skip_llm_judge,
                        "judge_model": args.judge_model or args.model,
                        "judge_base_url": args.judge_base_url or args.base_url,
                        "judge_reasoning_effort": (
                            args.judge_reasoning_effort
                            if args.judge_reasoning_effort is not None
                            else args.reasoning_effort
                        ),
                    },
                    "summary": build_summary(results),
                    "model_capability": capability,
                    "usage": aggregate_result_usage(
                        [{"usage": capability_usage}, *results]
                    ),
                    "results": results,
                },
            )
            if args.stop_on_failure and (
                not result["technical_completed"] or result["evaluation_errors"]
            ):
                return 1
    write_flat_csv(
        output.with_suffix(".csv"),
        [
            {
                "case_id": item["case_id"],
                "repetition": item["repetition"],
                "subtype": item["subtype"],
                "technical_completed": item["technical_completed"],
                "evaluation_completed": item["evaluation_completed"],
                **item["metrics"],
            }
            for item in results
        ],
    )
    print(f"[complete] {output}")
    benchmark_completed = all(
        item["technical_completed"] and not item["evaluation_errors"]
        for item in results
    )
    return 0 if benchmark_completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
