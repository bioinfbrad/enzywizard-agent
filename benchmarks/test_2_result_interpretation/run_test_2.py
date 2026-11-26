#!/usr/bin/env python3
"""Run Test 2 with Claude Haiku 5.5 as the fixed semantic judge."""

import argparse
import csv
from datetime import datetime, timezone
import os
from pathlib import Path
import time
from uuid import uuid4

from enzywizard_agent.conf import AgentConfig
from enzywizard_agent.data_models.runtime import RuntimeSession
from enzywizard_agent.services.llm_service import check_model_capabilities
from enzywizard_agent.services.runtime_service import utc_now

import benchmark_support as bench
from blind_review import export_blind_review
import judge
from metrics import METRICS, case_rates, macro_rates, summarize_metrics
from review_packet import build_review_packet


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api_key", default=os.environ.get("ENZYWIZARD_API_KEY"))
    parser.add_argument("--base_url", default=bench.DEFAULT_BASE_URL)
    parser.add_argument("--model", default=bench.DEFAULT_MODEL)
    parser.add_argument("--reasoning_effort", default=bench.DEFAULT_REASONING_EFFORT)
    parser.add_argument("--llm_timeout_seconds", type=float, default=180.0)
    parser.add_argument("--ontology_mode", choices=("off", "retrieve", "control"), default="retrieve")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument("--quick_case_count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--case_ids", nargs="*")
    parser.add_argument("--judge_api_key", default=os.environ.get("ANTHROPIC_API_KEY"))
    parser.add_argument("--judge_timeout_seconds", type=float, default=180.0)
    parser.add_argument("--show_details", action="store_true")
    parser.add_argument("--stop_on_failure", action="store_true")
    parser.add_argument("--bootstrap_iterations", type=int, default=1000)
    parser.add_argument("--validate_only", action="store_true")
    parser.add_argument("--workdir", default="~/enzywizard-benchmark-results/test_2_result_interpretation")
    parser.add_argument("--output")
    return parser


def _redact(value, args):
    text = str(value)
    for key in (args.api_key, args.judge_api_key):
        if key:
            text = text.replace(key, "[redacted]")
    return text


def run_case(catalog, case, repetition, args, case_dir):
    case_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    packet, review, rates = None, None, None
    agent_usage, judge_usage = {}, {}
    errors = []
    metric_errors = {}
    try:
        result, initial, state = bench._run_analysis(catalog, case, args, case_dir)
        agent_usage = bench._runtime_usage(state)
        bench._atomic_json(case_dir / "analysis_result.json", result.model_dump(mode="json"))
        if initial is not None:
            bench._atomic_json(case_dir / "initial_analysis_result.json", initial.model_dump(mode="json"))
        for stage, analysis in (("Final", result), ("Initial", initial)):
            if analysis is not None and analysis.analysis_status != "succeeded":
                errors.append(f"{stage} analysis did not succeed: {analysis.summary}")
        packet = build_review_packet(catalog, case, result, initial)
        review, judge_usage = judge.review_case(
            packet, args.judge_api_key, args.judge_timeout_seconds,
            save_raw=lambda data: bench._atomic_json(case_dir / "judge_response.json", data),
        )
        rates = case_rates(review)
        if review["citation_errors"]:
            metric_errors["Citation F1"] = [_redact(error, args) for error in review["citation_errors"]]
            errors.append("Citation review failed validation. See metric_errors and judge_response.json.")
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    return {
        "case_id": case["case_id"], "category_id": case["category_id"], "repetition": repetition,
        "passed": not errors, "errors": [_redact(error, args) for error in errors],
        "rates": rates, "metric_errors": metric_errors, "packet": packet, "review": review,
        "usage": {"agent": agent_usage, "judge": judge_usage},
        "case_workdir": str(case_dir), "duration_seconds": round(time.monotonic() - started, 3),
    }


def _write_csv(path, rows):
    fields = ["case_id", "category_id", "repetition", "passed"]
    fields.extend(f"{name} {part}" for name in METRICS for part in ("precision", "recall", "f1"))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            item = {key: row[key] for key in ("case_id", "category_id", "repetition", "passed")}
            if row["rates"] is not None:
                for name, rates in row["rates"].items():
                    for part in ("precision", "recall", "f1"):
                        item[f"{name} {part}"] = macro_rates([rates])[part]
            writer.writerow(item)


def _probe_model(args, run_dir):
    capability_dir = run_dir / "_model_capability"
    capability_dir.mkdir()
    return check_model_capabilities({
        "config": AgentConfig(
            api_key=args.api_key, base_url=args.base_url, model=args.model,
            reasoning_effort=args.reasoning_effort, ontology_mode="off",
            llm_timeout_seconds=args.llm_timeout_seconds, workdir=capability_dir,
        ),
        "session_id": f"test2-contract-{uuid4().hex}",
        "current_turn_id": 1,
        "current_user_message": "Model contract capability probe.",
        "runtime_session": RuntimeSession(started_at=utc_now(), started_monotonic=time.monotonic()),
    }).model_dump(mode="json")


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    if min(args.llm_timeout_seconds, args.judge_timeout_seconds) <= 0:
        parser.error("Timeouts must be positive")
    if args.bootstrap_iterations != 0 and args.bootstrap_iterations < 100:
        parser.error("--bootstrap_iterations must be 0 or at least 100")
    catalog, all_cases = bench.load_suite()
    try:
        cases = bench.select_cases(all_cases, args)
    except ValueError as error:
        parser.error(str(error))
    if args.validate_only:
        print(f"[validation passed] {len(all_cases)} cases and {len(catalog['reports'])} report fixtures")
        return 0
    if not args.api_key or not args.judge_api_key:
        parser.error("Provide ENZYWIZARD_API_KEY for the Agent and ANTHROPIC_API_KEY for the Judge")
    stamp = datetime.now(timezone.utc)
    workdir = Path(args.workdir).expanduser().resolve()
    run_dir = workdir / f"run_{stamp.strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"
    output = Path(args.output).expanduser().resolve() if args.output else run_dir / "results.json"
    if workdir.is_relative_to(bench.REPO) or output.is_relative_to(bench.REPO):
        parser.error("--workdir and --output must be outside the Agent repository")
    if output.suffix.lower() == ".csv":
        parser.error("--output must not have a .csv suffix")
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        capability = _probe_model(args, run_dir)
    except Exception as error:
        print(f"[model contract failed] {_redact(error, args)}", flush=True)
        return 2

    case_ids = [case["case_id"] for case in cases]
    metadata = {
        "test_id": "test_2_result_interpretation", "generated_at": stamp.isoformat(),
        "model": args.model, "base_url": args.base_url, "reasoning_effort": args.reasoning_effort,
        "ontology_mode": args.ontology_mode, "judge_model": judge.MODEL, "judge_base_url": judge.BASE_URL,
        "judge_api_version": judge.API_VERSION, "judge_reasoning_effort": "medium",
        "judge_protocol": "stable_evidence_ids_v5",
        "reference_policy": catalog["key_fact_policy_version"], "profile": args.profile,
        "repeat": args.repeat, "seed": args.seed, "selected_case_ids": case_ids,
        "bootstrap_iterations": args.bootstrap_iterations,
    }
    rows = []

    def save(*, complete=False):
        summary = {
            "case_count": len(case_ids), "expected_runs": len(case_ids) * args.repeat, "run_count": len(rows),
            "completed_all_case_runs": complete,
            "successful_runs": sum(row["passed"] for row in rows),
            "usable_judge_reviews": sum(row["review"] is not None and not row["metric_errors"] for row in rows),
            "metrics": summarize_metrics(
                rows, case_ids, args.repeat, iterations=args.bootstrap_iterations if complete else 0, seed=args.seed,
            ),
        }
        bench._atomic_json(output, {"metadata": metadata, "model_capability": capability, "summary": summary, "results": rows})

    print(f"[Test 2] {args.model} | judge={judge.MODEL} | cases={len(cases)} | repeat={args.repeat} | {run_dir}", flush=True)
    for repetition in range(1, args.repeat + 1):
        for case in cases:
            case_id = case["case_id"]
            print(f"[{case_id}] {repetition}/{args.repeat}", flush=True)
            case_dir = run_dir / case_id / f"repeat_{repetition:03d}"
            try:
                row = run_case(catalog, case, repetition, args, case_dir)
            except KeyboardInterrupt:
                save()
                _write_csv(output.with_suffix(".csv"), rows)
                return 130
            rows.append(row)
            for error in row["errors"]:
                print(f"  Error: {error}", flush=True)
            usage = row["usage"]["judge"]
            input_tokens = sum(usage.get(name, 0) for name in (
                "input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"
            ))
            if input_tokens > judge.STANDARD_INPUT_LIMIT:
                print(f"  Judge input exceeded 100,000 tokens: {input_tokens}", flush=True)
            if args.show_details and row["rates"] is not None:
                print({name: macro_rates([rates]) for name, rates in row["rates"].items()}, flush=True)
            finished = len(rows) == len(cases) * args.repeat
            if not finished:
                save()
            if args.stop_on_failure and not row["passed"] and not finished:
                _write_csv(output.with_suffix(".csv"), rows)
                return 1
    export_blind_review(run_dir, rows, case_ids, args.seed)
    save(complete=True)
    _write_csv(output.with_suffix(".csv"), rows)
    print(f"[complete] {output}", flush=True)
    return 0 if all(row["passed"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
