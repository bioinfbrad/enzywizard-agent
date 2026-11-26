#!/usr/bin/env python3
"""Run the 60-case Test 1 benchmark with Avg Score and execution success rate."""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from uuid import uuid4

from enzywizard_agent.conf import AgentConfig
from enzywizard_agent.data_models.runtime import RuntimeSession
from enzywizard_agent.services.llm_service import check_model_capabilities
from enzywizard_agent.services.runtime_service import utc_now

import benchmark_support as runner
from trajectory import run_trajectory
from workflow_metrics import METRICS, case_contracts, score_case, summarize_metrics


def _summary(rows, args, *, complete):
    iterations = args.bootstrap_iterations if complete else 0
    categories = sorted({row["category_id"] for row in rows})
    return {
        "case_count": len({row["case_id"] for row in rows}),
        "run_count": len(rows),
        "completed_all_case_runs": complete,
        "passed_runs": sum(row["passed"] for row in rows),
        "metrics": summarize_metrics(rows, iterations=iterations, seed=args.seed + 1),
        "by_category": {
            name: {
                "case_count": len({row["case_id"] for row in rows if row["category_id"] == name}),
                "metrics": summarize_metrics(
                    [row for row in rows if row["category_id"] == name],
                    iterations=iterations, seed=args.seed + sum(map(ord, name)),
                ),
            }
            for name in categories
        },
        "usage": {
            key: sum(row["usage"][key] for row in rows)
            for key in ("llm_calls", "input_tokens", "output_tokens", "total_tokens")
        },
    }


def _write_csv(path, rows, metrics):
    columns = ["case_id", "category_id", "repetition", "passed", *metrics, "duration_seconds"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, **row["metrics"]})


def main() -> int:
    parser = runner.build_parser()
    parser.description = "Test 1 benchmark with milestone similarity and scientific task success."
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    if args.bootstrap_iterations != 0 and args.bootstrap_iterations < 100:
        parser.error("--bootstrap_iterations must be 0 or at least 100")
    if args.case_ids and len(args.case_ids) != len(set(args.case_ids)):
        parser.error("--case_ids must not contain duplicates")

    catalog, all_cases = runner.load_cases()
    contracts = {case["case_id"]: case_contracts(case) for case in all_cases}
    cases = runner.select_cases(all_cases, args)
    if args.validate_only:
        count = sum(map(len, contracts.values()))
        print(f"[validation passed] {len(all_cases)} workflows, {count} valid milestone DAGs.")
        return 0
    if not args.api_key:
        parser.error("--api_key or ENZYWIZARD_API_KEY is required")
    workdir = Path(args.workdir).expanduser().resolve()
    stamp = datetime.now(timezone.utc)
    run_dir = workdir / f"run_{stamp.strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"
    output = Path(args.output).expanduser().resolve() if args.output else run_dir / "results.json"
    if workdir.is_relative_to(runner.REPO) or output.is_relative_to(runner.REPO):
        parser.error("--workdir and --output must be outside the Agent repository")
    if output == output.with_suffix(".csv"):
        parser.error("--output must not have a .csv suffix")
    run_dir.mkdir(parents=True, exist_ok=False)
    diagnostics = run_dir / "planning_diagnostics.jsonl"
    rubric = run_dir / "case_contracts.json"
    if output in {diagnostics, rubric}:
        parser.error("--output conflicts with a benchmark output name")
    runner._atomic_json(rubric, {case["case_id"]: contracts[case["case_id"]] for case in cases})

    capability_dir = run_dir / "_model_capability"
    capability_dir.mkdir()
    try:
        capability = check_model_capabilities({
            "config": AgentConfig(
                api_key=args.api_key, base_url=args.base_url, model=args.model,
                reasoning_effort=args.reasoning_effort, ontology_mode="off", workdir=capability_dir,
            ),
            "session_id": f"test1-contract-{uuid4().hex}",
            "current_turn_id": 1,
            "current_user_message": "Model contract capability probe.",
            "runtime_session": RuntimeSession(started_at=utc_now(), started_monotonic=time.monotonic()),
        }).model_dump(mode="json")
    except Exception as error:
        print(f"[model contract failed] {str(error).replace(args.api_key, '[redacted]')}", flush=True)
        return 2

    metrics = METRICS if args.execute else ("Plan Score",)
    metadata = {
        "test_id": "test_1_workflow_planning",
        "generated_at": stamp.isoformat(),
        "model": args.model, "base_url": args.base_url,
        "reasoning_effort": args.reasoning_effort, "ontology_mode": args.ontology_mode,
        "profile": args.profile, "execute": args.execute, "repeat": args.repeat,
        "seed": args.seed, "selected_case_ids": [case["case_id"] for case in cases],
        "bootstrap_iterations": args.bootstrap_iterations, "case_source": str(runner.ROOT / "cases"),
        "case_contracts": str(rubric),
    }
    rows = []

    def write_results(complete: bool):
        runner._atomic_json(output, {
            "metadata": metadata, "model_capability": capability,
            "summary": _summary(rows, args, complete=complete), "results": rows,
        })

    print(f"[Test 1 benchmark] cases={len(cases)} repeat={args.repeat} execute={args.execute} model={args.model}", flush=True)
    for case in cases:
        for repetition in range(1, args.repeat + 1):
            started = time.monotonic()
            case_dir = run_dir / case["case_id"] / f"repeat_{repetition:03d}"
            case_dir.mkdir(parents=True, exist_ok=False)
            print(f"\n[{case['case_id']}] {repetition}/{args.repeat} {case['description']}", flush=True)
            trace, state, errors = run_trajectory(catalog, case, args, case_dir)
            values, detail = score_case(contracts[case["case_id"]], trace, execute=args.execute)
            trace_path = case_dir / "trajectory.json"
            escaped_key = json.dumps(args.api_key, ensure_ascii=False)[1:-1]
            trace = json.loads(json.dumps(trace, ensure_ascii=False).replace(escaped_key, "[redacted]"))
            runner._atomic_json(trace_path, trace)
            row = {
                "case_id": case["case_id"], "category_id": case["category_id"], "repetition": repetition,
                "passed": not errors and all(value == 1.0 for value in values.values()),
                "errors": [error.replace(args.api_key, "[redacted]") for error in errors],
                "metrics": values, "details": detail, "trajectory": str(trace_path),
                "duration_seconds": round(time.monotonic() - started, 3),
                "usage": runner._runtime_usage(state),
            }
            rows.append(row)
            runner._write_planning_diagnostics(diagnostics, row, [
                event for event in trace["events"] if event["kind"] in {"routing", "history", "planning_state"}
            ], state, args)
            print(f"  Metrics: {values}", flush=True)
            for error in row["errors"]:
                print(f"  Error: {error}", flush=True)
            write_results(False)
            if args.stop_on_failure and not row["passed"]:
                _write_csv(output.with_suffix(".csv"), rows, metrics)
                return 1
    write_results(True)
    _write_csv(output.with_suffix(".csv"), rows, metrics)
    print(f"\n[complete] {output}", flush=True)
    return 0 if all(row["passed"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
