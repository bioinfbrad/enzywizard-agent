"""Publication Test 5: paired FAIR off/retrieve/control ablation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
from typing import Any

from publication_benchmark_utils import (
    DEFAULT_BASE_URL,
    DEFAULT_REPEAT,
    atomic_write_json,
    create_run_dir,
    mean,
    write_flat_csv,
)


HERE = Path(__file__).resolve().parent
TEST_SCRIPTS = {
    "test_1": HERE / "run_publication_test_1_workflows.py",
    "test_2": HERE / "run_publication_test_2_analysis.py",
    "test_3": HERE / "run_publication_test_3_recovery.py",
}
MODES = ("off", "retrieve", "control")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api_key", default=os.environ.get("ENZYWIZARD_API_KEY"))
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reasoning_effort", default=None)
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT)
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument("--sample_per_subtype", type=int, default=2)
    parser.add_argument(
        "--quick_case_count",
        type=int,
        help="Exact total number of subtype-balanced cases per quick test.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Sampling seed. Omit it to create one shared random ablation seed.",
    )
    parser.add_argument("--workdir", default="~/enzywizard-publication-test-5")
    parser.add_argument("--output")
    parser.add_argument("--judge_model")
    parser.add_argument("--judge_base_url")
    parser.add_argument("--judge_reasoning_effort")
    parser.add_argument("--skip_llm_judge", action="store_true")
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument("--execute_representative", action="store_true")
    execution.add_argument(
        "--execute_all",
        action="store_true",
        help="Run Test 1 as the full end-to-end execution benchmark in every mode.",
    )
    return parser


def _command(
    script: Path,
    test_id: str,
    mode: str,
    args: argparse.Namespace,
    run_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(script),
        "--model",
        args.model,
        "--base_url",
        args.base_url,
        "--ontology_mode",
        mode,
        "--repeat",
        str(args.repeat),
        "--profile",
        args.profile,
        "--sample_per_subtype",
        str(args.sample_per_subtype),
        "--seed",
        str(args.seed),
        "--workdir",
        str(run_dir / mode / test_id),
        "--output",
        str(run_dir / mode / f"{test_id}.json"),
    ]
    if args.quick_case_count is not None:
        command.extend(["--quick_case_count", str(args.quick_case_count)])
    command.extend(["--api_key", args.api_key])
    if args.reasoning_effort is not None:
        command.extend(["--reasoning_effort", args.reasoning_effort])
    if test_id == "test_1" and args.execute_representative:
        command.append("--execute_representative")
    if test_id == "test_1" and args.execute_all:
        command.append("--execute_all")
    if test_id == "test_2":
        if args.judge_model:
            command.extend(["--judge_model", args.judge_model])
        if args.judge_base_url:
            command.extend(["--judge_base_url", args.judge_base_url])
        if args.judge_reasoning_effort is not None:
            command.extend(
                ["--judge_reasoning_effort", args.judge_reasoning_effort]
            )
        if args.skip_llm_judge:
            command.append("--skip_llm_judge")
    return command


def _load_outputs(run_dir: Path) -> dict[str, dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        for test_id in TEST_SCRIPTS:
            path = run_dir / mode / f"{test_id}.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing ablation output: {path}")
            outputs[f"{mode}:{test_id}"] = json.loads(path.read_text(encoding="utf-8"))
    return outputs


def _paired_rows(outputs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for test_id in TEST_SCRIPTS:
        by_mode: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
        for mode in MODES:
            by_mode[mode] = {
                (item["case_id"], int(item["repetition"])): item
                for item in outputs[f"{mode}:{test_id}"]["results"]
            }
        keys = set(by_mode["off"])
        if any(set(by_mode[mode]) != keys for mode in MODES[1:]):
            raise ValueError(f"Unpaired result keys for {test_id}")
        for case_id, repetition in sorted(keys):
            row: dict[str, Any] = {
                "test_id": test_id,
                "case_id": case_id,
                "repetition": repetition,
                "fair_relevance": by_mode["off"][(case_id, repetition)].get(
                    "fair_relevance"
                ),
            }
            metric_names = set()
            for mode in MODES:
                item = by_mode[mode][(case_id, repetition)]
                row[f"{mode}_passed"] = bool(item.get("passed"))
                metric_names.update((item.get("metrics") or {}).keys())
            for metric in sorted(metric_names):
                values = {
                    mode: (by_mode[mode][(case_id, repetition)].get("metrics") or {}).get(
                        metric
                    )
                    for mode in MODES
                }
                for mode, value in values.items():
                    row[f"{mode}_{metric}"] = value
                if all(isinstance(values[mode], (int, float, bool)) for mode in MODES):
                    row[f"retrieve_minus_off_{metric}"] = float(values["retrieve"]) - float(
                        values["off"]
                    )
                    row[f"control_minus_off_{metric}"] = float(values["control"]) - float(
                        values["off"]
                    )
            rows.append(row)
    return rows


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "case_count_per_mode": len({(row["test_id"], row["case_id"]) for row in rows}),
        "run_count_per_mode": len(rows),
        "pass_rates": {
            mode: mean([row[f"{mode}_passed"] for row in rows]) for mode in MODES
        },
    }
    delta_names = sorted(
        {
            key
            for row in rows
            for key in row
            if key.startswith("retrieve_minus_off_")
            or key.startswith("control_minus_off_")
        }
    )
    summary["mean_paired_deltas"] = {
        name: mean([row.get(name) for row in rows]) for name in delta_names
    }
    summary["pass_rates_by_relevance"] = {
        relevance: {
            mode: mean(
                [
                    row[f"{mode}_passed"]
                    for row in rows
                    if row.get("fair_relevance") == relevance
                ]
            )
            for mode in MODES
        }
        for relevance in ("relevant", "neutral")
    }
    summary["pass_rates_by_test"] = {
        test_id: {
            mode: mean(
                [
                    row[f"{mode}_passed"]
                    for row in rows
                    if row["test_id"] == test_id
                ]
            )
            for mode in MODES
        }
        for test_id in TEST_SCRIPTS
    }
    return summary


def _usage_by_mode(outputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "llm_call_count",
        "input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "exact_usage_calls",
    )
    totals: dict[str, Any] = {}
    for mode in MODES:
        records = [
            outputs[f"{mode}:{test_id}"].get("usage") or {}
            for test_id in TEST_SCRIPTS
        ]
        totals[mode] = {
            field: sum(int(record.get(field, 0) or 0) for record in records)
            for field in fields
        }
        retrieval_records = [
            record.get("knowledge_retrieval") or {} for record in records
        ]
        totals[mode]["knowledge_retrieval"] = {
            field: sum(int(record.get(field, 0) or 0) for record in retrieval_records)
            for field in (
                "event_count",
                "matched_event_count",
                "injected_term_count",
                "estimated_input_tokens",
            )
        }
    totals["retrieve_minus_off"] = {
        field: totals["retrieve"][field] - totals["off"][field]
        for field in fields
    }
    totals["control_minus_off"] = {
        field: totals["control"][field] - totals["off"][field]
        for field in fields
    }
    return totals


def main() -> int:
    args = build_parser().parse_args()
    if not args.api_key:
        raise SystemExit("--api_key or ENZYWIZARD_API_KEY is required")
    if args.repeat < 1:
        raise SystemExit("--repeat must be positive")
    if args.seed is None:
        args.seed = secrets.randbits(32)
    run_dir = create_run_dir(args.workdir, "test_5")
    output = Path(args.output).expanduser().resolve() if args.output else run_dir / "results.json"
    process_results: list[dict[str, Any]] = []
    for mode in MODES:
        for test_id, script in TEST_SCRIPTS.items():
            command = _command(script, test_id, mode, args, run_dir)
            print(f"[Test 5] mode={mode} running {test_id}", flush=True)
            completed = subprocess.run(command, check=False)
            process_results.append(
                {"mode": mode, "test_id": test_id, "return_code": completed.returncode}
            )
    outputs = _load_outputs(run_dir)
    rows = _paired_rows(outputs)
    payload = {
        "test_id": "test_5",
        "model": args.model,
        "base_url": args.base_url,
        "reasoning_effort": args.reasoning_effort,
        "repeat": args.repeat,
        "profile": args.profile,
        "sample_per_subtype": args.sample_per_subtype,
        "quick_case_count": args.quick_case_count,
        "seed": args.seed,
        "modes": list(MODES),
        "fixed_case_count": len(
            {(row["test_id"], row["case_id"]) for row in rows}
        ),
        "summary": _summary(rows),
        "usage_by_mode": _usage_by_mode(outputs),
        "process_results": process_results,
        "source_outputs": {
            key: {
                "metadata": value.get("metadata"),
                "summary": value.get("summary"),
                "usage": value.get("usage"),
            }
            for key, value in outputs.items()
        },
        "paired_results": rows,
    }
    atomic_write_json(output, payload)
    write_flat_csv(output.with_suffix(".csv"), rows)
    print(f"[complete] {output}")
    return 0 if all(item["return_code"] == 0 for item in process_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
