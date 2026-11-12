#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
from statistics import mean
from typing import Any

from scoring import METRIC_NAMES


CONDITIONS = ("off", "control", "retrieve")
MATCHED_METADATA_FIELDS = (
    "model",
    "base_url",
    "reasoning_effort",
    "llm_timeout_seconds",
    "evaluation_mode",
    "metric_evaluation_mode",
    "iis_evaluator",
    "judge_model",
    "judge_base_url",
    "judge_reasoning_effort",
    "judge_timeout_seconds",
    "repeat",
    "profile",
    "seed",
    "selected_case_ids",
    "bootstrap_iterations",
    "fair_knowledge_base_sha256",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare matched off, control, and retrieve Test 4 runs."
    )
    parser.add_argument("--off", required=True, help="Off-condition results.json")
    parser.add_argument(
        "--control", required=True, help="Control-condition results.json"
    )
    parser.add_argument(
        "--retrieve", required=True, help="Retrieve-condition results.json"
    )
    parser.add_argument("--bootstrap_iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--output")
    return parser


def _load(path: str, expected_condition: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    if metadata.get("test_id") != "test_4_fair_ablation":
        raise ValueError(f"Not a Test 4 result: {resolved}")
    if metadata.get("ontology_mode") != expected_condition:
        raise ValueError(
            f"Expected {expected_condition!r}, found "
            f"{metadata.get('ontology_mode')!r}: {resolved}"
        )
    payload["_path"] = str(resolved)
    return payload


def _validate_matched(payloads: dict[str, dict[str, Any]]) -> None:
    baseline = payloads["off"]["metadata"]
    for condition in CONDITIONS[1:]:
        metadata = payloads[condition]["metadata"]
        for field in MATCHED_METADATA_FIELDS:
            if metadata.get(field) != baseline.get(field):
                raise ValueError(
                    f"Unmatched metadata field {field!r}: off="
                    f"{baseline.get(field)!r}, {condition}={metadata.get(field)!r}"
                )
    expected_keys: set[tuple[str, int]] | None = None
    for condition in CONDITIONS:
        results = payloads[condition].get("results")
        if not isinstance(results, list):
            raise ValueError(f"{condition} results must be a list")
        keys = {
            (str(item["case_id"]), int(item["repetition"])) for item in results
        }
        if len(keys) != len(results):
            raise ValueError(f"{condition} contains duplicate case/repetition rows")
        failed = [str(item["case_id"]) for item in results if not item.get("passed")]
        if failed:
            raise ValueError(
                f"{condition} contains incomplete or failed runs: {sorted(set(failed))}"
            )
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys:
            raise ValueError(f"{condition} does not contain the same case repetitions")

    rows = {
        condition: {
            (str(item["case_id"]), int(item["repetition"])): item
            for item in payloads[condition]["results"]
        }
        for condition in CONDITIONS
    }
    for key in sorted(rows["off"]):
        audits = {
            condition: rows[condition][key].get("retrieval_audit")
            for condition in CONDITIONS
        }
        if not all(isinstance(item, dict) for item in audits.values()):
            raise ValueError(f"Missing retrieval audit for case/repetition {key}")
        if any(not audits[condition].get("condition_valid") for condition in CONDITIONS):
            raise ValueError(f"Invalid ablation condition for case/repetition {key}")
        if audits["off"].get("injected_term_count") != 0:
            raise ValueError(f"Off condition injected FAIR metadata for {key}")
        if audits["retrieve"].get("relevant_injected_term_count", 0) < 1:
            raise ValueError(f"Retrieve condition lacked a relevant FAIR term for {key}")
        if audits["control"].get("relevant_injected_term_count", 0) != 0:
            raise ValueError(f"Control condition injected a relevant FAIR term for {key}")
        if audits["control"].get("injected_term_count") != audits["retrieve"].get(
            "injected_term_count"
        ):
            raise ValueError(f"Control/retrieve injection counts differ for {key}")
        hashes = {
            audits[condition].get("knowledge_base_sha256")
            for condition in CONDITIONS
        }
        if None in hashes or len(hashes) != 1:
            raise ValueError(f"Knowledge-base hashes differ for {key}")


def _metric_by_case(
    payload: dict[str, Any], metric: str
) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for item in payload["results"]:
        value = item.get("metrics", {}).get(metric)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values[str(item["case_id"])].append(float(value))
    return {case_id: mean(case_values) for case_id, case_values in values.items()}


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _paired_difference(
    left: dict[str, float],
    right: dict[str, float],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    case_ids = sorted(set(left) & set(right))
    differences = [left[case_id] - right[case_id] for case_id in case_ids]
    if not differences:
        return {"mean_difference": None, "bootstrap_95_ci": None, "case_count": 0}
    rng = random.Random(seed)
    estimates = [
        mean(rng.choice(differences) for _ in differences)
        for _ in range(iterations)
    ]
    return {
        "mean_difference": mean(differences),
        "bootstrap_95_ci": {
            "low": _percentile(estimates, 0.025),
            "high": _percentile(estimates, 0.975),
        },
        "case_count": len(differences),
    }


def compare(
    payloads: dict[str, dict[str, Any]], *, iterations: int, seed: int
) -> dict[str, Any]:
    _validate_matched(payloads)
    metrics: dict[str, Any] = {}
    for index, metric in enumerate(METRIC_NAMES):
        by_condition = {
            condition: _metric_by_case(payloads[condition], metric)
            for condition in CONDITIONS
        }
        metrics[metric] = {
            "condition_means": {
                condition: (
                    mean(values.values()) if values else None
                )
                for condition, values in by_condition.items()
            },
            "retrieve_minus_off": _paired_difference(
                by_condition["retrieve"],
                by_condition["off"],
                iterations=iterations,
                seed=seed + index * 2,
            ),
            "retrieve_minus_control": _paired_difference(
                by_condition["retrieve"],
                by_condition["control"],
                iterations=iterations,
                seed=seed + index * 2 + 1,
            ),
        }
    return {
        "test_id": "test_4_fair_ablation_comparison",
        "source_results": {
            condition: payloads[condition]["_path"] for condition in CONDITIONS
        },
        "bootstrap_iterations": iterations,
        "seed": seed,
        "metrics": metrics,
    }


def main() -> int:
    args = build_parser().parse_args()
    if args.bootstrap_iterations < 100:
        raise SystemExit("--bootstrap_iterations must be at least 100")
    payloads = {
        condition: _load(getattr(args, condition), condition)
        for condition in CONDITIONS
    }
    output = compare(
        payloads, iterations=args.bootstrap_iterations, seed=args.seed
    )
    rendered = json.dumps(output, indent=2, ensure_ascii=False)
    if args.output:
        path = Path(args.output).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
        print(path)
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
