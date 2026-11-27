#!/usr/bin/env python3
"""Compare paired Test 4 CDA runs from the three ontology conditions."""

import argparse
import json
from pathlib import Path
from statistics import fmean, stdev

from metrics import interval, score_map

MODES = ("off", "control", "retrieve")
MATCHED = (
    "test_id", "model", "base_url", "reasoning_effort", "llm_timeout_seconds",
    "repeat", "seed", "selected_case_ids", "judge_model", "judge_protocol",
    "judge_timeout_seconds", "dataset_sha256", "knowledge_base_sha256", "agent_source_sha256",
)


def compare(documents, *, iterations, seed):
    base = documents["off"]["metadata"]
    cases, repetitions = base["selected_case_ids"], base["repeat"]
    pop = {}
    audit_summary = {}
    for mode in MODES:
        doc = documents[mode]
        metadata = doc["metadata"]
        if metadata["ontology_mode"] != mode:
            raise ValueError(f"Incorrect run condition for {mode}.")
        fields = [name for name in MATCHED if metadata[name] != base[name]]
        if fields:
            raise ValueError(f"{mode} does not match off: {', '.join(fields)}")
        results = doc["results"]
        scores = score_map(results, cases, repetitions)
        if len(scores) != len(cases) * repetitions:
            raise ValueError(f"{mode} has incomplete case/repetition scores.")
        audits = [row.get("retrieval_audit") for row in results]
        if any(item is None or item["mode"] != mode for item in audits):
            raise ValueError(f"{mode} has missing or invalid retrieval audits.")
        if any(item["knowledge_base_sha256"] != base["knowledge_base_sha256"] for item in audits):
            raise ValueError(f"{mode} has mismatched knowledge-base hashes.")
        if mode == "off" and any(item["total_injected_terms"] for item in audits):
            raise ValueError("The off condition injected terminology.")
        if mode == "control":
            violations = []
            for row, audit in zip(results, audits, strict=True):
                problems = []
                if any(audit.get(key) for key in (
                    "target_injected_in_session",
                    "target_injected_on_definition_turn",
                    "target_injected_into_definition_answer",
                )):
                    problems.append("target term injected")
                if audit.get("control_count_matches") is not True:
                    problems.append("injection count does not match")
                if problems:
                    violations.append(
                        f"{row['case_id']} repeat {row['repetition']} ({', '.join(problems)})"
                    )
            if violations:
                raise ValueError(f"Invalid control condition: {'; '.join(violations)}")
        audit_summary[mode] = {
            "total_case_runs": len(results),
            "case_runs_with_target_injected_into_definition_answer": sum(
                item["target_injected_into_definition_answer"] for item in audits
            ),
            "case_runs_with_target_injected_in_session": sum(item["target_injected_in_session"] for item in audits),
            "case_runs_with_unmatched_control_count": sum(item["control_count_matches"] is False for item in audits),
        }
        pop[mode] = scores

    def statistics(values):
        means = [fmean(values[cid][i] for cid in cases) for i in range(repetitions)]
        result = {"by_repetition": means, "mean": fmean(means), "sd": stdev(means) if len(means) > 1 else None}
        if iterations:
            result["bootstrap_95_ci"] = interval([fmean(values[cid]) for cid in cases], iterations, seed)
        return result

    grouped = {
        mode: {cid: [pop[mode][cid, rep] for rep in range(1, repetitions + 1)] for cid in cases}
        for mode in MODES
    }
    differences = {
        f"retrieve_minus_{mode}": statistics({
            cid: [a - b for a, b in zip(grouped["retrieve"][cid], grouped[mode][cid], strict=True)]
            for cid in cases
        })
        for mode in ("off", "control")
    }
    return {
        "model": base["model"], "case_count": len(cases), "repeat": repetitions,
        "bootstrap_iterations": iterations, "CDA": {mode: statistics(grouped[mode]) for mode in MODES},
        "CDA_differences": differences, "retrieval_audit": audit_summary,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for mode in MODES:
        parser.add_argument(f"--{mode}", type=Path, required=True)
    parser.add_argument("--bootstrap_iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.bootstrap_iterations != 0 and args.bootstrap_iterations < 100:
        parser.error("Use 0 or at least 100 bootstrap iterations.")
    try:
        documents = {mode: json.loads(getattr(args, mode).read_text(encoding="utf-8")) for mode in MODES}
        output = compare(documents, iterations=args.bootstrap_iterations, seed=args.seed)
    except (KeyError, OSError, ValueError) as error:
        parser.error(str(error))
    payload = json.dumps(output, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
