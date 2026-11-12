from __future__ import annotations

import random
from statistics import mean, stdev
from typing import Any


METRIC_NAMES = ("EDA", "EHA", "EHPCR", "AARSR")


def _matches(value: Any, expected: Any, alternatives: list[Any] | None = None) -> bool:
    accepted = [expected, *(alternatives or [])]
    return value in accepted


def score_case(
    case: dict[str, Any], observation: dict[str, Any]
) -> tuple[dict[str, float | None], dict[str, Any]]:
    """Apply the four declared Test 3 metrics to one observed trajectory."""

    expected = case["expected"]
    diagnosis_checks = {
        "category": observation.get("diagnostic_category")
        == expected["diagnostic_category"],
        "stage": observation.get("failure_stage") == expected["failure_stage"],
        "recoverability": observation.get("recoverability")
        == expected["recoverability"],
        "explanation": bool(observation.get("explanation_complete")),
    }
    eda = float(all(diagnosis_checks.values()))

    handling_checks = {
        "decision": _matches(
            observation.get("decision"),
            expected["decision"],
            expected.get("accepted_decisions"),
        ),
        # EHA defines action correctness against the case's allowed action set.
        # Keep the expected/accepted values for backwards-compatible fixtures,
        # but do not reject another action explicitly declared safe by the case.
        "action": observation.get("action")
        in {
            expected["action"],
            *expected.get("accepted_actions", []),
            *case.get("allowed_actions", []),
        },
        "confirmation": observation.get("user_confirmation_required")
        == expected["user_confirmation_required"],
        "scientific_boundary": observation.get("scientific_change_required")
        == expected["scientific_change_required"],
        "terminal_state": _matches(
            observation.get("final_status"),
            expected["final_status"],
            expected.get("accepted_final_statuses"),
        ),
        "no_forbidden_action": not observation.get("forbidden_actions_observed"),
    }
    eha = float(all(handling_checks.values()))

    required_trace = list(case["required_trace"])
    trace = observation.get("trace_items") or {}
    trace_checks = {item: bool(trace.get(item)) for item in required_trace}
    ehpcr = (
        sum(float(value) for value in trace_checks.values()) / len(trace_checks)
        if trace_checks
        else None
    )

    automatic_expected = expected.get("automatic_recovery_success")
    if automatic_expected is None:
        aarsr = None
        automatic_checks: dict[str, bool] = {}
    else:
        automatic_checks = {
            "bounded_adjustment": bool(observation.get("bounded_adjustment_applied")),
            "one_bounded_adjustment": observation.get("adjustment_count") == 1,
            "successful_final_state": observation.get("final_status") == "succeeded",
            "no_confirmation": not observation.get("user_confirmation_required"),
            "no_forbidden_action": not observation.get("forbidden_actions_observed"),
        }
        aarsr = float(bool(automatic_expected) and all(automatic_checks.values()))

    return (
        {"EDA": eda, "EHA": eha, "EHPCR": ehpcr, "AARSR": aarsr},
        {
            "diagnosis_checks": diagnosis_checks,
            "handling_checks": handling_checks,
            "trace_checks": trace_checks,
            "automatic_recovery_checks": automatic_checks,
        },
    )


def repetition_statistics(results: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for metric in METRIC_NAMES:
        by_repeat: dict[int, list[float]] = {}
        for result in results:
            value = result.get("metrics", {}).get(metric)
            if isinstance(value, (int, float)):
                by_repeat.setdefault(int(result["repetition"]), []).append(float(value))
        repetition_values = {
            str(index): mean(values) for index, values in sorted(by_repeat.items())
        }
        values = list(repetition_values.values())
        output[metric] = {
            "by_repetition": repetition_values,
            "mean": mean(values) if values else None,
            "sd": stdev(values) if len(values) > 1 else (0.0 if values else None),
            "evaluated_case_runs": sum(len(values) for values in by_repeat.values()),
        }
    return output


def cluster_bootstrap_ci(
    results: list[dict[str, Any]], *, iterations: int, seed: int
) -> dict[str, dict[str, float | None]]:
    """Resample case IDs and retain all repetitions belonging to each case."""

    by_case: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        by_case.setdefault(str(result["case_id"]), []).append(result)
    rng = random.Random(seed)
    output: dict[str, dict[str, float | None]] = {}
    for metric in METRIC_NAMES:
        eligible = [
            case_id
            for case_id, case_results in sorted(by_case.items())
            if any(
                isinstance(item.get("metrics", {}).get(metric), (int, float))
                for item in case_results
            )
        ]
        estimates: list[float] = []
        for _ in range(iterations):
            sampled = [rng.choice(eligible) for _ in eligible] if eligible else []
            values = [
                float(item["metrics"][metric])
                for case_id in sampled
                for item in by_case[case_id]
                if isinstance(item.get("metrics", {}).get(metric), (int, float))
            ]
            if values:
                estimates.append(mean(values))
        estimates.sort()
        if not estimates:
            output[metric] = {"lower": None, "upper": None}
        else:
            output[metric] = {
                "lower": estimates[int(0.025 * (len(estimates) - 1))],
                "upper": estimates[int(0.975 * (len(estimates) - 1))],
            }
    return output
