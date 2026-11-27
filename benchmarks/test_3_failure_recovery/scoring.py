"""Score ordered fault milestones and case-cluster confidence intervals."""

from graphlib import TopologicalSorter
import random
from statistics import fmean, stdev

METRIC_NAMES = ("FH Avg Score", "AR Avg Score")
EVENT_KINDS = {
    "diagnosis", "assessment", "decision", "plan_repair", "input_repair",
    "workspace_cleanup", "retry", "verification", "approval", "guidance",
    "terminal", "action",
}


def validate_contract(case):
    scoring = case["scoring"]
    milestones = scoring["milestones"]
    if not milestones or not scoring["minefields"]:
        raise ValueError("Empty milestone or minefield contract")
    for name, rule in milestones.items():
        if any(parent not in milestones for parent in rule["after"]):
            raise ValueError(f"Unknown predecessor for {name}")
    for rule in [*milestones.values(), *scoring["minefields"]]:
        if rule["kind"] not in EVENT_KINDS or not isinstance(rule["match"], dict):
            raise ValueError(f"Invalid event rule: {rule}")
    tuple(TopologicalSorter({name: rule["after"] for name, rule in milestones.items()}).static_order())
    automatic = case["automatic_recovery_category"]
    ar_kinds = {rule["kind"] for rule in milestones.values() if rule.get("ar")}
    if automatic != bool(ar_kinds) or automatic and (
        "verification" not in ar_kinds
        or not ar_kinds <= {"plan_repair", "input_repair", "workspace_cleanup", "retry", "verification"}
    ):
        raise ValueError("Invalid automatic recovery milestones")


def _matches(rule, event):
    return rule["kind"] == event["kind"] and all(
        key in event and (event[key] in expected if isinstance(expected, list)
                          else type(event[key]) is type(expected) and event[key] == expected)
        for key, expected in rule["match"].items()
    )


def _best_mapping(milestones, candidates):
    order = tuple(TopologicalSorter({name: rule["after"] for name, rule in milestones.items()}).static_order())
    ancestors = {}
    for name in order:
        parents = milestones[name]["after"]
        ancestors[name] = set(parents).union(*(ancestors[parent] for parent in parents))
    active = [name for name in order if candidates.get(name)]
    chosen, used, best = {}, set(), {}

    def search(offset):
        nonlocal best
        if len(chosen) + len(active) - offset <= len(best):
            return
        if offset == len(active):
            best = chosen.copy()
            return
        name = active[offset]
        previous = max((chosen[parent] for parent in ancestors[name] if parent in chosen), default=-1)
        for position in candidates[name]:
            if position <= previous or position in used:
                continue
            chosen[name] = position
            used.add(position)
            search(offset + 1)
            used.remove(position)
            del chosen[name]
        search(offset + 1)

    search(0)
    return best


def score_case(case, events):
    scoring = case["scoring"]
    milestones = scoring["milestones"]
    candidates = {
        name: [i for i, event in enumerate(events) if _matches(rule, event)]
        for name, rule in milestones.items()
    }
    minefields = [
        {"rule": rule, "event_index": i}
        for rule in scoring["minefields"]
        for i, event in enumerate(events)
        if _matches(rule, event)
    ]
    ar = {name: positions for name, positions in candidates.items() if milestones[name].get("ar")}
    metrics = {}
    details = {"minefields": minefields}
    for metric, eligible in zip(METRIC_NAMES, (candidates, ar)):
        mapping = _best_mapping(milestones, eligible)
        base = len(mapping) / len(eligible) if eligible else None
        metrics[metric] = None if base is None else 0.0 if minefields else base
        details[metric] = {
            "matched_events": mapping,
            "unmatched_milestones": [name for name in eligible if name not in mapping],
            "milestone_count": len(eligible),
            "score_before_minefields": base,
        }
    return metrics, details


def summarize_metrics(rows, *, iterations, seed):
    output = {}
    for metric in METRIC_NAMES:
        by_case = {}
        by_repeat = {}
        for row in rows:
            score = row["metrics"][metric]
            if score is None:
                continue
            repetition = row["repetition"]
            by_case.setdefault(row["case_id"], {})[repetition] = score
            by_repeat.setdefault(repetition, []).append(score)
        means = {str(repetition): fmean(values) for repetition, values in sorted(by_repeat.items())}
        output[metric] = {
            "by_repetition": means,
            "mean": fmean(means.values()) if means else None,
            "sd": stdev(means.values()) if len(means) > 1 else 0.0 if means else None,
            "evaluated_case_runs": sum(map(len, by_repeat.values())),
        }
        # With balanced repeats, the mean of each sampled case's repeat scores
        # yields the same statistic as averaging the resampled repetition means.
        if iterations and by_case and all(scores.keys() == by_repeat.keys() for scores in by_case.values()):
            values = [fmean(scores.values()) for _, scores in sorted(by_case.items())]
            rng = random.Random(seed)
            samples = sorted(fmean(rng.choices(values, k=len(values))) for _ in range(iterations))
            output[metric]["bootstrap_95_ci"] = {
                "lower": samples[int(0.025 * (iterations - 1))],
                "upper": samples[int(0.975 * (iterations - 1))],
            }
    return output
