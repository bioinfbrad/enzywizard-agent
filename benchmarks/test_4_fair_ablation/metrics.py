"""Compute CDA and case-cluster confidence intervals."""

import random
from statistics import fmean, quantiles, stdev


def is_binary(value):
    return type(value) is int and value in (0, 1)


def score_map(rows, cases, repetitions):
    allowed = {(case_id, repetition) for case_id in cases for repetition in range(1, repetitions + 1)}
    seen = set()
    scores = {}
    for row in rows:
        key = (row["case_id"], row["repetition"])
        if key not in allowed or key in seen:
            raise ValueError(f"Unexpected or duplicate case/repetition: {key}")
        seen.add(key)
        value = row["score"]
        if is_binary(value):
            scores[key] = value
        elif value is not None:
            raise ValueError(f"Invalid score for {key}")
    return scores


def interval(case_values, iterations, seed):
    if iterations < 2 or not case_values:
        return None
    rng = random.Random(seed)
    estimates = [fmean(rng.choices(case_values, k=len(case_values))) for _ in range(iterations)]
    limits = quantiles(estimates, n=40, method="inclusive")
    return {"lower": limits[0], "upper": limits[-1]}


def summarize(rows, cases, repetitions, *, iterations, seed):
    scores = score_map(rows, cases, repetitions)
    by_rep = {}
    for repetition in range(1, repetitions + 1):
        values = [scores[(cid, repetition)] for cid in cases if (cid, repetition) in scores]
        by_rep[str(repetition)] = {
            "correct": sum(values), "scored": len(values), "applicable_cases": len(cases),
            "value": fmean(values) if values else None,
        }
    complete = len(scores) == len(cases) * repetitions
    values = [rep["value"] for rep in by_rep.values() if rep["value"] is not None]
    result = {
        "by_repetition": by_rep, "mean": fmean(values) if values else None,
        "sd": stdev(values) if len(values) > 1 else None,
        "complete_reviews": complete, "scored_case_runs": len(scores),
    }
    if complete and iterations:
        clustered = [fmean(scores[(cid, rep)] for rep in range(1, repetitions + 1)) for cid in cases]
        result["bootstrap_95_ci"] = interval(clustered, iterations, seed)
    elif iterations and len(values) == repetitions:
        rng = random.Random(seed)
        distribution = []
        for _ in range(iterations):
            sample = rng.choices(cases, k=len(cases))
            by_rep_sample = [
                [scores[cid, rep] for cid in sample if (cid, rep) in scores]
                for rep in range(1, repetitions + 1)
            ]
            if all(by_rep_sample):
                distribution.append(fmean(fmean(values) for values in by_rep_sample))
        if len(distribution) >= 2:
            limits = quantiles(distribution, n=40, method="inclusive")
            result["bootstrap_95_ci"] = {"lower": limits[0], "upper": limits[-1]}
            result["valid_bootstrap_draws"] = len(distribution)
    return result
