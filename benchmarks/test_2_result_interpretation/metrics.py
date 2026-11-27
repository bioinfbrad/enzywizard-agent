"""Case-macro Claim F1 and Citation F1 and clustered uncertainty estimates."""

from collections import defaultdict
import random
from statistics import mean, quantiles, stdev

METRICS = ("Claim F1", "Citation F1")


def case_rates(review):
    claims = review["claims"]
    required = review["required_claims"]
    claim_rates = None
    if required:
        claim_rates = {
            "precision": sum(item["supported"] for item in claims) / len(claims) if claims else 0.0,
            "recall": sum(item["covered"] for item in required) / len(required),
        }
    citation_rates = None
    citation_claims = [item for item in claims if item.get("requires_citation", True)]
    if citation_claims and not review.get("citation_errors"):
        precision, recall = [], []
        for item in citation_claims:
            actual, reference = item["citations"], item["reference_evidence"]
            matches = sum(citation["matched_reference_index"] is not None for citation in actual)
            precision.append(matches / len(actual) if actual else 0.0)
            recall.append(matches / len(reference) if reference else 0.0)
        citation_rates = {"precision": mean(precision), "recall": mean(recall)}
    return {"Claim F1": claim_rates, "Citation F1": citation_rates}


def macro_rates(values):
    applicable = [value for value in values if value is not None]
    if not applicable:
        return {"precision": None, "recall": None, "f1": None, "applicable_cases": 0}
    precision = mean(value["precision"] for value in applicable)
    recall = mean(value["recall"] for value in applicable)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "applicable_cases": len(applicable),
    }


def summarize_metrics(rows, case_ids, repetitions, *, iterations, seed):
    by_rep = defaultdict(dict)
    for row in rows:
        if row["rates"] is not None:
            by_rep[row["repetition"]][row["case_id"]] = {
                name: value for name, value in row["rates"].items()
                if name not in row.get("metric_errors", {})
            }
    output = {}
    for metric in METRICS:
        usable = {
            rep: {case_id: results[metric] for case_id, results in by_rep[rep].items() if metric in results}
            for rep in range(1, repetitions + 1)
        }
        complete = all(group.keys() == set(case_ids) for group in usable.values())
        by_repetition = {
            rep: macro_rates(usable[rep].values())
            for rep in range(1, repetitions + 1)
        }
        scores = [item["f1"] for item in by_repetition.values() if item["f1"] is not None]
        result = {
            "by_repetition": by_repetition,
            "mean": mean(scores) if scores else None,
            "sd": stdev(scores) if len(scores) > 1 else None,
            "complete_reviews": complete,
            "scored_or_inapplicable_reviews": sum(map(len, usable.values())),
        }
        if iterations and len(scores) == repetitions:
            # Restart with the same seed so the two metrics use the same case draws.
            rng = random.Random(seed)
            distribution = []
            for _ in range(iterations):
                sample = rng.choices(case_ids, k=len(case_ids))
                drawn = [
                    macro_rates(usable[rep].get(case_id) for case_id in sample)["f1"]
                    for rep in range(1, repetitions + 1)
                ]
                if all(score is not None for score in drawn):
                    distribution.append(mean(drawn))
            if len(distribution) >= 2:
                bounds = quantiles(distribution, n=40, method="inclusive")
                result["bootstrap_95_ci"] = {"lower": bounds[0], "upper": bounds[-1]}
                result["valid_bootstrap_draws"] = len(distribution)
        output[metric] = result
    return output
