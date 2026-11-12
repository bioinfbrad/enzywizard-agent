from __future__ import annotations

from collections import defaultdict
import random
from statistics import mean, stdev
from typing import Any


METRIC_NAMES = ("FP", "KFR", "IIS", "EA_F1", "TSIA")


def validate_iis_review(
    packet: dict[str, Any], review: dict[str, Any]
) -> list[str]:
    """Validate the IIS and TSIA semantic-review contract for Test 4."""

    errors: list[str] = []
    dimensions = {
        item["dimension_id"]: item for item in packet["iis_dimensions"]
    }
    judgments = review.get("interpretation_judgments")
    if not isinstance(judgments, list):
        return ["interpretation_judgments must be a list"]
    actual_ids = [str(item.get("dimension_id", "")) for item in judgments]
    if len(actual_ids) != len(set(actual_ids)):
        errors.append("interpretation_judgments contains duplicate identifiers")
    if set(actual_ids) != set(dimensions):
        errors.append(
            "interpretation_judgments identifiers differ: "
            f"expected={list(dimensions)}, actual={actual_ids}"
        )
    for item in judgments:
        expected = dimensions.get(item.get("dimension_id"))
        if expected is None:
            continue
        applicable = item.get("applicable")
        score = item.get("score")
        if bool(applicable) != bool(expected["applicable"]):
            errors.append(
                f"applicability changed for {item.get('dimension_id')}"
            )
        if expected["applicable"]:
            if (
                not isinstance(score, int)
                or isinstance(score, bool)
                or score not in {0, 1, 2}
            ):
                errors.append(
                    f"applicable IIS dimension {item.get('dimension_id')} "
                    "requires an integer score in {0, 1, 2}"
                )
        elif score is not None:
            errors.append(
                f"non-applicable IIS dimension {item.get('dimension_id')} "
                "must have score=null"
            )
    terminology = review.get("terminology_judgments")
    if not isinstance(terminology, list):
        errors.append("terminology_judgments must be a list")
        return errors
    expected_term_ids = {
        str(item["term_id"]) for item in packet.get("terminology_items", [])
    }
    actual_term_ids = [str(item.get("term_id", "")) for item in terminology]
    if any(not item for item in actual_term_ids) or len(actual_term_ids) != len(
        set(actual_term_ids)
    ):
        errors.append("terminology_judgments contains empty or duplicate identifiers")
    if set(actual_term_ids) != expected_term_ids:
        errors.append(
            "terminology_judgments identifiers differ: "
            f"expected={sorted(expected_term_ids)}, actual={actual_term_ids}"
        )
    for item in terminology:
        if not isinstance(item.get("understood"), bool):
            errors.append(
                "every terminology_judgments item requires understood=true/false"
            )
    return errors


def _score_tsia(
    packet: dict[str, Any], review: dict[str, Any]
) -> tuple[float | None, dict[str, Any]]:
    items = packet.get("terminology_items", [])
    judgments = review.get("terminology_judgments", [])
    understood = sum(bool(item["understood"]) for item in judgments)
    value = understood / len(items) if items else None
    return value, {
        "correctly_interpreted_terms": understood,
        "required_terms": len(items),
        "evaluation_source": review.get(
            "evaluation_mode", "unspecified_terminology_review"
        ),
        "applicable": bool(items),
    }


def validate_full_review(
    packet: dict[str, Any], review: dict[str, Any]
) -> list[str]:
    """Validate item-level semantic judgments for all five Test 4 metrics."""
    errors = validate_iis_review(packet, review)

    def validate_identifiers(
        field: str, expected: set[str], identifier: str, judgment_field: str
    ) -> None:
        judgments = review.get(field)
        if not isinstance(judgments, list):
            errors.append(f"{field} must be a list")
            return
        actual = [str(item.get(identifier, "")) for item in judgments]
        if any(not item for item in actual) or len(actual) != len(set(actual)):
            errors.append(f"{field} contains empty or duplicate identifiers")
        if set(actual) != expected:
            errors.append(
                f"{field} identifiers differ: expected={sorted(expected)}, "
                f"actual={actual}"
            )
        for item in judgments:
            if not isinstance(item.get(judgment_field), bool):
                errors.append(
                    f"every {field} item requires {judgment_field}=true/false"
                )

    factual = review.get("factual_judgments")
    if not isinstance(factual, list):
        errors.append("factual_judgments must be a list")
    else:
        claim_ids = [str(item.get("claim_id", "")) for item in factual]
        if any(not item for item in claim_ids) or len(claim_ids) != len(set(claim_ids)):
            errors.append("factual_judgments contains empty or duplicate claim IDs")
        for item in factual:
            if not str(item.get("claim", "")).strip():
                errors.append("every factual judgment requires a non-empty claim")
            if not isinstance(item.get("supported"), bool):
                errors.append("every factual judgment requires supported=true/false")

    required_ids = {str(item["fact_id"]) for item in packet["required_facts"]}
    validate_identifiers(
        "key_fact_judgments", required_ids, "fact_id", "covered"
    )
    association_judgments = review.get("evidence_association_judgments")
    if not isinstance(association_judgments, list):
        errors.append("evidence_association_judgments must be a list")
    else:
        association_ids = [
            str(item.get("association_id", "")) for item in association_judgments
        ]
        if any(not item for item in association_ids) or len(association_ids) != len(
            set(association_ids)
        ):
            errors.append(
                "evidence_association_judgments contains empty or duplicate identifiers"
            )
        for item in association_judgments:
            if not str(item.get("claim", "")).strip() or not str(
                item.get("evidence_reference", "")
            ).strip():
                errors.append(
                    "every evidence association judgment requires a claim and evidence reference"
                )
            if not isinstance(item.get("supported"), bool):
                errors.append(
                    "every evidence association judgment requires supported=true/false"
                )
    validate_identifiers(
        "evidence_coverage_judgments",
        required_ids,
        "fact_id",
        "evidence_supported",
    )
    return errors


def score_case(
    packet: dict[str, Any], review: dict[str, Any]
) -> tuple[dict[str, float | None], dict[str, Any]]:
    """Calculate automatic FP/KFR/EA-F1 and reviewed IIS/TSIA."""

    errors = validate_iis_review(packet, review)
    if errors:
        raise ValueError("Invalid IIS review: " + "; ".join(errors))

    objective_metrics_applicable = (
        packet.get("category_id") != "boundary_unrelated_analysis"
    )
    atomic = packet["atomic_facts"]
    fp = (
        sum(bool(item["automatically_supported"]) for item in atomic) / len(atomic)
        if objective_metrics_applicable and atomic
        else None
    )

    required = packet["required_facts"]
    kfr = (
        sum(bool(item["automatically_covered"]) for item in required)
        / len(required)
        if objective_metrics_applicable and required
        else None
    )

    interpretation = [
        item
        for item in review["interpretation_judgments"]
        if item["applicable"]
    ]
    iis = (
        sum(int(item["score"]) for item in interpretation)
        / (2 * len(interpretation))
        if interpretation
        else None
    )
    tsia, tsia_details = _score_tsia(packet, review)

    associations = packet["evidence_associations"]
    if objective_metrics_applicable and required:
        ea_precision = (
            sum(bool(item["automatically_supported"]) for item in associations)
            / len(associations)
            if associations
            else 0.0
        )
        ea_recall = (
            sum(bool(item["evidence_supported"]) for item in required)
            / len(required)
        )
        ea_f1 = (
            2 * ea_precision * ea_recall / (ea_precision + ea_recall)
            if ea_precision + ea_recall
            else 0.0
        )
    else:
        ea_precision = None
        ea_recall = None
        ea_f1 = None

    metrics = {
        "FP": fp,
        "KFR": kfr,
        "IIS": iis,
        "EA_F1": ea_f1,
        "TSIA": tsia,
    }
    details = {
        "FP": {
            "supported_atomic_facts": sum(
                bool(item["automatically_supported"]) for item in atomic
            ),
            "verifiable_atomic_facts": len(atomic),
            "evaluation_source": "structured_metric_facts_and_fixed_gold_reports",
            "applicable": objective_metrics_applicable,
        },
        "KFR": {
            "covered_required_facts": sum(
                bool(item["automatically_covered"]) for item in required
            ),
            "required_facts": len(required),
            "evaluation_source": "predefined_gold_metric_requirements",
            "applicable": objective_metrics_applicable,
        },
        "IIS": {
            "obtained_score": sum(int(item["score"]) for item in interpretation),
            "maximum_score": 2 * len(interpretation),
            "applicable_dimensions": len(interpretation),
            "evaluation_source": review.get(
                "evaluation_mode", "unspecified_iis_rubric"
            ),
        },
        "EA_F1": {
            "supported_associations": sum(
                bool(item["automatically_supported"]) for item in associations
            ),
            "generated_associations": len(associations),
            "evidence_covered_required_facts": sum(
                bool(item["evidence_supported"]) for item in required
            ),
            "required_facts": len(required),
            "precision": ea_precision,
            "recall": ea_recall,
            "evaluation_source": "structured_provenance_validation",
            "applicable": objective_metrics_applicable,
        },
        "TSIA": tsia_details,
    }
    return metrics, details


def score_case_full_review(
    packet: dict[str, Any], review: dict[str, Any]
) -> tuple[dict[str, float | None], dict[str, Any]]:
    """Calculate all metrics from item-level human or LLM semantic judgments."""
    errors = validate_full_review(packet, review)
    if errors:
        raise ValueError("Invalid full review: " + "; ".join(errors))

    objective_metrics_applicable = (
        packet.get("category_id") != "boundary_unrelated_analysis"
    )
    factual = review["factual_judgments"]
    fp = (
        sum(bool(item["supported"]) for item in factual) / len(factual)
        if objective_metrics_applicable and factual
        else None
    )
    key_facts = review["key_fact_judgments"]
    kfr = (
        sum(bool(item["covered"]) for item in key_facts) / len(key_facts)
        if objective_metrics_applicable and key_facts
        else None
    )
    interpretation = [
        item for item in review["interpretation_judgments"] if item["applicable"]
    ]
    iis = (
        sum(int(item["score"]) for item in interpretation)
        / (2 * len(interpretation))
        if interpretation
        else None
    )
    tsia, tsia_details = _score_tsia(packet, review)

    associations = review["evidence_association_judgments"]
    coverage = review["evidence_coverage_judgments"]
    if objective_metrics_applicable and key_facts:
        ea_precision = (
            sum(bool(item["supported"]) for item in associations) / len(associations)
            if associations
            else 0.0
        )
        ea_recall = sum(bool(item["evidence_supported"]) for item in coverage) / len(coverage)
        ea_f1 = (
            2 * ea_precision * ea_recall / (ea_precision + ea_recall)
            if ea_precision + ea_recall
            else 0.0
        )
    else:
        ea_precision = ea_recall = ea_f1 = None

    source = review.get("evaluation_mode", "full_semantic_review")
    metrics = {
        "FP": fp,
        "KFR": kfr,
        "IIS": iis,
        "EA_F1": ea_f1,
        "TSIA": tsia,
    }
    details = {
        "FP": {
            "supported_atomic_facts": sum(bool(item["supported"]) for item in factual),
            "verifiable_atomic_facts": len(factual),
            "evaluation_source": source,
            "applicable": objective_metrics_applicable,
        },
        "KFR": {
            "covered_required_facts": sum(bool(item["covered"]) for item in key_facts),
            "required_facts": len(key_facts),
            "evaluation_source": source,
            "applicable": objective_metrics_applicable,
        },
        "IIS": {
            "obtained_score": sum(int(item["score"]) for item in interpretation),
            "maximum_score": 2 * len(interpretation),
            "applicable_dimensions": len(interpretation),
            "evaluation_source": source,
        },
        "EA_F1": {
            "supported_associations": sum(bool(item["supported"]) for item in associations),
            "generated_associations": len(associations),
            "evidence_covered_required_facts": sum(
                bool(item["evidence_supported"]) for item in coverage
            ),
            "required_facts": len(coverage),
            "precision": ea_precision,
            "recall": ea_recall,
            "evaluation_source": source,
            "applicable": objective_metrics_applicable,
        },
        "TSIA": tsia_details,
    }
    return metrics, details


def repetition_statistics(results: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for metric in METRIC_NAMES:
        by_repeat: dict[int, list[float]] = defaultdict(list)
        for result in results:
            value = result.get("metrics", {}).get(metric)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                by_repeat[int(result["repetition"])].append(float(value))
        by_repetition = {
            str(repetition): mean(values)
            for repetition, values in sorted(by_repeat.items())
        }
        run_values = list(by_repetition.values())
        output[metric] = {
            "by_repetition": by_repetition,
            "mean": mean(run_values) if run_values else None,
            "sd": (
                stdev(run_values)
                if len(run_values) > 1
                else (0.0 if run_values else None)
            ),
            "evaluated_case_runs": sum(len(values) for values in by_repeat.values()),
        }
    return output


def cluster_bootstrap_ci(
    results: list[dict[str, Any]], *, iterations: int, seed: int
) -> dict[str, dict[str, float | None]]:
    """Resample case IDs while retaining all repetitions for a sampled case."""

    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_case[str(result["case_id"])].append(result)
    rng = random.Random(seed)
    output: dict[str, dict[str, float | None]] = {}
    for metric in METRIC_NAMES:
        eligible = [
            case_id
            for case_id, case_results in sorted(by_case.items())
            if any(
                isinstance(item.get("metrics", {}).get(metric), (int, float))
                and not isinstance(item.get("metrics", {}).get(metric), bool)
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
                and not isinstance(item.get("metrics", {}).get(metric), bool)
            ]
            if values:
                estimates.append(mean(values))
        estimates.sort()
        if not estimates:
            output[metric] = {"lower": None, "upper": None}
            continue
        lower_index = int(0.025 * (len(estimates) - 1))
        upper_index = int(0.975 * (len(estimates) - 1))
        output[metric] = {
            "lower": estimates[lower_index],
            "upper": estimates[upper_index],
        }
    return output
