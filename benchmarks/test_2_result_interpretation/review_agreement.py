#!/usr/bin/env python3
"""Compare completed blind labels against the paired Claude labels."""

import argparse
import json
from pathlib import Path

GROUPS = ("claim_support", "required_claim_coverage", "citation_correctness", "reference_evidence_coverage")


def _items(document):
    output = {group: {} for group in GROUPS}

    def assign(group, case_id, item, field):
        key = (case_id, item["id"])
        if key in output[group]:
            raise ValueError(f"Duplicate review item {case_id}/{item['id']} in {group}")
        output[group][key] = item[field]

    actual_case_ids = [case["case_id"] for case in document["cases"]]
    if actual_case_ids != document["sample_case_ids"] or len(actual_case_ids) != len(set(actual_case_ids)):
        raise ValueError("Case list does not match the sampled cases.")
    for case in document["cases"]:
        if not case["review_available"]:
            continue
        case_id = case["case_id"]
        for item in case["claims"]:
            assign("claim_support", case_id, item, "supported")
            for link in item["citations"]:
                assign("citation_correctness", case_id, link, "correct")
            for ref in item["reference_evidence"]:
                assign("reference_evidence_coverage", case_id, ref, "covered")
        for item in case["required_claims"]:
            assign("required_claim_coverage", case_id, item, "covered")
    return output


def _agreement(human, judge):
    if human.keys() != judge.keys():
        raise ValueError("Human and judge item IDs do not match.")
    for key in human:
        if any(type(value) is not int or value not in (0, 1) for value in (human[key], judge[key])):
            raise ValueError(f"Enter 0 or 1 for every judgment. Check {key[0]}/{key[1]}.")
    n = len(human)
    if not n:
        return {"item_count": 0, "agreement": None, "cohens_kappa": None}
    equal = sum(human[key] == judge[key] for key in human)
    agreement = equal / n
    h, j = sum(human.values()), sum(judge.values())
    expected_equal = h * j + (n - h) * (n - j)
    kappa = (equal * n - expected_equal) / (n * n - expected_equal) if expected_equal != n * n else None
    return {"item_count": n, "agreement": agreement, "cohens_kappa": kappa}


def compare(human, judge):
    identity = ("schema_version", "run_id", "repetition", "sample_seed", "sample_case_ids")
    if any(human[key] != judge[key] for key in identity):
        raise ValueError("Human and judge files are from different samples.")
    left, right = _items(human), _items(judge)
    human_unavailable = [case["case_id"] for case in human["cases"] if not case["review_available"]]
    judge_unavailable = [case["case_id"] for case in judge["cases"] if not case["review_available"]]
    if human_unavailable != judge_unavailable:
        raise ValueError("Case availability differs between the human and judge files.")
    human_bad_citations = [
        case["case_id"] for case in human["cases"] if not case.get("citation_review_available", True)
    ]
    judge_bad_citations = [
        case["case_id"] for case in judge["cases"] if not case.get("citation_review_available", True)
    ]
    if human_bad_citations != judge_bad_citations:
        raise ValueError("Citation review availability differs between the human and judge files.")
    return {
        "unavailable_citation_cases": human_bad_citations,
        "run_id": human["run_id"], "repetition": human["repetition"],
        "sampled_cases": human["sample_case_ids"], "unavailable_cases": human_unavailable,
        "agreement": {group: _agreement(left[group], right[group]) for group in GROUPS},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    try:
        human = json.loads((args.run_dir / "human_review.json").read_text(encoding="utf-8"))
        judge = json.loads((args.run_dir / "judge_review.json").read_text(encoding="utf-8"))
        result = compare(human, judge)
    except (KeyError, ValueError, OSError) as error:
        parser.error(str(error))
    rendered = json.dumps(result, indent=2)
    (args.run_dir / "review_agreement.json").write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
