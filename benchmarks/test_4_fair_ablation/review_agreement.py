#!/usr/bin/env python3
"""Compare human and Judge CDA scores for the saved blind sample."""

import argparse
import json
from pathlib import Path

from metrics import is_binary


def compare(human, judge):
    identity = ("schema_version", "review_id", "repetition", "sample_seed", "sample_case_ids")
    if any(human[key] != judge[key] for key in identity):
        raise ValueError("The human and Judge files describe different samples.")
    ids = human["sample_case_ids"]
    if len(ids) != len(set(ids)) or any([row["case_id"] for row in doc["cases"]] != ids for doc in (human, judge)):
        raise ValueError("The review items do not match the sample IDs.")
    observations = []
    unavailable = []
    for left, right in zip(human["cases"], judge["cases"], strict=True):
        if right["score"] is None:
            unavailable.append(left["case_id"])
            continue
        if not is_binary(left["score"]) or not is_binary(right["score"]):
            raise ValueError(f"Enter 0 or 1 for {left['case_id']}.")
        observations.append((left["score"], right["score"]))
    size = len(observations)
    matches = sum(left == right for left, right in observations)
    h, j = (sum(item[index] for item in observations) for index in (0, 1))
    expected = h * j + (size - h) * (size - j)
    kappa = (matches * size - expected) / (size * size - expected) if size * size != expected else None
    return {
        "review_id": human["review_id"], "item_count": size, "unavailable_cases": unavailable,
        "agreement": matches / size if size else None, "cohens_kappa": kappa,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    try:
        human, judge = [json.loads((args.run_dir / f"{name}_review.json").read_text(encoding="utf-8"))
                        for name in ("human", "judge")]
        result = compare(human, judge)
    except (KeyError, ValueError, OSError) as error:
        parser.error(str(error))
    payload = json.dumps(result, indent=2) + "\n"
    (args.run_dir / "review_agreement.json").write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
