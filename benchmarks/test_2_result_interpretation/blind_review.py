"""Write case-by-case, matched human and judge review files."""

import math
import random
from pathlib import Path

from benchmark_support import _atomic_json
from metrics import macro_rates


HUMAN_INSTRUCTIONS = [
    "Review one case at a time. Read the question and any prior turn, then review the candidate answer and sources.",
    "Enter 0 or 1 for every null judgment in claims and required_claims. Use source details to verify citations.",
    "Keep item IDs and lists intact. Note missing or incorrect claim splits in reviewer_notes.",
    "The items are aligned for agreement on judgments. Judge labels and reasons are in the separate judge_review.json.",
]


def _claim_items(claim, claim_id, *, human, citation_available):
    item = {
        "id": claim_id,
        "claim": claim["claim"],
        "supported": None if human else int(claim["supported"]),
    }
    if not human:
        item["reason"] = claim["reason"]
    if not citation_available or not claim.get("requires_citation", True):
        item["citations"] = []
        item["reference_evidence"] = []
        return item
    matches = {citation["matched_reference_index"] for citation in claim["citations"]}
    item["citations"] = []
    for n, citation in enumerate(claim["citations"], 1):
        index = citation["matched_reference_index"]
        link = {
            "id": f"{claim_id}.CIT{n:02}",
            "actual_source": citation["actual_source"],
            "source_id": citation.get("source_id"),
            "correct": None if human else int(index is not None),
        }
        if not human:
            link["matched_reference_id"] = f"{claim_id}.REF{index:02}" if index is not None else None
            link["reason"] = citation["reason"]
        item["citations"].append(link)
    item["reference_evidence"] = [
        {**ref, "id": f"{claim_id}.REF{n:02}", "covered": None if human else int(n in matches)}
        for n, ref in enumerate(claim["reference_evidence"], 1)
    ]
    return item


def _case_review(row, *, human):
    case = {"case_id": row["case_id"], "review_available": row["review"] is not None}
    if not case["review_available"]:
        case["review_status"] = "No usable judge review was produced for this case."
        return case
    packet, review = row["packet"], row["review"]
    citation_available = not review.get("citation_errors")
    case["citation_review_available"] = citation_available
    if not citation_available:
        case["citation_review_status"] = "Citation review failed validation. Claim judgments are available."
    if human:
        case["question"] = packet["question"]
        if packet["initial_question"]:
            case["prior_turn"] = {
                "question": packet["initial_question"],
                "answer_lines": (packet["initial_answer"] or "").splitlines(),
            }
        case["candidate_answer_lines"] = packet["candidate_answer"].splitlines()
        case["reviewer_notes"] = ""
    case["claims"] = [
        _claim_items(claim, f"CL{n:02}", human=human, citation_available=citation_available)
        for n, claim in enumerate(review["claims"], 1)
    ]
    by_id = {item["fact_id"]: item for item in review["required_claims"]}
    case["required_claims"] = [
        {
            "id": fact["fact_id"],
            **({"reference": {key: value for key, value in fact.items() if key != "fact_id"},
                "covered": None} if human else {
                    "covered": int(by_id[fact["fact_id"]]["covered"]),
                    "reason": by_id[fact["fact_id"]]["reason"],
                }),
        }
        for fact in packet["kfr_key_facts"]
    ]
    if human:
        case["case_requirements"] = packet["iis_requirements"]
        case["candidate_limitations"] = packet["candidate_limitations"]
        case["report_summaries"] = packet["report_summaries"]
        case["structured_citations"] = packet.get("claim_citations", [])
        case["evidence_registry"] = packet["candidate_evidence_registry"]
    else:
        case["case_scores"] = {
            metric: macro_rates([rates]) for metric, rates in row["rates"].items()
            if metric not in row.get("metric_errors", {})
        }
    return case


def export_blind_review(run_dir: Path, rows, case_ids, seed):
    first = {row["case_id"]: row for row in rows if row["repetition"] == 1}
    count = math.ceil(len(case_ids) / 5)
    selected = set(random.Random(seed).sample(sorted(case_ids), count))
    sample = [case_id for case_id in case_ids if case_id in selected]
    common = {
        "schema_version": 1, "run_id": run_dir.name, "repetition": 1,
        "sample_seed": seed, "sample_case_ids": sample,
    }
    _atomic_json(run_dir / "human_review.json", {
        **common, "instructions": HUMAN_INSTRUCTIONS,
        "cases": [_case_review(first[case_id], human=True) for case_id in sample],
    })
    _atomic_json(run_dir / "judge_review.json", {
        **common, "cases": [_case_review(first[case_id], human=False) for case_id in sample],
    })
