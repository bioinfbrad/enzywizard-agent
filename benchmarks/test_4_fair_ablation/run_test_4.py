#!/usr/bin/env python3
"""Run the independent concept definition benchmark."""

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import time
from uuid import uuid4

from enzywizard_agent.data_models.runtime import RuntimeSession
from enzywizard_agent.services.knowledge_retrieval_service import fair_terms_sha256
from enzywizard_agent.services.llm_service import check_model_capabilities
from enzywizard_agent.services.runtime_service import utc_now
from enzywizard_agent.utils.terminology_utils import scientific_answer_body, terminology_key

from cases import REPO, ROOT, load_suite, select_cases
from conversation import configuration, run_conversation
import judge
from metrics import summarize


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api_key", default=os.environ.get("ENZYWIZARD_API_KEY"))
    parser.add_argument("--base_url", default="https://api.openai.com/v1")
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--reasoning_effort", default="medium")
    parser.add_argument("--llm_timeout_seconds", type=float, default=180)
    parser.add_argument("--judge_api_key", default=os.environ.get("ANTHROPIC_API_KEY"))
    parser.add_argument("--judge_timeout_seconds", type=float, default=180)
    parser.add_argument("--ontology_mode", choices=("off", "control", "retrieve"), default="retrieve")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument("--quick_case_count", type=int, default=8)
    parser.add_argument("--case_ids", nargs="+")
    parser.add_argument("--seed", type=int, default=20261111)
    parser.add_argument("--bootstrap_iterations", type=int, default=1000)
    parser.add_argument("--workdir", type=Path, default=Path.home() / "enzywizard_benchmark_results" / ROOT.name)
    parser.add_argument("--validate_only", action="store_true")
    return parser


def _write(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def _error(error, args):
    message = str(error)
    for key in (args.api_key, args.judge_api_key):
        if key:
            message = message.replace(key, "[redacted]")
    return message


def _retrieval_audit(state, case, mode):
    runtime = state["runtime_session"]
    records = runtime.metadata.get("knowledge_retrieval", [])
    target = terminology_key(case["target_term"])
    definition_turn = 3 if "initial_question" in case else 2
    definition_records = [r for r in records if r.get("turn_id") == definition_turn]
    answer_records = [r for r in definition_records if r["stage"] in {
        "analysis.general_answer", "analysis.scientific_narrative",
    }]

    def has_target(events):
        return any(
            terminology_key(term["term"]) == target
            for event in events for term in event["matched_terms"]
        )

    return {
        "mode": mode, "knowledge_base_sha256": fair_terms_sha256(),
        "total_injected_terms": sum(r["injected_term_count"] for r in records),
        "target_injected_on_definition_turn": has_target(definition_records),
        "target_injected_into_definition_answer": has_target(answer_records),
        "target_injected_in_session": has_target(records),
        "control_count_matches": (
            all(r["injected_term_count"] == r["related_match_count"] for r in records)
            if mode == "control" else None
        ),
        "records": records,
    }


def run_case(catalog, case, repetition, args, run_dir):
    workdir = run_dir / case["case_id"] / f"repeat_{repetition:03d}"
    workdir.mkdir(parents=True)
    packet = {
        "target_term": case["target_term"], "question": case["definition_question"],
        "reference_definition": case["reference_definition"], "answer": "",
    }
    row = {
        "case_id": case["case_id"], "repetition": repetition, "category_id": case["category_id"],
        "score": None, "judge": None, "judge_usage": {}, "packet": packet,
        "case_workdir": str(workdir), "errors": [],
    }
    try:
        state, stop = run_conversation(catalog, case, args, workdir)
    except Exception as error:
        row["errors"].append(f"Conversation error: {_error(error, args)}")
        return row

    final = state.get("current_analysis_result")
    if stop or not final or final.turn_id != state["current_turn_id"]:
        final = None
    if final and final.analysis_status != "failed":
        packet["answer"] = scientific_answer_body(final.final_answer) or ""
    audit = _retrieval_audit(state, case, args.ontology_mode)
    _write(workdir / "retrieval_audit.json", audit)
    _write(workdir / "conversation.json", {
        "session_id": state["session_id"], "ontology_mode": args.ontology_mode,
        "user_messages": [message.model_dump(mode="json") for message in state.get("user_messages", [])],
        "analysis_results": [result.model_dump(mode="json") for result in state["analysis_results"]],
        "definition_turn": state["current_turn_id"] if final else None,
        "stop_reason": stop,
    })
    row["retrieval_audit"] = {key: value for key, value in audit.items() if key != "records"}
    runtime = state["runtime_session"]
    row["agent_llm_calls"] = len(runtime.llm_calls)
    if not packet["answer"].strip():
        row["score"] = 0
        row["judge"] = {"correct": 0, "reason": stop or "The Agent returned no usable definition answer."}
        row["scoring_source"] = "missing_answer"
        return row

    try:
        decision, usage = judge.review(
            packet, args.judge_api_key, args.judge_timeout_seconds,
            save_raw=lambda value: _write(workdir / "judge_response.json", value),
        )
        row.update(score=decision["correct"], judge=decision, judge_usage=usage, scoring_source="judge")
    except Exception as error:
        row["errors"].append(f"Judge error: {_error(error, args)}")
    return row


def write_reviews(run_dir, rows, case_ids, seed):
    first = {row["case_id"]: row for row in rows if row["repetition"] == 1}
    sampled = set(random.Random(seed).sample(sorted(case_ids), math.ceil(len(case_ids) / 5)))
    sample = [cid for cid in case_ids if cid in sampled]
    common = {
        "schema_version": 1, "review_id": run_dir.name, "repetition": 1,
        "sample_seed": seed, "sample_case_ids": sample,
    }
    _write(run_dir / "human_review.json", {
        **common,
        "instructions": "Read the question, reference definition and answer. Score 1 for the complete core meaning without a substantive error, otherwise 0. Accept equivalent wording.",
        "cases": [{
            "case_id": cid, **first[cid]["packet"], "score": None, "notes": "",
        } for cid in sample],
    })
    _write(run_dir / "judge_review.json", {
        **common,
        "cases": [{
            "case_id": cid, "score": first[cid]["score"], "judgment": first[cid]["judge"],
            "scoring_source": first[cid].get("scoring_source"), "errors": first[cid]["errors"],
        } for cid in sample],
    })


def _agent_source_sha256():
    digest = sha256()
    for path in sorted((REPO / "src/enzywizard_agent").rglob("*")):
        if path.suffix in {".py", ".yaml", ".json", ".txt"} and path.is_file():
            digest.update(str(path.relative_to(REPO)).encode())
            digest.update(sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.repeat < 1 or (args.bootstrap_iterations != 0 and args.bootstrap_iterations < 100):
        parser.error("Use a positive --repeat and --bootstrap_iterations of 0 or at least 100.")
    if min(args.llm_timeout_seconds, args.judge_timeout_seconds) <= 0:
        parser.error("Timeouts must be positive.")
    try:
        catalog, cases, dataset_hash = load_suite()
        selected = select_cases(cases, args)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    if args.validate_only:
        print(f"[validation passed] {len(cases)} definition cases and {len(catalog['reports'])} report fixtures.")
        return 0
    if not args.api_key or not args.judge_api_key:
        parser.error("Set ENZYWIZARD_API_KEY and ANTHROPIC_API_KEY.")
    workdir = args.workdir.expanduser().resolve()
    if workdir.is_relative_to(REPO):
        parser.error("--workdir must be outside the Agent repository.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = workdir / f"run_{stamp}_{uuid4().hex[:8]}"
    probe_dir = run_dir / "_model_capability"
    probe_dir.mkdir(parents=True, exist_ok=False)
    try:
        capability = check_model_capabilities({
            "config": configuration(args, probe_dir, ontology_mode="off"),
            "session_id": f"test4-contract-{uuid4().hex}",
            "current_turn_id": 1,
            "current_user_message": "Model contract capability probe.",
            "runtime_session": RuntimeSession(started_at=utc_now(), started_monotonic=time.monotonic()),
        }).model_dump(mode="json")
    except Exception as error:
        print(f"[model contract failed] {_error(error, args)}", flush=True)
        return 2

    ids = [case["case_id"] for case in selected]
    metadata = {
        "test_id": catalog["test_id"], "model": args.model, "base_url": args.base_url,
        "reasoning_effort": args.reasoning_effort, "llm_timeout_seconds": args.llm_timeout_seconds,
        "ontology_mode": args.ontology_mode, "repeat": args.repeat, "seed": args.seed,
        "profile": args.profile, "selected_case_ids": ids,
        "bootstrap_iterations": args.bootstrap_iterations,
        "judge_model": judge.MODEL, "judge_protocol": judge.PROTOCOL,
        "judge_timeout_seconds": args.judge_timeout_seconds,
        "dataset_sha256": dataset_hash, "knowledge_base_sha256": fair_terms_sha256(),
        "agent_source_sha256": _agent_source_sha256(),
    }
    rows = []

    def save(complete=False):
        score = summarize(rows, ids, args.repeat, iterations=args.bootstrap_iterations if complete else 0, seed=args.seed)
        _write(run_dir / "results.json", {
            "metadata": metadata, "model_capability": capability,
            "summary": {
                "case_count": len(ids), "expected_runs": len(ids) * args.repeat,
                "run_count": len(rows), "completed_all_case_runs": complete,
                "metrics": {"CDA": score},
            },
            "results": rows,
        })
        return score

    print(f"[Test 4] {args.model} | {args.ontology_mode} | cases={len(ids)} | repeat={args.repeat} | {run_dir}", flush=True)
    for repetition in range(1, args.repeat + 1):
        for case in selected:
            print(f"[{case['case_id']}] {repetition}/{args.repeat} {case['target_term']}", flush=True)
            row = run_case(catalog, case, repetition, args, run_dir)
            rows.append(row)
            for message in row["errors"]:
                print(f"  Error: {message}", flush=True)
            print(f"  CDA: {row['score']}", flush=True)
            save()
    write_reviews(run_dir, rows, ids, args.seed)
    fields = ("case_id", "category_id", "repetition", "score", "scoring_source")
    with (run_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    final = save(complete=True)
    print(f"[complete] {run_dir / 'results.json'}", flush=True)
    return 0 if final["complete_reviews"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
