"""Load the fixed definition cases and validate their local fixtures."""

from hashlib import sha256
import json
from pathlib import Path
import random

import yaml

from enzywizard_agent.services.knowledge_retrieval_service import load_fair_terms

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def _yaml(path):
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def definition_question(term):
    return f"What is the definition of ‘{term}’?"


def load_suite():
    catalog = _yaml(ROOT / "case_catalog.yaml")
    cases_dir = ROOT / "cases"
    index = _yaml(cases_dir / "index.yaml")
    if catalog["schema_version"] != 1 or index["schema_version"] != 1:
        raise ValueError("Unsupported case schema.")
    terms = {entry["term"]: entry for entry in load_fair_terms()}
    cases = []
    digest = sha256()
    for name in index["case_files"]:
        fixture = _yaml(cases_dir / name)
        if fixture["schema_version"] != 1 or len(fixture["cases"]) != fixture["case_count"]:
            raise ValueError(f"Invalid case count or schema in {name}")
        for case in fixture["cases"]:
            term = terms.get(case["target_term"])
            if (
                not term
                or not term.get("description", "").strip()
                or case["reference_definition"] != term["description"]
                or case["definition_question"] != definition_question(case["target_term"])
            ):
                raise ValueError(f"{case['case_id']}: invalid term, definition, or fixed question")
            original_keys = {"initial_question", "follow_up_question"} if "initial_question" in case else {"question"}
            if any(not isinstance(case.get(k), str) or not case[k].strip() for k in original_keys):
                raise ValueError(f"{case['case_id']}: missing original task")
            if not case["report_ids"] or any(k not in catalog["reports"] for k in case["report_ids"]):
                raise ValueError(f"{case['case_id']}: unknown report")
            cases.append({**case, "category_id": fixture["category_id"], "term_uri": term["uri"]})
    if len(cases) != catalog["case_count"] or len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("Unexpected suite size or duplicate case IDs.")
    digest.update(json.dumps([catalog, cases], sort_keys=True).encode())
    resources = (ROOT / "resources").resolve()
    for report in catalog["reports"].values():
        path = (ROOT / report["path"]).resolve()
        if not path.is_relative_to(resources) or not path.is_file():
            raise ValueError(f"Missing local report: {path}")
        digest.update(sha256(path.read_bytes()).digest())
    return catalog, cases, digest.hexdigest()


def select_cases(cases, args):
    if args.case_ids:
        by_id = {case["case_id"]: case for case in cases}
        if len(set(args.case_ids)) != len(args.case_ids):
            raise ValueError("Duplicate --case_ids.")
        if unknown := set(args.case_ids) - by_id.keys():
            raise ValueError(f"Unknown case IDs: {', '.join(sorted(unknown))}")
        return [by_id[case_id] for case_id in args.case_ids]
    if args.profile == "full":
        return cases
    if not 1 <= args.quick_case_count <= len(cases):
        raise ValueError("Invalid --quick_case_count.")
    groups = {}
    for case in cases:
        groups.setdefault(case["category_id"], []).append(case)
    rng = random.Random(args.seed)
    for group in groups.values():
        rng.shuffle(group)
    selected = [
        group[offset] for offset in range(max(map(len, groups.values())))
        for group in groups.values() if offset < len(group)
    ]
    return selected[:args.quick_case_count]
