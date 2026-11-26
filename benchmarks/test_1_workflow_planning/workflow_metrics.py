"""Binary milestone matching, case-specific success, and repeated-run statistics."""

from collections import Counter, defaultdict
from functools import cache
from graphlib import TopologicalSorter
import json
from math import isfinite
from pathlib import Path
import random
from statistics import mean, stdev
from typing import Any

import benchmark_support as checks


METRICS = ("Avg Score", "SR")


def tool_key(call: dict[str, Any]) -> str:
    return ":".join(checks.call_key(call))


@cache
def tool_specs():
    return checks.load_enzywizard_tool_specs()


def valid_call_arguments(call: dict) -> bool:
    spec = tool_specs().get(call["tool_name"])
    if spec is None or not isinstance(call.get("arguments"), dict):
        return False
    arguments = call["arguments"]
    required = {item["name"]: item.get("python_type") for item in spec.get("required_arguments", [])}
    optional = {item["name"]: item.get("python_type") for item in spec.get("optional_arguments", [])}
    fields = required | optional
    return required.keys() <= arguments.keys() <= fields.keys() and all(
        checks.value_type_is_valid(value, fields[name]) for name, value in arguments.items()
    )


def case_contracts(case: dict) -> list[dict]:
    """Validate and return the case's declared scoring alternatives."""
    alternatives = case["scoring"]
    if not alternatives:
        raise ValueError(f"{case['case_id']} has no scoring alternatives")
    expected_profiles = [item["profiles"] for item in alternatives]
    for field in ("reference_answers", "expected_trajectories"):
        if [item["workflow"] for item in case[field]] != expected_profiles:
            raise ValueError(f"{case['case_id']}: {field} must list every accepted workflow")
    valid_kinds = {
        "routing", "history", "plan", "plan_call", "revision", "approval",
        "execution_call", "artifact", "terminal",
    }
    for contract in alternatives:
        nodes = contract["milestones"]
        if not nodes:
            raise ValueError(f"{case['case_id']} contains an empty milestone graph")
        for node in nodes.values():
            if node["kind"] not in valid_kinds or not set(node["after"]) <= nodes.keys():
                raise ValueError(f"{case['case_id']} contains an invalid milestone")
        TopologicalSorter({name: node["after"] for name, node in nodes.items()}).prepare()
        if len(contract["minefields"]["forbidden_tools_by_plan_stage"]) != len(contract["profiles"]):
            raise ValueError(f"{case['case_id']} has inconsistent plan-stage restrictions")
        for artifact in contract["success"]["required_artifacts"]:
            if artifact["count"] < 1 or not artifact["producers"]:
                raise ValueError(f"{case['case_id']} has an invalid required artifact")
    return alternatives


def best_mapping(predecessors: dict[str, list[str]], candidates: dict[str, list[int]]) -> dict[str, int]:
    """Maximize the number of distinct matched events under the DAG's strict order."""
    ordered = list(TopologicalSorter(predecessors).static_order())
    ancestors: dict[str, set[str]] = {}
    for name in ordered:
        parents = predecessors[name]
        ancestors[name] = set(parents).union(*(ancestors[parent] for parent in parents))
    active = [name for name in ordered if candidates[name]]
    best: dict[str, int] = {}
    chosen: dict[str, int] = {}
    used: set[int] = set()

    def search(index: int) -> None:
        nonlocal best
        if len(chosen) + len(active) - index <= len(best):
            return
        if index == len(active):
            best = chosen.copy()
            return
        name = active[index]
        after = max((chosen[parent] for parent in ancestors[name] if parent in chosen), default=-1)
        for position in candidates[name]:
            if position > after and position not in used:
                chosen[name] = position
                used.add(position)
                search(index + 1)
                used.remove(position)
                del chosen[name]
        search(index + 1)

    search(0)
    return best


def valid_artifact(path: str, kind: str) -> bool:
    try:
        if not checks.artifact_file_is_valid(path, kind):
            return False
        if kind.endswith("_json") or Path(path).suffix.casefold() == ".json":
            value = json.loads(Path(path).read_text(encoding="utf-8"))
            if kind.endswith("_report_json"):
                expected = "enzywizard_" + kind.removesuffix("_report_json")
                return isinstance(value, dict) and bool(value) and value.get("report_type", expected) == expected
            return isinstance(value, (list, dict))
        return True
    except (ValueError, OSError, UnicodeError):
        return False


def matches_parameters(arguments: dict, expectations: list[dict]) -> bool:
    for rule in expectations:
        actual = arguments.get(rule["parameter"], rule.get("default_if_missing"))
        if not any(
            actual == expected
            and isinstance(actual, bool) == isinstance(expected, bool)
            for expected in rule["accepted_values"]
        ):
            return False
    return True


def matches_reuse(arguments: dict, fields: dict, artifacts: list[dict]) -> bool:
    for field, kind in fields.items():
        actual = arguments.get(field)
        if not isinstance(actual, str) or not actual:
            return False
        paths = [Path(item["path"]) for item in artifacts if item["artifact_type"] == kind]
        targets = [path.parent if field.endswith("_dir") else path for path in paths]
        if Path(actual).resolve() not in [target.resolve() for target in targets]:
            return False
        if kind == "docked_substrate_sdf":
            name = arguments.get("substrate_names")
            if not isinstance(name, str) or not any(
                path.parent.resolve() == Path(actual).resolve() and name == path.stem for path in paths
            ):
                return False
    return True


def _matches_report_statistics(event: dict, rule: dict) -> bool:
    fields = rule.get("required_statistics")
    if not fields:
        return True
    try:
        report = json.loads(Path(event["path"]).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    statistics = report.get("overall_statistics") if isinstance(report, dict) else None
    if not isinstance(statistics, dict):
        return False
    values = [statistics.get(field) for field in fields]
    return all(type(value) is int or (type(value) is float and isfinite(value)) for value in values)


def _matches(node: dict, event: dict, history_artifacts: list[dict]) -> bool:
    kind = node["kind"]
    if kind != event["kind"] or ("stage" in node and node["stage"] != event.get("stage")):
        return False
    if kind == "routing":
        return set(node["actions"]) == set(event["actions"])
    if kind == "history":
        return (
            event["status"] == "succeeded" and event["grounded"]
            and event["turns"] == [node["turn"]]
            and set(node["operations"]).issubset(event["operations"])
            and bool(set(node["accepted_artifact_types"]).intersection(event["artifact_types"]))
        )
    if kind == "plan":
        return bool(event["calls"]) and Counter(event["calls"]) == Counter(node["allowed"])
    if kind == "revision":
        return event["request"] == node["request"] and event["linked"]
    if kind == "approval":
        return event["validated"] and event["linked"]
    if kind in {"plan_call", "execution_call"}:
        arguments = event["arguments"]
        return (
            event["key"] == node["key"] and event["valid"]
            and set(node["dependencies"]).issubset(event["dependencies"])
            and matches_parameters(arguments, node["parameters"])
            and matches_reuse(arguments, node["reuse_inputs"], history_artifacts)
        )
    if kind == "artifact":
        return (
            event["valid"] and event["producer"] in node["producers"]
            and event["artifact_type"] == node["artifact_type"] and event["side"] == node["side"]
            and _matches_report_statistics(event, node)
        )
    return kind == "terminal" and event["status"] == "succeeded" and event["final"]


def _minefields(contract: dict, trace: dict) -> list[dict]:
    rules = contract["minefields"]
    forbidden = rules["forbidden_tools_by_plan_stage"]
    limits = rules["execution_call_limits"]
    reuse = rules["wrong_historical_input"]
    triggers = []
    executed: Counter = Counter()
    for position, event in enumerate(trace["events"]):
        kind = event["kind"]
        reasons = []
        if kind == "execution_call" and event["data"]["command"]:
            key = event["key"]
            executed[key] += 1
            if rules["executed_revoked_plan"] and event["revoked"]:
                reasons.append("executed_revoked_plan")
            if executed[key] > limits.get(key, 0):
                reasons.append("unexpected_execution_call")
            if (
                reuse and key.split(":")[0] == reuse["tool"]
                and not matches_reuse(event["arguments"], reuse["inputs"], trace["history_artifacts"])
            ):
                reasons.append("wrong_historical_input")
        elif kind == "plan_call":
            stage = event["stage"]
            if stage <= len(forbidden) and event["key"].split(":")[0] in forbidden[stage - 1]:
                reasons.append("forbidden_tool_in_plan")
        elif (
            kind == "history" and rules["wrong_historical_source"]
            and event["status"] == "succeeded" and not event["grounded"]
        ):
            reasons.append("wrong_historical_source")
        triggers.extend({"event": position, "reason": reason} for reason in reasons)
    return triggers


def _artifact_requirements(required: list[dict], events: list[dict]) -> bool:
    seen = set()
    for item in required:
        matches = [
            event for event in events
            if event["kind"] == "artifact" and event["valid"] and event["path"] not in seen
            and event["artifact_type"] == item["artifact_type"] and event["side"] == item["side"]
            and event["producer"] in item["producers"]
            and _matches_report_statistics(event, item)
        ]
        paths = list(dict.fromkeys(event["path"] for event in matches))
        if len(paths) < item["count"]:
            return False
        seen.update(paths[:item["count"]])
    return True


def _success(contract: dict, trace: dict) -> bool:
    rules = contract["success"]
    actual = [event for event in trace["events"] if event["kind"] == "execution_call"]
    terminal = [event for event in trace["events"] if event["kind"] == "terminal"]
    if not terminal or terminal[-1]["status"] != "succeeded" or not terminal[-1]["final"] or not actual:
        return False
    if any(event["revoked"] or event["key"].split(":")[0] in rules["forbidden_executions"] for event in actual):
        return False
    exclusive = rules["exclusive_execution_tools"]
    if exclusive and any(event["key"] not in exclusive for event in actual):
        return False
    actual = [event for event in actual if event["final"] and event["succeeded"]]
    if not actual or not all(event["target_inputs_valid"] for event in actual):
        return False
    checks_ok = all(
        any(
            matches_parameters(event["arguments"], item["requirements"])
            for event in actual if event["key"] == item["key"]
        )
        for item in rules["required_parameters"]
    )
    reuse = rules["required_historical_input"]
    reuse_ok = not reuse or any(
        event["key"].split(":")[0] == reuse["tool"]
        and matches_reuse(event["arguments"], reuse["inputs"], trace["history_artifacts"])
        for event in actual
    )
    return checks_ok and reuse_ok and _artifact_requirements(rules["required_artifacts"], trace["events"])


def score_case(contracts: list[dict], trace: dict, *, execute: bool) -> tuple[dict, dict]:
    results = []
    for contract in contracts:
        nodes = {
            name: node for name, node in contract["milestones"].items()
            if execute or node["kind"] not in {"execution_call", "artifact", "terminal"}
        }
        events = trace["events"]
        candidates = {
            name: [
                index for index, event in enumerate(events)
                if _matches(node, event, trace["history_artifacts"])
            ]
            for name, node in nodes.items()
        }
        matched = best_mapping({name: node["after"] for name, node in nodes.items()}, candidates)
        minefields = _minefields(contract, trace)
        base = len(matched) / len(nodes)
        results.append({
            "profiles": contract["profiles"], "score": 0.0 if minefields else base,
            "base_score": base, "matched": matched,
            "unmatched": [name for name in nodes if name not in matched],
            "minefields": minefields, "success": float(_success(contract, trace)) if execute else None,
        })
    best = max(results, key=lambda entry: (entry["score"], not entry["minefields"], entry["base_score"]))
    metrics = (
        {"Avg Score": best["score"], "SR": max(item["success"] for item in results)}
        if execute else {"Plan Score": best["score"]}
    )
    return metrics, {"chosen": best["profiles"], "alternatives": results}


def summarize_metrics(rows: list[dict], *, iterations: int, seed: int) -> dict:
    by_case: dict[str, dict[int, dict]] = defaultdict(dict)
    for row in rows:
        reps = by_case[row["case_id"]]
        if row["repetition"] in reps:
            raise ValueError("Duplicate case and repetition")
        reps[row["repetition"]] = row["metrics"]
    keys = sorted(by_case)
    rep_ids = sorted({rep for repeats in by_case.values() for rep in repeats})
    complete = all(sorted(by_case[key]) == rep_ids for key in keys)
    rng = random.Random(seed)
    samples = [rng.choices(keys, k=len(keys)) for _ in range(iterations)] if keys and complete else []
    metrics = {}
    for name in (METRICS if rows and "SR" in rows[0]["metrics"] else ("Plan Score",)):
        def repeat_means(sampled):
            return {
                str(rep): mean(by_case[key][rep][name] for key in sampled if rep in by_case[key])
                for rep in rep_ids
            }

        by_rep = repeat_means(keys)
        values = list(by_rep.values())
        out = {
            "by_repetition": by_rep,
            "mean": mean(values) if values else None,
            "sd": stdev(values) if len(values) > 1 else (0.0 if values else None),
            "evaluated_case_runs": len(rows),
        }
        if samples:
            distribution = sorted(mean(repeat_means(sample).values()) for sample in samples)
            out["bootstrap_95_ci"] = {
                "lower": distribution[int(0.025 * (len(distribution) - 1))],
                "upper": distribution[int(0.975 * (len(distribution) - 1))],
            }
        metrics[name] = out
    return metrics
