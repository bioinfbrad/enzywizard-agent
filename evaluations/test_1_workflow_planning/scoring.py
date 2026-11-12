from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import random
from statistics import mean, stdev
from typing import Any, Iterable

from enzywizard_agent.utils.tool_plan_utils import (
    PARAMETER_PROFILE_VALUES,
    load_enzywizard_tool_specs,
)


METRIC_NAMES = ("TSA", "PCVR", "WOVR", "RAC", "E2E_SR", "PPCR")
PROVENANCE_REQUIREMENTS = {
    "request_to_plan_link",
    "tool_selection_reason",
    "parameter_value_and_source",
    "order_and_dependency_record",
    "historical_reuse_origin",
    "plan_revision_trace",
    "final_validation_and_approval",
}


def call_key(call: dict[str, Any]) -> tuple[str, str]:
    return str(call.get("tool_name") or call.get("tool")), str(call.get("side"))


def expected_call_counter(profile: dict[str, Any]) -> Counter[tuple[str, str]]:
    return Counter(call_key(item) for item in profile.get("expected_calls", []))


def actual_call_counter(calls: list[dict[str, Any]]) -> Counter[tuple[str, str]]:
    return Counter(call_key(item) for item in calls)


def score_tsa(
    calls: list[dict[str, Any]], alternatives: list[dict[str, Any]]
) -> tuple[float, int | None]:
    actual = actual_call_counter(calls)
    for index, profile in enumerate(alternatives):
        forbidden = set(profile.get("forbidden_tools", []))
        if actual == expected_call_counter(profile) and not {
            tool for tool, _ in actual
        }.intersection(forbidden):
            return 1.0, index
    return 0.0, None


def _value_type_is_valid(value: Any, type_name: str | None) -> bool:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="python")
    if isinstance(value, dict) and "source_step_index" in value:
        return True
    if type_name == "bool":
        return isinstance(value, bool)
    if type_name == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "float":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if type_name == "str":
        return isinstance(value, str) and bool(value.strip())
    if type_name in {"list", "list[str]"}:
        return isinstance(value, list)
    return value is not None


def _parameter_expectations_for_call(
    expectations: list[dict[str, Any]], key: tuple[str, str]
) -> list[dict[str, Any]]:
    tool, side = key
    return [
        item
        for item in expectations
        if item.get("tool") == tool
        and (item.get("side") is None or item.get("side") == side)
    ]


def _call_parameters_correct(
    call: dict[str, Any], expectations: list[dict[str, Any]]
) -> bool:
    specs = load_enzywizard_tool_specs()
    tool_name, side = call_key(call)
    spec = specs.get(tool_name)
    arguments = call.get("arguments")
    if spec is None or not isinstance(arguments, dict):
        return False
    required = {
        str(item["name"]): item for item in spec.get("required_arguments", [])
    }
    optional = {
        str(item["name"]): item for item in spec.get("optional_arguments", [])
    }
    allowed = set(required) | set(optional)
    if not set(required).issubset(arguments) or not set(arguments).issubset(allowed):
        return False
    for name, value in arguments.items():
        field = required.get(name) or optional.get(name) or {}
        if not _value_type_is_valid(value, field.get("python_type")):
            return False
    for expectation in _parameter_expectations_for_call(
        expectations, (tool_name, side)
    ):
        if arguments.get(expectation["parameter"]) not in expectation[
            "accepted_values"
        ]:
            return False
    return True


def score_pcvr(
    calls: list[dict[str, Any]],
    profile: dict[str, Any],
    parameter_expectations: list[dict[str, Any]],
) -> tuple[float, list[dict[str, Any]]]:
    """Macro-ready per-case PCVR; missing expected calls count as incorrect."""

    actual_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for call in calls:
        actual_by_key.setdefault(call_key(call), []).append(call)
    checks: list[dict[str, Any]] = []
    for expected in profile.get("expected_calls", []):
        key = call_key(expected)
        matches = actual_by_key.get(key, [])
        correct = len(matches) == 1 and _call_parameters_correct(
            matches[0], parameter_expectations
        )
        checks.append({"tool": key[0], "side": key[1], "correct": correct})
    return (
        mean(float(item["correct"]) for item in checks) if checks else 1.0,
        checks,
    )


def _reference_is_valid(
    value: Any, current_index: int, outputs_by_step: dict[int, set[str]]
) -> bool:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="python")
    if isinstance(value, dict):
        if "source_step_index" in value:
            source = value.get("source_step_index")
            artifact_type = value.get("artifact_type")
            return (
                isinstance(source, int)
                and source < current_index
                and artifact_type in outputs_by_step.get(source, set())
            )
        return all(
            _reference_is_valid(nested, current_index, outputs_by_step)
            for nested in value.values()
        )
    if isinstance(value, (list, tuple)):
        return all(
            _reference_is_valid(nested, current_index, outputs_by_step)
            for nested in value
        )
    return True


def _referenced_step_indexes(value: Any) -> set[int]:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="python")
    if isinstance(value, dict):
        output: set[int] = set()
        if isinstance(value.get("source_step_index"), int):
            output.add(int(value["source_step_index"]))
        for nested in value.values():
            output.update(_referenced_step_indexes(nested))
        return output
    if isinstance(value, (list, tuple)):
        output: set[int] = set()
        for nested in value:
            output.update(_referenced_step_indexes(nested))
        return output
    return set()


def score_wovr(
    calls: list[dict[str, Any]], profile: dict[str, Any]
) -> tuple[float, dict[str, bool]]:
    indexes = [int(item.get("step_index", 0)) for item in calls]
    consecutive = indexes == list(range(1, len(calls) + 1))
    calls_by_id = {
        f"{call_key(item)[0]}:{call_key(item)[1]}": item for item in calls
    }
    positions = {identifier: index for index, identifier in enumerate(calls_by_id)}
    order_valid = all(
        before in positions
        and after in positions
        and positions[before] < positions[after]
        for before, after in profile.get("order_constraints", [])
    )
    outputs_by_step = {
        int(item.get("step_index", 0)): {
            str(output.get("artifact_type"))
            for output in item.get("output_artifacts", [])
        }
        for item in calls
    }
    references_valid = all(
        _reference_is_valid(value, int(call.get("step_index", 0)), outputs_by_step)
        for call in calls
        for value in (call.get("arguments") or {}).values()
    )
    staging_valid = all(
        rule.get("source_step_index") is None
        or (
            isinstance(rule.get("source_step_index"), int)
            and rule["source_step_index"] < int(call.get("step_index", 0))
        )
        for call in calls
        for rule in call.get("staging_rules", [])
    )
    dependency_links = all(
        int(calls_by_id[before]["step_index"])
        in (
            _referenced_step_indexes(calls_by_id[after].get("arguments") or {})
            | _referenced_step_indexes(calls_by_id[after].get("staging_rules") or [])
        )
        for before, after in profile.get("order_constraints", [])
        if before in calls_by_id and after in calls_by_id
    ) and all(
        before in calls_by_id and after in calls_by_id
        for before, after in profile.get("order_constraints", [])
    )
    detail = {
        "consecutive_steps": consecutive,
        "order_constraints": order_valid,
        "references": references_valid,
        "staging_dependencies": staging_valid,
        "dependency_links": dependency_links,
    }
    return float(all(detail.values())), detail


def flatten_required_artifacts(
    profile: dict[str, Any], artifact_groups: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for group_name in profile.get("required_artifact_groups", []):
        output.extend(artifact_groups[group_name])
    return output


def artifact_file_is_valid(path_value: str, artifact_type: str) -> bool:
    path = Path(path_value)
    if not path.is_file() or path.stat().st_size == 0:
        return False
    suffix = path.suffix.casefold()
    try:
        if artifact_type.endswith("_json") or suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
        elif suffix in {".fasta", ".fa", ".faa"}:
            return path.read_text(encoding="utf-8", errors="ignore").lstrip().startswith(">")
        elif suffix == ".sdf":
            return "$$$$" in path.read_text(encoding="utf-8", errors="ignore")
        elif suffix in {".cif", ".mmcif"}:
            return "data_" in path.read_text(encoding="utf-8", errors="ignore")[:4096]
        elif suffix == ".pdb":
            text = path.read_text(encoding="utf-8", errors="ignore")
            return "ATOM" in text or "HETATM" in text
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return True


def score_rac(
    execution: dict[str, Any] | None,
    required_artifacts: list[dict[str, Any]],
) -> tuple[float, list[dict[str, Any]]]:
    actual = [
        artifact
        for call in (execution or {}).get("call_executions", [])
        for artifact in call.get("artifacts", [])
        if artifact.get("source_tool") == call.get("tool_name")
    ]
    checks: list[dict[str, Any]] = []
    consumed: set[int] = set()
    for expected in required_artifacts:
        for _ in range(int(expected.get("count", 1))):
            match_index = next(
                (
                    index
                    for index, artifact in enumerate(actual)
                    if index not in consumed
                    and artifact.get("artifact_type") == expected["artifact_type"]
                    and artifact.get("side") == expected["side"]
                    and artifact_file_is_valid(
                        str(artifact.get("path", "")), expected["artifact_type"]
                    )
                ),
                None,
            )
            valid = match_index is not None
            if match_index is not None:
                consumed.add(match_index)
            checks.append({**expected, "valid": valid})
    return mean(float(item["valid"]) for item in checks) if checks else 1.0, checks


def score_e2e(
    tsa: float,
    pcvr: float,
    wovr: float,
    execution: dict[str, Any] | None,
    rac: float,
) -> float:
    execution_succeeded = bool(
        execution and execution.get("execution_status") == "succeeded"
    )
    return float(
        tsa == 1.0
        and pcvr == 1.0
        and wovr == 1.0
        and execution_succeeded
        and rac == 1.0
    )


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _plain_value(value: Any) -> Any:
    return value.model_dump(mode="python") if hasattr(value, "model_dump") else value


def _value_matches_recorded_source(
    value: Any,
    *,
    parameter_name: str,
    call: Any,
    state: dict[str, Any],
    history_result: Any | None,
) -> bool:
    dumped = _plain_value(value)
    if isinstance(dumped, dict) and "source_step_index" in dumped:
        return True

    agent_inputs = state.get("agent_inputs")
    input_values = (
        agent_inputs.model_dump(mode="python")
        if hasattr(agent_inputs, "model_dump")
        else dict(agent_inputs or {})
    )
    recorded_inputs = [item for item in input_values.values() if item is not None]
    if dumped in recorded_inputs:
        return True
    if (
        isinstance(dumped, list)
        and len(dumped) == 1
        and dumped[0] in recorded_inputs
    ):
        return True

    selected = getattr(history_result, "selected_artifacts", []) if history_result else []
    selected_paths = [Path(artifact.path) for artifact in selected]
    if dumped in [str(path) for path in selected_paths]:
        return True
    if isinstance(dumped, str):
        if parameter_name.endswith("_dir") and dumped in {
            str(path.parent) for path in selected_paths
        }:
            return True
        if parameter_name in {"substrate_names", "resolved_substrate_names"} and (
            dumped in {path.stem for path in selected_paths}
        ):
            return True

    execution_dir = Path(call.tool_execution_dir).resolve()
    if isinstance(dumped, str):
        try:
            candidate = Path(dumped).resolve()
            if candidate == execution_dir or execution_dir in candidate.parents:
                return True
            upstream_execution_dirs = [
                Path(upstream.tool_execution_dir).resolve()
                for result in state.get("planning_results", [])
                if result.tool_plan is not None
                and any(
                    planned.step_index == call.step_index
                    and planned.tool_execution_dir == call.tool_execution_dir
                    for planned in result.tool_plan.calls
                )
                for upstream in result.tool_plan.calls
                if upstream.step_index < call.step_index
            ]
            if any(
                candidate == upstream_dir or upstream_dir in candidate.parents
                for upstream_dir in upstream_execution_dirs
            ):
                return True
        except (OSError, RuntimeError, ValueError):
            pass
    return False


def score_ppcr_item(
    item: str,
    *,
    state: dict[str, Any],
    planning_results: list[Any],
    confirmations: list[Any],
    history_result: Any | None,
    revision_request: str | None,
) -> tuple[bool, str]:
    plans = [result.tool_plan for result in planning_results if result.tool_plan is not None]
    if item == "request_to_plan_link":
        turn = int(state.get("current_turn_id", 0))
        request = state.get("current_user_message")
        ok = (
            isinstance(request, str)
            and bool(request.strip())
            and bool(plans)
            and all(plan.turn_id == turn for plan in plans)
        )
        return ok, "plan turn IDs resolve to the active request turn"
    if item == "tool_selection_reason":
        ok = bool(planning_results) and all(
            result.tool_selection is not None
            and _nonempty(result.tool_selection.selection_reason)
            and result.tool_plan is not None
            and all(_nonempty(call.call_reason) for call in result.tool_plan.calls)
            for result in planning_results
        )
        return ok, "selection and per-call reasons are recorded"
    if item == "parameter_value_and_source":
        specs = load_enzywizard_tool_specs()
        ok = bool(planning_results)
        unresolved: list[str] = []
        current_turn = int(state.get("current_turn_id", 0))
        prior_calls = [
            call
            for prior_result in state.get("planning_results", [])
            if prior_result.tool_plan is not None
            and prior_result.tool_plan.turn_id < current_turn
            for call in prior_result.tool_plan.calls
        ]
        for result in planning_results:
            if result.tool_plan is None or result.parameter_extraction is None:
                ok = False
                break
            adjustments = {
                (entry.step_index, entry.parameter_name): entry
                for entry in result.parameter_extraction.parameter_adjustments
            }
            profiled_values: dict[tuple[int, str], Any] = {}
            for profile_request in result.parameter_extraction.profile_requests:
                for step_index in profile_request.step_indexes:
                    profile_call = next(
                        (
                            candidate
                            for candidate in result.tool_plan.calls
                            if candidate.step_index == step_index
                        ),
                        None,
                    )
                    if profile_call is None:
                        continue
                    for parameter_name, parameter_value in (
                        PARAMETER_PROFILE_VALUES.get(profile_call.tool_name, {})
                        .get(profile_request.profile, {})
                        .items()
                    ):
                        profiled_values[(step_index, parameter_name)] = parameter_value
            for call in result.tool_plan.calls:
                spec = specs.get(call.tool_name, {})
                defaults = {
                    entry["name"]: entry.get("default")
                    for entry in spec.get("optional_arguments", [])
                    if "default" in entry
                }
                for name, value in call.arguments.items():
                    dumped = _plain_value(value)
                    inherited = any(
                        prior.tool_name == call.tool_name
                        and prior.side == call.side
                        and name in prior.arguments
                        and (
                            prior.arguments[name].model_dump(mode="python")
                            if hasattr(prior.arguments[name], "model_dump")
                            else prior.arguments[name]
                        )
                        == dumped
                        for prior in prior_calls
                    )
                    source_resolves = (
                        (call.step_index, name) in adjustments
                        or (
                            (call.step_index, name) in profiled_values
                            and dumped == profiled_values[(call.step_index, name)]
                        )
                        or (name in defaults and dumped == defaults[name])
                        or inherited
                        or _value_matches_recorded_source(
                            value,
                            parameter_name=name,
                            call=call,
                            state=state,
                            history_result=history_result,
                        )
                    )
                    ok = ok and source_resolves
                    if not source_resolves:
                        unresolved.append(
                            f"{call.tool_name}:{call.side}.step{call.step_index}.{name}"
                        )
        detail = (
            "each argument resolves to input, default, profile, adjustment, "
            "historical reuse, or upstream output"
        )
        if unresolved:
            detail += "; unresolved=" + ", ".join(sorted(set(unresolved)))
        return bool(ok), detail
    if item == "order_and_dependency_record":
        ok = bool(plans) and all(
            [call.step_index for call in plan.calls] == list(range(1, len(plan.calls) + 1))
            for plan in plans
        )
        return ok, "ordered calls and structured upstream references are retained"
    if item == "historical_reuse_origin":
        selected = getattr(history_result, "selected_artifacts", []) if history_result else []
        current_turn = int(state.get("current_turn_id", 0))
        historical_calls = [
            call
            for execution in state.get("tool_executions", [])
            for call in execution.call_executions
            if call.turn_id < current_turn
        ]
        artifacts_resolve = not selected or all(
            artifact.turn_id >= 1
            and artifact.plan_id >= 1
            and _nonempty(artifact.source_tool)
            and Path(artifact.path).is_file()
            and any(
                call.turn_id == artifact.turn_id
                and call.plan_id == artifact.plan_id
                and call.tool_name == artifact.source_tool
                and any(
                    produced.path == artifact.path
                    and produced.artifact_type == artifact.artifact_type
                    and produced.side == artifact.side
                    for produced in call.artifacts
                )
                for call in historical_calls
            )
            for artifact in selected
        )
        all_plans = [
            result.tool_plan
            for result in state.get("planning_results", [])
            if result.tool_plan is not None
        ]
        prior_plans = [plan for plan in all_plans if plan.turn_id < current_turn]
        current_plans = [plan for plan in all_plans if plan.turn_id == current_turn]
        plan_resolves = not prior_plans or bool(current_plans)
        ok = (
            bool(selected or prior_plans)
            and artifacts_resolve
            and plan_resolves
        )
        return ok, "historical artifacts or plans resolve to their recorded origins"
    if item == "plan_revision_trace":
        signatures = [
            [(call.tool_name, call.side, call.arguments) for call in plan.calls]
            for plan in plans
        ]
        revision_confirmation = next(
            (
                confirmation
                for confirmation in confirmations
                if confirmation.decision == "revision_requested"
                and confirmation.revision_request == revision_request
            ),
            None,
        )
        ok = (
            len(plans) >= 2
            and signatures[-1] != signatures[-2]
            and _nonempty(revision_request)
            and _nonempty(plans[-1].plan_reason)
            and revision_confirmation is not None
            and revision_confirmation.turn_id == plans[-2].turn_id
            and revision_confirmation.plan_id == plans[-2].plan_id
            and plans[-1].turn_id == plans[-2].turn_id
            and plans[-1].plan_id > plans[-2].plan_id
        )
        return ok, "original plan, revision request, changed plan, and reason are linked"
    if item == "final_validation_and_approval":
        validations_resolve = bool(planning_results) and all(
            result.plan_validation is not None
            and result.plan_validation.passed
            and bool(result.plan_validation.checks)
            for result in planning_results
        )
        final_plan = plans[-1] if plans else None
        approval = confirmations[-1] if confirmations else None
        current_execution = state.get("current_tool_execution")
        ok = bool(
            validations_resolve
            and final_plan
            and approval
            and approval.decision == "approved"
            and approval.turn_id == final_plan.turn_id
            and approval.plan_id == final_plan.plan_id
            and (
                current_execution is None
                or (
                    current_execution.turn_id == final_plan.turn_id
                    and current_execution.plan_id == final_plan.plan_id
                )
            )
        )
        return ok, "validation and final approval resolve to the final plan and execution"
    return False, f"unknown provenance requirement: {item}"


def score_ppcr(
    requirements: list[str],
    *,
    state: dict[str, Any],
    planning_results: list[Any],
    confirmations: list[Any],
    history_result: Any | None,
    revision_request: str | None,
) -> tuple[float, list[dict[str, Any]]]:
    checks = []
    for item in requirements:
        complete, detail = score_ppcr_item(
            item,
            state=state,
            planning_results=planning_results,
            confirmations=confirmations,
            history_result=history_result,
            revision_request=revision_request,
        )
        checks.append({"item": item, "complete": complete, "detail": detail})
    return mean(float(item["complete"]) for item in checks) if checks else 1.0, checks


def repetition_statistics(results: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for metric in METRIC_NAMES:
        by_repeat: dict[int, list[float]] = {}
        for result in results:
            value = result.get("metrics", {}).get(metric)
            if isinstance(value, (int, float)):
                by_repeat.setdefault(int(result["repetition"]), []).append(float(value))
        repeat_values = {
            str(key): mean(values) for key, values in sorted(by_repeat.items())
        }
        values = list(repeat_values.values())
        output[metric] = {
            "by_repetition": repeat_values,
            "mean": mean(values) if values else None,
            "sd": stdev(values) if len(values) > 1 else (0.0 if values else None),
            "evaluated_case_runs": sum(len(items) for items in by_repeat.values()),
        }
    return output


def cluster_bootstrap_ci(
    results: list[dict[str, Any]], *, iterations: int, seed: int
) -> dict[str, dict[str, float | None]]:
    """Resample case IDs while retaining all repetitions within each case."""

    by_case: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        by_case.setdefault(str(result["case_id"]), []).append(result)
    case_ids = sorted(by_case)
    rng = random.Random(seed)
    output: dict[str, dict[str, float | None]] = {}
    for metric in METRIC_NAMES:
        eligible = [
            case_id
            for case_id in case_ids
            if any(
                isinstance(item.get("metrics", {}).get(metric), (int, float))
                for item in by_case[case_id]
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
            ]
            if values:
                estimates.append(mean(values))
        estimates.sort()
        if not estimates:
            output[metric] = {"lower": None, "upper": None}
            continue
        lower = estimates[int(0.025 * (len(estimates) - 1))]
        upper = estimates[int(0.975 * (len(estimates) - 1))]
        output[metric] = {"lower": lower, "upper": upper}
    return output
