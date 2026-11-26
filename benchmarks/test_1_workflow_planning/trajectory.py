"""Record the observable stages of the existing Test 1 workflow graph."""

from pathlib import Path
from typing import Any

from langgraph.types import Command

from enzywizard_agent.data_models.artifact import Artifact
from enzywizard_agent.data_models.planning import ToolCall
from enzywizard_agent.utils.tool_execution_utils import resolve_tool_call_arguments

import benchmark_support as runner
from workflow_metrics import tool_key, valid_artifact, valid_call_arguments


def _payload(value: Any) -> dict:
    return value.model_dump(mode="json")


def _identity(item: dict) -> tuple[int, int]:
    return item["turn_id"], item["plan_id"]


def _artifact_identity(artifact: dict) -> tuple:
    return (*_identity(artifact), artifact["source_tool"], artifact["side"],
            artifact["artifact_type"], artifact["path"])


def _plan_calls(calls: list[dict]) -> list[dict]:
    indexes = {call["step_index"]: call for call in calls}
    outputs = {
        index: {output["artifact_type"] for output in call["output_artifacts"]}
        for index, call in indexes.items()
    }
    normalized = []
    for position, call in enumerate(calls, 1):
        step = call["step_index"]
        arguments = call["arguments"]
        staging = call.get("staging_rules", [])
        references = runner.referenced_step_indexes(arguments) | runner.referenced_step_indexes(staging)
        reference_ok = all(ref in indexes and ref < step for ref in references)
        actual = {
            "kind": "plan_call", "key": tool_key(call), "arguments": arguments,
            "dependency_steps": sorted(references),
            "dependencies": sorted(tool_key(indexes[i]) for i in references if i in indexes and i < step),
            "valid": step == position and reference_ok
            and runner.reference_is_valid(arguments, step, outputs)
            and valid_call_arguments(call),
        }
        normalized.append(actual)
    return normalized


def _target_inputs_match(call: dict, inputs: dict) -> bool:
    fields = {
        "enzywizard-clean": {"input_path": "structure_file"},
        "enzywizard-mut-clean": {
            "wt_input_path": "structure_file", "mut_input_path": "mut_structure_file",
            "mutation": "amino_acid_substitution",
        },
    }
    arguments = call["resolved_arguments"]
    for argument, field in fields.get(call["tool_name"], {}).items():
        actual = arguments.get(argument)
        expected = inputs.get(field)
        if not isinstance(actual, str) or not isinstance(expected, str):
            return False
        if field.endswith("_file"):
            if Path(actual).resolve() != Path(expected).resolve():
                return False
        elif actual.upper() != expected.upper():
            return False
    return True


def _executed_events(execution: dict, calls: list[dict], inputs: dict, *, final: bool, revoked: bool):
    plan_by_step = {call["step_index"]: (call, facts) for call, facts in zip(calls, _plan_calls(calls))}
    available: list[Artifact] = []
    completed: dict[int, str] = {}
    seen_paths: set[Path] = set()
    for call in execution["call_executions"]:
        step = call["step_index"]
        key = tool_key(call)
        passed = call["execution_status"] == "succeeded"
        planned, facts = plan_by_step.get(step, ({}, {}))
        arguments = call["resolved_arguments"]
        reference_steps = facts.get("dependency_steps", [])
        dependencies = sorted(completed[ref] for ref in reference_steps if ref in completed)
        resolved = False
        if final and passed and planned and tool_key(planned) == key:
            try:
                expected = resolve_tool_call_arguments(ToolCall.model_validate(planned), available)
                # Check delayed bindings against real upstream artifacts. Runtime
                # recovery can legitimately repair other managed inputs.
                resolved = all(
                    arguments.get(name) == expected[name]
                    for name, value in planned["arguments"].items()
                    if isinstance(value, dict) and "source_step_index" in value
                )
            except (OSError, ValueError, KeyError):
                resolved = False
        target_ok = _target_inputs_match(call, inputs)
        yield {
            "kind": "execution_call", "key": key, "step_index": step, "data": call,
            "arguments": arguments, "dependencies": dependencies,
            "valid": resolved and target_ok and len(dependencies) == len(reference_steps),
            "final": final, "revoked": revoked, "succeeded": passed, "target_inputs_valid": target_ok,
        }
        if passed:
            completed[step] = key
        for artifact in call["artifacts"]:
            path = Path(artifact["path"]).resolve()
            valid = (
                final and passed and path not in seen_paths
                and _identity(artifact) == _identity(call)
                and artifact["source_tool"] == call["tool_name"]
                and artifact["metadata"].get("step_index", step) == step
                and valid_artifact(str(path), artifact["artifact_type"])
            )
            if valid:
                seen_paths.add(path)
            yield {
                "kind": "artifact", **artifact, "path": str(path), "producer": key,
                "valid": valid, "step_index": step,
            }
            if passed:
                available.append(Artifact.model_validate(artifact))
    yield {"kind": "terminal", "status": execution["execution_status"], "final": final}


def run_trajectory(catalog: dict, case: dict, args: Any, case_dir: Path) -> tuple[dict, dict, list[str]]:
    state = runner.build_state(catalog, case, args, case_dir)
    graph = runner.build_test_graph()
    graph_config = {"configurable": {"thread_id": state["session_id"]}}
    seeded = [_payload(item) for item in state["artifacts"]]
    seed_ids = {_artifact_identity(item) for item in seeded}
    trace = {
        "initial": {
            "request": state["current_user_message"],
            "agent_inputs": _payload(state["agent_inputs"]),
            "messages": [_payload(item) for item in state["user_messages"]],
            "prior_plans": [_payload(item) for item in state["planning_results"]],
            "prior_confirmations": [_payload(item) for item in state["user_confirmations"]],
            "prior_executions": [_payload(item) for item in state["tool_executions"]],
        },
        "history_artifacts": seeded,
        "events": [],
        "plans": [],
        "executions": [],
    }
    events = trace["events"]
    errors: list[str] = []
    counts = {name: len(state.get(name, [])) for name in (
        "request_routing_results", "history_query_results", "planning_results", "user_confirmations"
    )}
    confirmations = []
    stage_count = 2 if case["category_id"] == "plan_revision" else 1

    def new(current, key):
        items = current.get(key, [])
        start = counts[key]
        counts[key] = len(items)
        return [_payload(item) for item in items[start:]]

    def capture(current, stage, snapshot=None):
        for confirmation in new(current, "user_confirmations"):
            event = {
                "kind": "confirmation", "data": confirmation,
                "request": confirmation.get("revision_request"),
            }
            confirmations.append(event)
            events.append(event)
        for routing in new(current, "request_routing_results"):
            events.append({
                "kind": "routing", "stage": stage, "data": routing,
                "actions": [item["action_type"] for item in (routing.get("output") or {}).get("actions", [])],
            })
        for history in new(current, "history_query_results"):
            selected = history["selected_artifacts"]
            events.append({
                "kind": "history", "stage": stage, "data": history,
                "status": history["query_status"], "turns": history["resolved_turn_ids"],
                "operations": (history.get("query") or {}).get("operations", []),
                "artifact_types": sorted({item["artifact_type"] for item in selected}),
                "grounded": bool(selected) and all(_artifact_identity(item) in seed_ids for item in selected),
            })
        plans = new(current, "planning_results")
        if plans:
            events.extend({"kind": "planning_state", "stage": stage, "data": item} for item in plans)
        if snapshot is None:
            snapshot = next((plan["tool_plan"] for plan in reversed(plans) if plan.get("tool_plan")), None)
        if snapshot is not None:
            snapshot = runner._snapshot(snapshot)
            validation = next((
                item.get("plan_validation") for item in reversed(plans)
                if item.get("tool_plan") and _identity(item["tool_plan"]) == _identity(snapshot)
            ), None)
            trace["plans"].append({
                **snapshot, "stage": stage, "validated": bool(
                    validation and validation["passed"] and validation["checks"]
                ),
            })
            calls = snapshot["calls"]
            events.append({"kind": "plan", "stage": stage, "calls": [tool_key(call) for call in calls]})
            events.extend({**call, "stage": stage} for call in _plan_calls(calls))

    result = state
    stage = 1
    try:
        result = graph.invoke(state, config=graph_config)
        for stage in range(1, stage_count + 1):
            if not result.get("__interrupt__"):
                capture(result, stage)
                errors.append(f"Missing confirmation interrupt for plan stage {stage}; route={result.get('route')!r}")
                break
            snapshot = runner.get_single_interrupt_payload(result)
            capture(result, stage, snapshot)
            if args.show_plans:
                print(f"    plan {snapshot['plan_id']}: " + " -> ".join(map(tool_key, snapshot["calls"])), flush=True)
            revise = stage < stage_count
            decision = {
                "turn_id": snapshot["turn_id"], "plan_id": snapshot["plan_id"],
                "decision": "revision_requested" if revise else "approved",
                "revision_request": case.get("revision_request") if revise else None,
            }
            if not revise and not args.execute:
                result = runner.record_user_confirmation(result, decision)
                capture(result, stage)
                break
            result = graph.invoke(Command(resume=decision), config=graph_config)
            if not revise:
                capture(result, stage)
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
        try:
            checkpoint = graph.get_state(graph_config)
            if checkpoint.values:
                result = checkpoint.values
                capture(result, min(stage + 1, stage_count))
        except Exception as checkpoint_error:
            errors.append(f"Could not read checkpoint: {checkpoint_error}")

    plans = {_identity(plan): plan for plan in trace["plans"]}
    revoked = set()
    for event in confirmations:
        confirmation = event["data"]
        ref = _identity(confirmation)
        plan = plans.get(ref)
        if confirmation["decision"] == "revision_requested":
            revoked.add(ref)
            event["kind"] = "revision"
            event["linked"] = bool(plan and any(
                draft["stage"] == plan["stage"] + 1 and draft["turn_id"] == ref[0]
                and draft["plan_id"] > ref[1] and draft["calls"] != plan["calls"]
                for draft in plans.values()
            ))
        elif confirmation["decision"] == "approved":
            event["kind"] = "approval"
            event["linked"] = bool(plan and plan["stage"] == stage_count)
            event["validated"] = bool(plan and plan["validated"])
        event["stage"] = plan["stage"] if plan else None

    trace["final_route"] = result.get("route")
    active_turn = result.get("current_turn_id", state["current_turn_id"])
    executions = {}
    for execution in [*result.get("tool_executions", []), result.get("current_tool_execution")]:
        if execution is not None and execution.turn_id >= state["current_turn_id"]:
            data = _payload(execution)
            executions[_identity(data)] = data
    for ref, execution in executions.items():
        plan = plans.get(ref)
        final = bool(plan and plan["stage"] == stage_count and ref[0] == active_turn and ref not in revoked)
        trace["executions"].append(execution)
        events.extend(_executed_events(
            execution, plan["calls"] if plan else [], trace["initial"]["agent_inputs"],
            final=final, revoked=ref in revoked,
        ))
    if not executions:
        events.append({"kind": "terminal", "status": "not_executed", "final": False})
    return trace, result, errors
