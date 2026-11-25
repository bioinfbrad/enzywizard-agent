from __future__ import annotations

import json
import os
import re
import sys
from functools import lru_cache
from typing import Any


_MESSAGE_TYPE_LABELS = {
    "scientific_result": "Scientific result",
    "general_scientific_question": "General scientific answer",
    "planning_failure": "Workflow planning failure",
    "plan_rejected": "Plan rejected",
    "tool_execution_failure": "Tool execution failure",
    "artifact_query": "Artifact query",
    "software_help": "Software help",
    "conversation": "Conversation",
    "out_of_scope": "Out-of-scope response",
    "clarification": "Clarification request",
    "input_error": "Required input error",
    "request_routing_failure": "Request routing failure",
    "history_query_failure": "History query failure",
    "runtime_limit": "Runtime limit reached",
    "exit": "Session exit",
}

_BODY_SECTION_LABELS = {
    "scientific_result": "Scientific analysis",
    "general_scientific_question": "Scientific answer",
    "planning_failure": "Planning error report",
    "plan_rejected": "Plan decision",
    "tool_execution_failure": "Execution failure report",
    "artifact_query": "Artifacts",
    "software_help": "Software help",
    "conversation": "Response",
    "out_of_scope": "Scope response",
    "clarification": "Clarification required",
    "input_error": "Input error report",
    "request_routing_failure": "Routing error report",
    "history_query_failure": "History error report",
    "runtime_limit": "Runtime limit report",
    "exit": "Session status",
}

_DISPLAY_SECTION_LABELS = {
    "Evidence supporting the analysis:": "Evidence supporting the analysis",
    "Evidence for cited conclusions:": "Evidence for cited conclusions",
    "Terminology sources (definitions only, not computational-result evidence):": (
        "Terminology sources"
    ),
    "Terminology sources:": "Terminology sources",
    "Artifacts:": "Artifacts",
    "Software help:": "Software help",
    "Scientific answer:": "Scientific answer",
    "Scope response:": "Scope response",
    "Additional response:": "Additional response",
    "Session limit:": "Session limit",
}

_INLINE_EVIDENCE_PREFIXES = (
    "[Evidence:",
    "[Execution evidence:",
    "[Planning evidence:",
    "[Input evidence:",
)
_FIELD_PATH_PATTERN = re.compile(
    r"\bfield\s+(\$(?:\.[A-Za-z0-9_.$*\[\]-]+)?)"
)

_FIELD_PATH_LABELS = {
    "$": "Report root",
    "$.residue_mapping_old_to_new": (
        "Residue mapping (original residue to cleaned residue)"
    ),
    "$.clean_statistics.removed_heterogen_count": (
        "Clean statistics > Removed heterogen count"
    ),
    "$.clean_statistics.standardized_residue_name_count": (
        "Clean statistics > Standardized residue-name count"
    ),
    "$.clean_statistics.repaired_residue_count": (
        "Clean statistics > Repaired residue count"
    ),
    "$.clean_statistics.added_heavy_atom_count": (
        "Clean statistics > Added heavy-atom count"
    ),
    "$.clean_statistics.added_hydrogen_atom_count": (
        "Clean statistics > Added hydrogen-atom count"
    ),
    "$.clean_statistics.retained_residue_count": (
        "Clean statistics > Retained residue count"
    ),
}

_ANSI_RESET = "\033[0m"
_ANSI_BOLD = "\033[1m"
_ANSI_CYAN = "\033[36m"
_ANSI_GREEN = "\033[32m"
_ANSI_YELLOW = "\033[33m"
_ANSI_RED = "\033[31m"

_SIDE_DISPLAY_LABELS = {
    "wt": "Wild type",
    "mut": "Mutant",
    "shared": "Paired analysis",
}

_ARTIFACT_DISPLAY_LABELS = {
    "integrate_report_json": "Integration report",
    "integrate_nodes_json": "Graph nodes",
    "integrate_edges_json": "Graph edges",
    "log_file": "Execution log",
}

_PROGRAM_MANAGED_ARGUMENT_NAMES = {
    "input_path",
    "wt_input_path",
    "mut_input_path",
    "input_fasta",
    "wt_input_fasta",
    "mut_input_fasta",
    "input_msa",
    "wt_input_msa",
    "mut_input_msa",
    "input_dir",
    "wt_input_dir",
    "mut_input_dir",
    "output_dir",
    "wt_output_dir",
    "mut_output_dir",
    "substrate_dir",
}


def _humanize_identifier(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    return text.replace("_", " ").capitalize()


def _color_enabled() -> bool:
    """Use ANSI styling only for an interactive terminal that permits color."""

    return bool(getattr(sys.stdout, "isatty", lambda: False)()) and "NO_COLOR" not in os.environ


def _style(text: str, *codes: str) -> str:
    if not _color_enabled() or not codes:
        return text
    return "".join(codes) + text + _ANSI_RESET


def _banner(text: str) -> str:
    return _style(f"=== {text} ===", _ANSI_BOLD, _ANSI_CYAN)


def _section_heading(text: str) -> str:
    return _style(f"--- {text.upper()} ---", _ANSI_BOLD, _ANSI_CYAN)


def _status_style(text: str) -> str:
    normalized = text.casefold()
    if any(
        token in normalized
        for token in (
            "fail",
            "error",
            "restart required",
            "budget exceeded",
            "timed out",
            "timeout",
            "cancelled",
        )
    ):
        return _style(text, _ANSI_BOLD, _ANSI_RED)
    if any(
        token in normalized
        for token in ("waiting", "partial", "warning", "max turns")
    ):
        return _style(text, _ANSI_BOLD, _ANSI_YELLOW)
    return _style(text, _ANSI_BOLD, _ANSI_GREEN)


def _humanize_field_path(path: str) -> str:
    """Render a JSON field path clearly while retaining the path in stored data."""

    known = _FIELD_PATH_LABELS.get(path)
    if known is not None:
        return known
    if path == "$":
        return "Report root"
    parts = path.removeprefix("$.").split(".")
    labels: list[str] = []
    for part in parts:
        label = part.replace("[*]", " > All items")
        label = re.sub(
            r"\[(\d+)\]",
            lambda match: f" > Item {int(match.group(1)) + 1}",
            label,
        )
        label = label.replace("_", " ").strip()
        labels.append(
            f"{label[:1].upper()}{label[1:]}" if label else "Report field"
        )
    return " > ".join(labels)


def _render_field_paths(text: str) -> str:
    return _FIELD_PATH_PATTERN.sub(
        lambda match: f"source field: {_humanize_field_path(match.group(1))}",
        text,
    )


def _display_side_suffix(side: Any) -> str:
    label = _SIDE_DISPLAY_LABELS.get(str(side or "").casefold())
    return f" - {label}" if label else ""


def _humanize_side_references(text: str) -> str:
    rendered = re.sub(r"\s+\(single\)(?=\s*:)", "", text, flags=re.IGNORECASE)
    rendered = re.sub(r"\s+\[single\]", "", rendered, flags=re.IGNORECASE)
    for side, label in _SIDE_DISPLAY_LABELS.items():
        rendered = re.sub(
            rf"\({re.escape(side)}\)",
            f"({label})",
            rendered,
            flags=re.IGNORECASE,
        )
        rendered = re.sub(
            rf"\[{re.escape(side)}\]",
            f"[{label}]",
            rendered,
            flags=re.IGNORECASE,
        )
    return rendered


def _number_top_level_items(text: str) -> str:
    """Number deterministic top-level appendix items; preserve nested details."""

    counter = 0
    lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("- "):
            counter += 1
            lines.append(f"{counter}. {line[2:]}")
        else:
            lines.append(line)
    return "\n".join(lines)


def _source_origin(text: str) -> str:
    origin = text.strip()
    origin = re.sub(r"^calculated metric from\s+", "", origin)
    origin = re.sub(r"^metric from\s+", "", origin)
    origin = re.sub(
        r"\breports? from turn (\d+)",
        lambda match: f"report (turn {match.group(1)})",
        origin,
    )
    return origin[:1].upper() + origin[1:] if origin else "Recorded evidence"


def _render_source_item(source: str) -> list[str]:
    calculated = source.strip().casefold().startswith("calculated metric from ")
    field_match = _FIELD_PATH_PATTERN.search(source)
    if field_match is None:
        return [f"  - {_humanize_side_references(source.strip())}"]

    path = field_match.group(1)
    field_label = _humanize_field_path(path)
    origin = _FIELD_PATH_PATTERN.sub("", source).strip(" ,;")
    origin = _source_origin(origin)
    title = field_label
    if calculated:
        title += " (calculated from report entries)"
    return [f"  - {title}", f"    From: {origin} > {field_label}"]


def _display_evidence_finding(finding: str) -> str:
    return _humanize_side_references(finding)


def _render_evidence_appendix(text: str) -> str:
    """Render each finding and its evidence records as a readable source list."""

    lines = text.splitlines()
    rendered: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.startswith("- "):
            rendered.append(_humanize_side_references(line))
            index += 1
            continue

        raw_finding = line
        finding = _display_evidence_finding(raw_finding)
        rendered.append(finding)
        index += 1
        detail_lines: list[str] = []
        while index < len(lines) and not lines[index].startswith("- "):
            if lines[index].strip():
                detail_lines.append(lines[index].strip())
            index += 1
        if not detail_lines:
            continue

        details = " ".join(detail_lines)
        sources = list(
            dict.fromkeys(
                item.strip()
                for item in re.split(
                    r"(?:^|;\s*)(?:Evidence|Execution evidence|Planning evidence|Input evidence):\s*",
                    details,
                )
                if item.strip()
            )
        )
        if not sources:
            rendered.extend(f"  {item}" for item in detail_lines)
            continue
        rendered.append("  [Source]" if len(sources) == 1 else "  [Sources]")
        for source in sources:
            rendered.extend(_render_source_item(source))
    return "\n".join(rendered)


def _polish_appendix_text(label: str, text: str) -> str:
    """Remove labels that become redundant beneath explicit section headings."""

    if label == "Evidence supporting the analysis":
        return _render_evidence_appendix(text)
    if label == "Terminology sources":
        return re.sub(r"^- Terminology source:\s*", "- ", text, flags=re.MULTILINE)
    return _humanize_side_references(text)


def _strip_inline_evidence_labels(text: str) -> str:
    """Hide balanced inline evidence labels in the displayed narrative only."""

    rendered: list[str] = []
    cursor = 0
    while cursor < len(text):
        starts = [
            position
            for prefix in _INLINE_EVIDENCE_PREFIXES
            if (position := text.find(prefix, cursor)) >= 0
        ]
        if not starts:
            rendered.append(text[cursor:])
            break

        start = min(starts)
        depth = 0
        end: int | None = None
        for position in range(start, len(text)):
            character = text[position]
            if character == "[":
                depth += 1
            elif character == "]":
                depth -= 1
                if depth == 0:
                    end = position + 1
                    break

        if end is None:
            rendered.append(text[cursor:])
            break

        rendered.append(text[cursor:start].rstrip(" \t"))
        cursor = end

    return "".join(rendered)


def _display_sections(primary_label: str, final_answer: str) -> list[tuple[str, str]]:
    """Split deterministic appendices for display without changing final_answer."""

    sections: list[tuple[str, list[str]]] = [(primary_label, [])]
    for line in final_answer.splitlines():
        label = _DISPLAY_SECTION_LABELS.get(line.strip())
        if label is not None:
            sections.append((label, []))
            continue
        sections[-1][1].append(line)

    rendered: list[tuple[str, str]] = []
    for index, (label, lines) in enumerate(sections):
        body = "\n".join(lines).strip()
        if not body:
            continue
        if index == 0:
            body = _strip_inline_evidence_labels(body)
            body = re.sub(r"[ \t]+\n", "\n", body).strip()
        else:
            if label == "Evidence supporting the analysis":
                body = _polish_appendix_text(label, body)
                body = _number_top_level_items(body)
            else:
                body = _render_field_paths(body)
                body = _polish_appendix_text(label, body)
                body = _number_top_level_items(body)
        rendered.append((label, body))
    return rendered


def _format_argument_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


@lru_cache(maxsize=1)
def _tool_specs_for_display() -> dict[str, dict[str, Any]]:
    from .planning_utils import load_enzywizard_tool_specs

    return load_enzywizard_tool_specs()


def _parameter_annotation(tool_name: str, name: str, value: Any) -> str:
    spec = _tool_specs_for_display().get(tool_name, {})
    required_names = {
        item.get("name") for item in spec.get("required_arguments", [])
    }
    optional_specs = {
        item.get("name"): item for item in spec.get("optional_arguments", [])
    }
    labels: list[str] = []
    if name in required_names:
        labels.append("required")
    optional = optional_specs.get(name)
    if optional is not None and optional.get("default") == value:
        labels.append("default")
    if name in _PROGRAM_MANAGED_ARGUMENT_NAMES:
        labels.append("Agent-managed")
    return f" ({', '.join(labels)})" if labels else ""


def _artifact_display_label(artifact_type: str) -> str:
    known = _ARTIFACT_DISPLAY_LABELS.get(artifact_type)
    if known is not None:
        return known
    label = artifact_type.removesuffix("_json").replace("_", " ").strip()
    return label.capitalize() or "Output artifact"


def _message_type_label(response_mode: Any) -> str:
    return _MESSAGE_TYPE_LABELS.get(
        str(response_mode or ""),
        _humanize_identifier(response_mode, "Agent response"),
    )


def _body_section_label(response_mode: Any) -> str:
    return _BODY_SECTION_LABELS.get(
        str(response_mode or ""),
        "Response",
    )


def _session_state_label(
    *,
    analysis_route: Any,
    termination_reason: Any,
    awaiting_input: bool,
) -> str:
    # Terminal graph routes take precedence over a malformed/stale interrupt flag.
    # This prevents the UI from ever claiming that an ended session is waiting.
    if analysis_route == "restart_required":
        return "Ended; restart required"
    if analysis_route == "exit_requested":
        reason = _humanize_identifier(termination_reason, "completed")
        return f"Ended ({reason})"
    if awaiting_input:
        return "Waiting for the next user message"
    return "Completed"


def _display_message_envelope(
    *,
    response_mode: Any,
    analysis_status: Any,
    analysis_route: Any,
    termination_reason: Any,
    final_answer: str,
    awaiting_input: bool,
) -> None:
    """Render one answer with stable user-facing type and section labels."""

    message_type = _message_type_label(response_mode)
    status = _humanize_identifier(analysis_status, "Not available")
    session = _session_state_label(
        analysis_route=analysis_route,
        termination_reason=termination_reason,
        awaiting_input=awaiting_input,
    )
    print(f"\n{_banner('EnzyWizard-Agent')}")
    print(f"Message type: {_style(message_type, _ANSI_BOLD)}")
    print(f"Status: {_status_style(status)}")
    print(f"Session: {_status_style(session)}")
    for label, body in _display_sections(
        _body_section_label(response_mode),
        final_answer,
    ):
        print(f"\n{_section_heading(label)}")
        print(body)


def get_single_interrupt_payload(result: dict[str, Any]) -> dict[str, Any]:
    """从 LangGraph 执行结果中读取唯一的中断 payload。"""

    graph_interrupts = result.get("__interrupt__")
    if not graph_interrupts:
        raise ValueError("Agent graph result does not contain an interrupt.")
    if len(graph_interrupts) != 1:
        raise ValueError("CLI mode supports exactly one active interrupt at a time.")

    payload = graph_interrupts[0].value
    if not isinstance(payload, dict):
        raise ValueError("Agent interrupt payload must be a dictionary.")
    return payload


def display_tool_plan_confirmation(payload: dict[str, Any]) -> None:
    """以便于终端审阅的格式展示当前 ToolPlan。"""

    calls = payload.get("calls", [])
    print(f"\n{_banner('Tool Plan Confirmation')}")
    print(
        f"Turn {payload['turn_id']} | Plan {payload['plan_id']} | "
        f"{len(calls)} step{'s' if len(calls) != 1 else ''}"
    )
    print(
        "Purpose: "
        + _style(
            _humanize_identifier(payload["purpose_type"], "Not available"),
            _ANSI_BOLD,
        )
    )
    if payload.get("summary"):
        print(f"Summary: {payload['summary']}")
    print(f"Decision basis: {payload['plan_reason']}")
    if payload.get("plan_repairs"):
        print(
            "Internal validation recovery: Agent-managed plan structure was "
            "regenerated and revalidated before confirmation."
        )

    # 每个步骤都展示实际参数和预期产物，使用户确认的是完整可执行计划。
    for call in calls:
        side_suffix = _display_side_suffix(call.get("side"))
        step_title = (
            f"Step {call['step_index']}/{len(calls)}: "
            f"{call['tool_name']}{side_suffix}"
        )
        print(f"\n{_section_heading(step_title)}")
        print(f"Reason: {call['call_reason']}")
        print(_style("Parameters:", _ANSI_BOLD))
        arguments = call.get("arguments", {})
        if arguments:
            for index, (name, value) in enumerate(arguments.items(), start=1):
                annotation = _parameter_annotation(call["tool_name"], name, value)
                print(
                    f"  {index}. {name}: {_format_argument_value(value)}{annotation}"
                )
        else:
            print("  None")
        print(_style("Expected outputs:", _ANSI_BOLD))
        for index, artifact in enumerate(call["output_artifacts"], start=1):
            requirement = "required" if artifact["required"] else "optional"
            print(
                f"  {index}. {_artifact_display_label(artifact['artifact_type'])} "
                f"({requirement}): "
                f"{artifact['path_pattern']}"
            )


def read_tool_plan_confirmation(payload: dict[str, Any]) -> dict[str, Any]:
    """读取一次合法的终端确认，并生成 interrupt 恢复数据。"""

    while True:
        choice = input(
            "\nChoose [a]pprove, [m]odify, or [r]eject: "
        ).strip().casefold()

        if choice in {"a", "approve", "approved"}:
            decision = "approved"
            revision_request = None
            break
        if choice in {"m", "modify", "revision", "revision_requested"}:
            revision_request = input("Enter the plan revision request: ").strip()
            if not revision_request:
                print("[ERROR] A revision request cannot be empty.")
                continue
            decision = "revision_requested"
            break
        if choice in {"r", "reject", "rejected"}:
            decision = "rejected"
            revision_request = None
            break

        print("[ERROR] Enter a, m, or r.")

    # turn_id 和 plan_id 原样取自中断 payload，节点恢复时会再次与状态核对。
    return {
        "turn_id": payload["turn_id"],
        "plan_id": payload["plan_id"],
        "decision": decision,
        "revision_request": revision_request,
    }


def display_user_input_request(payload: dict[str, Any]) -> None:
    """展示上一轮回答和下一轮编号。"""

    _display_message_envelope(
        response_mode=payload.get("response_mode"),
        analysis_status=payload.get("analysis_status"),
        analysis_route=payload.get("analysis_route", "answer_generated"),
        termination_reason=payload.get("termination_reason"),
        final_answer=str(payload.get("final_answer", "")),
        awaiting_input=True,
    )
    turn_prompt = (
        f"Turn {payload['next_turn_id']}/{payload['max_turns']} | "
        "Enter the next message"
    )
    print(f"\n{_style(turn_prompt, _ANSI_BOLD, _ANSI_CYAN)}")


def read_next_user_input(payload: dict[str, Any]) -> dict[str, Any]:
    """读取非空下一轮消息；完整自然语言请求统一交给图内 Router 处理。"""

    while True:
        message = input("> ").strip()
        if message:
            return {
                "turn_id": payload["next_turn_id"],
                "message": message,
            }
        print("[ERROR] The next message cannot be empty.")


def display_final_analysis_result(result: dict[str, Any]) -> None:
    """图直接结束时展示最后一轮 Analysis 回答。"""

    analysis = result.get("current_analysis_result")
    if analysis is None:
        return
    final_answer = getattr(analysis, "final_answer", None)
    if final_answer:
        _display_message_envelope(
            response_mode=getattr(analysis, "response_mode", None),
            analysis_status=getattr(analysis, "analysis_status", None),
            analysis_route=getattr(analysis, "analysis_route", None),
            termination_reason=getattr(analysis, "termination_reason", None),
            final_answer=final_answer,
            awaiting_input=False,
        )
