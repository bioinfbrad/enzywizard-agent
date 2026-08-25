from __future__ import annotations

import re
from importlib.resources import files
from typing import Any

import yaml

from ..data_models.planning import PlanningPurposeType, ToolSelectionOutput


TOOL_SPEC_FIELD_NAMES = [
    "command_name",
    "description",
    "capabilities",
    "use_when",
    "do_not_use_when",
    "input_file_requirements",
    "planning_rules",
]


def purpose_needs_tool(purpose_type: PlanningPurposeType) -> bool:
    return purpose_type in (
        "single_protein_analysis",
        "wildtype_mutant_comparison",
    )


def load_enzywizard_tool_specs() -> dict[str, dict[str, Any]]:
    spec_path = files("enzywizard_agent.knowledge_base.enzywizard_tool_specs").joinpath(
        "enzywizard_tools.yaml"
    )
    spec_data = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    return spec_data["tools"]


def load_tool_specs_for_purpose(
    purpose_type: PlanningPurposeType,
) -> dict[str, dict[str, Any]]:
    tool_specs = load_enzywizard_tool_specs()
    selected_specs: dict[str, dict[str, Any]] = {}

    for tool_name, tool_spec in tool_specs.items():
        allowed_purpose_types = tool_spec.get("allowed_in_purpose_types", [])
        if purpose_type in allowed_purpose_types:
            selected_specs[tool_name] = select_tool_spec_fields(tool_spec)

    return selected_specs


def select_tool_spec_fields(tool_spec: dict[str, Any]) -> dict[str, Any]:
    return {
        field_name: tool_spec[field_name]
        for field_name in TOOL_SPEC_FIELD_NAMES
        if field_name in tool_spec
    }


def validate_selected_tools(
    tool_selection: ToolSelectionOutput,
    candidate_tool_names: set[str],
) -> ToolSelectionOutput:
    if tool_selection.tool_selection_status == "tool_selection_failed":
        error_message = tool_selection.error_message or tool_selection.selection_reason
        return create_tool_selection_failure(error_message)

    if not tool_selection.selected_tools:
        return create_tool_selection_failure(
            "Tool selection reported success without selecting any tools."
        )
    if not tool_selection.selection_reason:
        return create_tool_selection_failure(
            "Tool selection reported success without a selection reason."
        )
    if tool_selection.error_message:
        return create_tool_selection_failure(tool_selection.error_message)

    normalized_candidates = {
        re.sub(r"[_\s]+", "-", tool_name.strip().casefold()): tool_name
        for tool_name in candidate_tool_names
    }
    normalized_tool_names = [
        re.sub(r"[_\s]+", "-", tool_name.strip().casefold())
        for tool_name in tool_selection.selected_tools
    ]
    invalid_tool_names = [
        tool_name
        for tool_name, normalized_name in zip(
            tool_selection.selected_tools,
            normalized_tool_names,
        )
        if normalized_name not in normalized_candidates
    ]
    if invalid_tool_names:
        invalid_names = ", ".join(invalid_tool_names)
        return ToolSelectionOutput(
            tool_selection_status="tool_selection_failed",
            selected_tools=[],
            selection_reason="",
            error_message=(
                "Tool selection returned tools outside the allowed candidate set: "
                f"{invalid_names}."
            ),
        )

    canonical_tool_names = [
        normalized_candidates[normalized_name]
        for normalized_name in normalized_tool_names
    ]
    if len(canonical_tool_names) != len(set(canonical_tool_names)):
        return create_tool_selection_failure(
            "Tool selection returned duplicate tools after name normalization."
        )

    conflicting_tool_pairs = (
        {"enzywizard-batch", "enzywizard-integrate"},
        {"enzywizard-mut-batch", "enzywizard-mut-integrate"},
    )
    selected_tool_names = set(canonical_tool_names)
    for conflicting_pair in conflicting_tool_pairs:
        if conflicting_pair <= selected_tool_names:
            tool_names = " and ".join(sorted(conflicting_pair))
            return create_tool_selection_failure(
                f"Tool selection contains incompatible tools: {tool_names}."
            )

    return ToolSelectionOutput(
        tool_selection_status="tool_selection_succeeded",
        selected_tools=canonical_tool_names,
        selection_reason=tool_selection.selection_reason,
        error_message="",
    )


def create_tool_selection_failure(error_message: str) -> ToolSelectionOutput:
    message = str(error_message).strip() or "Tool selection failed without an error message."
    return ToolSelectionOutput(
        tool_selection_status="tool_selection_failed",
        selected_tools=[],
        selection_reason="",
        error_message=message,
    )
