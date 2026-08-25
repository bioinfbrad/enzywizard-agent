from __future__ import annotations

import json
from typing import Any

from ..data_models.planning import PlanningResult
from ..state import AgentState


def build_planning_context(state: AgentState) -> str:
    sections: list[str] = []

    current_user_message = state.get("current_user_message")
    if current_user_message:
        sections.append(f"Current user request:\n{current_user_message.strip()}")

    user_confirmation = state.get("current_user_confirmation")
    if user_confirmation is not None and user_confirmation.revision_request:
        sections.append(
            f"User revision request:\n{user_confirmation.revision_request.strip()}"
        )

    user_message_summary = state.get("user_message_summary")
    if user_message_summary:
        sections.append(f"Previous user request summary:\n{user_message_summary.strip()}")

    agent_inputs = state.get("agent_inputs")
    if agent_inputs is not None:
        input_lines = []
        for field_name, value in agent_inputs.model_dump().items():
            if value is not None:
                input_lines.append(f"- {field_name}: {value}")
        if input_lines:
            input_context = "\n".join(input_lines)
            sections.append(f"Provided agent inputs:\n{input_context}")

    return "\n\n".join(sections)


def build_tool_selection_context(
    state: AgentState,
    planning_result: PlanningResult,
    tool_specs: dict[str, dict[str, Any]],
) -> str:
    sections: list[str] = []

    agent_inputs = state.get("agent_inputs")
    cleaned_files = state.get("cleaned_files")
    input_structure_available = bool(
        agent_inputs is not None and agent_inputs.structure_file
    )
    input_mut_structure_available = bool(
        agent_inputs is not None and agent_inputs.mut_structure_file
    )
    input_amino_acid_substitution_available = bool(
        agent_inputs is not None and agent_inputs.amino_acid_substitution
    )
    trusted_single_cleaned_structure_available = bool(
        cleaned_files is not None
        and (
            cleaned_files.single_cleaned_structure_file
            or cleaned_files.single_cleaned_cif_file
            or cleaned_files.single_cleaned_pdb_file
        )
    )
    trusted_wt_cleaned_structure_available = bool(
        cleaned_files is not None
        and (
            cleaned_files.wt_cleaned_structure_file
            or cleaned_files.wt_cleaned_cif_file
            or cleaned_files.wt_cleaned_pdb_file
        )
    )
    trusted_mut_cleaned_structure_available = bool(
        cleaned_files is not None
        and (
            cleaned_files.mut_cleaned_structure_file
            or cleaned_files.mut_cleaned_cif_file
            or cleaned_files.mut_cleaned_pdb_file
        )
    )
    cleaned_amino_acid_substitution_available = bool(
        cleaned_files is not None
        and cleaned_files.cleaned_amino_acid_substitution
    )
    paired_cleaned_inputs_available = (
        trusted_wt_cleaned_structure_available
        and trusted_mut_cleaned_structure_available
        and cleaned_amino_acid_substitution_available
    )
    paired_raw_inputs_available = (
        input_structure_available
        and input_mut_structure_available
        and input_amino_acid_substitution_available
    )
    if planning_result.purpose_type == "single_protein_analysis":
        input_availability = {
            "initial_inputs": {
                "structure": input_structure_available,
                "msa": bool(agent_inputs is not None and agent_inputs.msa_file),
                "substrate": bool(
                    agent_inputs is not None and agent_inputs.substrate_name
                ),
            },
            "recoverable_intermediate_inputs": {
                "single_cleaned_structure": {
                    "currently_available": trusted_single_cleaned_structure_available,
                    "can_be_generated_in_current_plan": input_structure_available,
                    "producer_tool": "enzywizard-clean",
                },
            },
        }
    else:
        input_availability = {
            "initial_inputs": {
                "structure": input_structure_available,
                "msa": bool(agent_inputs is not None and agent_inputs.msa_file),
                "mut_structure": input_mut_structure_available,
                "mut_msa": bool(
                    agent_inputs is not None and agent_inputs.mut_msa_file
                ),
                "substrate": bool(
                    agent_inputs is not None and agent_inputs.substrate_name
                ),
                "amino_acid_substitution": (
                    input_amino_acid_substitution_available
                ),
            },
            "recoverable_intermediate_inputs": {
                "paired_cleaned_inputs": {
                    "currently_available": paired_cleaned_inputs_available,
                    "can_be_generated_in_current_plan": paired_raw_inputs_available,
                    "producer_tool": "enzywizard-mut-clean",
                },
            },
        }
    sections.append(
        "Program-resolved input and dependency availability (authoritative):\n"
        + json.dumps(input_availability, ensure_ascii=False, indent=2)
    )

    current_user_message = state.get("current_user_message")
    if current_user_message:
        sections.append(f"Current user request:\n{current_user_message.strip()}")

    user_confirmation = state.get("current_user_confirmation")
    if user_confirmation is not None and user_confirmation.revision_request:
        sections.append(
            f"User revision request:\n{user_confirmation.revision_request.strip()}"
        )

    user_message_summary = state.get("user_message_summary")
    if user_message_summary:
        sections.append(f"Previous user request summary:\n{user_message_summary.strip()}")

    if agent_inputs is not None:
        input_context = json.dumps(
            agent_inputs.model_dump(exclude_none=True),
            ensure_ascii=False,
            indent=2,
        )
        sections.append(f"Agent inputs:\n{input_context}")

    if cleaned_files is not None:
        cleaned_file_context = json.dumps(
            cleaned_files.model_dump(exclude_none=True),
            ensure_ascii=False,
            indent=2,
        )
        sections.append(f"Current cleaned file state:\n{cleaned_file_context}")

    artifact_summary = state.get("artifact_summary")
    if artifact_summary:
        sections.append(f"Artifact summary:\n{artifact_summary.strip()}")

    artifacts = state.get("artifacts")
    if artifacts:
        artifact_context = json.dumps(
            [artifact.model_dump() for artifact in artifacts],
            ensure_ascii=False,
            indent=2,
        )
        sections.append(f"Available artifacts:\n{artifact_context}")

    planning_context = {
        "purpose_type": planning_result.purpose_type,
        "purpose_reason": planning_result.purpose_reason,
        "needs_tool": planning_result.needs_tool,
    }
    sections.append(
        "Current planning intent:\n"
        + json.dumps(planning_context, ensure_ascii=False, indent=2)
    )

    tool_spec_context = json.dumps(tool_specs, ensure_ascii=False, indent=2)
    sections.append(f"Candidate EnzyWizard tool specifications:\n{tool_spec_context}")

    return "\n\n".join(sections)
