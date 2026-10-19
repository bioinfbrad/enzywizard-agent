"""Fast deterministic contracts for workflow requirement compilation.

This script does not call an external model or execute EnzyWizard tools. It verifies
that structured scientific requirements compile to stable tool sets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from enzywizard_agent.conf import AgentConfig
from enzywizard_agent.data_models.agent_inputs import AgentInputs
from enzywizard_agent.data_models.planning import (
    ParameterExtractionOutput,
    PlanningResult,
    ToolPlan,
    ToolSelectionOutput,
    WorkflowRequirementsDraft,
    WorkflowRequirementsOutput,
)
from enzywizard_agent.data_models.request_routing import (
    RequestedAction,
    RequestRoutingOutput,
    RequestRoutingResult,
)
from enzywizard_agent.data_models.user_confirmation import UserConfirmation
import enzywizard_agent.services.planning_service as planning_service
from enzywizard_agent.utils.planning_utils import (
    load_analysis_tool_specs_for_purpose,
)
from enzywizard_agent.utils.tool_plan_utils import create_tool_call


class FakeStructuredModel:
    def __init__(self, outputs: Any):
        self.outputs = outputs

    def invoke(self, _messages: list[Any]) -> WorkflowRequirementsOutput:
        return next(self.outputs)


class FakeModel:
    def __init__(self, outputs: Any):
        self.outputs = outputs

    def with_structured_output(
        self, _schema: type[Any], **_kwargs: Any
    ) -> FakeStructuredModel:
        return FakeStructuredModel(self.outputs)


def planning_result(purpose_type: str) -> PlanningResult:
    return PlanningResult(
        turn_id=1,
        plan_id=1,
        planning_route="tool_unused",
        planning_flag="intent_recognized",
        needs_tool=True,
        summary=None,
        purpose_type=purpose_type,
        purpose_reason="Contract fixture.",
        tool_selection=None,
        parameter_extraction=None,
        tool_plan=None,
        error_messages=[],
    )


def requirements(
    scope_mode: str,
    requested: list[str],
    excluded: list[str] | None = None,
) -> WorkflowRequirementsOutput:
    return WorkflowRequirementsOutput(
        workflow_requirements_status="workflow_requirements_succeeded",
        scope_mode=scope_mode,
        requested_analysis_types=requested,
        excluded_analysis_types=excluded or [],
        requirements_reason="Contract fixture requirements.",
        error_message="",
    )


def compile_tools(
    purpose_type: str,
    structured_requirements: WorkflowRequirementsOutput,
    preparation_tools: list[str],
) -> tuple[PlanningResult, list[str]]:
    result = planning_result(purpose_type)
    state = {
        "agent_inputs": AgentInputs(
            structure_file="/tmp/wt.pdb",
            mut_structure_file=(
                "/tmp/mut.pdb"
                if purpose_type == "wildtype_mutant_comparison"
                else None
            ),
            msa_file="/tmp/wt.sto",
            mut_msa_file=(
                "/tmp/mut.sto"
                if purpose_type == "wildtype_mutant_comparison"
                else None
            ),
            substrate_name="glucose",
            amino_acid_substitution=(
                "S70G"
                if purpose_type == "wildtype_mutant_comparison"
                else None
            ),
        )
    }
    selection = planning_service._compile_workflow_requirements(
        state,
        result,
        structured_requirements,
        preparation_tools,
        set(load_analysis_tool_specs_for_purpose(purpose_type)),
    )
    if selection.tool_selection_status != "tool_selection_succeeded":
        raise AssertionError(selection.error_message)
    return result, selection.selected_tools


def assert_tools(actual: list[str], expected: list[str], label: str) -> None:
    if actual != expected:
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def main() -> int:
    inconsistent_complete = WorkflowRequirementsDraft(
        workflow_requirements_status="workflow_requirements_succeeded",
        scope_mode="complete_analysis",
        requested_analysis_types=["energy", "pocket"],
        excluded_analysis_types=[],
        requirements_reason="The user requested the complete workflow.",
        error_message="",
    )
    normalized_complete = planning_service._normalize_workflow_requirements_draft(
        inconsistent_complete
    )
    if (
        normalized_complete.scope_mode != "complete_analysis"
        or normalized_complete.requested_analysis_types
    ):
        raise AssertionError("Complete-analysis draft was not normalized")
    print("[PASS] Repairable complete-analysis drafts reach strict validation.")

    overlapping_revision = WorkflowRequirementsDraft(
        workflow_requirements_status="workflow_requirements_succeeded",
        scope_mode="specific_analyses",
        requested_analysis_types=["energy", "pocket", "energy"],
        excluded_analysis_types=["pocket"],
        requirements_reason="Keep energy and remove pocket.",
        error_message="",
    )
    normalized_revision = planning_service._normalize_workflow_requirements_draft(
        overlapping_revision
    )
    if normalized_revision.requested_analysis_types != ["energy"]:
        raise AssertionError("Excluded analyses did not override the draft request")
    print("[PASS] Exclusions and duplicates are normalized deterministically.")

    draft = requirements("complete_analysis", [])
    reviewed = requirements(
        "specific_analyses",
        ["substrate", "docking", "interaction"],
    )
    fake_outputs = iter([draft, reviewed])
    original_factory = planning_service._create_planning_llm
    planning_service._create_planning_llm = lambda _state: FakeModel(fake_outputs)
    try:
        extracted = planning_service._extract_workflow_requirements(
            {
                "current_user_message": (
                    "Compare WT and mutant glucose docking and interactions."
                ),
                "agent_inputs": AgentInputs(
                    structure_file="/tmp/wt.pdb",
                    msa_file=None,
                    mut_structure_file="/tmp/mut.pdb",
                    mut_msa_file=None,
                    substrate_name="glucose",
                    amino_acid_substitution="S70G",
                ),
                "planning_results": [],
            },
            planning_result("wildtype_mutant_comparison"),
            ["enzywizard-mut-clean"],
        )
    finally:
        planning_service._create_planning_llm = original_factory
    if extracted != reviewed:
        raise AssertionError("Workflow requirement review did not correct the draft")
    print("[PASS] Independent requirement review can correct an over-broad draft.")

    rerun_workdir = Path("/tmp/enzywizard-planning-rerun-contract")
    rerun_state: dict[str, Any] = {
        "config": AgentConfig(
            api_key="not-used",
            base_url="https://example.invalid/v1",
            model="not-used",
            workdir=rerun_workdir,
        ),
        "session_id": "planning-rerun-contract",
        "current_turn_id": 2,
        "current_user_message": "Rerun the previous workflow with identical settings.",
        "current_request_routing": RequestRoutingResult(
            turn_id=2,
            routing_status="succeeded",
            route="request_to_planning",
            output=RequestRoutingOutput(
                request_type="tool_workflow",
                actions=[
                    RequestedAction(
                        action_type="reference_historical_context",
                        target="previous workflow settings",
                    ),
                    RequestedAction(
                        action_type="run_tool_workflow",
                        target="rerun the previous workflow",
                    ),
                ],
                requires_tool_execution=True,
                references_history=True,
                requires_history_artifacts=False,
                reuse_policy="none",
                requests_recomputation=True,
                clarification_question=None,
                routing_reason="Contract rerun.",
            ),
            summary="Contract rerun route.",
            error_message="",
        ),
        "user_confirmations": [
            UserConfirmation(
                turn_id=1,
                plan_id=1,
                decision="approved",
                revision_request=None,
            )
        ],
        "tool_executions": [],
    }
    source_call = create_tool_call(
        rerun_state,
        turn_id=1,
        plan_id=1,
        step_index=1,
        tool_name="enzywizard-energy",
        side="single",
        arguments={
            "input_path": "/tmp/old-input.cif",
            "output_dir": str(rerun_workdir / "old-output"),
            "minimize_energy": False,
            "minimization_iteration": 250,
        },
        staging_rules=[],
        call_reason="Original energy settings.",
    )
    source_plan = ToolPlan(
        turn_id=1,
        plan_id=1,
        purpose_type="single_protein_analysis",
        summary="Original approved energy plan.",
        plan_reason="Original approved energy plan.",
        calls=[source_call],
    )
    source_result = planning_result("single_protein_analysis").model_copy(
        update={
            "turn_id": 1,
            "plan_id": 1,
            "workflow_requirements": requirements(
                "specific_analyses", ["energy"]
            ),
            "tool_selection": ToolSelectionOutput(
                tool_selection_status="tool_selection_succeeded",
                selected_tools=["enzywizard-energy"],
                selection_reason="Original approved energy plan.",
                error_message="",
            ),
            "parameter_extraction": ParameterExtractionOutput(
                parameter_extraction_status="parameter_extraction_succeeded",
                parameter_adjustments=[],
                profile_requests=[],
                unsupported_requests=[],
                extraction_summary="Original settings.",
                error_message="",
            ),
            "tool_plan": source_plan,
        }
    )
    rerun_state["planning_results"] = [source_result]
    default_call = create_tool_call(
        rerun_state,
        turn_id=2,
        plan_id=1,
        step_index=1,
        tool_name="enzywizard-energy",
        side="single",
        arguments={
            "input_path": "/tmp/current-input.cif",
            "output_dir": str(rerun_workdir / "current-output"),
            "minimize_energy": True,
            "minimization_iteration": 100,
        },
        staging_rules=[],
        call_reason="Fresh current-turn energy plan.",
    )
    default_plan = source_plan.model_copy(
        update={
            "turn_id": 2,
            "plan_id": 1,
            "calls": [default_call],
        }
    )
    rerun_requirements = planning_service._extract_workflow_requirements(
        rerun_state,
        planning_result("single_protein_analysis").model_copy(
            update={"turn_id": 2}
        ),
        [],
    )
    inherited_plan = planning_service.apply_historical_rerun_settings(
        rerun_state,
        default_plan,
    )
    inherited_arguments = inherited_plan.calls[0].arguments
    if rerun_requirements.requested_analysis_types != ["energy"]:
        raise AssertionError("Historical rerun did not inherit scientific requirements")
    if (
        inherited_arguments["minimize_energy"] is not False
        or inherited_arguments["minimization_iteration"] != 250
    ):
        raise AssertionError("Historical rerun did not inherit editable parameters")
    if (
        inherited_arguments["input_path"] != "/tmp/current-input.cif"
        or inherited_arguments["output_dir"]
        != str(rerun_workdir / "current-output")
    ):
        raise AssertionError("Historical rerun copied program-managed paths")
    print("[PASS] Historical reruns inherit requirements and editable settings only.")

    result, tools = compile_tools(
        "wildtype_mutant_comparison",
        requirements(
            "specific_analyses",
            ["substrate", "docking", "interaction"],
        ),
        ["enzywizard-mut-clean"],
    )
    assert_tools(
        tools,
        [
            "enzywizard-mut-clean",
            "enzywizard-substrate",
            "enzywizard-dock",
            "enzywizard-interaction",
            "enzywizard-mut-integrate",
        ],
        "bounded paired docking workflow",
    )
    if result.workflow_requirements is None:
        raise AssertionError("Structured workflow requirements were not persisted")
    print("[PASS] Bounded paired docking cannot expand into mut-batch.")

    _, tools = compile_tools(
        "wildtype_mutant_comparison",
        requirements("complete_analysis", []),
        ["enzywizard-mut-clean"],
    )
    assert_tools(
        tools,
        ["enzywizard-mut-clean", "enzywizard-mut-batch"],
        "complete paired workflow",
    )
    print("[PASS] Explicit complete scope compiles to mut-batch.")

    _, tools = compile_tools(
        "wildtype_mutant_comparison",
        requirements(
            "specific_analyses",
            ["energy", "flexibility", "disorder"],
            ["pocket"],
        ),
        ["enzywizard-mut-clean"],
    )
    assert_tools(
        tools,
        [
            "enzywizard-mut-clean",
            "enzywizard-energy",
            "enzywizard-flexibility",
            "enzywizard-disorder",
            "enzywizard-mut-integrate",
        ],
        "revised paired workflow",
    )
    print("[PASS] Final revision preserves requested analyses and excludes pocket.")

    _, tools = compile_tools(
        "single_protein_analysis",
        requirements("preparation_only", []),
        ["enzywizard-clean"],
    )
    assert_tools(tools, ["enzywizard-clean"], "single preparation workflow")
    print("[PASS] Preparation-only scope compiles to clean only.")

    print("\n[ALL PASSED] Planning requirement compilation contracts are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
