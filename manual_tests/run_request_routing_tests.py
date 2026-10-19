"""Fast contract tests for request routing and structured history selection.

The script does not call an external model or execute EnzyWizard tools. It injects
strict structured model outputs and verifies graph routes plus deterministic artifact
selection.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

from enzywizard_agent.conf import AgentConfig
from enzywizard_agent.data_models.agent_inputs import AgentInputs
from enzywizard_agent.data_models.artifact import Artifact, CleanedFiles
from enzywizard_agent.data_models.history_query import (
    HistoryArtifactSelectionDraft,
    HistoryTurnSelectionDraft,
)
from enzywizard_agent.data_models.request_routing import (
    RequestedAction,
    RoutingDraftAction,
    RequestRoutingDraft,
    RequestRoutingOutput,
)
import enzywizard_agent.services.request_routing_service as routing_service
from enzywizard_agent.builders.request_routing_context_builder import (
    build_request_routing_context,
)
from enzywizard_agent.graph import build_graph
from enzywizard_agent.state import AgentState


class FakeStructuredModel:
    def __init__(self, output: Any = None, error: Exception | None = None):
        self.output = output
        self.error = error

    def invoke(self, _messages: list[Any]) -> Any:
        if self.error is not None:
            raise self.error
        return self.output


class FakeModel:
    def __init__(self, output: Any = None, error: Exception | None = None):
        self.output = output
        self.error = error

    def with_structured_output(
        self, _schema: type[Any], **_kwargs: Any
    ) -> FakeStructuredModel:
        return FakeStructuredModel(self.output, self.error)


class SequenceFakeModel:
    def __init__(self, outputs: list[Any]):
        self.outputs = iter(outputs)

    def with_structured_output(
        self, _schema: type[Any], **_kwargs: Any
    ) -> FakeStructuredModel:
        return FakeStructuredModel(next(self.outputs))


def empty_cleaned_files() -> CleanedFiles:
    return CleanedFiles(
        single_clean_output_dir=None,
        single_cleaned_structure_file=None,
        single_cleaned_cif_file=None,
        single_cleaned_pdb_file=None,
        single_cleaned_fasta_file=None,
        single_clean_report_file=None,
        mut_clean_output_dir=None,
        wt_cleaned_structure_file=None,
        wt_cleaned_cif_file=None,
        wt_cleaned_pdb_file=None,
        wt_cleaned_fasta_file=None,
        mut_cleaned_structure_file=None,
        mut_cleaned_cif_file=None,
        mut_cleaned_pdb_file=None,
        mut_cleaned_fasta_file=None,
        mut_clean_report_file=None,
        cleaned_amino_acid_substitution=None,
    )


def build_state() -> AgentState:
    artifacts = [
        Artifact(
            turn_id=1,
            plan_id=1,
            artifact_type="energy_report_json",
            path="/tmp/turn1_energy.json",
            source_tool="enzywizard-energy",
            side="single",
            metadata={},
        ),
        Artifact(
            turn_id=1,
            plan_id=1,
            artifact_type="cleaned_structure_cif",
            path="/tmp/turn1_cleaned.cif",
            source_tool="enzywizard-clean",
            side="single",
            metadata={},
        ),
        Artifact(
            turn_id=2,
            plan_id=1,
            artifact_type="pocket_report_json",
            path="/tmp/turn2_pocket.json",
            source_tool="enzywizard-pocket",
            side="single",
            metadata={},
        ),
    ]
    return {
        "config": AgentConfig(
            api_key="not-used",
            base_url="https://example.invalid/v1",
            model="not-used",
            workdir=Path("/tmp"),
            max_turns=5,
        ),
        "session_id": "routing-test",
        "agent_inputs": AgentInputs(
            structure_file="/tmp/input.pdb",
            msa_file=None,
            mut_structure_file=None,
            mut_msa_file=None,
            substrate_name=None,
            amino_acid_substitution=None,
        ),
        "current_turn_id": 3,
        "current_plan_id": 0,
        "current_user_message": "Synthetic structured-routing test.",
        "user_messages": [],
        "user_message_summary": "",
        "request_routing_results": [],
        "request_routing_summary": "",
        "history_query_results": [],
        "history_query_summary": "",
        "current_history_artifacts": [],
        "planning_results": [],
        "planning_summary": "",
        "user_confirmations": [],
        "user_confirmation_summary": "",
        "tool_executions": [],
        "tool_calling_summary": "",
        "analysis_results": [],
        "analysis_summary": "",
        "cleaned_files": empty_cleaned_files(),
        "current_plan_artifacts": [],
        "artifacts": artifacts,
        "artifact_summary": "",
    }


def routed_output(
    request_type: str,
    actions: list[str],
    *,
    tools: bool = False,
    history: bool = False,
    history_artifacts: bool | None = None,
    reuse: str = "none",
    clarification_question: str | None = None,
) -> RequestRoutingOutput:
    return RequestRoutingOutput(
        request_type=request_type,
        actions=[
            RequestedAction(
                action_type=action,
                target="test target",
                history_operation=(
                    "summarize"
                    if action == "analyze_history"
                    else "list_artifacts"
                    if action == "list_artifacts"
                    else "reuse_as_input"
                    if action == "reuse_historical_output"
                    else None
                ),
            )
            for action in actions
        ],
        requires_tool_execution=tools,
        references_history=history,
        requires_history_artifacts=(
            history if history_artifacts is None else history_artifacts
        ),
        reuse_policy=reuse,
        requests_recomputation=False,
        clarification_question=clarification_question,
        routing_reason="Test route.",
    )


def main() -> int:
    state = build_state()
    routing_context = json.loads(build_request_routing_context(state))
    if "available_artifact_catalog" in routing_context:
        raise AssertionError("Router context still exposes the artifact catalog")
    if "/tmp/input.pdb" in json.dumps(routing_context):
        raise AssertionError("Router context exposes current input paths")
    if routing_context["completed_turn_ids"] != [1, 2]:
        raise AssertionError("Router context lost completed-turn availability")
    original_create_llm = routing_service._create_llm
    try:
        route_cases = [
            (
                routed_output("tool_workflow", ["run_tool_workflow"], tools=True),
                "request_to_planning",
            ),
            (
                routed_output(
                    "tool_workflow",
                    ["reference_historical_context", "run_tool_workflow"],
                    tools=True,
                    history=True,
                    history_artifacts=False,
                ),
                "request_to_planning",
            ),
            (
                routed_output(
                    "history_analysis",
                    ["analyze_history"],
                    history=True,
                ),
                "request_to_history_query",
            ),
            (
                routed_output(
                    "mixed",
                    ["reuse_historical_output", "run_tool_workflow"],
                    tools=True,
                    history=True,
                    reuse="required",
                ),
                "request_to_history_query",
            ),
            (
                routed_output("software_help", ["answer_software_help"]),
                "request_to_analysis",
            ),
            (
                routed_output(
                    "general_scientific_question",
                    ["answer_general_question"],
                ),
                "request_to_analysis",
            ),
            (
                routed_output("conversation", ["continue_conversation"]),
                "request_to_analysis",
            ),
            (
                routed_output("out_of_scope", ["decline_out_of_scope"]),
                "request_to_analysis",
            ),
            (
                routed_output(
                    "clarification",
                    ["clarify_request"],
                    clarification_question="Which completed turn do you mean?",
                ),
                "request_to_analysis",
            ),
            (
                routed_output("exit", ["exit_session"]),
                "request_to_analysis",
            ),
        ]
        for output, expected_route in route_cases:
            routing_service._create_llm = lambda _state, item=output: FakeModel(item)
            result = routing_service.route_user_request(state)
            if result.routing_status != "succeeded" or result.route != expected_route:
                raise AssertionError(
                    f"Unexpected route for {output.request_type}: {result}"
                )
        print(f"[PASS] Verified {len(route_cases)} structured request routes.")

        mistaken_history = RequestRoutingDraft(
            actions=[
                RoutingDraftAction(
                    kind="compare_history",
                    target="existing energy reports",
                )
            ]
        )
        corrected_tool_route = RequestRoutingDraft(
            actions=[
                RoutingDraftAction(
                    kind="run_workflow",
                    target="compare the supplied WT and mutant energy",
                )
            ]
        )
        routing_outputs = iter(
            [mistaken_history, corrected_tool_route, corrected_tool_route]
        )
        routing_service._create_llm = lambda _state: FakeModel(
            next(routing_outputs)
        )
        reviewed_route = routing_service.route_user_request(state)
        if (
            reviewed_route.routing_status != "succeeded"
            or reviewed_route.route != "request_to_planning"
            or reviewed_route.output is None
            or reviewed_route.output.request_type != "tool_workflow"
        ):
            raise AssertionError("Router review did not correct the first draft")
        print("[PASS] Router review can correct a valid but incorrect first draft.")

        mixed_candidate = RequestRoutingDraft(
            actions=[
                RoutingDraftAction(
                    kind="list_history_files",
                    target="logs from the previous turn",
                ),
                RoutingDraftAction(
                    kind="run_workflow",
                    target="run a new energy workflow",
                ),
            ]
        )
        dropped_action_review = RequestRoutingDraft(
            actions=[
                RoutingDraftAction(
                    kind="run_workflow",
                    target="run a new energy workflow",
                )
            ]
        )
        routing_outputs = iter(
            [mixed_candidate, dropped_action_review, mixed_candidate]
        )
        routing_service._create_llm = lambda _state: FakeModel(
            next(routing_outputs)
        )
        preserved_mixed = routing_service.route_user_request(state)
        preserved_actions = {
            item.action_type
            for item in (
                preserved_mixed.output.actions
                if preserved_mixed.output is not None
                else []
            )
        }
        if preserved_actions != {"list_artifacts", "run_tool_workflow"}:
            raise AssertionError("Router review silently dropped a mixed action")
        print("[PASS] Router review cannot silently reduce a mixed request.")

        required_reuse_draft = RequestRoutingDraft(
            actions=[
                RoutingDraftAction(
                    kind="reuse_history_output_required",
                    target="cleaned structure from turn 1 for flexibility",
                )
            ]
        )
        normalized_reuse = routing_service._normalize_request_routing_draft(
            required_reuse_draft
        )
        if (
            normalized_reuse.request_type != "mixed"
            or not normalized_reuse.requires_tool_execution
            or not normalized_reuse.references_history
            or not normalized_reuse.requires_history_artifacts
            or normalized_reuse.reuse_policy != "required"
        ):
            raise AssertionError("Router draft normalization lost reuse semantics")

        optional_reuse_draft = RequestRoutingDraft(
            actions=[
                RoutingDraftAction(
                    kind="reuse_history_output_optional",
                    target="turn 2 docked substrate if compatible",
                )
            ]
        )
        normalized_optional_reuse = routing_service._normalize_request_routing_draft(
            optional_reuse_draft
        )
        if normalized_optional_reuse.reuse_policy != "allowed":
            raise AssertionError("Optional historical reuse was not preserved")

        compact_rerun_draft = RequestRoutingDraft(
            actions=[
                RoutingDraftAction(
                    kind="rerun_previous_workflow",
                    target="rerun the previous analysis with the same settings",
                )
            ]
        )
        normalized_rerun = routing_service._normalize_request_routing_draft(
            compact_rerun_draft
        )
        normalized_rerun_actions = {
            item.action_type for item in normalized_rerun.actions
        }
        if normalized_rerun_actions != {
            "reference_historical_context",
            "run_tool_workflow",
        }:
            raise AssertionError(
                "A previous-workflow rerun lost its historical plan reference"
            )
        if (
            not normalized_rerun.requests_recomputation
            or normalized_rerun.requires_history_artifacts
        ):
            raise AssertionError("Explicit rerun did not preserve recomputation")

        recompute_draft = RequestRoutingDraft(
            actions=[
                RoutingDraftAction(
                    kind="recompute_workflow",
                    target="freshly recompute energy from current inputs",
                )
            ]
        )
        normalized_recompute = routing_service._normalize_request_routing_draft(
            recompute_draft
        )
        if (
            {item.action_type for item in normalized_recompute.actions}
            != {"run_tool_workflow"}
            or normalized_recompute.references_history
            or not normalized_recompute.requests_recomputation
        ):
            raise AssertionError("Current-input recomputation gained history semantics")

        optional_legacy = routing_service._normalize_routing_candidate(
            {
                "reuse_policy": "allowed",
                "actions": [
                    {
                        "action_type": "reuse_historical_output",
                        "target": "turn 1 cleaned structure if compatible",
                    }
                ],
            }
        )
        if optional_legacy["actions"][0]["kind"] != "reuse_history_output_optional":
            raise AssertionError("Legacy optional reuse was strengthened to required")
        ambiguous_legacy = routing_service._normalize_routing_candidate(
            {
                "actions": [
                    {
                        "action_type": "reuse_historical_output",
                        "target": "turn 1 output",
                    }
                ]
            }
        )
        if ambiguous_legacy["actions"][0]["kind"] != "reuse_historical_output":
            raise AssertionError("Ambiguous legacy reuse semantics were guessed")

        legacy_rerun = routing_service._normalize_routing_candidate(
            {
                "requests_recomputation": True,
                "actions": [
                    {
                        "action_type": "reference_historical_context",
                        "target": "previous settings",
                    },
                    {
                        "action_type": "run_tool_workflow",
                        "target": "rerun the workflow",
                    },
                ],
            }
        )
        normalized_legacy_rerun = routing_service._normalize_request_routing_draft(
            RequestRoutingDraft.model_validate(legacy_rerun)
        )
        if not normalized_legacy_rerun.requests_recomputation:
            raise AssertionError("Legacy explicit recomputation flag was discarded")
        if {
            item.action_type for item in normalized_legacy_rerun.actions
        } != {"reference_historical_context", "run_tool_workflow"}:
            raise AssertionError("Legacy historical rerun actions were not preserved")

        legacy_current_recompute = routing_service._normalize_routing_candidate(
            {
                "requests_recomputation": True,
                "actions": [
                    {
                        "action_type": "run_tool_workflow",
                        "target": "freshly recompute current energy",
                    }
                ],
            }
        )
        normalized_legacy_current = routing_service._normalize_request_routing_draft(
            RequestRoutingDraft.model_validate(legacy_current_recompute)
        )
        if (
            not normalized_legacy_current.requests_recomputation
            or normalized_legacy_current.references_history
        ):
            raise AssertionError("Legacy current recomputation gained history context")

        previous_plan_draft = RequestRoutingDraft(
            actions=[
                RoutingDraftAction(
                    kind="reference_previous_plan",
                    target="use the previous plan as context",
                ),
                RoutingDraftAction(
                    kind="run_workflow",
                    target="add disorder analysis",
                ),
            ]
        )
        normalized_context = routing_service._normalize_request_routing_draft(
            previous_plan_draft
        )
        normalized_context_actions = {
            item.action_type for item in normalized_context.actions
        }
        if normalized_context_actions != {
            "reference_historical_context",
            "run_tool_workflow",
        } or normalized_context.reuse_policy != "none" or (
            normalized_context.requests_recomputation
        ):
            raise AssertionError(
                "Historical plan context was incorrectly compiled as file reuse"
            )

        history_summary_draft = RequestRoutingDraft(
            actions=[
                RoutingDraftAction(
                    kind="summarize_history",
                    target="turn 1 energy report",
                )
            ]
        )
        normalized_history = routing_service._normalize_request_routing_draft(
            history_summary_draft
        )
        if (
            normalized_history.request_type != "history_analysis"
            or normalized_history.requires_tool_execution
            or normalized_history.reuse_policy != "none"
            or normalized_history.requests_recomputation
        ):
            raise AssertionError("Router draft normalization retained invalid flags")

        mixed_exit_draft = RequestRoutingDraft(
            actions=[
                RoutingDraftAction(
                    kind="exit",
                    target="exit now",
                ),
                RoutingDraftAction(
                    kind="run_workflow",
                    target="calculate energy",
                ),
            ]
        )
        normalized_exit = routing_service._normalize_request_routing_draft(
            mixed_exit_draft
        )
        if (
            normalized_exit.request_type != "clarification"
            or normalized_exit.requires_tool_execution
            or not normalized_exit.clarification_question
        ):
            raise AssertionError("Mixed exit request was not made safe")

        science_draft = RequestRoutingDraft(
            actions=[
                RoutingDraftAction(
                    kind="science_question",
                    target="why glycine increases flexibility",
                )
            ]
        )
        normalized_science = routing_service._normalize_request_routing_draft(
            science_draft
        )
        if (
            normalized_science.request_type != "general_scientific_question"
            or normalized_science.actions[0].action_type
            != "answer_general_question"
        ):
            raise AssertionError("Science action did not compile correctly")

        exit_draft = RequestRoutingDraft(
            actions=[
                RoutingDraftAction(
                    kind="exit",
                    target="end this session",
                )
            ]
        )
        normalized_session = routing_service._normalize_request_routing_draft(
            exit_draft
        )
        if normalized_session.request_type != "exit":
            raise AssertionError("Exit action did not compile correctly")

        listing_draft = RequestRoutingDraft(
            actions=[
                RoutingDraftAction(
                    kind="list_history_files",
                    target="exact source files from turn 1",
                )
            ]
        )
        normalized_listing = routing_service._normalize_request_routing_draft(
            listing_draft
        )
        if (
            normalized_listing.request_type != "artifact_query"
            or normalized_listing.actions[0].action_type != "list_artifacts"
            or normalized_listing.actions[0].history_operation != "list_artifacts"
        ):
            raise AssertionError("Historical file listing did not compile correctly")
        print("[PASS] Router draft normalization derives consistent strict fields.")

        routing_service._create_llm = lambda _state: FakeModel(
            error=RuntimeError("synthetic router failure")
        )
        failed = routing_service.route_user_request(state)
        if failed.routing_status != "failed" or failed.route != "request_to_analysis":
            raise AssertionError("Router failure was not converted to a safe result")
        print("[PASS] Router exceptions become auditable non-execution results.")

        history_route = routed_output(
            "artifact_query",
            ["list_artifacts"],
            history=True,
        )
        routing_service._create_llm = lambda _state: FakeModel(history_route)
        state["current_request_routing"] = routing_service.route_user_request(state)
        turn_one = HistoryTurnSelectionDraft(
            action_id=0,
            turn_scope="specific",
            turn_ids=[1],
        )
        turn_one_energy = HistoryArtifactSelectionDraft(
            action_id=0,
            artifact_ids=["t1:a2"],
        )
        routing_service._create_llm = lambda _state: SequenceFakeModel(
            [turn_one, turn_one_energy]
        )
        history = routing_service.plan_history_query(state)
        if history.query_status != "succeeded":
            raise AssertionError(f"Structured history selection failed: {history}")
        if [item.artifact_type for item in history.selected_artifacts] != [
            "energy_report_json"
        ]:
            raise AssertionError("History selection returned the wrong artifact type")
        if (
            history.query is None
            or history.query.operations != ["list_artifacts"]
            or history.query.report_types != ["enzywizard_energy"]
        ):
            raise AssertionError("History compiler did not derive query metadata")
        print("[PASS] Program compiled turn and catalog-ID selections exactly.")

        invalid_turn = HistoryTurnSelectionDraft(
            action_id=0,
            turn_scope="specific",
            turn_ids=[99],
        )
        routing_service._create_llm = lambda _state: FakeModel(invalid_turn)
        missing = routing_service.plan_history_query(state)
        if missing.query_status != "not_found" or missing.selected_artifacts:
            raise AssertionError("An unavailable turn was not rejected")
        print("[PASS] Unavailable turn IDs are rejected without guessing.")

        unknown_artifact = HistoryArtifactSelectionDraft(
            action_id=0,
            artifact_ids=["invented-id"],
        )
        empty_artifact = HistoryArtifactSelectionDraft(action_id=0, artifact_ids=[])
        routing_service._create_llm = lambda _state: SequenceFakeModel(
            [turn_one, unknown_artifact, empty_artifact]
        )
        missing_artifact = routing_service.plan_history_query(state)
        if missing_artifact.query_status != "not_found":
            raise AssertionError("Invalid catalog IDs were not corrected safely")
        print("[PASS] Unknown catalog IDs receive one bounded correction attempt.")

        empty_mixed_output = RequestRoutingOutput(
            request_type="mixed",
            actions=[
                RequestedAction(
                    action_type="analyze_history",
                    target="summarize a missing result from turn 1",
                    history_operation="summarize",
                ),
                RequestedAction(
                    action_type="list_artifacts",
                    target="list a missing file from turn 2",
                    history_operation="list_artifacts",
                ),
            ],
            requires_tool_execution=False,
            references_history=True,
            requires_history_artifacts=True,
            reuse_policy="none",
            requests_recomputation=False,
            clarification_question=None,
            routing_reason="Exercise two independent empty history selections.",
        )
        state["current_request_routing"] = routing_service.RequestRoutingResult(
            turn_id=3,
            routing_status="succeeded",
            route="request_to_history_query",
            output=empty_mixed_output,
            summary="Test empty mixed history route.",
            error_message="",
        )
        empty_mixed_model = SequenceFakeModel(
            [
                HistoryTurnSelectionDraft(
                    action_id=0,
                    turn_scope="specific",
                    turn_ids=[1],
                ),
                HistoryArtifactSelectionDraft(action_id=0, artifact_ids=[]),
                HistoryTurnSelectionDraft(
                    action_id=1,
                    turn_scope="specific",
                    turn_ids=[2],
                ),
                HistoryArtifactSelectionDraft(action_id=1, artifact_ids=[]),
            ]
        )
        routing_service._create_llm = lambda _state: empty_mixed_model
        empty_mixed = routing_service.plan_history_query(state)
        if (
            empty_mixed.query_status != "not_found"
            or empty_mixed.query is None
            or empty_mixed.query.turn_scope != "specific"
            or empty_mixed.query.referenced_turn_ids != [1, 2]
            or empty_mixed.resolved_turn_ids != [1, 2]
        ):
            raise AssertionError(
                "Empty multi-action history query lost its searched turn IDs: "
                f"{empty_mixed}"
            )
        print("[PASS] Empty multi-action queries remain valid and auditable.")

        # History Planner 不能丢弃 Router 已识别出的独立 list/reuse 操作。
        state["artifacts"].extend(
            [
                Artifact(
                    turn_id=2,
                    plan_id=1,
                    artifact_type="docked_complex_structure",
                    path="/tmp/turn2_complex.pdb",
                    source_tool="enzywizard-dock",
                    side="single",
                    metadata={},
                ),
                Artifact(
                    turn_id=2,
                    plan_id=1,
                    artifact_type="docked_substrate_sdf",
                    path="/tmp/turn2_docked_glucose.sdf",
                    source_tool="enzywizard-dock",
                    side="single",
                    metadata={},
                ),
                Artifact(
                    turn_id=2,
                    plan_id=1,
                    artifact_type="docked_substrate_sdf",
                    path="/tmp/turn2_docked_fructose.sdf",
                    source_tool="enzywizard-dock",
                    side="single",
                    metadata={},
                ),
            ]
        )
        docked_entries = [
            entry
            for entry in routing_service._history_catalog(state)
            if entry.artifact_type == "docked_substrate_sdf"
        ]
        if (
            len(docked_entries) != 2
            or len({entry.catalog_id for entry in docked_entries}) != 2
            or {entry.model_payload()["name"] for entry in docked_entries}
            != {"turn2_docked_glucose.sdf", "turn2_docked_fructose.sdf"}
        ):
            raise AssertionError("Per-file artifact catalog entries were not preserved")
        if any("/tmp/" in str(entry.model_payload()) for entry in docked_entries):
            raise AssertionError("Artifact catalog exposed an absolute path")
        reuse_output = RequestRoutingOutput(
            request_type="mixed",
            actions=[
                RequestedAction(
                    action_type="reuse_historical_output",
                    target="docked output from turn 2",
                    history_operation="reuse_as_input",
                ),
                RequestedAction(
                    action_type="run_tool_workflow",
                    target="interaction analysis",
                ),
            ],
            requires_tool_execution=True,
            references_history=True,
            requires_history_artifacts=True,
            reuse_policy="required",
            requests_recomputation=False,
            clarification_question=None,
            routing_reason="Reuse a completed docking output.",
        )
        state["current_request_routing"] = routing_service.RequestRoutingResult(
            turn_id=3,
            routing_status="succeeded",
            route="request_to_history_query",
            output=reuse_output,
            summary="Test reuse route.",
            error_message="",
        )
        reusable_entry = next(
            entry
            for entry in docked_entries
            if entry.model_payload()["name"] == "turn2_docked_glucose.sdf"
        )
        routing_service._create_llm = lambda _state: SequenceFakeModel(
            [
                HistoryTurnSelectionDraft(
                    action_id=0,
                    turn_scope="specific",
                    turn_ids=[2],
                ),
                HistoryArtifactSelectionDraft(
                    action_id=0,
                    artifact_ids=[reusable_entry.catalog_id],
                ),
            ]
        )
        reuse_history = routing_service.plan_history_query(state)
        if (
            reuse_history.query_status != "succeeded"
            or [
                artifact.artifact_type
                for artifact in reuse_history.selected_artifacts
            ]
            != ["docked_substrate_sdf"]
        ):
            raise AssertionError("Dock reuse did not resolve the reusable workflow input")
        print("[PASS] Historical reuse selects YAML-approved workflow inputs only.")

        graph = build_graph()
        edges = {(edge.source, edge.target) for edge in graph.get_graph().edges}
        required_router_edges = {
            ("user_input_interrupt", "request_routing"),
            ("user_confirmation_interrupt", "request_routing"),
        }
        if not required_router_edges.issubset(edges):
            raise AssertionError("A multi-turn or plan-revision path bypasses Router")
        print("[PASS] New turns and plan revisions both return through Router.")
    finally:
        routing_service._create_llm = original_create_llm

    print("\n[ALL PASSED] Request Router and History Query contracts are valid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
