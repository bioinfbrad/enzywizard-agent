"""Offline contract tests for the frozen publication benchmark."""

from __future__ import annotations

from pathlib import Path
from collections import Counter
import unittest

from enzywizard_agent.builders.planning_context_builder import build_planning_context
from enzywizard_agent.data_models.agent_inputs import AgentInputs
from enzywizard_agent.data_models.artifact import Artifact
from enzywizard_agent.data_models.history_query import (
    HistoryQueryOutput,
    HistoryQueryResult,
)
from enzywizard_agent.data_models.planning import (
    ParameterExtractionOutput,
    PlanningResult,
    UnsupportedParameterRequest,
    WorkflowRequirementsOutput,
)
from enzywizard_agent.data_models.request_routing import (
    RequestedAction,
    RequestRoutingOutput,
)
from enzywizard_agent.data_models.tool_execution import ToolCallExecution
from enzywizard_agent.services.request_routing_service import (
    _apply_required_input_preflight,
    _explicit_turn_ids_for_action,
    _extract_explicit_turn_ids,
)
from enzywizard_agent.services.planning_service import (
    _compile_workflow_requirements,
    _normalize_resolved_history_input_requests,
    _remove_history_only_analyses,
)
from enzywizard_agent.services.analysis_service import (
    _deterministic_scientific_answer,
    _load_report_cautions,
)
from enzywizard_agent.services.tool_recovery_service import diagnose_tool_failure
from enzywizard_agent.data_models.analysis import ReportSummary
from enzywizard_agent.utils.analysis_utils import build_report_evidence

from publication_benchmark_utils import (
    AnalysisBenchmarkCase,
    BENCHMARK_ROOT,
    RecoveryBenchmarkCase,
    WorkflowBenchmarkCase,
    load_case_file,
    metric_statistics,
    resolve_project_path,
    select_profile_cases,
)
from run_publication_test_2_analysis import build_summary as build_analysis_summary
from run_publication_test_1_workflows import build_summary as build_workflow_summary
from run_full_workflow_tests import evaluate_plan_payload


class PublicationBenchmarkContracts(unittest.TestCase):
    def test_fixed_case_counts_and_unique_ids(self) -> None:
        workflow = load_case_file(
            BENCHMARK_ROOT / "test_1_workflow_cases.yaml",
            WorkflowBenchmarkCase,
            72,
        )
        analysis = load_case_file(
            BENCHMARK_ROOT / "test_2_analysis_cases.yaml",
            AnalysisBenchmarkCase,
            48,
        )
        recovery = load_case_file(
            BENCHMARK_ROOT / "test_3_recovery_cases.yaml",
            RecoveryBenchmarkCase,
            40,
        )
        identifiers = [item.case_id for item in [*workflow, *analysis, *recovery]]
        self.assertEqual(len(identifiers), 160)
        self.assertEqual(len(set(identifiers)), 160)

        self.assertEqual(Counter(item.subtype for item in workflow), {"workflow": 66, "history": 6})
        analysis_counts = Counter(item.subtype for item in analysis)
        self.assertGreaterEqual(min(analysis_counts.values()), 6)
        recovery_counts = Counter(item.subtype for item in recovery)
        self.assertGreaterEqual(min(recovery_counts.values()), 2)
        self.assertEqual(sum(item.scenario == "preflight" for item in recovery), 4)

    def test_primary_metric_sets_are_fixed(self) -> None:
        workflow_results = [
            {
                "case_id": "workflow",
                "passed": True,
                "execution_required": True,
                "execution_requested": True,
                "execution_completed": True,
                "task_success": True,
                "metrics": {
                    "tool_selection_accuracy": 1.0,
                    "parameter_correctness_validity": 1.0,
                    "workflow_order_validity": 1.0,
                    "required_artifact_completeness": 1.0,
                    "end_to_end_task_success": 1.0,
                },
            }
        ]
        self.assertEqual(
            set(build_workflow_summary(workflow_results, "execute_all")["metrics"]),
            {
                "tool_selection_accuracy",
                "parameter_correctness_validity",
                "workflow_order_validity",
                "required_artifact_completeness",
                "end_to_end_task_success",
            },
        )

    def test_quick_profile_is_reproducible_and_stratified(self) -> None:
        cases = load_case_file(
            BENCHMARK_ROOT / "test_2_analysis_cases.yaml",
            AnalysisBenchmarkCase,
            48,
        )
        first = select_profile_cases(
            cases,
            case_ids=None,
            profile="quick",
            sample_per_subtype=2,
            seed=17,
        )
        second = select_profile_cases(
            cases,
            case_ids=None,
            profile="quick",
            sample_per_subtype=2,
            seed=17,
        )
        self.assertEqual(
            [item.case_id for item in first],
            [item.case_id for item in second],
        )
        self.assertEqual(Counter(item.subtype for item in first), {
            subtype: 2 for subtype in {item.subtype for item in cases}
        })

    def test_quick_profile_supports_an_exact_total_and_variable_seed(self) -> None:
        cases = load_case_file(
            BENCHMARK_ROOT / "test_1_workflow_cases.yaml",
            WorkflowBenchmarkCase,
            72,
        )
        first = select_profile_cases(
            cases,
            case_ids=None,
            profile="quick",
            sample_per_subtype=4,
            quick_case_count=9,
            seed=1,
        )
        second = select_profile_cases(
            cases,
            case_ids=None,
            profile="quick",
            sample_per_subtype=4,
            quick_case_count=9,
            seed=2,
        )
        self.assertEqual(len(first), 9)
        self.assertEqual(len(second), 9)
        self.assertNotEqual(
            [item.case_id for item in first],
            [item.case_id for item in second],
        )

    def test_quick_profile_preserves_the_documented_seed_sample(self) -> None:
        cases = load_case_file(
            BENCHMARK_ROOT / "test_1_workflow_cases.yaml",
            WorkflowBenchmarkCase,
            72,
        )
        selected = select_profile_cases(
            cases,
            case_ids=None,
            profile="quick",
            sample_per_subtype=4,
            quick_case_count=8,
            seed=20261026,
        )
        self.assertEqual(
            [item.case_id for item in selected],
            [
                "T1-W17",
                "T1-W35",
                "T1-W59",
                "T1-W64",
                "T1-H03",
                "T1-H04",
                "T1-H05",
                "T1-H06",
            ],
        )

    def test_metric_sd_is_calculated_between_repetitions(self) -> None:
        results = [
            {
                "repetition": repetition,
                "metrics": {"score": value},
            }
            for repetition, values in ((1, (1.0, 0.0)), (2, (1.0, 1.0)))
            for value in values
        ]
        summary = metric_statistics(results)["score"]
        self.assertEqual(summary["mean"], 0.75)
        self.assertAlmostEqual(summary["sd"], 0.353553, places=6)
        self.assertEqual(summary["n"], 4)
        self.assertEqual(summary["repetition_n"], 2)

    def test_resolved_history_reference_is_not_a_path_override(self) -> None:
        artifact = Artifact(
            turn_id=1,
            plan_id=1,
            artifact_type="cleaned_structure_cif",
            path="/tmp/cleaned.cif",
            source_tool="enzywizard-clean",
            side="single",
            metadata={},
        )
        query = HistoryQueryOutput(
            turn_scope="specific",
            referenced_turn_ids=[1],
            artifact_categories=["clean"],
            artifact_types=["cleaned_structure_cif"],
            report_types=[],
            sides=["single"],
            operations=["reuse_as_input"],
            query_reason="test",
            clarification_question=None,
        )
        state = {
            "current_history_query": HistoryQueryResult(
                turn_id=2,
                query_status="succeeded",
                query=query,
                resolved_turn_ids=[1],
                selected_artifacts=[artifact],
                summary="test",
                error_message="",
            )
        }
        extraction = ParameterExtractionOutput(
            parameter_extraction_status="parameter_extraction_succeeded",
            parameter_adjustments=[],
            profile_requests=[],
            unsupported_requests=[
                UnsupportedParameterRequest(
                    step_index=1,
                    tool_name="enzywizard-flexibility",
                    parameter_name="input_path",
                    requested_value="Use the cleaned structure from turn 1",
                    request_type="input_path_override",
                )
            ],
            extraction_summary="test",
            error_message="",
        )
        normalized = _normalize_resolved_history_input_requests(state, extraction)
        self.assertEqual(normalized.unsupported_requests, [])

        explicit_path = extraction.model_copy(deep=True)
        explicit_path.unsupported_requests[0].requested_value = "/tmp/other.cif"
        preserved = _normalize_resolved_history_input_requests(state, explicit_path)
        self.assertEqual(len(preserved.unsupported_requests), 1)

    def test_execution_gate_accepts_any_valid_topological_order(self) -> None:
        expected = {
            "required_tools": [
                "enzywizard-clean",
                "enzywizard-energy",
                "enzywizard-substrate",
                "enzywizard-dock",
                "enzywizard-integrate",
            ],
            "allowed_tools": [
                "enzywizard-clean",
                "enzywizard-energy",
                "enzywizard-substrate",
                "enzywizard-dock",
                "enzywizard-integrate",
            ],
            "order_constraints": [
                ["enzywizard-clean", "enzywizard-energy"],
                ["enzywizard-substrate", "enzywizard-dock"],
                ["enzywizard-energy", "enzywizard-integrate"],
                ["enzywizard-dock", "enzywizard-integrate"],
            ],
        }
        payload = {
            "interrupt_type": "tool_plan_confirmation",
            "turn_id": 1,
            "plan_id": 1,
            "calls": [
                {"tool_name": name}
                for name in (
                    "enzywizard-clean",
                    "enzywizard-substrate",
                    "enzywizard-energy",
                    "enzywizard-dock",
                    "enzywizard-integrate",
                )
            ],
        }
        self.assertEqual(evaluate_plan_payload(payload, expected, 1), [])

        payload["calls"][1], payload["calls"][3] = (
            payload["calls"][3],
            payload["calls"][1],
        )
        self.assertTrue(evaluate_plan_payload(payload, expected, 1))

    def test_mixed_history_request_isolated_from_new_workflow(self) -> None:
        routing_output = RequestRoutingOutput(
            request_type="mixed",
            actions=[
                RequestedAction(
                    action_type="analyze_history",
                    target="energy reports from turn 2",
                    history_operation="compare",
                ),
                RequestedAction(
                    action_type="run_tool_workflow",
                    target="new binding-pocket analysis",
                ),
            ],
            requires_tool_execution=True,
            references_history=True,
            requires_history_artifacts=True,
            reuse_policy="none",
            requests_recomputation=False,
            clarification_question=None,
            routing_reason="test",
        )
        routing = type("Routing", (), {"output": routing_output})()
        artifact = Artifact(
            turn_id=2,
            plan_id=1,
            artifact_type="energy_report_json",
            path="/tmp/energy.json",
            source_tool="enzywizard-energy",
            side="single",
            metadata={},
        )
        query = HistoryQueryOutput(
            turn_scope="specific",
            referenced_turn_ids=[2],
            artifact_categories=["energy"],
            artifact_types=["energy_report_json"],
            report_types=["enzywizard_energy"],
            sides=["single"],
            operations=["compare"],
            query_reason="test",
            clarification_question=None,
        )
        history = HistoryQueryResult(
            turn_id=4,
            query_status="succeeded",
            query=query,
            resolved_turn_ids=[2],
            selected_artifacts=[artifact],
            summary="test",
            error_message="",
        )
        state = {
            "current_user_message": (
                "Compare energy reports from turn 2, then run a new "
                "binding-pocket analysis."
            ),
            "current_request_routing": routing,
            "current_history_query": history,
            "planning_results": [],
        }
        context = build_planning_context(state)
        self.assertIn(
            "Current new-computation request:\n- new binding-pocket analysis",
            context,
        )
        requirements = WorkflowRequirementsOutput(
            workflow_requirements_status="workflow_requirements_succeeded",
            scope_mode="specific_analyses",
            requested_analysis_types=["energy", "pocket"],
            excluded_analysis_types=[],
            requirements_reason="test",
            error_message="",
        )
        normalized = _remove_history_only_analyses(state, requirements)
        self.assertEqual(normalized.requested_analysis_types, ["pocket"])

    def test_workflow_oracles_follow_integration_contract(self) -> None:
        cases = load_case_file(
            BENCHMARK_ROOT / "test_1_workflow_cases.yaml",
            WorkflowBenchmarkCase,
            72,
        )
        preparation_tools = {
            "enzywizard-clean",
            "enzywizard-mut-clean",
            "enzywizard-substrate",
        }
        integration_tools = {
            "enzywizard-integrate",
            "enzywizard-mut-integrate",
        }
        batch_tools = {"enzywizard-batch", "enzywizard-mut-batch"}

        for case in cases:
            alternatives = [
                alternative
                for action in case.confirmation_actions
                for alternative in action.acceptable_plans
            ]
            if case.history_expectation is not None:
                alternatives.extend(case.history_expectation.acceptable_plans)
            for alternative in alternatives:
                with self.subTest(case=case.case_id, plan=alternative.name):
                    tools = set(alternative.expected_tools)
                    individual_tools = tools - preparation_tools - integration_tools - batch_tools
                    if not individual_tools or tools & batch_tools:
                        continue
                    expected_integration = (
                        "enzywizard-mut-integrate"
                        if "enzywizard-mut-clean" in tools
                        else "enzywizard-integrate"
                    )
                    self.assertIn(expected_integration, tools)

    def test_all_analysis_reports_exist(self) -> None:
        cases = load_case_file(
            BENCHMARK_ROOT / "test_2_analysis_cases.yaml",
            AnalysisBenchmarkCase,
            48,
        )
        missing = [
            resolve_project_path(spec.path)
            for case in cases
            for spec in case.artifacts
            if not resolve_project_path(spec.path).is_file()
        ]
        self.assertEqual(missing, [])

    def test_analysis_reports_quality_without_subjective_strict_pass(self) -> None:
        results = [
            {
                "case_id": "stable-pass",
                "passed": True,
                "technical_completed": True,
                "evaluation_completed": True,
                "metrics": {
                    "factual_precision": 0.9,
                    "key_fact_recall": 0.8,
                    "integrated_interpretation_score": 0.9,
                    "evidence_attribution_f1": 1.0,
                },
                "judge_result": {"identified_required_facts": [text]},
            }
            for text in ("first wording", "second wording", "third wording")
        ]
        results.extend(
            {
                "case_id": "unstable",
                "passed": True,
                "technical_completed": True,
                "evaluation_completed": True,
                "metrics": {
                    "factual_precision": quality,
                    "key_fact_recall": quality,
                    "integrated_interpretation_score": quality,
                    "evidence_attribution_f1": 1.0,
                },
                "judge_result": {"identified_required_facts": ["same wording"]},
            }
            for quality in (0.9, 0.7, 0.8)
        )
        summary = build_analysis_summary(results)
        self.assertNotIn("conclusion_stability", summary)
        self.assertNotIn("run_pass_rate", summary)
        self.assertEqual(summary["technical_completion_rate"], 1.0)
        self.assertNotIn("secondary_judge_assessment", summary)
        self.assertGreater(
            summary["repeated_metric_variability"][
                "integrated_interpretation_score"
            ]["max_within_case_sd"],
            0.0,
        )

    def test_workflow_summary_reports_execution_coverage_explicitly(self) -> None:
        results = [
            {
                "case_id": "workflow",
                "passed": True,
                "execution_required": True,
                "execution_requested": requested,
                "execution_completed": requested,
                "task_success": success,
                "metrics": {
                    "tool_selection_accuracy": 1.0,
                    "parameter_correctness_validity": 1.0,
                    "workflow_order_validity": 1.0,
                    "required_artifact_completeness": success,
                    "end_to_end_task_success": success,
                },
            }
            for requested, success in ((True, True), (False, None), (False, None))
        ]
        summary = build_workflow_summary(results, "execute_representative")
        self.assertEqual(summary["execution_required_runs"], 3)
        self.assertEqual(summary["execution_requested_runs"], 1)
        self.assertEqual(summary["execution_attempted_runs"], 1)
        self.assertAlmostEqual(summary["execution_coverage_rate"], 1 / 3, places=6)
        self.assertEqual(summary["execution_success_rate"], 1.0)
        self.assertEqual(summary["objective_task_evaluated_runs"], 1)

    def test_report_fallback_does_not_claim_execution_success(self) -> None:
        answer = _deterministic_scientific_answer(
            {},
            findings=[],
            evidence=[],
            historical=False,
        )
        self.assertIn("Analysis of the available reports", answer)
        self.assertNotIn("completed successfully", answer)

    def test_pocket_reports_have_a_stable_interpretation_caution(self) -> None:
        cautions = _load_report_cautions({"enzywizard_pocket"})
        self.assertEqual(len(cautions), 1)
        self.assertIn("require experimental or orthogonal validation", cautions[0])

    def test_calculated_comparisons_reference_both_source_reports(self) -> None:
        summaries = [
            ReportSummary(
                turn_id=1,
                plan_id=1,
                artifact_type="energy_report_json",
                report_type="enzywizard_energy",
                source_tool="enzywizard-energy",
                side=side,
                path=f"/{side}.json",
                key_metrics={"total_potential_energy": value},
                highlights=[f"total_potential_energy={value}"],
                warnings=[],
            )
            for side, value in (("wt", -10.0), ("mut", -12.0))
        ]
        evidence, _ = build_report_evidence(summaries)
        comparison = next(
            item for item in evidence if item.evidence_id.startswith("comparison-")
        )
        self.assertEqual(comparison.source_evidence_ids, ["report-1", "report-2"])

    def test_missing_input_preflight_is_deterministic(self) -> None:
        output = RequestRoutingOutput(
            request_type="tool_workflow",
            actions=[
                RequestedAction(
                    action_type="run_tool_workflow", target="energy analysis"
                )
            ],
            requires_tool_execution=True,
            references_history=False,
            requires_history_artifacts=False,
            reuse_policy="none",
            requests_recomputation=False,
            clarification_question=None,
            routing_reason="test",
        )
        inputs = AgentInputs(
            structure_file=None,
            msa_file=None,
            mut_structure_file=None,
            mut_msa_file=None,
            substrate_name=None,
            amino_acid_substitution=None,
        )
        result = _apply_required_input_preflight(
            {"current_user_message": "Analyze energy", "agent_inputs": inputs},
            output,
        )
        self.assertEqual(result.request_type, "clarification")
        self.assertFalse(result.requires_tool_execution)

    def test_preflight_preserves_independent_history_action(self) -> None:
        output = RequestRoutingOutput(
            request_type="mixed",
            actions=[
                RequestedAction(
                    action_type="analyze_history",
                    target="prior energy report",
                    history_operation="summarize",
                ),
                RequestedAction(
                    action_type="run_tool_workflow", target="new energy analysis"
                ),
            ],
            requires_tool_execution=True,
            references_history=True,
            requires_history_artifacts=True,
            reuse_policy="none",
            requests_recomputation=False,
            clarification_question=None,
            routing_reason="test",
        )
        inputs = AgentInputs(
            structure_file=None,
            msa_file=None,
            mut_structure_file=None,
            mut_msa_file=None,
            substrate_name=None,
            amino_acid_substitution=None,
        )
        result = _apply_required_input_preflight(
            {
                "current_user_message": (
                    "Summarize the prior energy report and run new energy analysis"
                ),
                "agent_inputs": inputs,
            },
            output,
        )
        self.assertEqual(result, output)

    def test_preflight_does_not_infer_conditional_inputs_from_words(self) -> None:
        output = RequestRoutingOutput(
            request_type="tool_workflow",
            actions=[
                RequestedAction(
                    action_type="run_tool_workflow", target="pocket analysis"
                )
            ],
            requires_tool_execution=True,
            references_history=False,
            requires_history_artifacts=False,
            reuse_policy="none",
            requests_recomputation=False,
            clarification_question=None,
            routing_reason="test",
        )
        inputs = AgentInputs(
            structure_file="protein.pdb",
            msa_file=None,
            mut_structure_file=None,
            mut_msa_file=None,
            substrate_name=None,
            amino_acid_substitution=None,
        )
        for request in (
            "Analyze internal interactions without a substrate.",
            "Locate a cavity that could accommodate a ligand.",
        ):
            with self.subTest(request=request):
                result = _apply_required_input_preflight(
                    {"current_user_message": request, "agent_inputs": inputs},
                    output,
                )
                self.assertEqual(result, output)

    def test_planning_requests_only_semantically_required_inputs(self) -> None:
        planning = PlanningResult(
            turn_id=1,
            plan_id=1,
            planning_route="tool_unused",
            planning_flag="intent_recognized",
            needs_tool=True,
            summary=None,
            purpose_type="single_protein_analysis",
            purpose_reason="test",
            tool_selection=None,
            parameter_extraction=None,
            tool_plan=None,
            error_messages=[],
        )
        inputs = AgentInputs(
            structure_file="protein.pdb",
            msa_file=None,
            mut_structure_file=None,
            mut_msa_file=None,
            substrate_name=None,
            amino_acid_substitution=None,
        )
        state = {"agent_inputs": inputs}

        pocket = WorkflowRequirementsOutput(
            workflow_requirements_status="workflow_requirements_succeeded",
            scope_mode="specific_analyses",
            requested_analysis_types=["pocket"],
            excluded_analysis_types=[],
            requirements_reason="test",
            error_message="",
        )
        pocket_selection = _compile_workflow_requirements(
            state,
            planning,
            pocket,
            ["enzywizard-clean"],
            {"enzywizard-pocket"},
        )
        self.assertEqual(
            pocket_selection.tool_selection_status, "tool_selection_succeeded"
        )

        docking = pocket.model_copy(
            update={"requested_analysis_types": ["substrate", "docking"]}
        )
        docking_selection = _compile_workflow_requirements(
            state,
            planning,
            docking,
            ["enzywizard-clean"],
            {"enzywizard-substrate", "enzywizard-dock"},
        )
        self.assertEqual(docking_selection.tool_selection_status, "input_required")
        self.assertEqual(
            docking_selection.missing_inputs, ["a substrate name or SMILES"]
        )

    def test_explicit_history_turns_are_deterministic(self) -> None:
        self.assertEqual(_extract_explicit_turn_ids("use turn 3"), [3])
        self.assertEqual(_extract_explicit_turn_ids("compare turns 1 and 2"), [1, 2])
        action = RequestedAction(
            action_type="reuse_historical_output",
            target="the cleaned structure for a new analysis",
            history_operation="reuse_as_input",
        )
        self.assertEqual(
            _explicit_turn_ids_for_action(
                "List outputs from turn 1 and reuse the cleaned structure.",
                action,
                history_action_count=2,
            ),
            [1],
        )

    def test_recovery_oracles_match_deterministic_classifier(self) -> None:
        cases = load_case_file(
            BENCHMARK_ROOT / "test_3_recovery_cases.yaml",
            RecoveryBenchmarkCase,
            40,
        )
        for case in cases:
            if case.scenario == "preflight":
                continue
            with self.subTest(case=case.case_id):
                call = ToolCallExecution(
                    turn_id=1,
                    plan_id=1,
                    step_index=1,
                    tool_name="enzywizard-energy",
                    side="single",
                    execution_status="failed",
                    command=["enzywizard-energy"] if case.command_present else [],
                    resolved_arguments={},
                    exit_code=1,
                    duration_seconds=0.0,
                    stdout_file=str(Path("stdout.txt")),
                    stderr_file=str(Path("stderr.txt")),
                    log_files=[],
                    stdout_excerpt="",
                    stderr_excerpt=case.error_message,
                    log_excerpt="",
                    artifacts=[],
                    error_message=case.error_message,
                )
                diagnosis = diagnose_tool_failure(call)
                self.assertEqual(diagnosis.category, case.expected_category)
                self.assertEqual(
                    diagnosis.recoverability, case.expected_recoverability
                )


if __name__ == "__main__":
    unittest.main()
