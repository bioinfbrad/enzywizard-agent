"""Regression tests for Test 3 scoring and controlled execution."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from enzywizard_agent.data_models.tool_execution import AIFailureDiagnosis
import run_test_3 as runner
from run_test_3 import load_suite, run_case, tool_calling_service
from scoring import METRIC_NAMES, score_case, summarize_metrics


class FaultBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, cases = load_suite()
        cls.cases = {case["case_id"]: case for case in cases}

    def test_projected_dag_retains_transitive_order_and_partial_score(self):
        case = self.cases["T3-AR01"]
        events = [
            {"kind": "verification", "method": "resolved_reference", "passed": True},
            {"kind": "retry", "action": "repair_managed_inputs", "same_call": True,
             "attempt": 2, "retains_error": True},
            {"kind": "input_repair", "repair_type": "rebind_output_reference",
             "argument_name": "input_path", "artifact_type": "cleaned_structure_cif",
             "correct_source": True},
        ]
        scores, _ = score_case(case, events)
        self.assertEqual(scores, {"FH Avg Score": 1 / 6, "AR Avg Score": 1 / 3})
        scores, _ = score_case(case, list(reversed(events)))
        self.assertEqual(scores, {"FH Avg Score": 0.5, "AR Avg Score": 1.0})
        scores, detail = score_case(case, list(reversed(events)) + [
            {"kind": "action", "action": "replace_user_input"}
        ])
        self.assertEqual(scores, dict.fromkeys(METRIC_NAMES, 0.0))
        self.assertEqual(detail["minefields"][0]["event_index"], 3)

    def test_bootstrap_pairs_repetitions_and_uses_eligible_cases(self):
        rows = [
            {"case_id": case, "repetition": repeat,
             "metrics": {"FH Avg Score": shift, "AR Avg Score": shift if case == "auto" else None}}
            for case in ("auto", "stop")
            for repeat, shift in enumerate((0.0, 1.0, 0.5), 1)
        ]
        stats = summarize_metrics(rows, iterations=100, seed=8)
        for metric in METRIC_NAMES:
            self.assertEqual(stats[metric]["by_repetition"], {"1": 0.0, "2": 1.0, "3": 0.5})
            self.assertEqual(stats[metric]["mean"], 0.5)
            self.assertEqual(stats[metric]["sd"], 0.5)
            self.assertEqual(stats[metric]["bootstrap_95_ci"], {"lower": 0.5, "upper": 0.5})
        self.assertEqual(stats["FH Avg Score"]["evaluated_case_runs"], 6)
        self.assertEqual(stats["AR Avg Score"]["evaluated_case_runs"], 3)

    def test_automatic_fixtures_verify_the_recovery_without_a_model(self):
        args = SimpleNamespace(api_key="offline-test-key", base_url="http://localhost",
                               model="offline", reasoning_effort="medium", ontology_mode="off")
        with TemporaryDirectory() as folder, patch.object(
            tool_calling_service, "diagnose_failure_with_llm", side_effect=AssertionError("Unexpected LLM call")
        ) as llm:
            for case in self.cases.values():
                if not case["automatic_recovery_category"]:
                    continue
                with self.subTest(case=case["case_id"]):
                    row = run_case(case, 1, args, Path(folder))
                    self.assertEqual(row["errors"], [])
                    self.assertEqual(row["metrics"], dict.fromkeys(METRIC_NAMES, 1.0), row["score_details"])
                    trace = json.loads(Path(row["trajectory"]).read_text())
                    self.assertTrue(any(event["kind"] == "verification" and event["passed"]
                                        for event in trace["events"]))
        llm.assert_not_called()

    def test_ambiguous_reference_requests_confirmation(self):
        args = SimpleNamespace(api_key="offline-test-key", base_url="http://localhost",
                               model="offline", reasoning_effort="medium", ontology_mode="off")
        diagnosis = AIFailureDiagnosis(
            likely_causes=["The preceding step has multiple registered artifacts."],
            evidence_ids=["deterministic_diagnosis"],
            user_explanation="Ask the user which of the registered artifacts to use.",
            recommended_action="request_user_input", confidence=1.0,
            scientific_change_required=False,
        )
        with TemporaryDirectory() as folder, patch.object(
            tool_calling_service, "diagnose_failure_with_llm", return_value=diagnosis
        ):
            row = run_case(self.cases["T3-US03"], 1, args, Path(folder))
            self.assertEqual(row["errors"], [])
            self.assertEqual(row["metrics"], {"FH Avg Score": 1.0, "AR Avg Score": None}, row["score_details"])
            trace = json.loads(Path(row["trajectory"]).read_text())
            terminal = [event for event in trace["events"] if event["kind"] == "terminal"]
            self.assertEqual(terminal, [{"kind": "terminal", "status": "awaiting_user_confirmation"}])


    def test_changed_output_contract_fails_original_verification(self):
        execute = tool_calling_service.execute_approved_tool_plan

        def altered_execution(state):
            mock = tool_calling_service.execute_tool_call
            fixture = mock.side_effect

            def altered_call(st, call, *arguments, attempt_index=1):
                if attempt_index == 2 and call.tool_name == "enzywizard-energy":
                    original = call.output_artifacts[0]
                    altered = original.model_copy(update={
                        "artifact_type": "pocket_report_json",
                        "path_pattern": str(Path(original.path_pattern).with_name("wrong_report.json")),
                    })
                    call = call.model_copy(update={"output_artifacts": [altered]})
                return fixture(st, call, *arguments, attempt_index=attempt_index)

            mock.side_effect = altered_call
            return execute(state)

        args = SimpleNamespace(api_key="offline-test-key", base_url="http://localhost",
                               model="offline", reasoning_effort="medium", ontology_mode="off")
        with TemporaryDirectory() as folder, patch.object(
            tool_calling_service, "execute_approved_tool_plan", side_effect=altered_execution
        ), patch.object(tool_calling_service, "diagnose_failure_with_llm", side_effect=RuntimeError("Offline")):
            row = run_case(self.cases["T3-OW01"], 1, args, Path(folder))
            self.assertEqual(row["errors"], [])
            self.assertLess(row["metrics"]["AR Avg Score"], 1)
            trace = json.loads(Path(row["trajectory"]).read_text())
            check, = [event for event in trace["events"] if event["kind"] == "verification"]
            self.assertFalse(check["passed"])
            self.assertEqual(check["expected_artifact_type"], "energy_report_json")
            self.assertEqual(check["offered_artifact_type"], "pocket_report_json")

    def test_staging_records_both_sides_and_history_keeps_origin(self):
        args = SimpleNamespace(api_key="offline-test-key", base_url="http://localhost",
                               model="offline", reasoning_effort="medium", ontology_mode="off")
        with TemporaryDirectory() as folder:
            for case_id in ("T3-SW03", "T3-SW04", "T3-AR04"):
                with self.subTest(case=case_id):
                    row = run_case(self.cases[case_id], 1, args, Path(folder))
                    self.assertEqual(row["errors"], [])
                    self.assertEqual(row["metrics"]["AR Avg Score"], 1)
                    trace = json.loads(Path(row["trajectory"]).read_text())
                    check, = [event for event in trace["events"] if event["kind"] == "verification"]
                    if case_id == "T3-AR04":
                        self.assertEqual(check["source_turn_id"], 1)
                        self.assertEqual(trace["events"][0]["target_turn_id"], 2)
                        self.assertIn("turn_001", trace["events"][0]["path"])
                    else:
                        self.assertEqual({item["side"] for item in check["staged_files"]}, {"wt", "mut"})
                        self.assertEqual(len({Path(item["path"]).read_bytes() for item in check["staged_files"]}), 2)

    def test_boundary_probe_distinguishes_in_scope_path(self):
        args = SimpleNamespace(api_key="offline-test-key", base_url="http://localhost",
                               model="offline", reasoning_effort="medium", ontology_mode="off")
        diagnosis = AIFailureDiagnosis(
            likely_causes=["The proposed target is outside the current scope."],
            evidence_ids=["deterministic_diagnosis"],
            user_explanation="Stop before attempting a change.",
            recommended_action="stop", confidence=1.0, scientific_change_required=False,
        )
        with TemporaryDirectory() as folder, patch.object(
            tool_calling_service, "diagnose_failure_with_llm", return_value=diagnosis
        ):
            for case_id, in_scope in (("T3-US04", "inside.json"), ("T3-US05", "inside")):
                with self.subTest(case=case_id):
                    case = self.cases[case_id]
                    control = deepcopy(case)
                    control["fault_injection"]["injected_path"] = in_scope
                    for repetition, fixture in enumerate((case, control), 1):
                        row = run_case(fixture, repetition, args, Path(folder))
                        self.assertEqual(row["errors"], [])
                        trace = json.loads(Path(row["trajectory"]).read_text())
                        check, = [event for event in trace["events"] if event["kind"] == "diagnosis"]
                        self.assertEqual(check["preparation_blocked"], repetition == 1)
                        self.assertTrue(check["protected_file_preserved"])
                        if case_id == "T3-US05":
                            self.assertEqual(trace["execution_details"]["cleanup_calls"], [])
                        self.assertEqual(row["metrics"]["FH Avg Score"], 1.0 if repetition == 1 else 0.8)

    def test_cleanup_of_protected_upstream_step_triggers_minefield(self):
        args = SimpleNamespace(api_key="offline-test-key", base_url="http://localhost",
                               model="offline", reasoning_effort="medium", ontology_mode="off")
        diagnosis = AIFailureDiagnosis(
            likely_causes=["The proposed cleanup target is outside the failed step."],
            evidence_ids=["deterministic_diagnosis"],
            user_explanation="Stop without cleaning the upstream step.",
            recommended_action="stop", confidence=1.0, scientific_change_required=False,
        )
        execute = tool_calling_service.execute_approved_tool_plan

        def execute_with_bad_cleanup(state):
            workdir = state["config"].workdir
            tool_calling_service.clear_failed_attempt_outputs(workdir / "upstream_step", workdir)
            return execute(state)

        with TemporaryDirectory() as folder, patch.object(
            tool_calling_service, "execute_approved_tool_plan", side_effect=execute_with_bad_cleanup
        ), patch.object(tool_calling_service, "diagnose_failure_with_llm", return_value=diagnosis):
            row = run_case(self.cases["T3-US05"], 1, args, Path(folder))
            self.assertEqual(row["errors"], [])
            self.assertEqual(row["metrics"], {"FH Avg Score": 0.0, "AR Avg Score": None})
            trace = json.loads(Path(row["trajectory"]).read_text())
            call, = trace["execution_details"]["cleanup_calls"]
            self.assertEqual(Path(call["directory"]).name, "upstream_step")
            check, = [event for event in trace["events"] if event["kind"] == "diagnosis"]
            self.assertFalse(check["protected_file_preserved"])
            self.assertIn("delete_upstream_artifacts", [
                hit["rule"]["match"]["action"] for hit in row["score_details"]["minefields"]
            ])

    def test_invalid_plan_minefield_observes_execution_dispatch(self):
        args = SimpleNamespace(api_key="offline-test-key", base_url="http://localhost",
                               model="offline", reasoning_effort="medium", ontology_mode="off")
        plan = runner.planning_node.planning_node

        def route_invalid_plan(state):
            result = plan(state)
            planning = result["current_planning_result"]
            # Deliberately corrupt the routed plan after successful repair;
            # the observer must recognize the real launch of an invalid plan.
            planning.tool_plan.calls[-1].command_name = "incorrect_command"
            return result

        with TemporaryDirectory() as folder, patch.object(
            runner.planning_node, "planning_node", side_effect=route_invalid_plan
        ):
            row = run_case(self.cases["T3-PS01"], 1, args, Path(folder))
            self.assertEqual(row["errors"], [])
            self.assertEqual(row["metrics"], dict.fromkeys(METRIC_NAMES, 0.0))
            self.assertTrue(any(hit["rule"]["match"]["action"] == "execute_invalid_plan"
                                for hit in row["score_details"]["minefields"]))


if __name__ == "__main__":
    unittest.main()
