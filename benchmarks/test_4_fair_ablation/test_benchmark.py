"""Offline regression checks for the Test 4 conversation and CDA."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from enzywizard_agent.data_models.request_routing import RequestRoutingOutput, RequestRoutingResult
from enzywizard_agent.services.knowledge_retrieval_service import enrich_messages_with_fair_terms
from langchain_core.messages import HumanMessage

from cases import load_suite
from compare_conditions import MATCHED, compare as compare_conditions
import conversation
from metrics import summarize
import run_test_4 as runner
from review_agreement import compare as review_agreement


class DefinitionBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog, cases, _ = load_suite()
        cls.cases = {case["case_id"]: case for case in cases}

    def test_binary_denominator_and_clustered_uncertainty(self):
        rows = [
            {"case_id": cid, "repetition": rep, "score": score}
            for rep, score in enumerate((0, 1, 0), 1) for cid in ("A", "B")
        ]
        stat = summarize(rows, ["A", "B"], 3, iterations=100, seed=42)
        self.assertAlmostEqual(stat["mean"], 1 / 3)
        self.assertAlmostEqual(stat["bootstrap_95_ci"]["lower"], 1 / 3)
        self.assertAlmostEqual(stat["bootstrap_95_ci"]["upper"], 1 / 3)
        rows[-1]["score"] = None
        stat = summarize(rows, ["A", "B"], 3, iterations=100, seed=42)
        self.assertFalse(stat["complete_reviews"])
        self.assertEqual(stat["scored_case_runs"], 5)
        self.assertAlmostEqual(stat["mean"], 1 / 3)
        self.assertEqual(stat["bootstrap_95_ci"], {"lower": 1 / 3, "upper": 1 / 3})
        rows[-2]["score"] = None
        stat = summarize(rows, ["A", "B"], 3, iterations=100, seed=42)
        self.assertNotIn("bootstrap_95_ci", stat)

    def test_invalid_judge_decision_is_retried_once(self):
        answer = {"answer": "A definition."}
        decision = {"correct": 0, "reason": "Incomplete."}
        with patch.object(runner.judge, "_review_once", side_effect=[ValueError("Invalid"), (decision, {})]) as request:
            self.assertEqual(runner.judge.review(answer, "fake", 5, save_raw=lambda data: None), (decision, {}))
        self.assertEqual(request.call_count, 2)
        self.assertEqual(request.call_args_list[0].args, request.call_args_list[1].args)

    def test_report_comparison_fixtures_read_raw_json(self):
        for case_id in ("T2-MC-01", "T2-MC-05", "T2-MC-06", "T2-MC-09"):
            with self.subTest(case=case_id):
                reports = conversation._reports(self.catalog, self.cases[case_id])
                summaries = [conversation._test_report_summary(artifact) for artifact in reports]
                if case_id == "T2-MC-06":
                    self.assertTrue(all("residue_72_embedding_coordinate_1" in summary.key_metrics
                                        for summary in summaries))
                else:
                    mutation = next(summary for summary in summaries
                                    if summary.source_tool == "enzywizard-mut-clean")
                    self.assertIn("wild_type_mutation_site_mapping", mutation.key_metrics)
                    self.assertIn("mutant_mutation_site_mapping", mutation.key_metrics)

    def test_paired_conditions_and_agreement(self):
        def document(mode, value):
            metadata = dict.fromkeys(MATCHED, "same")
            metadata.update(ontology_mode=mode, selected_case_ids=["A", "B"], repeat=3)
            rows = [
                {"case_id": cid, "repetition": rep, "score": value, "retrieval_audit": {
                     "mode": mode, "knowledge_base_sha256": "same", "total_injected_terms": 0,
                     "target_injected_into_definition_answer": False,
                     "target_injected_in_session": False, "control_count_matches": True,
                 }}
                for cid in ("A", "B") for rep in range(1, 4)
            ]
            return {"metadata": metadata, "results": rows}
        docs = {mode: document(mode, int(mode == "retrieve")) for mode in ("off", "control", "retrieve")}
        results = compare_conditions(docs, iterations=100, seed=42)
        self.assertEqual(results["CDA_differences"]["retrieve_minus_control"]["bootstrap_95_ci"],
                         {"lower": 1.0, "upper": 1.0})
        mismatched = deepcopy(docs)
        mismatched["control"]["metadata"]["model"] = "different"
        with self.assertRaises(ValueError):
            compare_conditions(mismatched, iterations=100, seed=42)

        for key, value, expected in (
            ("target_injected_in_session", True, r"A repeat 2 \(target term injected\)"),
            ("control_count_matches", False, r"A repeat 2 \(injection count does not match\)"),
        ):
            with self.subTest(audit_field=key):
                invalid = deepcopy(docs)
                invalid["control"]["results"][1]["retrieval_audit"][key] = value
                with self.assertRaisesRegex(ValueError, expected):
                    compare_conditions(invalid, iterations=100, seed=42)

        base = {"schema_version": 1, "review_id": "run", "repetition": 1,
                "sample_seed": 42, "sample_case_ids": ["A", "B"]}
        labels = {**base, "cases": [{"case_id": "A", "score": 1}, {"case_id": "B", "score": 0}]}
        agreement = review_agreement(labels, labels)
        self.assertEqual((agreement["agreement"], agreement["cohens_kappa"]), (1.0, 1.0))

    def test_normal_follow_up_history_and_ontology_mode(self):
        case = self.cases["T2-MF-07"]
        route = RequestRoutingResult(
            turn_id=3, routing_status="succeeded", route="request_to_analysis",
            summary="Definition question.", error_message="",
            output=RequestRoutingOutput(
                request_type="general_scientific_question",
                actions=[{"action_type": "answer_general_question", "target": case["target_term"]}],
                requires_tool_execution=False, references_history=False, requires_history_artifacts=False,
                requests_recomputation=False, reuse_policy="none", clarification_question=None,
                routing_reason="A concept definition was requested.",
            ),
        )

        def report_answer(state, _reports, *, historical):
            self.assertEqual(historical, state["current_turn_id"] == 2)
            self.assertEqual(len(state["analysis_results"]), state["current_turn_id"] - 1)
            if historical:
                self.assertEqual(state["current_history_query"].resolved_turn_ids, [1])
            enrich_messages_with_fair_terms(state, stage="analysis.scientific_narrative",
                                            messages=[HumanMessage(content="Report task.")])
            return conversation.analysis_service._empty_result(
                state, response_mode="scientific_result", summary="Prior report summary.",
                final_answer="Prior scientific report answer.",
            )

        def answer_definition(state, *, stage, llm, messages):
            self.assertEqual(stage, "analysis.general_answer")
            context = json.loads(messages[-1].content)
            self.assertEqual(context["user_request"], case["definition_question"])
            self.assertIn("Turn 2: Prior report summary.", context["prior_analysis_summary"])
            self.assertEqual([result.turn_id for result in state["analysis_results"]], [1, 2])
            enrich_messages_with_fair_terms(state, stage=stage, messages=messages)
            return case["reference_definition"]

        with TemporaryDirectory() as folder, patch.object(
            conversation.analysis_service, "_analyze_reports", side_effect=report_answer
        ), patch("enzywizard_agent.nodes.request_routing.route_user_request", return_value=route), patch.object(
            conversation.analysis_service, "invoke_text", side_effect=answer_definition
        ), patch.object(conversation.analysis_service, "_create_llm"), patch.object(
            runner.judge, "review", return_value=({"correct": 1, "reason": "Complete."}, {})
        ) as judge:
            for mode in ("off", "control", "retrieve"):
                with self.subTest(mode=mode):
                    args = SimpleNamespace(
                        api_key="test-agent", judge_api_key="test-judge", model="offline",
                        base_url="http://localhost", reasoning_effort="medium",
                        llm_timeout_seconds=5, judge_timeout_seconds=5, ontology_mode=mode,
                    )
                    workdir = Path(folder) / mode
                    row = runner.run_case(self.catalog, case, 1, args, workdir)
                    self.assertEqual(row["errors"], [])
                    self.assertEqual(row["score"], 1)
                    self.assertEqual(set(judge.call_args.args[0]), {
                        "target_term", "question", "reference_definition", "answer",
                    })
                    audit = row["retrieval_audit"]
                    self.assertEqual(audit["target_injected_into_definition_answer"], mode == "retrieve")
                    if mode == "off":
                        self.assertEqual(audit["total_injected_terms"], 0)
                    elif mode == "control":
                        self.assertTrue(audit["control_count_matches"])
                    case_dir = Path(row["case_workdir"])
                    saved = json.loads((case_dir / "conversation.json").read_text())
                    self.assertEqual(saved["ontology_mode"], mode)
                    self.assertEqual(saved["definition_turn"], 3)
                    self.assertEqual([m["turn_id"] for m in saved["user_messages"]], [1, 2, 3])
                    self.assertEqual([m["turn_id"] for m in saved["analysis_results"]], [1, 2, 3])

    def test_cli_writes_only_cda_and_two_blind_review_files(self):
        with TemporaryDirectory() as folder, patch.object(
            runner, "check_model_capabilities"
        ) as probe, patch.object(runner, "run_case") as case:
            probe.return_value.model_dump.return_value = {"fixture": True}
            case.side_effect = lambda _catalog, selected, repetition, *_: {
                "case_id": selected["case_id"], "repetition": repetition,
                "category_id": selected["category_id"], "score": 1, "errors": [],
                "packet": {"target_term": selected["target_term"], "answer": "Example answer."},
                "judge": {"correct": 1, "reason": "Correct."},
            }
            with patch.object(sys, "argv", [
                "run_test_4.py", "--workdir", folder, "--case_ids", "T2-SR-01",
                "--ontology_mode", "control", "--repeat", "3", "--bootstrap_iterations", "100",
                "--api_key", "test-agent", "--judge_api_key", "test-judge",
            ]):
                self.assertEqual(runner.main(), 0)
            run, = Path(folder).glob("run_*")
            result = json.loads((run / "results.json").read_text())
            self.assertEqual(set(result["summary"]["metrics"]), {"CDA"})
            self.assertEqual(result["summary"]["metrics"]["CDA"]["bootstrap_95_ci"], {"lower": 1.0, "upper": 1.0})
            sample = json.loads((run / "human_review.json").read_text())
            self.assertIsNone(sample["cases"][0]["score"])
            self.assertNotIn("ontology_mode", sample)
            self.assertTrue((run / "judge_review.json").is_file())
            self.assertTrue((run / "results.csv").is_file())


if __name__ == "__main__":
    unittest.main()
