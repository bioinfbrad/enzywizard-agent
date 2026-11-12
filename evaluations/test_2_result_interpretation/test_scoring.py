from __future__ import annotations

import io
from pathlib import Path
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_test_2 import (
    FullReviewOutput,
    IISReviewOutput,
    _artifacts,
    _case_key_facts,
    _full_review_packet,
    _history_result,
    _iis_review_packet,
    _scientific_answer_body,
    human_review,
    llm_full_review,
    llm_iis_review,
    load_suite,
)
from scoring import (
    cluster_bootstrap_ci,
    score_case,
    score_case_full_review,
    validate_full_review,
    validate_iis_review,
)


def evaluation_packet() -> dict:
    return {
        "case_id": "T2-TEST",
        "category_id": "multi_report_integration",
        "question": "Integrate the reports.",
        "initial_question": None,
        "initial_answer": None,
        "candidate_answer": "Claim one. Claim two.",
        "candidate_limitations": [],
        "expected_behavior": None,
        "required_limitations": [],
        "forbidden_claims": [],
        "iis_requirements": {
            "must_address": ["integrate the relevant reports"],
            "required_limitations": [],
            "forbidden_claims": [],
            "optional_metric_policy": "Available metrics are not a checklist.",
        },
        "report_summaries": [],
        "candidate_evidence_registry": [
            {
                "evidence_id": "EV-1",
                "report_id": "report-a",
                "source_kind": "report_metric",
                "inference_level": "direct_observation",
                "source_tool": "enzywizard-energy",
                "artifact_type": "energy_report",
                "side": "single",
                "turn_id": 1,
                "field_path": "$.one",
                "value_summary": "1",
                "source_evidence_ids": [],
            }
        ],
        "atomic_facts": [
            {"fact_id": "F-1", "text": "one", "automatically_supported": True},
            {"fact_id": "F-2", "text": "two", "automatically_supported": False},
        ],
        "required_facts": [
            {
                "fact_id": "KF-1",
                "text": "required one",
                "report_id": "report-a",
                "metric_name": "metric-one",
                "expected_value": 1,
                "field_path": "$.one",
                "accepted_aliases": [],
                "supporting_report_ids": ["report-a"],
                "automatically_covered": True,
                "evidence_supported": True,
            },
            {
                "fact_id": "KF-2",
                "text": "required two",
                "report_id": "report-a",
                "metric_name": "metric-two",
                "expected_value": 2,
                "field_path": "$.two",
                "accepted_aliases": [],
                "supporting_report_ids": ["report-a"],
                "automatically_covered": False,
                "evidence_supported": False,
            },
        ],
        "iis_dimensions": [
            {"dimension_id": "d1", "label": "Dimension 1", "applicable": True},
            {"dimension_id": "d2", "label": "Dimension 2", "applicable": True},
            {"dimension_id": "d3", "label": "Dimension 3", "applicable": False},
        ],
        "evidence_associations": [
            {"association_id": "EA-1", "automatically_supported": True},
            {"association_id": "EA-2", "automatically_supported": False},
        ],
    }


def iis_review() -> dict:
    return {
        "evaluation_mode": "automatic_plus_human_iis",
        "interpretation_judgments": [
            {"dimension_id": "d1", "applicable": True, "score": 2, "reason": "ok"},
            {"dimension_id": "d2", "applicable": True, "score": 1, "reason": "partial"},
            {"dimension_id": "d3", "applicable": False, "score": None, "reason": "n/a"},
        ],
        "overall_notes": "test",
    }


def full_review() -> dict:
    return {
        "evaluation_mode": "full_review_human",
        "factual_judgments": [
            {"claim_id": "C-1", "claim": "Claim one.", "supported": True, "reason": "ok"},
            {"claim_id": "C-2", "claim": "Claim two.", "supported": False, "reason": "no"},
        ],
        "key_fact_judgments": [
            {"fact_id": "KF-1", "covered": True, "reason": "ok"},
            {"fact_id": "KF-2", "covered": False, "reason": "missing"},
        ],
        "evidence_association_judgments": [
            {
                "association_id": "CAND-EA-1",
                "claim": "Claim one.",
                "evidence_reference": "report-a $.one",
                "supported": True,
                "reason": "ok",
            }
        ],
        "evidence_coverage_judgments": [
            {"fact_id": "KF-1", "evidence_supported": True, "reason": "ok"},
            {"fact_id": "KF-2", "evidence_supported": False, "reason": "missing"},
        ],
        **{
            key: value
            for key, value in iis_review().items()
            if key in {"interpretation_judgments", "overall_notes"}
        },
    }


class TestScoring(unittest.TestCase):
    def test_judge_answer_excludes_terminology_source_appendix(self) -> None:
        answer = (
            "Scientific interpretation.\n\nTerminology sources "
            "(definitions only, not computational-result evidence):\n- source"
        )
        self.assertEqual(_scientific_answer_body(answer), "Scientific interpretation.")

    def test_four_metrics_follow_declared_formulas(self) -> None:
        metrics, details = score_case(evaluation_packet(), iis_review())
        self.assertEqual(metrics, {"FP": 0.5, "KFR": 0.5, "IIS": 0.75, "EA_F1": 0.5})
        self.assertEqual(details["EA_F1"]["precision"], 0.5)
        self.assertEqual(details["EA_F1"]["recall"], 0.5)

    def test_non_applicable_automatic_metrics_are_none(self) -> None:
        packet = evaluation_packet()
        packet["atomic_facts"] = []
        packet["required_facts"] = []
        packet["evidence_associations"] = []
        metrics, _ = score_case(packet, iis_review())
        self.assertIsNone(metrics["FP"])
        self.assertIsNone(metrics["KFR"])
        self.assertEqual(metrics["IIS"], 0.75)
        self.assertIsNone(metrics["EA_F1"])

    def test_boundary_cases_report_only_iis(self) -> None:
        packet = evaluation_packet()
        packet["category_id"] = "boundary_unrelated_analysis"
        metrics, details = score_case(packet, iis_review())
        self.assertEqual(metrics["IIS"], 0.75)
        self.assertIsNone(metrics["FP"])
        self.assertIsNone(metrics["KFR"])
        self.assertIsNone(metrics["EA_F1"])
        self.assertFalse(details["FP"]["applicable"])

        review = full_review()
        metrics, details = score_case_full_review(packet, review)
        self.assertEqual(metrics["IIS"], 0.75)
        self.assertIsNone(metrics["FP"])
        self.assertIsNone(metrics["KFR"])
        self.assertIsNone(metrics["EA_F1"])
        self.assertFalse(details["EA_F1"]["applicable"])

    def test_full_review_uses_item_level_judgments_and_declared_formulas(self) -> None:
        metrics, details = score_case_full_review(evaluation_packet(), full_review())
        self.assertEqual(metrics["FP"], 0.5)
        self.assertEqual(metrics["KFR"], 0.5)
        self.assertEqual(metrics["IIS"], 0.75)
        self.assertAlmostEqual(metrics["EA_F1"], 2 / 3)
        self.assertEqual(details["EA_F1"]["precision"], 1.0)
        self.assertEqual(details["EA_F1"]["recall"], 0.5)

    def test_full_review_rejects_missing_item_judgments(self) -> None:
        review = full_review()
        review["key_fact_judgments"].pop()
        errors = validate_full_review(evaluation_packet(), review)
        self.assertTrue(any("key_fact_judgments identifiers differ" in item for item in errors))

    def test_iis_contract_rejects_missing_dimension(self) -> None:
        review = iis_review()
        review["interpretation_judgments"].pop()
        errors = validate_iis_review(evaluation_packet(), review)
        self.assertTrue(any("identifiers differ" in item for item in errors))
        with self.assertRaises(ValueError):
            score_case(evaluation_packet(), review)

    def test_human_review_requests_only_iis_scores(self) -> None:
        answers = iter(["2 1"])
        with redirect_stdout(io.StringIO()):
            review = human_review(evaluation_packet(), input_fn=lambda _: next(answers))
        self.assertEqual(set(review), {"evaluation_mode", "interpretation_judgments", "overall_notes"})
        metrics, _ = score_case(evaluation_packet(), review)
        self.assertEqual(metrics["IIS"], 0.75)

    def test_human_review_displays_generated_structured_facts(self) -> None:
        packet = evaluation_packet()
        for index, item in enumerate(packet["required_facts"], start=1):
            item.pop("text")
            item.update(
                {
                    "report_id": "report-a",
                    "metric_name": f"metric-{index}",
                    "expected_value": index,
                    "field_path": f"$.metric_{index}",
                }
            )
        with redirect_stdout(io.StringIO()):
            review = human_review(packet, input_fn=lambda _: "2 2")
        self.assertEqual(review["evaluation_mode"], "automatic_plus_human_iis")

    def test_llm_reviewer_uses_the_same_iis_contract(self) -> None:
        output = IISReviewOutput.model_validate(
            {
                "interpretation_judgments": iis_review()["interpretation_judgments"],
                "overall_notes": "simulated human rubric",
            }
        )
        with patch("run_test_2.create_chat_model_for_state", return_value=object()), patch(
            "run_test_2.invoke_structured", return_value=output
        ) as invoke:
            review = llm_iis_review(evaluation_packet(), {})
        self.assertEqual(review["evaluation_mode"], "automatic_plus_llm_iis")
        self.assertFalse(validate_iis_review(evaluation_packet(), review))
        supplied = invoke.call_args.kwargs["messages"][1].content
        self.assertIn('"iis_requirements"', supplied)
        self.assertNotIn('"atomic_facts"', supplied)
        self.assertNotIn('"required_facts"', supplied)
        self.assertNotIn('"evidence_associations"', supplied)

    def test_iis_review_packet_excludes_automatic_gold_labels(self) -> None:
        review_packet = _iis_review_packet(evaluation_packet())
        self.assertIn("iis_requirements", review_packet)
        self.assertIn("report_summaries", review_packet)
        self.assertNotIn("atomic_facts", review_packet)
        self.assertNotIn("required_facts", review_packet)
        self.assertNotIn("evidence_associations", review_packet)

    def test_full_review_packet_includes_semantic_gold_items(self) -> None:
        review_packet = _full_review_packet(evaluation_packet())
        self.assertEqual(len(review_packet["kfr_key_facts"]), 2)
        self.assertEqual(
            review_packet["candidate_evidence_registry"][0]["evidence_id"],
            "EV-1",
        )
        self.assertEqual(
            review_packet["kfr_key_facts"][0]["supporting_report_ids"],
            ["report-a"],
        )
        self.assertNotIn("atomic_facts", review_packet)

    def test_llm_full_reviewer_uses_the_full_contract(self) -> None:
        output = FullReviewOutput.model_validate(
            {
                key: value
                for key, value in full_review().items()
                if key != "evaluation_mode"
            }
        )
        with patch("run_test_2.create_chat_model_for_state", return_value=object()), patch(
            "run_test_2.invoke_structured", return_value=output
        ):
            review = llm_full_review(evaluation_packet(), {})
        self.assertEqual(review["evaluation_mode"], "full_review_llm")
        self.assertFalse(validate_full_review(evaluation_packet(), review))

    def test_cluster_bootstrap_retains_repeated_case_records(self) -> None:
        results = [
            {
                "case_id": case,
                "repetition": repetition,
                "metrics": {name: value for name in ("FP", "KFR", "IIS", "EA_F1")},
            }
            for case, value in (("A", 0.0), ("B", 1.0))
            for repetition in (1, 2, 3)
        ]
        intervals = cluster_bootstrap_ci(results, iterations=200, seed=7)
        self.assertEqual(set(intervals), {"FP", "KFR", "IIS", "EA_F1"})
        self.assertLessEqual(intervals["FP"]["lower"], intervals["FP"]["upper"])


class TestFixtures(unittest.TestCase):
    def test_suite_contains_six_categories_and_68_cases(self) -> None:
        catalog, cases = load_suite()
        self.assertEqual(len(cases), 68)
        self.assertEqual(len(catalog["categories"]), 6)
        self.assertEqual(sum(item["case_count"] for item in catalog["categories"].values()), 68)
        gold_reports = catalog["_gold_standard"]["reports"]
        self.assertEqual(len(gold_reports), 45)
        self.assertEqual(
            sum(len(item["facts"]) for item in gold_reports.values()), 289
        )
        self.assertTrue(
            all(
                fact["field_path"] and fact["accepted_aliases"]
                for report in gold_reports.values()
                for fact in report["facts"]
            )
        )

    def test_paired_report_sides_and_follow_up_history_are_valid(self) -> None:
        catalog, cases = load_suite()
        case = next(item for item in cases if item["case_id"] == "T2-MF-12")
        artifacts = _artifacts(catalog, case)
        self.assertEqual({item.side for item in artifacts}, {"wt", "mut"})
        history = _history_result(artifacts)
        self.assertEqual(history.query_status, "succeeded")
        self.assertEqual(history.resolved_turn_ids, [1])
        self.assertEqual(history.query.operations, ["explain"])

    def test_each_scientific_case_has_compact_report_covering_key_facts(self) -> None:
        catalog, cases = load_suite()
        for case in cases:
            selected = _case_key_facts(catalog, case)
            if case["category_id"] == "boundary_unrelated_analysis":
                self.assertFalse(selected)
                continue
            self.assertGreaterEqual(len(selected), 3, case["case_id"])
            self.assertLessEqual(len(selected), 6, case["case_id"])
            covered = {report_id for report_id, _ in selected}
            required = {
                report_id
                for report_id in case["report_ids"]
                if catalog["reports"][report_id]["source_tool"]
                != "enzywizard-mut-clean"
            }
            self.assertTrue(required.issubset(covered), case["case_id"])

            analysis_reports = [
                report_id
                for report_id in case["report_ids"]
                if catalog["reports"][report_id]["source_tool"]
                != "enzywizard-mut-clean"
            ]
            if len(analysis_reports) == 1 and not any(
                fact.get("synthetic_case_fact") for _, fact in selected
            ):
                self.assertEqual(len(selected), 3, case["case_id"])

    def test_mutation_identity_and_nonredundant_substrate_facts_are_selected(self) -> None:
        catalog, cases = load_suite()
        comparison = next(item for item in cases if item["case_id"] == "T2-MC-01")
        comparison_metrics = {
            fact["metric_name"] for _, fact in _case_key_facts(catalog, comparison)
        }
        self.assertIn("amino_acid_substitution", comparison_metrics)
        self.assertTrue(
            any(metric.startswith("comparison::") for metric in comparison_metrics)
        )

        comparison_facts = [
            fact
            for _, fact in _case_key_facts(catalog, comparison)
            if fact.get("derived_comparison_fact")
        ]
        self.assertEqual(len(comparison_facts), 1)
        self.assertEqual(len(comparison_facts[0]["supporting_report_ids"]), 2)
        self.assertTrue(
            all(
                report_id in comparison["report_ids"]
                for report_id in comparison_facts[0]["supporting_report_ids"]
            )
        )

        substrate = next(item for item in cases if item["case_id"] == "T2-SR-08")
        substrate_metrics = {
            fact["metric_name"] for _, fact in _case_key_facts(catalog, substrate)
        }
        self.assertIn("substrate_molecular_weight_mean", substrate_metrics)
        self.assertNotIn("substrate_molecular_weight_min", substrate_metrics)
        self.assertNotIn("substrate_molecular_weight_max", substrate_metrics)


if __name__ == "__main__":
    unittest.main()
