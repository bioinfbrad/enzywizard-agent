from __future__ import annotations

import argparse
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_test_4 import (
    _case_terminology_items,
    _iis_review_packet,
    _preflight_retrieve_cases,
    _review_terminology_items,
    _scientific_answer_body,
    load_suite,
)
from compare_conditions import compare
from scoring import (
    METRIC_NAMES,
    cluster_bootstrap_ci,
    repetition_statistics,
    score_case,
    score_case_full_review,
    validate_iis_review,
)


def packet() -> dict:
    return {
        "case_id": "T5-TEST",
        "category_id": "multi_report_integration",
        "atomic_facts": [
            {"automatically_supported": True},
            {"automatically_supported": False},
        ],
        "required_facts": [
            {
                "fact_id": "KF-1",
                "automatically_covered": True,
                "evidence_supported": True,
            },
            {
                "fact_id": "KF-2",
                "automatically_covered": False,
                "evidence_supported": False,
            },
        ],
        "evidence_associations": [
            {"automatically_supported": True},
            {"automatically_supported": False},
        ],
        "iis_dimensions": [
            {"dimension_id": "d1", "applicable": True},
            {"dimension_id": "d2", "applicable": True},
            {"dimension_id": "d3", "applicable": False},
        ],
        "terminology_items": [
            {
                "term_id": "binding_pocket",
                "label": "binding pocket",
                "accepted_interpretation": "A computationally identified cavity.",
                "forbidden_interpretations": ["It proves binding."],
            },
            {
                "term_id": "molecular_docking",
                "label": "molecular docking",
                "accepted_interpretation": "A modeled pose and score.",
                "forbidden_interpretations": ["It measures affinity."],
            },
        ],
    }


def review() -> dict:
    return {
        "evaluation_mode": "automatic_plus_human_iis_tsia",
        "interpretation_judgments": [
            {"dimension_id": "d1", "applicable": True, "score": 2},
            {"dimension_id": "d2", "applicable": True, "score": 1},
            {"dimension_id": "d3", "applicable": False, "score": None},
        ],
        "terminology_judgments": [
            {"term_id": "binding_pocket", "understood": True},
            {"term_id": "molecular_docking", "understood": False},
        ],
    }


def full_review() -> dict:
    return {
        **review(),
        "evaluation_mode": "full_review_human",
        "factual_judgments": [
            {"claim_id": "C-1", "claim": "one", "supported": True},
            {"claim_id": "C-2", "claim": "two", "supported": False},
        ],
        "key_fact_judgments": [
            {"fact_id": "KF-1", "covered": True},
            {"fact_id": "KF-2", "covered": False},
        ],
        "evidence_association_judgments": [
            {
                "association_id": "EA-1",
                "claim": "one",
                "evidence_reference": "report $.one",
                "supported": True,
            }
        ],
        "evidence_coverage_judgments": [
            {"fact_id": "KF-1", "evidence_supported": True},
            {"fact_id": "KF-2", "evidence_supported": False},
        ],
    }


class TestScoring(unittest.TestCase):
    def test_hybrid_formulas_include_tsia(self) -> None:
        metrics, details = score_case(packet(), review())
        self.assertEqual(METRIC_NAMES, ("FP", "KFR", "IIS", "EA_F1", "TSIA"))
        self.assertEqual(
            metrics,
            {"FP": 0.5, "KFR": 0.5, "IIS": 0.75, "EA_F1": 0.5, "TSIA": 0.5},
        )
        self.assertEqual(details["TSIA"]["required_terms"], 2)

    def test_full_review_formulas_include_tsia(self) -> None:
        metrics, details = score_case_full_review(packet(), full_review())
        self.assertEqual(metrics["FP"], 0.5)
        self.assertEqual(metrics["KFR"], 0.5)
        self.assertEqual(metrics["IIS"], 0.75)
        self.assertAlmostEqual(metrics["EA_F1"], 2 / 3)
        self.assertEqual(metrics["TSIA"], 0.5)
        self.assertEqual(details["TSIA"]["correctly_interpreted_terms"], 1)

    def test_tsia_contract_requires_every_term_once(self) -> None:
        bad = review()
        bad["terminology_judgments"].pop()
        errors = validate_iis_review(packet(), bad)
        self.assertTrue(any("terminology_judgments identifiers differ" in e for e in errors))

    def test_missing_terminology_is_scored_false(self) -> None:
        missing = review()
        for item in missing["terminology_judgments"]:
            item["understood"] = False
        metrics, details = score_case(packet(), missing)
        self.assertEqual(metrics["TSIA"], 0.0)
        self.assertEqual(details["TSIA"]["correctly_interpreted_terms"], 0)

    def test_statistics_and_bootstrap_include_tsia(self) -> None:
        results = []
        for case_id, value in (("A", 1.0), ("B", 0.0)):
            for repetition in (1, 2, 3):
                results.append(
                    {
                        "case_id": case_id,
                        "repetition": repetition,
                        "metrics": {name: value for name in METRIC_NAMES},
                    }
                )
        stats = repetition_statistics(results)
        self.assertEqual(stats["TSIA"]["mean"], 0.5)
        interval = cluster_bootstrap_ci(results, iterations=200, seed=7)
        self.assertIn("TSIA", interval)

    def test_suite_has_sixty_cases_and_source_neutral_terms(self) -> None:
        catalog, cases = load_suite()
        self.assertEqual(len(cases), 60)
        for case in cases:
            terms = _case_terminology_items(catalog, case)
            self.assertTrue(terms)
            self.assertNotIn("ontology_mode", case)
            for term in terms:
                self.assertNotIn("retrieve", str(term).casefold())
                self.assertNotIn("control", str(term).casefold())

    def test_all_cases_pass_relevant_retrieval_preflight(self) -> None:
        catalog, cases = load_suite()
        args = argparse.Namespace(
            api_key=None,
            base_url="https://example.test/v1",
            model="test-model",
            reasoning_effort="medium",
            llm_timeout_seconds=30.0,
            ontology_mode="retrieve",
        )
        _preflight_retrieve_cases(catalog, cases, args)

    def test_implicit_wt_mut_cases_receive_attribution_gold(self) -> None:
        catalog, cases = load_suite()
        implicit = [
            case
            for case in cases
            if any("_wt_" in item for item in case.get("report_ids", []))
            and any("_mut_" in item for item in case.get("report_ids", []))
            and case.get("scope") != "paired"
        ]
        self.assertEqual(len(implicit), 8)
        for case in implicit:
            term_ids = {
                item["term_id"] for item in _case_terminology_items(catalog, case)
            }
            self.assertIn("wild_type_mutant_attribution", term_ids)

    def test_human_and_llm_receive_same_source_blind_tsia_material(self) -> None:
        value = packet()
        value.update(
            {
                "question": "question",
                "initial_question": None,
                "initial_answer": None,
                "candidate_answer": "answer",
                "candidate_limitations": [],
                "expected_behavior": None,
                "iis_requirements": {},
                "report_summaries": [],
            }
        )
        for item in value["terminology_items"]:
            item["operationalization_note"] = "Operational test rule."
            item["fair_term"] = {"term": item["label"], "uri": "https://example.test"}
            item["source_version"] = "hash"
        expected = _review_terminology_items(value["terminology_items"])
        self.assertEqual(_iis_review_packet(value)["terminology_items"], expected)
        self.assertTrue(all("fair_term" not in item for item in expected))

    def test_judge_answer_excludes_terminology_source_appendix(self) -> None:
        answer = (
            "Scientific interpretation.\n\nTerminology sources "
            "(definitions only, not computational-result evidence):\n- source"
        )
        self.assertEqual(_scientific_answer_body(answer), "Scientific interpretation.")

    def test_condition_comparison_is_paired_by_case(self) -> None:
        common_metadata = {
            "test_id": "test_4_fair_ablation",
            "model": "model",
            "repeat": 1,
            "selected_case_ids": ["A", "B"],
        }
        payloads = {}
        for condition, value in (("off", 0.25), ("control", 0.5), ("retrieve", 1.0)):
            audit = {
                "condition_valid": True,
                "knowledge_base_sha256": "fixed-hash",
                "injected_term_count": 0 if condition == "off" else 2,
                "relevant_injected_term_count": 1 if condition == "retrieve" else 0,
            }
            payloads[condition] = {
                "_path": condition,
                "metadata": {**common_metadata, "ontology_mode": condition},
                "results": [
                    {
                        "case_id": case_id,
                        "repetition": 1,
                        "passed": True,
                        "metrics": {name: value for name in METRIC_NAMES},
                        "retrieval_audit": audit,
                    }
                    for case_id in ("A", "B")
                ],
            }
        output = compare(payloads, iterations=200, seed=11)
        self.assertEqual(
            output["metrics"]["TSIA"]["retrieve_minus_off"]["mean_difference"],
            0.75,
        )


if __name__ == "__main__":
    unittest.main()
