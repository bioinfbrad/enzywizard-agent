"""Regression tests for the Test 2 metrics, provider contract, and blind review."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import benchmark_support as bench
from blind_review import export_blind_review
from evidence_matching import evidence_index
import judge
from metrics import METRICS, case_rates, macro_rates, summarize_metrics
from review_agreement import _agreement, _items, compare
import run_test_2


def packet(case_id="T2-SR-01"):
    return {
        "case_id": case_id, "question": "What did the flexibility report find?",
        "initial_question": None, "initial_answer": None, "iis_requirements": {},
        "report_summaries": [{
            "report_id": "flexibility", "key_metrics": {"maximum": 0.98, "minimum": 0.04},
            "side": "single", "artifact_name": "flexibility.json",
            "metric_paths": {"maximum": "$.values[*]", "minimum": "$.values[*]"},
        }],
        "candidate_answer": "Maximum RMSF was 0.98 [RMSF].",
        "claim_citations": [{"text": "Maximum RMSF was 0.98.", "evidence_ids": ["RMSF"]}],
        "candidate_limitations": [], "candidate_evidence_registry": [{
            "evidence_id": "RMSF", "report_id": "flexibility", "side": "single",
            "field_path": "$.values[*]", "value_summary": "maximum=0.98",
            "source_kind": "report", "inference_level": "calculated",
        }],
        "kfr_key_facts": [{"fact_id": "finding", "expected_value": "Use either the mean or maximum."}],
    }


def review():
    return {
        "claims": [{
            "claim": "The reported maximum RMSF was 0.98.",
            "supported": True, "reason": "The value agrees with the report.",
            "reference_evidence": [{"requirement": "Maximum RMSF", "source": "flexibility, maximum"}],
            "citations": [{
                "actual_source": "[RMSF]", "matched_reference_index": 1,
                "reason": "The citation resolves to the matching field.",
            }],
        }],
        "required_claims": [{"fact_id": "finding", "covered": True, "reason": "The maximum qualifies."}],
    }


def raw_review():
    value = review()
    claim = value["claims"][0]
    claim.pop("reference_evidence")
    claim.update(
        reference_evidence_ids=["flexibility::maximum"],
        citations=[{"source_id": "RMSF", "actual_source": "[RMSF]"}],
        requires_citation=True, allow_derived=False, allow_report_level=False,
    )
    return value


def response(value):
    return {
        "stop_reason": "end_turn", "content": [{"type": "text", "text": json.dumps(value)}],
        "usage": {"input_tokens": 200, "output_tokens": 100},
    }


class MetricsTest(unittest.TestCase):
    def test_empty_answer_has_zero_claim_and_inapplicable_citation_rates(self):
        value = review()
        value["claims"].clear()
        value["required_claims"][0]["covered"] = False
        self.assertEqual(case_rates(value), {"Claim F1": {"precision": 0.0, "recall": 0.0}, "Citation F1": None})
        value["required_claims"].clear()
        self.assertIsNone(case_rates(value)["Claim F1"])

    def test_citations_average_written_claims_and_missing_claims_only_affect_claim_recall(self):
        value = review()
        value["required_claims"].append({"fact_id": "missing", "covered": False})
        original = deepcopy(value["claims"][0])
        value["claims"].append({
            **original, "claim": "The side-specific reports agree.", "citations": [],
            "reference_evidence": [*original["reference_evidence"], {"source": "mutant report"}],
        })
        value["claims"].append({**original, "claim": "Unsupported experimental claim.", "supported": False,
                                "reference_evidence": [], "citations": []})
        value["claims"][0]["citations"].append({"matched_reference_index": None})
        rates = case_rates(value)
        self.assertEqual(rates["Claim F1"], {"precision": 2 / 3, "recall": 0.5})
        self.assertEqual(rates["Citation F1"], {"precision": 0.5 / 3, "recall": 1 / 3})

    def test_harmonic_mean_applied_after_macro_averaging(self):
        result = macro_rates([{"precision": 1, "recall": 0}, {"precision": 0, "recall": 1}, None])
        self.assertEqual(result, {"precision": 0.5, "recall": 0.5, "f1": 0.5, "applicable_cases": 2})

    def test_case_bootstrap_keeps_repetitions_paired(self):
        rows = [
            {"case_id": case, "repetition": rep, "rates": {
                name: {"precision": score, "recall": score} for name in METRICS
            }}
            for case, scores in [("A", (0.0, 0.5, 1.0)), ("B", (1.0, 0.5, 0.0))]
            for rep, score in enumerate(scores, 1)
        ]
        result = summarize_metrics(rows, ["A", "B"], 3, iterations=100, seed=3)
        for metric in METRICS:
            self.assertEqual(result[metric]["mean"], 0.5)
            self.assertEqual(result[metric]["sd"], 0.0)
            self.assertEqual(result[metric]["bootstrap_95_ci"], {"lower": 0.5, "upper": 0.5})
        partial = summarize_metrics(rows[:-1], ["A", "B"], 3, iterations=100, seed=3)
        self.assertFalse(partial["Claim F1"]["complete_reviews"])
        self.assertNotIn("bootstrap_95_ci", partial["Claim F1"])


class JudgeTest(unittest.TestCase):
    def test_one_native_claude_request_returns_both_judgments(self):
        value = raw_review()
        with patch.object(judge, "_post", return_value=response(value)) as request:
            scored, usage = judge.review_case(packet(), "mock-key", 5)
        request.assert_called_once()
        endpoint, _, body, timeout = request.call_args.args
        self.assertEqual((endpoint, body["model"], timeout), ("/v1/messages", "claude-haiku-5-5", 5))
        supplied = json.loads(body["messages"][0]["content"])
        self.assertIn("RMSF", supplied.pop("evidence_keys"))
        self.assertEqual(supplied, packet())
        self.assertEqual(body["output_config"]["format"]["type"], "json_schema")
        self.assertEqual(scored["claims"][0]["citations"][0]["matched_reference_index"], 1)
        self.assertEqual(scored["citation_errors"], [])
        self.assertEqual(usage["input_tokens"], 200)

    def test_invalid_citation_id_keeps_claim_score(self):
        value = raw_review()
        value["claims"][0]["reference_evidence_ids"] = ["unknown"]
        with patch.object(judge, "_post", return_value=response(value)):
            scored, _ = judge.review_case(packet(), "mock-key", 5)
        rates = case_rates(scored)
        self.assertEqual(rates["Claim F1"], {"precision": 1.0, "recall": 1.0})
        self.assertIsNone(rates["Citation F1"])
        self.assertTrue(scored["citation_errors"])
        result = summarize_metrics(
            [{"case_id": "A", "repetition": 1, "rates": rates, "metric_errors": {"Citation F1": ["unknown"]}}],
            ["A"], 1, iterations=100, seed=5,
        )
        self.assertTrue(result["Claim F1"]["complete_reviews"])
        self.assertFalse(result["Citation F1"]["complete_reviews"])
        self.assertNotIn("bootstrap_95_ci", result["Citation F1"])

    def test_statistic_is_not_equivalent_to_another_statistic_on_same_raw_path(self):
        value = raw_review()
        value["claims"][0]["reference_evidence_ids"] = ["flexibility::minimum"]
        scored = judge.normalize_review(packet(), value, evidence_index(packet()))
        self.assertEqual(case_rates(scored)["Citation F1"], {"precision": 0.0, "recall": 0.0})

    def test_wrong_side_citation_cannot_match_expected_side(self):
        item = packet()
        wt = item["report_summaries"][0]
        mutant = {**wt, "report_id": "mut", "side": "mut", "artifact_name": "mut.json"}
        item["report_summaries"].append(mutant)
        value = raw_review()
        value["claims"][0]["reference_evidence_ids"] = ["mut::maximum"]
        scored = judge.normalize_review(item, value, evidence_index(item))
        self.assertEqual(case_rates(scored)["Citation F1"], {"precision": 0.0, "recall": 0.0})

    def test_comparison_and_equivalent_direct_links_count_once_per_side(self):
        item = packet()
        item["report_summaries"][0]["side"] = "wt"
        item["candidate_evidence_registry"][0]["side"] = "wt"
        item["report_summaries"].append({
            **item["report_summaries"][0], "report_id": "mut", "side": "mut", "artifact_name": "mut.json",
        })
        item["candidate_evidence_registry"].extend([
            {
                **item["candidate_evidence_registry"][0], "evidence_id": "MUT",
                "report_id": "mut", "side": "mut",
            },
            {
                "evidence_id": "comparison-1", "report_id": None, "side": "shared",
                "field_path": "maximum", "value_summary": "WT=0.98, MUT=0.98",
                "source_kind": "report", "inference_level": "calculated",
                "source_metric_evidence_ids": ["RMSF", "MUT"],
            },
        ])
        item["claim_citations"][0]["evidence_ids"].extend(["comparison-1", "MUT"])
        claim = raw_review()["claims"][0]
        claim.update(
            reference_evidence_ids=["flexibility::maximum", "mut::maximum"],
            allow_derived=True, citations=[{"source_id": "comparison-1", "actual_source": "comparison-1"}],
        )
        value = {**raw_review(), "claims": [claim]}
        index = evidence_index(item)
        def citation_rates():
            return case_rates(judge.normalize_review(item, value, index))["Citation F1"]
        self.assertEqual(citation_rates(), {"precision": 1.0, "recall": 1.0})
        claim["citations"].append({"source_id": "RMSF", "actual_source": "RMSF"})
        self.assertEqual(citation_rates(), {"precision": 1.0, "recall": 1.0})
        claim["citations"].pop(0)
        self.assertEqual(citation_rates(), {"precision": 1.0, "recall": 0.5})
        claim["citations"] = [{"source_id": "comparison-1", "actual_source": "comparison-1"}]
        claim["allow_derived"] = False
        self.assertEqual(citation_rates(), {"precision": 0.0, "recall": 0.0})

        # The judge can return a single entry for two visible inline markers.
        wt_link = "WT maximum (ID: RMSF)"
        mut_link = "MUT maximum (ID: MUT)"
        item["candidate_answer"] = f"Both were 0.98 [Evidence: {wt_link}; Evidence: {mut_link}]."
        claim["citations"] = [{"source_id": "RMSF", "actual_source": f"{wt_link}; {mut_link}"}]
        index = evidence_index(item)
        self.assertEqual(citation_rates(), {"precision": 1.0, "recall": 1.0})

        # A marker linked only elsewhere cannot be used to expand this citation.
        item["candidate_answer"] = f"WT maximum was 0.98 [Evidence: {wt_link}]."
        index = evidence_index(item)
        self.assertEqual(citation_rates(), {"precision": 1.0, "recall": 0.5})

    def test_known_but_unlinked_evidence_is_not_awarded(self):
        item = packet()
        item["candidate_answer"] = "Maximum RMSF was 0.98."
        item["claim_citations"].clear()
        value = judge.normalize_review(item, raw_review(), evidence_index(item))
        self.assertEqual(case_rates(value)["Citation F1"], {"precision": 0.0, "recall": 0.0})

    def test_appendix_copy_of_id_citation_counts_once(self):
        item = packet()
        display = "flexibility.json (turn 1) → Maximum: RMSF = 0.98"
        item["candidate_answer"] += "\n\nEvidence for cited conclusions:\n[1] " + display
        item["claim_citations"][0]["number"] = 1
        value = raw_review()
        value["claims"][0]["citations"].append({"source_id": None, "actual_source": display})

        def score():
            result = judge.normalize_review(item, value, evidence_index(item))
            return case_rates(result)["Citation F1"]

        self.assertEqual(score(), {"precision": 1.0, "recall": 1.0})
        # A distinct unknown appendix link retains its precision penalty.
        value["claims"][0]["citations"].append({"source_id": None, "actual_source": "wrong report or field"})
        self.assertEqual(score(), {"precision": 0.5, "recall": 1.0})

    def test_named_prose_report_resolves_when_report_granularity_is_allowed(self):
        item = packet()
        item["candidate_answer"] = "The **flexibility report** gives a maximum RMSF of 0.98."
        item["report_summaries"][0]["source_tool"] = "enzywizard-flexibility"
        item["claim_citations"].clear()
        value = raw_review()
        claim = value["claims"][0]
        claim["citations"] = [{"source_id": None, "actual_source": "flexibility report"}]
        claim["allow_report_level"] = True
        index = evidence_index(item)
        def score():
            return case_rates(judge.normalize_review(item, value, index))["Citation F1"]
        self.assertEqual(score(), {"precision": 1.0, "recall": 1.0})
        claim["allow_report_level"] = False
        self.assertEqual(score(), {"precision": 0.0, "recall": 0.0})

    def test_written_report_requirement_overrides_conflicting_judge_flag(self):
        item = packet()
        item["kfr_key_facts"][0].update(
            evidence_requirement="Cite the flexibility report for the result stated.",
            supporting_report_ids=["flexibility"],
        )
        claim = raw_review()["claims"][0]
        claim["reference_evidence_ids"] = ["flexibility"]
        value = {**raw_review(), "claims": [claim]}
        index = evidence_index(item)
        review = judge.normalize_review(item, value, index)
        self.assertEqual(case_rates(review)["Citation F1"], {"precision": 1.0, "recall": 1.0})

    def test_explicit_field_requirement_rejects_report_even_if_judge_allows_it(self):
        item = packet()
        item["question"] = "Identify the report and exact field for the maximum RMSF."
        item["candidate_answer"] = "The flexibility.json report gives a maximum RMSF of 0.98."
        item["kfr_key_facts"][0].update(
            evidence_requirement="Identify the report and matching field.",
            supporting_report_ids=["flexibility"],
        )
        value = raw_review()
        claim = value["claims"][0]
        claim.update(
            reference_evidence_ids=["flexibility::maximum"],
            citations=[{"source_id": "flexibility", "actual_source": "flexibility.json"}],
            allow_report_level=True,
        )
        review = judge.normalize_review(item, value, evidence_index(item))
        self.assertEqual(case_rates(review)["Citation F1"], {"precision": 0.0, "recall": 0.0})

        report_link = claim["citations"][0]
        item["candidate_answer"] += " Maximum RMSF was 0.98 [RMSF]."
        claim["citations"].append({"source_id": "RMSF", "actual_source": "[RMSF]"})
        review = judge.normalize_review(item, value, evidence_index(item))
        self.assertEqual(case_rates(review)["Citation F1"], {"precision": 1.0, "recall": 1.0})

        claim["citations"] = [report_link]

        # A separate statement identifying the report has a report-level reference.
        claim["claim"] = "This result comes from the flexibility.json report."
        claim["reference_evidence_ids"] = ["flexibility"]
        review = judge.normalize_review(item, value, evidence_index(item))
        self.assertEqual(case_rates(review)["Citation F1"], {"precision": 1.0, "recall": 1.0})

    def test_shared_report_name_needs_an_explicit_correct_side(self):
        item = packet()
        wt = item["report_summaries"][0]
        wt.update(side="wt", source_tool="enzywizard-flexibility")
        item["report_summaries"].append({**wt, "report_id": "mut", "side": "mut"})
        item["claim_citations"].clear()
        value = raw_review()
        claim = value["claims"][0]
        claim["reference_evidence_ids"] = ["mut::maximum"]
        claim["allow_report_level"] = True
        def score(visible_name, actual):
            item["candidate_answer"] = f"The {visible_name} gives a maximum RMSF of 0.98."
            claim["citations"] = [{"source_id": None, "actual_source": actual}]
            return case_rates(judge.normalize_review(item, value, evidence_index(item)))["Citation F1"]
        for report in ("flexibility report", "WT flexibility report"):
            self.assertEqual(score(report,report), {"precision": 0.0, "recall": 0.0})
        self.assertEqual(score("MUT flexibility report","MUT flexibility report"), {"precision": 1.0, "recall": 1.0})

    def test_citation_exemption_and_uncited_required_claim(self):
        value = raw_review()
        claim = value["claims"][0]
        value["claims"].append({**claim, "claim": "This is a computational prediction.",
                                "requires_citation": False, "reference_evidence_ids": [], "citations": []})
        index = evidence_index(packet())
        scored = judge.normalize_review(packet(), value, index)
        self.assertEqual(case_rates(scored)["Citation F1"], {"precision": 1.0, "recall": 1.0})
        claim["citations"].clear()
        scored = judge.normalize_review(packet(), value, index)
        self.assertEqual(case_rates(scored)["Citation F1"], {"precision": 0.0, "recall": 0.0})


class BlindReviewTest(unittest.TestCase):
    def test_full_export_uses_only_first_run_and_hides_judge_labels_and_reasons(self):
        cases = [f"T2-{n:02}" for n in range(60)]
        value = review()
        rows = [
            {"case_id": case, "repetition": rep, "packet": {
                **packet(case), "candidate_answer": f"Candidate from repetition {rep}."
            }, "review": value, "rates": case_rates(value)}
            for rep in (1, 2, 3) for case in cases
        ]
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            export_blind_review(directory, rows, cases, seed=42)
            human = json.loads((directory / "human_review.json").read_text())
            scored = json.loads((directory / "judge_review.json").read_text())
        self.assertEqual(len(human["cases"]), 12)
        for case in human["cases"]:
            self.assertEqual(case["candidate_answer_lines"], ["Candidate from repetition 1."])
            self.assertNotIn("reason", case["claims"][0])
            self.assertNotIn("matched_reference_id", case["claims"][0]["citations"][0])
        self.assertTrue(all(label is None for group in _items(human).values() for label in group.values()))
        with self.assertRaisesRegex(ValueError, "Enter 0 or 1"):
            compare(human, scored)

        for case in human["cases"]:
            item = case["claims"][0]
            item["supported"] = 1
            item["citations"][0]["correct"] = 1
            item["reference_evidence"][0]["covered"] = 1
            case["required_claims"][0]["covered"] = 1
        labels = compare(human, scored)["agreement"]
        self.assertTrue(all(item == {"item_count": 12, "agreement": 1.0, "cohens_kappa": None}
                            for item in labels.values()))

    def test_cohens_kappa_accounts_for_chance(self):
        self.assertEqual(_agreement(dict(a=1, b=1, c=0, d=0), dict(a=1, b=0, c=0, d=0)), {
            "item_count": 4, "agreement": 0.75, "cohens_kappa": 0.5,
        })


class RunnerTest(unittest.TestCase):
    def test_entry_point_exports_one_scored_review_per_repetition(self):
        analysis = SimpleNamespace(analysis_status="succeeded", model_dump=lambda **_: {})
        with tempfile.TemporaryDirectory() as folder:
            active = [
                "run_test_2.py", "--api_key", "mock-agent", "--judge_api_key", "mock-judge",
                "--case_ids", "T2-SR-01", "--repeat", "2", "--bootstrap_iterations", "0", "--workdir", folder,
            ]
            with (
                patch("sys.argv", active),
                patch.object(run_test_2, "_probe_model", return_value={}),
                patch.object(bench, "_run_analysis", return_value=(analysis, None, {})),
                patch.object(run_test_2, "build_review_packet", return_value=packet()),
                patch.object(judge, "_post", return_value=response(raw_review())) as request,
            ):
                self.assertEqual(run_test_2.main(), 0)
            self.assertEqual(request.call_count, 2)
            results = next(Path(folder).glob("run_*/results.json"))
            payload = json.loads(results.read_text())
            self.assertTrue(payload["summary"]["completed_all_case_runs"])
            self.assertEqual(payload["summary"]["metrics"]["Claim F1"]["mean"], 1.0)
            self.assertEqual(payload["summary"]["metrics"]["Citation F1"]["mean"], 1.0)
            self.assertTrue(results.with_suffix(".csv").is_file())
            self.assertTrue((results.parent / "human_review.json").is_file())
            self.assertTrue((results.parent / "judge_review.json").is_file())
            self.assertTrue((results.parent / "T2-SR-01/repeat_001/judge_response.json").is_file())


if __name__ == "__main__":
    unittest.main()
