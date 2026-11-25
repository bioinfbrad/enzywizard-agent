from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from enzywizard_agent.builders.analysis_context_builder import (
    _explicit_prior_analysis_payload,
    build_scientific_analysis_context,
)
from enzywizard_agent.data_models.agent_inputs import AgentInputs
from enzywizard_agent.data_models.analysis import (
    AnalysisEvidence,
    AnalysisFinding,
    AnalysisResult,
    ReportSummary,
)
from enzywizard_agent.data_models.artifact import Artifact
from enzywizard_agent.data_models.history_query import (
    HistoryQueryOutput,
    HistoryQueryResult,
)
from enzywizard_agent.services.analysis_service import (
    _evidence_display_label,
    _merge_artifact_listing,
    _merge_general_answer,
    _replace_internal_evidence_references,
)
from enzywizard_agent.utils.analysis_utils import (
    build_report_evidence,
    summarize_report_artifact,
)
from enzywizard_agent.utils.terminology_utils import (
    TERMINOLOGY_APPENDIX_MARKER,
    scientific_answer_body,
    terminology_number,
)


def _summary(
    *,
    turn_id: int,
    side: str = "single",
    value: float = -10.0,
) -> ReportSummary:
    return ReportSummary(
        turn_id=turn_id,
        plan_id=1,
        artifact_type="energy_report_json",
        report_type="enzywizard_energy",
        source_tool="enzywizard-energy",
        side=side,
        path=f"/tmp/turn_{turn_id}_energy.json",
        key_metrics={"total_potential_energy": value},
        metric_paths={
            "total_potential_energy": "$.energy.total_potential_energy"
        },
        metric_inference_levels={"total_potential_energy": "direct"},
        highlights=[f"total_potential_energy={value}"],
        warnings=[],
    )


def _analysis(turn_id: int, marker: str) -> AnalysisResult:
    evidence = AnalysisEvidence(
        evidence_id="report-1",
        source_kind="report",
        inference_level="direct",
        turn_id=turn_id,
        plan_id=1,
        artifact_type="energy_report_json",
        source_tool="enzywizard-energy",
        side="single",
        path=f"/tmp/{marker}.json",
        field_path="$",
        value_summary=marker,
    )
    finding = AnalysisFinding(
        title=marker,
        statement=marker,
        finding_kind="result",
        inference_level="direct",
        evidence_ids=["report-1"],
    )
    return AnalysisResult(
        turn_id=turn_id,
        response_mode="scientific_result",
        analysis_status="succeeded",
        analysis_route="answer_generated",
        termination_reason=None,
        summary=f"summary-{marker}",
        final_answer=f"answer-{marker}",
        report_summaries=[_summary(turn_id=turn_id)],
        findings=[finding],
        evidence=[evidence],
        referenced_artifacts=[],
        limitations=[f"limit-{marker}"],
        error_messages=[],
        used_language_model=True,
    )


class AnalysisContextAndProvenanceTests(unittest.TestCase):
    def test_old_report_summary_defaults_remain_compatible(self) -> None:
        summary = ReportSummary(
            turn_id=1,
            plan_id=1,
            artifact_type="energy_report_json",
            report_type="enzywizard_energy",
            source_tool="enzywizard-energy",
            side="single",
            path="/tmp/energy.json",
            key_metrics={"total_potential_energy": -10.0},
            highlights=["total_potential_energy=-10.0"],
            warnings=[],
        )
        self.assertEqual(summary.metric_paths, {})
        self.assertEqual(summary.metric_inference_levels, {})

    def test_field_evidence_is_precise_and_user_readable(self) -> None:
        evidence, findings = build_report_evidence([_summary(turn_id=1)])
        metric = next(item for item in evidence if "-metric-" in item.evidence_id)
        self.assertEqual(
            metric.field_path, "$.energy.total_potential_energy"
        )
        self.assertEqual(metric.metadata["role"], "report_metric")
        rendered = _replace_internal_evidence_references(
            f"Result [{metric.evidence_id}]", evidence
        )
        self.assertNotIn(metric.evidence_id, rendered)
        self.assertIn("$.energy.total_potential_energy", rendered)
        self.assertEqual(findings[0].evidence_ids, [metric.evidence_id])

    def test_array_summary_records_calculated_field_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            path.write_text(
                json.dumps(
                    {
                        "report_type": "test_report",
                        "residues": [
                            {"score": 1.0},
                            {"score": 3.0},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            summary = summarize_report_artifact(
                Artifact(
                    turn_id=1,
                    plan_id=1,
                    artifact_type="energy_report_json",
                    path=str(path),
                    source_tool="enzywizard-energy",
                    side="single",
                    metadata={},
                )
            )
        self.assertEqual(summary.metric_paths["score_mean"], "$.residues[*].score")
        self.assertEqual(
            summary.metric_inference_levels["score_mean"], "calculated"
        )
        evidence, _ = build_report_evidence([summary])
        calculated_metric = next(
            item
            for item in evidence
            if item.metadata.get("metric_name") == "score_mean"
        )
        label = _evidence_display_label(calculated_metric)
        self.assertIn("calculated metric", label)
        self.assertNotIn("comparison", label)

    def test_same_type_cross_turn_reports_are_compared_only_when_requested(self) -> None:
        summaries = [
            _summary(turn_id=1, value=-10.0),
            _summary(turn_id=2, value=-8.5),
        ]
        default_evidence, default_findings = build_report_evidence(summaries)
        self.assertFalse(
            any("later-earlier" in item.value_summary for item in default_evidence)
        )
        self.assertFalse(
            any(
                item.title.startswith("Cross-run report comparison:")
                for item in default_findings
            )
        )

        evidence, findings = build_report_evidence(
            summaries,
            enable_general_comparison=True,
        )
        comparison = next(
            item
            for item in evidence
            if item.inference_level == "calculated"
            and "later-earlier" in item.value_summary
        )
        self.assertEqual(len(comparison.source_evidence_ids), 2)
        self.assertEqual(
            comparison.source_evidence_ids, ["report-1", "report-2"]
        )
        self.assertTrue(
            all(
                "-metric-" in item
                for item in comparison.metadata["source_metric_evidence_ids"]
            )
        )
        self.assertTrue(
            any(
                item.title.startswith("Cross-run report comparison:")
                for item in findings
            )
        )

    def test_copied_report_paths_from_one_plan_are_not_compared(self) -> None:
        original = _summary(turn_id=1, value=-10.0)
        copied = original.model_copy(
            update={"path": "/tmp/copied_energy_report.json"}
        )
        evidence, findings = build_report_evidence(
            [original, copied],
            enable_general_comparison=True,
        )
        self.assertFalse(
            any("later-earlier" in item.value_summary for item in evidence)
        )
        self.assertFalse(
            any(
                item.title.startswith("Cross-run report comparison:")
                for item in findings
            )
        )

    def test_wt_mut_comparison_contract_is_preserved(self) -> None:
        evidence, findings = build_report_evidence(
            [
                _summary(turn_id=1, side="wt", value=-10.0),
                _summary(turn_id=1, side="mut", value=-8.5),
            ]
        )
        comparison = next(
            item
            for item in evidence
            if item.value_summary.startswith("total_potential_energy: WT=")
        )
        self.assertEqual(comparison.source_evidence_ids, ["report-1", "report-2"])
        self.assertEqual(
            comparison.metadata["source_field_paths"],
            [
                "$.energy.total_potential_energy",
                "$.energy.total_potential_energy",
            ],
        )
        self.assertTrue(
            any(item.title.startswith("WT/MUT comparison:") for item in findings)
        )

    def test_only_explicitly_referenced_prior_analysis_is_injected(self) -> None:
        artifact = Artifact(
            turn_id=2,
            plan_id=1,
            artifact_type="energy_report_json",
            path="/tmp/selected.json",
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
            operations=["explain"],
            query_reason="The user explicitly referenced turn 2.",
            clarification_question=None,
        )
        history = HistoryQueryResult(
            turn_id=3,
            query_status="succeeded",
            query=query,
            resolved_turn_ids=[2],
            selected_artifacts=[artifact],
            summary="Resolved turn 2.",
            error_message="",
        )
        payload = _explicit_prior_analysis_payload(
            {
                "current_turn_id": 3,
                "current_history_query": history,
                "analysis_results": [
                    _analysis(1, "excluded"),
                    _analysis(2, "included"),
                ],
            }
        )
        self.assertEqual([item["turn_id"] for item in payload], [2])
        self.assertEqual(payload[0]["final_answer"], "answer-included")
        self.assertNotIn("path", payload[0]["evidence"][0])
        self.assertEqual(
            payload[0]["evidence"][0]["evidence_id"],
            "prior-turn-2-report-1",
        )
        self.assertEqual(
            payload[0]["findings"][0]["evidence_ids"],
            ["prior-turn-2-report-1"],
        )
        self.assertEqual(
            payload[0]["evidence"][0]["artifact_name"], "included.json"
        )

    def test_mixed_response_sections_stay_in_the_body(self) -> None:
        state = {"current_turn_id": 2}
        artifact = Artifact(
            turn_id=1,
            plan_id=1,
            artifact_type="energy_report_json",
            path="/tmp/selected.json",
            source_tool="enzywizard-energy",
            side="single",
            metadata={},
        )
        general = _analysis(2, "additional").model_copy(
            update={"response_mode": "general_scientific_question"}
        )
        appendix = TERMINOLOGY_APPENDIX_MARKER + "(definitions only):\n- RMSF: a definition."
        with patch(
            "enzywizard_agent.services.analysis_service._answer_general_request",
            return_value=general,
        ):
            for glossary in ("", appendix):
                with self.subTest(glossary=bool(glossary)):
                    marker = terminology_number(1) if glossary else ""
                    initial = _analysis(2, "mixed").model_copy(
                        update={"final_answer": f"Mean RMSF{marker}.{glossary}"}
                    )
                    mixed = _merge_artifact_listing(initial, state, [artifact])
                    mixed = _merge_general_answer(mixed, state)
                    body = scientific_answer_body(mixed.final_answer)
                    self.assertIn("Mean RMSF.", body)
                    self.assertIn("Artifacts:\nFound these artifacts:", body)
                    self.assertIn("selected.json", body)
                    self.assertTrue(body.endswith("Additional response:\nanswer-additional"))
                    self.assertEqual(mixed.final_answer, body.replace("RMSF", f"RMSF{marker}") + glossary)

    def test_prior_evidence_ids_cannot_leak_into_user_answer(self) -> None:
        rendered = _replace_internal_evidence_references(
            "Prior claims [prior-turn-2-report-1; "
            "prior-turn-2-future-evidence-kind-1].",
            [],
        )
        self.assertNotIn("prior-turn-2-report-1", rendered)
        self.assertNotIn("prior-turn-2-future-evidence-kind-1", rendered)

    def test_unreferenced_or_reuse_only_history_does_not_inject_analysis(self) -> None:
        self.assertEqual(
            _explicit_prior_analysis_payload(
                {
                    "current_turn_id": 3,
                    "current_history_query": None,
                    "analysis_results": [_analysis(2, "excluded")],
                }
            ),
            [],
        )
        query = HistoryQueryOutput(
            turn_scope="specific",
            referenced_turn_ids=[2],
            artifact_categories=["energy"],
            artifact_types=["energy_report_json"],
            report_types=["enzywizard_energy"],
            sides=["single"],
            operations=["reuse_as_input"],
            query_reason="Reuse a report without asking to explain it.",
            clarification_question=None,
        )
        artifact = Artifact(
            turn_id=2,
            plan_id=1,
            artifact_type="energy_report_json",
            path="/tmp/reused.json",
            source_tool="enzywizard-energy",
            side="single",
            metadata={},
        )
        history = HistoryQueryResult(
            turn_id=3,
            query_status="succeeded",
            query=query,
            resolved_turn_ids=[2],
            selected_artifacts=[artifact],
            summary="Resolved turn 2.",
            error_message="",
        )
        self.assertEqual(
            _explicit_prior_analysis_payload(
                {
                    "current_turn_id": 3,
                    "current_history_query": history,
                    "analysis_results": [_analysis(2, "excluded")],
                }
            ),
            [],
        )

    def test_scientific_context_does_not_leak_global_analysis_summary(self) -> None:
        context = json.loads(
            build_scientific_analysis_context(
                {
                    "current_turn_id": 3,
                    "current_user_message": "Analyze the current report.",
                    "agent_inputs": AgentInputs(
                        structure_file="/tmp/input.cif",
                        msa_file=None,
                        mut_structure_file=None,
                        mut_msa_file=None,
                        substrate_name=None,
                        amino_acid_substitution=None,
                    ),
                    "analysis_summary": "UNREFERENCED_PRIOR_ANALYSIS",
                    "current_history_query": None,
                    "planning_results": [],
                    "user_confirmations": [],
                    "tool_executions": [],
                },
                [_summary(turn_id=3)],
                [],
                [],
            )
        )
        self.assertNotIn("prior_analysis_summary", context)
        self.assertNotIn("UNREFERENCED_PRIOR_ANALYSIS", json.dumps(context))


if __name__ == "__main__":
    unittest.main()
