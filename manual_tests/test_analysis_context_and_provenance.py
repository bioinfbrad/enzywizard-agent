from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from enzywizard_agent.builders.analysis_context_builder import (
    _explicit_prior_analysis_payload,
    build_scientific_analysis_context,
)
from enzywizard_agent.data_models.agent_inputs import AgentInputs
from enzywizard_agent.data_models.analysis import (
    AnalysisEvidence,
    AnalysisFinding,
    AnalysisNarrativeOutput,
    AnalysisResult,
    EvidenceDerivation,
    NarrativeClaimCitation,
    ReportSummary,
)
from enzywizard_agent.data_models.artifact import Artifact
from enzywizard_agent.data_models.history_query import (
    HistoryArtifactSelectionDraft,
    HistoryQueryOutput,
    HistoryQueryResult,
    HistoryTurnSelectionDraft,
)
from enzywizard_agent.data_models.request_routing import RequestedAction
from enzywizard_agent.services.analysis_service import (
    _analyze_reports,
    _confirmed_staged_report_pairs,
    _evidence_display_label,
    _merge_artifact_listing,
    _merge_general_answer,
    _readable_evidence_chain,
    _render_claim_citations,
    _replace_internal_evidence_references,
    scientific_review_answer,
)
from enzywizard_agent.services.request_routing_service import (
    _comparison_side_turns,
    _extract_explicit_turn_ids,
    _history_catalog,
    _resolve_comparison_report_pairs,
    plan_history_query,
)
from enzywizard_agent.utils.CLI_utils import _display_sections
from enzywizard_agent.utils.analysis_utils import (
    build_report_evidence,
    summarize_report_artifact,
)
from enzywizard_agent.utils.terminology_utils import (
    CLAIM_EVIDENCE_APPENDIX_MARKER,
    TERMINOLOGY_APPENDIX_MARKER,
    claim_evidence_number,
    order_answer_markers,
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
        self.assertEqual(summary.metric_operations["score_mean"], ["Mean"])
        self.assertEqual(summary.metric_operations["residues_count"], ["Count entries"])
        self.assertEqual(
            _readable_evidence_chain(calculated_metric, {}),
            "report.json (turn 1) \u2192 Score values from all entries in Residues \u2192 Mean: Score = 2.0",
        )

    def test_display_does_not_infer_an_operation_from_an_old_metric_name(self) -> None:
        summary = _summary(turn_id=1).model_copy(update={
            "key_metrics": {"score_mean": 2.0},
            "metric_paths": {"score_mean": "$.samples[*].score"},
            "metric_inference_levels": {"score_mean": "calculated"},
        })
        evidence, _ = build_report_evidence([summary])
        metric = evidence[1]
        for record in (metric, metric.model_copy(update={"derivation": None})):
            display = _readable_evidence_chain(record, {})
            self.assertIn("Calculated result = 2.0", display)
            self.assertNotIn("Mean", display)
        future_operation = metric.model_copy(update={
            "derivation": EvidenceDerivation(
                steps=["Remove missing values", "Median"], result_value="2.0"
            ),
        })
        self.assertIn(
            "Remove missing values \u2192 Median: Score = 2.0",
            _readable_evidence_chain(future_operation, {}),
        )

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
        line = _readable_evidence_chain(comparison, {item.evidence_id: item for item in evidence})
        self.assertLess(line.index("turn 1"), line.index("turn 2"))
        self.assertIn("Later minus earlier = 1.5", line)
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
                _summary(turn_id=1, side="mut", value=-8.5).model_copy(update={"path": "/tmp/mut.json"}),
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

    def test_wt_mut_pairing_does_not_mix_runs_or_guess_ambiguous_sides(self) -> None:
        wt_old = _summary(turn_id=1, side="wt", value=-10.0)
        mut_old = _summary(turn_id=1, side="mut", value=-8.0).model_copy(
            update={"path": "/tmp/turn_1_mut.json"}
        )
        wt_new = _summary(turn_id=2, side="wt", value=-20.0)
        wt_revision = wt_new.model_copy(update={"turn_id": 1, "plan_id": 2})
        for extra in (wt_new, wt_revision):
            for reports in ([wt_old, mut_old, extra], [extra, wt_old, mut_old]):
                evidence, _ = build_report_evidence(reports)
                comparisons = [e for e in evidence if e.evidence_id.startswith("comparison-")]
                self.assertEqual([e.derivation.result_value for e in comparisons], ["2.0"])
        extra = wt_old.model_copy(update={"path": "/tmp/another_wt.json"})
        evidence, _ = build_report_evidence([extra, wt_old, mut_old])
        self.assertFalse(any(e.evidence_id.startswith("comparison-") for e in evidence))

    def test_history_can_bind_an_explicit_cross_turn_pair(self) -> None:
        def artifact(turn: int, side: str) -> Artifact:
            return Artifact(
                turn_id=turn, plan_id=1, artifact_type="energy_report_json",
                path=f"/tmp/{turn}/{side}.json", source_tool="enzywizard-energy",
                side=side, metadata={},
            )

        wt, mut, other_wt = artifact(2, "wt"), artifact(1, "mut"), artifact(3, "wt")
        self.assertIsNone(_resolve_comparison_report_pairs([wt, mut], explicit_scope=False))
        self.assertEqual(
            _resolve_comparison_report_pairs(
                [wt, mut], explicit_scope=False, request="Compare wt.json and mut.json.",
            ),
            [(wt.path, mut.path)],
        )
        self.assertIsNone(_resolve_comparison_report_pairs([wt, mut, other_wt], explicit_scope=True))
        pairs = _resolve_comparison_report_pairs([mut, wt], explicit_scope=True)
        self.assertEqual(pairs, [(wt.path, mut.path)])
        summaries = [
            _summary(turn_id=a.turn_id, side=a.side, value=v).model_copy(update={"path": a.path})
            for a, v in ((mut, -8.0), (wt, -20.0))
        ]
        evidence, _ = build_report_evidence(summaries, comparison_report_pairs=pairs)
        comparison = next(e for e in evidence if e.evidence_id.startswith("comparison-"))
        self.assertEqual(comparison.derivation.result_value, "12.0")
        self.assertEqual(comparison.source_evidence_ids, ["report-2", "report-1"])
        state = {
            "current_turn_id": 4, "artifacts": [mut, wt, other_wt],
            "current_user_message": "Compare the mutant and wild-type reports from turns 1 and 2.",
        }
        action = RequestedAction(
            action_type="analyze_history", target="Compare reports from turns 1 and 2",
            history_operation="compare",
        )
        routing = "enzywizard_agent.services.request_routing_service"
        by_path = {item.artifact.path: item.catalog_id for item in _history_catalog(state)}
        with (
            patch(f"{routing}._history_actions", return_value=[action]),
            patch(f"{routing}._create_llm"),
            patch(f"{routing}.invoke_structured", return_value=
                HistoryArtifactSelectionDraft(action_id=0, artifact_ids=[by_path[mut.path], by_path[wt.path]]),
            ) as invoke,
        ):
            history = plan_history_query(state)
        self.assertEqual(history.query_status, "succeeded", history.error_message)
        self.assertEqual(history.comparison_report_pairs, [(wt.path, mut.path)])
        invoke.assert_called_once()
        analysis = "enzywizard_agent.services.analysis_service"
        summary_by_path = {summary.path: summary for summary in summaries}
        with patch(f"{analysis}.summarize_report_artifact", side_effect=lambda a: summary_by_path[a.path]):
            result = _analyze_reports(
                {**state, "current_history_query": history}, history.selected_artifacts, historical=True,
            )
        comparison = next(e for e in result.evidence if e.evidence_id.startswith("comparison-"))
        self.assertEqual(comparison.derivation.result_value, "12.0")

    def test_history_checks_requested_side_turns_before_accepting_model_selection(self) -> None:
        def artifact(turn: int, side: str) -> Artifact:
            return Artifact(
                turn_id=turn, plan_id=1, artifact_type="energy_report_json",
                path=f"/tmp/turn-{turn}-{side}.json", source_tool="enzywizard-energy",
                side=side, metadata={},
            )

        wt1, mut1, wt2, mut2 = [artifact(turn, side) for turn in (1, 2) for side in ("wt", "mut")]
        unassigned1, unassigned2 = [artifact(turn, "single") for turn in (1, 2)]
        available = [wt1, mut1, wt2, mut2, unassigned1, unassigned2]
        routing = "enzywizard_agent.services.request_routing_service"
        stated = "Compare the WT energy report from turn 1 with the MUT energy report from turn 2."
        named = "Compare turn-1-wt.json and turn-2-mut.json from turns 1 and 2."
        for phrasing in (
            stated,
            "Compare turn 1's wild type with turn 2's mutant.",
            "Compare WT (turn 1) with MUT (turn 2).",
            "\u6bd4\u8f83\u7b2c1\u8f6e\u7684WT\u548c\u7b2c2\u8f6e\u7684MUT\u3002",
        ):
            self.assertEqual(_comparison_side_turns(phrasing), {"wt": 1, "mut": 2})

        for request, chosen, success in (
            (stated, (wt1, mut2), True),
            (stated, (wt2, mut1), False),
            (stated, (wt1, mut1), False),
            (stated, (unassigned1, unassigned2), False),
            ("Compare the WT and MUT energy reports from turns 1 and 2.", (wt1, mut2), False),
            (named, (wt1, mut2), True),
            (named, (wt2, mut1), False),
            ("Compare the WT energy values from turns 1 and 2.", (wt1, wt2), True),
        ):
            with self.subTest(request=request, chosen=chosen):
                state = {"current_turn_id": 3, "current_user_message": request, "artifacts": available}
                action = RequestedAction(action_type="analyze_history", target=request, history_operation="compare")
                ids = {item.artifact.path: item.catalog_id for item in _history_catalog(state)}
                selection = HistoryArtifactSelectionDraft(
                    action_id=0, artifact_ids=[ids[report.path] for report in chosen],
                )
                with (
                    patch(f"{routing}._history_actions", return_value=[action]),
                    patch(f"{routing}._create_llm"),
                    patch(f"{routing}.invoke_structured", return_value=selection) as invoke,
                ):
                    result = plan_history_query(state)
                invoke.assert_called_once()
                if success:
                    self.assertEqual(result.query_status, "succeeded", result.error_message)
                    if chosen == (wt1, mut2):
                        self.assertEqual(result.comparison_report_pairs, [(wt1.path, mut2.path)])
                else:
                    self.assertEqual(result.query_status, "clarification_required", result.error_message)
                    self.assertFalse(result.comparison_report_pairs)
                    self.assertFalse(result.selected_artifacts)

    def test_history_checks_coverage_per_comparison_action(self) -> None:
        self.assertEqual(_extract_explicit_turn_ids("Compare turns 1, 2, and 3."), [1, 2, 3])
        self.assertEqual(_extract_explicit_turn_ids("\u6bd4\u8f83\u7b2c1\u30012\u30013\u8f6e\u3002"), [1, 2, 3])

        def artifact(turn: int, side: str, kind: str = "energy_report_json") -> Artifact:
            return Artifact(
                turn_id=turn, plan_id=1, artifact_type=kind,
                path=f"/tmp/{turn}-{side}-{kind}.json", source_tool="enzywizard-energy",
                side=side, metadata={},
            )

        wt1, mut1, wt2, mut2 = [artifact(turn, side) for turn in (1, 2) for side in ("wt", "mut")]
        log2 = artifact(2, "single", "stdout_file")
        reports = [wt1, mut1, wt2, mut2]
        available = [*reports, log2]
        service = "enzywizard_agent.services.request_routing_service"
        text = "Compare the WT and MUT energy reports from turns 1 and 2."
        compare = RequestedAction(action_type="analyze_history", target=text, history_operation="compare")

        def execute(request, actions, selections, *, prefix=()):
            state = {"current_turn_id": 3, "current_user_message": request, "artifacts": available}
            ids = {item.artifact.path: item.catalog_id for item in _history_catalog(state)}
            drafts = [
                HistoryArtifactSelectionDraft(action_id=index, artifact_ids=[ids[report.path] for report in selection])
                for index, selection in enumerate(selections)
            ]
            with (
                patch(f"{service}._history_actions", return_value=actions),
                patch(f"{service}._create_llm"),
                patch(f"{service}.invoke_structured", side_effect=[*prefix, *drafts]) as call,
            ):
                result = plan_history_query(state)
                self.assertEqual(call.call_count, len(prefix) + len(drafts))
                return result

        full = execute(text, [compare], [reports])
        self.assertEqual(full.query_status, "succeeded", full.error_message)
        self.assertEqual(full.comparison_report_pairs, [(wt1.path, mut1.path), (wt2.path, mut2.path)])
        for picked in ([wt1, mut1], [wt1, mut1, log2]):
            with self.subTest(picked=picked):
                # The router's abbreviated target must not drop the user's second turn.
                short = compare.model_copy(update={"target": "Compare reports from turn 1."})
                missing = execute(text, [short], [picked])
                self.assertEqual(missing.query_status, "clarification_required", missing.error_message)
                self.assertIn("turn 2", missing.query.clarification_question)
                self.assertFalse(missing.comparison_report_pairs)
        listing = RequestedAction(action_type="list_artifacts", target="List the reports from turn 2.")
        mixed = execute(f"{text} Then list the reports from turn 2.", [compare, listing], [[wt1, mut1], [wt2, mut2]])
        self.assertEqual(mixed.query_status, "clarification_required", mixed.error_message)
        self.assertIn("turn 2", mixed.query.clarification_question)

        names = f"Compare {Path(wt1.path).name} and {Path(mut2.path).name}."
        by_name = compare.model_copy(update={"target": names})
        selected = execute(names, [by_name], [[wt1, mut2]])
        self.assertEqual(selected.comparison_report_pairs, [(wt1.path, mut2.path)])
        omitted = execute(names, [by_name], [[wt1, mut1]])
        self.assertEqual(omitted.query_status, "clarification_required", omitted.error_message)
        self.assertIn(Path(mut2.path).name, omitted.query.clarification_question)

        vague = compare.model_copy(update={"target": "Compare the latest reports."})
        implicit = execute(vague.target, [vague], [[wt2, mut2]], prefix=[
            HistoryTurnSelectionDraft(action_id=0, turn_scope="latest_matching", turn_ids=[]),
        ])
        self.assertEqual(implicit.comparison_report_pairs, [(wt2.path, mut2.path)])

    def test_verified_staging_preserves_a_reused_report_pair(self) -> None:
        def artifact(turn: int, side: str) -> Artifact:
            return Artifact(
                turn_id=turn, plan_id=1, artifact_type="energy_report_json",
                path=f"/tmp/{turn}/{side}.json", source_tool="enzywizard-energy",
                side=side, metadata={},
            )

        old_wt, new_mut, unrelated = artifact(1, "wt"), artifact(2, "mut"), artifact(3, "wt")
        plan = SimpleNamespace(turn_id=2, plan_id=1, calls=[SimpleNamespace(
            tool_name="enzywizard-mut-integrate", step_index=2, staging_rules=[
                SimpleNamespace(side="wt", source_step_index=None, source_path_pattern=old_wt.path),
                SimpleNamespace(side="mut", source_step_index=1, source_path_pattern="/tmp/2/*.json"),
            ],
        )])
        execution = SimpleNamespace(
            turn_id=2, plan_id=1, execution_status="succeeded", call_executions=[
                SimpleNamespace(step_index=1, execution_status="succeeded", artifacts=[new_mut]),
                SimpleNamespace(step_index=2, execution_status="succeeded", artifacts=[]),
            ],
        )
        state = {
            "current_planning_result": SimpleNamespace(tool_plan=plan),
            "current_tool_execution": execution,
            "artifacts": [old_wt, unrelated],
        }
        reused, pairs = _confirmed_staged_report_pairs(state)
        self.assertEqual(reused, [old_wt])
        self.assertEqual(pairs, [(old_wt.path, new_mut.path)])
        execution.call_executions.pop()
        self.assertEqual(_confirmed_staged_report_pairs(state), ([], []))

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

    def test_scientific_answer_uses_one_call_and_cites_only_drafted_claims(self) -> None:
        artifact = Artifact(
            turn_id=1, plan_id=1, artifact_type="energy_report_json",
            path="/tmp/turn_1_energy.json", source_tool="enzywizard-energy",
            side="single", metadata={},
        )
        summary = _summary(turn_id=1).model_copy(update={
            "key_metrics": {"total_potential_energy": -10.0, "unused_metric": 12},
            "metric_paths": {
                "total_potential_energy": "$.energy.total_potential_energy",
                "unused_metric": "$.energy.unused_metric",
            },
        })
        claims = ["The total potential energy was -10.", "That result was negative."]
        prose = " ".join(claims) + "\n\nThese are computed results, not experimental measurements."
        output = AnalysisNarrativeOutput(
            summary="Energy analysis", final_answer=prose, limitations=[],
            claim_citations=[
                NarrativeClaimCitation(text=claim, evidence_ids=["report-1-metric-1"])
                for claim in claims
            ],
        )
        state = {
            "current_turn_id": 2,
            "agent_inputs": AgentInputs(
                structure_file=None, msa_file=None, mut_structure_file=None,
                mut_msa_file=None, substrate_name=None, amino_acid_substitution=None,
            ),
        }
        service = "enzywizard_agent.services.analysis_service"
        with (
            patch(f"{service}.summarize_report_artifact", return_value=summary),
            patch(f"{service}._create_llm"),
            patch(f"{service}.invoke_structured", return_value=output) as invoke,
        ):
            result = _analyze_reports(state, [artifact], historical=True)
            invoke.assert_called_once()
        self.assertTrue(result.used_language_model)
        self.assertEqual(scientific_answer_body(result.final_answer), prose)
        self.assertEqual(
            [(item.number, item.text, item.evidence_ids) for item in result.claim_citations],
            [(i, claim, ["report-1-metric-1"]) for i, claim in enumerate(claims, 1)],
        )
        self.assertEqual(len(result.evidence), 3)
        display = dict(_display_sections("Analysis", result.final_answer))
        self.assertIn(claim_evidence_number(2), display["Analysis"])
        appendix = display["Evidence for cited conclusions"]
        self.assertNotIn("unused", appendix.lower())
        self.assertNotIn("$.energy", appendix)
        self.assertIn("Total potential energy", appendix)
        review = scientific_review_answer(result)
        self.assertEqual(review.count("ID: report-1-metric-1"), 2)
        self.assertIn("$.energy.total_potential_energy", review)
        self.assertNotIn(CLAIM_EVIDENCE_APPENDIX_MARKER, review)
        self.assertEqual(order_answer_markers("RMSF" + terminology_number(1) + claim_evidence_number(1)),
                         "RMSF" + claim_evidence_number(1) + terminology_number(1))
        mixed = _merge_artifact_listing(result, state, [artifact])
        self.assertIn("Artifacts:", scientific_review_answer(mixed))
        self.assertIn("Artifacts:", scientific_answer_body(mixed.final_answer))
        with (
            patch(f"{service}.summarize_report_artifact", return_value=summary),
            patch(f"{service}._create_llm", side_effect=RuntimeError("offline")),
        ):
            fallback = _analyze_reports(state, [artifact], historical=True)
        self.assertEqual(fallback.analysis_status, "partial")
        self.assertTrue(all(c.evidence_ids for c in fallback.claim_citations))
        self.assertIn(CLAIM_EVIDENCE_APPENDIX_MARKER, fallback.final_answer)

    def test_citations_check_side_value_and_link_comparison_to_both_reports(self) -> None:
        wt = _summary(turn_id=1, side="wt", value=-10.0)
        mut = _summary(turn_id=1, side="mut", value=-9.0).model_copy(
            update={"path": "/tmp/mut_energy.json"}
        )
        evidence, _ = build_report_evidence([wt, mut])
        claims = [
            ("The mutant energy was -9.", ["report-1-metric-1"]),
            ("The wild-type energy was -8.", ["report-1-metric-1"]),
            ("The mutant was 1 higher than the wild type.", ["comparison-1", "missing"]),
            ("Unanchored citation.", ["report-1"]),
        ]
        prose = " ".join(claim for claim, _ in claims[:3])
        marked, appendix, citations, warnings = _render_claim_citations(
            prose, [NarrativeClaimCitation(text=claim, evidence_ids=ids) for claim, ids in claims],
            evidence,
        )
        self.assertEqual([c.evidence_ids for c in citations], [[], [], ["comparison-1"]])
        self.assertEqual(scientific_answer_body(marked + appendix), prose)
        self.assertIn("Mutant minus wild type = 1.0", appendix)
        self.assertIn("mut_energy.json", appendix)
        self.assertIn("turn_1_energy.json", appendix)
        self.assertIn("No validated computational source was linked", appendix)
        self.assertEqual(len(warnings), 3)

    def test_comparison_renders_each_sources_recorded_steps(self) -> None:
        common = {
            "metric_paths": {"score_mean": "$.samples[*].score"},
            "metric_operations": {"score_mean": ["Remove missing values", "Mean"]},
            "metric_inference_levels": {"score_mean": "calculated"},
        }
        wt = _summary(turn_id=1, side="wt").model_copy(update={
            **common, "key_metrics": {"score_mean": 3.0},
        })
        mut = _summary(turn_id=1, side="mut").model_copy(update={
            **common, "key_metrics": {"score_mean": 7.0}, "path": "/tmp/turn_1_mut.json",
        })
        evidence, _ = build_report_evidence([wt, mut])
        comparison = next(item for item in evidence if item.evidence_id == "comparison-1")
        by_id = {item.evidence_id: item for item in evidence}
        rendered = _readable_evidence_chain(comparison, by_id)
        self.assertIn("Remove missing values \u2192 Mean: Score = 3.0", rendered)
        self.assertIn("Remove missing values \u2192 Mean: Score = 7.0", rendered)
        self.assertTrue(rendered.endswith("Mutant minus wild type = 4.0"))
        self.assertNotIn("Samples = 3.0", rendered)
        legacy_trace = comparison.derivation.model_copy(update={"source_steps": []})
        legacy = comparison.model_copy(update={"derivation": legacy_trace})
        old_display = _readable_evidence_chain(legacy, by_id)
        self.assertIn("Value used in comparison = 3.0", old_display)
        self.assertNotIn("Mean:", old_display)

    def test_model_cannot_cite_evidence_excluded_from_its_prompt(self) -> None:
        artifact = Artifact(
            turn_id=1, plan_id=1, artifact_type="energy_report_json",
            path="/tmp/energy.json", source_tool="enzywizard-energy",
            side="single", metadata={},
        )
        state = {
            "current_turn_id": 2,
            "agent_inputs": AgentInputs(
                structure_file=None, msa_file=None, mut_structure_file=None,
                mut_msa_file=None, substrate_name=None, amino_acid_substitution=None,
            ),
        }
        summary = _summary(turn_id=1)
        seed = build_report_evidence([summary])[0][0]
        evidence = [
            seed.model_copy(update={"evidence_id": f"report-{i}"}) for i in range(1, 82)
        ]
        claims = ["The energy was -10.", "The report contains energy data."]
        output = AnalysisNarrativeOutput(
            summary="Energy report", final_answer=" ".join(claims), limitations=[],
            claim_citations=[
                NarrativeClaimCitation(text=claim, evidence_ids=[f"report-{index}"])
                for index, claim in enumerate(claims, 80)
            ],
        )
        service = "enzywizard_agent.services.analysis_service"
        with (
            patch(f"{service}.summarize_report_artifact", return_value=summary),
            patch(f"{service}.build_report_evidence", return_value=(evidence, [])),
            patch(f"{service}._create_llm"),
            patch(f"{service}.invoke_structured", return_value=output) as invoke,
        ):
            result = _analyze_reports(state, [artifact], historical=True)
        context = json.loads(invoke.call_args.kwargs["messages"][1].content)
        ids = {item["evidence_id"] for item in context["deterministic_evidence"]}
        self.assertIn("report-80", ids)
        self.assertNotIn("report-81", ids)
        self.assertEqual([item.evidence_ids for item in result.claim_citations], [["report-80"], []])
        self.assertEqual(len(result.evidence), 81)
        self.assertNotIn("report-81", scientific_review_answer(result))

    def test_citation_anchor_cannot_match_inside_a_value(self) -> None:
        evidence, _ = build_report_evidence([_summary(turn_id=1)])
        prose = "The total potential energy was -10.25."
        draft = NarrativeClaimCitation(
            text="The total potential energy was -10.", evidence_ids=["report-1-metric-1"]
        )
        marked, appendix, citations, warnings = _render_claim_citations(prose, [draft], evidence)
        self.assertEqual((marked, appendix, citations), (prose, "", []))
        self.assertEqual(len(warnings), 1)

    def test_report_read_failure_keeps_available_evidence_but_marks_analysis_partial(self) -> None:
        service = "enzywizard_agent.services.analysis_service"
        inputs = AgentInputs(
            structure_file=None, msa_file=None, mut_structure_file=None,
            mut_msa_file=None, substrate_name=None, amino_acid_substitution=None,
        )
        with tempfile.TemporaryDirectory() as directory:
            good_file = Path(directory) / "turn1-energy.json"
            good_file.write_text(json.dumps({
                "report_type": "enzywizard_energy", "energy": {"total_potential_energy": -10.0},
            }), encoding="utf-8")
            other_file = Path(directory) / "turn2-energy.json"
            reports = [
                Artifact(
                    turn_id=turn, plan_id=1, artifact_type="energy_report_json",
                    path=str(path), source_tool="enzywizard-energy", side="single", metadata={},
                )
                for turn, path in enumerate((good_file, other_file), start=1)
            ]
            query = HistoryQueryOutput(
                turn_scope="specific", referenced_turn_ids=[1, 2],
                artifact_categories=["energy"], artifact_types=["energy_report_json"],
                report_types=["enzywizard_energy"], sides=["single"], operations=["compare"],
                query_reason="The user explicitly requested both turns.", clarification_question=None,
            )
            history = HistoryQueryResult(
                turn_id=3, query_status="succeeded", query=query, resolved_turn_ids=[1, 2],
                selected_artifacts=reports, summary="Selected both reports.", error_message="",
            )
            state = {"current_turn_id": 3, "agent_inputs": inputs, "current_history_query": history}
            output = AnalysisNarrativeOutput(
                summary="Energy analysis", final_answer="Turn 1 energy was -10.",
                limitations=[], claim_citations=[],
            )
            with (
                patch(f"{service}._create_llm"),
                patch(f"{service}.invoke_structured", return_value=output) as invoke,
            ):
                missing = _analyze_reports(state, reports, historical=True)
                missing_context = json.loads(invoke.call_args.kwargs["messages"][1].content)
                other_file.write_text("{invalid json", encoding="utf-8")
                malformed = _analyze_reports(state, reports, historical=True)
                other_file.write_text(good_file.read_text(encoding="utf-8"), encoding="utf-8")
                complete = _analyze_reports(state, reports, historical=True)
                self.assertEqual(invoke.call_count, 3)
            for report, expected in ((missing, "missing"), (malformed, "unreadable")):
                self.assertEqual(report.analysis_status, "partial")
                self.assertIn("The requested comparison is incomplete.", report.final_answer)
                self.assertIn("turn2-energy.json (turn 2", report.final_answer)
                self.assertEqual([summary.read_status for summary in report.report_summaries], ["parsed", expected])
                self.assertEqual({item.path for item in report.evidence}, {str(good_file)})
                self.assertIn("Turn 1 energy was -10.", report.final_answer)
            self.assertEqual(missing_context["report_summaries"][1]["read_status"], "missing")
            self.assertEqual(complete.analysis_status, "succeeded")
            self.assertTrue(complete.limitations)  # Normal energy cautions are not read failures.
            self.assertTrue(any(e.evidence_id.startswith("comparison-") for e in complete.evidence))

            good_file.unlink()
            other_file.unlink()
            with patch(f"{service}.invoke_structured") as invoke:
                unavailable = _analyze_reports(state, reports, historical=True)
            invoke.assert_not_called()
            self.assertEqual(unavailable.analysis_status, "partial")
            self.assertFalse(unavailable.evidence)
            self.assertEqual(len(unavailable.report_summaries), 2)
            self.assertIn("turn1-energy.json", unavailable.final_answer)
            self.assertIn("turn2-energy.json", unavailable.final_answer)

    def test_report_read_status_distinguishes_size_and_ordinary_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.json"
            path.write_text("{}", encoding="utf-8")
            artifact = Artifact(
                turn_id=1, plan_id=1, artifact_type="energy_report_json", path=str(path),
                source_tool="enzywizard-energy", side="single", metadata={},
            )
            parsed = summarize_report_artifact(artifact)
            self.assertTrue(parsed.warnings)
            self.assertEqual(parsed.read_status, "parsed")
            with patch("enzywizard_agent.utils.analysis_utils.MAX_REPORT_BYTES", 1):
                oversized = summarize_report_artifact(artifact)
            self.assertEqual(oversized.read_status, "too_large")
            self.assertEqual(build_report_evidence([oversized]), ([], []))
            path.write_bytes(b"\xff")
            self.assertEqual(summarize_report_artifact(artifact).read_status, "unreadable")

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
