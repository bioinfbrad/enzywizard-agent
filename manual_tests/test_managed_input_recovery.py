from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from enzywizard_agent.conf import AgentConfig
from enzywizard_agent.data_models.artifact import Artifact
from enzywizard_agent.data_models.planning import (
    ParameterExtractionOutput,
    PlanValidationRecord,
    PlanningResult,
    ReportStagingRule,
    ToolCall,
    ToolOutputReference,
    ToolOutputArtifact,
    ToolPlan,
)
from enzywizard_agent.data_models.runtime import RuntimeSession
from enzywizard_agent.data_models.tool_execution import ToolCallExecution
from enzywizard_agent.data_models.user_confirmation import UserConfirmation
from enzywizard_agent.services.tool_calling_service import (
    clear_failed_attempt_outputs,
    execute_approved_tool_plan,
)
from enzywizard_agent.services.tool_recovery_service import (
    authorize_managed_input_repair,
    diagnose_tool_failure,
    prepare_managed_input_repair,
    select_recovery_decision,
)
from enzywizard_agent.services.runtime_service import utc_now
from enzywizard_agent.services.planning_service import repair_internal_tool_plan
from enzywizard_agent.utils.tool_execution_utils import stage_tool_reports


def _artifact(path: Path, *, step_index: int = 1) -> Artifact:
    return Artifact(
        turn_id=1,
        plan_id=1,
        artifact_type="energy_report_json",
        path=str(path),
        source_tool="enzywizard-energy",
        side="single",
        metadata={"step_index": step_index, "required": True},
    )


def _call(workdir: Path, *, step_index: int = 2) -> ToolCall:
    execution_dir = workdir / f"{step_index:02d}_integrate"
    return ToolCall(
        turn_id=1,
        plan_id=1,
        step_index=step_index,
        tool_name="enzywizard-integrate",
        command_name="enzywizard-integrate",
        side="single",
        tool_execution_dir=str(execution_dir),
        arguments={
            "input_dir": str(execution_dir / "input_reports"),
            "output_dir": str(execution_dir),
        },
        output_artifacts=[],
        staging_rules=[
            ReportStagingRule(
                source_step_index=1,
                source_path_pattern=str(workdir / "stale" / "energy_report_*.json"),
                destination_dir=str(execution_dir / "input_reports"),
                side="single",
            )
        ],
        call_reason="Integrate an upstream report.",
    )


def _execution(
    call: ToolCall,
    *,
    status: str,
    attempt_index: int,
    artifacts: list[Artifact] | None = None,
    error: str | None = None,
) -> ToolCallExecution:
    execution_dir = Path(call.tool_execution_dir)
    execution_dir.mkdir(parents=True, exist_ok=True)
    stdout = execution_dir / f"stdout_attempt_{attempt_index}.txt"
    stderr = execution_dir / f"stderr_attempt_{attempt_index}.txt"
    stdout.write_text("", encoding="utf-8")
    stderr.write_text(error or "", encoding="utf-8")
    return ToolCallExecution(
        turn_id=call.turn_id,
        plan_id=call.plan_id,
        step_index=call.step_index,
        tool_name=call.tool_name,
        side=call.side,
        attempt_index=attempt_index,
        execution_status=status,
        command=[call.command_name] if status == "succeeded" else [],
        resolved_arguments={},
        exit_code=0 if status == "succeeded" else None,
        duration_seconds=0.01,
        stdout_file=str(stdout),
        stderr_file=str(stderr),
        log_files=[],
        stdout_excerpt="",
        stderr_excerpt=error or "",
        log_excerpt="",
        artifacts=artifacts or [],
        error_message=error,
    )


class ManagedInputRecoveryTests(unittest.TestCase):
    def test_invalid_agent_plan_structure_is_regenerated_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workdir = Path(temporary)
            canonical_call = ToolCall(
                turn_id=1,
                plan_id=1,
                step_index=1,
                tool_name="enzywizard-energy",
                command_name="enzywizard-energy",
                side="single",
                tool_execution_dir=str(workdir / "01_energy"),
                arguments={"output_dir": str(workdir / "01_energy")},
                output_artifacts=[],
                staging_rules=[],
                call_reason="Canonical program-managed call.",
            )
            canonical_plan = ToolPlan(
                turn_id=1,
                plan_id=1,
                purpose_type="single_protein_analysis",
                summary="Canonical plan",
                plan_reason="Regenerate internal structure.",
                calls=[canonical_call],
            )
            planning = PlanningResult(
                turn_id=1,
                plan_id=1,
                planning_route="tool_unused",
                planning_flag="default_tool_plan_generated",
                needs_tool=True,
                summary=None,
                purpose_type="single_protein_analysis",
                purpose_reason="Test deterministic plan repair.",
                tool_selection=None,
                parameter_extraction=None,
                tool_plan=None,
                error_messages=[],
            )
            extraction = ParameterExtractionOutput(
                parameter_extraction_status="parameter_extraction_succeeded",
                parameter_adjustments=[],
                profile_requests=[],
                unsupported_requests=[],
                extraction_summary="No user parameter changes.",
                error_message="",
            )
            failed_validation = PlanValidationRecord(
                passed=False,
                checks=["registered_tools_and_commands"],
                warnings=[],
                error_messages=["Step 1 command was internally corrupted."],
            )
            state = {
                "config": AgentConfig(
                    api_key="test",
                    base_url="https://example.invalid/v1",
                    model="test",
                    workdir=workdir,
                    check_model_capabilities=False,
                )
            }
            passed_validation = PlanValidationRecord(
                passed=True,
                checks=["registered_tools_and_commands"],
                warnings=[],
                error_messages=[],
            )
            with (
                patch(
                    "enzywizard_agent.services.planning_service.build_default_tool_plan",
                    return_value=canonical_plan,
                ),
                patch(
                    "enzywizard_agent.services.planning_service.audit_tool_plan",
                    return_value=passed_validation,
                ),
            ):
                repaired, record = repair_internal_tool_plan(
                    state,
                    planning,
                    extraction,
                    failed_validation,
                )
            self.assertEqual(repaired, canonical_plan)
            self.assertEqual(record.repair_status, "succeeded")
            self.assertEqual(record.trigger_errors, failed_validation.error_messages)

            with patch(
                "enzywizard_agent.services.planning_service.build_default_tool_plan",
                side_effect=ValueError("deterministic rebuild failed"),
            ):
                failed_plan, failed_record = repair_internal_tool_plan(
                    state,
                    planning,
                    extraction,
                    failed_validation,
                )
            self.assertIsNone(failed_plan)
            self.assertEqual(failed_record.repair_status, "failed")
            self.assertIn(
                "deterministic rebuild failed",
                failed_record.validation.error_messages[0],
            )

    def test_unique_registered_report_enables_bounded_repair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workdir = Path(temporary)
            report = workdir / "01_energy" / "energy_report.json"
            report.parent.mkdir(parents=True)
            report.write_text("{}", encoding="utf-8")
            call = _call(workdir)

            repair = prepare_managed_input_repair(
                call, [_artifact(report)], {1}, workdir
            )
            self.assertIsNotNone(repair)
            repaired_call, records = repair or (None, [])
            self.assertEqual(
                repaired_call.staging_rules[0].source_path_pattern,
                str(report.resolve()),
            )
            self.assertEqual(records[0].artifact_type, "energy_report_json")
            self.assertEqual(repaired_call.arguments, call.arguments)
            self.assertEqual(repaired_call.output_artifacts, call.output_artifacts)
            stage_tool_reports(repaired_call, {1})
            self.assertTrue(
                (Path(repaired_call.staging_rules[0].destination_dir) / report.name)
                .is_file()
            )

            failed = _execution(
                call,
                status="failed",
                attempt_index=1,
                error=(
                    "enzywizard-integrate could not be started or prepared: "
                    "Staging pattern matched no files."
                ),
            )
            diagnosis = authorize_managed_input_repair(
                diagnose_tool_failure(failed), repair
            )
            self.assertEqual(diagnosis.recoverability, "automatic")
            decision = select_recovery_decision(
                diagnosis,
                AgentConfig(
                    api_key="test",
                    base_url="https://example.invalid/v1",
                    model="test",
                    workdir=workdir,
                    check_model_capabilities=False,
                ),
                0,
            )
            self.assertEqual(decision[:2], ("retry", "repair_managed_inputs"))

    def test_repair_rejects_ambiguous_or_outside_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workdir = Path(temporary)
            call = _call(workdir)
            reports = []
            for name in ("one.json", "two.json"):
                path = workdir / "01_energy" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf-8")
                reports.append(_artifact(path))
            self.assertIsNone(
                prepare_managed_input_repair(call, reports, {1}, workdir)
            )

            outside = Path(temporary).parent / "outside-energy-report.json"
            outside.write_text("{}", encoding="utf-8")
            try:
                self.assertIsNone(
                    prepare_managed_input_repair(
                        call, [_artifact(outside)], {1}, workdir
                    )
                )
            finally:
                outside.unlink(missing_ok=True)

    def test_repair_rejects_incompatible_artifact_side(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workdir = Path(temporary)
            report = workdir / "01_energy" / "energy_report.json"
            report.parent.mkdir(parents=True)
            report.write_text("{}", encoding="utf-8")
            artifact = _artifact(report)
            artifact.side = "mut"
            self.assertIsNone(
                prepare_managed_input_repair(
                    _call(workdir), [artifact], {1}, workdir
                )
            )

    def test_unrelated_path_failure_does_not_authorize_staging_repair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workdir = Path(temporary)
            report = workdir / "01_energy" / "energy_report.json"
            report.parent.mkdir(parents=True)
            report.write_text("{}", encoding="utf-8")
            call = _call(workdir)
            repair = prepare_managed_input_repair(
                call, [_artifact(report)], {1}, workdir
            )
            failed = _execution(
                call,
                status="failed",
                attempt_index=1,
                error=(
                    "enzywizard-integrate could not be started or prepared: "
                    "No such file or directory: user_input.cif."
                ),
            )
            diagnosis = authorize_managed_input_repair(
                diagnose_tool_failure(failed), repair
            )
            self.assertEqual(diagnosis.recoverability, "confirmation_required")
            self.assertNotIn("repair_managed_inputs", diagnosis.allowed_actions)

    def test_stale_output_reference_is_rebound_to_unique_registered_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workdir = Path(temporary)
            structure = workdir / "01_clean" / "cleaned.cif"
            structure.parent.mkdir(parents=True)
            structure.write_text("data_test\n", encoding="utf-8")
            artifact = Artifact(
                turn_id=1,
                plan_id=1,
                artifact_type="cleaned_structure_cif",
                path=str(structure),
                source_tool="enzywizard-clean",
                side="single",
                metadata={"step_index": 1, "required": True},
            )
            execution_dir = workdir / "03_energy"
            call = ToolCall(
                turn_id=1,
                plan_id=1,
                step_index=3,
                tool_name="enzywizard-energy",
                command_name="enzywizard-energy",
                side="single",
                tool_execution_dir=str(execution_dir),
                arguments={
                    "input_path": ToolOutputReference(
                        source_step_index=2,
                        artifact_type="cleaned_structure_cif",
                        expected_path_pattern=str(workdir / "stale" / "*.cif"),
                    ),
                    "output_dir": str(execution_dir),
                },
                output_artifacts=[],
                staging_rules=[],
                call_reason="Use a program-managed upstream structure.",
            )
            repair = prepare_managed_input_repair(call, [artifact], {1}, workdir)
            self.assertIsNotNone(repair)
            repaired_call, records = repair or (None, [])
            reference = repaired_call.arguments["input_path"]
            self.assertIsInstance(reference, ToolOutputReference)
            self.assertEqual(reference.source_step_index, 1)
            self.assertEqual(records[0].repair_type, "rebind_output_reference")

            failed = _execution(
                call,
                status="failed",
                attempt_index=1,
                error=(
                    "enzywizard-energy could not be started or prepared: Expected "
                    "exactly one artifact_type='cleaned_structure_cif' from step 2, "
                    "found 0."
                ),
            )
            diagnosis = authorize_managed_input_repair(
                diagnose_tool_failure(failed), repair
            )
            self.assertEqual(diagnosis.recoverability, "automatic")
            self.assertIn("repair_managed_inputs", diagnosis.allowed_actions)

    def test_agent_managed_workspace_collision_is_cleaned_and_retried(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workdir = Path(temporary)
            call = _call(workdir)
            failed = _execution(
                call,
                status="failed",
                attempt_index=1,
                error=(
                    "enzywizard-integrate could not be started or prepared: "
                    "Staging would overwrite an existing report."
                ),
            )
            diagnosis = diagnose_tool_failure(failed)
            self.assertEqual(diagnosis.recoverability, "automatic")
            decision = select_recovery_decision(
                diagnosis,
                AgentConfig(
                    api_key="test",
                    base_url="https://example.invalid/v1",
                    model="test",
                    workdir=workdir,
                    check_model_capabilities=False,
                ),
                0,
            )
            self.assertEqual(decision[:2], ("retry", "rebuild_managed_workspace"))

            execution_dir = Path(call.tool_execution_dir)
            conflict = execution_dir / "input_reports" / "stale.json"
            conflict.parent.mkdir(parents=True, exist_ok=True)
            conflict.write_text("{}", encoding="utf-8")
            audit_log = execution_dir / "stderr.txt"
            audit_log.write_text("failure", encoding="utf-8")
            clear_failed_attempt_outputs(execution_dir, workdir)
            self.assertFalse(conflict.exists())
            self.assertTrue(audit_log.exists())

    def test_old_plan_artifact_cannot_be_used_for_repair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workdir = Path(temporary)
            report = workdir / "old_plan" / "energy_report.json"
            report.parent.mkdir(parents=True)
            report.write_text("{}", encoding="utf-8")
            artifact = _artifact(report)
            artifact.plan_id = 99
            self.assertIsNone(
                prepare_managed_input_repair(
                    _call(workdir), [artifact], {1}, workdir
                )
            )

    def test_execution_repairs_only_failed_current_step(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workdir = Path(temporary)
            report = workdir / "01_energy" / "energy_report.json"
            report.parent.mkdir(parents=True)
            report.write_text("{}", encoding="utf-8")
            first_call = ToolCall(
                turn_id=1,
                plan_id=1,
                step_index=1,
                tool_name="enzywizard-energy",
                command_name="enzywizard-energy",
                side="single",
                tool_execution_dir=str(workdir / "01_energy"),
                arguments={},
                output_artifacts=[
                    ToolOutputArtifact(
                        artifact_type="energy_report_json",
                        path_pattern=str(report),
                        side="single",
                        required=True,
                    )
                ],
                staging_rules=[],
                call_reason="Produce the upstream report.",
            )
            second_call = _call(workdir)
            plan = ToolPlan(
                turn_id=1,
                plan_id=1,
                purpose_type="single_protein_analysis",
                summary="Recovery test",
                plan_reason="Verify bounded internal repair.",
                calls=[first_call, second_call],
            )
            planning = PlanningResult(
                turn_id=1,
                plan_id=1,
                planning_route="tool_plan_generated",
                planning_flag="tool_plan_generated",
                needs_tool=True,
                summary=plan.summary,
                purpose_type=plan.purpose_type,
                purpose_reason=plan.plan_reason,
                tool_selection=None,
                parameter_extraction=None,
                tool_plan=plan,
                error_messages=[],
            )
            state = {
                "config": AgentConfig(
                    api_key="test",
                    base_url="https://example.invalid/v1",
                    model="test",
                    workdir=workdir,
                    check_model_capabilities=False,
                    failure_diagnosis_mode="deterministic",
                ),
                "session_id": "managed-repair-test",
                "current_turn_id": 1,
                "current_plan_id": 1,
                "current_planning_result": planning,
                "current_user_confirmation": UserConfirmation(
                    turn_id=1,
                    plan_id=1,
                    decision="approved",
                    revision_request=None,
                ),
                "runtime_session": RuntimeSession(
                    started_at=utc_now(), started_monotonic=time.monotonic()
                ),
                "artifacts": [],
                "current_plan_artifacts": [],
            }
            calls: list[tuple[int, int, str]] = []

            def fake_execute(
                unused_state,
                call,
                unused_spec,
                unused_artifacts,
                unused_steps,
                *,
                attempt_index=1,
            ):
                calls.append(
                    (
                        call.step_index,
                        attempt_index,
                        call.staging_rules[0].source_path_pattern
                        if call.staging_rules
                        else "",
                    )
                )
                if call.step_index == 1:
                    return _execution(
                        call,
                        status="succeeded",
                        attempt_index=attempt_index,
                        artifacts=[_artifact(report)],
                    )
                if attempt_index == 1:
                    return _execution(
                        call,
                        status="failed",
                        attempt_index=attempt_index,
                        error=(
                            "enzywizard-integrate could not be started or prepared: "
                            "Staging pattern matched no files."
                        ),
                    )
                return _execution(
                    call,
                    status="succeeded",
                    attempt_index=attempt_index,
                )

            with (
                patch(
                    "enzywizard_agent.services.tool_calling_service.execute_tool_call",
                    side_effect=fake_execute,
                ),
                patch(
                    "enzywizard_agent.services.tool_calling_service.write_session_manifest"
                ),
            ):
                execution = execute_approved_tool_plan(state)

            self.assertEqual(execution.execution_status, "succeeded")
            self.assertEqual([item[:2] for item in calls], [(1, 1), (2, 1), (2, 2)])
            self.assertEqual(calls[-1][2], str(report.resolve()))
            self.assertEqual(len(execution.recovery_attempts), 1)
            attempt = execution.recovery_attempts[0]
            self.assertEqual(attempt.action, "repair_managed_inputs")
            self.assertEqual(len(attempt.managed_input_repairs), 1)


if __name__ == "__main__":
    unittest.main()
