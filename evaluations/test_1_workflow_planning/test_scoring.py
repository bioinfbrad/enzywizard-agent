from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_test_1 import (  # noqa: E402
    _build_agent_inputs,
    _resolve_inputs,
    _seed_artifacts,
    build_state,
    load_suite,
    validate_suite,
)
from scoring import (  # noqa: E402
    cluster_bootstrap_ci,
    score_e2e,
    score_pcvr,
    score_ppcr,
    score_rac,
    score_tsa,
    score_wovr,
)


def _call(
    tool: str,
    side: str,
    step: int,
    *,
    arguments: dict | None = None,
    outputs: list[dict] | None = None,
) -> dict:
    return {
        "tool_name": tool,
        "side": side,
        "step_index": step,
        "arguments": arguments or {},
        "output_artifacts": outputs or [],
        "staging_rules": [],
    }


class TestMetricScoring(unittest.TestCase):
    def test_agent_inputs_fill_omitted_optional_fields(self) -> None:
        inputs = _build_agent_inputs({"structure_file": "/tmp/input.cif"})

        self.assertEqual(inputs.structure_file, "/tmp/input.cif")
        self.assertIsNone(inputs.msa_file)
        self.assertIsNone(inputs.mut_structure_file)
        self.assertIsNone(inputs.mut_msa_file)
        self.assertIsNone(inputs.substrate_name)
        self.assertIsNone(inputs.amino_acid_substitution)

    def test_boundary_case_can_omit_input_profile(self) -> None:
        self.assertEqual(
            _resolve_inputs({"input_profiles": {}}, {"category_id": "boundary_non_tool"}),
            {},
        )

    def test_tsa_uses_exact_call_multiset(self) -> None:
        profile = {
            "expected_calls": [
                {"tool": "enzywizard-energy", "side": "wt"},
                {"tool": "enzywizard-energy", "side": "mut"},
            ]
        }
        correct = [
            _call("enzywizard-energy", "mut", 1),
            _call("enzywizard-energy", "wt", 2),
        ]
        duplicate = correct + [_call("enzywizard-energy", "wt", 3)]
        self.assertEqual(score_tsa(correct, [profile])[0], 1.0)
        self.assertEqual(score_tsa(duplicate, [profile])[0], 0.0)

    def test_pcvr_checks_schema_and_expected_values(self) -> None:
        profile = {
            "expected_calls": [{"tool": "enzywizard-energy", "side": "single"}]
        }
        arguments = {
            "input_path": "/tmp/input.cif",
            "output_dir": "/tmp/output",
            "minimization_iteration": 50,
        }
        expectations = [
            {
                "tool": "enzywizard-energy",
                "side": "single",
                "parameter": "minimization_iteration",
                "accepted_values": [50],
            }
        ]
        calls = [_call("enzywizard-energy", "single", 1, arguments=arguments)]
        self.assertEqual(score_pcvr(calls, profile, expectations)[0], 1.0)
        calls[0]["arguments"]["minimization_iteration"] = 100
        self.assertEqual(score_pcvr(calls, profile, expectations)[0], 0.0)

    def test_wovr_allows_independent_order_but_rejects_dependency_break(self) -> None:
        profile = {
            "order_constraints": [
                ["enzywizard-clean:single", "enzywizard-energy:single"]
            ]
        }
        valid = [
            _call("enzywizard-flexibility", "single", 1),
            _call(
                "enzywizard-clean",
                "single",
                2,
                outputs=[{"artifact_type": "cleaned_structure_cif"}],
            ),
            _call(
                "enzywizard-energy",
                "single",
                3,
                arguments={
                    "input_path": {
                        "source_step_index": 2,
                        "artifact_type": "cleaned_structure_cif",
                    }
                },
            ),
        ]
        self.assertEqual(score_wovr(valid, profile)[0], 1.0)
        missing_link = [
            _call("enzywizard-clean", "single", 1),
            _call("enzywizard-energy", "single", 2),
        ]
        self.assertEqual(score_wovr(missing_link, profile)[0], 0.0)
        invalid = [
            _call("enzywizard-flexibility", "single", 1),
            _call(
                "enzywizard-energy",
                "single",
                2,
                arguments={
                    "input_path": {
                        "source_step_index": 3,
                        "artifact_type": "cleaned_structure_cif",
                    }
                },
            ),
            _call(
                "enzywizard-clean",
                "single",
                3,
                outputs=[{"artifact_type": "cleaned_structure_cif"}],
            ),
        ]
        self.assertEqual(score_wovr(invalid, profile)[0], 0.0)

        nested_invalid = [
            _call(
                "enzywizard-clean",
                "single",
                1,
                outputs=[{"artifact_type": "cleaned_structure_cif"}],
            ),
            _call(
                "enzywizard-energy",
                "single",
                2,
                arguments={
                    "nested": [
                        {
                            "source_step_index": 3,
                            "artifact_type": "cleaned_structure_cif",
                        }
                    ]
                },
            ),
        ]
        self.assertEqual(score_wovr(nested_invalid, profile)[0], 0.0)

    def test_rac_uses_fixed_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "energy.json"
            report.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
            execution = {
                "execution_status": "succeeded",
                "call_executions": [
                    {
                        "tool_name": "enzywizard-energy",
                        "artifacts": [
                            {
                                "artifact_type": "energy_report_json",
                                "side": "single",
                                "source_tool": "enzywizard-energy",
                                "path": str(report),
                            }
                        ],
                    }
                ],
            }
            expected = [
                {"artifact_type": "energy_report_json", "side": "single"},
                {"artifact_type": "integrate_report_json", "side": "single"},
            ]
            rac, checks = score_rac(execution, expected)
            self.assertEqual(rac, 0.5)
            self.assertEqual([item["valid"] for item in checks], [True, False])
            self.assertEqual(score_e2e(1.0, 1.0, 1.0, execution, rac), 0.0)

    def test_ppcr_scores_only_applicable_requirements(self) -> None:
        call = SimpleNamespace(
            step_index=1,
            tool_name="enzywizard-energy",
            side="single",
            tool_execution_dir="/tmp/out",
            arguments={"input_path": "/tmp/input.cif", "output_dir": "/tmp/out"},
            call_reason="Energy analysis was explicitly requested.",
        )
        plan = SimpleNamespace(turn_id=1, plan_id=1, calls=[call], plan_reason="Requested analysis.")
        planning = SimpleNamespace(
            tool_plan=plan,
            tool_selection=SimpleNamespace(selection_reason="The request asks for energy."),
            parameter_extraction=SimpleNamespace(
                parameter_adjustments=[], profile_requests=[]
            ),
            plan_validation=SimpleNamespace(passed=True, checks=["schema", "dependencies"]),
        )
        confirmation = SimpleNamespace(
            decision="approved", revision_request=None, turn_id=1, plan_id=1
        )
        requirements = [
            "request_to_plan_link",
            "tool_selection_reason",
            "parameter_value_and_source",
            "order_and_dependency_record",
            "final_validation_and_approval",
        ]
        score, checks = score_ppcr(
            requirements,
            state={
                "current_turn_id": 1,
                "current_user_message": "Evaluate the enzyme energy.",
                "agent_inputs": {"structure_file": "/tmp/input.cif"},
                "planning_results": [planning],
            },
            planning_results=[planning],
            confirmations=[confirmation],
            history_result=None,
            revision_request=None,
        )
        self.assertEqual(score, 1.0)
        self.assertTrue(all(item["complete"] for item in checks))
        call.arguments["input_path"] = "/tmp/unrecorded-input.cif"
        score, checks = score_ppcr(
            requirements,
            state={
                "current_turn_id": 1,
                "current_user_message": "Evaluate the enzyme energy.",
                "agent_inputs": {"structure_file": "/tmp/input.cif"},
                "planning_results": [planning],
            },
            planning_results=[planning],
            confirmations=[confirmation],
            history_result=None,
            revision_request=None,
        )
        self.assertLess(score, 1.0)
        self.assertFalse(
            next(
                item["complete"]
                for item in checks
                if item["item"] == "parameter_value_and_source"
            )
        )

    def test_ppcr_resolves_values_derived_from_historical_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            docked = Path(directory) / "docked_glucose.sdf"
            docked.write_text("test\n$$$$\n", encoding="utf-8")
            call = SimpleNamespace(
                step_index=1,
                tool_name="enzywizard-interaction",
                side="single",
                tool_execution_dir=str(Path(directory) / "interaction"),
                arguments={
                    "input_path": "/tmp/input.cif",
                    "output_dir": str(Path(directory) / "interaction"),
                    "substrate_names": docked.stem,
                    "substrate_dir": str(docked.parent),
                },
                call_reason="Reuse the selected historical docking output.",
            )
            plan = SimpleNamespace(
                turn_id=2,
                plan_id=1,
                calls=[call],
                plan_reason="Reuse historical docking output.",
            )
            planning = SimpleNamespace(
                tool_plan=plan,
                tool_selection=SimpleNamespace(selection_reason="Interaction requested."),
                parameter_extraction=SimpleNamespace(
                    parameter_adjustments=[], profile_requests=[]
                ),
                plan_validation=SimpleNamespace(passed=True, checks=["schema"]),
            )
            history = SimpleNamespace(
                selected_artifacts=[SimpleNamespace(path=str(docked))]
            )

            score, checks = score_ppcr(
                ["parameter_value_and_source"],
                state={
                    "current_turn_id": 2,
                    "current_user_message": "Reuse the docked substrate.",
                    "agent_inputs": {"structure_file": "/tmp/input.cif"},
                    "planning_results": [planning],
                },
                planning_results=[planning],
                confirmations=[],
                history_result=history,
                revision_request=None,
            )

            self.assertEqual(score, 1.0)
            self.assertTrue(checks[0]["complete"])

            call.arguments["input_path"] = docked.stem
            score, checks = score_ppcr(
                ["parameter_value_and_source"],
                state={
                    "current_turn_id": 2,
                    "current_user_message": "Reuse the docked substrate.",
                    "agent_inputs": {"structure_file": "/tmp/input.cif"},
                    "planning_results": [planning],
                },
                planning_results=[planning],
                confirmations=[],
                history_result=history,
                revision_request=None,
            )
            self.assertEqual(score, 0.0)
            self.assertIn("input_path", checks[0]["detail"])

    def test_ppcr_resolves_directory_from_an_upstream_plan_step(self) -> None:
        upstream = SimpleNamespace(
            turn_id=1,
            plan_id=1,
            step_index=1,
            tool_name="enzywizard-substrate",
            side="single",
            tool_execution_dir="/tmp/plan/01_substrate",
            arguments={
                "substrate_names": "glucose",
                "output_dir": "/tmp/plan/01_substrate",
            },
            call_reason="Prepare the requested substrate.",
        )
        downstream = SimpleNamespace(
            turn_id=1,
            plan_id=1,
            step_index=2,
            tool_name="enzywizard-dock",
            side="single",
            tool_execution_dir="/tmp/plan/02_dock",
            arguments={
                "input_path": {
                    "source_step_index": 1,
                    "artifact_type": "cleaned_structure_cif",
                },
                "substrate_names": {
                    "source_step_index": 1,
                    "artifact_type": "substrate_report_json",
                },
                "substrate_dir": "/tmp/plan/01_substrate",
                "output_dir": "/tmp/plan/02_dock",
            },
            call_reason="Dock the prepared substrate.",
        )
        plan = SimpleNamespace(
            turn_id=1,
            plan_id=1,
            calls=[upstream, downstream],
            plan_reason="Prepare and dock glucose.",
        )
        planning = SimpleNamespace(
            tool_plan=plan,
            tool_selection=SimpleNamespace(selection_reason="Docking requested."),
            parameter_extraction=SimpleNamespace(
                parameter_adjustments=[], profile_requests=[]
            ),
            plan_validation=SimpleNamespace(passed=True, checks=["dependencies"]),
        )

        score, checks = score_ppcr(
            ["parameter_value_and_source"],
            state={
                "current_turn_id": 1,
                "current_user_message": "Prepare and dock glucose.",
                "agent_inputs": {
                    "structure_file": "/tmp/input.cif",
                    "substrate_name": "glucose",
                },
                "planning_results": [planning],
            },
            planning_results=[planning],
            confirmations=[],
            history_result=None,
            revision_request=None,
        )

        self.assertEqual(score, 1.0)
        self.assertTrue(checks[0]["complete"])

    def test_ppcr_profile_only_sources_parameters_controlled_by_that_profile(self) -> None:
        call = SimpleNamespace(
            turn_id=1,
            plan_id=1,
            step_index=1,
            tool_name="enzywizard-energy",
            side="single",
            tool_execution_dir="/tmp/plan/01_energy",
            arguments={
                "input_path": "/tmp/input.cif",
                "output_dir": "/tmp/plan/01_energy",
                "minimize_energy": True,
                "minimization_iteration": 25,
            },
            call_reason="Use the requested fast energy profile.",
        )
        plan = SimpleNamespace(
            turn_id=1,
            plan_id=1,
            calls=[call],
            plan_reason="Run a fast energy analysis.",
        )
        planning = SimpleNamespace(
            tool_plan=plan,
            tool_selection=SimpleNamespace(selection_reason="Energy requested."),
            parameter_extraction=SimpleNamespace(
                parameter_adjustments=[],
                profile_requests=[
                    SimpleNamespace(
                        step_indexes=[1],
                        profile="fast",
                        profile_reason="The user requested a fast calculation.",
                    )
                ],
            ),
            plan_validation=SimpleNamespace(passed=True, checks=["parameters"]),
        )
        state = {
            "current_turn_id": 1,
            "current_user_message": "Run a fast energy analysis.",
            "agent_inputs": {"structure_file": "/tmp/input.cif"},
            "planning_results": [planning],
        }

        score, _ = score_ppcr(
            ["parameter_value_and_source"],
            state=state,
            planning_results=[planning],
            confirmations=[],
            history_result=None,
            revision_request=None,
        )
        self.assertEqual(score, 1.0)

        call.arguments["minimize_energy"] = False
        score, checks = score_ppcr(
            ["parameter_value_and_source"],
            state=state,
            planning_results=[planning],
            confirmations=[],
            history_result=None,
            revision_request=None,
        )
        self.assertEqual(score, 0.0)
        self.assertIn("minimize_energy", checks[0]["detail"])

    def test_cluster_bootstrap_keeps_case_repetitions_together(self) -> None:
        results = [
            {"case_id": "A", "repetition": repeat, "metrics": {"TSA": 1.0}}
            for repeat in range(1, 4)
        ] + [
            {"case_id": "B", "repetition": repeat, "metrics": {"TSA": 0.0}}
            for repeat in range(1, 4)
        ]
        interval = cluster_bootstrap_ci(results, iterations=500, seed=7)["TSA"]
        self.assertEqual(interval, {"lower": 0.0, "upper": 1.0})


class TestSuiteContract(unittest.TestCase):
    def test_resource_manifest_has_three_complete_representative_enzymes(self) -> None:
        manifest = HERE / "resources" / "resource_manifest.csv"
        with manifest.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        self.assertEqual({row["enzyme_id"] for row in rows}, {"1MXT", "1IU4", "1TCA"})
        path_fields = [
            field for field in rows[0] if field.endswith(("structure", "sequence", "alignment"))
        ]
        for row in rows:
            self.assertTrue(row["mutation"])
            self.assertTrue(row["substrate"])
            for field in path_fields:
                self.assertTrue((HERE / "resources" / row[field]).is_file())

    def test_every_case_builds_an_isolated_valid_state(self) -> None:
        catalog, cases = load_suite()
        args = SimpleNamespace(
            api_key="fixture-only",
            base_url="https://example.invalid/v1",
            model="fixture-only",
            reasoning_effort="medium",
            ontology_mode="retrieve",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for case in cases:
                case_dir = root / case["case_id"]
                case_dir.mkdir()
                state = build_state(catalog, case, args, case_dir)
                artifacts = state["artifacts"]

                self.assertEqual(
                    bool(artifacts), bool((case.get("history") or {}).get("artifacts")),
                    case["case_id"],
                )
                self.assertEqual(
                    len({item.path for item in artifacts}),
                    len(artifacts),
                    case["case_id"],
                )
                for artifact in artifacts:
                    path = Path(artifact.path)
                    self.assertTrue(path.is_file(), case["case_id"])
                    self.assertTrue(path.is_relative_to(case_dir), case["case_id"])
                for execution in state["tool_executions"]:
                    for call in execution.call_executions:
                        self.assertTrue(
                            set(item.path for item in call.artifacts).issubset(
                                {item.path for item in artifacts}
                            ),
                            case["case_id"],
                        )

    def test_history_fixtures_are_staged_inside_case_workdir(self) -> None:
        _, cases = load_suite()
        case = next(item for item in cases if item["case_id"] == "T1-HR09")
        with tempfile.TemporaryDirectory() as directory:
            workdir = Path(directory).resolve()
            artifacts = _seed_artifacts(case, workdir)

            self.assertTrue(artifacts)
            self.assertTrue(
                all(Path(item.path).is_relative_to(workdir) for item in artifacts)
            )
            self.assertTrue(all(Path(item.path).is_file() for item in artifacts))
            self.assertEqual(
                {Path(item.path).name for item in artifacts},
                {
                    "cleaned_1TCA-assembly1.cif",
                    "cleaned_1TCA-assembly1.fasta",
                    "clean_report_1TCA-assembly1.json",
                    "substrate_report_4-nitrophenyl_butyrate.json",
                    "dock_report_cleaned_1TCA-assembly1_4-nitrophenyl_butyrate.json",
                    "docked_4-nitrophenyl_butyrate.sdf",
                },
            )

    def test_reusable_docking_requires_complete_history_chain(self) -> None:
        catalog, cases = load_suite()
        broken = deepcopy(cases)
        case = next(item for item in broken if item["case_id"] == "T1-HR09")
        case["history"]["artifacts"] = [
            item
            for item in case["history"]["artifacts"]
            if item["artifact_type"] != "cleaned_sequence_fasta"
        ]

        with self.assertRaisesRegex(
            ValueError, "T1-HR09: reusable clean fixture is incomplete"
        ):
            validate_suite(catalog, broken)

    def test_suite_has_fixed_categories_and_metric_applicability(self) -> None:
        catalog, cases = load_suite()
        counts: dict[str, int] = {}
        for case in cases:
            counts[case["category_id"]] = counts.get(case["category_id"], 0) + 1
        self.assertEqual(len(cases), 68)
        self.assertEqual(
            counts,
            {
                "single_tool_requests": 12,
                "multi_tool_requests": 12,
                "parameters_constraints": 12,
                "historical_reuse": 12,
                "plan_revision": 12,
                "boundary_non_tool": 8,
            },
        )
        self.assertEqual(catalog["metrics"]["boundary"], ["TSA"])
        for case in cases:
            if case["category_id"] == "boundary_non_tool":
                continue
            names = (
                [case["expected_initial_plan"], case["expected_final_plan"]]
                if case["category_id"] == "plan_revision"
                else list(case.get("acceptable_plans") or [case["expected_plan"]])
            )
            for name in names:
                profile = catalog["plan_profiles"][name]
                self.assertTrue(profile["expected_calls"])
                self.assertTrue(profile["required_artifact_groups"])


if __name__ == "__main__":
    unittest.main()
