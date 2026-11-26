from copy import deepcopy
from graphlib import TopologicalSorter
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import benchmark_support as runner
from benchmark_support import load_cases
from langgraph.types import interrupt
from workflow_metrics import best_mapping, case_contracts, score_case, summarize_metrics


def clean_trace(ph=6.5):
    key = "enzywizard-clean:single"
    arguments = {"pH": ph, "add_H": False}
    return {
        "history_artifacts": [],
        "events": [
            {"kind": "routing", "stage": 1, "actions": ["run_tool_workflow"]},
            {"kind": "plan", "stage": 1, "calls": [key]},
            {"kind": "plan_call", "stage": 1, "key": key, "arguments": arguments,
             "valid": True, "dependencies": []},
            {"kind": "approval", "stage": 1, "validated": True, "linked": True},
            {"kind": "execution_call", "key": key, "arguments": arguments.copy(),
             "data": {"command": ["enzywizard-clean"]}, "dependencies": [], "valid": True,
             "final": True, "revoked": False, "succeeded": True, "target_inputs_valid": True},
            *[
                {"kind": "artifact", "artifact_type": kind, "side": "single",
                 "producer": key, "valid": True, "path": f"/test/{kind}"}
                for kind in (
                    "cleaned_structure_cif", "cleaned_structure_pdb",
                    "cleaned_sequence_fasta", "clean_report_json",
                )
            ],
            {"kind": "terminal", "status": "succeeded", "final": True},
        ],
    }


def workflow_trace(contract, arguments):
    cleaned = "/test/history/cleaned.cif"
    docked = "/test/history/docked_ligand.sdf"
    trace = {"events": [], "history_artifacts": [
        {"artifact_type": "cleaned_structure_cif", "path": cleaned},
        {"artifact_type": "docked_substrate_sdf", "path": docked},
    ]}
    history_inputs = {"input_path": cleaned, "substrate_dir": str(Path(docked).parent)}
    nodes = contract["milestones"]
    for name in TopologicalSorter({key: node["after"] for key, node in nodes.items()}).static_order():
        event = deepcopy(nodes[name])
        kind = event["kind"]
        if kind == "history":
            event.update(status="succeeded", grounded=True, turns=[event["turn"]],
                         artifact_types=event["accepted_artifact_types"])
        if kind == "plan":
            event["calls"] = event["allowed"]
        if kind in {"plan_call", "execution_call"}:
            values = arguments.get(event["key"], {}).copy()
            for key in event["reuse_inputs"]:
                values[key] = history_inputs[key]
            if "substrate_dir" in event["reuse_inputs"]:
                values["substrate_names"] = Path(docked).stem
            event.update(arguments=values, valid=True)
        if kind == "execution_call":
            event.update(data={"command": [event["key"]]}, final=True, revoked=False,
                         succeeded=True, target_inputs_valid=True)
        if kind == "artifact":
            event.update(valid=True, producer=event["producers"][0], path=f"/test/{name}")
        if kind == "approval":
            event.update(validated=True, linked=True)
        if kind == "terminal":
            event.update(status="succeeded", final=True)
        trace["events"].append(event)
    return trace


class WorkflowMetricsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog, cases = load_cases()
        cls.cases = {case["case_id"]: case for case in cases}
        cls.contracts = {key: case_contracts(case) for key, case in cls.cases.items()}

    def test_workflow_subset_and_alternatives(self):
        self.assertEqual(len(self.cases), 60)
        self.assertEqual(sum(map(len, self.contracts.values())), 61)
        self.assertEqual(
            [item["profiles"] for item in self.contracts["T1-HR10"]],
            [["reused_flexibility"], ["flexibility"]],
        )
        self.assertTrue(all(not key.startswith("T1-B") for key in self.cases))

    def test_confirmation_interrupt_uses_the_runner_graph_dependencies(self):
        def confirm(state):
            return {"approved": interrupt("confirm a plan")}

        graph = runner.StateGraph(dict)
        graph.add_node("confirm", confirm)
        graph.set_entry_point("confirm")
        graph.add_edge("confirm", runner.END)
        compiled = graph.compile(checkpointer=runner.create_in_memory_checkpointer())
        result = compiled.invoke({}, config={"configurable": {"thread_id": "test1-interrupt"}})
        self.assertEqual(result["__interrupt__"][0].value, "confirm a plan")

    def test_partial_order_and_transitive_dependency(self):
        dag = {"clean": [], "energy": ["clean"], "flex": ["clean"], "integrate": ["energy", "flex"]}
        self.assertEqual(len(best_mapping(dag, {
            "clean": [0], "energy": [2], "flex": [1], "integrate": [3],
        })), 4)
        self.assertEqual(len(best_mapping(dag, {
            "clean": [4], "energy": [], "flex": [], "integrate": [3],
        })), 1)

    def test_maximum_mapping_uses_distinct_events(self):
        dag = {"a": [], "b": [], "c": ["b"]}
        self.assertEqual(best_mapping(dag, {
            "a": [0, 2], "b": [0, 3], "c": [1],
        }), {"a": 2, "b": 0, "c": 1})

    def test_full_score_and_invalid_artifact(self):
        trace = clean_trace()
        rules = self.contracts["T1-PC01"]
        self.assertEqual(score_case(rules, trace, execute=True)[0], {"Avg Score": 1.0, "SR": 1.0})
        trace["events"][5]["valid"] = False
        metrics, detail = score_case(rules, trace, execute=True)
        self.assertEqual(metrics["SR"], 0.0)
        self.assertGreater(metrics["Avg Score"], 0.0)
        self.assertLess(metrics["Avg Score"], 1.0)
        self.assertEqual(detail["alternatives"][0]["minefields"], [])

    def test_review_text_does_not_change_the_scores(self):
        case = deepcopy(self.cases["T1-PC01"])
        case["reference_answers"][0]["answer"] = "A differently worded reference answer."
        case["expected_trajectories"][0]["events"] = ["A different written summary of the trajectory."]
        original = score_case(self.contracts["T1-PC01"], clean_trace(), execute=True)
        self.assertEqual(score_case(case_contracts(case), clean_trace(), execute=True), original)

    def test_plan_error_and_execution_success_are_distinct(self):
        trace = clean_trace()
        trace["events"][2]["arguments"] = {"pH": 7.0, "add_H": False}
        metrics, _ = score_case(self.contracts["T1-PC01"], trace, execute=True)
        self.assertLess(metrics["Avg Score"], 1.0)
        self.assertEqual(metrics["SR"], 1.0)
        trace["events"][4]["arguments"]["pH"] = 7.0
        self.assertEqual(score_case(self.contracts["T1-PC01"], trace, execute=True)[0]["SR"], 0.0)

    def test_extra_execution_and_revoked_plan(self):
        trace = clean_trace()
        trace["events"].insert(5, {**trace["events"][4], "key": "enzywizard-energy:single"})
        metrics, detail = score_case(self.contracts["T1-PC01"], trace, execute=True)
        self.assertEqual(metrics["Avg Score"], 0.0)
        self.assertEqual(metrics["SR"], 0.0)
        self.assertIn("unexpected_execution_call", {
            x["reason"] for x in detail["alternatives"][0]["minefields"]
        })
        trace = clean_trace()
        trace["events"][4]["revoked"] = True
        self.assertEqual(score_case(self.contracts["T1-PC01"], trace, execute=True)[0], {
            "Avg Score": 0.0, "SR": 0.0,
        })

    def test_unprepared_extra_tool_does_not_trigger_execution_minefield(self):
        rules = self.contracts["T1-PC01"]
        trace = clean_trace()
        trace["events"][-1]["status"] = "failed"
        partial, _ = score_case(rules, trace, execute=True)
        self.assertGreater(partial["Avg Score"], 0.0)
        self.assertEqual(partial["SR"], 0.0)

        extra = {
            **trace["events"][4], "key": "enzywizard-energy:single",
            "arguments": {}, "valid": False, "succeeded": False,
            "data": {"command": []},
        }
        trace["events"].insert(-1, extra)
        metrics, detail = score_case(rules, trace, execute=True)
        self.assertEqual(metrics, partial)
        self.assertEqual(detail["alternatives"][0]["minefields"], [])

        extra["data"]["command"] = ["enzywizard-energy"]
        metrics, detail = score_case(rules, trace, execute=True)
        self.assertEqual(metrics, {"Avg Score": 0.0, "SR": 0.0})
        self.assertEqual(detail["alternatives"][0]["minefields"][0]["reason"], "unexpected_execution_call")

    def test_revision_initial_tool_is_allowed_until_cancelled(self):
        rules = self.contracts["T1-RV08"]
        initial = {"kind": "plan_call", "stage": 1, "key": "enzywizard-dock:single",
                   "arguments": {}, "dependencies": [], "valid": True}
        trace = {"history_artifacts": [], "events": [initial]}
        _, detail = score_case(rules, trace, execute=True)
        self.assertEqual(detail["alternatives"][0]["minefields"], [])
        trace["events"].append({
            **initial, "kind": "execution_call", "revoked": True, "final": False,
            "succeeded": True, "data": {"command": ["enzywizard-dock"]},
        })
        _, detail = score_case(rules, trace, execute=True)
        self.assertIn("executed_revoked_plan", {
            x["reason"] for x in detail["alternatives"][0]["minefields"]
        })

    def test_historical_wrong_input_is_minefield(self):
        trace = {"history_artifacts": [{"path": "/fixed/cleaned.cif", "artifact_type": "cleaned_structure_cif"}],
                 "events": [{
                     "kind": "execution_call", "key": "enzywizard-energy:single", "valid": True,
                     "revoked": False, "data": {"command": ["enzywizard-energy"]},
                     "arguments": {"input_path": "/different/cleaned.cif"}, "dependencies": [],
                 }]}
        metrics, detail = score_case(self.contracts["T1-HR01"], trace, execute=True)
        self.assertEqual(metrics["Avg Score"], 0.0)
        self.assertEqual(detail["alternatives"][0]["minefields"][0]["reason"], "wrong_historical_input")

    def test_plan_only_and_empty_execution(self):
        trace = clean_trace()
        trace["events"] = trace["events"][:4]
        rules = self.contracts["T1-PC01"]
        self.assertEqual(score_case(rules, trace, execute=False)[0], {"Plan Score": 1.0})
        self.assertEqual(score_case(rules, trace, execute=True)[0]["SR"], 0.0)

    def test_energy_iterations_require_minimization_and_accept_tool_default(self):
        for case_id, iterations in (("T1-PC02", 25), ("T1-PC10", 250)):
            with self.subTest(case=case_id):
                rules = self.contracts[case_id]
                energy = "enzywizard-energy:single"
                trace = workflow_trace(rules[0], {
                    energy: {"minimization_iteration": iterations},
                    "enzywizard-flexibility:single": {"cutoff": 12.0, "n_modes": 12},
                })
                self.assertEqual(score_case(rules, trace, execute=True)[0], {"Avg Score": 1.0, "SR": 1.0})
                for event in trace["events"]:
                    if event.get("key") == energy:
                        event["arguments"]["minimize_energy"] = True
                self.assertEqual(score_case(rules, trace, execute=True)[0], {"Avg Score": 1.0, "SR": 1.0})
                for event in trace["events"]:
                    if event.get("key") == energy:
                        event["arguments"]["minimize_energy"] = False
                metrics, _ = score_case(rules, trace, execute=True)
                self.assertLess(metrics["Avg Score"], 1.0)
                self.assertEqual(metrics["SR"], 0.0)

    def test_explicit_substrate_identity_accepts_equivalent_representations(self):
        rules = self.contracts["T1-MT09"]
        key = "enzywizard-substrate:single"
        for ligand, sr in (
            ("4-nitrophenyl butyrate", 1.0),
            ("4-nitrophenyl butanoate", 1.0),
            ("CCCC(=O)Oc1ccc([N+](=O)[O-])cc1", 1.0),
            ("Cholesterol", 0.0),
        ):
            with self.subTest(ligand=ligand):
                trace = workflow_trace(rules[0], {key: {"substrate_names": ligand}})
                metrics, _ = score_case(rules, trace, execute=True)
                self.assertEqual(metrics["Avg Score"] == 1.0, bool(sr))
                self.assertEqual(metrics["SR"], sr)

    def test_batch_requires_substrate_and_docking_and_interaction_statistics(self):
        for case, key, kind, prefixes in (
            ("T1-ST11", "enzywizard-batch:single", "integrate_report_json", ("",)),
            ("T1-ST12", "enzywizard-mut-batch:shared", "mut_integrate_report_json", ("wild_type_", "mutant_")),
        ):
            with self.subTest(case=case), TemporaryDirectory() as directory:
                rules = self.contracts[case]
                trace = workflow_trace(rules[0], {key: {"substrate_names": "4-nitrophenyl butyrate"}})
                report = Path(directory) / "report.json"
                stats = {}
                for prefix in prefixes:
                    stats[prefix + "enzyme_substrate_binding_affinity"] = -6.4
                    stats[prefix + "hydrogen_bond_count"] = 0
                report.write_text(json.dumps({"overall_statistics": stats}), encoding="utf-8")
                for event in trace["events"]:
                    if event.get("artifact_type") == kind:
                        event["path"] = str(report)
                self.assertEqual(score_case(rules, trace, execute=True)[0], {"Avg Score": 1.0, "SR": 1.0})
                for missing in stats:
                    incomplete = {field: value for field, value in stats.items() if field != missing}
                    report.write_text(json.dumps({"overall_statistics": incomplete}), encoding="utf-8")
                    metrics, _ = score_case(rules, trace, execute=True)
                    self.assertLess(metrics["Avg Score"], 1.0)
                    self.assertEqual(metrics["SR"], 0.0)
                report.write_text(json.dumps({"overall_statistics": stats}), encoding="utf-8")
                for event in trace["events"]:
                    if event.get("key") == key:
                        event["arguments"].pop("substrate_names")
                metrics, _ = score_case(rules, trace, execute=True)
                self.assertLess(metrics["Avg Score"], 1.0)
                self.assertEqual(metrics["SR"], 0.0)

    def test_historical_requests_reject_prohibited_extra_executions_in_sr(self):
        for case, extra in (("T1-HR09", "enzywizard-dock:single"), ("T1-HR11", "enzywizard-disorder:single")):
            with self.subTest(case=case):
                rules = self.contracts[case]
                trace = workflow_trace(rules[0], {
                    "enzywizard-clean:single": {"add_H": True, "pH": 7.0},
                    "enzywizard-energy:single": {"minimize_energy": True, "minimization_iteration": 100},
                })
                self.assertEqual(score_case(rules, trace, execute=True)[0], {"Avg Score": 1.0, "SR": 1.0})
                existing = next(event for event in trace["events"] if event["kind"] == "execution_call")
                trace["events"].insert(-1, {**existing, "key": extra})
                self.assertEqual(score_case(rules, trace, execute=True)[0], {"Avg Score": 0.0, "SR": 0.0})

    def test_case_cluster_bootstrap_uses_all_repetitions(self):
        rows = [
            {"case_id": case_id, "repetition": repeat, "metrics": {"Avg Score": value, "SR": value}}
            for case_id, values in (("a", (0, 0, 0)), ("b", (1, 1, 1)))
            for repeat, value in enumerate(values, 1)
        ]
        scores = summarize_metrics(rows, iterations=200, seed=7)
        self.assertEqual(scores["SR"]["by_repetition"], {"1": 0.5, "2": 0.5, "3": 0.5})
        self.assertEqual(scores["SR"]["mean"], 0.5)
        self.assertEqual(scores["SR"]["sd"], 0.0)
        self.assertEqual(scores["SR"]["bootstrap_95_ci"], {"lower": 0.0, "upper": 1.0})


if __name__ == "__main__":
    unittest.main()
