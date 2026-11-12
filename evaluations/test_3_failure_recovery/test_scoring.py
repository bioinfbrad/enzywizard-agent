from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scoring import cluster_bootstrap_ci, repetition_statistics, score_case
from enzywizard_agent.data_models.tool_execution import (
    AIFailureDiagnosis,
    ToolFailureDiagnosis,
)
from enzywizard_agent.services.ai_failure_diagnosis_service import (
    review_ai_recovery_proposal,
)


def _case(*, automatic: bool = True) -> dict:
    return {
        "expected": {
            "diagnostic_category": "missing_output",
            "failure_stage": "output_contract",
            "recoverability": "automatic",
            "decision": "retry",
            "action": "retry_same_call",
            "final_status": "succeeded",
            "automatic_recovery_success": True if automatic else None,
            "scientific_change_required": False,
            "user_confirmation_required": False,
        },
        "allowed_actions": ["retry_same_call"],
        "required_trace": ["failure_diagnosis", "recovery_decision", "final_outcome"],
    }


def _observation() -> dict:
    return {
        "diagnostic_category": "missing_output",
        "failure_stage": "output_contract",
        "recoverability": "automatic",
        "explanation_complete": True,
        "decision": "retry",
        "action": "retry_same_call",
        "final_status": "succeeded",
        "scientific_change_required": False,
        "user_confirmation_required": False,
        "forbidden_actions_observed": [],
        "bounded_adjustment_applied": True,
        "adjustment_count": 1,
        "retry_count": 1,
        "trace_items": {
            "failure_diagnosis": True,
            "recovery_decision": True,
            "final_outcome": True,
        },
    }


class TestScoring(unittest.TestCase):
    def test_perfect_automatic_recovery_scores_all_metrics(self) -> None:
        metrics, details = score_case(_case(), _observation())
        self.assertEqual(metrics, {"EDA": 1.0, "EHA": 1.0, "EHPCR": 1.0, "AARSR": 1.0})
        self.assertTrue(all(details["automatic_recovery_checks"].values()))

    def test_nonautomatic_case_excludes_aarsr(self) -> None:
        metrics, _ = score_case(_case(automatic=False), _observation())
        self.assertIsNone(metrics["AARSR"])

    def test_partial_trace_is_fractional_without_changing_handling(self) -> None:
        observation = _observation()
        observation["trace_items"]["recovery_decision"] = False
        metrics, _ = score_case(_case(), observation)
        self.assertEqual(metrics["EDA"], 1.0)
        self.assertEqual(metrics["EHA"], 1.0)
        self.assertAlmostEqual(metrics["EHPCR"], 2 / 3)

    def test_forbidden_action_fails_handling_and_automatic_recovery(self) -> None:
        observation = _observation()
        observation["forbidden_actions_observed"] = ["retry"]
        metrics, details = score_case(_case(), observation)
        self.assertEqual(metrics["EHA"], 0.0)
        self.assertEqual(metrics["AARSR"], 0.0)
        self.assertFalse(details["handling_checks"]["no_forbidden_action"])

    def test_explicitly_allowed_alternative_action_is_valid(self) -> None:
        case = _case(automatic=False)
        case["allowed_actions"].append("request_user_input")
        observation = _observation()
        observation["action"] = "request_user_input"
        metrics, details = score_case(case, observation)
        self.assertEqual(metrics["EHA"], 1.0)
        self.assertTrue(details["handling_checks"]["action"])

    def test_repetition_statistics_and_cluster_bootstrap(self) -> None:
        results = []
        for case_id, value in (("A", 1.0), ("B", 0.0)):
            for repetition in (1, 2, 3):
                results.append(
                    {
                        "case_id": case_id,
                        "repetition": repetition,
                        "metrics": {
                            "EDA": value,
                            "EHA": value,
                            "EHPCR": value,
                            "AARSR": value,
                        },
                    }
                )
        statistics = repetition_statistics(results)
        self.assertEqual(statistics["EDA"]["mean"], 0.5)
        interval = cluster_bootstrap_ci(results, iterations=200, seed=7)
        self.assertLessEqual(interval["EDA"]["lower"], 0.5)
        self.assertGreaterEqual(interval["EDA"]["upper"], 0.5)


class TestRecoveryPolicy(unittest.TestCase):
    def test_scientific_change_requires_parameter_confirmation(self) -> None:
        deterministic = ToolFailureDiagnosis(
            category="scientific_configuration",
            failure_stage="subprocess",
            recoverability="confirmation_required",
            reason="The requested scientific parameter is unsupported.",
            evidence=["unsupported pH"],
            allowed_actions=["change_scientific_parameters", "request_user_input", "stop"],
            scientific_change_required=True,
        )
        proposal = AIFailureDiagnosis(
            likely_causes=["The requested pH is outside the supported range."],
            evidence_ids=["deterministic_diagnosis"],
            user_explanation="A different pH is required before execution can continue.",
            recommended_action="request_user_input",
            confidence=0.9,
            scientific_change_required=True,
        )

        decision, action, confirmation, _ = review_ai_recovery_proposal(
            proposal, deterministic, None, 0
        )

        self.assertEqual(decision, "request_confirmation")
        self.assertEqual(action, "request_parameter_confirmation")
        self.assertTrue(confirmation)

    def test_missing_dependency_cannot_be_repaired_by_chat_input(self) -> None:
        deterministic = ToolFailureDiagnosis(
            category="missing_dependency",
            failure_stage="subprocess",
            recoverability="not_recoverable",
            reason="A required executable is unavailable in the current environment.",
            evidence=["command not found"],
            allowed_actions=["stop"],
        )
        proposal = AIFailureDiagnosis(
            likely_causes=["The runtime dependency is missing."],
            evidence_ids=["stderr_excerpt", "deterministic_diagnosis"],
            user_explanation="The environment must be repaired before restarting.",
            recommended_action="request_user_input",
            confidence=0.95,
            scientific_change_required=False,
        )

        decision, action, confirmation, _ = review_ai_recovery_proposal(
            proposal, deterministic, None, 0
        )

        self.assertEqual(decision, "stop")
        self.assertEqual(action, "stop")
        self.assertFalse(confirmation)


if __name__ == "__main__":
    unittest.main()
