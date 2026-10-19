"""离线验证统一模型契约、预算、manifest 和隐私边界。"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
from typing import Any

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ConfigDict

from enzywizard_agent.conf import AgentConfig
from enzywizard_agent.data_models.agent_inputs import AgentInputs
from enzywizard_agent.data_models.analysis import AnalysisResult
from enzywizard_agent.data_models.artifact import Artifact
from enzywizard_agent.data_models.runtime import (
    ModelContractError,
    RuntimeBudgetError,
    RuntimeSession,
    SessionTimeoutError,
)
import enzywizard_agent.nodes.analysis as analysis_node_module
import enzywizard_agent.services.llm_service as llm_service
from enzywizard_agent.services.manifest_service import write_session_manifest
from enzywizard_agent.services.runtime_service import (
    enforce_session_deadline,
    ensure_runtime_session,
    utc_now,
)
from enzywizard_agent.state import AgentState
from enzywizard_agent.utils.security_utils import redact_text


class ContractOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str


class FakeRawResponse:
    def __init__(self) -> None:
        self.usage_metadata = {
            "input_tokens": 11,
            "output_tokens": 3,
            "total_tokens": 14,
        }
        self.response_metadata: dict[str, Any] = {}


class FakeNullableUsageRawResponse:
    def __init__(self) -> None:
        self.usage_metadata = None
        self.response_metadata = {
            "token_usage": {
                "prompt_tokens": 11,
                "completion_tokens": 3,
                "total_tokens": 14,
                "completion_tokens_details": None,
            }
        }


class FakeContentRawResponse(FakeRawResponse):
    def __init__(self, content: Any) -> None:
        super().__init__()
        self.content = content


class FakeNoUsageContentResponse:
    def __init__(self, content: Any) -> None:
        self.content = content
        self.usage_metadata = None
        self.response_metadata: dict[str, Any] = {}


class SequenceStructuredModel:
    def __init__(self, results: list[Any]) -> None:
        self.results = iter(results)

    def invoke(self, _messages: list[Any]) -> Any:
        result = next(self.results)
        if isinstance(result, BaseException):
            raise result
        return result


class SequenceOpenAICompatibleModel:
    def __init__(self, results: list[Any]) -> None:
        self.structured = SequenceStructuredModel(results)

    def with_structured_output(
        self,
        _schema: type[BaseModel],
        **kwargs: Any,
    ) -> SequenceStructuredModel:
        if kwargs.get("method") != "json_schema" or kwargs.get("strict") is not True:
            raise AssertionError("Repair calls must retain native strict JSON Schema")
        return self.structured


class FakeTextModel:
    def __init__(self, content: Any) -> None:
        self.content = content

    def invoke(self, _messages: list[Any]) -> FakeContentRawResponse:
        return FakeContentRawResponse(self.content)


class FakeStructuredModel:
    def __init__(self, schema: type[BaseModel], calls: list[dict[str, Any]]) -> None:
        self.schema = schema
        self.calls = calls

    def invoke(self, messages: list[Any]) -> dict[str, Any]:
        contains_json_instruction = any(
            "json" in str(message.content).casefold() for message in messages
        )
        if not contains_json_instruction:
            raise AssertionError("Structured requests must explicitly mention JSON")
        self.calls.append(
            {
                "message_count": len(messages),
                "schema": self.schema.__name__,
                "contains_json_instruction": contains_json_instruction,
            }
        )
        if "actions" in self.schema.model_fields:
            parsed = self.schema(
                actions=[
                    {
                        "kind": "run_workflow",
                        "target": "capability probe",
                    }
                ]
            )
        elif {"action_id", "turn_scope", "turn_ids"}.issubset(
            self.schema.model_fields
        ):
            parsed = self.schema(
                action_id=0,
                turn_scope="specific",
                turn_ids=[1],
            )
        elif "parameter_adjustments" in self.schema.model_fields:
            parsed = self.schema(
                status="ready",
                parameter_adjustments=[
                    {
                        "step_index": 1,
                        "parameter_name": "probe_parameter",
                        "parameter_value": 7,
                        "adjustment_reason": "capability probe",
                    }
                ],
                optional_note=None,
                enabled=True,
            )
        else:
            parsed = self.schema(value="ok")
        return {
            "raw": FakeRawResponse(),
            "parsed": parsed,
            "parsing_error": None,
        }


class FakeOpenAICompatibleModel:
    def __init__(self, profile: str) -> None:
        self.profile = profile
        self.calls: list[dict[str, Any]] = []

    def with_structured_output(
        self,
        schema: type[BaseModel],
        **kwargs: Any,
    ) -> FakeStructuredModel:
        if kwargs.get("method") != "json_schema" or kwargs.get("strict") is not True:
            raise AssertionError(f"{self.profile} did not receive the strict JSON Schema contract")
        return FakeStructuredModel(schema, self.calls)


class LegacyStructuredModel:
    def with_structured_output(self, _schema: type[BaseModel]) -> FakeStructuredModel:
        raise AssertionError("Legacy structured output must not be invoked")


def make_config(workdir: Path, **updates: Any) -> AgentConfig:
    values: dict[str, Any] = {
        "api_key": "private-api-key-123456789",
        "base_url": "https://example.invalid/v1",
        "model": "contract-model",
        "workdir": workdir,
        "check_model_capabilities": False,
    }
    values.update(updates)
    return AgentConfig(**values)


def make_state(workdir: Path, **config_updates: Any) -> AgentState:
    return {
        "config": make_config(workdir, **config_updates),
        "session_id": "runtime-contract",
        "current_turn_id": 1,
        "current_plan_id": 0,
        "current_user_message": "Test runtime contracts.",
        "user_messages": [],
        "planning_results": [],
        "tool_executions": [],
        "analysis_results": [],
        "artifacts": [],
        "runtime_session": RuntimeSession(
            started_at=utc_now(),
            started_monotonic=time.monotonic(),
        ),
    }


def test_factory_contract(workdir: Path) -> None:
    captured: list[dict[str, Any]] = []
    original = llm_service.ChatOpenAI
    llm_service.ChatOpenAI = lambda **kwargs: captured.append(kwargs) or kwargs
    try:
        llm_service.create_chat_model(make_config(workdir))
        llm_service.create_chat_model(
            make_config(
                workdir,
                temperature=0.2,
                max_completion_tokens=1024,
                reasoning_effort="low",
                llm_extra_body={"enable_thinking": False},
            )
        )
    finally:
        llm_service.ChatOpenAI = original

    default, explicit = captured
    if default.get("use_responses_api") is not False:
        raise AssertionError("The factory did not pin OpenAI Chat Completions")
    if any(name in default for name in ("temperature", "reasoning_effort", "extra_body")):
        raise AssertionError("Optional provider parameters were sent without user input")
    if explicit.get("extra_body") != {"enable_thinking": False}:
        raise AssertionError("Explicit provider parameters were not preserved")
    print("[PASS] 统一 LLM Factory 只发送标准参数和显式扩展参数。")


def test_two_compatible_profiles(workdir: Path) -> None:
    original = llm_service.create_chat_model_for_state
    llm_service._CAPABILITY_CACHE.clear()
    try:
        for profile in ("qwen-openai-compatible", "gemini-openai-compatible"):
            state = make_state(
                workdir / profile,
                api_key=f"{profile}-key",
                base_url=f"https://{profile}.invalid/v1",
                model=f"{profile}-model",
                check_model_capabilities=True,
            )
            fake_model = FakeOpenAICompatibleModel(profile)
            llm_service.create_chat_model_for_state = lambda _state, model=fake_model: model
            result = llm_service.check_model_capabilities(state)
            if result.status != "passed" or not result.strict_structured_output:
                raise AssertionError(f"Compatibility profile failed: {profile}")
            expected_schemas = [
                "_CapabilityRoutingProbe",
                "_CapabilityHistoryProbe",
                "_CapabilityPlanningProbe",
            ]
            if [call["schema"] for call in fake_model.calls] != expected_schemas:
                raise AssertionError(
                    f"Capability probes were incomplete for {profile}: "
                    f"{fake_model.calls}"
                )

        diagnostic_state = make_state(
            workdir / "planning-probe-failure",
            api_key="planning-probe-failure-key",
            base_url="https://planning-probe-failure.invalid/v1",
            model="planning-probe-failure-model",
            check_model_capabilities=True,
        )
        diagnostic_model = SequenceOpenAICompatibleModel(
            [
                {
                    "raw": FakeRawResponse(),
                    "parsed": llm_service._CapabilityRoutingProbe(
                        actions=[
                            {
                                "kind": "run_workflow",
                                "target": "capability probe",
                            }
                        ]
                    ),
                    "parsing_error": None,
                },
                {
                    "raw": FakeRawResponse(),
                    "parsed": llm_service._CapabilityHistoryProbe(
                        action_id=0,
                        turn_scope="specific",
                        turn_ids=[1],
                    ),
                    "parsing_error": None,
                },
                {
                    "raw": FakeContentRawResponse('{"status": "ready"}'),
                    "parsed": None,
                    "parsing_error": ValueError("missing planning fields"),
                },
            ]
        )
        llm_service.create_chat_model_for_state = lambda _state: diagnostic_model
        try:
            llm_service.check_model_capabilities(diagnostic_state)
        except ModelContractError:
            diagnostic = diagnostic_state["runtime_session"].model_capability
            if diagnostic is None:
                raise AssertionError("Failed probe did not record capability details")
            if not diagnostic.chat_completions or not diagnostic.system_messages:
                raise AssertionError(
                    "A later probe failure erased earlier transport capabilities"
                )
            if "planning_structure" not in diagnostic.detail:
                raise AssertionError(
                    f"Failed probe stage was not reported: {diagnostic.detail}"
                )
        else:
            raise AssertionError("An invalid planning structure probe was accepted")

        routing_failure_state = make_state(
            workdir / "routing-probe-structure-failure",
            api_key="routing-probe-structure-failure-key",
            base_url="https://routing-probe-structure-failure.invalid/v1",
            model="routing-probe-structure-failure-model",
            check_model_capabilities=True,
        )
        routing_failure_model = SequenceOpenAICompatibleModel(
            [
                {
                    "raw": FakeContentRawResponse(
                        '{"kind": "run_workflow", "target": "capability probe"}'
                    ),
                    "parsed": None,
                    "parsing_error": ValueError("missing actions wrapper"),
                }
            ]
        )
        llm_service.create_chat_model_for_state = (
            lambda _state: routing_failure_model
        )
        try:
            llm_service.check_model_capabilities(routing_failure_state)
        except ModelContractError:
            diagnostic = routing_failure_state["runtime_session"].model_capability
            if diagnostic is None:
                raise AssertionError("Routing failure did not record capability details")
            if not diagnostic.chat_completions or not diagnostic.system_messages:
                raise AssertionError(
                    "A returned malformed response was mistaken for transport failure"
                )
            if "strict_schema_validation" not in diagnostic.detail:
                raise AssertionError(
                    f"Schema failure was not classified correctly: {diagnostic.detail}"
                )
        else:
            raise AssertionError("A malformed routing capability response was accepted")

        legacy_state = make_state(
            workdir / "legacy-structured-output",
            api_key="legacy-key",
            base_url="https://legacy.invalid/v1",
            model="legacy-model",
            check_model_capabilities=True,
        )
        legacy_model = LegacyStructuredModel()
        llm_service.create_chat_model_for_state = lambda _state: legacy_model
        try:
            llm_service.check_model_capabilities(legacy_state)
        except ModelContractError:
            pass
        else:
            raise AssertionError(
                "A client without native strict JSON Schema support was accepted"
            )
    finally:
        llm_service.create_chat_model_for_state = original
        llm_service._CAPABILITY_CACHE.clear()
    print(
        "[PASS] OpenAI-compatible 端点需通过三阶段严格 JSON Schema 契约。"
    )


def test_usage_and_budgets(workdir: Path) -> None:
    state = make_state(workdir, max_llm_calls=1)
    model = FakeOpenAICompatibleModel("budget")
    output = llm_service.invoke_structured(
        state,
        stage="runtime.contract",
        schema=ContractOutput,
        messages=[HumanMessage(content="Return the contract output.")],
        llm=model,
    )
    if output.value != "ok" or state["runtime_session"].input_tokens != 11:
        raise AssertionError("Exact token usage was not recorded")
    try:
        llm_service.invoke_structured(
            state,
            stage="runtime.contract.second",
            schema=ContractOutput,
            messages=[HumanMessage(content="This call must be blocked.")],
            llm=model,
        )
    except RuntimeBudgetError:
        pass
    else:
        raise AssertionError("The LLM call budget did not stop a second call")
    if state["runtime_session"].status != "budget_exceeded":
        raise AssertionError("Budget exhaustion did not update runtime status")

    timeout_state = make_state(workdir / "deadline", session_timeout_seconds=0.01)
    timeout_state["runtime_session"].started_monotonic = time.monotonic() - 1
    try:
        enforce_session_deadline(timeout_state, "runtime.deadline")
    except SessionTimeoutError:
        pass
    else:
        raise AssertionError("The session deadline did not stop execution")
    print("[PASS] LLM 调用、Token 和会话时间预算具有确定性终止行为。")


def test_nullable_usage_metadata() -> None:
    usage = llm_service._extract_usage(FakeNullableUsageRawResponse())
    if usage != (11, 3, 14, None):
        raise AssertionError(f"Nullable token details were parsed incorrectly: {usage}")

    missing_usage = type(
        "MissingUsageRawResponse",
        (),
        {
            "usage_metadata": None,
            "response_metadata": {"token_usage": None},
        },
    )()
    if llm_service._extract_usage(missing_usage) != (None, None, None, None):
        raise AssertionError("Missing token usage was not normalized safely")

    zero_reasoning = type(
        "ZeroReasoningRawResponse",
        (),
        {
            "usage_metadata": {
                "input_tokens": 11,
                "output_tokens": 3,
                "total_tokens": 14,
                "output_token_details": {"reasoning": 0},
            },
            "response_metadata": {},
        },
    )()
    if llm_service._extract_usage(zero_reasoning) != (11, 3, 14, 0):
        raise AssertionError("A zero reasoning-token count was not preserved")
    print("[PASS] 可空或缺失的 OpenAI-compatible Token 元数据可安全解析。")


def test_structured_recovery_and_text(workdir: Path) -> None:
    normalized_state = make_state(workdir / "normalize")
    malformed = {
        "raw": FakeContentRawResponse('{"answer": "normalized"}'),
        "parsed": None,
        "parsing_error": ValueError("wrong field name"),
    }
    normalized = llm_service.invoke_structured(
        normalized_state,
        stage="runtime.normalize",
        schema=ContractOutput,
        messages=[HumanMessage(content="Return a contract value.")],
        llm=SequenceOpenAICompatibleModel([malformed]),
        normalizer=lambda candidate: {"value": candidate.get("answer")},
    )
    if normalized.value != "normalized":
        raise AssertionError("Deterministic structured normalization was not used")
    if len(normalized_state["runtime_session"].llm_calls) != 1:
        raise AssertionError("Local normalization unexpectedly called the model twice")

    repaired_state = make_state(workdir / "repair")
    invalid = {
        "raw": FakeContentRawResponse('{"wrong": "value"}'),
        "parsed": None,
        "parsing_error": ValueError("missing value"),
    }
    repaired = {
        "raw": FakeContentRawResponse('{"value": "repaired"}'),
        "parsed": ContractOutput(value="repaired"),
        "parsing_error": None,
    }
    repaired_output = llm_service.invoke_structured(
        repaired_state,
        stage="runtime.repair",
        schema=ContractOutput,
        messages=[HumanMessage(content="Return a contract value.")],
        llm=SequenceOpenAICompatibleModel([invalid, repaired]),
    )
    if repaired_output.value != "repaired":
        raise AssertionError("One bounded strict-schema repair was not accepted")
    if [call.status for call in repaired_state["runtime_session"].llm_calls] != [
        "failed",
        "succeeded",
    ]:
        raise AssertionError("Structured repair calls were not recorded independently")

    estimated_failure_state = make_state(workdir / "estimated-failure")
    malformed_without_usage = {
        "raw": FakeNoUsageContentResponse('{"wrong": "value"}'),
        "parsed": None,
        "parsing_error": ValueError("missing value"),
    }
    repaired_without_usage = {
        "raw": FakeContentRawResponse('{"value": "repaired"}'),
        "parsed": ContractOutput(value="repaired"),
        "parsing_error": None,
    }
    llm_service.invoke_structured(
        estimated_failure_state,
        stage="runtime.estimated_failure",
        schema=ContractOutput,
        messages=[HumanMessage(content="Return a contract value.")],
        llm=SequenceOpenAICompatibleModel(
            [malformed_without_usage, repaired_without_usage]
        ),
    )
    failed_call = estimated_failure_state["runtime_session"].llm_calls[0]
    if not failed_call.output_tokens or failed_call.usage_quality != "estimated":
        raise AssertionError("Malformed output tokens were not estimated")
    if not estimated_failure_state["runtime_session"].errors[0].recoverable:
        raise AssertionError("A repairable contract failure was not marked recoverable")

    api_failure_state = make_state(workdir / "api-failure")
    try:
        llm_service.invoke_structured(
            api_failure_state,
            stage="runtime.api_failure",
            schema=ContractOutput,
            messages=[HumanMessage(content="Return a contract value.")],
            llm=SequenceOpenAICompatibleModel([RuntimeError("provider unavailable")]),
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("Provider failure was unexpectedly repaired")
    if api_failure_state["runtime_session"].errors[0].recoverable:
        raise AssertionError("Provider failure was incorrectly marked recoverable")

    text_state = make_state(workdir / "text")
    text_output = llm_service.invoke_text(
        text_state,
        stage="runtime.text",
        messages=[HumanMessage(content="Return plain text.")],
        llm=FakeTextModel([{"type": "text", "text": "plain answer"}]),
    )
    if text_output != "plain answer":
        raise AssertionError("Plain response content blocks were not normalized")
    print("[PASS] 结构规范化、一次严格修复和纯文本回答共享统一运行时台账。")


def test_runtime_mapping_rehydration(workdir: Path) -> None:
    state = make_state(workdir)
    state["runtime_session"] = state["runtime_session"].model_dump(mode="json")

    runtime = ensure_runtime_session(state)
    if not isinstance(runtime, RuntimeSession):
        raise AssertionError("Runtime mapping was not rehydrated as RuntimeSession")
    if state["runtime_session"] is not runtime:
        raise AssertionError("Rehydrated RuntimeSession was not written back to state")
    print("[PASS] 旧 checkpoint 的运行时映射可恢复为 RuntimeSession。")


def test_budget_analysis_exit(workdir: Path) -> None:
    state = make_state(workdir, max_turns=3)
    state["runtime_session"].status = "budget_exceeded"
    state["runtime_session"].stop_reason = "max_llm_calls"
    fallback = AnalysisResult(
        turn_id=1,
        response_mode="scientific_result",
        analysis_status="partial",
        analysis_route="answer_generated",
        termination_reason=None,
        summary="Partial evidence is available.",
        final_answer="Partial evidence is available.",
        report_summaries=[],
        findings=[],
        evidence=[],
        referenced_artifacts=[],
        limitations=[],
        error_messages=[],
        used_language_model=False,
    )
    original = analysis_node_module.analyze_agent_state
    analysis_node_module.analyze_agent_state = lambda _state: fallback
    try:
        result_state = analysis_node_module.analysis_node(state)
    finally:
        analysis_node_module.analyze_agent_state = original
    result = result_state["current_analysis_result"]
    if result.analysis_route != "exit_requested" or result.termination_reason != "budget_exceeded":
        raise AssertionError("Budget exhaustion did not stop the multi-turn loop")
    print("[PASS] Analysis 在预算耗尽时保留部分回答并结束多轮循环。")


def test_manifest_and_redaction(workdir: Path) -> None:
    secret_key = "private-api-key-123456789"
    workdir.mkdir(parents=True, exist_ok=True)
    state = make_state(
        workdir,
        api_key=secret_key,
        base_url=(
            "https://private-user:private-password@example.invalid/v1"
            "?api_key=query-secret"
        ),
        llm_extra_body={"Authorization": f"Bearer {secret_key}"},
    )
    input_path = workdir / "private-input.pdb"
    input_path.write_text("PRIVATE_INPUT_CONTENT", encoding="utf-8")
    session_dir = workdir / "session_runtime-contract"
    session_dir.mkdir(parents=True, exist_ok=True)
    report_path = session_dir / "private-report.json"
    report_path.write_text('{"patient_name": "PRIVATE_REPORT_CONTENT"}', encoding="utf-8")
    state["agent_inputs"] = AgentInputs(
        structure_file=str(input_path),
        msa_file=None,
        mut_structure_file=None,
        mut_msa_file=None,
        substrate_name="private-substrate",
        amino_acid_substitution=None,
    )
    state["artifacts"] = [
        Artifact(
            turn_id=1,
            plan_id=1,
            artifact_type="energy_report_json",
            path=str(report_path),
            source_tool="enzywizard-energy",
            side="single",
            metadata={},
        )
    ]
    manifest_path = write_session_manifest(state)
    manifest_text = manifest_path.read_text(encoding="utf-8")
    forbidden = (
        secret_key,
        "private-user",
        "private-password",
        "query-secret",
        "PRIVATE_INPUT_CONTENT",
        "PRIVATE_REPORT_CONTENT",
        str(Path.home()),
    )
    leaked = [value for value in forbidden if value in manifest_text]
    if leaked:
        raise AssertionError(f"Manifest leaked private values: {leaked}")
    payload = json.loads(manifest_text)
    if not payload["inputs"]["structure_file"]["sha256"]:
        raise AssertionError("Input checksum is missing from the manifest")
    if payload["artifacts"][0]["path"] != "private-report.json":
        raise AssertionError("Session artifact path was not stored relatively")
    redacted = redact_text(f"Authorization: Bearer {secret_key}", api_key=secret_key)
    if secret_key in redacted:
        raise AssertionError("Direct error redaction retained the API key")
    print("[PASS] Session manifest 可复现但不保存 API key、绝对私有路径或报告正文。")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="enzywizard-agent-runtime-") as root:
        workdir = Path(root)
        test_factory_contract(workdir / "factory")
        test_two_compatible_profiles(workdir / "compatibility")
        test_usage_and_budgets(workdir / "budgets")
        test_nullable_usage_metadata()
        test_structured_recovery_and_text(workdir / "model-output")
        test_runtime_mapping_rehydration(workdir / "rehydration")
        test_budget_analysis_exit(workdir / "analysis")
        test_manifest_and_redaction(workdir / "manifest")
    print("\n[ALL PASSED] Runtime, model compatibility, privacy, and manifest contracts are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
