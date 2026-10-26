from __future__ import annotations

from collections.abc import Callable, Mapping
from hashlib import sha256
import json
import time
from typing import Any, Literal, TypeVar, cast

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field

from ..conf import AgentConfig
from ..data_models.runtime import (
    LLMCallRecord,
    ModelCapabilityResult,
    ModelContractError,
)
from ..state import AgentState
from .runtime_service import (
    classify_exception,
    ensure_runtime_session,
    enforce_llm_budget,
    is_runtime_limit_error,
    record_runtime_error,
    remaining_session_seconds,
    utc_now,
)
from .knowledge_retrieval_service import enrich_messages_with_fair_terms
from ..utils.security_utils import redact_text


StructuredOutputT = TypeVar("StructuredOutputT", bound=BaseModel)
StructuredNormalizer = Callable[[Any], Any]
_CAPABILITY_CACHE: dict[str, ModelCapabilityResult] = {}


class _CapabilityAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["run_workflow"]
    target: Literal["capability probe"]


class _CapabilityRoutingProbe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    actions: list[_CapabilityAction] = Field(min_length=1, max_length=1)


class _CapabilityHistoryProbe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_id: Literal[0]
    turn_scope: Literal["specific"]
    turn_ids: list[Literal[1]] = Field(min_length=1, max_length=1)


class _CapabilityParameterAdjustment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_index: Literal[1]
    parameter_name: Literal["probe_parameter"]
    parameter_value: str | int | float | bool
    adjustment_reason: Literal["capability probe"]


class _CapabilityPlanningProbe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ready"]
    parameter_adjustments: list[_CapabilityParameterAdjustment] = Field(
        min_length=1,
        max_length=1,
    )
    optional_note: str | None
    enabled: bool


class _StructuredContractFailure(ModelContractError):
    """Keep a malformed structured candidate available for one repair request."""

    def __init__(self, message: str, *, candidate: Any = None) -> None:
        super().__init__(message)
        self.candidate = candidate


def create_chat_model(
    config: AgentConfig,
    *,
    timeout_seconds: float | None = None,
) -> ChatOpenAI:
    """从统一配置创建 OpenAI-compatible Chat Completions 客户端。"""

    kwargs: dict[str, Any] = {
        "api_key": config.api_key,
        "base_url": config.base_url,
        "model": config.model,
        "timeout": timeout_seconds or config.llm_timeout_seconds,
        "max_retries": config.llm_max_retries,
        # 明确使用 Chat Completions，避免部分 OpenAI SDK 版本按模型切换 API。
        "use_responses_api": False,
    }
    optional_values = {
        "temperature": config.temperature,
        "max_completion_tokens": config.max_completion_tokens,
        "reasoning_effort": config.reasoning_effort,
        "extra_body": config.llm_extra_body,
    }
    kwargs.update({key: value for key, value in optional_values.items() if value is not None})
    return ChatOpenAI(**kwargs)


def create_chat_model_for_state(state: AgentState) -> ChatOpenAI:
    """创建受当前会话剩余时间约束的统一模型客户端。"""

    config = state["config"]
    remaining = remaining_session_seconds(state)
    timeout = config.llm_timeout_seconds
    if remaining is not None:
        timeout = max(0.001, min(timeout, remaining))
    return create_chat_model(config, timeout_seconds=timeout)


def _estimate_message_tokens(messages: list[BaseMessage]) -> int:
    # 估算只用于请求前预算保护；最终台账优先采用服务端 usage。
    characters = sum(len(str(message.content)) for message in messages)
    return max(1, (characters + 3) // 4)


def _extract_usage(raw: Any) -> tuple[int | None, int | None, int | None, int | None]:
    """读取不同 OpenAI-compatible 服务可能缺失或为 null 的用量字段。"""

    usage_value = getattr(raw, "usage_metadata", None)
    usage = usage_value if isinstance(usage_value, Mapping) else {}
    response_value = getattr(raw, "response_metadata", None)
    response_metadata = response_value if isinstance(response_value, Mapping) else {}
    token_usage_value = response_metadata.get("token_usage")
    token_usage = token_usage_value if isinstance(token_usage_value, Mapping) else {}

    input_tokens = usage.get("input_tokens")
    if input_tokens is None:
        input_tokens = token_usage.get("prompt_tokens")
    output_tokens = usage.get("output_tokens")
    if output_tokens is None:
        output_tokens = token_usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    if total_tokens is None:
        total_tokens = token_usage.get("total_tokens")

    usage_details_value = usage.get("output_token_details")
    usage_details = (
        usage_details_value if isinstance(usage_details_value, Mapping) else {}
    )
    token_details_value = token_usage.get("completion_tokens_details")
    token_details = (
        token_details_value if isinstance(token_details_value, Mapping) else {}
    )
    reasoning_tokens = usage_details.get("reasoning")
    if reasoning_tokens is None:
        reasoning_tokens = usage_details.get("reasoning_tokens")
    if reasoning_tokens is None:
        reasoning_tokens = token_details.get("reasoning_tokens")
    if reasoning_tokens is None:
        reasoning_tokens = token_details.get("reasoning")
    return input_tokens, output_tokens, total_tokens, reasoning_tokens


def _extract_text_content(raw: Any) -> str:
    """Normalize OpenAI-compatible string and content-block responses to text."""

    content = getattr(raw, "content", raw)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, Mapping):
        for key in ("text", "content", "output_text"):
            value = content.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, Mapping):
                for key in ("text", "content", "output_text"):
                    value = item.get(key)
                    if isinstance(value, str) and value:
                        chunks.append(value)
                        break
        return "\n".join(chunks).strip()
    return str(content).strip() if content is not None else ""


def _decode_json_candidate(value: Any) -> Any:
    """Decode a JSON candidate without interpreting domain semantics."""

    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, (Mapping, list)):
        return value
    text = _extract_text_content(value)
    if not text:
        raise ValueError("structured response did not contain any content")
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            stripped = "\n".join(lines[1:-1]).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as direct_error:
        decoder = json.JSONDecoder()
        for index, character in enumerate(stripped):
            if character not in "[{":
                continue
            try:
                candidate, _ = decoder.raw_decode(stripped[index:])
                return candidate
            except json.JSONDecodeError:
                continue
        raise direct_error


def _with_json_schema_instruction(
    messages: list[BaseMessage],
) -> list[BaseMessage]:
    """Ensure compatible gateways recognize a strict JSON response request."""

    if any("json" in str(message.content).casefold() for message in messages):
        return messages
    instruction = (
        "Return valid JSON that exactly matches the supplied JSON Schema. "
        "Do not add prose or Markdown outside the JSON value."
    )
    updated = list(messages)
    for index, message in enumerate(updated):
        if isinstance(message, SystemMessage) and isinstance(message.content, str):
            updated[index] = message.model_copy(
                update={"content": f"{message.content.rstrip()}\n\n{instruction}"}
            )
            return updated
    return [SystemMessage(content=instruction), *updated]


def _record_llm_call(
    state: AgentState,
    *,
    stage: str,
    started_at: str,
    started_monotonic: float,
    estimated_input: int,
    raw: Any,
    output_text: str,
    error: BaseException | None = None,
    recoverable: bool = False,
) -> None:
    """Record one remote call and enforce cumulative post-call token budgets."""

    runtime = ensure_runtime_session(state)
    input_tokens, output_tokens, total_tokens, reasoning_tokens = _extract_usage(raw)
    usage_quality = "exact" if input_tokens is not None else "estimated"
    if input_tokens is None:
        input_tokens = estimated_input
    if output_tokens is None and output_text:
        output_tokens = max(1, (len(output_text) + 3) // 4)
        usage_quality = "estimated"
    if total_tokens is None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    error_category = None
    if error is not None:
        error_category = "timeout" if classify_exception(error) == "timeout" else "model"
    runtime.llm_calls.append(
        LLMCallRecord(
            stage=stage,
            started_at=started_at,
            duration_seconds=time.monotonic() - started_monotonic,
            status="failed" if error is not None else "succeeded",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            reasoning_tokens=reasoning_tokens,
            usage_quality=usage_quality,
            error_category=error_category,
        )
    )
    if error is not None:
        record_runtime_error(
            state,
            stage=stage,
            error=error,
            category=cast(Any, error_category),
            recoverable=recoverable,
        )

    config = state.get("config")
    exceeded_reason: str | None = None
    if (
        config is not None
        and config.max_input_tokens is not None
        and runtime.input_tokens > config.max_input_tokens
    ):
        exceeded_reason = "max_input_tokens"
    elif (
        config is not None
        and config.max_output_tokens is not None
        and runtime.output_tokens > config.max_output_tokens
    ):
        exceeded_reason = "max_output_tokens"
    if exceeded_reason is not None and runtime.status != "budget_exceeded":
        runtime.status = "budget_exceeded"
        runtime.stop_reason = exceeded_reason
        record_runtime_error(
            state,
            stage=stage,
            error=f"Recorded token usage exceeded {exceeded_reason}.",
            category="budget",
        )


def _invoke_structured_once(
    state: AgentState,
    *,
    stage: str,
    schema: type[StructuredOutputT],
    messages: list[BaseMessage],
    llm: Any,
    normalizer: StructuredNormalizer | None,
    recoverable_failure: bool,
) -> StructuredOutputT:
    messages = _with_json_schema_instruction(messages)
    estimated_input = _estimate_message_tokens(messages)
    enforce_llm_budget(state, stage, estimated_input)
    started_at = utc_now()
    started_monotonic = time.monotonic()
    raw: Any = None
    output_text = ""
    try:
        structured = llm.with_structured_output(
            schema,
            method="json_schema",
            strict=True,
            include_raw=True,
        )
        result = structured.invoke(messages)
        is_envelope = isinstance(result, dict) and bool(
            {"raw", "parsed", "parsing_error"}.intersection(result)
        )
        raw = result.get("raw") if is_envelope else None
        parsed = result.get("parsed") if is_envelope else result
        parsing_error = result.get("parsing_error") if is_envelope else None
        output_text = _extract_text_content(raw)
        if isinstance(parsed, schema):
            validated = parsed
        else:
            source = parsed if parsed is not None else raw
            try:
                candidate = _decode_json_candidate(source)
                if not output_text:
                    output_text = json.dumps(candidate, ensure_ascii=False, default=str)
                if normalizer is not None:
                    candidate = normalizer(candidate)
                validated = schema.model_validate(candidate)
            except Exception as validation_error:
                detail = parsing_error or validation_error
                raise _StructuredContractFailure(
                    f"Structured output validation failed for {schema.__name__}: {detail}",
                    candidate=(
                        candidate if "candidate" in locals() else output_text or None
                    ),
                ) from validation_error
        _record_llm_call(
            state,
            stage=stage,
            started_at=started_at,
            started_monotonic=started_monotonic,
            estimated_input=estimated_input,
            raw=raw,
            output_text=validated.model_dump_json(),
        )
        return cast(StructuredOutputT, validated)
    except Exception as error:
        _record_llm_call(
            state,
            stage=stage,
            started_at=started_at,
            started_monotonic=started_monotonic,
            estimated_input=estimated_input,
            raw=raw,
            output_text=output_text,
            error=error,
            recoverable=(
                recoverable_failure
                and isinstance(error, _StructuredContractFailure)
            ),
        )
        raise


def invoke_structured(
    state: AgentState,
    *,
    stage: str,
    schema: type[StructuredOutputT],
    messages: list[BaseMessage],
    llm: Any | None = None,
    normalizer: StructuredNormalizer | None = None,
    allow_repair: bool = True,
    retrieval_context: str | None = None,
) -> StructuredOutputT:
    """Invoke native strict JSON Schema, normalize, and repair at most once."""

    messages = enrich_messages_with_fair_terms(
        state,
        stage=stage,
        messages=messages,
        retrieval_context=retrieval_context,
    )
    model = llm or create_chat_model_for_state(state)
    contract_failure: _StructuredContractFailure | None = None
    try:
        return _invoke_structured_once(
            state,
            stage=stage,
            schema=schema,
            messages=messages,
            llm=model,
            normalizer=normalizer,
            recoverable_failure=allow_repair,
        )
    except _StructuredContractFailure as first_error:
        if not allow_repair or first_error.candidate is None:
            raise
        contract_failure = first_error

    assert contract_failure is not None
    repair_payload = {
        "invalid_candidate": contract_failure.candidate,
        "validation_error": str(contract_failure),
        "required_json_schema": schema.model_json_schema(),
    }
    repair_messages: list[BaseMessage] = [
        SystemMessage(
            content=(
                "Repair the supplied candidate so it matches the supplied JSON Schema. "
                "Preserve its meaning. Only correct structure, field names, enum values, "
                "missing required defaults, and wrappers. Do not invent turns, files, "
                "actions, scientific facts, or user intent. Return only the schema."
            )
        ),
        HumanMessage(content=json.dumps(repair_payload, ensure_ascii=False, indent=2)),
    ]
    try:
        return _invoke_structured_once(
            state,
            stage=f"{stage}.repair",
            schema=schema,
            messages=repair_messages,
            llm=model,
            normalizer=normalizer,
            recoverable_failure=False,
        )
    except Exception as repair_error:
        raise ModelContractError(
            f"Structured output repair failed for {schema.__name__}: {repair_error}"
        ) from repair_error


def invoke_text(
    state: AgentState,
    *,
    stage: str,
    messages: list[BaseMessage],
    llm: Any | None = None,
    retrieval_context: str | None = None,
) -> str:
    """Invoke a normal text response while preserving budgets and usage records."""

    messages = enrich_messages_with_fair_terms(
        state,
        stage=stage,
        messages=messages,
        retrieval_context=retrieval_context,
    )
    estimated_input = _estimate_message_tokens(messages)
    enforce_llm_budget(state, stage, estimated_input)
    model = llm or create_chat_model_for_state(state)
    started_at = utc_now()
    started_monotonic = time.monotonic()
    raw: Any = None
    try:
        raw = model.invoke(messages)
        output = _extract_text_content(raw)
        if not output:
            raise ModelContractError("Text response was empty.")
        _record_llm_call(
            state,
            stage=stage,
            started_at=started_at,
            started_monotonic=started_monotonic,
            estimated_input=estimated_input,
            raw=raw,
            output_text=output,
        )
        return output
    except Exception as error:
        _record_llm_call(
            state,
            stage=stage,
            started_at=started_at,
            started_monotonic=started_monotonic,
            estimated_input=estimated_input,
            raw=raw,
            output_text="",
            error=error,
        )
        raise


def check_model_capabilities(state: AgentState) -> ModelCapabilityResult:
    """用最小请求检查当前端点是否满足 Agent 的结构化模型契约。"""

    config = state["config"]
    runtime = ensure_runtime_session(state)
    if not config.check_model_capabilities:
        result = ModelCapabilityResult(
            status="skipped",
            chat_completions=False,
            system_messages=False,
            strict_structured_output=False,
            token_usage_available=False,
            checked_at=utc_now(),
            detail="Model capability check was disabled by the user.",
        )
        runtime.model_capability = result
        return result

    cache_key = json.dumps(
        {
            "base_url": config.base_url.rstrip("/"),
            "model": config.model,
            # 缓存只保存哈希，防止不同凭据共享一次通过结果。
            "credential_hash": sha256(config.api_key.encode("utf-8")).hexdigest(),
            "temperature": config.temperature,
            "max_completion_tokens": config.max_completion_tokens,
            "reasoning_effort": config.reasoning_effort,
            "extra_body": config.llm_extra_body,
        },
        sort_keys=True,
        default=str,
    )
    cached = _CAPABILITY_CACHE.get(cache_key)
    if cached is not None:
        result = cached.model_copy(update={"cached": True, "checked_at": utc_now()})
        runtime.model_capability = result
        return result

    completed_probes: list[str] = []
    successful_probe_calls: list[LLMCallRecord] = []
    current_probe = "routing"
    try:
        invoke_structured(
            state,
            stage="model_capability_check.routing",
            schema=_CapabilityRoutingProbe,
            messages=[
                SystemMessage(
                    content=(
                        "Return exactly one routing action. Its kind must be "
                        "run_workflow and its target must be capability probe. Return "
                        "only the supplied schema without any additional fields."
                    )
                ),
                HumanMessage(content="Run the model contract capability probe."),
            ],
            allow_repair=False,
        )
        completed_probes.append(current_probe)
        successful_probe_calls.append(runtime.llm_calls[-1])

        current_probe = "history_selection"
        invoke_structured(
            state,
            stage="model_capability_check.history_selection",
            schema=_CapabilityHistoryProbe,
            messages=[
                SystemMessage(
                    content=(
                        "Return the supplied schema with action_id 0, turn_scope "
                        "specific, and turn_ids containing only integer 1."
                    )
                ),
                HumanMessage(content="Select turn 1 for action 0."),
            ],
            allow_repair=False,
        )
        completed_probes.append(current_probe)
        successful_probe_calls.append(runtime.llm_calls[-1])

        current_probe = "planning_structure"
        invoke_structured(
            state,
            stage="model_capability_check.planning_structure",
            schema=_CapabilityPlanningProbe,
            messages=[
                SystemMessage(
                    content=(
                        "Return the supplied planning capability JSON schema with status "
                        "ready, one parameter adjustment for step 1 named "
                        "probe_parameter with integer value 7 and reason capability "
                        "probe, optional_note null, and enabled true."
                    )
                ),
                HumanMessage(content="Return the planning structure capability probe."),
            ],
            allow_repair=False,
        )
        completed_probes.append(current_probe)
        successful_probe_calls.append(runtime.llm_calls[-1])

        result = ModelCapabilityResult(
            status="passed",
            chat_completions=True,
            system_messages=True,
            strict_structured_output=True,
            token_usage_available=all(
                call.usage_quality == "exact" for call in successful_probe_calls
            ),
            checked_at=utc_now(),
            detail=(
                "OpenAI-compatible Chat Completions and routing, history-selection, "
                "and planning-structure strict JSON Schema probes passed."
            ),
        )
        _CAPABILITY_CACHE[cache_key] = result
        runtime.model_capability = result
        return result
    except Exception as error:
        if is_runtime_limit_error(error):
            raise
        basic_contract_passed = "routing" in completed_probes
        structured_response_received = isinstance(error, _StructuredContractFailure)
        failure_kind = (
            "strict_schema_validation"
            if structured_response_received
            else "request_or_transport"
        )
        result = ModelCapabilityResult(
            status="failed",
            chat_completions=basic_contract_passed or structured_response_received,
            system_messages=basic_contract_passed or structured_response_received,
            strict_structured_output=False,
            token_usage_available=(
                bool(successful_probe_calls)
                and all(
                    call.usage_quality == "exact" for call in successful_probe_calls
                )
            ),
            checked_at=utc_now(),
            detail=(
                f"Model contract check failed at {current_probe} "
                f"({failure_kind}): "
                + redact_text(error, api_key=config.api_key)
            ),
        )
        runtime.model_capability = result
        raise ModelContractError(result.detail) from error
