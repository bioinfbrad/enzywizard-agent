from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from ..builders.request_routing_context_builder import build_request_routing_context
from ..data_models.artifact import Artifact
from ..data_models.history_query import (
    HistoryArtifactSelectionDraft,
    HistoryQueryOutput,
    HistoryQueryResult,
    HistoryTurnSelectionDraft,
)
from ..data_models.request_routing import (
    RequestedAction,
    RequestRoutingDraft,
    RequestRoutingOutput,
    RequestRoutingResult,
)
from ..prompts.request_routing_prompt import (
    HISTORY_ARTIFACT_SELECTION_SYSTEM_PROMPT,
    HISTORY_TURN_SELECTION_SYSTEM_PROMPT,
    REQUEST_ROUTING_REVIEW_SYSTEM_PROMPT,
    REQUEST_ROUTING_SYSTEM_PROMPT,
)
from ..state import AgentState
from ..utils.planning_utils import load_enzywizard_tool_specs
from ..utils.tool_execution_utils import is_safe_historical_artifact
from .llm_service import create_chat_model_for_state, invoke_structured
from .runtime_service import is_runtime_limit_error


ARTIFACT_CATEGORY_TYPES: dict[str, set[str]] = {
    "clean": {
        "clean_report_json",
        "mut_clean_report_json",
        "cleaned_structure_cif",
        "cleaned_structure_pdb",
        "cleaned_sequence_fasta",
        "wt_cleaned_structure_cif",
        "wt_cleaned_structure_pdb",
        "wt_cleaned_sequence_fasta",
        "mut_cleaned_structure_cif",
        "mut_cleaned_structure_pdb",
        "mut_cleaned_sequence_fasta",
    },
    "aaprops": {"aaprops_report_json"},
    "hydrocluster": {"hydrocluster_report_json"},
    "energy": {"energy_report_json"},
    "flexibility": {"flexibility_report_json"},
    "disorder": {"disorder_report_json"},
    "conservation": {
        "conservation_report_json",
        "cleaned_msa_sto",
        "hmm_profile",
    },
    "embedding": {"embedding_report_json"},
    "pocket": {"pocket_report_json"},
    "substrate": {"substrate_report_json", "substrate_sdf"},
    "dock": {
        "dock_report_json",
        "docked_substrate_sdf",
        "docked_complex_structure",
    },
    "interaction": {"interaction_report_json"},
    "integrate": {
        "integrate_report_json",
        "mut_integrate_report_json",
        "integrate_nodes_json",
        "integrate_edges_json",
        "wt_integrate_nodes_json",
        "wt_integrate_edges_json",
        "mut_integrate_nodes_json",
        "mut_integrate_edges_json",
    },
    "structure": {
        "cleaned_structure_cif",
        "cleaned_structure_pdb",
        "wt_cleaned_structure_cif",
        "wt_cleaned_structure_pdb",
        "mut_cleaned_structure_cif",
        "mut_cleaned_structure_pdb",
        "docked_complex_structure",
    },
    "sequence": {
        "cleaned_sequence_fasta",
        "wt_cleaned_sequence_fasta",
        "mut_cleaned_sequence_fasta",
        "cleaned_msa_sto",
        "hmm_profile",
    },
    "graph": {
        "integrate_nodes_json",
        "integrate_edges_json",
        "wt_integrate_nodes_json",
        "wt_integrate_edges_json",
        "mut_integrate_nodes_json",
        "mut_integrate_edges_json",
    },
    "log": {"log_file", "agent_stdout_log", "agent_stderr_log"},
}

REPORT_TYPE_TO_ARTIFACT_TYPE = {
    "enzywizard_clean": "clean_report_json",
    "enzywizard_mut_clean": "mut_clean_report_json",
    "enzywizard_aaprops": "aaprops_report_json",
    "enzywizard_hydrocluster": "hydrocluster_report_json",
    "enzywizard_energy": "energy_report_json",
    "enzywizard_flexibility": "flexibility_report_json",
    "enzywizard_disorder": "disorder_report_json",
    "enzywizard_conservation": "conservation_report_json",
    "enzywizard_embedding": "embedding_report_json",
    "enzywizard_pocket": "pocket_report_json",
    "enzywizard_substrate": "substrate_report_json",
    "enzywizard_dock": "dock_report_json",
    "enzywizard_interaction": "interaction_report_json",
    "enzywizard_integrate": "integrate_report_json",
    "enzywizard_mut_integrate": "mut_integrate_report_json",
}

_HISTORY_ANALYSIS_ACTION_TYPES = {"analyze_history", "list_artifacts"}
_HISTORY_DRAFT_KINDS = {
    "summarize_history",
    "explain_history",
    "compare_history",
    "list_history_files",
}
_WORKFLOW_DRAFT_KINDS = {
    "run_workflow",
    "rerun_previous_workflow",
    "recompute_workflow",
}
_EXPLICIT_HISTORY_REFERENCE = re.compile(
    r"\b(?:previous|earlier|prior|historical|history|already|existing|reuse|"
    r"re-use|turn\s*#?\s*\d+|last\s+(?:turn|run|workflow|request))\b",
    re.IGNORECASE,
)
_EXPLICIT_ARTIFACT_REQUEST = re.compile(
    r"\b(?:list|show|find|open|download|inspect|analy[sz]e|compare|summari[sz]e|"
    r"explain)\b.{0,80}\b(?:report|result|output|artifact|file|log)s?\b",
    re.IGNORECASE | re.DOTALL,
)


def _create_llm(state: AgentState):
    return create_chat_model_for_state(state)


def _has_explicit_history_request(state: AgentState) -> bool:
    message = str(state.get("current_user_message", ""))
    return bool(
        _EXPLICIT_HISTORY_REFERENCE.search(message)
        or _EXPLICIT_ARTIFACT_REQUEST.search(message)
    )


def _prune_unsupported_mixed_history_actions(
    state: AgentState,
    draft: RequestRoutingDraft | RequestRoutingOutput,
) -> RequestRoutingDraft | RequestRoutingOutput:
    """Drop unsupported history analysis added beside a current workflow."""

    if not isinstance(draft, RequestRoutingDraft):
        return draft
    kinds = {action.kind for action in draft.actions}
    if (
        not kinds.intersection(_WORKFLOW_DRAFT_KINDS)
        or not kinds.intersection(_HISTORY_DRAFT_KINDS)
        or _has_explicit_history_request(state)
    ):
        return draft
    actions = [
        action for action in draft.actions if action.kind not in _HISTORY_DRAFT_KINDS
    ]
    return draft.model_copy(update={"actions": actions}) if actions else draft


def _route_for_output(output: RequestRoutingOutput) -> str:
    if output.requires_history_artifacts:
        return "request_to_history_query"
    if output.requires_tool_execution:
        return "request_to_planning"
    return "request_to_analysis"


def _normalize_routing_candidate(candidate: Any) -> Any:
    """Canonicalize structural aliases without reinterpreting request semantics."""

    if isinstance(candidate, list):
        candidate = {"actions": candidate}
    if not isinstance(candidate, Mapping):
        return candidate
    normalized = dict(candidate)
    actions = normalized.get("actions")
    if isinstance(actions, Mapping):
        actions = [actions]
    if not isinstance(actions, list):
        return normalized
    raw_kinds = [
        item.get("kind") or item.get("action") or item.get("action_type")
        for item in actions
        if isinstance(item, Mapping)
    ]
    explicit_recomputation = normalized.get("requests_recomputation") is True
    workflow_count = sum(
        kind in {"run_workflow", "run_tool_workflow"} for kind in raw_kinds
    )
    if explicit_recomputation and workflow_count != 1:
        # A global legacy flag cannot be assigned safely to zero or multiple
        # workflow actions. Preserve the full invalid candidate for strict repair.
        return normalized
    references_previous_plan = any(
        kind in {"reference_previous_plan", "reference_historical_context"}
        for kind in raw_kinds
    )
    kind_aliases = {
        "run_tool_workflow": "run_workflow",
        "rerun_workflow": "rerun_previous_workflow",
        "list_artifacts": "list_history_files",
        "reference_historical_context": "reference_previous_plan",
        "answer_software_help": "software_help",
        "answer_general_question": "science_question",
        "continue_conversation": "conversation",
        "decline_out_of_scope": "out_of_scope",
        "clarify_request": "clarify",
        "exit_session": "exit",
    }
    normalized_actions: list[Any] = []
    for item in actions:
        if not isinstance(item, Mapping):
            normalized_actions.append(item)
            continue
        action = dict(item)
        kind = action.get("kind")
        if kind is None:
            kind = action.get("action") or action.get("action_type")
        if kind == "analyze_history":
            operation = action.get("history_operation")
            kind = (
                f"{operation}_history"
                if operation in {"summarize", "explain", "compare"}
                else kind
            )
        if (
            explicit_recomputation
            and kind in {"run_workflow", "run_tool_workflow"}
        ):
            kind = (
                "rerun_previous_workflow"
                if references_previous_plan
                else "recompute_workflow"
            )
        if kind == "reuse_historical_output":
            explicit_policy = action.get("reuse_policy", normalized.get("reuse_policy"))
            if explicit_policy == "required":
                kind = "reuse_history_output_required"
            elif explicit_policy == "allowed":
                kind = "reuse_history_output_optional"
            # Missing or unknown policy is semantically ambiguous. Keep the legacy
            # value invalid so the strict repair call must resolve or reject it.
        target = action.get("target")
        if target is None:
            target = action.get("description") or action.get("request")
        normalized_actions.append(
            {
                "kind": kind_aliases.get(kind, kind),
                "target": target,
            }
        )
    return {"actions": normalized_actions}


def _normalize_request_routing_draft(
    draft: RequestRoutingDraft | RequestRoutingOutput,
) -> RequestRoutingOutput:
    """把最小语义动作编译成完整、严格且内部一致的路由结果。"""

    if isinstance(draft, RequestRoutingOutput):
        return draft

    # 每个 kind 唯一决定正式动作、历史操作、复用策略和重算语义。LLM 不再
    # 重复填写这些相关字段，从源头避免多个枚举相互矛盾。
    compiled_actions: list[RequestedAction] = []
    reuse_policy = "none"
    requests_recomputation = False
    for action in draft.actions:
        if action.kind == "run_workflow":
            compiled_actions.append(
                RequestedAction(action_type="run_tool_workflow", target=action.target)
            )
        elif action.kind == "rerun_previous_workflow":
            requests_recomputation = True
            compiled_actions.extend(
                [
                    RequestedAction(
                        action_type="reference_historical_context",
                        target=action.target,
                    ),
                    RequestedAction(
                        action_type="run_tool_workflow",
                        target=action.target,
                    ),
                ]
            )
        elif action.kind == "recompute_workflow":
            requests_recomputation = True
            compiled_actions.append(
                RequestedAction(action_type="run_tool_workflow", target=action.target)
            )
        elif action.kind in {
            "summarize_history",
            "explain_history",
            "compare_history",
        }:
            operation = action.kind.removesuffix("_history")
            compiled_actions.append(
                RequestedAction(
                    action_type="analyze_history",
                    target=action.target,
                    history_operation=operation,
                )
            )
        elif action.kind == "list_history_files":
            compiled_actions.append(
                RequestedAction(
                    action_type="list_artifacts",
                    target=action.target,
                    history_operation="list_artifacts",
                )
            )
        elif action.kind in {
            "reuse_history_output_required",
            "reuse_history_output_optional",
        }:
            if action.kind == "reuse_history_output_required":
                reuse_policy = "required"
            elif reuse_policy != "required":
                reuse_policy = "allowed"
            compiled_actions.extend(
                [
                    RequestedAction(
                        action_type="reuse_historical_output",
                        target=action.target,
                        history_operation="reuse_as_input",
                    ),
                    RequestedAction(
                        action_type="run_tool_workflow",
                        target=action.target,
                    ),
                ]
            )
        elif action.kind == "reference_previous_plan":
            compiled_actions.append(
                RequestedAction(
                    action_type="reference_historical_context",
                    target=action.target,
                )
            )
        else:
            direct_action_types = {
                "software_help": "answer_software_help",
                "science_question": "answer_general_question",
                "conversation": "continue_conversation",
                "out_of_scope": "decline_out_of_scope",
                "clarify": "clarify_request",
                "exit": "exit_session",
            }
            compiled_actions.append(
                RequestedAction(
                    action_type=direct_action_types[action.kind],
                    target=action.target,
                )
            )

    # 只有历史上下文而没有执行或回答动作时，补充会话回答动作。该规则只读取
    # 结构化 action，不解析用户原文。
    compiled_types = {item.action_type for item in compiled_actions}
    if (
        "reference_historical_context" in compiled_types
        and not compiled_types.intersection(
            {"run_tool_workflow", "continue_conversation"}
        )
    ):
        compiled_actions.append(
            RequestedAction(
                action_type="continue_conversation",
                target="answer the request using bounded historical context",
            )
        )

    # 同类动作由后续阶段作为一个用户目标处理，避免模型把单条工作流的内部
    # 前置步骤展开成多个重复动作。不同目标保留在合并后的 target 中。
    grouped_targets: dict[tuple[str, str | None], list[str]] = {}
    grouped_actions: dict[tuple[str, str | None], RequestedAction] = {}
    for action in compiled_actions:
        key = (action.action_type, action.history_operation)
        grouped_actions.setdefault(key, action)
        targets = grouped_targets.setdefault(key, [])
        if action.target not in targets:
            targets.append(action.target)
    actions = [
        grouped_actions[key].model_copy(update={"target": "; ".join(targets)})
        for key, targets in grouped_targets.items()
    ]
    action_types = {action.action_type for action in actions}
    if "clarify_request" in action_types or (
        "exit_session" in action_types and len(action_types) > 1
    ):
        # 歧义动作以及“退出并执行其他工作”都不能静默选择其中一侧。
        actions = [
            RequestedAction(
                action_type="clarify_request",
                target="resolve the ambiguous or conflicting requested actions",
            )
        ]
        action_types = {"clarify_request"}
    action_to_request_type = {
        "run_tool_workflow": "tool_workflow",
        "analyze_history": "history_analysis",
        "list_artifacts": "artifact_query",
        "answer_software_help": "software_help",
        "answer_general_question": "general_scientific_question",
        "continue_conversation": "conversation",
        "decline_out_of_scope": "out_of_scope",
        "clarify_request": "clarification",
        "exit_session": "exit",
    }
    primary_types = {
        request_type
        for action_type, request_type in action_to_request_type.items()
        if action_type in action_types
    }
    # 文件复用既读取历史产物又触发新工具执行，因此规范为 mixed；仅把
    # 旧计划作为上下文不构成独立历史产物动作。
    if "reuse_historical_output" in action_types:
        primary_types.add("artifact_query")
    request_type = (
        next(iter(primary_types))
        if len(primary_types) == 1
        else "mixed"
        if len(primary_types) > 1
        else "clarification"
    )
    history_artifact_actions = {
        "analyze_history",
        "list_artifacts",
        "reuse_historical_output",
    }
    history_actions = history_artifact_actions | {"reference_historical_context"}
    requires_tool_execution = "run_tool_workflow" in action_types
    references_history = bool(action_types & history_actions)
    requires_history_artifacts = bool(action_types & history_artifact_actions)

    if "reuse_historical_output" not in action_types:
        reuse_policy = "none"

    routing_reason = "Compiled requested action kinds: " + ", ".join(
        action.kind for action in draft.actions
    ) + "."

    return RequestRoutingOutput(
        request_type=request_type,
        actions=actions,
        requires_tool_execution=requires_tool_execution,
        references_history=references_history,
        requires_history_artifacts=requires_history_artifacts,
        reuse_policy=reuse_policy,
        requests_recomputation=(requests_recomputation and requires_tool_execution),
        clarification_question=(
            "Could you clarify the target and action you want me to perform?"
            if request_type == "clarification"
            else None
        ),
        routing_reason=routing_reason,
    )


def _review_request_routing_draft(
    state: AgentState,
    draft: RequestRoutingDraft | RequestRoutingOutput,
) -> RequestRoutingDraft | RequestRoutingOutput:
    """用独立结构化审核纠正首轮 Router 的语义偏差；失败时保留有效草稿。"""

    review_context = {
        "original_routing_context": json.loads(
            build_request_routing_context(state)
        ),
        "candidate_routing_draft": draft.model_dump(mode="json"),
    }
    try:
        reviewed = invoke_structured(
            state,
            stage="routing.request_review",
            schema=RequestRoutingDraft,
            llm=_create_llm(state),
            messages=[
                SystemMessage(content=REQUEST_ROUTING_REVIEW_SYSTEM_PROMPT),
                HumanMessage(content=json.dumps(review_context, indent=2)),
            ],
            normalizer=_normalize_routing_candidate,
        )
        candidate_output = _normalize_request_routing_draft(draft)
        reviewed_output = _normalize_request_routing_draft(reviewed)
        candidate_signature = _routing_semantic_signature(candidate_output)
        reviewed_signature = _routing_semantic_signature(reviewed_output)
        final_draft = reviewed
        final_output = reviewed_output
        if candidate_signature != reviewed_signature:
            adjudication_context = {
                "original_routing_context": review_context["original_routing_context"],
                "first_candidate": draft.model_dump(mode="json"),
                "review_candidate": reviewed.model_dump(mode="json"),
                "instruction": (
                    "Resolve the semantic disagreement. Return the complete minimal "
                    "routing draft that best matches the original user request."
                ),
            }
            try:
                final_draft = invoke_structured(
                    state,
                    stage="routing.request_adjudication",
                    schema=RequestRoutingDraft,
                    llm=_create_llm(state),
                    messages=[
                        SystemMessage(content=REQUEST_ROUTING_REVIEW_SYSTEM_PROMPT),
                        HumanMessage(
                            content=json.dumps(adjudication_context, indent=2)
                        ),
                    ],
                    normalizer=_normalize_routing_candidate,
                )
                final_output = _normalize_request_routing_draft(final_draft)
            except Exception as error:
                if is_runtime_limit_error(error):
                    raise
                return draft
        if (
            candidate_output.request_type == "mixed"
            and final_output.request_type != "clarification"
            and len(final_output.actions) < len(candidate_output.actions)
        ):
            candidate_types = {
                action.action_type for action in candidate_output.actions
            }
            final_types = {action.action_type for action in final_output.actions}
            removed_types = candidate_types - final_types
            if (
                not removed_types.issubset(_HISTORY_ANALYSIS_ACTION_TYPES)
                or _has_explicit_history_request(state)
            ):
                # Preserve deleted independent tasks and explicit history requests.
                return draft
        return final_draft
    except Exception as error:
        if is_runtime_limit_error(error):
            raise
        return draft


def _apply_required_input_preflight(
    state: AgentState,
    output: RequestRoutingOutput,
) -> RequestRoutingOutput:
    """Check only the universal structure input before scientific planning."""

    if output.request_type != "tool_workflow" or not output.requires_tool_execution:
        return output
    inputs = state.get("agent_inputs")
    if inputs is None:
        return output
    missing: list[str] = []
    if not inputs.structure_file and "reuse_historical_output" not in {
        item.action_type for item in output.actions
    }:
        missing.append("a protein structure")
    missing = list(dict.fromkeys(missing))
    if not missing:
        return output
    if len(missing) == 1:
        requirement = missing[0]
    else:
        requirement = ", ".join(missing[:-1]) + f", and {missing[-1]}"
    return RequestRoutingOutput(
        request_type="clarification",
        actions=[
            RequestedAction(
                action_type="clarify_request",
                target=f"provide {requirement} for the requested workflow",
            )
        ],
        requires_tool_execution=False,
        references_history=False,
        requires_history_artifacts=False,
        reuse_policy="none",
        requests_recomputation=False,
        clarification_question=(
            f"Please provide {requirement} before I build this workflow."
        ),
        routing_reason="Required workflow input is unavailable.",
    )


def _routing_semantic_signature(output: RequestRoutingOutput) -> tuple[Any, ...]:
    return (
        output.request_type,
        tuple(
            sorted(
                (action.action_type, action.history_operation or "")
                for action in output.actions
            )
        ),
        output.requires_tool_execution,
        output.references_history,
        output.requires_history_artifacts,
        output.reuse_policy,
        output.requests_recomputation,
    )


def route_user_request(state: AgentState) -> RequestRoutingResult:
    """调用专用 Router LLM，并校验混合动作是否有用户请求依据。"""

    turn_id = state.get("current_turn_id", 1)
    try:
        draft = invoke_structured(
            state,
            stage="routing.request",
            schema=RequestRoutingDraft,
            llm=_create_llm(state),
            messages=[
                SystemMessage(content=REQUEST_ROUTING_SYSTEM_PROMPT),
                HumanMessage(content=build_request_routing_context(state)),
            ],
            normalizer=_normalize_routing_candidate,
        )
        draft = _review_request_routing_draft(state, draft)
        draft = _prune_unsupported_mixed_history_actions(state, draft)
        output = _normalize_request_routing_draft(draft)
        output = _apply_required_input_preflight(state, output)
        return RequestRoutingResult(
            turn_id=turn_id,
            routing_status="succeeded",
            route=_route_for_output(output),
            output=output,
            summary=(
                f"Classified request as {output.request_type} with "
                f"{len(output.actions)} action(s)."
            ),
            error_message="",
        )
    except Exception as exc:
        return RequestRoutingResult(
            turn_id=turn_id,
            routing_status="failed",
            route="request_to_analysis",
            output=None,
            summary="Request routing failed before any tool execution.",
            error_message=f"Request routing failed: {exc}",
        )


@dataclass(frozen=True)
class _HistoryCatalogEntry:
    catalog_id: str
    turn_id: int
    source_tool: str
    side: str
    artifact_type: str
    reusable_as_workflow_input: bool
    artifact: Artifact

    def model_payload(self) -> dict[str, Any]:
        return {
            "id": self.catalog_id,
            "turn_id": self.turn_id,
            "source_tool": self.source_tool,
            "side": self.side,
            "artifact_type": self.artifact_type,
            "name": Path(self.artifact.path).name,
            "report_type": ARTIFACT_TYPE_TO_REPORT_TYPE.get(self.artifact_type),
            "reusable_as_workflow_input": self.reusable_as_workflow_input,
        }


ARTIFACT_TYPE_TO_REPORT_TYPE = {
    artifact_type: report_type
    for report_type, artifact_type in REPORT_TYPE_TO_ARTIFACT_TYPE.items()
}


def _reusable_artifact_types() -> set[str]:
    reusable: set[str] = set()
    for tool_spec in load_enzywizard_tool_specs().values():
        for output_spec in tool_spec.get("output_files", []):
            artifact_type = output_spec.get("artifact_type")
            if (
                isinstance(artifact_type, str)
                and output_spec.get("historical_workflow_input") is True
            ):
                reusable.add(artifact_type)
    return reusable


def _history_catalog(state: AgentState) -> list[_HistoryCatalogEntry]:
    current_turn_id = state.get("current_turn_id", 1)
    reusable_types = _reusable_artifact_types()
    turn_indexes: dict[int, int] = defaultdict(int)
    entries: list[_HistoryCatalogEntry] = []
    artifacts = sorted(
        (
            artifact
            for artifact in state.get("artifacts", [])
            if artifact.turn_id < current_turn_id
            and is_safe_historical_artifact(
                artifact,
                state.get("tool_executions", []),
            )
        ),
        key=lambda item: (
            item.turn_id,
            item.source_tool,
            item.side,
            item.artifact_type,
            Path(item.path).name,
            item.path,
        ),
    )
    for artifact in artifacts:
        turn_indexes[artifact.turn_id] += 1
        entries.append(
            _HistoryCatalogEntry(
                catalog_id=(
                    f"t{artifact.turn_id}:a{turn_indexes[artifact.turn_id]}"
                ),
                turn_id=artifact.turn_id,
                source_tool=artifact.source_tool,
                side=artifact.side,
                artifact_type=artifact.artifact_type,
                reusable_as_workflow_input=(
                    artifact.artifact_type in reusable_types
                ),
                artifact=artifact,
            )
        )
    return entries


def _history_actions(state: AgentState) -> list[RequestedAction]:
    routing = state.get("current_request_routing")
    actions = (
        routing.output.actions
        if routing is not None and routing.output is not None
        else []
    )
    return [
        action
        for action in actions
        if action.action_type
        in {"analyze_history", "list_artifacts", "reuse_historical_output"}
    ]


def _operation_for_action(action: RequestedAction) -> str:
    if action.history_operation is not None:
        return action.history_operation
    if action.action_type == "list_artifacts":
        return "list_artifacts"
    if action.action_type == "reuse_historical_output":
        return "reuse_as_input"
    # Backward-compatible checkpoints may predate explicit history_operation.
    return "summarize"


def _normalize_turn_candidate(candidate: Any, action_id: int) -> Any:
    if isinstance(candidate, list) and len(candidate) == 1:
        candidate = candidate[0]
    if not isinstance(candidate, Mapping):
        return candidate
    normalized = dict(candidate)
    nested_scope = normalized.get("turn_scope")
    if isinstance(nested_scope, Mapping):
        normalized["turn_scope"] = nested_scope.get("type") or nested_scope.get("scope")
        normalized.setdefault(
            "turn_ids",
            nested_scope.get("turn_ids") or nested_scope.get("referenced_turn_ids"),
        )
    normalized["action_id"] = action_id
    normalized["turn_ids"] = normalized.get(
        "turn_ids", normalized.get("referenced_turn_ids", [])
    )
    if normalized.get("turn_ids") is None:
        normalized["turn_ids"] = []
    return {
        "action_id": normalized.get("action_id"),
        "turn_scope": normalized.get("turn_scope"),
        "turn_ids": normalized.get("turn_ids"),
    }


_EXPLICIT_TURN_REFERENCE = re.compile(
    r"\bturns?\s+(#?\d+(?:\s*(?:,|and|&)\s*(?:turns?\s+)?#?\d+)*)",
    flags=re.IGNORECASE,
)


def _extract_explicit_turn_ids(text: str) -> list[int]:
    """Extract only explicit numeric turn references, preserving first-use order."""

    turn_ids: list[int] = []
    for match in _EXPLICIT_TURN_REFERENCE.finditer(text):
        turn_ids.extend(int(value) for value in re.findall(r"\d+", match.group(1)))
    return list(dict.fromkeys(turn_ids))


def _explicit_turn_ids_for_action(
    request: str,
    action: RequestedAction,
    *,
    history_action_count: int,
) -> list[int]:
    """Resolve unambiguous numeric references without asking the language model."""

    action_turn_ids = _extract_explicit_turn_ids(action.target)
    if action_turn_ids:
        return action_turn_ids
    request_turn_ids = _extract_explicit_turn_ids(request)
    if len(request_turn_ids) == 1 or history_action_count == 1:
        return request_turn_ids
    # Multiple actions and multiple global turn IDs require semantic assignment.
    return []


def _normalize_artifact_candidate(candidate: Any, action_id: int) -> Any:
    if isinstance(candidate, list):
        candidate = {"artifact_ids": candidate}
    if not isinstance(candidate, Mapping):
        return candidate
    normalized = dict(candidate)
    artifact_ids = normalized.get("artifact_ids")
    if artifact_ids is None:
        artifact_ids = normalized.get("selected_artifact_ids", normalized.get("ids", []))
    if isinstance(artifact_ids, str):
        artifact_ids = [artifact_ids]
    if isinstance(artifact_ids, list):
        artifact_ids = [
            item.get("id") or item.get("artifact_id")
            if isinstance(item, Mapping)
            else item
            for item in artifact_ids
        ]
    return {"action_id": action_id, "artifact_ids": artifact_ids or []}


def _resolve_turn_selection(
    selection: HistoryTurnSelectionDraft,
    available_turn_ids: list[int],
) -> tuple[list[int], str | None]:
    if not available_turn_ids:
        return [], "No completed artifact-bearing turn is available."
    if selection.turn_scope == "specific":
        invalid = sorted(set(selection.turn_ids) - set(available_turn_ids))
        if invalid:
            return [], f"Requested turn IDs are not available: {invalid}."
        return sorted(selection.turn_ids), None
    if selection.turn_scope == "previous":
        return [available_turn_ids[-1]], None
    return list(available_turn_ids), None


def _artifact_categories_for_types(artifact_types: set[str]) -> list[str]:
    return sorted(
        category
        for category, category_types in ARTIFACT_CATEGORY_TYPES.items()
        if artifact_types.intersection(category_types)
    )


def _compile_history_query(
    turn_selections: list[HistoryTurnSelectionDraft],
    resolved_turn_ids: list[int],
    selected: list[Artifact],
    operations: list[str],
    *,
    clarification_question: str | None = None,
) -> HistoryQueryOutput:
    scopes = {selection.turn_scope for selection in turn_selections}
    if len(turn_selections) == 1:
        turn_scope = turn_selections[0].turn_scope
        referenced_turn_ids = list(turn_selections[0].turn_ids)
    elif scopes == {"all_previous"}:
        turn_scope = "all_previous"
        referenced_turn_ids = []
    else:
        turn_scope = "specific"
        referenced_turn_ids = sorted(set(resolved_turn_ids))

    artifact_types = sorted({artifact.artifact_type for artifact in selected})
    report_types = sorted(
        {
            ARTIFACT_TYPE_TO_REPORT_TYPE[artifact_type]
            for artifact_type in artifact_types
            if artifact_type in ARTIFACT_TYPE_TO_REPORT_TYPE
        }
    )
    return HistoryQueryOutput.model_validate(
        {
            "turn_scope": turn_scope,
            "referenced_turn_ids": referenced_turn_ids,
            "artifact_categories": _artifact_categories_for_types(set(artifact_types)),
            "artifact_types": artifact_types,
            "report_types": report_types,
            "sides": sorted({artifact.side for artifact in selected}),
            "operations": list(dict.fromkeys(operations)),
            "query_reason": (
                "Compiled routed history actions from validated turn and artifact "
                "catalog selections."
            ),
            "clarification_question": clarification_question,
        }
    )


def plan_history_query(state: AgentState) -> HistoryQueryResult:
    """Resolve each history action through small schemas and exact catalog IDs."""

    current_turn_id = state.get("current_turn_id", 1)
    actions = _history_actions(state)
    catalog = _history_catalog(state)
    available_turn_ids = sorted({entry.turn_id for entry in catalog})
    if not actions:
        return HistoryQueryResult(
            turn_id=current_turn_id,
            query_status="failed",
            query=None,
            resolved_turn_ids=[],
            selected_artifacts=[],
            summary="History query planning failed.",
            error_message="No routed history action was available.",
        )

    turn_selections: list[HistoryTurnSelectionDraft] = []
    resolved_turn_ids: list[int] = []
    selected_artifacts: list[Artifact] = []
    selected_paths: set[str] = set()
    operations: list[str] = []
    comparison_artifacts: list[list[Artifact]] = []

    try:
        for action_id, action in enumerate(actions):
            operation = _operation_for_action(action)
            operations.append(operation)
            current_request = state.get("current_user_message", "")
            turn_context = {
                "current_turn_id": current_turn_id,
                "current_user_request": current_request,
                "routed_action": {
                    "action_id": action_id,
                    "operation": operation,
                    "target": action.target,
                },
                "available_turn_ids": available_turn_ids,
            }
            model = _create_llm(state)
            explicit_turn_ids = _explicit_turn_ids_for_action(
                current_request,
                action,
                history_action_count=len(actions),
            )
            if explicit_turn_ids:
                turn_selection = HistoryTurnSelectionDraft(
                    action_id=action_id,
                    turn_scope="specific",
                    turn_ids=explicit_turn_ids,
                )
            else:
                turn_selection = invoke_structured(
                    state,
                    stage=f"routing.history.turn_selection.{action_id}",
                    schema=HistoryTurnSelectionDraft,
                    llm=model,
                    messages=[
                        SystemMessage(content=HISTORY_TURN_SELECTION_SYSTEM_PROMPT),
                        HumanMessage(content=json.dumps(turn_context, indent=2)),
                    ],
                    normalizer=lambda candidate, expected=action_id: _normalize_turn_candidate(
                        candidate, expected
                    ),
                    retrieval_context=action.target,
                )
            turn_selections.append(turn_selection)
            candidate_turn_ids, turn_error = _resolve_turn_selection(
                turn_selection, available_turn_ids
            )
            if turn_error:
                query = _compile_history_query(
                    turn_selections,
                    resolved_turn_ids,
                    selected_artifacts,
                    operations,
                )
                return HistoryQueryResult(
                    turn_id=current_turn_id,
                    query_status="not_found",
                    query=query,
                    resolved_turn_ids=resolved_turn_ids,
                    selected_artifacts=[],
                    summary=turn_error,
                    error_message="",
                )

            candidate_entries = [
                entry for entry in catalog if entry.turn_id in candidate_turn_ids
            ]
            if operation == "reuse_as_input":
                candidate_entries = [
                    entry
                    for entry in candidate_entries
                    if entry.reusable_as_workflow_input
                ]
            artifact_context = {
                "current_user_request": state.get("current_user_message", ""),
                "routed_action": {
                    "action_id": action_id,
                    "operation": operation,
                    "target": action.target,
                    "turn_scope": turn_selection.turn_scope,
                },
                "candidate_artifacts": [
                    entry.model_payload() for entry in candidate_entries
                ],
            }
            artifact_selection = invoke_structured(
                state,
                stage=f"routing.history.artifact_selection.{action_id}",
                schema=HistoryArtifactSelectionDraft,
                llm=model,
                messages=[
                    SystemMessage(content=HISTORY_ARTIFACT_SELECTION_SYSTEM_PROMPT),
                    HumanMessage(content=json.dumps(artifact_context, indent=2)),
                ],
                normalizer=lambda candidate, expected=action_id: _normalize_artifact_candidate(
                    candidate, expected
                ),
                retrieval_context=action.target,
            )
            entries_by_id = {
                entry.catalog_id: entry for entry in candidate_entries
            }
            invalid_ids = sorted(
                set(artifact_selection.artifact_ids) - set(entries_by_id)
            )
            if invalid_ids:
                correction_context = {
                    **artifact_context,
                    "invalid_artifact_ids": invalid_ids,
                    "instruction": "Replace invalid IDs using only candidate_artifacts.",
                }
                artifact_selection = invoke_structured(
                    state,
                    stage=f"routing.history.artifact_selection.{action_id}.retry",
                    schema=HistoryArtifactSelectionDraft,
                    llm=model,
                    messages=[
                        SystemMessage(content=HISTORY_ARTIFACT_SELECTION_SYSTEM_PROMPT),
                        HumanMessage(content=json.dumps(correction_context, indent=2)),
                    ],
                    normalizer=lambda candidate, expected=action_id: _normalize_artifact_candidate(
                        candidate, expected
                    ),
                    retrieval_context=action.target,
                )
                invalid_ids = sorted(
                    set(artifact_selection.artifact_ids) - set(entries_by_id)
                )
                if invalid_ids:
                    raise ValueError(
                        f"Artifact selection returned unknown catalog IDs: {invalid_ids}"
                    )

            selected_entries = [
                entries_by_id[artifact_id]
                for artifact_id in artifact_selection.artifact_ids
            ]
            action_resolved_turn_ids = list(candidate_turn_ids)
            if turn_selection.turn_scope == "latest_matching" and selected_entries:
                latest_turn = max(entry.turn_id for entry in selected_entries)
                selected_entries = [
                    entry for entry in selected_entries if entry.turn_id == latest_turn
                ]
                action_resolved_turn_ids = [latest_turn]
            action_artifacts = [
                entry.artifact for entry in selected_entries
            ]
            if operation == "compare":
                comparison_artifacts.append(action_artifacts)
            for artifact in action_artifacts:
                if artifact.path in selected_paths:
                    continue
                selected_paths.add(artifact.path)
                selected_artifacts.append(artifact)
            # Keep the searched turns even when no artifact matches. This lets the
            # deterministic compiler produce a valid, auditable not-found query.
            resolved_turn_ids.extend(action_resolved_turn_ids)
    except Exception as exc:
        return HistoryQueryResult(
            turn_id=current_turn_id,
            query_status="failed",
            query=None,
            resolved_turn_ids=[],
            selected_artifacts=[],
            summary="History query planning failed.",
            error_message=f"History query planning failed: {exc}",
        )

    resolved_turn_ids = sorted(set(resolved_turn_ids))
    query = _compile_history_query(
        turn_selections,
        resolved_turn_ids,
        selected_artifacts,
        operations,
    )
    if not selected_artifacts:
        return HistoryQueryResult(
            turn_id=current_turn_id,
            query_status="not_found",
            query=query,
            resolved_turn_ids=resolved_turn_ids,
            selected_artifacts=[],
            summary="No artifact matched the validated history selections.",
            error_message="",
        )

    for artifacts in comparison_artifacts:
        formal_reports = [
            artifact
            for artifact in artifacts
            if artifact.artifact_type.endswith("_report_json")
        ]
        contains_paired_report = any(
            artifact.artifact_type == "mut_integrate_report_json"
            for artifact in formal_reports
        )
        if len(formal_reports) < 2 and not contains_paired_report:
            clarified_query = query.model_copy(
                update={
                    "clarification_question": (
                        "Which additional completed report or turn should be included "
                        "in the comparison?"
                    )
                }
            )
            return HistoryQueryResult(
                turn_id=current_turn_id,
                query_status="clarification_required",
                query=clarified_query,
                resolved_turn_ids=resolved_turn_ids,
                selected_artifacts=[],
                summary="The comparison has fewer than two compatible reports.",
                error_message="",
            )

    return HistoryQueryResult(
        turn_id=current_turn_id,
        query_status="succeeded",
        query=query,
        resolved_turn_ids=resolved_turn_ids,
        selected_artifacts=selected_artifacts,
        summary=(
            f"Selected {len(selected_artifacts)} artifact(s) from turns "
            f"{resolved_turn_ids} through validated catalog IDs."
        ),
        error_message="",
    )
