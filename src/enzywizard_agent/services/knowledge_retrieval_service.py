from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from importlib.resources import files
import re
import time
import unicodedata
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage
import yaml

from ..state import AgentState
from ..utils.terminology_utils import (
    find_terminology_mentions,
    find_user_terminology_mentions,
    scientific_answer_body,
    terminology_forms,
    terminology_key,
)
from .runtime_service import ensure_runtime_session, utc_now


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_ONTOLOGY_STAGE_PREFIXES = (
    "routing.request",
    "routing.history.",
    "planning.intent_classification",
    "planning.workflow_requirements",
    "planning.parameter_extraction",
    "analysis.scientific_narrative",
    "analysis.general_answer",
)


@dataclass(frozen=True)
class RetrievalProfile:
    """Configure one deterministic matching policy over a shared scorer."""

    field_weights: Mapping[str, float]
    threshold: float
    top_k: int | None
    require_exact_phrase: bool
    allow_partial_match: bool
    prefer_longest_match: bool = True
    suppress_contained_matches: bool = False


@dataclass(frozen=True)
class KnowledgeMatch:
    entry: Mapping[str, Any]
    score: float
    matched_field: str
    matched_text: str
    match_reason: str


_FAIR_CONTEXT_MAX_CHARACTERS = 2_400

FAILURE_HINT_PROFILE = RetrievalProfile(
    field_weights={"contains": 1.0},
    threshold=1.0,
    top_k=3,
    require_exact_phrase=True,
    allow_partial_match=False,
    suppress_contained_matches=True,
)


def _normalized_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(_TOKEN_PATTERN.findall(text))


def _tokens(value: str) -> set[str]:
    return set(value.split())


def _field_values(entry: Mapping[str, Any], field_name: str) -> list[str]:
    value = entry.get(field_name)
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value if str(item).strip()]
    return []


def _applies_to_category(entry: Mapping[str, Any], category: str | None) -> bool:
    categories = entry.get("usage_categories", [])
    return not (
        category is not None
        and isinstance(categories, list)
        and categories
        and category not in categories
    )


def retrieve_terminology(
    entries: Iterable[Mapping[str, Any]],
    query: str,
    *,
    usage_category: str | None = None,
    user_input: bool = False,
) -> list[KnowledgeMatch]:
    eligible = tuple(entry for entry in entries if _applies_to_category(entry, usage_category))
    mentions = (
        find_user_terminology_mentions(query, eligible)
        if user_input
        else find_terminology_mentions(query, eligible)
    )
    matches: dict[str, KnowledgeMatch] = {}
    for mention in mentions:
        key = terminology_key(mention.entry["term"])
        score = 0.90 if mention.match_reason == "separator_normalized" else (
            1.0 if mention.field == "term" else 0.95
        )
        previous = matches.get(key)
        if previous is None or score > previous.score:
            matched_text = query[mention.start:mention.end]
            matches[key] = KnowledgeMatch(
                mention.entry,
                score,
                mention.field,
                matched_text,
                mention.match_reason
                or ("exact_value" if matched_text == query.strip() else "exact_phrase"),
            )
    return sorted(
        matches.values(),
        key=lambda match: (-match.score, -len(match.matched_text)),
    )


def _score_value(
    query: str,
    query_tokens: set[str],
    value: str,
    *,
    weight: float,
    profile: RetrievalProfile,
) -> tuple[float, str] | None:
    candidate = _normalized_text(value)
    if not candidate:
        return None
    padded_query = f" {query} "
    if candidate == query:
        return weight, "exact_value"
    if f" {candidate} " in padded_query:
        return weight, "exact_phrase"
    if profile.require_exact_phrase or not profile.allow_partial_match:
        return None

    candidate_tokens = _tokens(candidate)
    if not candidate_tokens:
        return None
    overlap = candidate_tokens.intersection(query_tokens)
    coverage = len(overlap) / len(candidate_tokens)
    if not overlap:
        return None
    # Full token coverage remains below phrase matching; sparse overlaps fall off
    # quickly so broad definitions do not dominate label and alias matches.
    return weight * (0.35 + 0.55 * coverage), "token_coverage"


def retrieve_entries(
    entries: Iterable[Mapping[str, Any]],
    query: str,
    profile: RetrievalProfile,
    *,
    usage_category: str | None = None,
) -> list[KnowledgeMatch]:
    """Rank knowledge entries deterministically with a configurable policy."""

    normalized_query = _normalized_text(query)
    if not normalized_query:
        return []
    query_tokens = _tokens(normalized_query)

    matches: list[KnowledgeMatch] = []
    for entry in entries:
        categories = entry.get("usage_categories", [])
        if (
            usage_category is not None
            and isinstance(categories, list)
            and categories
            and usage_category not in categories
        ):
            continue

        best: KnowledgeMatch | None = None
        for field_name, weight in profile.field_weights.items():
            for value in _field_values(entry, field_name):
                scored = _score_value(
                    normalized_query,
                    query_tokens,
                    value,
                    weight=weight,
                    profile=profile,
                )
                if scored is None:
                    continue
                score, reason = scored
                candidate = KnowledgeMatch(
                    entry=entry,
                    score=score,
                    matched_field=field_name,
                    matched_text=value,
                    match_reason=reason,
                )
                if best is None or (
                    candidate.score,
                    len(_normalized_text(candidate.matched_text)),
                    candidate.matched_field,
                ) > (
                    best.score,
                    len(_normalized_text(best.matched_text)),
                    best.matched_field,
                ):
                    best = candidate
        if best is not None and best.score >= profile.threshold:
            matches.append(best)

    matches.sort(
        key=lambda item: (
            -item.score,
            -len(_normalized_text(item.matched_text))
            if profile.prefer_longest_match
            else 0,
            str(item.entry.get("term") or item.entry.get("contains") or "").casefold(),
            str(item.entry.get("uri") or ""),
        )
    )

    if profile.suppress_contained_matches:
        selected: list[KnowledgeMatch] = []
        for match in matches:
            normalized = _normalized_text(match.matched_text)
            if any(
                normalized != _normalized_text(existing.matched_text)
                and f" {normalized} "
                in f" {_normalized_text(existing.matched_text)} "
                for existing in selected
            ):
                continue
            selected.append(match)
        matches = selected

    if profile.top_k is not None:
        matches = matches[: profile.top_k]
    return matches


@lru_cache(maxsize=1)
def load_fair_terms() -> tuple[dict[str, Any], ...]:
    path = files("enzywizard_agent.knowledge_base").joinpath("fair_terms.yaml")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeError, yaml.YAMLError):
        return ()
    terms = payload.get("terms", []) if isinstance(payload, dict) else []
    return tuple(item for item in terms if isinstance(item, dict))


@lru_cache(maxsize=1)
def fair_terms_sha256() -> str | None:
    path = files("enzywizard_agent.knowledge_base").joinpath("fair_terms.yaml")
    try:
        return sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def retrieve_failure_hints(
    error_text: str,
    rules: Iterable[Mapping[str, Any]],
) -> list[dict[str, str]]:
    matches = retrieve_entries(rules, error_text, FAILURE_HINT_PROFILE)
    return [
        {
            "cause": str(match.entry["cause"]),
            "action": str(match.entry["action"]),
        }
        for match in matches
        if match.entry.get("cause") and match.entry.get("action")
    ]


def _stage_category(stage: str) -> str | None:
    if stage.startswith("routing.history."):
        return "history_query"
    if stage.startswith("routing.request"):
        return "request_routing"
    if stage.startswith("planning.parameter_extraction"):
        return "parameter_extraction"
    if stage.startswith("planning."):
        return "workflow_planning"
    if stage.startswith("analysis."):
        return "scientific_analysis"
    return None


def _ontology_enabled_for_stage(state: AgentState, stage: str) -> bool:
    config = state.get("config")
    return bool(
        config is not None
        and config.ontology_mode in {"retrieve", "control"}
        and not stage.endswith(".repair")
        and stage.startswith(_ONTOLOGY_STAGE_PREFIXES)
    )


def _retrieval_queries(
    state: AgentState,
    retrieval_context: str | None,
) -> dict[str, str]:
    """Keep user intent ahead of stage context and resolved history."""
    queries: dict[str, str] = {}
    turn_id = state.get("current_turn_id", 0)
    confirmation = state.get("current_user_confirmation")
    if (
        confirmation is not None
        and confirmation.turn_id == turn_id
        and confirmation.decision == "revision_requested"
        and confirmation.revision_request
    ):
        queries["revision_request"] = confirmation.revision_request[:6_000]
    user_request = str(state.get("current_user_message", "")).strip()
    if user_request:
        queries["current_user_message"] = user_request[:6_000]
    stage_focus = str(retrieval_context or "").strip()
    routing = state.get("current_request_routing")
    if (
        not stage_focus
        and routing is not None
        and routing.turn_id == turn_id
        and routing.output is not None
    ):
        stage_focus = "; ".join(action.target for action in routing.output.actions)
    if stage_focus and _normalized_text(stage_focus) != _normalized_text(user_request):
        queries["stage_focus"] = stage_focus[:2_000]
    history = state.get("current_history_query")
    if history is not None and history.turn_id == turn_id and history.resolved_turn_ids:
        resolved_turns = set(history.resolved_turn_ids)
        prior_answers = [
            scientific_answer_body(result.final_answer)
            for result in state.get("analysis_results", [])
            if result.turn_id in resolved_turns
            and result.turn_id < turn_id
            and result.response_mode == "scientific_result"
        ][-2:]
        history_context = "\n\n".join(
            answer.split("\n\nEvidence supporting the analysis:", 1)[0][:1_000]
            for answer in prior_answers
            if answer
        )
        if history_context:
            queries["referenced_history"] = history_context
    return queries


def _render_fair_context(matches: Sequence[KnowledgeMatch]) -> str:
    lines = [
        "Local FAIR terminology (reference only):",
        "Use these definitions only to interpret concepts already present in the "
        "request or current context. Ignore irrelevant definitions. Do not infer "
        "additional requested actions or use definitions as computational evidence.",
    ]
    for match in matches:
        term = str(match.entry.get("term", "")).strip()
        description = str(match.entry.get("description", "")).strip()
        lines.append(f"- {term}: {description}")
    return "\n".join(lines)


def _fit_fair_context(
    matches: Sequence[KnowledgeMatch],
) -> tuple[list[KnowledgeMatch], str]:
    """Keep complete definitions while enforcing a small prompt budget."""

    selected: list[KnowledgeMatch] = []
    context = ""
    for match in matches:
        candidate = [*selected, match]
        candidate_context = _render_fair_context(candidate)
        if len(candidate_context) > _FAIR_CONTEXT_MAX_CHARACTERS:
            continue
        selected = candidate
        context = candidate_context
    return selected, context


def _record_retrieval(
    state: AgentState,
    *,
    stage: str,
    query: str,
    query_sources: Sequence[str],
    matches: Sequence[KnowledgeMatch],
    context: str,
    started_monotonic: float,
    mode: str,
    related_match_count: int,
) -> None:
    runtime = ensure_runtime_session(state)
    records = runtime.metadata.setdefault("knowledge_retrieval", [])
    if not isinstance(records, list):
        records = []
        runtime.metadata["knowledge_retrieval"] = records
    records.append(
        {
            "timestamp": utc_now(),
            "stage": stage,
            "mode": mode,
            "knowledge_base_sha256": fair_terms_sha256(),
            "related_match_count": related_match_count,
            "query_character_count": len(query),
            "query_sources": list(query_sources),
            "matched_terms": [
                {
                    "term": str(match.entry.get("term", "")),
                    "uri": str(match.entry.get("uri", "")),
                    "description": str(match.entry.get("description", "")),
                    "source_documents": list(match.entry.get("source_documents", [])),
                    "related_tools": list(match.entry.get("related_tools", [])),
                    "usage_categories": list(
                        match.entry.get("usage_categories", [])
                    ),
                    "score": round(match.score, 6),
                    "matched_field": match.matched_field,
                    "match_reason": match.match_reason,
                }
                for match in matches
            ],
            "injected_term_count": len(matches),
            "injected_character_count": len(context),
            "estimated_input_tokens": (len(context) + 3) // 4 if context else 0,
            "retrieval_duration_ms": round(
                (time.monotonic() - started_monotonic) * 1000,
                3,
            ),
            "no_match_reason": None if matches else "below_threshold",
        }
    )


def _control_matches(
    entries: Sequence[Mapping[str, Any]],
    related: Sequence[KnowledgeMatch],
    query: str,
    usage_category: str | None,
    matched_terms: set[str],
) -> list[KnowledgeMatch]:
    """Choose a deterministic same-count irrelevant-metadata control."""

    if not related:
        return []
    query_tokens = _tokens(_normalized_text(query))
    candidates: list[Mapping[str, Any]] = []
    for entry in entries:
        if not _applies_to_category(entry, usage_category):
            continue
        if terminology_key(entry["term"]) in matched_terms:
            continue
        labels = " ".join(terminology_forms(entry))
        if _tokens(_normalized_text(labels)) & query_tokens:
            continue
        candidates.append(entry)
    candidates.sort(
        key=lambda entry: sha256(
            f"{query}|{terminology_key(entry['term'])}".encode("utf-8")
        ).hexdigest()
    )
    selected: list[KnowledgeMatch] = []
    for entry in candidates:
        match = KnowledgeMatch(
            entry=entry,
            score=0.0,
            matched_field="control",
            matched_text=str(entry.get("term", "")),
            match_reason="irrelevant_metadata_control",
        )
        if len(_render_fair_context([*selected, match])) > _FAIR_CONTEXT_MAX_CHARACTERS:
            continue
        selected.append(match)
        if len(selected) == len(related):
            break
    return selected


def enrich_messages_with_fair_terms(
    state: AgentState,
    *,
    stage: str,
    messages: Sequence[BaseMessage],
    retrieval_context: str | None = None,
) -> list[BaseMessage]:
    """Append compact local terminology to dynamic context when enabled."""

    original = list(messages)
    if not _ontology_enabled_for_stage(state, stage):
        return original

    started_monotonic = time.monotonic()
    queries = _retrieval_queries(state, retrieval_context)
    query = "\n\n".join(queries.values())
    entries = load_fair_terms()
    matches_by_term: dict[str, KnowledgeMatch] = {}
    for source, text in queries.items():
        user_input = source in {"current_user_message", "revision_request"}
        matches = retrieve_terminology(
            entries, text, usage_category=_stage_category(stage), user_input=user_input
        )
        if user_input:
            matches.sort(key=lambda match: not (
                match.matched_field == "aliases"
                and match.match_reason in {"exact_value", "exact_phrase"}
            ))
        for match in matches:
            key = terminology_key(match.entry["term"])
            matches_by_term.setdefault(key, match)
    related_matches = list(matches_by_term.values())
    related_matches, related_context = (
        _fit_fair_context(related_matches) if related_matches else ([], "")
    )
    mode = state["config"].ontology_mode
    if mode == "control":
        matches = _control_matches(
            entries,
            related_matches,
            query,
            _stage_category(stage),
            set(matches_by_term),
        )
        context = _render_fair_context(matches) if matches else ""
    else:
        matches = related_matches
        context = related_context
    _record_retrieval(
        state,
        stage=stage,
        query=query,
        query_sources=list(queries),
        matches=matches,
        context=context,
        started_monotonic=started_monotonic,
        mode=mode,
        related_match_count=len(related_matches),
    )
    if not context:
        return original

    for index in range(len(original) - 1, -1, -1):
        message = original[index]
        if isinstance(message, HumanMessage) and isinstance(message.content, str):
            original[index] = message.model_copy(
                update={"content": f"{message.content.rstrip()}\n\n{context}"}
            )
            return original
    return [*original, HumanMessage(content=context)]
