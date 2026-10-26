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


FAIR_TERM_PROFILE = RetrievalProfile(
    field_weights={"term": 1.0, "aliases": 0.95, "description": 0.75},
    threshold=0.70,
    top_k=3,
    require_exact_phrase=False,
    allow_partial_match=True,
)

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


def _retrieval_query(
    state: AgentState,
    retrieval_context: str | None,
) -> tuple[str, list[str]]:
    """Build a bounded query from user intent and an optional stage focus."""

    chunks: list[str] = []
    sources: list[str] = []
    user_request = str(state.get("current_user_message", "")).strip()
    if user_request:
        chunks.append(user_request)
        sources.append("current_user_message")
    stage_focus = str(retrieval_context or "").strip()
    if stage_focus and _normalized_text(stage_focus) != _normalized_text(user_request):
        chunks.append(stage_focus)
        sources.append("stage_focus")
    return "\n\n".join(chunks)[-6_000:], sources


def _render_fair_context(matches: Sequence[KnowledgeMatch]) -> str:
    lines = [
        "Local FAIR terminology (reference only):",
        "Use these definitions only to interpret concepts already present in the "
        "request or current context. Do not infer additional requested actions.",
    ]
    for match in matches:
        term = str(match.entry.get("term", "")).strip()
        description = str(match.entry.get("description", "")).strip()
        uri = str(match.entry.get("uri", "")).strip()
        lines.append(f"- {term}: {description} Source: {uri}")
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
) -> list[KnowledgeMatch]:
    """Choose a deterministic same-count irrelevant-metadata control."""

    if not related:
        return []
    related_uris = {str(item.entry.get("uri", "")) for item in related}
    query_tokens = _tokens(_normalized_text(query))
    candidates: list[Mapping[str, Any]] = []
    for entry in entries:
        categories = entry.get("usage_categories", [])
        if (
            usage_category is not None
            and isinstance(categories, list)
            and categories
            and usage_category not in categories
        ):
            continue
        if str(entry.get("uri", "")) in related_uris:
            continue
        labels = " ".join(
            _field_values(entry, "term") + _field_values(entry, "aliases")
        )
        if _tokens(_normalized_text(labels)).intersection(query_tokens):
            continue
        candidates.append(entry)
    candidates.sort(
        key=lambda entry: sha256(
            f"{query}|{entry.get('uri', '')}|{entry.get('term', '')}".encode("utf-8")
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
    query, query_sources = _retrieval_query(state, retrieval_context)
    entries = load_fair_terms()
    related_matches = retrieve_entries(
        entries,
        query,
        FAIR_TERM_PROFILE,
        usage_category=_stage_category(stage),
    )
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
        )
        context = _render_fair_context(matches) if matches else ""
    else:
        matches = related_matches
        context = related_context
    _record_retrieval(
        state,
        stage=stage,
        query=query,
        query_sources=query_sources,
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
