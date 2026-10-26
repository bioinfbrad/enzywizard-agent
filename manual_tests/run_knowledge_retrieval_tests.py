"""Offline contracts for deterministic FAIR-term and failure-hint retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
import yaml

from build_fair_terms import (
    DEFAULT_REVIEWED_DESCRIPTIONS_PATH,
    SOURCE_TOOLS,
    extract_term_sources,
    load_reviewed_descriptions,
    validate_reviewed_sources,
)
from evaluation_utils import aggregate_usage, runtime_usage_payload
from enzywizard_agent.conf import AgentConfig
from enzywizard_agent.services.knowledge_retrieval_service import (
    FAIR_TERM_PROFILE,
    RetrievalProfile,
    enrich_messages_with_fair_terms,
    load_fair_terms,
    retrieve_entries,
    retrieve_failure_hints,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_BASE_PATH = (
    PROJECT_ROOT / "src/enzywizard_agent/knowledge_base/fair_terms.yaml"
)


def _state(mode: str) -> dict[str, Any]:
    return {
        "config": AgentConfig(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            model="test-model",
            workdir=Path("/tmp/enzywizard-knowledge-retrieval-test"),
            ontology_mode=mode,
        ),
        "session_id": "knowledge-retrieval-test",
        "current_user_message": "",
    }


def check_shared_matcher_profiles() -> None:
    entries = [
        {
            "term": "amino acid substitution",
            "aliases": ["residue substitution"],
            "description": "Replacement of one amino acid by another.",
            "usage_categories": ["request_routing"],
        },
        {
            "term": "binding pocket",
            "aliases": [],
            "description": "A region where a ligand can bind.",
            "usage_categories": ["scientific_analysis"],
        },
    ]
    alias_matches = retrieve_entries(
        entries,
        "Compare the residue substitution before docking.",
        FAIR_TERM_PROFILE,
        usage_category="request_routing",
    )
    assert [match.entry["term"] for match in alias_matches] == [
        "amino acid substitution"
    ]
    assert alias_matches[0].matched_field == "aliases"

    category_filtered = retrieve_entries(
        entries,
        "Explain the binding pocket.",
        FAIR_TERM_PROFILE,
        usage_category="request_routing",
    )
    assert category_filtered == []

    top_one = RetrievalProfile(
        field_weights={"term": 1.0},
        threshold=0.6,
        top_k=1,
        require_exact_phrase=False,
        allow_partial_match=True,
    )
    ranked = retrieve_entries(
        [{"term": "pocket"}, {"term": "binding pocket"}],
        "binding pocket",
        top_one,
    )
    assert [match.entry["term"] for match in ranked] == ["binding pocket"]


def check_failure_hint_precision() -> None:
    rules = [
        {
            "contains": "out of memory",
            "cause": "generic memory failure",
            "action": "reduce memory use",
        },
        {
            "contains": "cuda out of memory",
            "cause": "GPU memory failure",
            "action": "reduce GPU memory use",
        },
    ]
    matches = retrieve_failure_hints("RuntimeError: CUDA out of memory", rules)
    assert matches == [
        {
            "cause": "GPU memory failure",
            "action": "reduce GPU memory use",
        }
    ]
    assert retrieve_failure_hints("unrelated transport error", rules) == []


def check_enrichment_modes_and_tracking() -> None:
    default_config = AgentConfig(
        api_key="test-key",
        base_url="https://example.invalid/v1",
        model="test-model",
        workdir=Path("/tmp/enzywizard-knowledge-retrieval-default-test"),
    )
    assert default_config.ontology_mode == "retrieve"

    messages = [
        SystemMessage(content="Classify the request."),
        HumanMessage(content="Explain this amino acid substitution."),
    ]

    off_state = _state("off")
    off_messages = enrich_messages_with_fair_terms(
        off_state,
        stage="routing.request",
        messages=messages,
    )
    assert [message.content for message in off_messages] == [
        message.content for message in messages
    ]
    assert "runtime_session" not in off_state

    retrieve_state = _state("retrieve")
    retrieve_state["current_user_message"] = (
        "Explain this amino acid substitution."
    )
    enriched = enrich_messages_with_fair_terms(
        retrieve_state,
        stage="routing.request",
        messages=messages,
    )
    assert len(enriched) == len(messages)
    assert "Local FAIR terminology" in str(enriched[-1].content)
    assert "amino acid substitution" in str(enriched[-1].content).casefold()
    records = retrieve_state["runtime_session"].metadata["knowledge_retrieval"]
    assert len(records) == 1
    assert records[0]["stage"] == "routing.request"
    assert records[0]["injected_term_count"] >= 1
    assert records[0]["injected_term_count"] <= 3
    assert records[0]["matched_terms"]
    assert records[0]["query_sources"] == ["current_user_message"]
    usage = runtime_usage_payload(retrieve_state["runtime_session"])
    assert usage["knowledge_retrieval"]["event_count"] == 1
    assert usage["knowledge_retrieval"]["events"][0]["stage"] == "routing.request"
    aggregate = aggregate_usage([usage])
    assert aggregate["knowledge_retrieval"]["event_count"] == 1

    control_state = _state("control")
    control_state["current_user_message"] = retrieve_state["current_user_message"]
    controlled = enrich_messages_with_fair_terms(
        control_state,
        stage="routing.request",
        messages=messages,
    )
    control_record = control_state["runtime_session"].metadata[
        "knowledge_retrieval"
    ][0]
    assert control_record["mode"] == "control"
    assert control_record["related_match_count"] == records[0]["injected_term_count"]
    assert control_record["injected_term_count"] == records[0]["injected_term_count"]
    assert control_record["matched_terms"] != records[0]["matched_terms"]
    assert "Local FAIR terminology" in str(controlled[-1].content)

    capability_state = _state("retrieve")
    capability_messages = enrich_messages_with_fair_terms(
        capability_state,
        stage="model_capability_check.routing",
        messages=messages,
    )
    assert [message.content for message in capability_messages] == [
        message.content for message in messages
    ]
    assert "runtime_session" not in capability_state

    no_match_state = _state("retrieve")
    no_match_state["current_user_message"] = "quux_xyzzy_no_match"
    unchanged = enrich_messages_with_fair_terms(
        no_match_state,
        stage="routing.request",
        messages=[HumanMessage(content="quux_xyzzy_no_match")],
    )
    assert unchanged[0].content == "quux_xyzzy_no_match"
    no_match_records = no_match_state["runtime_session"].metadata[
        "knowledge_retrieval"
    ]
    assert no_match_records[0]["injected_term_count"] == 0
    assert no_match_records[0]["no_match_reason"] == "below_threshold"

    scoped_state = _state("retrieve")
    scoped_state["current_user_message"] = "Dock glucose as the substrate."
    scoped = enrich_messages_with_fair_terms(
        scoped_state,
        stage="routing.request",
        messages=[
            HumanMessage(
                content=(
                    "Dynamic catalog mentions amino acid substitution, mutant, "
                    "energy, and protein flexibility."
                )
            )
        ],
    )
    scoped_content = str(scoped[-1].content).casefold()
    assert "local fair terminology" in scoped_content
    assert "amino acid substitution:" not in scoped_content
    scoped_record = scoped_state["runtime_session"].metadata[
        "knowledge_retrieval"
    ][0]
    assert scoped_record["query_character_count"] == len(
        scoped_state["current_user_message"]
    )

    focused_state = _state("retrieve")
    focused_state["current_user_message"] = "Explain this result."
    focused = enrich_messages_with_fair_terms(
        focused_state,
        stage="analysis.scientific_narrative",
        messages=[HumanMessage(content="Large serialized analysis context")],
        retrieval_context="binding pocket",
    )
    assert "binding pocket:" in str(focused[-1].content).casefold()
    focused_record = focused_state["runtime_session"].metadata[
        "knowledge_retrieval"
    ][0]
    assert focused_record["query_sources"] == [
        "current_user_message",
        "stage_focus",
    ]


def check_expanded_terminology_retrieval() -> None:
    entries = load_fair_terms()
    assert len(entries) == 151

    workflow_matches = retrieve_entries(
        entries,
        "Prepare the ligand with AutoDock Vina before docking.",
        FAIR_TERM_PROFILE,
        usage_category="workflow_planning",
    )
    assert workflow_matches
    assert workflow_matches[0].entry["term"] == "AutoDock Vina"

    scientific_matches = retrieve_entries(
        entries,
        "Explain the net electric charge in this report.",
        FAIR_TERM_PROFILE,
        usage_category="scientific_analysis",
    )
    assert any(
        match.entry["term"] == "net electric charge"
        for match in scientific_matches
    )


def check_knowledge_base_provenance() -> None:
    payload = yaml.safe_load(KNOWLEDGE_BASE_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert set(payload) == {"terms"}
    terms = payload["terms"]
    assert isinstance(terms, list) and terms

    allowed_documents = set(SOURCE_TOOLS)
    source_readmes = [
        PROJECT_ROOT.parent / tool_name / "README.md"
        for tool_name in SOURCE_TOOLS
    ]
    expected_pairs: set[tuple[str, str]] | None = None
    expected_reviewed_descriptions: dict[tuple[str, str], str] = {}
    if all(path.is_file() for path in source_readmes):
        source_terms, _ = extract_term_sources(PROJECT_ROOT.parent)
        reviewed = load_reviewed_descriptions(DEFAULT_REVIEWED_DESCRIPTIONS_PATH)
        validate_reviewed_sources(source_terms, reviewed)
        assert len(reviewed) == 94
        reviewed_candidates: dict[tuple[str, str], list[str]] = {}
        for item in reviewed.values():
            reviewed_candidates.setdefault(
                (item.term.casefold(), item.uri), []
            ).append(item.description)
        expected_reviewed_descriptions = {
            pair: max(descriptions, key=lambda value: (len(value), value))
            for pair, descriptions in reviewed_candidates.items()
        }
        assert len(expected_reviewed_descriptions) == 91
        expected_pairs = {
            (
                item.term.casefold(),
                reviewed[(item.term.casefold(), item.uri)].uri
                if (item.term.casefold(), item.uri) in reviewed
                else item.uri,
            )
            for item in source_terms
        }
        assert len(expected_pairs) == 151
    observed_pairs: set[tuple[str, str]] = set()
    required_fields = {
        "term",
        "aliases",
        "uri",
        "description",
        "source_documents",
        "related_tools",
        "usage_categories",
    }
    for item in terms:
        assert isinstance(item, dict)
        assert set(item) == required_fields
        pair = (str(item["term"]).casefold(), str(item["uri"]))
        assert pair not in observed_pairs
        observed_pairs.add(pair)
        assert isinstance(item["aliases"], list)
        assert str(item["description"]).strip()
        if pair in expected_reviewed_descriptions:
            assert item["description"] == expected_reviewed_descriptions[pair]
        assert set(item["source_documents"]).issubset(allowed_documents)
        assert item["related_tools"] == item["source_documents"]
        assert isinstance(item["usage_categories"], list)
        assert item["usage_categories"]

    assert len(terms) == 151
    if expected_pairs is not None:
        assert observed_pairs == expected_pairs

    loaded = load_fair_terms()
    assert len(loaded) == len(terms)


def main() -> None:
    checks = (
        check_shared_matcher_profiles,
        check_failure_hint_precision,
        check_enrichment_modes_and_tracking,
        check_expanded_terminology_retrieval,
        check_knowledge_base_provenance,
    )
    for check in checks:
        check()
        print(f"[PASS] {check.__name__}")
    print(f"All {len(checks)} knowledge retrieval checks passed.")


if __name__ == "__main__":
    main()
