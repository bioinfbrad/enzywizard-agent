"""Offline contracts for deterministic FAIR-term and failure-hint retrieval."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from langchain_core.messages import HumanMessage, SystemMessage
import yaml

from build_fair_terms import (
    DEFAULT_REVIEWED_DESCRIPTIONS_PATH,
    EXCLUDED_TERMS,
    PREFERRED_TERM_SOURCES,
    SOURCE_TOOLS,
    TERM_LABEL_OVERRIDES,
    build_terms,
    extract_term_sources,
    keep_term_source,
    load_cached_descriptions,
    load_reviewed_aliases,
    load_reviewed_descriptions,
    validate_reviewed_sources,
)
from evaluation_utils import aggregate_usage, runtime_usage_payload
from enzywizard_agent.conf import AgentConfig
from enzywizard_agent.data_models.user_confirmation import UserConfirmation
from enzywizard_agent.services.analysis_service import _scientific_terminology
from enzywizard_agent.services.knowledge_retrieval_service import (
    RetrievalProfile,
    enrich_messages_with_fair_terms,
    load_fair_terms,
    retrieve_entries,
    retrieve_failure_hints,
    retrieve_terminology,
)
from enzywizard_agent.utils.CLI_utils import _display_sections
from enzywizard_agent.utils.terminology_utils import (
    annotate_terminology,
    claim_evidence_number,
    scientific_answer_body,
    terminology_key,
    terminology_number,
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
    alias_matches = retrieve_terminology(
        entries,
        "Compare the residue substitution before docking.",
        usage_category="request_routing",
    )
    assert [match.entry["term"] for match in alias_matches] == [
        "amino acid substitution"
    ]
    assert alias_matches[0].matched_field == "aliases"

    category_filtered = retrieve_terminology(
        entries,
        "Explain the binding pocket.",
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
    assert "do not infer" in str(enriched[-1].content).casefold()
    assert "computational evidence" in str(enriched[-1].content).casefold()
    records = retrieve_state["runtime_session"].metadata["knowledge_retrieval"]
    assert len(records) == 1
    assert records[0]["stage"] == "routing.request"
    assert records[0]["injected_term_count"] >= 1
    assert records[0]["injected_character_count"] <= 2_400
    assert records[0]["matched_terms"]
    assert records[0]["matched_terms"][0]["uri"] not in enriched[-1].content
    assert "Source:" not in enriched[-1].content
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
    assert control_record["matched_terms"][0]["uri"] not in controlled[-1].content

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
    assert "ignore irrelevant definitions" in str(
        focused[-1].content
    ).casefold()
    focused_record = focused_state["runtime_session"].metadata[
        "knowledge_retrieval"
    ][0]
    assert focused_record["query_sources"] == [
        "current_user_message",
        "stage_focus",
    ]


def check_expanded_terminology_retrieval() -> None:
    entries = load_fair_terms()
    assert len(entries) == 136

    workflow_matches = retrieve_terminology(
        entries,
        "Prepare the ligand with AutoDock Vina before docking.",
        usage_category="workflow_planning",
    )
    assert workflow_matches
    assert workflow_matches[0].entry["term"] == "AutoDock Vina"

    scientific_matches = retrieve_terminology(
        entries,
        "Explain the net electric charge in this report.",
        usage_category="scientific_analysis",
    )
    assert any(
        match.entry["term"] == "net electric charge"
        for match in scientific_matches
    )

    state = _state("retrieve")
    state["current_turn_id"] = 3
    state["current_user_message"] = "Explain RMSF for the report."
    state["current_history_query"] = SimpleNamespace(turn_id=3, resolved_turn_ids=[2])
    state["analysis_results"] = [
        SimpleNamespace(
            turn_id=turn, response_mode="scientific_result", final_answer=answer
        )
        for turn, answer in ((1, "AutoDock Vina"), (2, "The isoelectric point."))
    ]
    enrich_messages_with_fair_terms(
        state,
        stage="analysis.scientific_narrative",
        messages=[HumanMessage(content="Report context")],
        retrieval_context="protein flexibility; secondary structure; binding pocket; IDR",
    )
    record = state["runtime_session"].metadata["knowledge_retrieval"][0]
    terms = [match["term"] for match in record["matched_terms"]]
    assert terms[0] == "root mean square fluctuation"
    assert len(terms) > 3 and record["injected_character_count"] <= 2_400
    enrich_messages_with_fair_terms(
        state,
        stage="analysis.general_answer",
        messages=[HumanMessage(content="Which term was described in that turn?")],
    )
    record = state["runtime_session"].metadata["knowledge_retrieval"][-1]
    terms = [match["term"] for match in record["matched_terms"]]
    assert "isoelectric point" in terms and "AutoDock Vina" not in terms
    assert "referenced_history" in record["query_sources"]
    assert not retrieve_terminology(entries, "a root fluctuation")

    variants = (
        "Intrinsically-disordered regions (IDRs), then IDR and binding-pockets."
    )
    hits = {match.entry["term"] for match in retrieve_terminology(entries, variants)}
    assert hits == {"intrinsically disordered region", "binding pocket"}
    _, displayed = annotate_terminology(variants, entries)
    assert [entry["term"] for entry in displayed] == [
        "intrinsically disordered region", "binding pocket"
    ]
    assert not retrieve_terminology(entries, "an IDR2 field or intrinsically random region")
    control = _state("control")
    control["current_user_message"] = variants
    enrich_messages_with_fair_terms(
        control,
        stage="analysis.scientific_narrative",
        messages=[HumanMessage(content=variants)],
    )
    audit = control["runtime_session"].metadata["knowledge_retrieval"][0]
    assert audit["injected_term_count"] == audit["related_match_count"] == 2
    assert not hits.intersection(item["term"] for item in audit["matched_terms"])


def check_user_terminology_matching() -> None:
    entries = load_fair_terms()
    for spelling, term, reason in (
        ("predicted_binding_pockets.", "binding pocket", "separator_normalized"),
        ("root_mean_square_fluctuation", "root mean square fluctuation", "exact_value"),
    ):
        keys = {
            match.entry["term"]: match.match_reason
            for match in retrieve_terminology(entries, spelling, user_input=True)
        }
        assert keys.get(term) == reason, (spelling, keys)
        if reason != "exact_value":
            assert not annotate_terminology(spelling, entries)[1]

    for query, terms in (
        ("potental energy", {"energy"}),
        ("chemcal bond", {"bond"}),
        ("potential energy", {"potential energy"}),
        ("$.predicted_binding_pockets", {"binding pocket"}),
        ("$['predicted_binding_pockets']", {"binding pocket"}),
        ("$.entries[*].predicted_binding_pockets", {"binding pocket"}),
        ("What does $.reports[-1].predicted_binding_pockets mean?", {"binding pocket"}),
        ("What does $.reports[0]['predicted_binding_pockets'] mean?", {"binding pocket"}),
        ("Compare WT/MUT", {"wild-type", "mutant"}),
        ("30 cubic angstroms", {"cubic angstrom"}),
    ):
        matches = retrieve_terminology(entries, query, user_input=True)
        assert {match.entry["term"] for match in matches} == terms, (query, matches)
    for separator in (" ", "\t", "-", "\u2010", "\u2011", "\u2013", "\u2014", "\u2015", "\u2212", "\ufe63", "\uff0d"):
        for first, last, term in (
            ("binding", "pockets", "binding pocket"),
            ("secondary", "structure", "secondary structure"),
        ):
            label = f"{first}{separator}{last}"
            assert [m.entry["term"] for m in retrieve_terminology(
                entries, label, user_input=True
            )] == [term], label
            marked, used = annotate_terminology(label, entries)
            assert [entry["term"] for entry in used] == [term], label
            assert marked == label + terminology_number(1)
    assert retrieve_terminology(entries, "chemcal bond", user_input=True)[0].matched_text == (
        "bond"
    )
    for volume in ("30 cubic angstroms", "30 \u00c5\u00b3", "30 \u00c5^3"):
        answer = f"The reported volume is {volume}."
        assert [m.entry["term"] for m in retrieve_terminology(entries, answer, user_input=True)] == [
            "cubic angstrom"
        ]
        assert [e["term"] for e in annotate_terminology(answer, entries)[1]] == [
            "cubic angstrom"
        ]

    for query in (
        "RMSD",
        "binding site",
        "binding/pocket",
        "binding\u207bpocket",
        "use 12 modes",
        "binding pokcet",
        "binding pockt",
        "binding pockeet",
        "binding pockat",
        "dihederal angle",
        "bindng pokcet",
        "predicted_binding_pokcets",
        "$.entries[*].predicted_binding_pokcets",
        "/tmp/predicted_binding_pockets.json",
        "https://example.invalid/?term=binding_pokcet",
        "https://example.invalid/?term='binding_pokcet'",
        "https://example.invalid/?term=binding_pocket",
        "https://example.invalid/?term=$.predicted_binding_pockets",
        "https://example.invalid/?term=$.reports[-1].predicted_binding_pockets",
    ):
        assert not retrieve_terminology(entries, query, user_input=True), query

    ambiguous = [{"term": "binding pocket"}, {"term": "binding packet"}]
    assert not retrieve_terminology(ambiguous, "binding picket", user_input=True)
    assert [m.entry["term"] for m in retrieve_terminology(
        ambiguous, "binding pocket", user_input=True
    )] == ["binding pocket"]

    def inject(state: dict[str, Any], focus: str | None = None) -> dict[str, Any]:
        enrich_messages_with_fair_terms(
            state,
            stage="analysis.scientific_narrative",
            messages=[HumanMessage(content="Report context")],
            retrieval_context=focus,
        )
        return state["runtime_session"].metadata["knowledge_retrieval"][-1]

    follow_up = _state("retrieve")
    follow_up.update(current_turn_id=2, current_user_message="Compare WT binding pokcets.")
    audit = inject(follow_up)
    matches = {item["term"]: item["match_reason"] for item in audit["matched_terms"]}
    assert matches == {"wild-type": "exact_phrase"}

    revision = UserConfirmation(
        turn_id=2, plan_id=1, decision="revision_requested",
        revision_request="Use binding-pockets analysis instead.",
    )
    revised = _state("retrieve")
    revised.update(current_turn_id=2, current_user_confirmation=revision)
    assert inject(revised)["matched_terms"][0]["term"] == "binding pocket"
    assert inject(revised)["query_sources"] == ["revision_request"]

    context_only = _state("retrieve")
    context_only.update(
        current_turn_id=3,
        current_user_confirmation=revision,
        current_history_query=SimpleNamespace(turn_id=3, resolved_turn_ids=[2]),
        analysis_results=[
            SimpleNamespace(
                turn_id=2, response_mode="scientific_result", final_answer="binding pokcet"
            )
        ],
    )
    assert inject(context_only, focus="predicted_binding_pockets")["matched_terms"] == []

    control = _state("control")
    control["current_user_message"] = "predicted_binding_pockets"
    audit = inject(control)
    assert audit["related_match_count"] == audit["injected_term_count"] == 1
    assert not any(item["term"] == "binding pocket" for item in audit["matched_terms"])
    control["current_user_message"] = "binding pokcet"
    assert inject(control)["matched_terms"] == []


def check_unit_terminology_matching() -> None:
    entries = load_fair_terms()
    for term, amounts in (
        ("cubic angstrom", (
            "30\u00c5\u00b3",
            "30\u212b\u00b3",
            "30\u00c5^3",
            "30 angstrom^3",
            "30 angstroms**3",
            "30\u00c5 ^ 3",
            "30\u00c5^{3}",
            "-2.5e+3\u00c5\u00b3",
        )),
        ("cubic centimeter", ("2cm^3", "2 cm\u00b3", "2 centimetres^3", ".5cm**3")),
        ("kilojoule per mole", (
            "80kJ/mol", "80 kJ / mol", "80 kJ mol^-1", "80 kJ mol\u207b\u00b9",
            "80 kJ mol^\u22121", "80 kJ mol^{\u22121}", "80 kJ mol ** \u22121",
        )),
        ("dalton", ("42Da", "42 daltons", "42 Daltons", "42 DALTONS")),
        ("degree", ("90\u00b0", "90 degree of arc")),
        ("angstrom", ("1.2\u00c5", "1.2 angstroms")),
        ("kilocalorie", ("5kcal", "5 kilocalories")),
        ("mole", ("2mol", "2 moles")),
    ):
        for amount in amounts:
            for phrase in (f"The value is {amount}.", f"The value is **{amount}**."):
                assert [m.entry["term"] for m in retrieve_terminology(
                    entries, phrase, user_input=True
                )] == [term], phrase
                assert [e["term"] for e in annotate_terminology(phrase, entries)[1]] == [
                    term
                ], phrase

    for phrase in (
        "30 \u00c5^3.5",
        "30 \u00c5^30",
        "30 \u00c5\u00b2",
        "30 \u00c5\u207b\u00b3",
        "30 \u00c5^-3",
        "30 angstrom^2",
        "30 \u00c5\u00b3 / mol",
        "30 \u00c5\u00b3\u00b7mol",
        "30 \u00c5\u00b3 * mol",
        "30 \u00c5\u00b3 mol.",
        "30 mol \u00c5\u00b3.",
        "30 \u00c5^3\tmol",
        "30 mol^{\u22122} \u00c5\u00b3",
        "30 kJ mol^\u22121 Da",
        "30 cubic angstroms per mole",
        "30 kcal/mol",
        "30 kcal mol^-1",
        "90\u00b0C",
        "42 dA",
        "42 DA",
        "42 da",
        "2 CM^3",
        "80 KJ/mol",
        "80 kJ/MOL",
        "5 KCAL",
        "The model has six degrees of freedom.",
        "The label is PDB42Da.",
        "The label is PDB30\u00c5\u00b3.",
        "Compare WT10 and RMSF12.",
    ):
        assert not retrieve_terminology(entries, phrase, user_input=True), phrase
        assert not annotate_terminology(phrase, entries)[1], phrase

    for phrase in ("2 mol and 42 Da.", "2 mol; 42 Da.", "2 mol\nDa is the symbol."):
        assert {m.entry["term"] for m in retrieve_terminology(
            entries, phrase, user_input=True
        )} == {"mole", "dalton"}, phrase
        assert {e["term"] for e in annotate_terminology(phrase, entries)[1]} == {
            "mole", "dalton"
        }, phrase
    phrase = "The value is 80 kJ mol^{\u22121}."
    marked, _ = annotate_terminology(phrase, entries)
    assert marked == phrase[:-1] + terminology_number(1) + "."
    assert not annotate_terminology("The field is `30\u00c5\u00b3`.", entries)[1]
    assert not retrieve_terminology(
        entries, "See https://example.invalid/?value=30\u00c5\u00b3", user_input=True
    )


def check_scientific_terminology_output() -> None:
    answer = (
        "`IDR` field. RMSF and IDR. RMSF again. "
        "[Evidence: secondary structure, $.data[*].value]"
    )
    state = _state("retrieve")
    state["current_turn_id"] = 3
    marked, evidence, appendix = _scientific_terminology(state, answer)
    first, second = terminology_number(1), terminology_number(2)
    assert marked == answer.replace("RMSF and IDR", f"RMSF{first} and IDR{second}")
    assert [item.source_label for item in evidence] == [
        "root mean square fluctuation", "intrinsically disordered region"
    ]
    assert all(
        item.turn_id == 3 and item.source_kind == "knowledge_base" for item in evidence
    )
    assert all(
        item.value_summary in appendix and item.source_uri in appendix for item in evidence
    )
    assert appendix.index(first) < appendix.index(second)
    assert "secondary structure" not in appendix
    assert scientific_answer_body(marked + appendix) == answer
    assert scientific_answer_body("1 nm\u00b2" + appendix) == "1 nm\u00b2"
    for mode in ("off", "control"):
        assert _scientific_terminology(_state(mode), answer) == (answer, [], "")

    ambiguous = [
        {"term": "pocket", "uri": uri, "description": "a definition"}
        for uri in ("https://example.invalid/a", "https://example.invalid/b")
    ]
    assert annotate_terminology("A pocket.", ambiguous) == ("A pocket.", [])


def check_turn_requires_explicit_scientific_label() -> None:
    entries = load_fair_terms()
    marker = claim_evidence_number(1)
    for prose in (
        "Turn 1 flexibility report",
        "The report from the previous turn",
        "This, in turn, affects the mean",
        "Turn to the flexibility report",
    ):
        answer = f"{prose}: mean RMSF = 0.318195{marker}."
        marked, evidence, appendix = _scientific_terminology(_state("retrieve"), answer)
        assert marked == answer.replace("RMSF", "RMSF" + terminology_number(1)), marked
        assert [item.source_label for item in evidence] == ["root mean square fluctuation"]
        assert "hydrogen-bonded turn" not in appendix
        assert scientific_answer_body(marked + appendix, keep_claim_markers=True) == answer
        for user_input in (False, True):
            assert [
                match.entry["term"]
                for match in retrieve_terminology(entries, answer, user_input=user_input)
            ] == ["root mean square fluctuation"]

    term = "hydrogen-bonded turn"
    for label in (term, "hydrogen-bonded turns", "DSSP T", "DSSP code T"):
        answer = f"From the previous turn: {label}."
        marked, evidence, appendix = _scientific_terminology(_state("retrieve"), answer)
        assert marked == f"From the previous turn: {label}{terminology_number(1)}."
        assert [item.source_label for item in evidence] == [term]
        assert term in appendix
        for user_input in (False, True):
            assert [
                match.entry["term"]
                for match in retrieve_terminology(entries, answer, user_input=user_input)
            ] == [term]

    _, used = annotate_terminology("The graph has a node and an HMM profile.", entries)
    assert {"node", "profile"} <= {entry["term"] for entry in used}


def check_terminology_appendix_uses_single_plain_numbers() -> None:
    citation = claim_evidence_number(1)
    answer = f"The mean RMSF is 0.318195{citation}."
    marked, _, appendix = _scientific_terminology(_state("retrieve"), answer)
    evidence = "\n\nEvidence for cited conclusions:\n[1] flexibility.json"
    sections = dict(_display_sections("Analysis", marked + evidence + appendix))
    assert sections["Analysis"] == marked
    assert sections["Evidence for cited conclusions"] == "[1] flexibility.json"
    glossary = sections["Terminology sources"]
    assert glossary.startswith("(1) root mean square fluctuation:"), glossary
    assert terminology_number(1) not in glossary
    stored = f"\n\nTerminology sources:\n- {terminology_number(12)} Example: A definition."
    assert dict(_display_sections("Analysis", answer + stored))["Terminology sources"] == (
        "(12) Example: A definition."
    )


def check_knowledge_base_provenance() -> None:
    payload = yaml.safe_load(KNOWLEDGE_BASE_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert set(payload) == {"terms"}
    terms = payload["terms"]
    assert isinstance(terms, list) and terms
    reviewed_aliases, reviewed_unit_symbols = load_reviewed_aliases()
    assert {item["term"].casefold(): item["aliases"] for item in terms if item["aliases"]} == (
        reviewed_aliases
    )
    assert {
        item["term"].casefold(): item["unit_symbols"] for item in terms if "unit_symbols" in item
    } == reviewed_unit_symbols

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
        expected_pairs = {
            (TERM_LABEL_OVERRIDES.get(pair, pair[0]).casefold(), pair[1])
            for pair in expected_pairs
            if keep_term_source(*pair)
        }
        assert len(expected_pairs) == 136
        with patch("build_fair_terms.fetch_description", return_value=None):
            rebuilt, _ = build_terms(
                source_terms, timeout_seconds=1, workers=1,
                reviewed_descriptions=reviewed,
                cached_descriptions=load_cached_descriptions(KNOWLEDGE_BASE_PATH),
            )
        assert rebuilt == terms
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
        if item["uri"].split("://", 1)[-1].startswith("qudt.org/vocab/unit/"):
            expected_fields = required_fields | {"kind"}
            if item["term"].casefold() in reviewed_unit_symbols:
                expected_fields.add("unit_symbols")
            assert set(item) == expected_fields
            assert item["kind"] == "unit"
        else:
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

    assert len(terms) == 136
    assert len({terminology_key(term) for term, _ in observed_pairs}) == 136
    assert not EXCLUDED_TERMS.intersection(term for term, _ in observed_pairs)
    assert set(PREFERRED_TERM_SOURCES.items()) <= observed_pairs
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
        check_user_terminology_matching,
        check_unit_terminology_matching,
        check_scientific_terminology_output,
        check_turn_requires_explicit_scientific_label,
        check_terminology_appendix_uses_single_plain_numbers,
        check_knowledge_base_provenance,
    )
    for check in checks:
        check()
        print(f"[PASS] {check.__name__}")
    print(f"All {len(checks)} knowledge retrieval checks passed.")


if __name__ == "__main__":
    main()
