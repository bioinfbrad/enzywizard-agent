"""Build the local FAIR terminology YAML from canonical tool term links."""

from __future__ import annotations

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import socket
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

import yaml


SOURCE_TOOLS = (
    "enzywizard-clean",
    "enzywizard-mut-clean",
    "enzywizard-aaprops",
    "enzywizard-hydrocluster",
    "enzywizard-energy",
    "enzywizard-flexibility",
    "enzywizard-disorder",
    "enzywizard-conservation",
    "enzywizard-embedding",
    "enzywizard-pocket",
    "enzywizard-substrate",
    "enzywizard-dock",
    "enzywizard-interaction",
    "enzywizard-integrate",
    "enzywizard-batch",
    "enzywizard-mut-batch",
    "enzywizard-mut-integrate",
)

TERM_LINK_PATTERN = re.compile(
    r"['\"](?P<term>[^'\"\n]+)['\"]\s*:\s*"
    r"(?P<url>https?://[^\s;,)]+)",
)
SPACE_PATTERN = re.compile(r"\s+")
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
USER_AGENT = "EnzyWizard FAIR terminology builder/1.0"
UNIT_HOST = "qudt.org"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEWED_DESCRIPTIONS_PATH = (
    PROJECT_ROOT
    / "manual_tests"
    / "fixtures"
    / "fair_terms_descriptions_reviewed.txt"
)
SOFTWARE_TERMS = {
    "autodock vina",
    "dssp",
    "esm 2",
    "openmm force",
    "pyvol",
    "rdkit",
    "ring interaction type",
}

# These links did not expose a reliable, broadly applicable definition for the
# agent context. This reviewed gate excludes page titles, API signatures,
# examples, and definitions whose scope would be misleading here.
UNRELIABLE_DESCRIPTION_TERMS = {
    "binding pocket",
    "box",
    "centroid",
    "dssp",
    "emission probability",
    "enzyme",
    "hmmer",
    "logp",
    "node",
    "openmm force",
    "protein language model",
    "rdkit",
    "substrate",
    "wild type",
}

SCIENTIFIC_ONLY_TERMS = {
    "alanine",
    "arginine",
    "asparagine",
    "aspartic acid",
    "beta bridge",
    "classification",
    "coordinate",
    "cysteine",
    "diameter",
    "distance",
    "extended strand",
    "glutamic acid",
    "glutamine",
    "glycine",
    "histidine",
    "index",
    "information content",
    "interaction type",
    "isoleucine",
    "leucine",
    "lysine",
    "methionine",
    "one letter code",
    "phenylalanine",
    "proline",
    "serine",
    "size",
    "standard deviation",
    "statistics",
    "threonine",
    "tryptophan",
    "turn",
    "tyrosine",
    "uncharged polar",
    "valine",
    "volume",
}


@dataclass(frozen=True)
class TermSource:
    term: str
    uri: str
    source_documents: tuple[str, ...]


@dataclass(frozen=True)
class ReviewedDescription:
    term: str
    original_uri: str
    uri: str
    description: str


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.skip_depth = 0
        self.blocks: list[str] = []
        self._current: list[str] = []
        self.meta_descriptions: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self.skip_depth += 1
            return
        values = {name.casefold(): value or "" for name, value in attrs}
        if tag == "meta" and values.get("content"):
            marker = (values.get("name") or values.get("property")).casefold()
            if marker in {"description", "og:description", "twitter:description"}:
                self.meta_descriptions.append(values["content"])
        if tag in {"p", "li", "td", "dd", "section", "article", "h1", "h2", "h3"}:
            self._flush()

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self.skip_depth = max(0, self.skip_depth - 1)
            return
        if tag in {"p", "li", "td", "dd", "section", "article", "h1", "h2", "h3"}:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self.skip_depth == 0 and data.strip():
            self._current.append(data)

    def _flush(self) -> None:
        text = _clean_text(" ".join(self._current))
        if text:
            self.blocks.append(text)
        self._current = []

    def close(self) -> None:
        super().close()
        self._flush()


def parse_args() -> Any:
    agent_root = PROJECT_ROOT
    parser = ArgumentParser()
    parser.add_argument(
        "--tools_root",
        type=Path,
        default=agent_root.parent,
        help="Directory containing the 17 EnzyWizard tool repositories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            agent_root
            / "src"
            / "enzywizard_agent"
            / "knowledge_base"
            / "fair_terms.yaml"
        ),
    )
    parser.add_argument(
        "--reviewed_descriptions",
        type=Path,
        default=DEFAULT_REVIEWED_DESCRIPTIONS_PATH,
        help=(
            "Reviewed term descriptions keyed by the original term and URI. "
            "Reviewed entries override network extraction."
        ),
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refetch unreviewed descriptions instead of reusing the output YAML.",
    )
    parser.add_argument("--timeout_seconds", type=float, default=20.0)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def _clean_text(value: str) -> str:
    without_tags = HTML_TAG_PATTERN.sub(" ", value)
    return SPACE_PATTERN.sub(" ", unescape(without_tags)).strip(" \t\r\n-–—|:")


def _normalized(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def extract_term_sources(tools_root: Path) -> tuple[list[TermSource], int]:
    occurrences: dict[tuple[str, str], dict[str, Any]] = {}
    occurrence_count = 0
    missing = []
    for tool_name in SOURCE_TOOLS:
        path = tools_root / tool_name / "README.md"
        if not path.is_file():
            missing.append(str(path))
            continue
        for match in TERM_LINK_PATTERN.finditer(path.read_text(encoding="utf-8")):
            term = _clean_text(match.group("term"))
            uri = match.group("url").rstrip(".,")
            if not term or not uri:
                continue
            occurrence_count += 1
            key = (term.casefold(), uri)
            item = occurrences.setdefault(
                key,
                {"term": term, "uri": uri, "source_documents": set()},
            )
            item["source_documents"].add(tool_name)
    if missing:
        raise FileNotFoundError(
            "Missing canonical source documents:\n" + "\n".join(missing)
        )
    sources = [
        TermSource(
            term=item["term"],
            uri=item["uri"],
            source_documents=tuple(sorted(item["source_documents"])),
        )
        for item in occurrences.values()
    ]
    sources.sort(key=lambda item: (item.term.casefold(), item.uri))
    return sources, occurrence_count


def load_reviewed_descriptions(
    path: Path,
) -> dict[tuple[str, str], ReviewedDescription]:
    """Load reviewed descriptions without exposing review metadata at runtime."""

    if not path.is_file():
        raise FileNotFoundError(f"Reviewed description file not found: {path}")

    text = path.read_text(encoding="utf-8")
    matches = list(re.finditer(r"(?m)^\[(\d{3})\]\s*$", text))
    reviewed: dict[tuple[str, str], ReviewedDescription] = {}
    required_fields = {"term", "uri", "updated_uri", "description"}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        fields: dict[str, str] = {}
        for line in text[match.end() : end].strip().splitlines():
            name, separator, value = line.partition(":")
            if not separator or name not in required_fields:
                raise ValueError(
                    f"Invalid reviewed description line in [{match.group(1)}]: {line}"
                )
            if name in fields:
                raise ValueError(
                    f"Duplicate field {name!r} in reviewed entry [{match.group(1)}]"
                )
            fields[name] = value.strip()
        if set(fields) != required_fields:
            missing = sorted(required_fields.difference(fields))
            raise ValueError(
                f"Reviewed entry [{match.group(1)}] is missing fields: {missing}"
            )

        term = _clean_text(fields["term"])
        original_uri = fields["uri"]
        updated_uri = fields["updated_uri"]
        uri = original_uri if updated_uri == "same" else updated_uri
        description = _clean_text(fields["description"])
        if not term or not original_uri.startswith(("http://", "https://")):
            raise ValueError(f"Invalid term or original URI in [{match.group(1)}]")
        if not uri.startswith(("http://", "https://")):
            raise ValueError(f"Invalid effective URI in [{match.group(1)}]")
        if not description:
            raise ValueError(f"Empty description in [{match.group(1)}]")

        key = (term.casefold(), original_uri)
        if key in reviewed:
            raise ValueError(
                f"Duplicate reviewed term and original URI in [{match.group(1)}]: {key}"
            )
        reviewed[key] = ReviewedDescription(
            term=term,
            original_uri=original_uri,
            uri=uri,
            description=description,
        )

    if not reviewed:
        raise ValueError(f"No reviewed descriptions found in: {path}")
    return reviewed


def load_cached_descriptions(path: Path) -> dict[tuple[str, str], str]:
    """Reuse already accepted descriptions so ordinary rebuilds stay offline."""

    if not path.is_file():
        return {}
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        return {}
    terms = payload.get("terms", []) if isinstance(payload, dict) else []
    cached: dict[tuple[str, str], str] = {}
    for item in terms:
        if not isinstance(item, dict):
            continue
        term = _clean_text(str(item.get("term", "")))
        uri = str(item.get("uri", "")).strip()
        description = _clean_text(str(item.get("description", "")))
        if term and uri and description:
            cached[(term.casefold(), uri)] = description
    return cached


def validate_reviewed_sources(
    sources: list[TermSource],
    reviewed_descriptions: dict[tuple[str, str], ReviewedDescription],
) -> None:
    """Reject reviewed records that do not correspond to a canonical term link."""

    source_keys = {(source.term.casefold(), source.uri) for source in sources}
    unknown = sorted(set(reviewed_descriptions).difference(source_keys))
    if unknown:
        details = "\n".join(f"- {term}: {uri}" for term, uri in unknown)
        raise ValueError(
            "Reviewed descriptions contain term links absent from the 17 canonical "
            f"source documents:\n{details}"
        )


def _request(url: str, timeout_seconds: float) -> tuple[str, str]:
    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json,text/html,application/xhtml+xml;q=0.9,*/*;q=0.1",
        },
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        content_type = response.headers.get("Content-Type", "")
        charset = response.headers.get_content_charset() or "utf-8"
        body = response.read(2_000_000).decode(charset, errors="replace")
    return content_type, body


def _ontology_name(uri: str) -> str | None:
    obo_match = re.search(r"/obo/([A-Za-z]+)_\d+$", uri)
    if obo_match:
        return obo_match.group(1).casefold()
    if urlsplit(uri).netloc.casefold() == "edamontology.org":
        return "edam"
    return None


def _description_from_ols(
    source: TermSource,
    timeout_seconds: float,
) -> str | None:
    ontology = _ontology_name(source.uri)
    if ontology is None:
        return None
    query = urlencode(
        {
            "q": source.term,
            "ontology": ontology,
            "exact": "true",
            "rows": "50",
        }
    )
    _, body = _request(
        f"https://www.ebi.ac.uk/ols4/api/search?{query}",
        timeout_seconds,
    )
    payload = json.loads(body)
    documents = payload.get("response", {}).get("docs", [])
    source_path = urlsplit(source.uri).path.rstrip("/")
    for document in documents:
        iri = str(document.get("iri", ""))
        if urlsplit(iri).path.rstrip("/") != source_path:
            continue
        descriptions = document.get("description") or []
        if isinstance(descriptions, str):
            descriptions = [descriptions]
        for description in descriptions:
            cleaned = _clean_text(str(description))
            if 20 <= len(cleaned) <= 800:
                return cleaned
    return None


def _candidate_sentences(parser: _VisibleTextParser) -> list[str]:
    candidates: list[str] = []
    for block in [*parser.meta_descriptions, *parser.blocks]:
        cleaned_block = _clean_text(block)
        if 20 <= len(cleaned_block) <= 800:
            candidates.append(cleaned_block)
        for sentence in SENTENCE_PATTERN.split(_clean_text(block)):
            sentence = _clean_text(sentence)
            if 20 <= len(sentence) <= 800:
                candidates.append(sentence)
    return candidates


def _description_from_qudt(body: str) -> str | None:
    match = re.search(
        r"dcterms:description.*?<td[^>]*>(?P<value>.*?)</td>",
        body,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        return None
    description = _clean_text(match.group("value"))
    return description if 20 <= len(description) <= 800 else None


def _looks_like_definition(term: str, sentence: str) -> bool:
    normalized = _normalized(sentence)
    if not normalized or normalized == _normalized(term):
        return False
    rejected_markers = (
        "accessed today",
        "available from",
        "cookie",
        "privacy policy",
        "enable javascript",
        "user s guide pdf",
        "documentation",
        "api reference",
        "network architecture",
        "relationship to a single",
        "can be specified similarly",
        "given two bounding boxes",
        "pictorial example",
        "you can open an issue",
        "with apologies prepared",
        "below is an acted in",
        "this method should be overridden",
        "this method is a shortcut",
        "skip to content",
        "copyright",
        "residue specification",
        "gmx dssp f",
    )
    if any(marker in normalized for marker in rejected_markers):
        return False
    if "http://" in sentence or "https://" in sentence or ">>>" in sentence:
        return False
    title_markers = (" glossary ", " schema ", " documentation ", " manual ")
    padded = f" {normalized} "
    if len(sentence.split()) < 5 and any(marker in padded for marker in title_markers):
        return False
    return True


def _description_from_html(term: str, body: str) -> str | None:
    parser = _VisibleTextParser()
    parser.feed(body)
    parser.close()
    term_normalized = _normalized(term)
    term_tokens = set(term_normalized.split())
    if not term_tokens:
        return None

    ranked: list[tuple[float, int, str]] = []
    for sentence in _candidate_sentences(parser):
        normalized = _normalized(sentence)
        sentence_tokens = set(normalized.split())
        coverage = len(term_tokens.intersection(sentence_tokens)) / len(term_tokens)
        phrase = f" {term_normalized} " in f" {normalized} "
        if not phrase and coverage < 1.0:
            continue
        if not _looks_like_definition(term, sentence):
            continue
        definition_cues = (
            " is a ",
            " is an ",
            " refers to ",
            " defined as ",
            " represents ",
            " describes ",
            " computes ",
            " measures ",
            " consists of ",
            " equal to ",
            " unit of ",
            " measure of ",
        )
        cue_bonus = 0.35 if any(cue in f" {normalized} " for cue in definition_cues) else 0.0
        code_definition_bonus = 0.25 if re.match(r"^[A-Z]\s+[—-]", sentence) else 0.0
        length_bonus = min(len(sentence), 300) / 3000
        score = (1.0 if phrase else 0.7) + cue_bonus + code_definition_bonus + length_bonus
        ranked.append((score, len(sentence), sentence))
    if not ranked:
        return None
    ranked.sort(reverse=True)
    return ranked[0][2]


def fetch_description(source: TermSource, timeout_seconds: float) -> str | None:
    if _normalized(source.term) in UNRELIABLE_DESCRIPTION_TERMS:
        return None
    if _ontology_name(source.uri) is not None:
        try:
            return _description_from_ols(source, timeout_seconds)
        except (HTTPError, URLError, TimeoutError, socket.timeout, json.JSONDecodeError):
            return None

    try:
        content_type, body = _request(source.uri, timeout_seconds)
    except (HTTPError, URLError, TimeoutError, socket.timeout, UnicodeError):
        return None
    if urlsplit(source.uri).netloc.casefold() == UNIT_HOST:
        return _description_from_qudt(body)
    if "json" in content_type.casefold():
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return None
        for key in ("description", "definition", "comment"):
            value = payload.get(key) if isinstance(payload, dict) else None
            if isinstance(value, str) and 20 <= len(_clean_text(value)) <= 800:
                return _clean_text(value)
        return None
    return _description_from_html(source.term, body)


def usage_categories(source: TermSource) -> list[str]:
    host = urlsplit(source.uri).netloc.casefold()
    term = _normalized(source.term)
    if term in SCIENTIFIC_ONLY_TERMS:
        return ["scientific_analysis"]
    if term == "report":
        return ["history_query", "scientific_analysis"]
    if host == UNIT_HOST:
        return ["parameter_extraction", "scientific_analysis"]
    if term in SOFTWARE_TERMS or any(
        marker in host for marker in ("docs.", "manual.", "bio.tools")
    ):
        return ["workflow_planning", "scientific_analysis"]
    return [
        "request_routing",
        "history_query",
        "workflow_planning",
        "parameter_extraction",
        "scientific_analysis",
    ]


def build_terms(
    sources: list[TermSource],
    *,
    timeout_seconds: float,
    workers: int,
    reviewed_descriptions: dict[tuple[str, str], ReviewedDescription] | None = None,
    cached_descriptions: dict[tuple[str, str], str] | None = None,
) -> tuple[list[dict[str, Any]], list[TermSource]]:
    reviewed_descriptions = reviewed_descriptions or {}
    cached_descriptions = cached_descriptions or {}
    descriptions: dict[tuple[str, str], str] = {}
    sources_to_fetch = [
        source
        for source in sources
        if (source.term.casefold(), source.uri) not in reviewed_descriptions
        and (source.term.casefold(), source.uri) not in cached_descriptions
    ]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(fetch_description, source, timeout_seconds): source
            for source in sources_to_fetch
        }
        for future in as_completed(futures):
            source = futures[future]
            try:
                description = future.result()
            except Exception:
                description = None
            if description:
                descriptions[(source.term.casefold(), source.uri)] = description

    terms_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    omitted: list[TermSource] = []
    for source in sources:
        original_key = (source.term.casefold(), source.uri)
        reviewed = reviewed_descriptions.get(original_key)
        effective_source = (
            TermSource(
                term=source.term,
                uri=reviewed.uri,
                source_documents=source.source_documents,
            )
            if reviewed is not None
            else source
        )
        description = (
            reviewed.description
            if reviewed is not None
            else cached_descriptions.get(original_key)
            or descriptions.get(original_key)
        )
        if not description:
            omitted.append(source)
            continue
        effective_key = (effective_source.term.casefold(), effective_source.uri)
        existing = terms_by_pair.get(effective_key)
        if existing is None:
            terms_by_pair[effective_key] = {
                "term": effective_source.term,
                "aliases": [],
                "uri": effective_source.uri,
                "description": description,
                "source_documents": list(effective_source.source_documents),
                "related_tools": list(effective_source.source_documents),
                "usage_categories": usage_categories(effective_source),
            }
            continue

        documents = sorted(
            set(existing["source_documents"]).union(effective_source.source_documents)
        )
        existing["source_documents"] = documents
        existing["related_tools"] = documents
        if (len(description), description) > (
            len(str(existing["description"])),
            str(existing["description"]),
        ):
            existing["description"] = description

    terms = sorted(
        terms_by_pair.values(),
        key=lambda item: (str(item["term"]).casefold(), str(item["uri"])),
    )
    return terms, omitted


def main() -> int:
    args = parse_args()
    sources, occurrence_count = extract_term_sources(args.tools_root.resolve())
    reviewed_descriptions = load_reviewed_descriptions(
        args.reviewed_descriptions.resolve()
    )
    validate_reviewed_sources(sources, reviewed_descriptions)
    cached_descriptions = (
        {} if args.refresh else load_cached_descriptions(args.output.resolve())
    )
    terms, omitted = build_terms(
        sources,
        timeout_seconds=args.timeout_seconds,
        workers=args.workers,
        reviewed_descriptions=reviewed_descriptions,
        cached_descriptions=cached_descriptions,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_name(f".{args.output.name}.tmp")
    temporary_output.write_text(
        yaml.safe_dump(
            {"terms": terms},
            allow_unicode=False,
            sort_keys=False,
            width=100,
        ),
        encoding="utf-8",
    )
    temporary_output.replace(args.output)

    print(f"Source documents: {len(SOURCE_TOOLS)}")
    print(f"Term-link occurrences: {occurrence_count}")
    print(f"Unique term-link pairs: {len(sources)}")
    print(f"Reviewed descriptions: {len(reviewed_descriptions)}")
    print(f"Cached accepted descriptions: {len(cached_descriptions)}")
    print(f"Included terms: {len(terms)}")
    print(f"Omitted without a reliable description: {len(omitted)}")
    review_excluded = sum(
        _normalized(source.term) in UNRELIABLE_DESCRIPTION_TERMS
        for source in omitted
    )
    print(f"Omitted by reviewed quality gate: {review_excluded}")
    print(f"Omitted by fetch or automatic validation: {len(omitted) - review_excluded}")
    print(f"Knowledge base: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
