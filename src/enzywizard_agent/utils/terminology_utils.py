from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any


TERMINOLOGY_APPENDIX_MARKER = "\n\nTerminology sources "
CLAIM_EVIDENCE_APPENDIX_MARKER = "\n\nEvidence for cited conclusions:\n"
_SUPERSCRIPTS = str.maketrans(
    "0123456789", "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079"
)
_UNIT_SUPERSCRIPTS = {**_SUPERSCRIPTS, ord("+"): "\u207a", ord("-"): "\u207b"}
_UNIT_ORDINARY = str.maketrans({value: chr(key) for key, value in _UNIT_SUPERSCRIPTS.items()})
_UNIT_SUPER_POWER = re.compile(r"[\u207a\u207b]?[\u2070\u00b9\u00b2\u00b3\u2074-\u2079]+")
_UNIT_OPERATOR = re.compile(r"[ \t]*(/|\^|\*\*)[ \t]*")
_BRACED_POWER = re.compile(r"\^\{[ \t]*([-+]?[0-9]+)[ \t]*\}")
_UNIT_GLYPHS = str.maketrans({"\u2212": "-", "\u212b": "\u00c5"})
_UNIT_POWER = re.compile(r"\^[-+]?[0-9]+")
_SYMBOL_POWER_PATTERN = (
    r"(?:[ \t]*(?:\^|\*\*)[ \t]*[-+]?[0-9]+"
    r"|[ \t]*\^[ \t]*\{[ \t]*[-+]?[0-9]+[ \t]*\}"
    r"|[\u207a\u207b]?[\u2070\u00b9\u00b2\u00b3\u2074-\u2079]+)"
)
_UNIT_POWER_WORDS = {"square": 2, "cubic": 3}
_NUMBER_AT_END = re.compile(
    r"(?<![\w.])[-+\u2212]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+\u2212]?\d+)?$"
)
_UNIT_LEFT_OPERATOR = re.compile(
    r"(?:[/\u00b7\u22c5\u00d7][ \t]*|\bper[ \t]+|\w\*|\w[ \t]*\*[ \t]+)$",
    re.IGNORECASE,
)
_UNIT_RIGHT_OPERATOR = re.compile(
    r"[\w\u207a\u207b]|\.\d|[ \t]*(?:[/\u00b7\u22c5\u00d7^]|\*\*[ \t]*[-+]?\d|\*[ \t]*[^\W\d_])"
    r"|[ \t]+(?:per[ \t]+|[^\W\d_]+[ \t]*(?:\^|\*\*|[\u207a\u207b\u2070\u00b9\u00b2\u00b3\u2074-\u2079]))",
    re.IGNORECASE,
)
_TERM_MARKER = re.compile(r"\u207d[\u2070\u00b9\u00b2\u00b3\u2074-\u2079]+\u207e")
CLAIM_EVIDENCE_MARKER = re.compile(r"\[[\u2070\u00b9\u00b2\u00b3\u2074-\u2079]+\]")
# Leave Markdown code, links, URLs and square-bracketed evidence references alone.
_PROTECTED_TEXT = re.compile(
    r"```[\s\S]*?```|`[^`\n]*`|\[[^\]\n]*\](?:\([^\n)]*\))?|https?://\S+",
    re.IGNORECASE,
)
_TERM_SEPARATOR_PATTERN = r"[\s-]+"
_TERM_SEPARATOR = re.compile(_TERM_SEPARATOR_PATTERN)
_USER_URL = re.compile(r"\b(?:(?:https?|ftp)://|www\.)[^\s`<>\"]+", re.IGNORECASE)
_JSON_PATH = re.compile(
    r"""(?<![\w$])\$(?:\.[A-Za-z_]\w*|\[\s*(?:-?\d+|\*|'[A-Za-z_]\w*'|"[A-Za-z_]\w*")\s*\])+"""
)
_JSON_FIELD = re.compile(r"""\.([A-Za-z_]\w*)|\[\s*['"]([A-Za-z_]\w*)['"]\s*\]""")
_PAIRED_SIDES = re.compile(
    r"(?<![\w./\\])(?:WT\s*/\s*MUT|MUT\s*/\s*WT)(?![\w/\\]|\.\w)",
    re.IGNORECASE,
)
_USER_FIELD = re.compile(
    r"(?<![\w./\\$@-])[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+(?![\w/\\@-]|\.\w)"
)
# Only inflect reviewed countable heads, not units, software names or every label.
_REGULAR_PLURAL_HEADS = frozenset(
    "angle atom bond bridge cluster coordinate embedding fingerprint fluctuation "
    "force interaction model node pocket profile region residue sequence strand "
    "structure substitution substrate term vector".split()
)


def _normalize_term_dashes(text: str) -> str:
    if text.isascii():
        return text
    # Keep character offsets intact so annotations preserve the original spelling.
    return "".join(
        "-" if char == "\u2212" or unicodedata.category(char) == "Pd" else char
        for char in text
    )


def terminology_key(label: str) -> str:
    normalized = _normalize_term_dashes(label.casefold())
    return " ".join(_TERM_SEPARATOR.split(normalized)).strip()


def _label_forms(label: str) -> tuple[str, ...]:
    key = terminology_key(label)
    prefix, sep, head = key.rpartition(" ")
    if head in _REGULAR_PLURAL_HEADS:
        return key, f"{prefix}{sep}{head}s"
    if head == "helix":
        return key, f"{prefix}{sep}helices"
    return (key,)


def _entry_labels(entry: Mapping[str, Any]) -> Iterator[tuple[str, str]]:
    for field in ("term", "aliases"):
        labels = entry.get(field, [])
        for label in (labels,) if isinstance(labels, str) else labels:
            if isinstance(label, str) and label.strip():
                yield field, label


def _unit_key(label: str) -> str:
    label = _UNIT_SUPER_POWER.sub(
        lambda match: "^" + match.group().translate(_UNIT_ORDINARY),
        label.translate(_UNIT_GLYPHS),
    )
    label = _UNIT_OPERATOR.sub(
        lambda match: "^" if match[1] == "**" else match[1], " ".join(label.split())
    )
    return _BRACED_POWER.sub(r"^\1", label)


def _entry_forms(entry: Mapping[str, Any]) -> Iterator[tuple[str, str]]:
    symbols = set(entry.get("unit_symbols", ()))
    for field, label in _entry_labels(entry):
        if entry.get("kind") != "unit":
            for form in _label_forms(label):
                yield field, form
            continue
        key = _unit_key(label if label in symbols else label.casefold())
        yield field, key
        power_word, _, base = key.partition(" ")
        if base and power_word in _UNIT_POWER_WORDS:
            yield field, f"{base}^{_UNIT_POWER_WORDS[power_word]}"


def terminology_forms(entry: Mapping[str, Any]) -> set[str]:
    return {form for _, form in _entry_forms(entry)}


@dataclass(frozen=True)
class TerminologyMention:
    start: int
    end: int
    entry: Mapping[str, Any]
    field: str
    match_reason: str | None = None


@lru_cache(maxsize=32)
def _mention_pattern(names: tuple[str, ...]) -> re.Pattern[str]:
    alternatives = "|".join(
        _TERM_SEPARATOR_PATTERN.join(map(re.escape, name.split()))
        for name in names
    )
    return re.compile(rf"(?<![\w/])(?:{alternatives})(?![\w/])", re.IGNORECASE)


@lru_cache(maxsize=128)
def _unit_pattern(names: tuple[str, ...], *, case_sensitive: bool) -> re.Pattern[str]:
    def expression(name: str) -> str:
        parts = []
        for part in re.split(r"(\^[-+]?\d+|/| )", name):
            if part.startswith("^") and len(part) > 1:
                power = part[1:]
                parts.append(
                    rf"(?:[ \t]*(?:\^|\*\*)[ \t]*{re.escape(power)}"
                    rf"|[ \t]*\^[ \t]*\{{[ \t]*{re.escape(power)}[ \t]*\}}"
                    rf"|{re.escape(power.translate(_UNIT_SUPERSCRIPTS))})"
                )
            elif part == "/":
                parts.append(r"[ \t]*/[ \t]*")
            elif part == " ":
                parts.append(r"\s+")
            else:
                parts.append(re.escape(part))
        return "".join(parts)

    alternatives = "|".join(map(expression, names))
    return re.compile(rf"(?<![^\W\d])(?:{alternatives})", 0 if case_sensitive else re.IGNORECASE)


@lru_cache(maxsize=128)
def _spaced_unit_pattern(symbols: tuple[str, ...]) -> re.Pattern[str]:
    # Protect adjacent registered symbols, not ordinary words separated by spaces.
    components = {
        _UNIT_POWER.sub("", part)
        for symbol in symbols
        for part in re.split(r"[/ ]+", symbol)
    }
    alternatives = "|".join(map(re.escape, sorted(components - {""}, key=lambda x: (-len(x), x))))
    atom = rf"(?:{alternatives}){_SYMBOL_POWER_PATTERN}?"
    return re.compile(rf"(?<![^\W\d]){atom}(?:[ \t]+{atom})+(?!\w|\.[0-9]|\^)")


def _unit_boundary_ok(text: str, start: int, end: int) -> bool:
    prefix = text[:start]
    if prefix and prefix[-1].isdigit() and not _NUMBER_AT_END.search(prefix):
        return False
    return not (_UNIT_LEFT_OPERATOR.search(prefix) or _UNIT_RIGHT_OPERATOR.match(text, end))


def find_terminology_mentions(
    text: str,
    entries: Iterable[Mapping[str, Any]],
    *,
    prose_only: bool = False,
) -> Iterator[TerminologyMention]:
    """Match the longest whole term or known variant without guessing a meaning."""
    candidates: dict[str, list[tuple[Mapping[str, Any], str]]] = {}
    unit_candidates: dict[str, list[tuple[Mapping[str, Any], str]]] = {}
    symbol_candidates: dict[str, list[tuple[Mapping[str, Any], str]]] = {}
    for entry in entries:
        is_unit = entry.get("kind") == "unit"
        symbols = {_unit_key(label) for label in entry.get("unit_symbols", ())} if is_unit else set()
        forms: dict[str, str] = {}
        for field, form in _entry_forms(entry):
            forms.setdefault(form, field)
        for form, field in forms.items():
            target = symbol_candidates if form in symbols else unit_candidates if is_unit else candidates
            target.setdefault(form, []).append((entry, field))

    protected = list(_PROTECTED_TEXT.finditer(text)) if prose_only else []
    unit_text = text.translate(_UNIT_GLYPHS) if unit_candidates or symbol_candidates else text
    compounds = (
        list(_spaced_unit_pattern(tuple(sorted(symbol_candidates))).finditer(unit_text))
        if symbol_candidates else []
    )
    mentions = []
    for pool, is_unit, is_symbol in (
        (candidates, False, False), (unit_candidates, True, False), (symbol_candidates, True, True)
    ):
        if not pool:
            continue
        names = tuple(sorted(pool, key=lambda name: (-len(name), name)))
        pattern = _unit_pattern(names, case_sensitive=is_symbol) if is_unit else _mention_pattern(names)
        match_text = unit_text if is_unit else _normalize_term_dashes(text)
        for match in pattern.finditer(match_text):
            start, end = match.span()
            if any(start < span.end() and span.start() < end for span in protected):
                continue
            if is_unit:
                if not _unit_boundary_ok(unit_text, start, end) or any(
                    start < span.end() and span.start() < end
                    and not (start <= span.start() and end >= span.end())
                    for span in compounds
                ):
                    continue
                key = _unit_key(match.group() if is_symbol else match.group().casefold())
            else:
                key = terminology_key(match.group())
            matches = pool.get(key, [])
            if len(matches) == 1:
                entry, field = matches[0]
                mentions.append(TerminologyMention(start, end, entry, field))

    last_end = 0
    for mention in sorted(mentions, key=lambda item: (item.start, -item.end)):
        if mention.start >= last_end:
            yield mention
            last_end = mention.end


def _overlaps(first: TerminologyMention, second: TerminologyMention) -> bool:
    return first.start < second.end and second.start < first.end


def _prepare_user_text(text: str) -> str:
    """Expose path keys, but not URL contents, without shifting match offsets."""
    def keep_json_fields(match: re.Match[str]) -> str:
        path = match.group()
        masked = ["|"] * len(path)
        for field in _JSON_FIELD.finditer(path):
            start, end = field.span(1 if field[1] is not None else 2)
            masked[start:end] = path[start:end]
        return "".join(masked)

    text = _USER_URL.sub(lambda match: "|" * len(match.group()), text)
    text = _JSON_PATH.sub(keep_json_fields, text)
    return _PAIRED_SIDES.sub(lambda match: match.group().replace("/", "|"), text)


def find_user_terminology_mentions(
    text: str, entries: Sequence[Mapping[str, Any]]
) -> Iterator[TerminologyMention]:
    """Match exact terms and known variants in user-provided text and field names."""
    text = _prepare_user_text(text)
    exact = list(find_terminology_mentions(text, entries))
    normalized = _USER_FIELD.sub(lambda match: match.group().replace("_", " "), text)
    if normalized != text:
        normalized_matches = [
            replace(mention, match_reason="separator_normalized")
            for mention in find_terminology_mentions(normalized, entries)
            if not any(_overlaps(mention, found) for found in exact)
        ]
        exact.extend(normalized_matches)
    yield from exact


def terminology_number(number: int) -> str:
    return "\u207d" + str(number).translate(_SUPERSCRIPTS) + "\u207e"


def claim_evidence_number(number: int) -> str:
    return "[" + str(number).translate(_SUPERSCRIPTS) + "]"


def order_answer_markers(answer: str) -> str:
    return re.sub(
        rf"({_TERM_MARKER.pattern})({CLAIM_EVIDENCE_MARKER.pattern})", r"\2\1", answer
    )


def scientific_answer_body(
    answer: str | None, *, keep_claim_markers: bool = False
) -> str | None:
    """Keep prose and legacy inline evidence without rendered appendices."""
    if answer is None:
        return None
    body, marker, _ = answer.partition(TERMINOLOGY_APPENDIX_MARKER)
    if marker:
        body = _TERM_MARKER.sub("", body)
    body, evidence_marker, _ = body.partition(CLAIM_EVIDENCE_APPENDIX_MARKER)
    if evidence_marker and not keep_claim_markers:
        body = CLAIM_EVIDENCE_MARKER.sub("", body)
    return body.rstrip()


def annotate_terminology(
    answer: str, entries: Sequence[Mapping[str, Any]]
) -> tuple[str, list[Mapping[str, Any]]]:
    """Annotate the first prose mention of each unambiguous knowledge-base entry."""
    eligible = (entry for entry in entries if entry.get("uri") and entry.get("description"))
    used: dict[str, Mapping[str, Any]] = {}
    chunks: list[str] = []
    cursor = 0
    for mention in find_terminology_mentions(answer, eligible, prose_only=True):
        key = terminology_key(mention.entry["term"])
        if key in used:
            continue
        used[key] = mention.entry
        chunks.append(answer[cursor:mention.end])
        chunks.append(terminology_number(len(used)))
        cursor = mention.end
    chunks.append(answer[cursor:])
    return "".join(chunks), list(used.values())
