"""Match cited sources to stable, side-specific evidence units."""

from collections import defaultdict
import re
from typing import NamedTuple

from enzywizard_agent.utils.terminology_utils import CLAIM_EVIDENCE_APPENDIX_MARKER


class EvidenceIndex(NamedTuple):
    units: dict
    aliases: dict
    derived: dict
    selectable: dict
    visible: set
    rendered: dict
    report_granularity: dict
    body: str


def _visible_token(value, text, *, marker=False):
    prefix = r"(?:\bID:\s*|\[)" if marker else r"(?<![\w-])"
    return re.search(prefix + re.escape(value) + r"(?![\w-])", text) is not None


def _display_text(text):
    return " ".join(text.replace("**", "").replace("`", "").split())


def _appendix_sources(appendix, spans):
    # The Agent renderer writes one numbered line per structured span, in
    # evidence-ID order. Skip lines that cannot be mapped one-to-one.
    by_number = {span["number"]: span["evidence_ids"] for span in spans if "number" in span}
    descriptions = defaultdict(set)
    for line in appendix.splitlines():
        match = re.fullmatch(r"\s*\[(\d+)\]\s+(.+)", line)
        if not match:
            continue
        number, description = match.groups()
        ids = by_number.get(int(number), [])
        parts = description.split("; ")
        if not ids or len(ids) != len(parts):
            continue
        for source_id, part in zip(ids, parts, strict=True):
            display = _display_text(part)
            descriptions[display].add(source_id)
            descriptions[f"[{number}] {display}"].add(source_id)
    return {display: next(iter(ids)) for display, ids in descriptions.items() if len(ids) == 1}


def _report_labels(report):
    tool = report.get("source_tool", "")
    labels = {report["artifact_name"]}
    if tool:
        labels |= {f"{tool} report", f'{tool.removeprefix("enzywizard-")} report'}
    return labels


def _visible_report_labels(reports, answer):
    by_label = defaultdict(list)
    for report in reports:
        for label in _report_labels(report):
            by_label[label].append(report)
    visible = {}
    sides = {"wt": ("WT", "wild type", "wild-type"), "mut": ("MUT", "mutant")}
    for label, candidates in by_label.items():
        if len(candidates) == 1 and _visible_token(label, answer):
            visible[label] = candidates[0]["report_id"]
        for report in candidates:
            for side in sides.get(report["side"], ()):
                for scoped in (f"{side} {label}", f"{label} ({side})"):
                    if _visible_token(scoped, answer):
                        visible[scoped] = report["report_id"]
    return visible


_FIELD_GRANULARITY = re.compile(r"\b(?:fields?|coordinates?)\b", re.IGNORECASE)
_REPORT_ALTERNATIVE = re.compile(r"\breports? or (?:[^.]*? )?fields?\b", re.IGNORECASE)
_REPORT_SOURCE = re.compile(r"\breports?\b", re.IGNORECASE)


def _report_granularity(packet, report_ids):
    """Use written source requirements for their named reports."""
    question = packet.get("question", "")
    if _FIELD_GRANULARITY.search(question) and not _REPORT_ALTERNATIVE.search(question):
        return dict.fromkeys(report_ids, False)

    policy = {}
    for fact in packet.get("kfr_key_facts", []):
        rule = fact.get("evidence_requirement", "")
        if not rule:
            continue
        if _REPORT_ALTERNATIVE.search(rule):
            allow = True
        elif _FIELD_GRANULARITY.search(rule):
            allow = False
        elif _REPORT_SOURCE.search(rule):
            allow = True
        else:
            continue
        for report_id in fact.get("supporting_report_ids", []):
            policy[report_id] = policy.get(report_id, True) and allow
    return policy


def evidence_index(packet):
    units = {}
    aliases = {}
    derived = {}
    selectable = {}
    reports = packet.get("report_summaries", [])
    for report in reports:
        report_id, side = report["report_id"], report["side"]
        units[report_id] = (report_id, side, "$", "")
        selectable[report_id] = f'{report_id} ({side}): ' + ", ".join(sorted(_report_labels(report)))
        for metric, value in report["key_metrics"].items():
            key = f"{report_id}::{metric}"
            path = report["metric_paths"].get(metric)
            units[key] = (report_id, side, path, metric)
            selectable[key] = f"{path}: {value}"

    registry = {item["evidence_id"]: item for item in packet.get("candidate_evidence_registry", [])}
    by_identity = {identity: key for key, identity in units.items()}
    spans = packet.get("claim_citations") or []
    cited = {key for span in spans for key in span["evidence_ids"]}
    visible_answer = packet.get("candidate_answer", "")
    body, _, appendix = visible_answer.partition(CLAIM_EVIDENCE_APPENDIX_MARKER)
    rendered = _appendix_sources(appendix, spans)
    body = _display_text(body)
    report_labels = _visible_report_labels(reports, body)
    rendered.update(report_labels)
    for key, item in registry.items():
        report_id, side, path = item["report_id"], item["side"], item["field_path"]
        parents = item.get("source_metric_evidence_ids", [])
        if item["inference_level"] == "calculated" and parents:
            derived[key] = (tuple(parents), path)
            selectable[key] = "Calculated from " + ", ".join(parents)
            continue
        if report_id is None:
            if item["source_kind"] == "artifact":
                units[key] = (None, side, path, key)
                selectable[key] = item["value_summary"]
            continue
        metric, separator, _ = item["value_summary"].partition("=")
        identity = (report_id, side, path, metric if separator and path != "$" else "")
        canonical = by_identity.get(identity)
        if canonical is None:
            canonical = key
            units[key] = identity
            by_identity[identity] = key
        aliases[key] = canonical
        selectable[key] = canonical

    direct_visible = cited | set(report_labels.values())
    for report in reports:
        key = report["report_id"]
        if _visible_token(key, visible_answer):
            direct_visible.add(key)
        for metric, path in report["metric_paths"].items():
            if (key in direct_visible and path and path in visible_answer
                and list(report["metric_paths"].values()).count(path) == 1):
                direct_visible.add(f"{key}::{metric}")
    for key in registry:
        if _visible_token(key, visible_answer, marker=True):
            direct_visible.add(key)
    direct_visible.update(aliases.get(key, key) for key in tuple(direct_visible))
    report_granularity = _report_granularity(packet, [report["report_id"] for report in reports])
    return EvidenceIndex(units, aliases, derived, selectable, direct_visible, rendered, report_granularity, body)


def _resolve(key, units, aliases, derived, seen=frozenset()):
    canonical = aliases.get(key, key)
    if canonical in units:
        return {canonical}
    if key not in derived or key in seen:
        return set()
    parents, statistic = derived[key]
    leaves = [_resolve(parent, units, aliases, derived, seen | {key}) for parent in parents]
    if not all(leaves):
        return set()
    resolved = set.union(*leaves)
    if {units[leaf][1] for leaf in resolved} != {"wt", "mut"}:
        return set()
    if any(units[leaf][3] != statistic for leaf in resolved):
        return set()
    return resolved


def _citation_links(links, body):
    for link in links:
        displayed = _display_text(link["actual_source"])
        parts = [part.removeprefix("Evidence: ") for part in displayed.split("; ")]
        if len(parts) > 1 and f"Evidence: {'; Evidence: '.join(parts)}" in body:
            markers = [re.search(r"\(ID:\s*([\w-]+)\)$", part) for part in parts]
            if all(markers) and link["source_id"] in {marker[1] for marker in markers}:
                for part, marker in zip(parts, markers, strict=True):
                    yield {"actual_source": part, "source_id": marker[1]}
                continue
        yield link


def match_citations(claim, index):
    units, aliases, derived, selectable, visible, rendered, report_granularity, body = index
    ids = claim["reference_evidence_ids"] if claim["supported"] else []
    if any(key not in selectable for key in ids):
        raise ValueError("Unknown reference evidence ID")
    reference = set()
    for key in ids:
        resolved = _resolve(key, units, aliases, derived)
        if not resolved:
            raise ValueError("Reference evidence ID has no resolvable source")
        reference.update(resolved)

    # A single citation may resolve to several atomic evidence units. Count
    # equivalent direct and derived citations only once for each unit.
    paired_reference = {units[key][1] for key in reference} >= {"wt", "mut"}
    matched = {}
    unmatched = {}
    for link in _citation_links(claim["citations"], body):
        source, displayed = link["source_id"], link["actual_source"]
        if not displayed.strip():
            raise ValueError("Empty actual citation")
        source = rendered.get(_display_text(displayed), source)
        actual = _resolve(source, units, aliases, derived) if source in visible else set()
        if source in derived and not claim["allow_derived"]:
            actual = set()
        same = {key for key in actual & reference if units[key][2] != "$"}
        for unit in actual:
            report, side, path, _ = units[unit]
            written = report_granularity.get(report)
            if written is not True and not claim["allow_report_level"]:
                continue
            if path == "$":
                # An attribution claim may have a report as its reference even
                # when the question also requires fields for other claims.
                same.update(
                    ref for ref in reference if units[ref][:2] == (report, side)
                    and (units[ref][2] == "$" or written is not False)
                )
            elif report in reference:
                same.add(report)
        if same:
            for ref in sorted(same):
                matched.setdefault(ref, {"actual_source": displayed, "source_id": source})
            if paired_reference and source in derived:
                for unit in sorted(actual - same):
                    unmatched.setdefault((unit,), {"actual_source": displayed, "source_id": unit})
        else:
            identity = tuple(sorted(actual)) or (source if source in visible else displayed.strip(),)
            unmatched.setdefault(identity, {"actual_source": displayed, "source_id": source})

    # A broad prose attribution adds no separate citation when this claim
    # already cites an exact field from the same report and side.
    matched_reports = {units[key][:2] for key in matched if units[key][2] != "$"}
    for identity, link in list(unmatched.items()):
        source = link["source_id"]
        unit = units.get(aliases.get(source, source))
        if (
            unit and unit[2] == "$" and unit[:2] in matched_reports
            and re.search(r"\breports?\b|\.json\b", link["actual_source"], re.IGNORECASE)
            and not re.search(r"\bID:|\[\d+\]|\$", link["actual_source"])
        ):
            del unmatched[identity]

    refs = sorted(reference)
    return {
        **{key: claim[key] for key in ("claim", "supported", "reason", "requires_citation")},
        "reference_evidence": [
            {"id": key, "requirement": "Supports this claim", "source": key,
             "location": selectable[key]} for key in refs
        ],
        "citations": [
            {**matched[key], "matched_reference_index": i, "reason": f"Verified source: {key}"}
            for i, key in enumerate(refs, 1) if key in matched
        ] + [
            {**link, "matched_reference_index": None, "reason": "No matching evidence unit"}
            for link in unmatched.values()
        ],
    }
