"""Claude Haiku 5.5 structured review for both Test 2 metrics."""

from __future__ import annotations

import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, ValidationError

from evidence_matching import evidence_index, match_citations

MODEL = "claude-haiku-5-5"
BASE_URL = "https://api.anthropic.com"
API_VERSION = "2026-10-21"
MAX_OUTPUT_TOKENS = 16384
STANDARD_INPUT_LIMIT = 100_000

PROMPT = """Review one English scientific-result interpretation case. Return the
item-level judgments for both metrics in one response. The program scores them.

Claims and required answers
- Review the final answer and concrete results in its visible evidence section.
  The earlier answer is context for follow-ups. It earns no credit by itself.
- List distinct independently verifiable, case-specific scientific conclusions.
  Keep a comparison and the values necessary to express that comparison together.
  A reported score, ligand identity, and named complex are separate claims
  if all three are stated, even when one sentence or evidence block contains
  them. Likewise, different contact types or different metrics are separate.
  A WT/MUT comparison of one metric remains a single claim. Do not repeat
  the component values of that comparison as additional claims.
- Exclude headings, advice, general definitions, and generic cautions about
  prediction, validation, or interpretation. Statements about what information
  was supplied should not be transformed into scientific conclusions about the
  underlying report. If a task explicitly asks for such a distinction, judge
  whether it was addressed under the corresponding required fact.
- Report supported for each extracted claim using the fixed facts or supplied
  report evidence. Preserve the object, side, value, unit, and qualifications.
  For claims that require scientific verification, set requires_citation=true
  regardless of whether a citation was supplied or the claim is true.
- Return one judgment for each supplied kfr_key_facts fact_id. Accept equivalent
  scientific wording and any one allowed alternative for a single requirement.
  Coverage judges the answer content. Evaluate required citations separately.
  For a follow-up correction, independently check the earlier answer for any
  targeted errors. Recognize the difference between a lack of named residues
  and an available aggregate range for per-residue results.

Sources
- Use the stable evidence IDs in evidence_keys; all IDs refer to the supplied
  reports, statistics, comparison records, or artifact evidence. Use exact IDs,
  including for different statistics that share the same raw field path.
- For each claim that requires citation, select reference_evidence_ids for
  the independently needed evidence. Select them based on the claim and fixed
  materials, independently of what the answer cited. Use an empty list if no
  supplied source supports the claim. Prefer a matching metric ID to a vague
  report ID when the claim states a specific metric. Wrong sides or fields do
  not support the claim.
- For an allowed derived comparison, list the corresponding side-specific
  evidence IDs as references. In citations, keep the comparison ID the answer
  actually used. The program follows its existing source links. Set
  allow_derived=true when verified traceability satisfies the question's source
  requirement; set it false when the question explicitly requires direct
  citations. Set allow_report_level=true only when a report-level citation
  satisfies the required granularity. A question asking for the exact field
  needs a field-level citation.
- For citations, preserve each distinct actual link in the same claim, starting
  with claim_citations and any visible evidence section. Return separate
  citation entries for different explicit IDs, even inside one bracket. One citation can
  appear in the prose with an ID and again in the evidence appendix with a
  readable label. Return it only once, using the ID. A prose statement
  explicitly attributing results to a named report is a report-level citation
  even without a numbered marker. Use that report's ID and preserve the
  actual report name used. Retain the WT/MUT qualifier when names are shared.
  Cite a comparison as one original link. Use its exact source ID. For an explicit but invalid or
  unresolvable link, preserve the visible description and set source_id=null.
  Do not add links the answer did not cite. A general introduction naming the
  available reports does not cite each later claim. Count a prose attribution
  only when it directly links a specific conclusion to a source.
- If no citation is required, use empty lists for both citations and
  reference_evidence_ids. Do not assign reference IDs to omitted claims.
"""


class ReviewModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class FactualClaim(ReviewModel):
    claim: str
    supported: bool
    reason: str


class Citation(ReviewModel):
    actual_source: str
    source_id: str | None


class CitationReview(ReviewModel):
    requires_citation: bool
    allow_derived: bool
    allow_report_level: bool
    reference_evidence_ids: list[str]
    citations: list[Citation]


class Claim(FactualClaim, CitationReview):
    pass


class RequiredClaim(ReviewModel):
    fact_id: str
    covered: bool
    reason: str


class Review(ReviewModel):
    claims: list[Claim]
    required_claims: list[RequiredClaim]


def _post(endpoint, api_key, payload, timeout):
    request = Request(
        BASE_URL + endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "anthropic-version": API_VERSION,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as error:
        with error:
            try:
                details = json.load(error).get("error", {})
            except ValueError:
                details = {}
        message = details.get("message", "Request failed.")
        raise RuntimeError(f"Claude HTTP {error.code}: {message}") from None


def normalize_review(packet, raw, index):
    if not isinstance(raw, dict) or not isinstance(raw.get("claims"), list):
        raise ValueError("Claude did not return a claim list")
    fact_fields = FactualClaim.model_fields
    citation_fields = CitationReview.model_fields
    claims = []
    citation_errors = []
    for n, item in enumerate(raw["claims"], 1):
        if not isinstance(item, dict):
            raise ValueError(f"Claim {n} is not an object")
        factual = FactualClaim.model_validate({key: item[key] for key in fact_fields if key in item}).model_dump()
        if not factual["claim"].strip():
            raise ValueError("Claude returned an empty claim")
        try:
            cit = CitationReview.model_validate({key: item[key] for key in citation_fields if key in item}).model_dump()
            if not cit["requires_citation"] and (cit["citations"] or cit["reference_evidence_ids"]):
                raise ValueError("A claim marked exempt contains citation judgments")
            claims.append(match_citations({**factual, **cit}, index))
        except (ValidationError, ValueError) as error:
            citation_errors.append(f"Claim {n}: {error}")
            claims.append({**factual, "requires_citation": None, "citations": [], "reference_evidence": []})

    required_claims = [RequiredClaim.model_validate(item).model_dump() for item in raw.get("required_claims", [])]
    required = [item["fact_id"] for item in packet["kfr_key_facts"]]
    if sorted(item["fact_id"] for item in required_claims) != sorted(required):
        raise ValueError("Claude must judge each supplied required claim exactly once")
    return {"claims": claims, "required_claims": required_claims, "citation_errors": citation_errors}


def review_case(packet, api_key, timeout, *, save_raw=None):
    index = evidence_index(packet)
    augmented = {**packet, "evidence_keys": index.selectable}
    response = _post("/v1/messages", api_key, {
        "model": MODEL,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "system": PROMPT,
        "messages": [{"role": "user", "content": json.dumps(augmented, ensure_ascii=False)}],
        "output_config": {
            "effort": "medium",
            "format": {"type": "json_schema", "schema": Review.model_json_schema()},
        },
    }, timeout)
    text = "".join(block["text"] for block in response["content"] if block["type"] == "text")
    usage = response.get("usage", {})
    if save_raw:
        save_raw({"id": response.get("id"), "stop_reason": response.get("stop_reason"),
                  "usage": usage, "text": text})
    if response.get("stop_reason") != "end_turn":
        raise ValueError(f"Claude review ended with stop reason {response.get('stop_reason')!r}")
    return normalize_review(packet, json.loads(text), index), usage
