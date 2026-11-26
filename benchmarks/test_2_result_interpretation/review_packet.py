"""Build the same per-case materials used by the previous full review."""

from pathlib import Path

from benchmark_support import (
    ROOT,
    _answer_for_review,
    _case_key_facts,
    _case_semantic_requirements,
    _compact_report_summaries,
    _iis_dimensions,
)


def _evidence_registry(result, report_ids_by_path):
    evidence_by_id = {item.evidence_id: item for item in result.evidence}
    referenced = {
        evidence_id
        for item in [*result.findings, *(result.claim_citations or [])]
        for evidence_id in item.evidence_ids
    }
    pending = list(referenced)
    while pending:
        item = evidence_by_id.get(pending.pop())
        if item:
            for source in (*item.source_evidence_ids, *item.metadata.get("source_metric_evidence_ids", [])):
                if source not in referenced:
                    referenced.add(source)
                    pending.append(source)
    registry = []
    for evidence_id in sorted(referenced & evidence_by_id.keys()):
        item = evidence_by_id[evidence_id]
        registry.append({
            "evidence_id": evidence_id,
            "report_id": report_ids_by_path.get(str(Path(item.path).resolve())) if item.path else None,
            "source_kind": item.source_kind,
            "inference_level": item.inference_level,
            "source_tool": item.source_tool,
            "artifact_type": item.artifact_type,
            "side": item.side,
            "turn_id": item.turn_id,
            "field_path": item.field_path,
            "value_summary": item.value_summary,
            "source_evidence_ids": item.source_evidence_ids,
            "source_field_paths": item.metadata.get("source_field_paths", []),
            "source_metric_evidence_ids": item.metadata.get("source_metric_evidence_ids", []),
        })
    return registry


def build_review_packet(catalog, case, result, initial=None):
    report_ids_by_path = {
        str((ROOT / spec["path"]).resolve()): report_id for report_id, spec in catalog["reports"].items()
    }
    required = [
        {
            "fact_id": fact["fact_id"],
            "report_id": report_id,
            "metric_name": fact["metric_name"],
            "expected_value": fact["expected_value"],
            "field_path": fact["field_path"],
            "accepted_aliases": fact.get("accepted_aliases", []),
            "supporting_report_ids": fact.get("supporting_report_ids", [report_id]),
        }
        for report_id, fact in _case_key_facts(catalog, case)
    ]
    required.extend(
        {**fact, "accepted_aliases": []} for fact in _case_semantic_requirements(catalog, case)
    )
    packet = {
        "case_id": case["case_id"],
        "category_id": case["category_id"],
        "question": case.get("follow_up_question") or case["question"],
        "initial_question": case.get("initial_question"),
        "initial_answer": _answer_for_review(initial) if initial is not None else None,
        "candidate_answer": _answer_for_review(result),
        "candidate_limitations": result.limitations,
        "expected_behavior": case.get("expected_behavior"),
        "iis_requirements": {
            "must_address": case.get("required_facts", []),
            "required_limitations": case.get("required_limitations", []),
            "forbidden_claims": case.get("forbidden_claims", []),
            "optional_metric_policy": (
                "Report metrics are available evidence, not an exhaustive narrative "
                "checklist. Omission is not penalized unless required by the question "
                "or must_address entries."
            ),
        },
        "report_summaries": _compact_report_summaries(result, report_ids_by_path),
        "iis_dimensions": _iis_dimensions(case),
        "objective_metrics_applicable": True,
        "kfr_key_facts": required,
        "candidate_evidence_registry": _evidence_registry(result, report_ids_by_path),
    }
    if result.claim_citations is not None:
        packet["claim_citations"] = [citation.model_dump(mode="json") for citation in result.claim_citations]
    return packet
