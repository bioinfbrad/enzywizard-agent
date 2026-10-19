from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

from ..data_models.analysis import (
    AnalysisArtifactReference,
    AnalysisEvidence,
    AnalysisFinding,
    ReportSummary,
)
from ..data_models.artifact import Artifact


# 报告读取上限防止异常文件耗尽 CLI 进程内存。正常 embedding 报告远低于此值。
MAX_REPORT_BYTES = 100 * 1024 * 1024
MAX_METRICS_PER_REPORT = 60

REPORT_ARTIFACT_TYPES = {
    "clean_report_json",
    "mut_clean_report_json",
    "aaprops_report_json",
    "hydrocluster_report_json",
    "energy_report_json",
    "flexibility_report_json",
    "disorder_report_json",
    "conservation_report_json",
    "embedding_report_json",
    "pocket_report_json",
    "substrate_report_json",
    "dock_report_json",
    "interaction_report_json",
    "integrate_report_json",
    "mut_integrate_report_json",
}

REPORT_PRIORITY_FIELDS: dict[str, tuple[str, ...]] = {
    "enzywizard_clean": (
        "original_residue_count",
        "cleaned_residue_count",
        "removed_residue_count",
    ),
    "enzywizard_energy": ("total_potential_energy",),
    "enzywizard_flexibility": (
        "residue_root_mean_square_fluctuation_mean",
        "residue_root_mean_square_fluctuation_max",
    ),
    "enzywizard_disorder": (
        "disordered_region_count",
        "max_disordered_region_length",
        "total_disordered_region_length",
    ),
    "enzywizard_pocket": (
        "binding_pocket_count",
        "max_binding_pocket_volume",
        "total_binding_pocket_volume",
    ),
    "enzywizard_dock": ("enzyme_substrate_binding_affinity",),
    "enzywizard_interaction": (
        "hydrogen_bond_count",
        "ionic_bond_count",
        "van_der_waals_contact_count",
        "pi_pi_stacking_count",
        "pi_cation_interaction_count",
        "disulfide_bond_count",
    ),
    "enzywizard_integrate": (
        "sequence_length",
        "total_potential_energy",
        "binding_pocket_count",
        "enzyme_substrate_binding_affinity",
        "hydrogen_bond_count",
    ),
    "enzywizard_mut_integrate": (
        "difference_total_potential_energy",
        "difference_binding_pocket_count",
        "difference_max_binding_pocket_volume",
        "difference_residue_root_mean_square_fluctuation",
    ),
}


def artifact_to_reference(artifact: Artifact) -> AnalysisArtifactReference:
    return AnalysisArtifactReference(
        turn_id=artifact.turn_id,
        plan_id=artifact.plan_id,
        artifact_type=artifact.artifact_type,
        source_tool=artifact.source_tool,
        side=artifact.side,
        path=artifact.path,
        exists=Path(artifact.path).is_file(),
    )


def select_report_artifacts(artifacts: Iterable[Artifact]) -> list[Artifact]:
    """仅选择正式 JSON 报告，不把 nodes/edges、stdout 或日志当成报告。"""

    selected: list[Artifact] = []
    seen_paths: set[str] = set()
    seen_shared_reports: set[tuple[int, int, str, str]] = set()
    for artifact in artifacts:
        if artifact.artifact_type not in REPORT_ARTIFACT_TYPES:
            continue
        if artifact.path in seen_paths:
            continue
        shared_key = (
            artifact.turn_id,
            artifact.plan_id,
            artifact.source_tool,
            artifact.artifact_type,
        )
        # mut-integrate 可能在 WT/MUT 输出目录各写一份内容相同的 shared 报告；
        # Analysis 只需保留一个逻辑报告，但仍在全局 artifacts 中追踪两个文件。
        if artifact.side == "shared" and shared_key in seen_shared_reports:
            continue
        seen_paths.add(artifact.path)
        if artifact.side == "shared":
            seen_shared_reports.add(shared_key)
        selected.append(artifact)
    return selected


def _normalize_scalar(value: Any) -> str | int | float | bool | None:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return round(value, 6)
    return None


def _flatten_scalar_mapping(
    value: Any,
    output: dict[str, Any],
    prefix: str = "",
) -> None:
    """递归提取对象中的标量；数组由专门规则压缩，避免上下文爆炸。"""

    if len(output) >= MAX_METRICS_PER_REPORT:
        return
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            _flatten_scalar_mapping(child, output, child_prefix)
            if len(output) >= MAX_METRICS_PER_REPORT:
                break
        return
    if isinstance(value, list):
        return
    normalized = _normalize_scalar(value)
    if normalized is not None:
        # 末级字段名通常已经在 schema 中全局唯一；冲突时保留完整路径。
        short_key = prefix.rsplit(".", 1)[-1]
        output[short_key if short_key not in output else prefix] = normalized


def _summarize_record_array(
    records: list[Any],
    metrics: dict[str, Any],
    prefix: str,
) -> None:
    metrics[f"{prefix}_count"] = len(records)
    dict_records = [item for item in records if isinstance(item, dict)]
    if not dict_records:
        return

    numeric_keys: list[str] = []
    for key in dict_records[0]:
        if key.endswith("index") or key.endswith("count"):
            continue
        values = [item.get(key) for item in dict_records]
        if values and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            numeric_keys.append(key)

    for key in numeric_keys[:6]:
        values = [float(item[key]) for item in dict_records]
        metrics[f"{key}_mean"] = round(fmean(values), 6)
        metrics[f"{key}_min"] = round(min(values), 6)
        metrics[f"{key}_max"] = round(max(values), 6)


def _prioritize_metrics(report_type: str, metrics: dict[str, Any]) -> dict[str, Any]:
    priority = REPORT_PRIORITY_FIELDS.get(report_type, ())
    ordered: dict[str, Any] = {}
    for requested_key in priority:
        for key, value in metrics.items():
            if key == requested_key or key.endswith(f".{requested_key}"):
                ordered[key] = value
    for key, value in metrics.items():
        if key not in ordered:
            ordered[key] = value
        if len(ordered) >= MAX_METRICS_PER_REPORT:
            break
    return ordered


def summarize_report_artifact(artifact: Artifact) -> ReportSummary:
    """读取并压缩一个工具报告；格式异常会转成 warning 而不是静默忽略。"""

    path = Path(artifact.path)
    warnings: list[str] = []
    metrics: dict[str, Any] = {}
    report_type = "unknown"

    if not path.is_file():
        warnings.append("The report file does not exist.")
    elif path.stat().st_size > MAX_REPORT_BYTES:
        warnings.append(
            f"The report exceeds the {MAX_REPORT_BYTES}-byte safety limit and "
            "was not parsed."
        )
    else:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("report root must be a JSON object")
            report_type = str(data.get("report_type") or "unknown")

            # 所有 schema 的统计对象和能量项都可安全压缩为标量字典。
            for key, value in data.items():
                if key == "report_type":
                    continue
                if isinstance(value, dict):
                    _flatten_scalar_mapping(value, metrics, key)
                elif isinstance(value, list):
                    _summarize_record_array(value, metrics, key)

            # embedding 的向量维度具有意义，但绝不把向量本身送入模型。
            embeddings = data.get("sequence_embeddings")
            if isinstance(embeddings, list) and embeddings:
                first = embeddings[0]
                if isinstance(first, dict):
                    vector = next(
                        (value for value in first.values() if isinstance(value, list)),
                        None,
                    )
                    if vector is not None:
                        metrics["embedding_dimension"] = len(vector)

            # integrated_graph 是节点数组；其长度比完整图内容更适合作为摘要。
            for graph_key in (
                "integrated_graph",
                "wild_type_integrated_graph",
                "mutant_integrated_graph",
            ):
                graph = data.get(graph_key)
                if isinstance(graph, list):
                    metrics[f"{graph_key}_node_count"] = len(graph)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            warnings.append(f"Report parsing failed: {exc}")

    metrics = _prioritize_metrics(report_type, metrics)
    highlights = [
        f"{key}={value}"
        for key, value in list(metrics.items())[:8]
    ]
    if not highlights and not warnings:
        warnings.append("The report contains no scalar metrics to summarize.")

    return ReportSummary(
        turn_id=artifact.turn_id,
        plan_id=artifact.plan_id,
        artifact_type=artifact.artifact_type,
        report_type=report_type,
        source_tool=artifact.source_tool,
        side=artifact.side,
        path=artifact.path,
        key_metrics=metrics,
        highlights=highlights,
        warnings=warnings,
    )


def build_report_evidence(
    summaries: Iterable[ReportSummary],
) -> tuple[list[AnalysisEvidence], list[AnalysisFinding]]:
    """为报告摘要和 WT/MUT 数值差异建立可验证证据链。"""

    evidence: list[AnalysisEvidence] = []
    findings: list[AnalysisFinding] = []
    summaries = list(summaries)

    for index, summary in enumerate(summaries, start=1):
        evidence_id = f"report-{index}"
        value_summary = "; ".join(summary.highlights) or "No scalar metrics extracted"
        evidence.append(
            AnalysisEvidence(
                evidence_id=evidence_id,
                source_kind="report",
                inference_level="direct",
                turn_id=summary.turn_id,
                plan_id=summary.plan_id,
                artifact_type=summary.artifact_type,
                source_tool=summary.source_tool,
                side=summary.side,
                path=summary.path,
                field_path=None,
                value_summary=value_summary,
            )
        )
        findings.append(
            AnalysisFinding(
                title=f"{summary.source_tool} ({summary.side})",
                statement=value_summary,
                finding_kind="warning" if summary.warnings else "result",
                inference_level="direct",
                evidence_ids=[evidence_id],
            )
        )

    # 对同一轮、同一计划、同一侧的单蛋白整合报告执行保守一致性检查。
    # 只比较完全相同的数值字段名，不推测别名，也不把 MUT-WT 差值误当原值。
    consistency_index = 1
    grouped_by_scope: dict[
        tuple[int, int, str], list[ReportSummary]
    ] = {}
    for summary in summaries:
        grouped_by_scope.setdefault(
            (summary.turn_id, summary.plan_id, summary.side), []
        ).append(summary)

    for scoped_summaries in grouped_by_scope.values():
        integrated = [
            item
            for item in scoped_summaries
            if item.report_type == "enzywizard_integrate"
            or item.source_tool == "enzywizard-integrate"
        ]
        individual = [
            item
            for item in scoped_summaries
            if item not in integrated
            and item.report_type != "enzywizard_mut_integrate"
            and item.source_tool != "enzywizard-mut-integrate"
        ]
        for integrated_summary in integrated:
            for individual_summary in individual:
                shared_numeric_keys = [
                    key
                    for key, value in individual_summary.key_metrics.items()
                    if key in integrated_summary.key_metrics
                    and isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and isinstance(integrated_summary.key_metrics[key], (int, float))
                    and not isinstance(integrated_summary.key_metrics[key], bool)
                ]
                for key in shared_numeric_keys:
                    individual_value = float(individual_summary.key_metrics[key])
                    integrated_value = float(integrated_summary.key_metrics[key])
                    if math.isclose(
                        individual_value,
                        integrated_value,
                        rel_tol=1e-6,
                        abs_tol=1e-6,
                    ):
                        continue
                    evidence_id = f"consistency-{consistency_index}"
                    consistency_index += 1
                    statement = (
                        f"{key} differs between {individual_summary.source_tool} "
                        f"({individual_summary.key_metrics[key]}) and "
                        f"{integrated_summary.source_tool} "
                        f"({integrated_summary.key_metrics[key]})."
                    )
                    evidence.append(
                        AnalysisEvidence(
                            evidence_id=evidence_id,
                            source_kind="report",
                            inference_level="calculated",
                            turn_id=integrated_summary.turn_id,
                            plan_id=integrated_summary.plan_id,
                            artifact_type=integrated_summary.artifact_type,
                            source_tool=integrated_summary.source_tool,
                            side=integrated_summary.side,
                            path=integrated_summary.path,
                            field_path=key,
                            value_summary=statement,
                        )
                    )
                    findings.append(
                        AnalysisFinding(
                            title=f"Cross-report inconsistency: {key}",
                            statement=statement,
                            finding_kind="warning",
                            inference_level="calculated",
                            evidence_ids=[evidence_id],
                        )
                    )
                    if consistency_index > 10:
                        break
                if consistency_index > 10:
                    break
            if consistency_index > 10:
                break
        if consistency_index > 10:
            break

    # 仅比较同一工具、同一产物类型且分别属于 WT/MUT 的报告。
    grouped: dict[tuple[str, str], dict[str, ReportSummary]] = {}
    for summary in summaries:
        if summary.side not in {"wt", "mut"}:
            continue
        grouped.setdefault((summary.source_tool, summary.artifact_type), {})[
            summary.side
        ] = summary

    comparison_index = 1
    for (source_tool, artifact_type), pair in grouped.items():
        if set(pair) != {"wt", "mut"}:
            continue
        wt_metrics = pair["wt"].key_metrics
        mut_metrics = pair["mut"].key_metrics
        shared_keys = [
            key
            for key in wt_metrics
            if key in mut_metrics
            and isinstance(wt_metrics[key], (int, float))
            and not isinstance(wt_metrics[key], bool)
            and isinstance(mut_metrics[key], (int, float))
            and not isinstance(mut_metrics[key], bool)
        ]
        for key in shared_keys[:5]:
            difference = round(float(mut_metrics[key]) - float(wt_metrics[key]), 6)
            evidence_id = f"comparison-{comparison_index}"
            comparison_index += 1
            statement = (
                f"{key}: WT={wt_metrics[key]}, MUT={mut_metrics[key]}, "
                f"MUT-WT={difference}"
            )
            evidence.append(
                AnalysisEvidence(
                    evidence_id=evidence_id,
                    source_kind="report",
                    inference_level="calculated",
                    turn_id=pair["mut"].turn_id,
                    plan_id=pair["mut"].plan_id,
                    artifact_type=artifact_type,
                    source_tool=source_tool,
                    side="shared",
                    path=None,
                    field_path=key,
                    value_summary=statement,
                )
            )
            findings.append(
                AnalysisFinding(
                    title=f"WT/MUT comparison: {key}",
                    statement=statement,
                    finding_kind="comparison",
                    inference_level="calculated",
                    evidence_ids=[evidence_id],
                )
            )
    return evidence, findings


def compact_history_summary(items: Iterable[Any], limit: int = 12) -> str:
    """生成供后续轮次使用的有界历史摘要。"""

    values = list(items)[-limit:]
    lines: list[str] = []
    for item in values:
        turn_id = getattr(item, "turn_id", "?")
        summary = getattr(item, "summary", None)
        if summary:
            lines.append(f"Turn {turn_id}: {summary}")
    return "\n".join(lines)
