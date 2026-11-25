from __future__ import annotations

import json
import math
from itertools import combinations
from pathlib import Path
from stat import S_ISREG
from statistics import fmean
from typing import Any, Iterable

from ..data_models.analysis import (
    AnalysisArtifactReference,
    AnalysisEvidence,
    AnalysisFinding,
    EvidenceDerivation,
    InferenceLevel,
    ReportReadStatus,
    ReportSummary,
)
from ..data_models.artifact import Artifact


# 报告读取上限防止异常文件耗尽 CLI 进程内存。正常 embedding 报告远低于此值。
MAX_REPORT_BYTES = 100 * 1024 * 1024
MAX_METRICS_PER_REPORT = 60
MAX_FIELD_EVIDENCE_PER_REPORT = 8
MAX_GENERAL_REPORT_COMPARISONS = 12

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
    try:
        exists = Path(artifact.path).is_file()
    except OSError:
        exists = False
    return AnalysisArtifactReference(
        turn_id=artifact.turn_id,
        plan_id=artifact.plan_id,
        artifact_type=artifact.artifact_type,
        source_tool=artifact.source_tool,
        side=artifact.side,
        path=artifact.path,
        exists=exists,
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
    metric_paths: dict[str, str],
    metric_inference_levels: dict[str, InferenceLevel],
    prefix: str = "",
) -> None:
    """递归提取对象中的标量；数组由专门规则压缩，避免上下文爆炸。"""

    if len(output) >= MAX_METRICS_PER_REPORT:
        return
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            _flatten_scalar_mapping(
                child,
                output,
                metric_paths,
                metric_inference_levels,
                child_prefix,
            )
            if len(output) >= MAX_METRICS_PER_REPORT:
                break
        return
    if isinstance(value, list):
        return
    normalized = _normalize_scalar(value)
    if normalized is not None:
        # 末级字段名通常已经在 schema 中全局唯一；冲突时保留完整路径。
        short_key = prefix.rsplit(".", 1)[-1]
        metric_key = short_key if short_key not in output else prefix
        output[metric_key] = normalized
        metric_paths[metric_key] = f"$.{prefix}"
        metric_inference_levels[metric_key] = "direct"


def _summarize_record_array(
    records: list[Any],
    metrics: dict[str, Any],
    metric_paths: dict[str, str],
    metric_inference_levels: dict[str, InferenceLevel],
    metric_operations: dict[str, list[str]],
    prefix: str,
) -> None:
    count_key = f"{prefix}_count"
    metrics[count_key] = len(records)
    metric_paths[count_key] = f"$.{prefix}"
    metric_inference_levels[count_key] = "calculated"
    metric_operations[count_key] = ["Count entries"]
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
        source_path = f"$.{prefix}[*].{key}"
        for suffix, operation, metric_value in (
            ("mean", "Mean", round(fmean(values), 6)),
            ("min", "Minimum", round(min(values), 6)),
            ("max", "Maximum", round(max(values), 6)),
        ):
            metric_key = f"{key}_{suffix}"
            metrics[metric_key] = metric_value
            metric_paths[metric_key] = source_path
            metric_inference_levels[metric_key] = "calculated"
            metric_operations[metric_key] = [operation]


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
    metric_paths: dict[str, str] = {}
    metric_inference_levels: dict[str, InferenceLevel] = {}
    metric_operations: dict[str, list[str]] = {}
    report_type = "unknown"

    read_status: ReportReadStatus = "parsed"
    try:
        file_info = path.stat()
    except FileNotFoundError:
        read_status = "missing"
        warnings.append("The report file does not exist.")
    except OSError:
        read_status = "unreadable"
        warnings.append("The report file could not be accessed.")
    else:
        if not S_ISREG(file_info.st_mode):
            read_status = "unreadable"
            warnings.append("The report path is not a regular file.")
        elif file_info.st_size > MAX_REPORT_BYTES:
            read_status = "too_large"
            warnings.append(
                f"The report exceeds the {MAX_REPORT_BYTES}-byte safety limit and "
                "was not parsed."
            )

    if read_status == "parsed":
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
                    _flatten_scalar_mapping(
                        value,
                        metrics,
                        metric_paths,
                        metric_inference_levels,
                        key,
                    )
                elif isinstance(value, list):
                    _summarize_record_array(
                        value,
                        metrics,
                        metric_paths,
                        metric_inference_levels,
                        metric_operations,
                        key,
                    )

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
                        metric_paths["embedding_dimension"] = (
                            "$.sequence_embeddings[0]"
                        )
                        metric_inference_levels["embedding_dimension"] = (
                            "calculated"
                        )
                        metric_operations["embedding_dimension"] = ["Measure vector length"]

            # integrated_graph 是节点数组；其长度比完整图内容更适合作为摘要。
            for graph_key in (
                "integrated_graph",
                "wild_type_integrated_graph",
                "mutant_integrated_graph",
            ):
                graph = data.get(graph_key)
                if isinstance(graph, list):
                    metric_key = f"{graph_key}_node_count"
                    metrics[metric_key] = len(graph)
                    metric_paths[metric_key] = f"$.{graph_key}"
                    metric_inference_levels[metric_key] = "calculated"
                    metric_operations[metric_key] = ["Count nodes"]
        except FileNotFoundError:
            read_status = "missing"
            warnings.append("The report file does not exist.")
        except OSError:
            read_status = "unreadable"
            warnings.append("The report file could not be read.")
        except ValueError as exc:
            read_status = "unreadable"
            warnings.append(f"Report parsing failed: {exc}")

    if read_status != "parsed":
        metrics.clear()
    metrics = _prioritize_metrics(report_type, metrics)
    metric_paths = {
        key: metric_paths.get(key, "$")
        for key in metrics
    }
    metric_inference_levels = {
        key: metric_inference_levels.get(key, "direct")
        for key in metrics
    }
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
        metric_paths=metric_paths,
        metric_inference_levels=metric_inference_levels,
        metric_operations={
            key: metric_operations[key]
            for key in metrics
            if key in metric_operations and metric_inference_levels[key] == "calculated"
        },
        read_status=read_status,
        highlights=highlights,
        warnings=warnings,
    )


def unique_wt_mut_report_pairs(artifacts: Iterable[Artifact]) -> list[tuple[str, str]]:
    """Pair already selected reports only when each side has one distinct candidate."""
    grouped: dict[tuple[str, str], dict[str, set[str]]] = {}
    for artifact in artifacts:
        if artifact.side not in {"wt", "mut"} or not artifact.artifact_type.endswith("_report_json"):
            continue
        group = grouped.setdefault((artifact.source_tool, artifact.artifact_type), {})
        group.setdefault(artifact.side, set()).add(artifact.path)
    return [
        (next(iter(sides["wt"])), next(iter(sides["mut"])))
        for sides in grouped.values()
        if len(sides.get("wt", ())) == 1 and len(sides.get("mut", ())) == 1
        and sides["wt"] != sides["mut"]
    ]


def build_report_evidence(
    summaries: Iterable[ReportSummary],
    *,
    enable_general_comparison: bool = False,
    comparison_report_pairs: Iterable[tuple[str, str]] = (),
) -> tuple[list[AnalysisEvidence], list[AnalysisFinding]]:
    """为报告摘要和 WT/MUT 数值差异建立可验证证据链。"""

    evidence: list[AnalysisEvidence] = []
    findings: list[AnalysisFinding] = []
    summaries = [summary for summary in summaries if summary.read_status == "parsed"]
    report_evidence_ids: dict[int, str] = {}
    metric_evidence_ids: dict[tuple[int, str], str] = {}

    def metric_provenance(
        key: str, source_summaries: Iterable[ReportSummary]
    ) -> dict[str, list[str]]:
        sources = list(source_summaries)
        return {
            "source_metric_evidence_ids": [
                evidence_id
                for summary in sources
                if (evidence_id := metric_evidence_ids.get((id(summary), key)))
                is not None
            ],
            "source_field_paths": [
                summary.metric_paths.get(key, "$") for summary in sources
            ],
        }

    for index, summary in enumerate(summaries, start=1):
        evidence_id = f"report-{index}"
        report_evidence_ids[id(summary)] = evidence_id
        value_summary = "; ".join(summary.highlights) or "No scalar metrics extracted"
        summary_metric_evidence_ids: list[str] = []
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
                # This evidence summarizes the report as a whole. ``$`` is the
                # stable JSON-root location; calculated comparisons below retain
                # their exact scalar field path.
                field_path="$",
                value_summary=value_summary,
            )
        )
        for metric_index, (key, value) in enumerate(
            list(summary.key_metrics.items())[:MAX_FIELD_EVIDENCE_PER_REPORT],
            start=1,
        ):
            metric_evidence_id = f"{evidence_id}-metric-{metric_index}"
            summary_metric_evidence_ids.append(metric_evidence_id)
            metric_evidence_ids[(id(summary), key)] = metric_evidence_id
            inference_level = summary.metric_inference_levels.get(key, "direct")
            evidence.append(
                AnalysisEvidence(
                    evidence_id=metric_evidence_id,
                    source_kind="report",
                    inference_level=inference_level,
                    turn_id=summary.turn_id,
                    plan_id=summary.plan_id,
                    artifact_type=summary.artifact_type,
                    source_tool=summary.source_tool,
                    side=summary.side,
                    path=summary.path,
                    field_path=summary.metric_paths.get(key, "$"),
                    value_summary=f"{key}={value}",
                    source_evidence_ids=(
                        [evidence_id] if inference_level == "calculated" else []
                    ),
                    metadata={"role": "report_metric", "metric_name": key},
                    derivation=EvidenceDerivation(
                        steps=summary.metric_operations.get(key, []),
                        result_value=str(value),
                    ),
                )
            )
        findings.append(
            AnalysisFinding(
                title=f"{summary.source_tool} ({summary.side})",
                statement=value_summary,
                finding_kind="warning" if summary.warnings else "result",
                inference_level="direct",
                evidence_ids=summary_metric_evidence_ids or [evidence_id],
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
                            source_evidence_ids=[
                                report_evidence_ids[id(individual_summary)],
                                report_evidence_ids[id(integrated_summary)],
                            ],
                            metadata=metric_provenance(
                                key,
                                [individual_summary, integrated_summary],
                            ),
                            derivation=EvidenceDerivation(
                                steps=["Check whether source values agree"],
                                result_label="Consistency check",
                                result_value="Values differ",
                                source_values=[
                                    str(individual_summary.key_metrics[key]),
                                    str(integrated_summary.key_metrics[key]),
                                ],
                                source_steps=[
                                    summary.metric_operations.get(key, [])
                                    for summary in (individual_summary, integrated_summary)
                                ],
                            ),
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

    grouped: dict[tuple[int, int, str, str, str], dict[str, dict[str, ReportSummary]]] = {}
    for summary in summaries:
        if summary.side not in {"wt", "mut"}:
            continue
        scope = (
            summary.turn_id, summary.plan_id, summary.source_tool,
            summary.artifact_type, summary.report_type,
        )
        grouped.setdefault(scope, {}).setdefault(summary.side, {})[summary.path] = summary

    by_path = {summary.path: summary for summary in summaries}
    wt_mut_pairs: dict[tuple[str, str], tuple[ReportSummary, ReportSummary]] = {}
    for wt_path, mut_path in comparison_report_pairs:
        wt, mut = by_path.get(wt_path), by_path.get(mut_path)
        if wt is None or mut is None or wt.side != "wt" or mut.side != "mut":
            continue
        if (wt.source_tool, wt.artifact_type, wt.report_type) == (
            mut.source_tool, mut.artifact_type, mut.report_type
        ):
            wt_mut_pairs[wt_path, mut_path] = (wt, mut)
    explicit_paths = {path for pair in wt_mut_pairs for path in pair}
    for sides in grouped.values():
        if len(sides.get("wt", ())) != 1 or len(sides.get("mut", ())) != 1:
            continue
        wt, mut = next(iter(sides["wt"].values())), next(iter(sides["mut"].values()))
        if wt.path == mut.path or {wt.path, mut.path} & explicit_paths:
            continue
        wt_mut_pairs[wt.path, mut.path] = (wt, mut)

    comparison_index = 1
    for wt, mut in wt_mut_pairs.values():
        wt_metrics, mut_metrics = wt.key_metrics, mut.key_metrics
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
                    turn_id=mut.turn_id,
                    plan_id=mut.plan_id,
                    artifact_type=mut.artifact_type,
                    source_tool=mut.source_tool,
                    side="shared",
                    path=None,
                    field_path=key,
                    value_summary=statement,
                    source_evidence_ids=[
                        report_evidence_ids[id(wt)],
                        report_evidence_ids[id(mut)],
                    ],
                    metadata=metric_provenance(
                        key, [wt, mut]
                    ),
                    derivation=EvidenceDerivation(
                        steps=["Subtract wild type from mutant"],
                        result_label="Mutant minus wild type",
                        result_value=str(difference),
                        source_values=[str(wt_metrics[key]), str(mut_metrics[key])],
                        source_steps=[
                            summary.metric_operations.get(key, []) for summary in (wt, mut)
                        ],
                    ),
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

    if not enable_general_comparison:
        return evidence, findings

    # Conservatively compare repeated reports of the same type and side.  This
    # supports current-vs-history and cross-run analysis without equating unlike
    # scientific metrics.  Existing WT/MUT comparison behavior above is unchanged.
    grouped_repeated: dict[
        tuple[str, str, str, str], list[ReportSummary]
    ] = {}
    for summary in summaries:
        grouped_repeated.setdefault(
            (
                summary.report_type,
                summary.source_tool,
                summary.artifact_type,
                summary.side,
            ),
            [],
        ).append(summary)

    general_comparison_count = 0
    for repeated in grouped_repeated.values():
        if len(repeated) < 2:
            continue
        ordered = sorted(
            repeated,
            key=lambda item: (item.turn_id, item.plan_id, item.path),
        )
        for earlier, later in combinations(ordered, 2):
            # Multiple registered paths from one plan can be copies of the same
            # logical report. Compare distinct runs only.
            if (earlier.turn_id, earlier.plan_id) == (
                later.turn_id,
                later.plan_id,
            ):
                continue
            shared_keys = [
                key
                for key, value in earlier.key_metrics.items()
                if key in later.key_metrics
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
                and isinstance(later.key_metrics[key], (int, float))
                and not isinstance(later.key_metrics[key], bool)
            ]
            for key in shared_keys[:5]:
                difference = round(
                    float(later.key_metrics[key])
                    - float(earlier.key_metrics[key]),
                    6,
                )
                evidence_id = f"comparison-{comparison_index}"
                comparison_index += 1
                general_comparison_count += 1
                statement = (
                    f"{key}: turn {earlier.turn_id}/plan {earlier.plan_id}="
                    f"{earlier.key_metrics[key]}, turn {later.turn_id}/plan "
                    f"{later.plan_id}={later.key_metrics[key]}, "
                    f"later-earlier={difference}"
                )
                evidence.append(
                    AnalysisEvidence(
                        evidence_id=evidence_id,
                        source_kind="report",
                        inference_level="calculated",
                        turn_id=later.turn_id,
                        plan_id=later.plan_id,
                        artifact_type=later.artifact_type,
                        source_tool=later.source_tool,
                        side=later.side,
                        path=None,
                        field_path=key,
                        value_summary=statement,
                        source_evidence_ids=[
                            report_evidence_ids[id(earlier)],
                            report_evidence_ids[id(later)],
                        ],
                        metadata=metric_provenance(key, [earlier, later]),
                        derivation=EvidenceDerivation(
                            steps=["Subtract earlier value from later value"],
                            result_label="Later minus earlier",
                            result_value=str(difference),
                            source_values=[
                                str(earlier.key_metrics[key]), str(later.key_metrics[key])
                            ],
                            source_steps=[
                                summary.metric_operations.get(key, [])
                                for summary in (earlier, later)
                            ],
                        ),
                    )
                )
                findings.append(
                    AnalysisFinding(
                        title=f"Cross-run report comparison: {key}",
                        statement=statement,
                        finding_kind="comparison",
                        inference_level="calculated",
                        evidence_ids=[evidence_id],
                    )
                )
                if general_comparison_count >= MAX_GENERAL_REPORT_COMPARISONS:
                    break
            if general_comparison_count >= MAX_GENERAL_REPORT_COMPARISONS:
                break
        if general_comparison_count >= MAX_GENERAL_REPORT_COMPARISONS:
            break
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
