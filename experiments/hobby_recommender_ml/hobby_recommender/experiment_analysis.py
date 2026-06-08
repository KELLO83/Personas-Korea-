from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | Mapping[str, "JsonValue"] | Sequence["JsonValue"]
MetricRow: TypeAlias = dict[str, str | float]
AuditValue: TypeAlias = str | bool | list[str]


@dataclass(frozen=True, slots=True)
class SegmentGap:
    dimension: str
    worst_segment: str
    best_segment: str
    worst_recall: float
    best_recall: float
    recall_gap: float
    worst_count: int
    best_count: int


@dataclass(frozen=True, slots=True)
class FeatureAblationGroup:
    name: str
    remove_columns: tuple[str, ...]


def compare_metric_reports(
    reports: Mapping[str, Mapping[str, JsonValue]],
    baseline: str,
    metrics: Sequence[str],
) -> list[MetricRow]:
    baseline_metrics = _metric_mapping(reports[baseline])
    rows: list[MetricRow] = []
    for name, report in reports.items():
        report_metrics = _metric_mapping(report)
        row: MetricRow = {"experiment": name}
        for metric_name in metrics:
            value = _float(report_metrics.get(metric_name))
            base_value = _float(baseline_metrics.get(metric_name))
            row[metric_name] = value
            row[f"{metric_name}_delta"] = value - base_value
        rows.append(row)
    return rows


def segment_gap_report(metrics: Mapping[str, JsonValue]) -> list[SegmentGap]:
    per_segment = metrics.get("per_segment")
    if not isinstance(per_segment, Mapping):
        return []

    rows: list[SegmentGap] = []
    for dimension, raw_segments in per_segment.items():
        if not isinstance(dimension, str) or not isinstance(raw_segments, Mapping):
            continue
        segment_rows = _segment_rows(raw_segments)
        if len(segment_rows) < 2:
            continue
        worst = min(segment_rows, key=lambda item: item[1])
        best = max(segment_rows, key=lambda item: item[1])
        rows.append(
            SegmentGap(
                dimension=dimension,
                worst_segment=worst[0],
                best_segment=best[0],
                worst_recall=worst[1],
                best_recall=best[1],
                recall_gap=best[1] - worst[1],
                worst_count=worst[2],
                best_count=best[2],
            )
        )
    return sorted(rows, key=lambda item: item.recall_gap, reverse=True)


def alias_audit_report(
    experiment_id: str,
    train_status: Mapping[str, JsonValue],
    validation_metrics: Mapping[str, JsonValue],
    test_metrics: Mapping[str, JsonValue] | None = None,
) -> dict[str, AuditValue]:
    summary = train_status.get("summary")
    summary_map = summary if isinstance(summary, Mapping) else {}
    uses_alias = "alias" in experiment_id.lower()
    audit_pass = bool(summary_map.get("text_embedding_audit_pass"))
    caveats: list[str] = []
    if uses_alias:
        caveats.append("alias_candidate_text_requires_provenance_review")
    if not audit_pass:
        caveats.append("text_embedding_audit_missing_or_failed")
    if test_metrics is None:
        caveats.append("test_metrics_missing")
    validation_recall = _recall_at_10(validation_metrics)
    test_recall = _recall_at_10(test_metrics or {})
    if validation_recall > 0.0 and test_recall > 0.0 and test_recall < validation_recall:
        caveats.append("test_recall_below_validation_recall")
    promotion_state = "hold" if caveats else "candidate"
    return {
        "experiment_id": experiment_id,
        "uses_alias_candidate_text": uses_alias,
        "text_embedding_audit_pass": audit_pass,
        "promotion_state": promotion_state,
        "caveats": caveats,
    }


def build_feature_ablation_manifest(
    baseline_feature_columns: Sequence[str],
    groups: Sequence[FeatureAblationGroup],
) -> dict[str, JsonValue]:
    baseline = list(baseline_feature_columns)
    variants: list[dict[str, JsonValue]] = []
    for group in groups:
        removed = set(group.remove_columns)
        variants.append(
            {
                "changed_group": group.name,
                "removed_columns": list(group.remove_columns),
                "feature_columns": [column for column in baseline if column not in removed],
            }
        )
    return {
        "baseline": {"feature_columns": baseline},
        "policy": "one_feature_group_removed_per_variant",
        "variants": variants,
    }


def _metric_mapping(report: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    raw_metrics = report.get("metrics")
    return raw_metrics if isinstance(raw_metrics, Mapping) else report


def _segment_rows(raw_segments: Mapping[str, JsonValue]) -> list[tuple[str, float, int]]:
    rows: list[tuple[str, float, int]] = []
    for segment_name, raw_payload in raw_segments.items():
        if not isinstance(segment_name, str) or not isinstance(raw_payload, Mapping):
            continue
        if segment_name.endswith("_gap"):
            continue
        recall = _float(raw_payload.get("recall"))
        count = int(_float(raw_payload.get("count")))
        rows.append((segment_name, recall, count))
    return rows


def _recall_at_10(report: Mapping[str, JsonValue]) -> float:
    metrics = _metric_mapping(report)
    return _float(metrics.get("recall@10"))


def _float(value: JsonValue | object) -> float:
    if isinstance(value, int | float):
        return float(value)
    return 0.0
