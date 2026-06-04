from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import TypeAlias

MetricRow: TypeAlias = dict[str, str | float]
ReviewRow: TypeAlias = Mapping[str, str | tuple[str, ...]]


def classify_failure_modes(row: Mapping[str, str]) -> tuple[str, ...]:
    shared_hobby_count = _int(row.get("shared_hobby_count", "0"))
    shared_skill_count = _int(row.get("shared_skill_count", "0"))
    explanation_count = _int(row.get("explanation_feature_count", "0"))
    has_shared_reason = shared_hobby_count > 0 or shared_skill_count > 0
    labels: list[str] = []
    if explanation_count <= 2:
        labels.append("low_information")
    if _flag(row, "same_occupation") and not has_shared_reason and not _has_location_match(row):
        labels.append("occupation_overfit")
    elif _is_location_only(row) and not has_shared_reason:
        labels.append("location_overfit")
    elif not has_shared_reason and _has_demographic_match(row):
        labels.append("demographic_only")
    return tuple(labels or ["unclassified"])


def summarize_failure_taxonomy(rows: Sequence[ReviewRow]) -> dict[str, int | dict[str, int]]:
    counts: Counter[str] = Counter()
    for row in rows:
        raw_modes = row.get("failure_modes", ())
        if isinstance(raw_modes, tuple):
            counts.update(raw_modes)
        elif isinstance(raw_modes, str):
            counts.update(mode.strip() for mode in raw_modes.split("|") if mode.strip())
    return {
        "row_count": len(rows),
        "mode_counts": dict(sorted(counts.items())),
    }


def compare_experiment_metrics(
    reports: Mapping[str, Mapping[str, Mapping[str, float]]],
    baseline: str,
    metrics: Sequence[str],
) -> list[MetricRow]:
    baseline_metrics = reports[baseline]["metrics"]
    rows: list[MetricRow] = []
    for name, report in reports.items():
        row: MetricRow = {"experiment": name}
        report_metrics = report["metrics"]
        for metric_name in metrics:
            value = float(report_metrics.get(metric_name, 0.0))
            baseline_value = float(baseline_metrics.get(metric_name, 0.0))
            row[metric_name] = value
            row[f"{metric_name}_delta"] = value - baseline_value
        rows.append(row)
    return rows


def _has_demographic_match(row: Mapping[str, str]) -> bool:
    return any(
        _flag(row, column)
        for column in (
            "same_occupation",
            "same_province",
            "same_district",
            "same_age_group",
            "same_education",
            "same_field",
        )
    )


def _has_location_match(row: Mapping[str, str]) -> bool:
    return _flag(row, "same_province") or _flag(row, "same_district")


def _is_location_only(row: Mapping[str, str]) -> bool:
    return _has_location_match(row) and not any(
        _flag(row, column)
        for column in ("same_occupation", "same_age_group", "same_education", "same_field")
    )


def _flag(row: Mapping[str, str], column: str) -> bool:
    return _int(row.get(column, "0")) > 0


def _int(value: str) -> int:
    try:
        return int(float(value))
    except ValueError:
        return 0
