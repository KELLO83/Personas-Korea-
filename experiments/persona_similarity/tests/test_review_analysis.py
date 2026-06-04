from __future__ import annotations

from experiments.persona_similarity.scripts.review_analysis import (
    classify_failure_modes,
    compare_experiment_metrics,
    summarize_failure_taxonomy,
)


def test_classify_failure_modes_marks_low_information_pair() -> None:
    row = {
        "same_occupation": "0",
        "same_province": "1",
        "same_district": "0",
        "same_age_group": "1",
        "shared_hobby_count": "0",
        "shared_skill_count": "0",
        "explanation_feature_count": "2",
    }

    labels = classify_failure_modes(row)

    assert "low_information" in labels
    assert "demographic_only" in labels


def test_classify_failure_modes_marks_occupation_overfit() -> None:
    row = {
        "same_occupation": "1",
        "same_province": "0",
        "same_district": "0",
        "same_age_group": "0",
        "shared_hobby_count": "0",
        "shared_skill_count": "0",
        "explanation_feature_count": "4",
    }

    assert classify_failure_modes(row) == ("occupation_overfit",)


def test_summarize_failure_taxonomy_counts_multiple_labels() -> None:
    rows = [
        {"failure_modes": ("low_information", "demographic_only")},
        {"failure_modes": ("low_information",)},
    ]

    summary = summarize_failure_taxonomy(rows)

    assert summary["row_count"] == 2
    assert summary["mode_counts"]["low_information"] == 2
    assert summary["mode_counts"]["demographic_only"] == 1


def test_compare_experiment_metrics_reports_text_feature_delta() -> None:
    rows = compare_experiment_metrics(
        {
            "structured": {"metrics": {"ndcg@10": 0.99, "strong_reason_coverage@10": 0.88}},
            "structured_text": {"metrics": {"ndcg@10": 0.98, "strong_reason_coverage@10": 0.90}},
        },
        baseline="structured",
        metrics=("ndcg@10", "strong_reason_coverage@10"),
    )

    assert rows[1]["experiment"] == "structured_text"
    assert rows[1]["ndcg@10_delta"] == -0.010000000000000009
    assert rows[1]["strong_reason_coverage@10_delta"] == 0.020000000000000018
