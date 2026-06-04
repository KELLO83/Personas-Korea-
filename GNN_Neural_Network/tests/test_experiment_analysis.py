from __future__ import annotations

from GNN_Neural_Network.gnn_recommender.experiment_analysis import (
    FeatureAblationGroup,
    alias_audit_report,
    build_feature_ablation_manifest,
    compare_metric_reports,
    segment_gap_report,
)


def test_compare_metric_reports_computes_deltas_from_baseline() -> None:
    reports = {
        "baseline": {"metrics": {"recall@10": 0.7, "ndcg@10": 0.45}},
        "phase6": {"metrics": {"recall@10": 0.71, "ndcg@10": 0.44}},
    }

    rows = compare_metric_reports(reports, "baseline", ("recall@10", "ndcg@10"))

    assert rows[0]["experiment"] == "baseline"
    assert rows[0]["recall@10_delta"] == 0.0
    assert rows[1]["experiment"] == "phase6"
    assert rows[1]["recall@10_delta"] == 0.010000000000000009
    assert rows[1]["ndcg@10_delta"] == -0.010000000000000009


def test_segment_gap_report_identifies_worst_and_gap() -> None:
    metrics = {
        "per_segment": {
            "age_group": {
                "10대": {"recall": 0.8, "count": 10},
                "30대": {"recall": 0.6, "count": 8},
            },
            "sex": {
                "남자": {"recall": 0.72, "count": 7},
                "여자": {"recall": 0.74, "count": 9},
            },
        }
    }

    rows = segment_gap_report(metrics)

    assert rows[0].dimension == "age_group"
    assert rows[0].worst_segment == "30대"
    assert rows[0].best_segment == "10대"
    assert rows[0].recall_gap == 0.20000000000000007


def test_alias_audit_report_holds_alias_candidate_with_caveats() -> None:
    report = alias_audit_report(
        experiment_id="phase6_domain_text_hard1_aliases_full_validation",
        train_status={"summary": {"text_embedding_audit_pass": True}},
        validation_metrics={"metrics": {"recall@10": 0.73}},
        test_metrics={"metrics": {"recall@10": 0.71}},
    )

    assert report["uses_alias_candidate_text"] is True
    assert report["text_embedding_audit_pass"] is True
    assert report["promotion_state"] == "hold"


def test_feature_ablation_manifest_marks_one_changed_group_per_variant() -> None:
    manifest = build_feature_ablation_manifest(
        baseline_feature_columns=("a", "b", "c"),
        groups=(
            FeatureAblationGroup(name="without_text", remove_columns=("b",)),
            FeatureAblationGroup(name="without_demo", remove_columns=("c",)),
        ),
    )

    assert manifest["baseline"]["feature_columns"] == ["a", "b", "c"]
    assert manifest["variants"][0]["changed_group"] == "without_text"
    assert manifest["variants"][0]["feature_columns"] == ["a", "c"]
    assert manifest["variants"][1]["removed_columns"] == ["c"]
