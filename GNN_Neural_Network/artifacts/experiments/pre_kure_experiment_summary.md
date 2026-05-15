# Pre-KRUE Experiment Summary

Date: 2026-05-15

Scope: experiments and closure checks completed before running the KRUE/KURE semantic similarity feature ablation. The KRUE/KURE semantic similarity feature itself was not executed in this summary.

## Current Baseline

The comparison baseline remains the closed Phase 2.5 default:

```text
Stage 1: popularity + cooccurrence
Stage 2: LightGBM learned ranker
include_source_features: false
include_text_embedding_feature: false
MMR: false
```

## Canonical/Fallback Baseline Closure

Prepare-only was refreshed with:

```powershell
.\.venv\Scripts\python.exe GNN_Neural_Network\scripts\train_lightgcn.py --prepare-only
```

Result:

```text
rare_item_policy: keep_with_fallback
raw_edges: 50000
retained_edges: 48811
rare_items_count: 7423
fallback_edges_count: 7457
dropped_edges: 0
```

Decision: canonical/fallback preparation is current for the local 50K data slice. Rare items are preserved with fallback and `dropped_edges=0`.

## Feature Balance Probe

Both runs used the 50K validation candidate pool with text embedding disabled.

| Run | Recall@10 | NDCG@10 | Delta Recall@10 vs Stage1 | Delta NDCG@10 vs Stage1 | Candidate Recall@50 | v2 Fallback | Cold-start Recall@10 | Cold-start NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| feature_fraction=0.7 | 0.680712 | 0.419497 | +0.000251 | +0.000128 | 0.993011 | 19677 | 0.687084 | 0.424718 |
| feature_fraction=0.8 | 0.680863 | 0.419572 | +0.000402 | +0.000203 | 0.993011 | 19677 | 0.687200 | 0.424785 |

Decision: no default change. `feature_fraction=0.8` is slightly better than `0.7`, but both are only marginally above Stage 1 and both have a large fallback count. These are closed as pre-KURE blockers, not promoted candidates.

Artifacts:

- `GNN_Neural_Network/artifacts/experiments/phase5_b2_feature_balance/feature_fraction_0_7/validation_metrics.json`
- `GNN_Neural_Network/artifacts/experiments/phase5_b2_feature_balance/feature_fraction_0_8/validation_metrics.json`

## Closed Phase 2.5 Cold-Start Baseline

The sparse-user slice is defined as `known_hobbies <= 1`. These metrics are for the closed Phase 2.5 default, not for a KURE/KRUE semantic feature run.

| Split | People | V2 Recall@10 | V2 NDCG@10 | V2 Coverage@10 | V2 Novelty@10 | V2 ILD@10 | Stage1 Recall@10 | Stage1 NDCG@10 | Candidate Recall@50 | V2 Fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | 8,563 | 0.592199 | 0.367798 | 0.002802 | 4.570526 | 0.967444 | 0.582389 | 0.363706 | 0.827669 | 0 |
| test | 8,563 | 0.589513 | 0.368271 | 0.002802 | 4.570526 | 0.967444 | 0.578302 | 0.364737 | 0.827208 | 0 |

The validation run used the older single-process feature builder and took about 1,687 seconds. The test run used the new process-pool-by-person feature builder with `os.cpu_count() - 2` default CPU threads (`22 -> 20`) and finished in about 344 seconds. Metric semantics are unchanged; this is an evaluation throughput change only.

Artifacts:

- `GNN_Neural_Network/artifacts/experiments/phase2_5_cold_start_baseline/validation_metrics.json`
- `GNN_Neural_Network/artifacts/experiments/phase2_5_cold_start_baseline/test_metrics.json`

## Taxonomy Risk

The taxonomy over-merge check was refreshed.

```text
total_hobbies: 49558
avg_user_top_category_ratio: 0.750987
rare_hobby_count: 49262
warning categories: 기타/다양
```

Decision: taxonomy/category balance remains a documented warning. It does not change the default model, and future KRUE/KURE coverage or novelty gains must be reviewed with this risk in mind.

Artifact:

- `GNN_Neural_Network/artifacts/experiments/phase5_taxonomy_overmerge/overmerge_report.json`

## Final Decision

Pre-KRUE blockers are closed or explicitly documented as warnings for the current handoff. No candidate replaces the closed Phase 2.5 default. The project is ready for a gated KRUE/KURE semantic similarity ablation only if `include_text_embedding_feature=true` is explicitly enabled for that run and leakage audit passes. Any future result must compare both overall metrics and the fixed cold-start baseline above before changing defaults.
