# Phase 5-C Text Embedding Ablation Summary

Date: 2026-05-05

Run: `kure_text_feature_001`

## Decision

Status: `disabled`

The KURE-v1 text embedding feature run was stopped before LightGBM training because the post-mask leakage audit failed above the configured threshold.

```text
threshold: 0.05
failure_rate: 0.989843
passed_person_count: 202
failed_person_count: 19686
```

No validation metric comparison was performed and no test run was executed.

## Baseline

The default remains the closed Phase 2.5 path:

```text
Stage 1: popularity + cooccurrence
Stage 2: LightGBM learned ranker
include_text_embedding_feature: false
MMR: false
```

Baseline validation reference:

```text
Recall@10: 0.7390509094604207
NDCG@10: 0.45797028878684237
Coverage@10: 0.15555555555555556
Novelty@10: 4.584286633989583
Candidate Recall@50: 0.9776445483182603
```

## Artifacts

- `GNN_Neural_Network/artifacts/experiments/phase5_c_text_embedding/kure_text_feature_001/text_leakage_audit.json`
- `GNN_Neural_Network/artifacts/experiments/phase5_c_text_embedding/kure_text_feature_001/ranker_params.json`
- `GNN_Neural_Network/artifacts/experiments/phase5_c_text_embedding/kure_text_feature_001/ranker_train.status.json`
- `GNN_Neural_Network/artifacts/experiments/phase5_c_text_embedding/kure_text_feature_001/validation_metrics.status.json`

## Next Step

Improve split-aware masking and canonical alias coverage, or redesign the text split, before re-running `include_text_embedding_feature=true`.
