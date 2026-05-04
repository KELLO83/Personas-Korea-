# Phase 2.5 Source One-Hot Ablation Summary

## Status

- Date: 2026-05-01
- Experiment: `Phase2.5_Source_OneHot_Ablation_Single_Config`
- Execution policy: single-config train/eval, validation-only selection, no test unless validation beats current default
- Final decision: `rejected`
- Default path after experiment: keep `include_source_features=false`

## What Was Tested

The experiment tested whether explicit candidate-source indicators help the LightGBM ranker balance Stage 1 provider signals.

Added features:

- `source_is_popularity`
- `source_is_cooccurrence`
- `source_count`

Fixed settings inherited from the current default:

- `num_leaves=31`
- `min_data_in_leaf=50`
- `learning_rate=0.05`
- `reg_alpha=0.1`
- `reg_lambda=0.1`
- `neg_ratio=4`
- `hard_ratio=0.8`
- `include_text_embedding_feature=false`
- Stage 1 providers: `popularity + cooccurrence`
- `candidate_k=50`

## Validation Result

| Config | Recall@10 | NDCG@10 | coverage@10 | novelty@10 | Decision |
|:---|---:|---:|---:|---:|:---|
| Current default, no source features | 0.739051 | 0.457970 | 0.155556 | 4.584287 | baseline |
| Source one-hot enabled | 0.737933 | 0.457141 | 0.138889 | 4.584875 | rejected |

Delta versus current default:

- `Recall@10=-0.001117772584087`
- `NDCG@10=-0.0008293137054019`
- `coverage@10=-0.0166666666666667`
- `novelty@10=+0.000588524901453`

## Feature Importance Observation

The trained model assigned non-zero gain to `source_count`, but not to individual source flags.

Observed from `ranker_feature_importance.json` / training output:

- `source_count`: non-zero gain
- `source_is_popularity`: zero gain
- `source_is_cooccurrence`: zero gain

Interpretation:

- The current merged candidate pool already preserves provider information through normalized `source_scores`.
- Explicit source flags did not improve ranking under the current `popularity + cooccurrence` provider set.
- Validation coverage dropped, so this feature policy does not help the ranking collapse issue.

## Decision

Do not promote source one-hot features.

Reason:

- Validation `Recall@10` and `NDCG@10` are both below the current default.
- `coverage@10` also regressed from `0.155556` to `0.138889`.
- The tiny novelty improvement is not enough to justify lower accuracy and lower coverage.
- Per split discipline, test evaluation was skipped because validation did not beat the default.

Selected default after this experiment:

```text
num_leaves=31
neg_ratio=4
hard_ratio=0.8
include_source_features=false
include_text_embedding_feature=false
```

## Artifacts

- `artifacts/experiments/phase2_5_source_onehot/ranker_model.txt`
- `artifacts/experiments/phase2_5_source_onehot/ranker_params.json`
- `artifacts/experiments/phase2_5_source_onehot/ranker_feature_importance.json`
- `artifacts/experiments/phase2_5_source_onehot/validation_metrics.json`
- `artifacts/experiments/phase2_5_source_onehot/validation_metrics.status.json`

## Next Step

Phase 2.5 default ranking path remains unchanged. The next meaningful experiments are:

- KURE dense embedding MMR re-evaluation
- leakage-safe text embedding feature ablation
- final default-path closure for `num_leaves=31`, `neg_ratio=4`, `hard_ratio=0.8`, `include_source_features=false`
