# Phase 2.5 Negative Sampling Ablation Summary

## Status

- Date: 2026-05-01
- Experiment: `Phase2.5_Negative_Sampling_Ablation_Single_Config`
- Execution policy: single-config train/eval runs, validation selection, one final test check
- Final decision: `rejected_default_change`
- Default path after experiment: keep `num_leaves=31`, `neg_ratio=4`, `hard_ratio=0.8`

## Baseline

Current default before this ablation:

- Stage 1: `popularity + cooccurrence`
- Stage 2: LightGBM binary classifier
- `num_leaves=31`
- `min_data_in_leaf=50`
- `learning_rate=0.05`
- `reg_alpha=0.1`
- `reg_lambda=0.1`
- `neg_ratio=4`
- `hard_ratio=0.8`
- `include_source_features=false`
- `include_text_embedding_feature=false`

Baseline validation:

- `Recall@10=0.7390509094604207`
- `NDCG@10=0.45797028878684237`
- `coverage@10=0.15555555555555556`
- `novelty@10=4.584286633989583`

Baseline test:

- `Recall@10=0.7096839752057718`
- `NDCG@10=0.447712669317698`
- `coverage@10=0.15555555555555556`
- `novelty@10=4.584286633989583`

## What Was Tested

The ablation tested whether changing the negative sampling ratio or hard/easy negative mix improves the currently promoted LightGBM reranker.

Negative sampling definitions:

- `neg_ratio`: number of sampled negatives per positive training row.
- `hard_ratio`: fraction of negatives drawn from the Stage 1 candidate pool.
- Hard negatives: Stage 1 candidates that are not held-out positives and not train-known hobbies.
- Easy negatives: random canonical hobbies outside positives, train-known hobbies, and the Stage 1 candidate pool.

Validation selection rule:

- Primary metric: validation `Recall@10`
- Tie-breaker: validation `NDCG@10`
- Guardrail metrics: `coverage@10`, `novelty@10`, `candidate_recall@50`
- Test split is used only once for the selected validation candidate.

## Validation Results

| Config | Recall@10 | NDCG@10 | coverage@10 | novelty@10 | Decision |
|:---|---:|---:|---:|---:|:---|
| `neg_ratio=1`, `hard_ratio=0.8` | 0.693934 | 0.435216 | 0.155556 | 4.484836 | rejected |
| `neg_ratio=2`, `hard_ratio=0.8` | 0.726857 | 0.451182 | 0.138889 | 4.568721 | rejected |
| `neg_ratio=4`, `hard_ratio=0.8` | 0.739051 | 0.457970 | 0.155556 | 4.584287 | baseline |
| `neg_ratio=8`, `hard_ratio=0.8` | 0.737323 | 0.457867 | 0.155556 | 4.584538 | rejected |
| `neg_ratio=4`, `hard_ratio=0.5` | 0.730210 | 0.452869 | 0.138889 | 4.588810 | rejected |
| `neg_ratio=4`, `hard_ratio=1.0` | 0.742404 | 0.458620 | 0.155556 | 4.600143 | selected for test |

Validation outcome:

- `neg_ratio=4`, `hard_ratio=1.0` produced the best validation `Recall@10` and `NDCG@10`.
- The validation improvement over the current default was small but real: `Recall@10 +0.003353`, `NDCG@10 +0.000650`.
- `hard_ratio=1.0` also improved validation novelty versus current default: `4.600143` vs `4.584287`.

## Final Test Check

Selected validation candidate:

- `neg_ratio=4`
- `hard_ratio=1.0`

Test result:

- `Recall@10=0.7089726653795346`
- `NDCG@10=0.44667618759401445`
- `coverage@10=0.15555555555555556`
- `novelty@10=4.600143`

Delta versus current default test result:

- `Recall@10=-0.0007113098262372`
- `NDCG@10=-0.0010364817236835`
- `coverage@10=0.0`
- `novelty@10=+0.015856366010417`

Gate versus v1 deterministic reranker:

- `Recall@10 delta=+0.004674321715272822`
- `NDCG@10 delta=+0.006347231753099258`
- Gate status: `promoted` versus v1

## Decision

Do not change the default negative sampling policy.

Reason:

- `hard_ratio=1.0` won validation, but the single final test check underperformed the existing `hard_ratio=0.8` default on both `Recall@10` and `NDCG@10`.
- The novelty gain is not enough to justify lowering the primary accuracy metrics.
- Ranking collapse remains unresolved because `coverage@10` did not improve.

Selected default after this experiment:

```text
num_leaves=31
neg_ratio=4
hard_ratio=0.8
include_source_features=false
include_text_embedding_feature=false
```

## Artifacts

- `artifacts/experiments/phase2_5_neg_ratio_1/ranker_model.txt`
- `artifacts/experiments/phase2_5_neg_ratio_1/validation_metrics.json`
- `artifacts/experiments/phase2_5_neg_ratio_2/ranker_model.txt`
- `artifacts/experiments/phase2_5_neg_ratio_2/validation_metrics.json`
- `artifacts/experiments/phase2_5_neg_ratio_8/ranker_model.txt`
- `artifacts/experiments/phase2_5_neg_ratio_8/validation_metrics.json`
- `artifacts/experiments/phase2_5_neg_ratio_4_hard_0_5/ranker_model.txt`
- `artifacts/experiments/phase2_5_neg_ratio_4_hard_0_5/validation_metrics.json`
- `artifacts/experiments/phase2_5_neg_ratio_4_hard_1_0/ranker_model.txt`
- `artifacts/experiments/phase2_5_neg_ratio_4_hard_1_0/validation_metrics.json`
- `artifacts/experiments/phase2_5_neg_ratio_4_hard_1_0/test_metrics.json`

## Next Step

Run source one-hot ablation next:

```text
train_ranker.py --include-source-features
```

Goal:

- Test whether explicit source indicators help the learned ranker balance `popularity` and `cooccurrence` signals.
- Keep validation/test split discipline: validation-only selection, one final test check for the selected setting.
