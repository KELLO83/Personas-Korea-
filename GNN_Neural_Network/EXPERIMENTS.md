# GNN Experiment Summary

This document summarizes the offline recommender experiments under `GNN_Neural_Network/artifacts/experiments`.
It is written as a human-readable decision record for developers who need to understand what was tested, what worked, what failed, and what the current baseline is.

Last updated: 2026-05-05

## Current Default

The current default recommender path is the closed Phase 2.5 accuracy-oriented baseline.

```text
Stage 1 = popularity + cooccurrence
Stage 2 = LightGBM learned ranker
MMR     = false
```

Default model artifact:

```text
GNN_Neural_Network/artifacts/experiments/phase2_5_num_leaves_31/ranker_model.txt
```

Default LightGBM configuration:

```text
num_leaves=31
min_data_in_leaf=50
learning_rate=0.05
reg_alpha=0.1
reg_lambda=0.1
neg_ratio=4
hard_ratio=0.8
include_source_features=false
include_text_embedding_feature=false
```

## Final Default Performance

| Split | Path | Recall@10 | NDCG@10 | Coverage@10 | Novelty@10 | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| validation | Stage 1 `popularity + cooccurrence` | 0.694035 | 0.435455 | 0.127778 | 4.483649 | baseline |
| validation | v1 deterministic reranker | 0.709887 | 0.442340 | 0.516667 | 4.732133 | fallback / comparison |
| validation | Phase 2.5 LightGBM default | 0.739051 | 0.457970 | 0.155556 | 4.584287 | selected default |
| test | Stage 1 `popularity + cooccurrence` | 0.690885 | 0.437556 | 0.127778 | 4.483649 | baseline |
| test | v1 deterministic reranker | 0.704298 | 0.440329 | 0.516667 | 4.732133 | fallback / comparison |
| test | Phase 2.5 LightGBM default | 0.709684 | 0.447713 | 0.155556 | 4.584287 | selected default |

Interpretation:

- LightGBM improves Recall@10 and NDCG@10 over both Stage 1 and the v1 deterministic reranker.
- v1 deterministic reranker still has much better catalog coverage and novelty.
- The current default is the best accuracy-oriented path, not the best diversity-oriented path.
- CandidateRecall@50 is about 0.977, so the main unresolved problem is ranking collapse, not candidate retrieval failure.

## Experiment Timeline

| Order | Experiment | What was tested | Main result | Decision |
| ---: | --- | --- | --- | --- |
| 1 | Stage 1 provider selection | Popularity, cooccurrence, LightGCN merge, segment popularity, BM25/PMI/IDF/Jaccard variants | `popularity + cooccurrence` was the most stable default candidate generator | accepted |
| 2 | LightGBM learned ranker | Learned Stage 2 reranker over Stage 1 candidates | Improved Recall/NDCG over Stage 1 and v1 deterministic reranker | promoted |
| 3 | LightGBM regularization | Regularization and tree-size tuning | `num_leaves=31` passed validation and test gates | accepted |
| 4 | Negative sampling ablation | `neg_ratio` and `hard_ratio` variations | `hard_ratio=1.0` won validation but lost on final test versus current default | rejected default change |
| 5 | Source one-hot ablation | Explicit source flags: popularity/cooccurrence/source_count | Lower Recall/NDCG/Coverage than default | rejected |
| 6 | Phase 2.5 closure | Fixed comparison baseline for future experiments | Current default locked as the reference baseline | closed |
| 7 | KURE dense MMR sweep | MMR with KURE dense embeddings, lambda 0.5/0.7/0.8/0.9 | All candidates failed accuracy gates | no-go |
| 8 | LambdaRank smoke | Listwise objective smoke tests | Diversity improved slightly but Recall/NDCG dropped heavily | blocked |
| 9 | DPP diversity rerank | DPP-style diversity reranking | Novelty improved greatly, but Recall/NDCG collapsed | blocked |
| 10 | Feature balance / 50K probes | Feature-fraction and small probe datasets | Probe-only results, not valid default candidates | non-default |
| 11 | Taxonomy over-merge check | Category/canonicalization quality check | Found over-merge and category concentration risk | data risk |
| 12 | Text embedding leakage check | Leakage-safe text embedding pre-check | Warning state; additional validation needed before text features | blocked / needs follow-up |

## Phase 2.5 Negative Sampling Ablation

Baseline before this experiment:

```text
num_leaves=31
neg_ratio=4
hard_ratio=0.8
include_source_features=false
include_text_embedding_feature=false
```

Validation results:

| Config | Recall@10 | NDCG@10 | Coverage@10 | Novelty@10 | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `neg_ratio=1`, `hard_ratio=0.8` | 0.693934 | 0.435216 | 0.155556 | 4.484836 | rejected |
| `neg_ratio=2`, `hard_ratio=0.8` | 0.726857 | 0.451182 | 0.138889 | 4.568721 | rejected |
| `neg_ratio=4`, `hard_ratio=0.8` | 0.739051 | 0.457970 | 0.155556 | 4.584287 | baseline |
| `neg_ratio=8`, `hard_ratio=0.8` | 0.737323 | 0.457867 | 0.155556 | 4.584538 | rejected |
| `neg_ratio=4`, `hard_ratio=0.5` | 0.730210 | 0.452869 | 0.138889 | 4.588810 | rejected |
| `neg_ratio=4`, `hard_ratio=1.0` | 0.742404 | 0.458620 | 0.155556 | 4.600143 | selected for test |

Final test check for `neg_ratio=4`, `hard_ratio=1.0`:

| Split | Recall@10 | NDCG@10 | Coverage@10 | Novelty@10 | Delta vs default Recall@10 | Delta vs default NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 0.708973 | 0.446676 | 0.155556 | 4.600143 | -0.000711 | -0.001036 |

Decision:

- Do not change the default negative sampling policy.
- `hard_ratio=1.0` was better on validation but worse on final test.
- The small novelty gain does not justify lower Recall/NDCG.
- Keep `neg_ratio=4`, `hard_ratio=0.8`.

Artifacts:

```text
artifacts/experiments/phase2_5_neg_ratio_1/
artifacts/experiments/phase2_5_neg_ratio_2/
artifacts/experiments/phase2_5_neg_ratio_8/
artifacts/experiments/phase2_5_neg_ratio_4_hard_0_5/
artifacts/experiments/phase2_5_neg_ratio_4_hard_1_0/
artifacts/experiments/phase2_5_negative_sampling_summary.md
```

## Phase 2.5 Source One-Hot Ablation

This experiment tested whether explicit candidate-source indicators improve ranking.

Added features:

```text
source_is_popularity
source_is_cooccurrence
source_count
```

Validation results:

| Config | Recall@10 | NDCG@10 | Coverage@10 | Novelty@10 | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Current default, source off | 0.739051 | 0.457970 | 0.155556 | 4.584287 | baseline |
| Source one-hot enabled | 0.737933 | 0.457141 | 0.138889 | 4.584875 | rejected |

Decision:

- Do not promote source one-hot features.
- `source_count` had non-zero gain, but individual source flags had no useful gain under the current candidate pool.
- Recall, NDCG, and Coverage all regressed.
- Keep `include_source_features=false`.

Artifact:

```text
artifacts/experiments/phase2_5_source_onehot/
artifacts/experiments/phase2_5_source_onehot_summary.md
```

## Phase 5-A KURE Dense MMR Sweep

This experiment re-tested MMR diversity reranking with dense KURE embeddings.
The comparison baseline was the closed Phase 2.5 default.

Baseline validation reference:

| Metric | Value |
| --- | ---: |
| Recall@10 | 0.739051 |
| NDCG@10 | 0.457970 |
| Coverage@10 | 0.155556 |
| Novelty@10 | 4.584287 |
| CandidateRecall@50 | 0.977645 |

Validation sweeps:

| Lambda | Recall@10 | NDCG@10 | Coverage@10 | Novelty@10 | CandidateRecall@50 | Gate result |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.5 | 0.702571 | 0.444050 | 0.172222 | 4.600135 | 0.977645 | blocked |
| 0.7 | 0.723097 | 0.452158 | 0.144444 | 4.574652 | 0.977645 | blocked |
| 0.8 | 0.729397 | 0.454478 | 0.144444 | 4.572330 | 0.977645 | blocked |
| 0.9 | 0.728889 | 0.454644 | 0.150000 | 4.574799 | 0.977645 | blocked |

Decision:

- Final status: `NO-GO`.
- No lambda passed the accuracy gate.
- `lambda=0.5` improved Coverage and Novelty slightly, but Recall/NDCG loss was too large.
- Default remains `MMR=false`.

Artifacts:

```text
artifacts/experiments/phase5_kure_mmr_lambda_0.5/
artifacts/experiments/phase5_kure_mmr_lambda_0.7/
artifacts/experiments/phase5_kure_mmr_lambda_0.8/
artifacts/experiments/phase5_kure_mmr_lambda_0.9/
artifacts/experiments/phase5_kure_mmr_summary.md
```

## Phase 5 Additional Experiments

These experiments attempted to improve diversity or rebalance ranker behavior after the Phase 2.5 default was closed.

| Experiment | Split | Recall@10 | NDCG@10 | Coverage@10 | Novelty@10 | Decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| LambdaRank smoke f07 | validation | 0.708363 | 0.442592 | 0.188889 | 4.627144 | blocked |
| LambdaRank smoke f08 | validation | 0.708363 | 0.442592 | 0.188889 | 4.627144 | blocked |
| DPP candidate_k50 | validation | 0.518138 | 0.348219 | 0.183333 | 6.267843 | blocked |
| KURE MMR lambda 0.9 | test | 0.709278 | 0.447621 | 0.155556 | 4.577922 | blocked |

Interpretation:

- LambdaRank smoke tests improved diversity slightly but lost too much accuracy.
- DPP reranking improved novelty strongly but collapsed Recall/NDCG.
- KURE MMR lambda 0.9 test was close to default accuracy, but still not better and did not improve diversity enough.
- None of these should change the default path.

## Probe-Only Experiments

Some artifacts are probes or simulations, not full default-candidate evaluations.
Do not compare them directly with the main Phase 2.5 metrics.

| Experiment | Metric | Value | Interpretation |
| --- | --- | ---: | --- |
| `phase5_pre_50k_baseline` | validation_auc | 0.499992 | Not a useful ranker result |
| `phase5_pre_50k_baseline` | simulated_recall@10 | 0.000000 | Not a default candidate |
| `feature_fraction_0.8` | validation_auc | 0.999596 | Probe result; ranking recall is 0 |
| `feature_fraction_0.85` | validation_auc | 0.999596 | Probe result; ranking recall is 0 |
| `probe_lgbm_balance` | validation_auc | 0.952474 | Feature-balance probe only |

## Data Quality And Safety Findings

### Taxonomy Over-Merge

The taxonomy over-merge report indicates category concentration risk.
A large portion of canonical hobbies is concentrated into a broad miscellaneous/general category.
This can distort both diversity metrics and qualitative recommendations.

Artifact:

```text
artifacts/experiments/phase5_taxonomy_overmerge/
```

### Text Embedding Leakage

The text embedding leakage check produced a warning.
Text embedding features should not be promoted until leakage-safe validation passes.

Observed warning indicators:

```text
sample size: 5000 edges
train/validation common text count: 13
text leakage ratio: 0.0152
TF-IDF average cosine similarity: 0.2668
TF-IDF max cosine similarity: 1.0000
```

Artifact:

```text
artifacts/experiments/phase5_text_embedding_leakage/
```

## Accepted Components

| Component | Role | Status |
| --- | --- | --- |
| `popularity` | Stage 1 provider | accepted |
| `cooccurrence` | Stage 1 provider | accepted |
| LightGBM learned ranker | Stage 2 ranker | promoted |
| `num_leaves=31` regularized config | LightGBM default | accepted |
| v1 deterministic reranker | fallback/comparison | retained |

## Rejected Or Non-Default Components

| Component | Status | Reason |
| --- | --- | --- |
| `segment_popularity` | disabled | Degraded Recall/NDCG |
| BM25 / PMI / IDF / Jaccard / pop-capped Stage 1 variants | not selected | Did not beat selected Stage 1 baseline |
| LightGCN merge into Stage 1 | not selected | Lower validation recall than selected baseline |
| Category one-hot MMR | no-go | Binary similarity made MMR ineffective |
| KURE dense MMR | no-go | Accuracy gate failed for all lambdas |
| Negative sampling change to `hard_ratio=1.0` | rejected | Validation won, but final test lost against current default |
| Source one-hot features | rejected | Recall/NDCG/Coverage regressed |
| LambdaRank smoke | blocked | Accuracy below default |
| DPP diversity rerank | blocked | Recall/NDCG collapsed |
| Text embedding feature | blocked / needs follow-up | Leakage warning remains unresolved |

## Known Limitations

- Ranking collapse remains unresolved.
- CandidateRecall@50 is high, so candidate generation is not the primary bottleneck.
- The learned ranker still concentrates top-k recommendations around popular/cooccurring hobbies.
- Coverage@10 remains far below the v1 deterministic reranker.
- Novelty@10 remains below the v1 deterministic reranker.
- Feature importance remains dominated by `cooccurrence_score` and `popularity_prior`.
- Taxonomy over-merge can distort diversity and recommendation quality.
- Text embedding features require leakage-safe validation before promotion.

## Rules For Future Experiments

Use the closed Phase 2.5 default as the baseline unless a newer default decision is explicitly recorded.

Future experiment policy:

- Select candidates on validation only.
- Run test only once for the selected validation winner.
- Do not promote a candidate that lowers Recall/NDCG unless the experiment explicitly defines and passes a different product gate.
- Record metrics in `artifacts/experiments/<experiment_id>/validation_metrics.json`.
- Record gated decisions in `validation_metrics.status.json` when applicable.
- Preserve all metrics, status files, params, feature importance, summaries, model weights, and ranker datasets.
- Only generated caches are safe cleanup targets without explicit confirmation.

## Artifact Index

Core decision artifacts:

```text
artifacts/experiment_decisions.json
artifacts/experiment_run_summary.md
artifacts/experiments/phase2_5_default_decision_closure.md
```

Main experiment artifacts:

```text
artifacts/experiments/phase2_5_num_leaves_31/
artifacts/experiments/phase2_5_neg_ratio_*/
artifacts/experiments/phase2_5_source_onehot/
artifacts/experiments/phase5_kure_mmr_lambda_*/
artifacts/experiments/phase5_b1_listwise/
artifacts/experiments/phase5_b3_diversity_rerank/
artifacts/experiments/phase5_pre_50k_baseline/
artifacts/experiments/phase5_taxonomy_overmerge/
artifacts/experiments/phase5_text_embedding_leakage/
```

Recovery and cleanup records:

```text
artifacts/experiments/cleanup_removed_generated_files.json
artifacts/experiments/cleanup_recovery_report.md
artifacts/experiments/deleted_artifact_recovery_plan.json
artifacts/experiments/recovery_completion_report.md
artifacts/experiments/RECOVER_DELETED_EXPERIMENTS.md
scripts/recover_deleted_experiment_artifacts.py
```
