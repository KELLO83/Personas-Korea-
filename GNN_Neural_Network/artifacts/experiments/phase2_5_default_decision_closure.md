# Phase 2.5 Default Decision Closure

## Status

- Date: 2026-05-01
- Scope: Phase 2.5 ranking-collapse mitigation and learned-ranker default selection
- Decision status: closed
- Default path status: promoted as current best accuracy-oriented default

## Why This Closure Exists

Phase 2.5 included multiple experiments that could have changed the learned-ranker default path.

- LightGBM regularization tuning
- negative sampling ablation
- source one-hot ablation
- MMR failure analysis
- logging, cache, and status artifact governance

Before starting the next experiment family, the current best default must be fixed as the comparison baseline. This prevents later experiments from moving the baseline implicitly or using inconsistent validation/test claims.

## Final Phase 2.5 Default

```text
Stage 1 = popularity + cooccurrence
Stage 2 = LightGBM learned ranker
MMR = false
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

Default artifact:

```text
GNN_Neural_Network/artifacts/experiments/phase2_5_num_leaves_31/ranker_model.txt
```

## Final Metrics

### Validation

| Path | Recall@10 | NDCG@10 | coverage@10 | novelty@10 | Status |
|:---|---:|---:|---:|---:|:---|
| Stage 1 `popularity + cooccurrence` | 0.694035 | 0.435455 | 0.127778 | 4.483649 | baseline |
| v1 deterministic reranker | 0.709887 | 0.442340 | 0.516667 | 4.732133 | fallback/comparison |
| Phase 2.5 default LightGBM | 0.739051 | 0.457970 | 0.155556 | 4.584287 | selected default |

### Test

| Path | Recall@10 | NDCG@10 | coverage@10 | novelty@10 | Status |
|:---|---:|---:|---:|---:|:---|
| Stage 1 `popularity + cooccurrence` | 0.690885 | 0.437556 | 0.127778 | 4.483649 | baseline |
| v1 deterministic reranker | 0.704298 | 0.440329 | 0.516667 | 4.732133 | fallback/comparison |
| Phase 2.5 default LightGBM | 0.709684 | 0.447713 | 0.155556 | 4.584287 | selected default |

## Accepted Decisions

- Keep Stage 1 default as `popularity + cooccurrence`.
- Promote LightGBM learned ranker over Stage 1 and v1 deterministic reranker for accuracy-oriented default ranking.
- Promote `num_leaves=31` regularized LightGBM configuration.
- Keep `neg_ratio=4`, `hard_ratio=0.8` after negative sampling ablation.
- Keep `include_source_features=false` after source one-hot ablation.
- Keep `include_text_embedding_feature=false` until leakage-safe text experiments pass gates.
- Keep `MMR=false` as default until KURE dense embedding MMR passes validation and final test gates.

## Rejected Or Non-Default Decisions

- `segment_popularity`: disabled because it degraded recall and NDCG.
- BM25/PMI/IDF/Jaccard/pop-capped providers: not selected because they did not beat the selected Stage 1 baseline.
- LightGCN merged into default Stage 1: not selected because it lowered validation recall.
- Category one-hot MMR: no-go because binary cosine similarity made lambda sweep ineffective.
- Negative sampling default change to `hard_ratio=1.0`: rejected because validation improved but final test Recall/NDCG underperformed current default.
- Source one-hot features: rejected because validation Recall/NDCG/coverage were below current default.

## Known Limitations

- Ranking collapse is not fully solved.
- Candidate recall is high (`candidate_recall@50 ~= 0.977`), so retrieval coverage is not the primary bottleneck.
- The learned ranker still concentrates top-k recommendations toward popular/cooccurring hobbies.
- `coverage@10=0.155556` remains far below v1 deterministic `coverage@10=0.516667`.
- `novelty@10=4.584287` remains below v1 deterministic `novelty@10=4.732133`.
- Feature importance remains dominated by `cooccurrence_score` and `popularity_prior`.

## Baseline For Next Experiments

All future experiments should compare against this fixed Phase 2.5 default unless explicitly documented otherwise.

Baseline validation reference:

```text
Recall@10=0.7390509094604207
NDCG@10=0.45797028878684237
coverage@10=0.15555555555555556
novelty@10=4.584286633989583
candidate_recall@50=0.9776445483182603
```

Baseline test reference:

```text
Recall@10=0.7096839752057718
NDCG@10=0.447712669317698
coverage@10=0.15555555555555556
novelty@10=4.584286633989583
candidate_recall@50=0.977136469870948
```

## Next Experiment Family

The next experiment family is not part of Phase 2.5 default tuning unless it explicitly passes promotion gates.

Recommended next path:

- KURE dense embedding MMR re-evaluation
- leakage-safe text embedding feature ablation
- optional future ranking objective work such as LambdaRank/pairwise reranking if diversity remains unresolved

Promotion rule for next experiments:

- Select on validation only.
- Run test only once for the selected validation winner.
- Do not change default unless the selected experiment preserves accuracy and improves diversity under documented gates.
