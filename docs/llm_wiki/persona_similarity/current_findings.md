# Person -> Person Current Findings

## Current Production Behavior

```text
FastRP/KNN SIMILAR_TO + post-hoc explanation API
```

LightGBM rerankers and E5 text features are offline experimental artifacts.

## Local Dataset Shape

| Artifact | Rows | Columns | Meaning |
|---|---:|---:|---|
| `candidate_pairs.parquet` | 2,500,000 | 29 | FastRP/KNN topK=50 directed candidate pairs |
| `pair_features.parquet` | 2,500,000 | 45 | structured pair features and weak labels |
| `pair_features_with_text.parquet` | 2,500,000 | 53 | structured plus text cosine features |
| `persona_texts.parquet` | 50,000 | 12 | persona text source table |

## Current Offline Best

```text
offline_best_weak_label_reranker = structured_lambdarank
NDCG@5  = 0.993136
NDCG@10 = 0.993145
status  = offline experimental
```

## Interpretation

- Structured LambdaRank is the strongest weak-label reranker.
- It is not production-promoted because manual review is not approved.
- E5 text-only LambdaRank shows real semantic signal versus raw FastRP, but structured+text does not beat structured LambdaRank under current weak-label NDCG.
- Diversity reranking is an optional tradeoff, not a default.

## First Experiment Priority

Run manual-review-first validation before new model promotion:

1. Review text-driven and structured LambdaRank recommendation samples.
2. Identify semantic failure modes not captured by weak labels.
3. Only then compare KURE/Snowflake-ko text backbones or two-tower retrieval.
