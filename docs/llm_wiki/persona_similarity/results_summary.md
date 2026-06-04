# Person -> Person Existing Results Summary

## Current Decision

Production behavior remains unchanged:

```text
FastRP/KNN SIMILAR_TO + post-hoc explanation API
```

LightGBM rerankers and E5 text features are offline experimental artifacts. Manual review is still required before production promotion.

## Dataset

```text
candidate pairs = 2,500,000
source personas = 50,000
candidate width = topK 50
train rows = 2,000,000
valid rows = 250,000
test rows = 250,000
```

## Main Test Metrics

| Experiment | NDCG@5 | NDCG@10 | Strong@5 | Low-info@5 | Decision |
|---|---:|---:|---:|---:|---|
| FastRP baseline | 0.519541 | 0.557837 | 0.627080 | 0.338200 | production baseline |
| deterministic baseline | 0.957142 | 0.963170 | 0.975120 | 0.024680 | strong weak-label baseline |
| structured LambdaRank | 0.993136 | 0.993145 | 0.916560 | 0.083120 | best current weak-label reranker |
| structured rank_xendcg | 0.988806 | 0.987662 | 0.943600 | 0.056160 | lower NDCG, stronger reason mix |
| E5 text-only LambdaRank | 0.691256 | 0.711463 | 0.746880 | 0.238760 | experimental text signal |
| E5 structured+text LambdaRank | 0.992754 | 0.992738 | 0.914680 | 0.085200 | not promoted |
| E5 structured+text rank_xendcg | 0.987959 | 0.987090 | 0.946880 | 0.052800 | not promoted |
| E5 structured+text hybrid alpha=0.9 | 0.991623 | 0.991685 | 0.928760 | 0.070840 | not promoted |

## E5 Text Feature Build

```text
model = dragonkue/multilingual-e5-small-ko-v2
device = cuda
batch_size = 128
embedding_rows = 400,000
embedding_dim = 384
runtime_seconds = 1433.094
features_with_text_rows = 2,500,000
```

Artifacts:

- `experiments/persona_similarity/artifacts/datasets/persona_text_embeddings.npz`
- `experiments/persona_similarity/artifacts/datasets/pair_features_with_text.parquet`
- `experiments/persona_similarity/artifacts/metrics/text_embedding_metadata.json`
- `experiments/persona_similarity/artifacts/metrics/text_feature_status.json`

## Text Feature Interpretation

- E5 text-only LambdaRank beats raw FastRP by a wide margin on weak-label NDCG.
- Structured+text LambdaRank is slightly below structured LambdaRank.
- The current weak-label policy is heavily structured-overlap based, so text can be useful without winning the current automatic metric.
- Manual review of text-driven examples is the next required gate.

## Diversity Rerank Follow-Up

| Base | Lambda | NDCG@5 | NDCG@10 | Low-info@5 | Occupation diversity@5 | Province diversity@5 | Community diversity@5 | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| structured LambdaRank | 0.05 | 0.992938 | 0.992896 | 0.082520 | 0.573480 | 0.573440 | 0.835080 | not promoted |
| structured LambdaRank | 0.1 | 0.992617 | 0.992480 | 0.080640 | 0.579600 | 0.582800 | 0.841200 | not promoted |
| structured LambdaRank | 0.2 | 0.991906 | 0.991646 | 0.075680 | 0.590120 | 0.601400 | 0.852840 | optional tradeoff |
| structured+text LambdaRank | 0.05 | 0.992521 | 0.992420 | 0.084280 | 0.573480 | 0.568960 | 0.833360 | not promoted |
| structured+text LambdaRank | 0.1 | 0.992268 | 0.992119 | 0.083040 | 0.580480 | 0.580760 | 0.841560 | not promoted |
| structured+text LambdaRank | 0.2 | 0.991518 | 0.991277 | 0.078080 | 0.590440 | 0.600080 | 0.854160 | optional tradeoff |

Decision: diversity rerank improves diversity and low-information metrics, but NDCG drops versus direct structured LambdaRank.

## Domain-Specific Text Cosine Ablation

| Experiment | Feature shape | NDCG@5 | NDCG@10 | Strong@5 | Low-info@5 | Decision |
|---|---|---:|---:|---:|---:|---|
| text-only all-text | `all_text_cosine` only | 0.628669 | 0.658979 | 0.695640 | 0.290200 | weaker control |
| text-only domain-specific | 8 E5 text cosine columns | 0.691256 | 0.711463 | 0.746880 | 0.238760 | better text-only shape |
| structured+all-text | structured + `all_text_cosine` | 0.992646 | 0.992597 | 0.911640 | 0.088200 | weaker control |
| structured+domain-specific | structured + 8 E5 text cosine columns | 0.992754 | 0.992738 | 0.914680 | 0.085200 | better text feature shape |

Decision: keep domain-specific E5 cosine as the preferred text feature shape for future text experiments, but do not promote it because structured LambdaRank remains the stronger weak-label reranker.

## Text Builder Ablation

| Builder | Embedding seconds | Feature seconds | NDCG@5 | NDCG@10 | Strong@5 | Low-info@5 | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| `domain_tagged_blocks` | 1433.094 | 36.000 | 0.992754 | 0.992738 | 0.914680 | 0.085200 | control |
| `structured_plus_narrative` | 2061.497 | 65.393 | 0.992346 | 0.992281 | 0.916360 | 0.082800 | not promoted |
| `narrative_only` | 1315.223 | 35.366 | 0.992032 | 0.991582 | 0.917680 | 0.082160 | not promoted |
| `structured_only` | 147.010 | 33.496 | 0.991181 | 0.991208 | 0.924080 | 0.075120 | leakage control, not promoted |

Decision: keep `domain_tagged_blocks` as the default text builder for this experiment family.

## Promotion Gate Status

Automatic checks passed:

- candidate width is exactly topK=50 for every source,
- train/valid/test splits are source-disjoint,
- raw `uuid`, `display_name`, and raw text identifier features are not model features,
- structured, text-only, and structured+text comparisons use the same test split size,
- rollback remains raw FastRP/KNN `SIMILAR_TO` ordering via `fastrp_score`.

Blocked gate:

```text
manual_review = not approved
```

## Current Follow-Up Implication

The next useful similar-persona experiment is not a new model first. It is manual review and failure-mode labeling for:

1. structured LambdaRank,
2. E5 text-only LambdaRank,
3. structured+text LambdaRank,
4. diversity-reranked variants.

Backbone swaps to KURE-v1 or Snowflake-ko should wait until E5 review identifies what semantic failure a new backbone should solve.
