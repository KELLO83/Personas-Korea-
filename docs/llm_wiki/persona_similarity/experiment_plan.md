# Person -> Person Experiment Plan

## Objective

Decide whether any similar-persona reranker is trustworthy enough to move beyond FastRP/KNN production behavior.

## First Branch: Manual Review Gate

Recommended first branch:

```text
manual review of structured LambdaRank and text-driven examples
```

Reason:

```text
automatic weak-label NDCG is already high
production blocker is trust and semantic quality
```

## Baselines To Reproduce First

Use these as comparison anchors:

```text
FastRP baseline
deterministic baseline
structured LambdaRank
E5 text-only LambdaRank
structured+text LambdaRank
```

Required metrics:

- NDCG@K
- explanation coverage
- strong-reason coverage
- low-information dominance
- diversity
- runtime
- model size
- manual review status

## Gate

Do not promote unless:

```text
candidate width is topK >= 50
source-disjoint split is preserved
no raw uuid/display_name/raw text identifier feature is used
manual review approves semantic recommendation quality
rollback remains FastRP/KNN ordering through fastrp_score
```

## No-Go

- Do not promote from weak-label NDCG alone.
- Do not run KURE/Snowflake-ko swaps before E5 review identifies the target failure mode.
- Do not merge this result with hobby recommendation metrics.
