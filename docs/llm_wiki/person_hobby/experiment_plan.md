# Person -> Hobby Experiment Plan

## Objective

Find whether a ranker-side alternative can improve hobby recommendation without sacrificing the accuracy gate that selected the Phase 2.5 LightGBM default.

## First Branch: Feature Interaction Ranker

Recommended first branch:

```text
FM / FFM / DeepFM / Wide&Deep-style ranker
```

Reason:

```text
candidate recall is already high
current problem is rank ordering and coverage/novelty tradeoff
```

## Baseline To Reproduce First

Use the current default as the comparison anchor:

```text
Stage 1 = popularity + cooccurrence
Stage 2 = LightGBM learned ranker
```

Required metrics:

- Recall@K
- NDCG@K
- CandidateRecall@K
- Coverage@K
- Novelty@K
- runtime
- qualitative hobby quality

## Gate

Do not promote unless:

```text
Recall/NDCG beats or matches Phase 2.5 default within the agreed tolerance
coverage/novelty improves enough to justify any small accuracy tradeoff
known-hobby masking remains correct
text leakage audit passes if any text features are used
```

## No-Go

- Do not prioritize a new retriever unless candidate recall falls below the current level.
- Do not use raw persona text as a direct tree-model feature.
- Do not merge this result with similar-persona recommendation metrics.
