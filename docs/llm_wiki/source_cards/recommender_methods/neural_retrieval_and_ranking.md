# Source Card: NCF / Wide&Deep / Two-Tower Retrieval

## Metadata

- Type: Neural collaborative filtering, neural ranking, two-stage retrieval/ranking
- URL: https://arxiv.org/abs/1708.05031, https://arxiv.org/abs/1606.07792, https://research.google.com/pubs/archive/45530.pdf
- Authors: NCF, Wide&Deep, and YouTube DNN recommender authors
- Year: 2016-2017
- Local status: Not implemented as a distinct track
- Compatibility label: Benchmark/defer by task

## Summary

NCF replaces simple matrix-factorization interaction functions with neural networks. Wide&Deep combines memorization and generalization for ranking. Two-tower systems separate candidate retrieval from ranking and are useful at scale.

## Relevance To This Project

For hobby recommendation, a neural ranker or Wide&Deep-style interaction model is a plausible alternative to LightGBM if it can improve ranking without harming coverage and novelty. A two-tower retriever is less urgent because candidate recall is already high.

For similar-persona recommendation, two-tower text/structured retrieval may be useful as an offline comparison against FastRP/KNN, but promotion still depends on manual review and weak-label caveats.

## Protocol Match

| Item | Source | This Project | Match |
|---|---|---|---|
| Task | retrieval/ranking | both systems | Partial |
| Data unit | user-item or query-item | hobby edge or source-target pair | Good |
| Candidate pool | retrieved candidates | Stage 1 hobby candidates or `SIMILAR_TO` topK | Good |
| Label | interaction / engagement | held-out edge or weak pair label | Partial |
| Metric | ranking/engagement | Recall/NDCG plus quality gates | Good |

## Adopt / Defer / Avoid

- Adopt: Benchmark Wide&Deep-style feature interaction if LightGBM feature interactions look saturated.
- Defer: Two-tower retrieval for hobby until candidate recall becomes a real bottleneck.
- Avoid: Treating weak-label NDCG as enough for similar-persona production promotion.

## Claim Boundary Impact

This supports benchmark design. It does not imply online-scale architecture is needed now.
