# Source Card: Diversity Reranking

## Metadata

- Type: Post-ranking tradeoff method
- URL: local experiment artifacts
- Authors: local experiment track
- Year: 2026
- Local status: Tried in multiple forms
- Compatibility label: Accuracy-gated

## Summary

Diversity reranking reorders a scored candidate list to improve coverage, novelty, or attribute diversity. It is useful when the default ranker collapses toward narrow high-confidence recommendations.

## Relevance To This Project

The hobby recommender has a clear diversity/coverage gap: the current LightGBM default improves accuracy but has much lower coverage than the deterministic reranker. Prior MMR and DPP attempts improved novelty or coverage in places but failed accuracy gates. This makes diversity reranking relevant but not safe as a default without stricter gates.

Similar-persona diversity reranking improved diversity and low-information metrics but lowered NDCG versus structured LambdaRank, so it remains an optional offline tradeoff.

## Protocol Match

| Item | Source | This Project | Match |
|---|---|---|---|
| Task | rerank candidate list | both systems | Good |
| Data unit | ranked candidate list | hobby candidates or similar-persona candidates | Good |
| Candidate pool | existing candidates | Stage 1 or `SIMILAR_TO` topK | Good |
| Label | original relevance | held-out hobby or weak pair label | Good |
| Metric | accuracy/diversity tradeoff | Recall/NDCG plus coverage, novelty, diversity | Good |

## Adopt / Defer / Avoid

- Adopt: Keep as controlled tradeoff experiment.
- Defer: Default promotion until accuracy loss is within an agreed tolerance.
- Avoid: Optimizing novelty while collapsing Recall/NDCG.

## Claim Boundary Impact

Diversity reranking is an experiment track, not a default change.
