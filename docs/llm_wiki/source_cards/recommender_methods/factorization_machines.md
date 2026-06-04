# Source Card: Factorization Machines

## Metadata

- Type: Tabular feature interaction model
- URL: https://www.gabormelli.com/RKB/2010_FactorizationMachines
- Authors: Steffen Rendle
- Year: 2010
- Local status: Not implemented as a separate ranker
- Compatibility label: High-priority

## Summary

Factorization Machines model pairwise feature interactions efficiently in sparse tabular settings. Field-aware or deep variants are common recommender ranker alternatives when structured categorical/context features matter.

## Relevance To This Project

This is one of the best first alternatives for the hobby ranker because the local gap is ranking/diversity, not candidate recall. The hobby system already has structured context, candidate-provider signals, popularity/co-occurrence signals, and LightGBM ranker features. FM/FFM/DeepFM-style models can directly test whether learned feature interactions outperform the current tree ranker or improve coverage/novelty tradeoffs.

For similar-persona ranking, FM-style interaction models are also plausible over pair features, but the current structured LambdaRank is already extremely strong on weak-label NDCG. Manual review remains the stronger blocker.

## Protocol Match

| Item | Source | This Project | Match |
|---|---|---|---|
| Task | tabular recommendation ranking | both systems | Good |
| Data unit | sparse feature row | ranker row or source-target pair row | Good |
| Candidate pool | external candidates | existing candidate pools | Good |
| Label | implicit/explicit target | held-out hobby or weak pair label | Partial |
| Metric | ranking metrics | Recall/NDCG plus project gates | Good |

## Adopt / Defer / Avoid

- Adopt: Try first for hobby ranker alternatives.
- Defer: Similar-persona FM until manual-review analysis shows structured LambdaRank failure modes.
- Avoid: Adding raw IDs or raw text identifiers as feature crosses.

## Claim Boundary Impact

This changes the experiment priority: FM/FFM/DeepFM-style rankers should be evaluated before new candidate retrievers for hobby recommendation.
