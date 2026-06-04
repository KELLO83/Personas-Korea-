# Source Card: RecBole

## Metadata

- Type: Recommendation benchmark framework
- URL: https://recbole.io/docs/
- Authors: RecBole project
- Year: active project
- Local status: Not integrated
- Compatibility label: Benchmark/tooling reference

## Summary

RecBole is a PyTorch-based recommender-system library with model families across general, sequential, context-aware, and knowledge-based recommendation. Its model list is useful as a taxonomy for deciding which alternatives fit the local data.

## Relevance To This Project

Use RecBole as a reference taxonomy, not as an immediate dependency. The current project already has custom pipelines for hobby recommendation and similar-persona ranking. A direct RecBole integration would require adapter work for local artifacts and strict separation of the two recommender boundaries.

## Protocol Match

| Item | Source | This Project | Match |
|---|---|---|---|
| Task | General recommender families | `Person -> Hobby`, `Person -> Person` | Partial |
| Data unit | user-item interactions, context, sequences | hobby edges, directed persona pairs | Partial |
| Candidate pool | framework-specific | Stage 1 candidates or `SIMILAR_TO` topK | Partial |
| Label | implicit/explicit feedback | held-out hobby or weak pair label | Partial |
| Metric | ranking metrics | Recall/NDCG plus project-specific gates | Good |

## Adopt / Defer / Avoid

- Adopt: Use its model taxonomy for shortlist coverage.
- Defer: Full framework integration until the custom benchmark scripts are exhausted.
- Avoid: Using RecBole defaults without preserving local split, masking, and manual-review rules.

## Claim Boundary Impact

RecBole supports experiment planning only. It does not change the current default recommender.
