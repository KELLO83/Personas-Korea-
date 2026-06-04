# Source Card: LightGCN / XSimGCL

## Metadata

- Type: Graph collaborative filtering
- URL: https://arxiv.org/abs/2002.02126, https://arxiv.org/abs/2209.02544
- Authors: LightGCN and XSimGCL authors
- Year: 2020, 2022
- Local status: LightGCN exists in the hobby recommender codebase; XSimGCL is a planned direction
- Compatibility label: Benchmark

## Summary

LightGCN simplifies graph convolution for collaborative filtering to neighborhood aggregation over a user-item graph. XSimGCL adds simple contrastive perturbations to improve representation uniformity and long-tail behavior.

## Relevance To This Project

For `Person -> Hobby`, these methods fit the bipartite person-hobby graph. However, the current hobby system already reports high candidate recall at 50, so graph-CF work should be framed as a controlled candidate-generation or embedding-feature comparison, not as the first ranking fix.

For `Person -> Person`, these methods do not directly fit unless the directed candidate-pair task is reformulated as graph embedding or KNN retrieval.

## Protocol Match

| Item | Source | This Project | Match |
|---|---|---|---|
| Task | user-item recommendation | `Person -> Hobby` | Good |
| Data unit | implicit user-item edge | `person_uuid,hobby_name` | Good |
| Candidate pool | all items or sampled candidates | retained hobby set / Stage 1 candidates | Good |
| Label | observed or held-out interactions | held-out hobby edges | Good |
| Metric | Recall/NDCG | Recall@K, NDCG@K, candidate recall, coverage, novelty | Good |

## Adopt / Defer / Avoid

- Adopt: Keep as candidate-generation and embedding-feature benchmark.
- Defer: Making it the primary experiment until ranking/diversity alternatives are tested.
- Avoid: Claiming graph-CF solves the current top gap without showing Recall/NDCG and coverage/novelty improvements.

## Claim Boundary Impact

This supports a benchmark track, not a production change.
