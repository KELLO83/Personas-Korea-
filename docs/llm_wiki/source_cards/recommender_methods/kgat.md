# Source Card: KGAT

## Metadata

- Type: Knowledge-graph-aware recommendation
- URL: https://arxiv.org/abs/1905.07854
- Authors: Xiang Wang, Xiangnan He, Yixin Cao, Meng Liu, Tat-Seng Chua
- Year: 2019
- Local status: Not implemented
- Compatibility label: Benchmark/defer

## Summary

KGAT models high-order relations in a knowledge graph with attention and embedding propagation. It is relevant when side information beyond direct user-item edges is central to the recommendation task.

## Relevance To This Project

The local Neo4j graph contains Person, Hobby, Occupation, District, Province, Education, FamilyType, HousingType, and other nodes. This makes KG-aware methods conceptually relevant. The risk is cost and complexity: the current hobby issue is ranking/diversity after strong candidate recall, and similar-persona promotion is blocked by manual review rather than graph-feature absence.

## Protocol Match

| Item | Source | This Project | Match |
|---|---|---|---|
| Task | KG-enhanced user-item recommendation | hobby graph or persona graph | Partial |
| Data unit | user-item plus KG triples | Person-Hobby plus heterogeneous graph | Good for hobby |
| Candidate pool | items | hobbies or similar personas | Partial |
| Label | interaction label | held-out hobby or weak pair label | Partial |
| Metric | ranking metrics | Recall/NDCG plus project gates | Good |

## Adopt / Defer / Avoid

- Adopt: Keep as a later KG-aware comparison if simpler ranker alternatives plateau.
- Defer: First execution until dataset-shape and baseline reports prove the KG features are the limiting factor.
- Avoid: Using KGAT to bypass text leakage/manual-review rules.

## Claim Boundary Impact

KGAT is a future offline comparison. It does not change current default behavior.
