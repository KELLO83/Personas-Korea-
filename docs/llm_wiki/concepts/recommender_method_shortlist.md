# Recommender Method Shortlist

## Person -> Hobby Priority

| Priority | Method | Decision | Reason |
|---:|---|---|---|
| 1 | FM/FFM/DeepFM or Wide&Deep-style ranker | try-first | Current gap is ranker interaction/diversity, not candidate recall. |
| 2 | Diversity-aware reranking with strict accuracy tolerance | benchmark | Coverage/novelty are weak, but prior MMR/DPP lost too much Recall/NDCG. |
| 3 | LightGCN/XSimGCL embedding feature comparison | benchmark | Fits bipartite graph, but candidate recall is already strong. |
| 4 | KG-aware method such as KGAT | defer | Heterogeneous graph exists, but complexity is high and ranking gap is more immediate. |
| 5 | Text embedding hobby features | manual-review-only | Leakage risk is high unless masking/audit is complete. |

## Person -> Person Priority

| Priority | Method | Decision | Reason |
|---:|---|---|---|
| 1 | Manual review of structured/text examples | try-first | Promotion blocker is trust, not automatic metric. |
| 2 | Text-driven reranking with E5 baseline review | manual-review gated | Text-only has useful signal, but structured+text is not better than structured LambdaRank. |
| 3 | KURE/Snowflake-ko backbone swap | defer | Only after E5 manual review establishes a useful target. |
| 4 | Two-tower/text retrieval comparison | benchmark | Useful offline comparison against FastRP/KNN, not a promotion shortcut. |
| 5 | KG-aware persona embedding | defer | Needs careful weak-label/manual-review design. |

## Rejected For Current Data

- Reinforcement learning and bandits: no real online feedback loop.
- Sequential/session recommenders: no session or time-order interaction logs.
- Raw text tree-model features: violates feature policy and leakage controls.
