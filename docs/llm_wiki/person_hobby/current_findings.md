# Person -> Hobby Current Findings

## Current Result State

There are two result layers:

- `GNN_Neural_Network/EXPERIMENTS.md` records the closed Phase 2.5 baseline state from 2026-05-05.
- `GNN_Neural_Network/artifacts/experiment_run_summary.md` records later E5-domain and Phase 6 follow-ups from 2026-05-17 to 2026-05-20.

Use `results_summary.md` as the LLM Wiki consolidation of both layers.

## Current Default In Latest Artifact Summary

The latest run summary records the current default path as:

```text
Stage 1 = popularity + cooccurrence
Stage 2 = LightGBM learned ranker + E5-small-ko-v2 single + domain-specific text similarities
production_embedding_model = dragonkue/multilingual-e5-small-ko-v2
include_source_features = false
include_text_embedding_feature = true
include_domain_text_embedding_features = true
MMR = false
```

## Closed Phase 2.5 Baseline

```text
Stage 1 = popularity + cooccurrence
Stage 2 = LightGBM learned ranker
MMR     = false
```

Closed Phase 2.5 artifact:

```text
GNN_Neural_Network/artifacts/experiments/phase2_5_num_leaves_31/ranker_model.txt
```

## Local Dataset Shape

| Artifact | Rows | Columns | Meaning |
|---|---:|---:|---|
| `GNN_Neural_Network/data/person_hobby_edges.csv` | 50,000 | 2 | `person_uuid,hobby_name` edges |
| `GNN_Neural_Network/data/person_context.csv` | 50,000 | 21 | structured context and persona text |

## Phase 2.5 Metrics

| Split | Path | Recall@10 | NDCG@10 | Coverage@10 | Novelty@10 |
|---|---|---:|---:|---:|---:|
| test | Stage 1 `popularity + cooccurrence` | 0.690885 | 0.437556 | 0.127778 | 4.483649 |
| test | v1 deterministic reranker | 0.704298 | 0.440329 | 0.516667 | 4.732133 |
| test | Phase 2.5 LightGBM default | 0.709684 | 0.447713 | 0.155556 | 4.584287 |

## Interpretation

- Phase 2.5 was the best accuracy-oriented path at the time of `EXPERIMENTS.md`.
- CandidateRecall@50 is about `0.977`, so candidate retrieval is not the first bottleneck.
- The main unresolved issue is ranking collapse and weaker coverage/novelty versus the deterministic reranker.
- Later E5-domain Stage2 runs changed the active baseline in `artifacts/experiment_run_summary.md`.
- Text features remain governance-sensitive because persona text and candidate aliases can inject leakage or taxonomy/canonicalization bias.

## First Experiment Priority

Try ranker-side alternatives before new candidate retrievers:

1. FM/FFM/DeepFM or Wide&Deep-style feature interaction ranker.
2. Diversity-aware reranking with strict Recall/NDCG tolerance.
3. LightGCN/XSimGCL embedding features only as a controlled comparison.
