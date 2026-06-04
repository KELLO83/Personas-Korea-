# Current Recommender Findings

## Current Status

The project has two separate recommender systems. Their datasets, labels, metrics, artifacts, and promotion decisions must stay separate.

```text
GNN_Neural_Network/              Person -> Hobby
experiments/persona_similarity/  Person -> Person
```

## Person -> Hobby

There are two local result layers. `GNN_Neural_Network/EXPERIMENTS.md` records the Phase 2.5-era baseline. `GNN_Neural_Network/artifacts/experiment_run_summary.md` records later E5-domain and Phase 6 results. Use `docs/llm_wiki/person_hobby/results_summary.md` for the consolidated view.

Latest artifact summary default:

```text
Stage 1 = popularity + cooccurrence
Stage 2 = LightGBM learned ranker + E5-small-ko-v2 single + domain-specific text similarities
production_embedding_model = dragonkue/multilingual-e5-small-ko-v2
include_text_embedding_feature = true
include_domain_text_embedding_features = true
MMR = false
```

Latest Phase 6 caveat:

```text
phase6_domain_text_hard1_aliases_full_validation
test Recall@10 = 0.710786
test NDCG@10 = 0.464645
test + topic calibration Recall@10 = 0.711338
test + topic calibration NDCG@10 = 0.464943
```

This is the strongest stored test artifact in the run summary, but it carries validation-selection and alias provenance caveats.

Closed Phase 2.5 default:

```text
Stage 1 = popularity + cooccurrence
Stage 2 = LightGBM learned ranker
MMR     = false
```

Key recorded metrics:

| Split | Path | Recall@10 | NDCG@10 | Coverage@10 | Novelty@10 |
|---|---|---:|---:|---:|---:|
| test | Stage 1 popularity + cooccurrence | 0.690885 | 0.437556 | 0.127778 | 4.483649 |
| test | v1 deterministic reranker | 0.704298 | 0.440329 | 0.516667 | 4.732133 |
| test | Phase 2.5 LightGBM default | 0.709684 | 0.447713 | 0.155556 | 4.584287 |

Interpretation:

- The current default is the best accuracy-oriented path.
- The deterministic reranker remains much stronger for catalog coverage.
- CandidateRecall@50 is about `0.977`, so the main unresolved problem is ranking collapse and diversity, not candidate retrieval.
- Text features are risky because held-out hobby names can appear in persona text.

## Person -> Person

Current production behavior:

```text
FastRP/KNN SIMILAR_TO + post-hoc explanation API
```

Current offline best:

```text
structured_lambdarank
NDCG@5  = 0.993136
NDCG@10 = 0.993145
status  = offline experimental
```

Interpretation:

- Structured LambdaRank is the best current weak-label reranker.
- Production is not changed because manual review is not approved.
- E5 text-only reranking shows useful signal versus raw FastRP, but structured+text does not beat structured LambdaRank under the current weak-label metric.
- Diversity reranking is an optional tradeoff, not a promoted default.

## Active Research Direction

For hobby recommendation, prioritize governance-safe ranker improvements, feature interaction alternatives, and diversity-aware gates before replacing the candidate retriever.

For similar-persona recommendation, prioritize manual review of text-driven/weak-label examples before additional backbone swaps or production promotion.
