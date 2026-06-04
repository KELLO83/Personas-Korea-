# Person -> Hobby Existing Results Summary

## Source Layer Caveat

Two local result summaries exist and they are not the same snapshot:

| Source | Date / state | Role |
|---|---|---|
| `GNN_Neural_Network/EXPERIMENTS.md` | Last updated 2026-05-05 | Phase 2.5-era decision record |
| `GNN_Neural_Network/artifacts/experiment_run_summary.md` | 2026-05-17 plus 2026-05-20 follow-up | Later E5-domain and Phase 6 result record |

When they differ, prefer the newer artifact summary for "current result state", but keep the Phase 2.5 summary as a baseline reference.

## Current Default From Latest Artifact Summary

```text
Stage 1 = popularity + cooccurrence
Stage 2 = LightGBM learned ranker + E5-small-ko-v2 single + domain-specific text similarities
production_embedding_model = dragonkue/multilingual-e5-small-ko-v2
include_source_features = false
include_text_embedding_feature = true
include_domain_text_embedding_features = true
MMR = false
```

Latest default decision source:

```text
GNN_Neural_Network/artifacts/experiment_run_summary.md
GNN_Neural_Network/artifacts/experiment_decisions.json
```

## Latest Phase 6 Follow-Up

Best recorded Phase 6 candidate:

```text
run_id = phase6_domain_text_hard1_aliases_full_validation
Stage 1 = popularity + cooccurrence
Stage 2 = LightGBM(num_leaves=31)
negative_sampling = neg_ratio=4, hard_ratio=1.0
candidate_text_builder = name_plus_aliases
include_text_embedding_feature = true
include_domain_text_embedding_features = true
KURE semantic Stage1 provider = false
topic calibration = optional lambda=0.02 post-ranker
```

| Split / Variant | Recall@10 | NDCG@10 | ILD@10 | Decision |
|---|---:|---:|---:|---|
| validation | 0.732523 | 0.480773 | 0.969728 | eligible_for_test |
| validation + topic calibration `lambda=0.02` | 0.732707 | 0.480842 | 0.969934 | optional post-ranker |
| test | 0.710786 | 0.464645 | 0.969734 | promoted test artifact |
| test + topic calibration `lambda=0.02` | 0.711338 | 0.464943 | 0.969949 | optional post-ranker |

Caveat:

- It does not beat the old validation Recall@10 leader `phase2_5_neg_ratio_4_hard_1_0` (`0.742404` vs `0.732523`).
- It improves validation NDCG@10 substantially (`0.458620` -> `0.480773`).
- It is the strongest stored test artifact found in the run summary, but test results must not be used to retroactively select models without the validation caveat.
- `candidate_text_builder=name_plus_aliases` still requires alias source-field provenance and canonicalization bias approval for production wiring.

## E5-Small Domain-Specific Stage2 Promotion

Run:

```text
artifacts/experiments/phase5_c_text_embedding/e5_domain_features_validation_thread18/
artifacts/experiments/phase5_c_text_embedding/e5_domain_features_test_thread18/
```

| Split | Recall@10 | NDCG@10 | CandidateRecall@50 | Decision |
|---|---:|---:|---:|---|
| validation | 0.699180 | 0.448862 | 0.827669 | passed |
| test | 0.680943 | 0.436665 | 0.827208 | promoted current SOTA/default in artifact summary |

Test deltas recorded in the artifact:

- vs Stage1: Recall@10 `+0.111173`, NDCG@10 `+0.080306`
- vs E5-small single: Recall@10 `+0.057106`, NDCG@10 `+0.042744`
- vs Snowflake-ko single: Recall@10 `+0.043290`, NDCG@10 `+0.033860`
- vs KURE-v1 single: Recall@10 `+0.063461`, NDCG@10 `+0.050407`

## E5-Domain Rank/Margin Follow-Up

Added features:

```text
e5_similarity_rank
e5_similarity_percentile
e5_similarity_gap_to_top
e5_similarity_gap_to_mean
```

| Split | Recall@10 | NDCG@10 | CandidateRecall@50 | Decision |
|---|---:|---:|---:|---|
| validation | 0.702404 | 0.449661 | 0.827669 | passed |
| test | 0.682509 | 0.436354 | 0.827208 | mixed |

Decision: keep the E5-domain default unchanged because Recall improved slightly but NDCG regressed slightly on test.

## Candidate Hobby Text Expansion

Tested builders:

```text
name_only
name_plus_aliases
name_plus_category
name_plus_short_description
```

| Builder | Split | Recall@10 | NDCG@10 | CandidateRecall@50 | Decision |
|---|---|---:|---:|---:|---|
| `name_plus_category` | validation | 0.676062 | 0.424624 | 0.827669 | rejected; test skipped |
| `name_plus_aliases` | validation | 0.711615 | 0.461267 | 0.827669 | metric-positive, not promoted |
| `name_plus_aliases` | test | 0.694207 | 0.445550 | 0.827208 | excluded by governance |
| `name_plus_short_description` | validation | 0.674035 | 0.426106 | 0.827669 | rejected; test skipped |

Decision: expanded candidate text builders stay non-default because alias/category/description metadata can inject taxonomy or canonicalization bias.

## Closed Phase 2.5 Baseline

| Split | Path | Recall@10 | NDCG@10 | Coverage@10 | Novelty@10 | Status |
|---|---|---:|---:|---:|---:|---|
| validation | Stage 1 `popularity + cooccurrence` | 0.694035 | 0.435455 | 0.127778 | 4.483649 | baseline |
| validation | v1 deterministic reranker | 0.709887 | 0.442340 | 0.516667 | 4.732133 | fallback / comparison |
| validation | Phase 2.5 LightGBM default | 0.739051 | 0.457970 | 0.155556 | 4.584287 | selected default |
| test | Stage 1 `popularity + cooccurrence` | 0.690885 | 0.437556 | 0.127778 | 4.483649 | baseline |
| test | v1 deterministic reranker | 0.704298 | 0.440329 | 0.516667 | 4.732133 | fallback / comparison |
| test | Phase 2.5 LightGBM default | 0.709684 | 0.447713 | 0.155556 | 4.584287 | selected default |

Interpretation at Phase 2.5:

- LightGBM improved Recall/NDCG over Stage 1 and deterministic reranker.
- Deterministic reranker retained much better coverage and novelty.
- CandidateRecall@50 was about `0.977`, so ranking collapse was the main unresolved issue.

## Rejected / Non-Default Experiments

| Experiment | Outcome |
|---|---|
| Negative sampling `hard_ratio=1.0` | validation won but final test lost vs current default |
| Source one-hot features | Recall/NDCG/Coverage regressed |
| KURE dense MMR sweep | all lambdas failed accuracy gates |
| LambdaRank smoke | diversity improved slightly, Recall/NDCG dropped heavily |
| DPP diversity rerank | novelty improved, Recall/NDCG collapsed |
| KURE semantic Stage1 | validation candidate recall dropped from `0.977645` to `0.794971`; rejected |
| Candidate text `name_plus_category` / `name_plus_short_description` | rejected |
| Candidate text `name_plus_aliases` | metric-positive but governance-excluded |

## Current Follow-Up Implication

The next useful hobby experiment is not a new semantic Stage1 retriever by default. The recorded results point to:

1. ranker-side feature interaction alternatives,
2. governance-safe text feature improvements,
3. diversity/coverage tradeoff experiments with strict Recall/NDCG gates.
