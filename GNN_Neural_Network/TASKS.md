# GNN Recommender Tasks

This file tracks executable tasks for `GNN_Neural_Network/` experiments. For requirements and design decisions, use `PRD.md`. For historical v2 reranker checklist details, use `CHECKLIST_GNN_Reranker_v2.md`.

## Phase 2.5: Default Decision Closure

- [x] Regularization tuning completed
  - [x] `num_leaves=31` selected as current best LightGBM setting
  - [x] validation/test metrics recorded
- [x] Negative sampling ablation completed
  - [x] `neg_ratio=4`, `hard_ratio=1.0` selected by validation
  - [x] final test underperformed current default
  - [x] default remains `neg_ratio=4`, `hard_ratio=0.8`
- [x] Source one-hot ablation completed
  - [x] `include_source_features=true` evaluated on validation
  - [x] validation recall/ndcg/coverage below current default
  - [x] default remains `include_source_features=false`
- [x] Category one-hot MMR recorded as NO-GO
  - [x] binary category similarity made lambda sweep ineffective
  - [x] default remains `MMR=false`
- [x] Phase 2.5 default decision closure recorded
  - [x] `artifacts/experiments/phase2_5_default_decision_closure.md`
  - [x] `artifacts/experiment_decisions.json`
  - [x] `artifacts/experiment_run_summary.md`

Closed Phase 2.5 default:

```text
Stage 1 = popularity + cooccurrence
Stage 2 = LightGBM learned ranker
model = artifacts/experiments/phase2_5_num_leaves_31/ranker_model.txt
num_leaves=31
min_data_in_leaf=50
learning_rate=0.05
reg_alpha=0.1
reg_lambda=0.1
neg_ratio=4
hard_ratio=0.8
include_source_features=false
include_text_embedding_feature=false
MMR=false
```

## Phase 5-A: KURE Dense Embedding MMR Re-Evaluation

> Goal: Use the closed Phase 2.5 default as a fixed baseline and test whether KURE-v1 dense hobby embeddings make MMR useful for ranking-collapse mitigation.

### Baseline Lock

- [x] Phase 2.5 closed default confirmed
- [x] validation baseline recorded
  - [x] `Recall@10=0.7390509094604207`
  - [x] `NDCG@10=0.45797028878684237`
  - [x] `coverage@10=0.15555555555555556`
  - [x] `novelty@10=4.584286633989583`
  - [x] `candidate_recall@50=0.9776445483182603`
- [x] test baseline recorded
  - [x] `Recall@10=0.7096839752057718`
  - [x] `NDCG@10=0.447712669317698`
  - [x] `coverage@10=0.15555555555555556`
  - [x] `novelty@10=4.584286633989583`
  - [x] `candidate_recall@50=0.977136469870948`
- [ ] baseline artifact paths verified
  - [ ] `artifacts/experiments/phase2_5_num_leaves_31/validation_metrics.json`
  - [ ] `artifacts/experiments/phase2_5_num_leaves_31/test_metrics.json`

### Implementation Design Before Code

- [ ] KURE hobby embedding generation path designed
  - [ ] model: `nlpai-lab/KURE-v1`
  - [ ] output: L2-normalized dense embedding matrix
- [ ] `HobbyEmbeddingCache` reuse policy defined
  - [ ] cache directory layout
  - [ ] metadata fields: model name, hobby list/hash, embedding dimension, created timestamp
- [ ] `evaluate_ranker.py` option plan finalized
  - [ ] `--mmr-embedding-method category_onehot|kure`
  - [ ] `--embedding-cache-dir <path>`
  - [ ] `--embedding-batch-size <int>`
- [ ] MMR application scope fixed
  - [ ] apply MMR to full `candidate_k=50` pool
  - [ ] do not apply MMR only after truncating to top-k
- [ ] category one-hot fallback behavior preserved
- [ ] KURE load/device/batch policy defined
  - [ ] CUDA when available
  - [ ] CPU fallback
  - [ ] configurable batch size
- [ ] existing `sweep_mmr_lambda.py` disposition decided
  - [ ] keep as legacy, or
  - [ ] refactor to use full candidate pool and KURE method

### Validation Sweep Plan

- [ ] lambda candidates confirmed
  - [ ] `0.5`
  - [ ] `0.7`
  - [ ] `0.8`
  - [ ] `0.9`
  - [ ] optional `0.3` only if accuracy/diversity curve needs a low-lambda point
- [ ] each lambda executed as one validation run
- [ ] no test execution until validation winner is selected
- [ ] validation failure means test is skipped

### Promotion Gates

- [ ] accuracy gate finalized
  - [ ] `delta_recall@10 >= -0.002` vs closed default
  - [ ] `delta_ndcg@10 >= -0.002` vs closed default
- [ ] diversity gate finalized
  - [ ] at least 2 of these improve: `coverage@10`, `novelty@10`, `intra_list_diversity@10`
  - [ ] decide whether `coverage@10` improvement is mandatory for default promotion
- [ ] stability gate finalized
  - [ ] `v2_fallback_count=0`
  - [ ] `candidate_recall@50` remains effectively unchanged
  - [ ] KURE embedding cache is reusable

### Artifacts

- [ ] lambda validation output paths finalized
  - [ ] `artifacts/experiments/phase5_kure_mmr_lambda_<lambda>/validation_metrics.json`
  - [ ] `artifacts/experiments/phase5_kure_mmr_lambda_<lambda>/validation_metrics.status.json`
- [ ] selected lambda test output path finalized
  - [ ] `artifacts/experiments/phase5_kure_mmr_lambda_<lambda>/test_metrics.json`
  - [ ] `artifacts/experiments/phase5_kure_mmr_lambda_<lambda>/test_metrics.status.json`
- [ ] summary artifact planned
  - [ ] `artifacts/experiments/phase5_kure_mmr_summary.md`
- [ ] decision artifact schema planned
  - [ ] `artifacts/experiment_decisions.json` key: `phase5_kure_mmr`
- [ ] run summary update planned
  - [ ] `artifacts/experiment_run_summary.md`

### Implementation Gate

- [ ] `PRD.md` and this `TASKS.md` KURE MMR plan reviewed for consistency
- [ ] code-change scope approved
- [ ] implementation starts only after the above planning items are accepted
