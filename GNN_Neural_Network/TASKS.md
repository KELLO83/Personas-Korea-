# GNN Recommender Tasks

This file tracks executable tasks for `GNN_Neural_Network/` experiments. For requirements and design decisions, use `PRD.md`. For historical v2 reranker checklist details, use `CHECKLIST_GNN_Reranker_v2.md`.

## Global Execution Policy

- [x] All post-`Phase2.5` default promotion candidates use an accuracy-first hard gate.
- [x] Default promotion hard accuracy gate: `delta_recall@10 >= -0.002`, `delta_ndcg@10 >= -0.002` (vs closed `phase2_5_default`).
- [x] Ranking-collapse exploration may additionally record a non-promoting `diversity_probe` status.
- [x] Diversity probe accuracy gate: `delta_recall@10 >= -0.010`, `delta_ndcg@10 >= -0.010` (vs closed `phase2_5_default`).
- [x] `diversity_probe` status cannot change the default path without a later default-promotion pass.
- [x] Diversity is secondary: at least 2 of `coverage@10`, `novelty@10`, `intra_list_diversity@10` must satisfy minimum gains.
- [x] Diversity minimum gain thresholds
  - `coverage@10`: `+0.025`
  - `novelty@10`: `+0.10`
  - `intra_list_diversity@10`: `+0.02`
- [x] Stability gate baseline: `v2_fallback_count=0`, `candidate_recall@50` drift within tolerance.
- [x] Validation-first + winner-only test for any metric tie-break change.
- [x] Default experiment scope: 10K offline pilot before any full-scale follow-up.
- [x] Phase 5+ evaluations record cold-start subset metrics (`known_hobbies <= 1`) separately from overall metrics.
- [x] Keep artifact governance fixed:
  - `GNN_Neural_Network/artifacts/experiments/<phase>/<run>/...`
  - `validation_metrics.json` + `validation_metrics.status.json` must exist every trial.
  - `test_metrics.json` / `test_metrics.status.json` only for selected winner.

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
- [x] baseline artifact paths verified
  - [x] `artifacts/experiments/phase2_5_num_leaves_31/validation_metrics.json`
  - [x] `artifacts/experiments/phase2_5_num_leaves_31/test_metrics.json`

### Implementation Design Before Code

- [x] KURE hobby embedding generation path designed
  - [x] model: `nlpai-lab/KURE-v1`
  - [x] output: L2-normalized dense embedding matrix
- [x] `HobbyEmbeddingCache` reuse policy defined
  - [x] cache directory layout
  - [x] metadata fields: model name, hobby list/hash, embedding dimension, created timestamp
- [x] `evaluate_ranker.py` option plan finalized
  - [x] `--mmr-embedding-method category_onehot|kure`
  - [x] `--embedding-cache-dir <path>`
  - [x] `--embedding-batch-size <int>`
- [x] MMR application scope fixed
  - [x] apply MMR to full `candidate_k=50` pool
  - [x] do not apply MMR only after truncating to top-k
- [x] category one-hot fallback behavior preserved
- [x] KURE load/device/batch policy defined
  - [x] CUDA when available
  - [x] CPU fallback
  - [x] configurable batch size
- [x] existing `sweep_mmr_lambda.py` disposition decided
  - [x] keep as legacy, or
  - [x] refactor to use full candidate pool and KURE method

### Validation Sweep Plan

- [x] lambda candidates confirmed
  - [x] `0.5`
  - [x] `0.7`
  - [x] `0.8`
  - [x] `0.9`
  - [ ] optional `0.3` only if accuracy/diversity curve needs a low-lambda point
- [x] each lambda executed as one validation run
- [x] no test execution until validation winner is selected
- [x] validation failure means test is skipped

#### Validation outcome summary

- [x] `0.5` validation complete (`blocked`)
- [x] `0.7` validation complete (`blocked`)
- [x] `0.8` validation complete (`blocked`)
- [x] `0.9` validation complete (`blocked`)

  - no validation winner passed the promotion gate
  - test runs were skipped intentionally

### Promotion Gates

- [x] accuracy gate finalized
  - [x] `delta_recall@10 >= -0.002` vs closed default
  - [x] `delta_ndcg@10 >= -0.002` vs closed default
- [x] diversity gate finalized
  - [x] at least 2 of these improve: `coverage@10`, `novelty@10`, `intra_list_diversity@10`
  - [x] decide whether `coverage@10` improvement is mandatory for default promotion
- [x] stability gate finalized
  - [x] `v2_fallback_count=0`
  - [x] `candidate_recall@50` remains effectively unchanged
  - [x] KURE embedding cache is reusable

- [x] Result

  - [x] all candidates failed accuracy gate (`recall@10`, `ndcg@10`) and then skipped test
  - [x] diversity/stability gates were secondary after gate fail
  - [x] Phase 5-A final status: `NO-GO`

### Artifacts

- [x] lambda validation output paths finalized
  - [x] `artifacts/experiments/phase5_kure_mmr_lambda_<lambda>/validation_metrics.json`
  - [x] `artifacts/experiments/phase5_kure_mmr_lambda_<lambda>/validation_metrics.status.json`
 - [x] selected lambda test output policy finalized
  - [x] `artifacts/experiments/phase5_kure_mmr_lambda_<lambda>/test_metrics.json` intentionally absent (no winner selected)
  - [x] `artifacts/experiments/phase5_kure_mmr_lambda_<lambda>/test_metrics.status.json` intentionally absent (no winner selected)
- [x] summary artifact planned
  - [x] `artifacts/experiments/phase5_kure_mmr_summary.md`
- [x] decision artifact schema planned
  - [x] `artifacts/experiment_decisions.json` key: `phase5_kure_mmr`
- [x] run summary update planned
  - [x] `artifacts/experiment_run_summary.md`

### Implementation Gate

- [x] `PRD.md` and this `TASKS.md` KURE MMR plan reviewed for consistency
- [x] code-change scope approved
- [x] implementation starts only after the above planning items are accepted

## Phase 5-B: Ranking Collapse Mitigation

> Goal: reduce `v2 LightGBM` top-k popularity concentration while preserving the closed Phase 2.5 baseline.

### Baseline Lock

- [x] `artifacts/experiments/phase2_5_num_leaves_31/validation_metrics.json`
  - `Recall@10=0.7390509094604207`
  - `NDCG@10=0.45797028878684237`
  - `candidate_recall@50=0.9776445483182603`
  - `coverage@10=0.15555555555555556`
  - `novelty@10=4.584286633989583`
- [x] stability reference
  - `v2_fallback_count=0`
  - `candidate_recall@50` drift tolerance for single-run checks

### Execution Policy

- [x] follows single-config policy (validation-first, winner-only testing)
- [x] no blind multi-config ablation without prior Phase 2.5-locked baseline comparison
- [x] one run path per subtask

### Step 1: listwise objective probe

- [x] add/enable listwise objective experiment path in ranking pipeline (예: `LambdaRank` 단일 설정)
- [x] run validation passes
  - `smoke_lambdarank_f07`: completed, blocked
  - `smoke_lambdarank_f08`: completed, blocked
- [x] no winner passed gates → test pass skipped

### Step 2: feature balance probe

- [ ] run `feature_fraction` probe #1 (`0.7`) under closed baseline config
  - training completed
  - validation run was started but not completed before handoff
- [ ] run `feature_fraction` probe #2 (`0.8`) under closed baseline config
  - planned next
- [ ] compare against closure gates; only qualifying winner enters test

### Step 3: constrained diversification rerank probe

- [x] define one constrained rerank experiment (DPP / greedy ILD variant)
- [x] keep input set as full `candidate_k=50` pool before rerank
- [x] single validation run
- [x] DPP rerank validation completed (`phase5_b3_diversity_rerank/dpp_candidate_k50`), blocked on primary/secondary gates

### Step 4: taxonomy over-merge tracking (continuous)

- [x] track `canonical singleton ratio`, `raw singleton ratio`, and candidate set drift in artifacts (ongoing)
- [ ] record whether over-merge risk contributes to ranking collapse

### Promotion Gates

- [ ] default promotion accuracy gate
  - [ ] `delta_recall@10 >= -0.002` (closed Phase 2.5 baseline 대비)
  - [ ] `delta_ndcg@10 >= -0.002` (closed Phase 2.5 baseline 대비)
- [ ] diversity probe gate (non-promoting)
  - [ ] `delta_recall@10 >= -0.010` (closed Phase 2.5 baseline 대비)
  - [ ] `delta_ndcg@10 >= -0.010` (closed Phase 2.5 baseline 대비)
  - [ ] probe result is recorded as `diversity_probe`, not `promoted`
- [ ] diversity gate
  - [ ] at least 2/3 metrics improve by thresholds: `coverage@10`, `novelty@10`, `intra_list_diversity@10`
- [ ] stability gate
  - [ ] `v2_fallback_count=0`
  - [ ] `candidate_recall@50` 유지 또는 허용 오차 이내
  - [ ] rerank cache/KURE embedding cache 정책 일관성 유지
- [ ] cold-start subset metrics recorded
  - [ ] recall@10 / ndcg@10
  - [ ] coverage@10 / novelty@10 / intra_list_diversity@10

### Artifacts

- [x] `artifacts/experiments/phase5_b1_listwise/*/validation_metrics.json`
- [x] `artifacts/experiments/phase5_b1_listwise/*/validation_metrics.status.json`
- [ ] `artifacts/experiments/phase5_b1_listwise/*/test_metrics.json` (winner only)
- [ ] `artifacts/experiments/phase5_b1_listwise/*/test_metrics.status.json` (winner only)
- [ ] `artifacts/experiments/phase5_b2_feature_balance/*/validation_metrics.json`
- [ ] `artifacts/experiments/phase5_b2_feature_balance/*/validation_metrics.status.json`
- [ ] `artifacts/experiments/phase5_b2_feature_balance/*/test_metrics.json` (winner only)
- [ ] `artifacts/experiments/phase5_b2_feature_balance/*/test_metrics.status.json` (winner only)
- [x] `artifacts/experiments/phase5_b3_diversity_rerank/*/validation_metrics.json`
- [x] `artifacts/experiments/phase5_b3_diversity_rerank/*/validation_metrics.status.json`
- [ ] `artifacts/experiment_decisions.json`
  - [ ] `phase5_ranking_collapse_mitigation` 항목 갱신
- [ ] `artifacts/experiment_run_summary.md`
  - [ ] Phase 5-B 결과 요약 반영

## Phase 5-C: Leakage-Safe Text Embedding Ablation

> Goal: test whether masked persona text contains useful persona-aware signal that can reduce ranking collapse without leaking held-out hobbies.

### Policy Lock

- [x] Default path remains `include_text_embedding_feature=false` until a separate default-promotion decision is accepted.
- [x] Text embedding ablation may use `diversity_probe` status for exploration, but cannot promote the default path by itself.
- [x] Validation-first + winner-only test remains required.
- [x] If masking/audit fails, the run is excluded from validation/test metric comparison.

### Implementation Tasks

- [ ] add training CLI/config path for `include_text_embedding_feature=true`
- [ ] pass `include_text_embedding_feature` into `build_ranker_dataset()` from `train_ranker.py`
- [ ] compute `text_embedding_similarity` during ranker dataset construction
- [ ] apply `mask_holdout_hobbies()` before encoding persona text
- [ ] run `post_mask_leakage_audit()` and persist audit summary in artifacts
- [ ] cache persona/hobby text embeddings with reusable keys independent of `experiment_id`
- [ ] make evaluation feature construction match training feature construction
- [ ] record `feature_policy.include_text_embedding_feature=true` in train/eval artifacts

### Validation Plan

- [ ] run one validation ablation with closed Phase 2.5 baseline settings plus text embedding feature
- [ ] compare against closed Phase 2.5 baseline
- [ ] evaluate default promotion gate
- [ ] evaluate diversity probe gate
- [ ] evaluate diversity gate (`coverage@10`, `novelty@10`, `intra_list_diversity@10`)
- [ ] evaluate stability gate (`candidate_recall@50`, `v2_fallback_count`)
- [ ] record cold-start subset metrics
- [ ] run test only if validation selects a winner under the applicable policy

### Artifacts

- [ ] `artifacts/experiments/phase5_c_text_embedding/<run>/ranker_model.txt`
- [ ] `artifacts/experiments/phase5_c_text_embedding/<run>/ranker_params.json`
- [ ] `artifacts/experiments/phase5_c_text_embedding/<run>/validation_metrics.json`
- [ ] `artifacts/experiments/phase5_c_text_embedding/<run>/validation_metrics.status.json`
- [ ] `artifacts/experiments/phase5_c_text_embedding/<run>/text_leakage_audit.json`
- [ ] `artifacts/experiments/phase5_c_text_embedding/<run>/test_metrics.json` (winner only)
- [ ] `artifacts/experiments/phase5_c_text_embedding/<run>/test_metrics.status.json` (winner only)
- [ ] `artifacts/experiment_decisions.json`
  - [ ] `phase5_text_embedding_ablation` 항목 갱신
- [ ] `artifacts/experiment_run_summary.md`
  - [ ] Phase 5-C 결과 요약 반영

## Phase 5-D: Cold-Start Evaluation Slice

> Goal: determine whether ranking-collapse mitigation helps users with sparse known hobbies, where persona text may carry the most value.

- [ ] define cold-start subset as `known_hobbies <= 1`
- [ ] add subset metric computation to ranker evaluation artifacts
- [ ] report cold-start recall@10 and ndcg@10
- [ ] report cold-start coverage@10, novelty@10, intra_list_diversity@10
- [ ] compare cold-start results for closed Phase 2.5 default, Phase 5-B candidates, and Phase 5-C text embedding ablation
- [ ] record whether cold-start gains justify follow-up even if overall default-promotion gate fails
