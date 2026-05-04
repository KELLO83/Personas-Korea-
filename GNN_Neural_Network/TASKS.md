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

### Step 1: listwise objective probe (Closed)

- [x] add/enable listwise objective experiment path in ranking pipeline (예: `LambdaRank` 단일 설정)
- [x] run validation passes
  - `smoke_lambdarank_f07`: completed, blocked
  - **결과:** Listwise objective(LambdaRank) 적용 시 ranking collapse(coverage@10 미달) 완화에 실패. 정형 feature 한계 확인.

### Step 2: Text-Embedding Feature Integration (Phase 5-B Execution)
   - [ ] **Feature Toggle:** `include_text_embedding_feature` ablation 실행. (기본값 `false` 유지 보장). 추론(Inference) 시 이 플래그가 `false`이면 KURE feature 계산 로직 자체를 스킵하여 Shape 불일치 오류를 방지한다.
   - [ ] **Prerequisite:** 실험 전용 `mask_holdout_hobbies()` + `post_mask_leakage_audit()` 파이프라인 가동. 누수율(Leakage rate) 5% 초과 시 실험 즉시 `disabled`.
   - [ ] **[ACT] 플레이스홀더 강화:** `mask_holdout_hobbies()` 함수 내부 로직 수정. 단순 삭제가 아닌 문법 유지(예: "~한다" -> "[ACT]를 한다")를 위한 정규식/치환 룰셋 적용 후 임베딩 계산.
   - [ ] **Model:** `gnn_recommender/text_embedding.py` 내 `_load_kure_model()` 활용 (KRUE/KURE-v1). CUDA 우선 / CPU Fallback 및 입력 max_length=512 제한 적용.
   - [ ] **도메인 태깅:** 임베딩 전, 7개 도메인 프리픽스(`[PROF]...[FAM]`)를 결합하여 Context 강화된 단일 문장 생성.
   - [ ] **Feature 주입:** `persona_hobby_semantic_sim` (Cosine Similarity) Feature를 LightGBM 학습/추론에 추가.
   - [ ] **CLI 가이드:** `train_ranker.py` 실행 시, KURE 캐싱을 활성화하기 위해 `--use-kure-cache` 플래그를 추가할 수 있도록 인자 확장.
- [ ] **Validation Gate (diversity_probe 전용):**
  - delta_recall@10 >= -0.010 (단, -0.005 이하일 경우 qualitative review 필수)
  - delta_ndcg@10 >= -0.010
  - delta_coverage@10 >= +0.030 (Primary Target - 핵심 정복 지표)
- [ ] **Success Condition:** Recall 하락 최소화(-0.005 이내) 유지하되, Coverage@10 또는 Novelty@10이 +0.03 이상 개선 시 `diversity_probe` 승격 검토.
- [ ] **Failure Condition:** Recall@10 감소 폭 -0.010 초과 또는 누수 audit 실패 시 즉시 Roll-back (기존 운영 모델 유지).

### Step 3: Post-Hoc Analysis & Reporting
- [ ] 최종 실험 결과 `artifacts/experiments/phase5_text_embedding/` 경로에 저장.
- [ ] `validation_metrics.status.json`, `experiment_decisions.json` 업데이트 (상태: `promoted`, `experimental`, `disabled` 중 택 1).
- [ ] `PRD.md` 최종 Feature Policy 갱신 (승인된 설정만 상위 문서 반영).

---

## Overall Summary & Default Path

### 현재 운영 중인 Default 승인 경로 (Locked)
- **Stage 1:** `popularity + cooccurrence` (LightGCN XSimGCL 후보 생성)
- **Stage 2:** `LightGBM learned ranker` (정형 데이터 Feature 기반)
- **Feature Policy:**
  - `include_source_features=false` (권장, 안정성 우선)
  - **`include_text_embedding_feature=false` (기본값, Leakage 방지)**
- **MMR:** `false` (필요 시 `--use-mmr` flag에 의한 개별 실험용)
- **최종 모델:** `artifacts/experiments/phase2_5_num_leaves_31/ranker_model.txt`

### 변경 관리 규정
- Text Embedding(KURE) Feature는 `diversity_probe` 또는 `ablation` 목적의 특정 Run에서만 `true`로 일시 허용됨.
- 어떠한 변경이든 기본 하드 게이트(`recall@10` 손실 -0.002, `ndcg@10` 손실 -0.002)를 깨뜨릴 경우 승인 불가.
- 모든 텍스트 기반 실험은 `DATASET_EXPLAIN.md`의 29.3% 누수 사례를 반드시 `masking`으로 제거한 후 진행.
  - `smoke_lambdarank_f08`: completed, blocked
- [x] no winner passed gates → test pass skipped

### [PRE-REQUISITE] 50K Dataset & Fallback Policy Validation
> **Notice**: Before running Phase 5-B (KURE) experiments, the 2-Stage baseline MUST be stabilized on 50K dataset.
> 1. Re-run Phase 2.5 config (`num_leaves=31`, `neg_ratio=4`, `hard_ratio=0.8`) on **50K validation set** to confirm new base-line metrics.
> 2. Confirm `rare_item_policy=keep_with_fallback` does NOT break `candidate_recall@50` (> -0.01 delta).
> 3. If `coverage@10` of baseline is still < 0.20 after regularization, it justifies the need for KRUE embedding injection.

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

- [ ] **Pre-check**: Verify `data.py` `prepare_hobby_edges()` successfully runs with `rare_item_policy=keep_with_fallback` and generates valid `vocabulary_report.json` including `rare_items_count`.
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

## Phase 5-Pre: Rare Hobby Fallback Policy (KRUE 도입 전 필수)

> Goal: 모델의 편향(랭킹 붕괴) 해소 및 long-tail 추천 품질 확보. 희귀 취미(raw hobby)를 삭제(drop)하지 않고, parent canonical 또는 category로 백오프(fallback)하여 학습/추천에 활용한다.

### Policy Lock

- [x] `rare_item_policy=drop` 제거. 기본값을 `keep_with_fallback` 또는 `backoff_to_canonical_or_category`로 변경한다.
- [x] `raw_hobby`는 원본으로 반드시 보존하며, `canonical_hobby` 및 `category`로 백오프 매핑 테이블을 구축한다.
- [x] 학습(Backbone)과 추천(Display/Expansion)은 분리한다. LightGCN 등은 안정적인 백오프 item으로 학습하고, 최종 Top-K에는 raw hobby가 복원되어 노출되어야 한다.

### Implementation Tasks

- [x] `configs/lightgcn_hobby.yaml`에 `rare_item_policy: keep_with_fallback` 및 `fallback_order` 추가.
- [x] `gnn_recommender/data.py` 내 `prepare_hobby_edges()` 수정. `rare_item_policy != "drop"`에 대한 예외 처리 제거 및 fallback 로직 구축.
- [x] `raw_hobby_to_fallback_item.json` artifact 생성 (raw hobby -> canonical -> category 매핑 저장).
- [x] `vocabulary_report.json`에 `dropped_hobbies` 대신 `fallback_hobbies` 및 `fallback_edges` 통계 추가.

### Experiment Plan

- [x] 기존 `drop` 정책(`min_item_degree=3` 적용) Baseline 재측정.
- [x] 신규 `fallback` 정책 적용 후 지표 비교.
- [x] 비교 지표: `raw_hobbies`, `canonical_hobbies`, `retained_edges`, `candidate_recall@50`, `coverage@10`, `novelty@10`, `cold_start_recall@10`.

### Gate

- [x] fallback 적용 시 `candidate_recall@50` 하락폭이 `-0.01` 이내로 유지되어야 함.
- [x] `coverage@10` 또는 `novelty@10` 지표가 기존 drop 정책 대비 **반드시 개선**되어야 함.

### Artifacts

- [x] `artifacts/vocabulary_report.json`
- [x] `artifacts/raw_hobby_to_fallback_item.json`
- [x] `artifacts/fallback_policy_report.json`

## Phase 5-D: Cold-Start Evaluation Slice

> Goal: determine whether ranking-collapse mitigation helps users with sparse known hobbies, where persona text may carry the most value.

- [ ] define cold-start subset as `known_hobbies <= 1`
- [ ] add subset metric computation to ranker evaluation artifacts
- [ ] report cold-start recall@10 and ndcg@10
- [ ] report cold-start coverage@10, novelty@10, intra_list_diversity@10
- [ ] compare cold-start results for closed Phase 2.5 default, Phase 5-B candidates, and Phase 5-C text embedding ablation
   - [ ] record whether cold-start gains justify follow-up even if overall default-promotion gate fails

## 🚀 Phase 5-E: Text Embedding 구조화 및 [ACT] 플레이스홀더 룰셋 적용 (PRD 패치 반영)

### 배경
단순 취미 단어 매칭을 넘어선 'Persona 맥락(Context)'을 KURE 임베딩으로 계산하기 위한 핵심 전처리 규칙을 구체화합니다.

### 실행 태스크
- [ ] **도메인별 Prefix 태깅 스크립트 구현** (`src/embeddings/text_preprocessor.py` 또는 `data.py` 내부 함수 추가)
  - Persona 텍스트를 7개 도메인(`[PROF]`, `[SPORT]`, `[ART]`, `[TRAVEL]`, `[FOOD]`, `[FAM]`, `[CULT]`)으로 분리/태깅하여 단일 문장으로 결합.
  - 예: `f"[PROF] ... [SPORT] ... [ACT]를 즐깁니다"`
- [ ] **Leakage-safe [ACT] 플레이스홀더 로직 강화** (`gnn_recommender/text_embedding.py` 수정)
  - `mask_holdout_hobbies()` 함수 내부 수정: 단순 삭제가 아닌, 문법적 흐름 유지(예: "~한다" -> "[ACT]를 한다")을 위한 정규식 룰셋 적용.
  - 문장 끝의 불필요한 구두점/여백 정제 로직 추가.
- [ ] **KURE 임베딩 품질 검증 프로세스 정의**
  - [ACT] 플레이스홀더 적용 전/후의 Persona 벡터 간 코사인 유사도 모니터링 (하한선: 0.75 이상 유지).
  - 임베딩 차원 분산(Variance) 분석을 통한 '문맥 손실' 최소화 검증.
- [ ] **게이트 적용**
  - `post_mask_leakage_audit()` 시 [ACT] 태깅 상태를 함께 로깅.
  - PRD 제52항 준수: 마스킹/audit 실패 또는 문맥 왜곡 발생 시 해당 Run 즉시 `disabled` 처리.
--------------------------------------------------
# ?�� Phase 5-B/C: Text Embedding (KURE) ?�험 구현 가?�드
--------------------------------------------------

## ?�� ?�이�?규칙 (Naming Convention)
기존 2-Stage LightGBM ?�련(`train.py`)�? KURE ?�베??Feature�?추�??�는 ?�련??구분?�기 ?�한 규칙?�니??
- **?�이??로더 ?�이�?** `gnn_recommender/data_loader.py` (?�는 `dataset.py`)
  - ??��: ?�형 ?�이??+ `[PROF]...[FAM]` ?�깅???�스???�이?��? 결합?�여 최종 ?�습??`ranker_dataset.csv` ?�성 (KURE ?�처�?로직 ?�함).
- **모델 ?�이�?** `gnn_recommender/ranker.py` (?�는 `train.py`)
  - ??��: `data_loader.py`?�서 ?�성???�이?�셋??바탕?�로 ?�수 LightGBM 모델 ?�습 진행.

## ?���?[TASK] 구현 ?�행 ?�서 (Execution Pipeline)
- [ ] **1. ?�이??결합 �??�처�?(`data_loader.py` 로직 ?�성)**
  - `build_domain_tagged_persona_text()`�??�용??7�??�메???�스?�에 `[PROF]`, `[SPORT]` ??Prefix ?�깅.
  - `mask_holdout_hobbies()`�??�용???�보 취�? 마스??`[ACT]`).
  - `KUREEncoder.compute_similarity()`�??�해 Persona???�스??벡터 �??�보 취�??�???�사???�렬 ?�출.
  - `ranker_dataset.csv`???�스???�사??Feature 칼럼 결합.
- [ ] **2. LightGBM ?�학??(`ranker.py` 로직 ?�성)**
  - ?�성???�로???�이?�셋???�??기존 LightGBM ?�라미터�??�학???�행.
- [ ] **3. Gate 검�?�?결과 로깅 (`evaluate.py` ?�행)**
  - 기�?(Baseline) ?��?`Recall@10` ?�락 ???�인 (Gate 1: -0.005 ?�하 ?��?).
  - `Coverage@10` ?�승 ???�인 (Gate 2: +0.03 ?�상 ?�승 ?�요).
  - 결과???�라 `experiment_decisions.json` ?�태 `promoted` ?�는 `disabled` 기록.

--------------------------------------------------
