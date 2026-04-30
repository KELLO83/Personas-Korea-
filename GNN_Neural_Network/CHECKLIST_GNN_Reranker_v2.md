# GNN Reranker v2.6 구현 체크리스트 (A-단계형)

> `PRD_GNN_Reranker_v2.md` (v2.6) 기반 구현 진행 상황.  
> **중요**: `PRD.md`/`CHECKLIST.md`가 v2 실행을 포함한 기준 문서이며, 본 문서는 v2 실험 실행/체크만 위한 보조 문서입니다.  
> 전략: Learned Ranker + MMR + SHAP을 단계별로 잘라서 구현. 각 Phase마다 Go/No-Go Gate를 두고, 통과 시에만 다음 Phase로 진행.  
> 각 항목 완료 시 [x] 표시, 진행 중이면 [~] 표시.  
> 버전: v2.6 (2026-04-30). Phase 2 PROMOTED, Phase 2.5 진행 중 (ranking collapse 완화), Phase 3 NO-GO.  
> **핵심 진단**: 현재 병목은 retrieval 부족이 아니라 **ranking collapse** — candidate_recall@50=97.8%임에도 v2 LightGBM이 coverage@10=13.9%로 수축.  
>   
> **Current default path**:  
> ```
> Stage 1 = popularity + cooccurrence (candidate generation)  
> Stage 2 = LightGBM learned ranker (relevance scoring)  
> MMR     = off (default disabled, --use-mmr flag only)  
> ```

> ## 충돌 우선순위
> - 실무 의사결정/판단은 `GNN_Neural_Network/PRD.md` → `GNN_Neural_Network/CHECKLIST.md` → `README.md` 순으로 적용합니다.
> - 본 체크리스트는 실험 상태와 가동률을 기록하기 위한 보조 문서로서, 상위 문서와 충돌할 경우 상위 문서를 최우선으로 수정해 정합성을 유지합니다.

---

## Phase 0: Pre-Flight (반드시 먼저 수행)

> 목표: v1 baseline을 정확히 측정하고, Stage1 candidate pool 품질을 확인하여 Stage3 진행 가치를 판정.

### 0.1 v1 Baseline 측정
- [x] `evaluate_reranker.py --mode=deterministic` 실행하여 v1 전체 메트릭 측정 (기존 artifacts 활용)
  - [x] `recall@5,10,20` 기록 — **test: recall@10 = 0.7043**
  - [x] `ndcg@5,10,20` 기록 — **test: ndcg@10 = 0.4403**
  - [x] `hit_rate@5,10,20` 기록 — **test: hit_rate@10 = 0.7043**
  - [x] `coverage@5,10,20` 신규 메트릭 측정 및 기록 — **Phase 1에서 완료** (v1: coverage@10=0.517)
  - [x] `novelty@5,10,20` 신규 메트릭 측정 및 기록 — **Phase 1에서 완료** (v1: novelty@10=4.732)
  - [x] `intra_list_diversity@5,10,20` 신규 메트릭 측정 및 기록 — **Phase 1에서 완료** (v1: ILD@10=0.99)
- [x] v1 per-segment recall 측정 (age_group별, sex별) — **Phase 1에서 완료**
  - [x] 연령대별 recall@10 분포 기록 — **Phase 1에서 완료**
  - [x] 성별별 recall@10 분포 기록 — **Phase 1에서 완료**
  - [x] segment recall gap (max - min) 계산 — **Phase 1에서 완료**
- [x] 측정 결과를 `artifacts/v1_baseline_metrics.json`에 저장 — **Phase 1에서 생성 완료**

### 0.2 Stage1 Candidate Pool 품질 + Reranker 개선 여지 측정
- [x] **Pool Quality 측정**:
  - [x] `candidate_recall@50` 측정: test 정답이 Stage1 candidate pool에 포함된 비율 — **0.9771**
  - [x] `candidate_pool_coverage@50` 측정: 전체 hobby 중 candidate pool에 포함된 고유 hobby 비율 — **Phase 1에서 완료**
- [x] **Reranker Improvement Room 측정**:
  - [x] `oracle_recall@10` 측정: "candidate pool을 완벽히 정렬했을 때"의 theoretical upper bound — **0.9771**
  - [x] `v1_recall@10` 측정: 현재 v1 deterministic의 실제 recall@10 — **0.7043**
  - [x] 개선 여지: `improvement_room = 0.9771 - 0.7043 = 0.2728`
- [x] 결과 해석 (2개 gate 독립 평가):
  - [x] **Pool Quality Gate**: `0.9771 >= 0.70` → **PASS**
  - [x] **Reranker Improvement Room Gate**: `0.2728 >= 0.05` → **PASS**

### 0.3 데이터 스키마 확정
- [x] `ranker_dataset.csv` 스키마 확정 (PRD §4.2와 동일)
  ```csv
  person_id,candidate_hobby_id,split,label,lightgcn_score,cooccurrence_score,segment_popularity_score,known_hobby_compatibility,age_group_fit,occupation_fit,region_fit,popularity_prior,mismatch_penalty,popularity_penalty,novelty_bonus,category_diversity_reward,text_embedding_similarity,is_cold_start
  ```
  > **참고**: `source_popularity`, `source_cooccurrence`, `source_lightgcn` one-hot 특성은 초기 Phase 2에서 제외.
  > `source_scores` dict에 이미 provider 정보가 연속형 점수로 인코딩되어 있으므로,
  > LightGBM이 스스로 분할 학습할 수 있음. one-hot 특성은 Phase 2 ablation에서 추가 여부 결정.
- [x] `split` column values: `train`, `validation`, `test`
- [x] `label` definition: 1 if (person_id, candidate_hobby_id) is in held-out positive edges for that split, else 0
- [x] Negative sampling strategy 확정: **기본 ratio = 4:1** (negative:positive), Hard Negative + Easy Negative 혼합 (Mixed Negative Sampling).
  - Hard Negative: Stage1이 추천한 50개 후보 중 정답이 아닌 항목 (변별력 확보)
  - Easy Negative: 전체 catalog에서 랜덤 샘플링 (popularity bias 완화)
  - 혼합 비율은 config로 조정 가능 (기본 4:1 Hard-heavy)
  - 1:1은 너무 인위적이어서 음성 클래스 과소평가 위험
  - 자연 비율(약 1:9~1:11)에 가까운 4:1이 시작점으로 합리적
  - Ablation으로 1:1, 4:1, 8:1 비교
- [x] 데이터 크기 추정: validation split 기준 약 40K 엣지, 180개 취미. 학습 데이터는 추정 수만~수십만 rows (PRD 초안 "~50M rows"는 과대 추정이었음). **데이터 스케일이 작으므로 과적합 방지가 최우선**.

### 0.4 LightGBM Training Mode 확정
- [x] Objective: `binary` (pointwise binary classification)
- [x] Metric: `auc` (early stopping)
- [x] Hyperparameters 초기값 (보수적 설정, 데이터 스케일이 작으므로 과적합 방지 우선):
  ```python
  params = {
      'objective': 'binary',
      'metric': 'auc',
      'boosting_type': 'gbdt',
      'num_leaves': 15,
      'learning_rate': 0.05,
      'feature_fraction': 0.9,
      'bagging_fraction': 0.8,
      'bagging_freq': 5,
      'min_data_in_leaf': 50,
      'lambda_l1': 0.1,
      'lambda_l2': 0.1,
      'verbose': -1,
      'early_stopping_rounds': 50,
  }
  # Categorical Features 명시: ['candidate_hobby_id', 'age_group_fit', ...]
  ```

### 0.5 Text Embedding Policy 확정
- [x] 기본값: `use_text_embedding=false`
- [x] 마스킹 방식 확정: alias-aware + regex word-boundary
- [x] Post-mask leakage audit 절차 확정 (PRD §3.2.3 참고)
- [x] Promotion path에서 text feature 제외 (leakage audit 통과 후에만 ablation으로 평가)

### 0.6 Promotion Gate 수치 확정
- [x] Accuracy Gate: delta_recall@10 >= -0.002, delta_ndcg@10 >= +0.005
- [x] Diversity Gate: delta_coverage@10 >= 0, delta_novelty@10 >= 0, delta_intra_list_diversity@10 >= 0
- [x] Fairness Gate: segment recall gap regression <= 3%p
- [x] MMR은 optional이므로 accuracy gate 없이 diversity gate만 적용 가능 (사용자가 λ 선택)

### 0.7 Runtime Budget 확정
- [x] Feature generation: 수만~수십만 rows (과대 추정 수정), 목표 < 30분 (병렬화, 캐싱)
- [x] LightGBM 학습: < 10분 (CPU)
- [x] SHAP computation: < 5분 (batch, 1000 samples)
- [x] 전체 ablation 실행: < 1시간

---

### 🚦 Phase 0 Go/No-Go Gate — **PASS** ✅

> Phase 0 핵심 게이트는 모두 통과. Phase 1에서 미측정 항목(coverage, novelty, ILD, per-segment, candidate_pool_coverage)을 완료하여 v1_baseline_metrics.json 생성.

```
[GO]  아래 핵심 gate 조건 충족:

      1. Pool Quality Gate:
         candidate_recall@50 = 0.9771 >= 0.70
         → PASS (Stage1이 test 정답의 97.7%를 후보 pool에 담고 있음)

      2. Reranker Improvement Room Gate:
         oracle_recall@10 - v1_recall@10 = 0.9771 - 0.7043 = 0.2728 >= 0.05
         → PASS (learned ranker가 v1보다 최대 27%p 개선할 공간 있음)

      Phase 1에서 보완 완료:
         - coverage@5,10,20 / novelty@5,10,20 / intra_list_diversity@5,10,20 ✅
         - candidate_pool_coverage@50 ✅
         - per-segment recall (age_group, sex) ✅
         - v1_baseline_metrics.json 생성 완료 ✅

      → Phase 1 Foundation 진행 완료 → Phase 2 진행
```

---

## Phase 1: Foundation (1-2일)

> 목표: metrics 확장 및 learned ranker를 위한 인프라 구축.

### 1.1 의존성 추가
- [x] `requirements-gnn.txt`에 `lightgbm>=4.0.0` 추가
- [x] `requirements-gnn.txt`에 `shap>=0.45.0` 추가 (Phase 1에서 설치만, 실사용은 Phase 4)
- [x] `requirements-gnn.txt`에 `scikit-learn>=1.3.0` 추가
- [x] `.venv`에 설치 테스트 완료

### 1.2 평가 메트릭 확장 (`gnn_recommender/metrics.py`)
- [x] `catalog_coverage_at_k()` — 이미 구현됨, v1 baseline 측정 완료
- [x] `novelty_at_k()` — 이미 구현됨, v1 baseline 측정 완료
- [x] `intra_list_diversity_at_k()` — 구현 완료 (category 기반)
- [x] `oracle_recall_at_k()` — 구현 완료, 버그 수정 (pool[:k] → full pool + min cap)
- [x] `per_segment_metrics()` — 구현 완료 (age_group, sex별 recall)
- [x] `summarize_ranking_metrics()`에 신규 메트릭 통합 완료
- [x] `v1_baseline_metrics.json` 생성 완료

### 1.3 신규 Feature 추가 (`gnn_recommender/rerank.py`)
- [x] `is_cold_start` flag 추가 (`known_hobbies` <= 1)
- [x] `RerankerWeights` dataclass에 `is_cold_start: float = 0.0` 추가
- [x] `score_rerank_features()` 업데이트 (신규 feature 포함)
- [ ] `text_embedding_similarity` feature 추가 — **Phase 4에서 구현** (`--use-text-embedding` 시에만)
- [ ] `source_popularity`, `source_cooccurrence`, `source_lightgcn` one-hot — **Phase 2 ablation에서 결정** (초기 제외)

### 1.4 Leakage-Safe Text Embedding 유틸리티 — **Phase 4로 연기**
- [ ] `gnn_recommender/text_embedding.py` 신규 모듈 → Phase 4에서 구현
- [ ] `mask_holdout_hobbies()`, `post_mask_leakage_audit()` → Phase 4에서 구현
- [ ] `compute_text_embedding_similarity()` → Phase 4에서 구현
- [ ] 단위 테스트 → Phase 4에서 구현

### 1.5 KURE Embedding 캐싱 유틸리티 — **Phase 4로 연기**
- [ ] `gnn_recommender/embedding_cache.py` 신규 모듈 → Phase 4에서 구현
- [ ] `PersonEmbeddingCache`, `HobbyEmbeddingCache` → Phase 4에서 구현

---

### 🚦 Phase 1 Go/No-Go Gate — **PASS** ✅

```
[GO]  신규 메트릭(intra_list_diversity, oracle_recall, per_segment)이 v1 baseline에 대해 정상 측정됨
      AND 기존 메트릭(catalog_coverage, novelty) 검증 통과
      AND LightGBM 설치/동작 확인 (import + 간단한 fit/predict 테스트)
      → Phase 2 진행 완료

✅ 전체 188 tests passed (17 metrics + 23 ranker + 기존)
✅ v1_baseline_metrics.json 생성 완료
✅ LightGBM 4.6.0, SHAP 0.51.0, sklearn 1.8.0 설치 확인
```

---

## Phase 2: Learned Ranker Core (3-5일)

> 목표: LightGBM binary classifier만 구현. **SHAP은 아직 포함하지 않음.**  
> 핵심 질문: "v2 LightGBM이 v1 deterministic보다 accuracy가 나은가?"  
> **⚠️ Phase 2를 통과하지 못하면 Phase 3 이후는 진행하지 않음.**

### 2.1 Ranker 모듈 생성 (`gnn_recommender/ranker.py`)
- [x] `RankerDataset` dataclass + `RankerRow` dataclass
- [x] `build_ranker_dataset()` — val_edges 기반 데이터 구성, train_known masking
- [x] `sample_negatives()` — Hard+Easy Mixed 4:1 MNS (hard_ratio=0.8)
- [x] `LightGBMRanker` 클래스 — fit, predict, save, load, feature_importance

### 2.2 학습 스크립트 (`scripts/train_ranker.py`)
- [x] CLI argument parsing: --config, --output-dir, --neg-ratio, --hard-ratio, --seed, --num-boost-round, --early-stopping-rounds
- [x] val_edges 기반 데이터 로드 (train_edges 아님)
- [x] Stage1 candidate pool 생성 (popularity + cooccurrence)
- [x] Feature engineering: `build_rerank_features()` + leakage feature 제외
- [x] LightGBM 학습: best_iteration=122, best_auc=0.8893
- [x] 모델 저장: `artifacts/ranker_model.txt`, `artifacts/ranker_params.json`, `artifacts/ranker_feature_importance.json`

### 2.3 평가 스크립트 (`scripts/evaluate_ranker.py`)
- [x] v1 deterministic baseline 평가
- [x] v2 learned ranker 평가 (ranker.predict()로 relevance score 생성)
- [x] Delta vs v1 baseline (recall, ndcg, coverage, novelty, diversity)
- [x] Promotion gate 판정 (recall@10 >= -0.002, ndcg@10 >= +0.005)
- [x] 산출물: `artifacts/ranker_eval_metrics.json`

### 2.4 버그 수정 및 검증
- [x] oracle_recall_at_k 버그 수정 (pool[:k] → full pool + min(hits, k) cap)
- [x] 데이터 구성 버그 수정 (train_edges → val_edges 80/20 split)
- [x] to_numpy() empty dataset shape 버그 수정
- [x] to_lgb_dataset() .construct() 호출 추가
- [x] 전체 regression test 212 passed (188 기존 + 24 diversity)

---

### 🚦 Phase 2 Go/No-Go Gate (Accuracy Gate) — **PASS** ✅

```
[GO]  delta_recall@10 = +0.004 (>= -0.002) ✅
      delta_ndcg@10 = +0.007 (>= +0.005) ✅
      → Phase 3 진행 완료 (MMR)

Validation: delta_recall@10 = +0.020, delta_ndcg@10 = +0.013
Test: delta_recall@10 = +0.004, delta_ndcg@10 = +0.007
LightGBM AUC = 0.8893, best_iteration = 122

Diversity note: v2 LightGBM has lower coverage/novelty than v1 deterministic.
                Coverage@10: 0.150 (v1=0.517), Novelty@10: 4.580 (v1=4.732)
                → Phase 3 MMR 실험으로 diversity 개선 시도
```

---

## Phase 2.5: LightGBM Regularization Tuning + Negative Sampling Ablation + Source One-Hot

> **목표**: Phase 2 PROMOTED 모델의 ranking collapse 완화 및 feature balance 개선.  
> **핵심 진단**: 현재 병목은 **retrieval 부족이 아니라 ranking collapse**. candidate_recall@50=97.8%로 pool은 충분하나, v2 LightGBM이 top-k를 인기 취미로 수축시킴 (coverage@10: v1=51.7% → v2=13.9%).  
> **핵심 질문**: "Regularization, neg sampling, source feature가 ranking collapse를 완화하고 feature 기여 균형을 개선하는가?"  
> **Feature importance 현황**: cooccurrence_score(53.5%) + popularity_prior(37.5%) = 91% 독점. 나머지 11개 feature는 현재 설정에서 실질 기여가 작음 (무의미가 아님).

### 2.5-A. Regularization Tuning
- [ ] `scripts/tune_ranker_regularization.py` — Sequential greedy search
  - [ ] num_leaves=[7,15,31], min_data_in_leaf=[20,50,100], max_depth=[3,5,-1]
  - [ ] feature_fraction=[0.7,0.8,0.9,1.0], bagging_fraction=[0.7,0.8,1.0]
  - [ ] reg_alpha=[0.05,0.1,0.5,1.0], reg_lambda=[0.05,0.1,0.5,1.0]
  - [ ] **핵심**: feature_fraction 0.7~0.8로 cooccurrence/popularity 독점(91%) 완화
  - [ ] Validation recall@10 기준 best config 탐색 (최대 3 round greedy)
  - [ ] 산출물: `artifacts/regularization_tuning.json`, `artifacts/regularization_tuning_summary.md`

### 2.5-B. Negative Sampling Ablation
- [ ] `scripts/ablation_neg_sampling.py` — Phase 1 + Phase 2 ablation
  - [ ] Phase 1: neg_ratio [1,2,4,8] × hard_ratio=0.8 (기본값)
  - [ ] Phase 2: best neg_ratio × hard_ratio [0.5,0.8,1.0]
  - [ ] **핵심**: hard/easy 비율이 popularity bias에 미치는 영향 측정
  - [ ] accuracy + coverage@10 동시 평가
  - [ ] 산출물: `artifacts/neg_sampling_ablation.json`, `artifacts/neg_sampling_ablation_summary.md`

### 2.5-C. Source One-Hot Feature Ablation
- [ ] `gnn_recommender/rerank.py`에 source one-hot feature 3개 추가
  - [ ] `source_is_popularity` (0/1): 후보가 popularity provider에서 왔는지
  - [ ] `source_is_cooccurrence` (0/1): 후보가 cooccurrence provider에서 왔는지
  - [ ] `source_count` (1~2): 후보를 생성한 provider 수
  - [ ] `RANKER_FEATURE_COLUMNS` 및 `RANKER_CATEGORICAL_FEATURES` 업데이트
- [ ] Source one-hot 포함/미포함 비교 평가 (coverage@10 변화 측정)
- [ ] 산출물: `artifacts/source_onehot_ablation.json`
- [ ] **참고**: source_scores에 이미 연속형 점수로 정보가 인코딩되어 효과 제한적일 수 있음

---

### 🚦 Phase 2.5 Go/No-Go Gate (Stability + Diversity Gate)

```
[GO]  regularization tuning 또는 neg_ratio/hard_ratio 변경이
      test recall@10 >= -0.002 AND test ndcg@10 >= +0.005 (vs v1 deterministic) 유지
      AND coverage@10 개선 (v2 baseline 13.9% 대비 유의미 증가)
      AND 과적합 지표(val-test gap) 개선 또는 동등
      → best config을 새 기본값으로 승격

[NO-GO] 모든 설정이 baseline보다 불안정하거나 promotion gate 미충족
        → 기존 기본값(neg_ratio=4, hard_ratio=0.8) 유지
        → tuning 결과는 실험 기록으로만 보존
```

> **핵심 진단 요약**: 현재 v2 LightGBM의 core 문제는 **ranking collapse**.
> candidate_recall@50=97.8%로 Stage 1 pool은 충분하나,
> v2가 top-k를 인기 취미로 수축시켜 coverage@10이 v1의 51.7%에서 13.9%로 급감.
> candidate_pool_size 확장(50→100)은 1순위가 아님 — ranking collapse를 먼저 해결해야 함.

---

## Phase 3: MMR Diversity Reordering — **NO-GO** ❌

> 목표: Phase 2를 통과한 learned ranker 위에 MMR을 추가. **SHAP은 아직 포함하지 않음.**  
> 핵심 질문: "MMR이 accuracy를 크게 희생하지 않고 diversity를 개선하는가?"  
> **결론: NO-GO** — category one-hot embedding으로는 MMR이 diversity를 개선하지 못함.  
> 원인: one-hot similarity가 0 또는 1이어서 lambda sweep이 무의미. KURE dense embedding 필요.

### 3.1 MMR 모듈 (`gnn_recommender/diversity.py`) ✅
- [x] `compute_hobby_embeddings()` — **category one-hot fallback 구현 완료**, Phase 5에서 KURE 교체 예정
- [x] `mmr_rerank()` — Greedy MMR selection 구현 (PRD §5.1 공식 준수)
- [x] `compute_intra_list_diversity()` — Pairwise cosine similarity, `1 - mean_similarity` 반환
- [x] `mmr_rerank_with_hobbies()` — convenience wrapper

### 3.2 MMR Sweep 스크립트 (`scripts/sweep_mmr_lambda.py`) ✅
- [x] λ Grid Search: 0.1부터 0.9까지 0.1 단위 정밀 스윕
- [x] 각 λ에 대해 recall@10, ndcg@10, coverage@10, intra_list_diversity@10 측정
- [x] Pareto frontier 시각화 (`artifacts/mmr_pareto.png`)
- [x] 산출물: `artifacts/mmr_sweep.json`

### 3.3 평가 CLI 업데이트 ✅
- [x] `evaluate_ranker.py`에 `--use-mmr` 플래그 추가 (기본값 false)
- [x] `--mmr-lambda` 파라미터 추가 (기본값 0.7, range 0.0~1.0)

### 3.4 단위 테스트 ✅
- [x] `tests/test_diversity.py` — 24 tests (compute_hobby_embeddings, mmr_rerank, intra_list_diversity, _get_category)
- [x] `tests/test_ranker.py` — 23 tests (RankerDataset, sample_negatives, build_ranker_dataset, LightGBMRanker)

---

### 🚦 Phase 3 Go/No-Go Gate (Diversity Gate) — **NO-GO** ❌

```
[NO-GO] MMR with category one-hot embedding:
        - Accuracy gate: PASS (recall drop within tolerance)
        - Diversity metrics improved: 0 out of 3 (coverage DOWN, novelty DOWN, ILD marginal)
        - Lambda sweep (0.1-0.9): ALL lambda values produce IDENTICAL results to v2 baseline
        - Root cause: one-hot cosine similarity is 0 or 1, making MMR a no-op within same-category items
        
        → Learned ranker only (Phase 2 result) remains default
        → MMR available as --use-mmr flag (default: false)
        → KURE embedding integration (Phase 5 planned) needed for MMR re-evaluation
        
정식 기록: "MMR이 이 candidate pool에서는 diversity-accuracy trade-off를 해결하지 못함" 
            (category one-hot embedding의 한계, KURE dense embedding 필요)
```

---

## Phase 4: SHAP Explanation + Text Embedding (2-3일)

> 목표: Phase 2 PROMOTED learned ranker에 SHAP 기반 해석 추가. Phase 3 MMR은 NO-GO이므로 기본 경로는 no-MMR LightGBM 기준으로 진행. 동시에 Phase 1/3에서 연기된 Text Embedding utility 및 KURE embedding cache를 구현.  
> 핵심 질문: "SHAP reason이 90% 이상의 추천에 대해 의미 있게 생성되는가?"

### 4.1 SHAP 해석 모듈 (`gnn_recommender/ranker_explain.py`)
- [ ] `compute_shap_values(ranker, X_sample, feature_names)`:
  - [ ] `shap.TreeExplainer(ranker.model)` 사용
  - [ ] Batch 처리 (1000 samples 단위)
  - [ ] SHAP 값 반환 (numpy array)
- [ ] `generate_reason(person_context, candidate, shap_values, feature_names)`:
  - [ ] Positive SHAP feature 상위 3개 선택
  - [ ] Feature 값과 SHAP 값을 한국어 템플릿에 매핑
  - [ ] 예: "당신의 연령대(30대)에서 인기가 높고(age_group_fit), 기존 취미(등산)와 함께 자주 등장합니다(known_hobby_compatibility)."
  - [ ] **주의**: 템플릿에 "모델이 고려한 요인"이라고 명시, "실제 이유"라고 하지 않음

### 4.2 Reason 자동 검증
- [ ] `validate_reason_batch(recommendations, shap_values)`:
  - [ ] 90% 이상의 추천에서 non-empty reason 생성 확인
  - [ ] 각 reason이 NaN feature를 참조하지 않음 확인
  - [ ] 각 reason이 masked hold-out hobby를 참조하지 않음 확인
  - [ ] 결과를 `artifacts/reason_quality.json`에 저장

### 4.3 추천 CLI 업데이트
- [ ] `recommend_for_persona.py`에 `--explain` 플래그 추가
  - [ ] True: SHAP 기반 reason 출력
  - [ ] False: 기존 출력 (reason 없음)

### 4.4 Leakage-Safe Text Embedding Utility (Phase 1에서 연기됨 → Phase 4에서 구현)
- [ ] `gnn_recommender/text_embedding.py` — 신규 모듈
  - [ ] `mask_holdout_hobbies()` — alias-aware + regex word-boundary masking
  - [ ] `post_mask_leakage_audit()` — 마스킹 후 잔존 leakage 검사
  - [ ] `compute_text_embedding_similarity()` — KURE-v1 기반 cosine similarity
- [ ] 단위 테스트 (`tests/test_text_embedding.py`)

### 4.5 KURE Embedding Cache (Phase 1에서 연기됨 → Phase 4에서 구현)
- [ ] `gnn_recommender/embedding_cache.py` — 신규 모듈
  - [ ] `PersonEmbeddingCache` — persona text embedding 캐시
  - [ ] `HobbyEmbeddingCache` — hobby name embedding 캐시
- [ ] Phase 3 MMR의 `compute_hobby_embeddings()`를 KURE embedding으로 교체 가능하도록 인터페이스 확인

---

### 🚦 Phase 4 Go/No-Go Gate (Explanation Gate)

```
[GO]  자동 검증 기준 통과:
      - 90% 이상의 추천에서 non-empty reason 생성
      - reason에 NaN/masked hobby 미포함
      → Phase 5 진행 (Integration)

[NO-GO] 자동 검증 기준 미달
        → 템플릿/SHAP 계산 로직 수정 후 재시도 (1회)
        → 여전히 실패 시:
            - SHAP은 --explain flag로만 제공 (기본값 false)
            - Phase 5로 진행 (SHAP 없이 learned+MMR만 제공)
```

---

## Phase 5: Integration & Final Evaluation (2일)

> 목표: 모든 컴포넌트 통합 및 최종 ablation 실행.

### 5.1 전체 CLI 통합
- [ ] `recommend_for_persona.py`에 모든 플래그 통합:
  - [ ] `--use-learned-ranker`
  - [ ] `--use-mmr`
  - [ ] `--mmr-lambda` (기본값 0.7)
  - [ ] `--explain`
  - [ ] `--use-text-embedding` (optional)
- [ ] `evaluate_reranker.py`에 모든 모드 통합:
  - [ ] `deterministic` (기존 v1)
  - [ ] `learned` (v2 LightGBM only)
  - [ ] `learned+mmr` (v2 + MMR)

### 5.2 전체 Ablation 실행
- [ ] **Ablation 1**: v1 deterministic baseline (기존 그대로)
- [ ] **Ablation 2**: v2 LightGBM only
- [ ] **Ablation 3**: v2 LightGBM + MMR (λ=0.7)
- [ ] **Ablation 4**: MMR λ sweep 결과 분석 (Pareto curve)
- [ ] **Ablation 5**: v2 LightGBM + MMR + text embedding (leakage audit 통과 후)
- [ ] **Ablation 6**: Cold-start users only (known hobbies <= 1)
- [ ] **Ablation 7**: Dense users only (known hobbies >= 5)

### 5.3 결과 기록
- [ ] `experiment_decisions.json` 업데이트:
  - [ ] 각 ablation의 accept/reject/promote 상태 기록
  - [ ] 실패/저하 원인 기록 (degraded recall, toxic in ablation 등)
  - [ ] Selected baseline 선언 (v1 또는 v2 중 최종 선택)
- [ ] `experiment_run_summary.md` 업데이트:
  - [ ] Human-readable summary
  - [ ] Pareto curve 시각화 결과 포함
  - [ ] v1 vs v2 metric comparison table

### 5.4 문서화
- [ ] `GNN_Neural_Network/README.md` 업데이트:
  - [ ] v2 아키텍처 다이어그램 추가
  - [ ] Learned ranker 사용법 (`--use-learned-ranker`)
  - [ ] MMR 사용법 (`--use-mmr`, `--mmr-lambda`)
  - [ ] SHAP explanation 사용법 (`--explain`)
  - [ ] Text embedding 사용법 (`--use-text-embedding`, leakage 주의)
- [ ] `PRD_GNN_Reranker_v2.md` 상태 업데이트 (완료된 항목 체크)
- [ ] `CHANGELOG.md`에 v2 추가

---

### 🚦 Phase 5 Go/No-Go Gate (Final Gate)

```
[GO]  모든 ablation 결과가 promotion gate 기준 충족:
      - Accuracy Gate: delta_recall@10 >= -0.002, delta_ndcg@10 >= +0.005
      - Diversity Gate: delta_coverage/novelty/diversity >= 0 (최소 2개)
      - Fairness Gate: segment recall gap regression <= 3%p
      AND 기존 v1 테스트 전체 통과 (regression 없음)
      → v2를 promoted default로 승격

[NO-GO] promotion gate 미충족 OR regression 발견
        → v1 deterministic을 계속 default로 유지
        → v2는 --use-learned-ranker flag로만 제공
        → experiment_decisions.json에 "v2 promotion 보류" 기록
```

---

## Phase 6: Regression Test (항상 수행)

### 6.1 기존 기능 보존
- [ ] `train_lightgcn.py`가 기존과 동일하게 동작하는지 확인
- [ ] `evaluate_lightgcn.py`가 기존과 동일하게 동작하는지 확인
- [ ] `evaluate_stage1_ablation.py`가 기존과 동일하게 동작하는지 확인
- [ ] `evaluate_reranker.py --mode=deterministic`이 v1과 동일한 결과를 내는지 확인
  - [ ] Delta == 0 (floating point tolerance 1e-6)

### 6.2 테스트 자동화
- [x] `tests/test_ranker.py` — RankerDataset, LightGBMRanker 단위 테스트 (23 tests)
- [x] `tests/test_diversity.py` — MMR, intra-list diversity 단위 테스트 (24 tests)
- [x] `tests/test_metrics_extended.py` — coverage, novelty, diversity, oracle_recall, segment metrics (17 tests)
- [ ] `tests/test_text_embedding.py` — Leakage-safe embedding 단위 테스트 (Phase 4)
- [ ] `tests/test_integration.py` — End-to-end: prepare → train ranker → evaluate → recommend

---

## 검수 게이트 (Go/No-Go Criteria)

v2 코드를 main 브랜치에 병합하기 전에 반드시 충족해야 할 조건:

| 조건 | 기준 | 검증 방법 |
|:---|:---|:---|
| **성능** | v2가 v1 대비 recall@10 delta >= -0.002, ndcg@10 delta >= +0.005 | `evaluate_ranker.py --mode=learned` vs `--mode=deterministic` |
| **Coverage** | delta_coverage@10 >= 0 | `ranker_metrics.json` |
| **Novelty** | delta_novelty@10 >= 0 | `ranker_metrics.json` |
| **Diversity** | delta_intra_list_diversity@10 >= 0 | `ranker_metrics.json` |
| **Fairness** | segment recall gap regression <= 3%p | `ranker_metrics.json` |
| **리그레션** | 기존 `train_lightgcn.py`, `evaluate_lightgcn.py`, `evaluate_reranker.py --mode=deterministic` 정상 동작 | 기존 테스트 전체 통과 |
| **해석** | SHAP 기반 reason이 90% 이상의 추천에 대해 non-empty 문자열 생성 | `tests/test_ranker_explain.py` 자동 검증 |
| **누수** | Text embedding이 post_mask_leakage_audit 통과 | `tests/test_text_embedding.py` |
| **Pool Quality** | candidate_recall@50 >= 0.70 (Stage1이 충분한 후보 제공) | `evaluate_stage1_ablation.py` |
| **Improvement Room** | oracle_recall@10 - v1_recall@10 >= 0.05 (learned ranker 도입 가치 있음) | `metrics.py` |

> **주의**: 기존 "manual check 100개 샘플" gate는 제거되었습니다. 모든 gate는 자동화 테스트로 검증 가능해야 합니다.

---

## 변경 이력

| 날짜 | 버전 | 변경 내용 |
|:---|:---|:---|
| 2026-04-29 | v2.0 | 초안 작성 |
| 2026-04-29 | v2.1 | Metis 검수 반영: Phase 0 Pre-Flight 추가, promotion gate delta 기준 추가, LightGBM binary classifier 명시, ranker row schema 추가, text embedding leakage 정책 강화, MMR 공식 수정, coverage/novelty target 상대 개선으로 변경, XGBoost/DPP 제거, manual QA gate 자동화로 대체, KURE embedding caching 추가 |
| 2026-04-29 | v2.2 | A-단계형 실행 방식 반영: Phase별 Go/No-Go Gate 추가, Phase 2 (LightGBM only) / Phase 3 (MMR) / Phase 4 (SHAP) 순차 분리, 각 Phase 성공/실패 시 다음 행동 명시, Phase 0 기준 재정의 (Pool Quality / Improvement Room 분리) |
| 2026-04-29 | v2.3 | 검토 피드백 반영: Phase 1 범위 축소 (text embedding/cache → Phase 4 연기), negative sampling 기본 4:1 + ablation, source_* one-hot → Phase 2 ablation으로 이연, 데이터 스케일 과대 추정 수정, LightGBM regularization 보수적 설정 (num_leaves 15, min_data_in_leaf 50, L1/L2 regularization), Phase 2 통과 후에만 Phase 3 진행 명시 강화 |
| 2026-04-29 | v2.4 | 최종 교차 검증(Oracle/Explore/Librarian) 반영: negative sampling 자가모순 해소 (Hard Only → Hard+Easy 4:1 MNS 통일), Phase 4에 Text Embedding+KURE cache 구현 항목 추가 (§4.4, §4.5), Phase 3 MMR embedding fallback 정책 명시 (category one-hot → Phase 5 KURE 교체), 버전 헤더 v2.4 통일 |
| 2026-04-30 | v2.5 | Phase 2 PROMOTED (validation recall@10 +0.020, ndcg@10 +0.013; test recall@10 +0.004, ndcg@10 +0.007). Phase 3 MMR NO-GO (category one-hot embedding으로 diversity 개선 불가, lambda sweep 무효). oracle_recall_at_k 버그 수정. tests 212 passed (188 + 24 diversity). experiment_decisions.json + experiment_run_summary.md 업데이트. |
| 2026-04-30 | v2.6 | Phase 2.5 추가: LightGBM regularization tuning (tune_ranker_regularization.py) + negative sampling ablation (ablation_neg_sampling.py) + source one-hot ablation. 핵심 진단 정정: 병목은 retrieval 부족이 아니라 **ranking collapse** (v2 LightGBM이 top-k를 인기 취미로 수축, coverage@10 51.7%→13.9%). feature importance 91% 독집은 "무의미"가 아니라 "현재 설정에서 실질 기여가 작다"로 해석. source one-hot 3순위로 편입. PRD/체크리스트/실험 결정 문서에 반영. |

---

**작성일**: 2026-04-30  
**버전**: v2.6 (A-단계형)  
**상태**: Phase 2 PROMOTED, Phase 2.5 진행 중 (ranking collapse 완화), Phase 3 NO-GO (MMR one-hot 한계)  
**다음 단계**: Regularization tuning → Neg sampling ablation → Source one-hot ablation → KURE hobby embedding
