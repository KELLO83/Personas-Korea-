# PRD: GNN Hobby Recommendation Engine v2.6

> **Product Requirements Document** for `GNN_Neural_Network` offline recommendation PoC.  
> 본 문서는 LightGCN 기반 2-stage 추천 엔진의 개선 계획을 정의합니다.  
> 기존 v1.0 deterministic baseline을 유지하며, learned ranker 및 diversity layer를 추가합니다.  
> **위치 및 적용 범위**: 본 문서는 `GNN_Neural_Network/PRD.md`/`GNN_Neural_Network/TASKS.md`의 기준을 보완하는 v2 실험용 PRD이며, 기본 요구사항/체크 규칙의 대체 문서가 아닙니다.  
> 실행 방식: A-단계형 (Phase 0~5 순차 진행, 각 Phase마다 Go/No-Go Gate).  
> **v2.6 상태**: Phase 2 PROMOTED (LightGBM), Phase 2.5 진행 중 (ranking collapse 완화), Phase 3 NO-GO (MMR + category one-hot).  
> **핵심 진단**: 현재 병목은 retrieval 부족이 아니라 **ranking collapse** — v2 LightGBM이 top-k를 인기 취미로 수축시킴 (coverage@10: v1=51.7% → v2=15.56%).

> ## 문서 역할 및 충돌 우선순위
> - 이 문서는 `GNN_Neural_Network/PRD.md`와 `GNN_Neural_Network/TASKS.md`를 보완하는 실험 전용 PRD입니다.
> - 실무 의사결정은 `PRD.md`(요구사항/결정) → `TASKS.md`(진행 게이트) 순으로 적용합니다.
> - 본 문서의 단계 게이트/실험 기록은 해당 순위를 대체할 수 없으며, 충돌 시 즉시 상위 문서를 우선 반영합니다.

---

## 1. 개요 (Executive Summary)

### 1.1 목표
현재 `GNN_Neural_Network`의 deterministic 2-stage 추천 구조를, 학습된 랭커(learned ranker) 기반 구조로 고도화하여 추천 정확도(Recall/NDCG)와 리스트 품질(Diversity/Novelty)을 동시에 향상시킵니다.

### 1.2 현재 상태 (v2.6 — Phase 2 PROMOTED, Phase 2.5 진행 중, Phase 3 NO-GO)

| 항목 | 현재 값 | 비고 |
|:---|:---|:---|
| Stage1 | `popularity + cooccurrence` | LightGCN 및 6개 대안 provider는 ablation에서 baseline 미달 |
| Stage2 | **LightGBM learned ranker (promoted)** | v1 deterministic reranker는 fallback으로 유지. AUC=0.8890555966387075, best_iteration=84 |
| Text Feature | 비활성 (`use_text_fit=false`) | Leakage 이슈로 인해 persona text 미사용. Phase 4에서 leakage audit 후 도입 예정 |
| Diversity | `category_diversity_reward=0`, MMR **NO-GO** | Phase 3에서 category one-hot MMR 실험 → diversity gate 미충족 (0/3 metrics improved). `--use-mmr` 플래그로만 제공 (기본값 false) |
| 평가 지표 | Recall@K, NDCG@K, Hit Rate@K, **Coverage@K, Novelty@K, ILD@K, oracle_recall@K, per-segment** | Phase 1에서 확장 완료 |
| 최종 산출물 | `ranker_eval_metrics.json`, `experiment_decisions.json`, `experiment_run_summary.md` | LightGBM promotion gate 통과 기록 |

**v2.6 핵심 메트릭**:

| 지표 | Stage 1 (pop+cooc) | v1 Deterministic | v2 LightGBM | 비고 |
|:---|:---|:---|:---|:---|
| recall@10 (test) | 0.6909 | 0.7043 | **0.7097** | v2 > v1 > Stage1 |
| ndcg@10 (test) | 0.4376 | 0.4403 | **0.4477** | v2 > v1 > Stage1 |
| coverage@10 (val) | 0.128 | **0.517** | 0.1556 | **v1이 더 높음 (ranking collapse)** |
| novelty@10 (val) | 4.48 | **4.73** | 4.5843 | v1이 더 높음 |
| ILD@10 (val) | 1.00 | 0.99 | 0.99 | 유사 |
| candidate_recall@50 | 0.977 | — | 0.978 | Pool은 충분 (retrieval 병목 아님) |

> **핵심 진단**: v2 LightGBM은 accuracy를 개선했지만 **ranking collapse** 문제가 있음. candidate_recall@50=97.8%로 Stage 1 pool은 충분하나, v2가 top-k를 인기 취미로 수축시켜 coverage@10이 v1(51.7%)에서 15.56%로 급감. 동일 pool에서 v1은 51.7%를 달성했으므로, 문제는 pool 크기가 아니라 ranking objective/feature balance임.

### 1.3 v2 목표 지표

| 지표 | v1 Baseline | v2 Target | 측정 방법 |
|:---|:---|:---|:---|
| `recall@10` | 현재값 | **delta >= -0.002** | `evaluate_reranker.py` |
| `ndcg@10` | 현재값 | **delta >= +0.005** | `evaluate_reranker.py` |
| `candidate_recall@50` | 현재값 | **변화 없음** | `evaluate_stage1_ablation.py` |
| `coverage@10` | **0.517** (v1 deterministic) | **delta >= 0** | `metrics.py` 확장 |
| `novelty@10` | **4.732** (v1 deterministic) | **delta >= 0** | `metrics.py` 확장 |
| `intra_list_diversity@10` | **0.990** (v1 deterministic) | **delta >= 0** | `metrics.py` 확장 |
| `segment recall gap` | **age_group=5.5%p, sex=2.2%p** (v1) | **regression <= 3%p** | Per-segment evaluation 완료 |

> **원칙**: v2는 v1 대비 **accuracy를 크게 희생하지 않으면서 diversity/novelty를 개선**하는 것이 목표입니다.

### 1.4 핵심 전제 (Scope Boundary)

- 본 프로젝트는 **synthetic 데이터 기반 offline PoC**입니다.
- 성공의 기준은 **held-out edge prediction 개선**이며, 실제 사용자 선호도 일반화를 증명하는 것은 아닙니다.
- v2 LightGBM은 **promoted default**이며, v1 deterministic은 `--rerank` 플래그로 fallback 가능합니다.

---

## 2. 아키텍처 개요

### 2.1 전체 파이프라인

```
[Raw Data]
  person_hobby_edges.csv + person_context.csv
      │
      ▼
[Canonicalization & Split]  (기존과 동일)
  train/validation/test edges + hobby_profile.json
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Candidate Generation (변경 없음)                    │
│   → popularity provider                                     │
│   → cooccurrence provider                                   │
│   → LightGCN provider (auxiliary, optional)                 │
│   → candidate pool = 50 ~ 100개                             │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 2a: Feature Engineering (확장)                         │
│   → 기존 14개 demographic/compatibility feature             │
│     (이 중 similar_person_score, persona_text_fit는         │
│      Ranker Row Schema에서 제외/대체)                        │
│   → (▲) Candidate source one-hot — Phase 2.5 ablation 대상  │
│   → (+) Cold-start segment flag                             │
│   → (◇) Leakage-safe text embedding similarity — Phase 4, 기본 비활성 │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ ★ Stage 2b: Learned Ranker — PROMOTED ✅                    │
│   → LightGBM binary classifier (AUC=0.8890556)                │
│   → 입력: Stage2 feature vector per (person, candidate)     │
│   → 출력: predicted relevance score                         │
│   → 학습: pointwise binary cross-entropy (label 1/0)        │
│   → 평가: ranking metrics grouped by person                 │
│   → 해석: SHAP 값으로 feature importance 확인               │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 2c: Diversity Reordering — NO-GO ❌ (optional)         │
│   → MMR (Maximal Marginal Relevance)                        │
│   → relevance-diversity trade-off 파라미터 λ                 │
│   → ℹ category one-hot similarity → binary cosine → no-op (재정렬 불가) │
│   → Phase 5에서 KURE dense embedding으로 재평가 예정         │
└─────────────────────────────────────────────────────────────┘
      │
      ▼
[Final Top-K Output]
  hobby_name + predicted_score + reason (SHAP-based) + display_examples
```

> 주의: 이 문서의 `Stage 2a/2b/2c` 표기는 **2-stage 추천 구조 내부의 세부 단계**를 설명하기 위한 것이다. 상위 아키텍처 관점에서는 여전히 `Stage 1 = candidate generation`, `Stage 2 = ranking/re-ranking`으로 해석한다.

### 2.2 핵심 설계 결정

| 결정 사항 | 선택 | 근거 |
|:---|:---|:---|
| Stage1 변경 여부 | **유지** | popularity+cooccurrence가 이미 강한 baseline. 변경 시 regression risk. |
| Learned Ranker 종류 | **LightGBM binary classifier** (1순위) | PoC에 가장 적합, SHAP 해석 가능, 빠른 학습. 향후 실험(LambdaRank) 검토 가능. |
| 학습 데이터 구성 | pointwise binary classification | Implicit feedback에 적합. label=1 (held-out positive), label=0 (Hard Negative + Easy Negative 혼합, 4:1 비율). Hard Negative = Stage1이 추천한 50개 중 정답이 아닌 항목, Easy Negative = 전체 catalog에서 랜덤 샘플링. Hard-heavy 혼합으로 변별력 확보와 popularity bias 완화를 동시 달성 (산업 표준 Mixed Negative Sampling). |
| Text Embedding | **Leakage-safe**: alias-aware masking 후 KURE-v1 sentence embedding. **기본 비활성**, `--use-text-embedding` 플래그로만 활성화. | 마스킹 후 추가 leakage audit 통과 시에만 사용. |
| Diversity 알고리즘 | **MMR with dense embeddings** (재평가 후보, 현재 default disabled) | Phase 3에서 category one-hot MMR → NO-GO (binary cosine similarity). KURE dense embedding으로 재평가 예정 (Phase 5). `--use-mmr` 플래그로만 제공 (기본값 false). |
| Baseline 대비 평가 | 반드시 동일 split, 동일 candidate pool, 동일 known-hobby masking | 공정한 비교를 위해 v1과 동일한 평가 조건 고정. |

### 2.3 고려했던 대안 경로 (Historical — Not Taken)

> 이 경로는 v2 설계 초기에 검토되었으나, 실제로는 **LightGBM learned ranker를 먼저 구현하는 방향**으로 진행되었습니다.

```
Step 1: v1에 coverage/novelty/diversity 메트릭 추가 → 측정
Step 2: deterministic Stage2 가중치를 grid search / logistic regression으로 자동 탐색
Step 3: MMR만 v1 위에 추가 → diversity 개선 확인
Step 4: 여전히 plateau면 그때 LightGBM 추가
```

**결과**: Step 1~2는 Phase 1/2에서 일부 반영되었으나, deterministic weight 탐색 대신 **LightGBM binary classifier로 직접 전환**하는 것이 더 효과적이라고 판단되어 현재의 A-단계형 접근(Phase 0→2→2.5→3)으로 대첸되었습니다.

---

## 3. Stage 2: Feature Engineering (확장)

### 3.1 기존 Feature 유지 (v1과 동일)

`rerank.py:144-167`의 모든 feature는 그대로 유지 (코드상 14개):

- `lightgcn_score`, `cooccurrence_score`, `segment_popularity_score`
- `known_hobby_compatibility`
- `age_group_fit`, `occupation_fit`, `region_fit`
- `popularity_prior`, `mismatch_penalty`
- `popularity_penalty`, `novelty_bonus`, `category_diversity_reward`
- `similar_person_score` — 현재 항상 0 (유사 페르소나 데이터 미구축). Ranker Row Schema에서 **제외**.
- `persona_text_fit` — leakage 이슈로 비활성. Ranker Row Schema에서 `text_embedding_similarity`로 **대체** (Phase 4).

### 3.2 신규 Feature

#### 3.2.1 Candidate Source One-Hot

**목적**: 어떤 Stage1 provider가 해당 candidate를 생성했는지를 explicit하게 알려주어, ranker가 source별로 다른 가중치를 학습하도록 함.

**출력 feature**: `source_popularity`, `source_cooccurrence`, `source_lightgcn` (0/1)

**주의**: 동일 hobby가 여러 provider에서 생성되면 해당 source 플래그를 모두 1로 설정.

#### 3.2.2 Cold-Start Segment Flag

**목적**: Known hobby가 매우 적은 사용자(0~1개)는 다른 패턴을 보일 수 있음.

**출력 feature**: `is_cold_start` (1 if len(known_hobbies) <= 1 else 0)

#### 3.2.3 Leakage-Safe Text Embedding Similarity (Optional)

**문제**: `persona_text`에 "산책" 등의 hobby 이름이 직접 포함되어 있어, text match는 information leakage.

**해결 (강화된 정책)**:
1. `build_leakage_audit()`에서 추적 중인 hold-out hobby 목록 로드.
2. **alias-aware masking**: canonical hobby뿐만 아니라 `hobby_taxonomy.json`의 `include_keywords`에 포함된 모든 variant를 마스킹.
3. **regex word-boundary masking**: 단순 `str.replace`가 아닌 `\bkeyword\b` 패턴으로 치환.
4. `persona_text`, `hobbies_text`, `embedding_text` 등에서 `[MASK]` 토큰으로 치환.
5. KURE-v1 모델로 masking된 text의 sentence embedding 생성.
6. Candidate hobby의 canonical name embedding과 cosine similarity 계산.

**마스킹 후 추가 leakage audit**:
```python
# 마스킹된 텍스트에서 여전히 hold-out hobby 또는 alias가 남아있는지 검사
def post_mask_leakage_audit(masked_text, holdout_hobbies, alias_map):
    normalized = normalize_hobby_name(masked_text)
    for hobby in holdout_hobbies:
        if hobby in normalized:
            return False  # leakage 존재
    return True
```

**출력 feature**: `text_embedding_similarity` (float, 0~1)

**활성화 정책**: 기본값 `false`. `--use-text-embedding` 플래그로만 활성화. promotion path에 포함되지 않음.

---

## 4. Stage 3: Learned Ranker

### 4.1 모델 선택

| 모델 | 장점 | 단점 | 우선순위 |
|:---|:---|:---|:---:|
| **LightGBM binary classifier** | 빠름, SHAP 해석 가능, overfitting 적음, 구현 단순 | 비선형 상호작용 limited | 1 |
| LightGBM LambdaRank | ranking에 특화 | group 관리 복잡 | 2 (v2.2 이후 실험) |
| XGBoost | LightGBM과 유사 | 학습 속도 약간 느림, 추가 의존성 | 3 (v2.2 이후 실험) |
| Small DNN | Flexible | 해석 어려움, hyperparameter 많음 | 4 (v2.2 이후 실험) |

### 4.2 학습 데이터 구성 (Pointwise Binary Classification)

**Ranker Row Schema** (필수 정의):

```csv
person_id,candidate_hobby_id,split,label,lightgcn_score,cooccurrence_score,segment_popularity_score,known_hobby_compatibility,age_group_fit,occupation_fit,region_fit,popularity_prior,mismatch_penalty,popularity_penalty,novelty_bonus,category_diversity_reward,text_embedding_similarity,is_cold_start
```

> **참고**: `source_popularity`, `source_cooccurrence`, `source_lightgcn` one-hot 특성은 초기 Phase 2에서 제외.
> `source_scores` dict에 이미 provider 정보가 연속형 점수로 인코딩되어 있으므로,
> LightGBM이 스스로 분할 학습할 수 있음. one-hot 특성은 Phase 2 ablation 실험에서 추가 여부를 결정함.

**Positive samples**:
- Train split의 모든 `(person_id, hobby_id)` edge.
- Feature: Stage2 feature vector.
- Label: 1

**Negative samples**:
- Stage1 candidate pool 근처에서 샘플링 (hard negative) + 전체 catalog random (easy negative) 혼합.
- **Negative:Positive ratio = 4:1** (기본값). Ablation으로 1:1, 4:1, 8:1 비교.
  - 1:1은 너무 인위적이어서 음성 클래스 과소평가 위험.
  - 자연 비율(약 1:9~1:11)에 가까운 4:1이 시작점으로 합리적.
- 기존 `sample_negative()` 함수 재활용.
- Label: 0

**학습/검증 분할**:
- Training: train split의 positive + negative pairs.
- Validation: validation split의 positive + negative pairs.
- Early stopping: validation AUC 또는 validation NDCG@10.

### 4.3 학습 / 검증 절차

```
1. train_lightgcn.py --prepare-only 실행
   → train_edges, validation_edges, test_edges, hobby_profile, person_context 생성

2. Stage1 candidate generation (popularity + cooccurrence)
   → train/validation/test 사용자별 candidate pool 50개

3. Feature engineering per (person, candidate)
   → ranker_dataset.csv 생성 (Phase 4.2 스키마 준수)

4. LightGBM binary classifier 학습
   → train(X_train, y_train), early stopping on validation AUC

5. Evaluation
   → test split에서 Recall@K, NDCG@K vs v1 deterministic baseline 비교
   → 동일 split, 동일 candidate pool, 동일 known-hobby masking
```

### 4.4 해석 가능성 (Interpretability)

- **SHAP 값**: 각 feature가 해당 (person, candidate) 쌍의 점수에 미친 영향.
- **Feature importance plot**: 전체 데이터셋에서 어떤 feature가 가장 중요한지.
- **Reason generation**: SHAP 값을 바탕으로 "이 취미가 추천된 이유"를 자동 생성.
  - 예: "당신의 연령대(30대)에서 인기가 높고(age_group_fit=0.8), 기존 취미(등산)와 함께 자주 등장합니다(known_hobby_compatibility=0.6)."
  - **주의**: SHAP reason은 "모델이 사용한 factor"이지 "실제 사용자가 이 취미를 좋아하는 causal reason"은 아닙니다.

---

## 5. Stage 4: Diversity Reordering (MMR)

### 5.1 MMR (Maximal Marginal Relevance)

**목적**: relevance와 diversity를 동시에 최적화.

**Grid Search 필수**: λ 파라미터를 0.1부터 0.9까지 0.1 단위로 탐색하여 정확도 하락(Recall/NDCG drop)을 최소화하면서 글로벌 다양성(Coverage)이 상승하는 최적의 Sweet-spot 도출.

**공식** (수정됨):
```
MMR(i) = λ * relevance(i) - (1 - λ) * max_similarity_to_selected(i)
```

- `relevance(i)`: Stage3 learned ranker 출력값 (또는 v1 deterministic score).
- `max_similarity_to_selected(i)`: 이미 선택된 hobby 중 i와 가장 유사한 hobby와의 cosine similarity.
- `similarity(i, j)`: hobby i와 j의 embedding cosine similarity (KURE-v1 기반, 또는 category one-hot 기반 fallback).
- `λ`: trade-off 파라미터 (0.5 ~ 0.9 실험).

**구현**:
```python
def mmr_rerank(candidates, relevance_scores, embeddings, lambda_param=0.7, top_k=10):
    selected = []
    remaining = list(range(len(candidates)))
    while len(selected) < top_k and remaining:
        best_idx = max(remaining, key=lambda i: 
            lambda_param * relevance_scores[i] - 
            (1 - lambda_param) * max((cosine_sim(embeddings[i], embeddings[j]) for j in selected), default=0)
        )
        selected.append(best_idx)
        remaining.remove(best_idx)
    return [candidates[i] for i in selected]
```

### 5.2 DPP (Determinantal Point Process)

- v2.2 scope에서 **제외**.
- 향후 실험 항목으로 검토.

---

## 6. 평가 전략 (Evaluation)

### 6.1 필수 메트릭

| 메트릭 | 정의 | 도구 |
|:---|:---|:---|
| `recall@K` | 정답 중 추천에 포함된 비율 | `metrics.py` |
| `ndcg@K` | 순서를 고려한 정확도 | `metrics.py` |
| `hit_rate@K` | 최소 1개 정답 포함 여부 | `metrics.py` |
| `coverage@K` | 전체 hobby 중 추천된 고유 hobby 비율 | `metrics.py` 확장 |
| `novelty@K` | 추천된 hobby의 inverse popularity 평균 | `metrics.py` 확장 |
| `intra_list_diversity@K` | 추천 리스트 내 hobby 간 category 기반 Jaccard distance 평균 | `metrics.py` 확장 |
| `segment recall gap` | 연령대/성별별 recall 차이 (max - min) | 신규 |
| `candidate_recall@K` | Stage1 candidate pool 내 정답 포함률 | 기존 |

### 6.2 Ablations (반드시 수행)

| 실험 | 목적 |
|:---|:---|
| v1 deterministic baseline | 현재 기준선 측정 (coverage/novelty/diversity 포함) |
| v2 LightGBM only | learned ranker의 순수 accuracy 영향 |
| v2 LightGBM + MMR (λ=0.7) | diversity reordering의 accuracy-diversity trade-off |
| MMR λ sweep [0.3, 0.5, 0.7, 0.9] | 최적 λ 탐색, Pareto curve 확인 |
| v2 LightGBM + MMR + text embedding | text feature 기여도 (leakage audit 통과 후) |
| Cold-start users only (known <= 1) | cold-start segment에서의 성능 |
| Dense users only (known >= 5) | dense segment에서의 성능 |

### 6.3 Promotion Gate (엄격화된 기준)

```text
v2 learned ranker를 promoted default로 승격하려면 아래 조건을 모두 충족해야 함:

1. Accuracy Gate:
   - delta_recall@10 >= -0.002  (v1 대비 2%p 이내 하락 허용)
   - delta_ndcg@10 >= +0.005    (v1 대비 5%p 이상 개선)
   - delta_candidate_recall@50 == 0 (Stage1 변경 없음을 확인)

2. Diversity Gate:
   - delta_coverage@10 >= 0
   - delta_novelty@10 >= 0
   - delta_intra_list_diversity@10 >= 0

3. Fairness Gate:
   - segment recall gap regression <= 3%p
   - (age_group별 recall 최대-최소 차이가 v1 대비 3%p 이상 증가하지 않음)

4. Leakage Gate (text feature 사용 시에만):
   - post_mask_leakage_audit 통과
   - text feature이 ablation에서 accuracy 저하를 유발하지 않음

단, MMR은 accuracy를 희생할 수 있는 알고리즘이므로:
- MMR 적용 버전은 accuracy gate와 diversity gate를 동시에 평가.
- λ 값 선택은 accuracy-diversity Pareto frontier 상에서 사용자가 결정.
- MMR은 optional flag (`--use-mmr`)로 제공되며, 기본값은 False.
```

### 6.4 Upper-Bound Oracle Metric

Stage1 candidate pool의 품질과 learned ranker의 개선 여지를 평가하기 위해 oracle metric을 측정:

```text
oracle_recall@10 = "만약 ranker가 candidate pool을 완벽히 정렬하면 달성 가능한 recall@10"
```

계산 방법: test 사용자별로 candidate pool에 정답 hobby가 포함되어 있다면 1, 아니면 0. 평균.

**활용 방법**:
- `candidate_recall@50`과 함께 사용하여 2개 gate를 판정:
  1. **Pool Quality Gate**: `candidate_recall@50 >= 0.70` — Stage1이 충분한 후보를 담고 있는가?
  2. **Reranker Improvement Room Gate**: `oracle_recall@10 - v1_recall@10 >= 0.05` — v1보다 나아질 여지가 있는가?
- Pool Quality Gate 실패 시: Stage1 개선이 선행되어야 함.
- Improvement Room Gate 실패 시: v1이 이미 이론 상한에 근접하므로 learned ranker 도입 가치가 낮음.

---

## 7. 리스크 및 대응

| ID | 리스크 | 영향 | 대응 |
|:---|:---|:---:|:---|
| R1 | Learned ranker가 overfitting | 🔴 높음 | Early stopping, feature selection, train-only profile 강제 검증, LightGBM regularization |
| R2 | Text embedding leakage 미완전 차단 | 🔴 높음 | alias-aware + word-boundary regex masking, post_mask leakage audit, 기본 비활성 |
| R3 | MMR이 accuracy를 크게 저하 | 🟡 중간 | λ 파라미터 스윕 (0.1 ~ 0.9), accuracy-diversity Pareto curve 확인, optional flag |
| R4 | LightGBM 설치 의존성 충돌 | 🟡 중간 | `requirements-gnn.txt`에 `lightgbm` 추가, CPU/GPU 버전 명시, Windows wheel 테스트 |
| R5 | SHAP 계산 비용 | 🟢 낮음 | Batch SHAP (per 1000 samples), reason 생성은 offline cache |
| R6 | v2가 v1보다 낮은 성능 | 🔴 높음 | v1 deterministic을 fallback으로 유지, v2는 optional flag (`--use-learned-ranker`)로 배포 |
| R7 | 대규모 데이터 처리 (수만~수십만 rows) | 🟡 중간 | feature generation 병렬화, person/hobby embedding 캐싱, 데이터 스케일이 작으므로 오히려 과적합 주의 |
| R8 | Synthetic data ceiling | 🟡 중간 | scope를 "held-out edge prediction 개선"으로 명확히 한정 |

---

## 8. 구현 계획 (Implementation Plan)

> **전략**: A (Learned Ranker + MMR + SHAP)를 단계별로 잘라서 구현.  
> **핵심 원칙**: 각 Phase마다 Go/No-Go Gate를 두고, 통과 시에만 다음 Phase로 진행.

### Phase 0: Pre-Flight (반드시 먼저 수행)

**목표**: v1 baseline을 정확히 측정하고, Stage1 candidate pool 품질을 확인하여 Stage3 진행 가치를 판정.

- [ ] v1 baseline의 coverage@10, novelty@10, intra_list_diversity@10 측정
- [ ] Stage1 candidate_recall@50, oracle_recall@10 측정
- [ ] ranker_dataset.csv 스키마 확정 (Phase 4.2와 동일)
- [ ] LightGBM training mode 확정 (binary classifier)
- [ ] Negative sampling strategy 확정 (4:1 ratio, Hard+Easy MNS)
- [ ] Text leakage policy 확정 (기본 비활성, flag로만 활성)
- [ ] Promotion gate 수치 확정 (delta tolerance)
- [ ] Runtime budget 및 sample-size strategy 확정

**Phase 0 Go/No-Go Gate:**

> **참고**: Phase 0의 핵심 gate(Pool Quality, Improvement Room)는 기존 아티팩트로 확인 완료.  
> 단, coverage/novelty/intra_list_diversity/candidate_pool_coverage/per-segment recall 측정은 Phase 1에서 신규 메트릭 구현 후 수행.  
> `v1_baseline_metrics.json`은 Phase 1에서 신규 메트릭을 포함하여 재생성할 때까지 `rerank_metrics.json`으로 임시 대체.

```
[GO]  아래 2개 조건을 모두 충족:

      1. Pool Quality Gate:
         candidate_recall@50 >= 0.70
         (Stage1이 test 정답의 70% 이상을 후보 pool에 담고 있어야 함)

      2. Reranker Improvement Room Gate:
         oracle_recall@10 - v1_recall@10 >= 0.05
         (완벽 정렬 시 이론 상한과 v1 실제 성능 간에 5%p 이상 개선 여지가 있어야 함)
         → 이는 learned ranker가 v1보다 나아질 "공간"이 있음을 의미

      → Phase 1 진행

[NO-GO] Pool Quality Gate 실패:
        candidate_recall@50 < 0.70
        → Stage1 개선이 선행되어야 함 (popularity/cooccurrence 로직 변경 또는 candidate_k 증가)
        → v2 learned ranker 보류
        → experiment_decisions.json에 "Stage1 candidate pool 품질 부족" 기록

[NO-GO] Reranker Improvement Room Gate 실패:
        oracle_recall@10 - v1_recall@10 < 0.05
        → v1 deterministic이 이미 이론 상한에 근접 (ranking 개선 여지 없음)
        → learned ranker 도입 가치 없음
        → v1 유지, diversity 개선만 별도 검토 (MMR만 Phase 3부터 시도)
```

---

### Phase 1: Foundation (1-2일)

**목표**: metrics 확장 및 의존성 구축. **text embedding과 embedding cache는 Phase 4 이전으로 연기.**

- [ ] `requirements-gnn.txt`에 `lightgbm`, `shap`, `scikit-learn` 추가
  - shap은 Phase 1에서 설치만 하고 **실사용은 Phase 4**에서 함 (환경 이슈 사전 방지)
- [ ] `metrics.py`에 신규 메트릭 추가
  - `catalog_coverage`는 이미 구현됨 → 검증만 수행
  - `novelty`는 이미 구현됨 → 검증만 수행
  - `intra_list_diversity_at_k` 추가 (category 기반 Jaccard distance)
  - `oracle_recall_at_k` 추가
  - `per_segment_metrics` 추가 (age_group, sex별 recall 분리 측정)
- [ ] 신규 메트릭 단위 테스트 (`tests/test_metrics_extended.py`)
- [ ] LightGBM 설치/동작 확인 (`import lightgbm; lgb.train()` smoke test)
- [ ] ~~text embedding 유틸리티~~ → **Phase 4로 연기**
- [ ] ~~KURE embedding cache~~ → **Phase 4로 연기**

**Phase 1 Go/No-Go Gate:**
```
[GO]  모든 신규 메트릭이 v1 baseline에 대해 정상 측정됨
      AND LightGBM 설치/동작 확인
      → Phase 2 진행

[NO-GO] 메트릭 측정 실패 또는 LightGBM 설치 오류
        → Foundation fix 후 재시도
```

---

### Phase 2: Learned Ranker Core (3-5일)

**목표**: LightGBM binary classifier만 구현. **SHAP은 아직 포함하지 않음.**  
**핵심 질문**: "v2 LightGBM이 v1 deterministic보다 accuracy가 나은가?"

> **⚠️ Phase 2를 통과하지 못하면 Phase 3 이후는 진행하지 않음.**  
> Learned ranker가 deterministic보다 나은 성능을 보이지 않으면 MMR/SHAP은 의미가 없음.  
> Phase 2 NO-GO 시 v1 deterministic을 유지하고 실험 종료.

- [ ] `gnn_recommender/ranker.py` — LightGBM binary classifier 클래스 구현
  - [ ] `RankerDataset` (Phase 4.2 스키마 준수)
  - [ ] `LightGBMRanker.fit()` (early stopping on validation AUC)
  - [ ] `LightGBMRanker.predict()`
  - [ ] `LightGBMRanker.save()` / `load()`
- [ ] `scripts/train_ranker.py` — ranker 학습 CLI
- [ ] `scripts/evaluate_ranker.py` — v1 vs v2 비교 평가 (동일 조건 보장)

**Phase 2 Go/No-Go Gate (Accuracy Gate):**
```
[GO]  delta_recall@10 >= -0.002
      AND delta_ndcg@10 >= +0.005
      → Phase 3 진행 (MMR 추가)

[NO-GO] delta_recall@10 < -0.002 OR delta_ndcg@10 < +0.005
        → LightGBM hyperparameter 튜닝 시도 (1회)
        → 여전히 실패 시 "Learned ranker가 이 데이터에서는 deterministic보다 유의미한 개선을 제공하지 않음"으로 기록
        → v1 유지, MMR/SHAP 구현 보류
```

---

### Phase 3: MMR Diversity Reordering (2-3일)

**목표**: Phase 2를 통과한 learned ranker 위에 MMR을 추가. **SHAP은 아직 포함하지 않음.**  
**핵심 질문**: "MMR이 accuracy를 크게 희생하지 않고 diversity를 개선하는가?"

> **⚠️ Embedding Fallback 정책**: Phase 3에서는 KURE embedding cache가 아직 구현되지 않았으므로,  
> MMR의 `similarity(i, j)` 계산에 **`hobby_taxonomy.json` category 기반 one-hot embedding을 fallback으로 사용**.  
> Phase 4에서 KURE embedding utility 구현 후, Phase 5 통합 시 KURE embedding으로 교체하여 재평가.

- [x] `gnn_recommender/diversity.py` — MMR 구현 ✅ (NO-GO: one-hot similarity binary 한계)
  - [x] `compute_hobby_embeddings()` — **Phase 3: category one-hot fallback only**, Phase 5에서 KURE 교체
  - [x] `mmr_rerank()` (PRD §5.1 공식 준수)
  - [x] `compute_intra_list_diversity()`
- [x] `scripts/sweep_mmr_lambda.py` — λ 파라미터 스윕 + Pareto curve ✅ (all λ identical to baseline)
- [x] `evaluate_reranker.py`에 `--use-mmr` 플래그 추가 (기본값 false) ✅

**Phase 3 Go/No-Go Gate (Diversity Gate):**
```
[GO]  accuracy gate 유지 (delta_recall@10 >= -0.002)
      AND diversity metrics 중 2개 이상 개선 (coverage, novelty, intra_list_diversity)
      → Phase 4 진행 (SHAP 추가)

[NO-GO] accuracy gate 실패 OR diversity metrics가 2개 미만 개선
        → λ 파라미터 재조정 (Pareto frontier 상에서 최적 λ 재선택)
        → 여전히 실패 시 "MMR이 이 candidate pool에서는 diversity-accuracy trade-off를 해결하지 못함"으로 기록
        → Learned ranker만 유지, MMR은 optional flag로 제공 (기본값 false)
```

---

### Phase 2.5: LightGBM Regularization Tuning + Negative Sampling Ablation + Source One-Hot (2-3일)

**목표**: Phase 2 PROMOTED 모델의 ranking collapse 완화 및 feature balance 개선.  
**핵심 진단**: 현재 병목은 **retrieval 부족이 아니라 ranking collapse**임. candidate_recall@50=97.8%로 Stage 1 pool은 충분하나, v2 LightGBM이 top-k를 인기 취미로 수축시킴 (coverage@10: v1=51.7% → v2=15.56%). 동일 pool에서 v1 deterministic은 coverage@10=51.7%를 달성했으므로, 문제는 pool 크기가 아니라 ranking objective/feature balance임.
**핵심 질문**: "Regularization, negative sampling, source feature가 ranking collapse를 완화하고 feature 기여 균형을 개선하는가?"

**Feature importance 현황** (`phase2_5_num_leaves_31` 기준):
| Feature | Gain | 비율 |
|:---|---|:---|
| cooccurrence_score | 91,013.19 | 60.73% |
| popularity_prior | 42,170.82 | 28.14% |
| **두 개 합계** | **133,184.01** | **88.87%** |
| age_group_fit | 8,638.08 | 5.76% |
| known_hobby_compatibility | 2,908.98 | 1.94% |
| 나머지 feature | 5,177.86 | 3.45% |

> **참고**: `popularity_penalty=0`, `category_diversity_reward=0`은 feature 값이 상수(0.0)라 학습 불가능한 것. `lightgcn_score=0`은 현재 Stage 1 provider 조합(pop+cooc)에서 LightGCN 점수가 들어오지 않아서임. "무의미"가 아니라 "현재 설정에서 실질 기여가 작다"로 해석해야 함.

#### 2.5-A. Regularization Tuning

- [x] `scripts/tune_ranker_regularization.py` — Sequential greedy search
  - [x] Baseline 대비 `num_leaves`, `min_data_in_leaf`, `max_depth`, `feature_fraction`, `bagging_fraction`, `reg_alpha`, `reg_lambda` 탐색
  - [x] validation recall@10 기준 best config 선정: `num_leaves=31`, `min_data_in_leaf=50`, `learning_rate=0.05`, `reg_alpha=0.1`, `reg_lambda=0.1`
  - [x] fixed-config validation/test 평가 완료: validation `recall@10=0.7390509`, test `recall@10=0.7096839752057718`
  - [x] 산출물: `artifacts/regularization_tuning.json`, `artifacts/regularization_tuning_summary.md`

#### 2.5-B. Negative Sampling Ablation

- [x] Negative sampling ablation — single-config train/eval 방식으로 실행
  - [x] **핵심 목표**: hard/easy 비율이 popularity bias에 미치는 영향 측정
  - [x] validation-only selection 후 선택 후보만 final test 평가
  - [x] train/val row 수, AUC, recall@10, ndcg@10, **coverage@10** 비교 (accuracy + diversity 동시 평가)
  - [x] 산출물: `artifacts/experiments/phase2_5_negative_sampling_summary.md`
  - [x] 결정: validation best `hard_ratio=1.0`은 final test에서 current default보다 낮아 `neg_ratio=4`, `hard_ratio=0.8` 유지

#### 2.5-C. Source One-Hot Feature Ablation

- [x] `gnn_recommender/rerank.py`에 source one-hot feature 3개 추가
  - [x] `source_is_popularity`: 이 후보가 popularity provider에서 왔으면 1, 아니면 0
  - [x] `source_is_cooccurrence`: 이 후보가 cooccurrence provider에서 왔으면 1, 아니면 0
  - [x] `source_count`: 이 후보를 생성한 provider 수 (1 또는 2)
  - [x] `RANKER_FEATURE_COLUMNS`에 3개 추가, `RANKER_CATEGORICAL_FEATURES`에 `source_is_popularity`, `source_is_cooccurrence` 추가
  - [x] `build_rerank_features()`에서 `HobbyCandidate.source_scores`를 기반으로 feature 생성
- [x] `scripts/train_ranker.py` 및 평가 파이프라인 업데이트
- [x] Source one-hot 포함/미포함 비교 평가 (coverage@10 변화 측정)
- [x] 산출물: `artifacts/experiments/phase2_5_source_onehot_summary.md`
- [x] 결정: validation recall/ndcg/coverage가 current default보다 낮아 `include_source_features=false` 유지, test 생략

> **참고**: source one-hot은 구현 비용이 낮으나, `source_scores` dict에 이미 연속형 점수로 provider 정보가 인코딩되어 있어 효과가 제한적일 수 있음. Ablation으로 실제 기여도를 검증해야 함.

**Phase 2.5 Go/No-Go Gate (Stability + Diversity Gate):**
```
[GO]  regularization tuning 또는 neg_ratio/hard_ratio 변경이
      test recall@10 >= -0.002 AND test ndcg@10 >= +0.005 (vs v1 deterministic) 유지
      AND coverage@10 개선 (v2 baseline 15.56% 대비 유의미 증가)
      AND 과적합 지표(val-test gap) 개선 또는 동등
      → best config을 새 기본값으로 승격

[NO-GO] 모든 설정이 baseline보다 불안정하거나 promotion gate 미충족
        → 기존 기본값(neg_ratio=4, hard_ratio=0.8) 유지
        → tuning 결과는 실험 기록으로만 보존
```

> **핵심 진단 요약**: 현재 v2 LightGBM의 core 문제는 **ranking collapse**임.
> candidate_recall@50=97.8%로 Stage 1 pool은 충분하나,
> v2가 top-k를 인기 취미로 수축시켜 coverage@10이 v1의 51.7%에서 15.56%로 급감.
> 같은 pool에서 v1 deterministic은 51.7%를 달성했으므로,
> 문제는 pool 크기가 아니라 ranking objective / feature balance임.
> candidate_pool_size 확장(50→100)은 1순위가 아님.

---

### Phase 4: SHAP Explanation + Text Embedding (2-3일)

**목표**: Phase 2를 통과한 모델(learned ranker)에 SHAP 기반 해석 추가. Phase 3은 NO-GO였으므로 MMR은 optional flag로만 제공. 동시에 Phase 1/3에서 연기된 Text Embedding utility 및 KURE embedding cache를 구현.  
**핵심 질문**: "SHAP reason이 90% 이상의 추천에 대해 의미 있게 생성되는가?"

#### 4-A. SHAP 해석

- [ ] `gnn_recommender/ranker_explain.py` — SHAP 해석 모듈
  - [ ] `compute_shap_values()` (batch 처리)
  - [ ] `generate_reason()` (한국어 템플릿)
  - [ ] Reason 자동 검증 (non-empty, NaN 미참조, masked hobby 미참조)
- [ ] `recommend_for_persona.py`에 `--explain` 플래그 추가

#### 4-B. Leakage-Safe Text Embedding Utility (Phase 1에서 연기됨)

- [ ] `gnn_recommender/text_embedding.py` — 신규 모듈
  - [ ] `mask_holdout_hobbies()` — alias-aware + regex word-boundary masking
  - [ ] `post_mask_leakage_audit()` — 마스킹 후 잔존 leakage 검사
  - [ ] `compute_text_embedding_similarity()` — KURE-v1 기반 cosine similarity
- [ ] 단위 테스트 (`tests/test_text_embedding.py`)

#### 4-C. KURE Embedding Cache (Phase 1에서 연기됨)

- [ ] `gnn_recommender/embedding_cache.py` — 신규 모듈
  - [ ] `PersonEmbeddingCache` — persona text embedding 캐시
  - [ ] `HobbyEmbeddingCache` — hobby name embedding 캐시
- [ ] Phase 3 MMR의 `compute_hobby_embeddings()`를 KURE embedding으로 교체 가능하도록 인터페이스 확인

**Phase 4 Go/No-Go Gate (Explanation Gate):**
```
[GO]  자동 검증 기준 통과:
      - 90% 이상의 추천에서 non-empty reason 생성
      - reason에 NaN/masked hobby 미포함
      → Phase 5 진행 (Integration)

[NO-GO] 자동 검증 기준 미달
        → 템플릿/SHAP 계산 로직 수정 후 재시도
        → 여전히 실패 시 SHAP은 optional flag로 제공 (기본값 false)
```

---

### Phase 5: Integration & Final Evaluation (2일)

**목표**: 모든 컴포넌트 통합 및 최종 ablation 실행.

- [ ] `recommend_for_persona.py`에 모든 플래그 통합 (`--use-learned-ranker`, `--use-mmr`, `--mmr-lambda`, `--explain`, `--use-text-embedding`)
- [ ] `evaluate_reranker.py`에 모든 모드 통합 (deterministic / learned / learned+mmr)
- [ ] 전체 ablation 실행 (Phase 6.2 참고)
- [ ] `experiment_decisions.json`에 최종 결과 기록
- [ ] `experiment_run_summary.md` 업데이트
- [ ] `README.md`에 v2 아키텍처 반영

**Phase 5 Go/No-Go Gate (Final Gate):**
```
[GO]  모든 ablation 결과가 promotion gate 기준 충족
      AND 기존 v1 테스트 전체 통과 (regression 없음)
      → v2를 promoted default로 승격 (또는 optional flag로 제공)

[NO-GO] promotion gate 미충족 OR regression 발견
        → v1 deterministic을 계속 default로 유지
        → v2는 --use-learned-ranker flag로만 제공
```

---

## 9. 단계별 의사결정 트리

```
Phase 0 Pre-Flight
  └─ oracle_recall >= threshold?
      ├─ YES → Phase 1 Foundation
      │          └─ metrics/LightGBM 준비 완료?
      │              ├─ YES → Phase 2 Learned Ranker
      │              │          └─ accuracy gate 통과?
      │              │              ├─ YES → Phase 3 MMR
      │              │              │          └─ diversity gate 통과?
      │              │              │              ├─ YES → Phase 4 SHAP
      │              │              │              │          └─ explanation gate 통과?
      │              │              │              │              ├─ YES → Phase 5 Integration → v2 promoted
      │              │              │              │              └─ NO  → SHAP optional, learned+MMR 제공
      │              │              │              └─ NO  → MMR optional, learned only 제공
      │              │              └─ NO  → learned ranker 보류, v1 유지
      │              └─ NO  → Foundation fix
      └─ NO  → Stage1 개선 선행, v2 보류
```

---

## 10. 문서 관리

| 파일 | 목적 |
|:---|:---|
| `GNN_Neural_Network/README.md` | 사용자/운영자용 가이드 (v1+v2 통합) |
| `GNN_Neural_Network/PRD_GNN_Reranker_v2.md` | 본 문서 — 제품 요구사항 |
| `GNN_Neural_Network/CHECKLIST_GNN_Reranker_v2.md` | 구현 진행 상황 추적 |
| `GNN_Neural_Network/artifacts/experiment_decisions.json` | ablation 결과 기록 |
| `GNN_Neural_Network/artifacts/experiment_run_summary.md` | 인간이 읽는 요약 |

---

## 11. 변경 이력 (Changelog)

| 날짜 | 버전 | 변경 내용 | 검수자 |
|:---|:---|:---|:---|
| 2026-04-29 | v2.0 | 초안 작성 | - |
| 2026-04-29 | v2.1 | Metis 검토 반영: promotion gate delta 기준 추가, LightGBM binary classifier 명시, ranker row schema 추가, text embedding leakage 정책 강화, MMR 공식 수정, coverage/novelty target 상대 개선으로 변경, XGBoost/DPP v2.1 이관, manual QA gate 자동화로 대체, Phase 0 Pre-Flight 추가 | Metis |
| 2026-04-29 | v2.3 | 검토 피드백 반영: Phase 1 범위 축소 (text embedding/cache → Phase 4 연기), negative sampling 기본 4:1 + ablation 1:1/4:1/8:1, source_* one-hot → Phase 2 ablation으로 이연, 데이터 스케일 과대 추정 수정 (수만~수십만 rows), LightGBM regularization 보수적 설정 (num_leaves 15, min_data_in_leaf 50, L1/L2), Phase 2 통과 후에만 Phase 3 진행 명시 강화, SHAP 의존성은 Phase 1 설치/Phase 4 실사용 분리 | 사용자 검토 |
| 2026-04-29 | v2.4 | 최종 교차 검증(Oracle/Explore/Librarian) 반영: §2.2 negative sampling 자가모순 해소 (Hard Only → Hard+Easy 4:1 MNS 통일), §1.2/§3.1 feature 수 12→14 정정 및 제외 사유 명시, Phase 4에 Text Embedding+KURE cache 구현 항목 추가, Phase 3에 MMR embedding fallback 정책 명시, 버전 헤더 v2.4 통일 | Oracle+Explore+Librarian |
| 2026-05-01 | v2.5 | Phase 2 PROMOTED (LightGBM AUC=0.8890555966387075, val recall@10=0.7391, test recall@10=0.7097). Phase 3 NO-GO: category one-hot MMR 실험 결과 모든 lambda(0.1~0.9)에서 baseline과 동일 (one-hot similarity가 binary이어서 reorder 불가). KURE dense embedding 필요. oracle_recall_at_k 버그 수정 (pool[:k]→full pool). tests 212 passed. experiment_decisions.json + experiment_run_summary.md 업데이트 | 구현 검증 |
| 2026-05-01 | v2.6 | Phase 2.5 추가: regularization tuning + neg sampling ablation + source one-hot ablation. 핵심 진단 정정: 병목은 retrieval 부족이 아니라 **ranking collapse** (v2 LightGBM이 top-k를 인기 취미로 수축, coverage@10 51.7%→15.56%). cooccurrence_score(60.73%) + popularity_prior(28.14%)로의 편중이 확인되어 feature diversity 완화 필요성이 남음. Go/No-Go Gate에 diversity 지표 추가. source one-hot을 Phase 2.5-C로 편입. | 진단 정정 |

---

**작성일**: 2026-05-01  
**버전**: v2.6 (A-단계형)  
**상태**: Phase 2 PROMOTED, Phase 2.5 진행 중 (ranking collapse 완화), Phase 3 NO-GO (MMR one-hot 한계)  
**다음 단계**: Regularization tuning → Neg sampling ablation → Source one-hot ablation → KURE hobby embedding MMR 재평가
