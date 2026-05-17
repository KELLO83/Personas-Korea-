# GNN_Neural_Network: 취미 추천 시스템

이 폴더는 `Nemotron-Personas-Korea` 데이터에서 `Person -> Hobby` 추천을
실험하는 오프라인 추천 시스템 workspace입니다. 현재 문서의 기준일은
2026-05-17입니다.

## 현재 결론

현재 로컬 데이터와 현재 split 기준 accuracy SOTA와 production default는 모두
E5-small-ko-v2 domain-specific Stage2입니다.

```text
Stage1 = popularity + cooccurrence
Stage2 = LightGBM(num_leaves=31)
Embedding model = dragonkue/multilingual-e5-small-ko-v2
Feature policy = text_embedding_similarity + E5 domain-specific similarities
MMR = false
KURE Stage1 semantic provider = false
```

현재 기본 모델:

```text
GNN_Neural_Network/artifacts/experiments/phase5_c_text_embedding/e5_domain_features_validation_thread18/ranker_model.txt
```

## 현재 Test 결과

같은 current split, 같은 Stage1 후보풀, 같은 LightGBM recipe를 고정했습니다.
바뀐 것은 Stage2에 들어가는 embedding feature 형태입니다.

| Model | Test Recall@10 | Test NDCG@10 | Candidate Recall@50 | Decision |
| --- | ---: | ---: | ---: | --- |
| no-text LightGBM | 0.579626 | 0.360270 | 0.827208 | baseline |
| KURE-v1 Stage2 single | 0.617482 | 0.386258 | 0.827208 | historical baseline |
| E5-small-ko-v2 Stage2 single | 0.623837 | 0.393921 | 0.827208 | previous production default |
| Snowflake-ko Stage2 single | 0.637653 | 0.402805 | 0.827208 | previous accuracy reference |
| E5-small-ko-v2 Stage2 domain-specific | 0.680943 | 0.436665 | 0.827208 | current SOTA/default |

E5 domain-specific delta:

| 비교 대상 | Recall@10 delta | NDCG@10 delta |
| --- | ---: | ---: |
| Stage1 | +0.111173 | +0.080306 |
| E5 single | +0.057106 | +0.042744 |
| Snowflake single | +0.043290 | +0.033860 |
| KURE-v1 single | +0.063461 | +0.050407 |

## 왜 E5 Domain-Specific이 기본값인가

기존 E5/Snowflake/KURE 단일 feature는 persona 전체 텍스트와 후보 hobby 텍스트의
cosine similarity 하나만 LightGBM에 넣었습니다. 현재 기본값은 여기에 더해
persona text를 domain별로 나눈 similarity를 함께 넣습니다.

추가 feature:

```text
e5_professional_similarity
e5_sports_similarity
e5_arts_similarity
e5_travel_similarity
e5_food_similarity
e5_family_similarity
```

즉 LightGBM은 "전체적으로 비슷한가"뿐 아니라 "스포츠/예술/여행/음식/가족/직업
맥락 중 어디에서 매칭됐는가"를 함께 학습합니다. 이 구조가 test에서 Snowflake
single-feature보다도 높게 나왔으므로 현재 SOTA로 승격했습니다.

## 모델 역할

| 구성 | 현재 역할 | 결정 |
| --- | --- | --- |
| `popularity + cooccurrence` | Stage1 후보 생성 | 유지 |
| LightGBM ranker | Stage2 정렬 | 유지 |
| E5-small domain-specific features | Stage2 semantic feature | 현재 SOTA/default |
| Snowflake single feature | 이전 accuracy reference | 보존 |
| E5-small single feature | 이전 lightweight baseline | 보존 |
| KURE-v1 Stage2 single | historical Stage2 baseline | 보존 |
| KURE-v1 Stage1 semantic provider | Stage1 semantic 후보 생성 | 거절 |
| KURE dense MMR | diversity reranker | 거절 |

Stage1에 semantic embedding을 넣는 방식은 현재 추천하지 않습니다. KURE Stage1
semantic provider는 candidate_recall@50을 떨어뜨렸고, Stage2가 맞출 수 있는
정답 후보 자체를 줄였습니다. 현재 증거상 embedding은 Stage2 feature로 쓰는 것이
맞습니다.

## 실행 명령

백엔드, export, Neo4j, API는 기본 `.venv` Python 3.11로 실행합니다.
검증된 ML 학습/평가는 `.venv314t` Python 3.14t로 실행할 수 있습니다.

현재 기본 모델 test 평가:

```powershell
.\.venv314t\Scripts\python.exe GNN_Neural_Network\scripts\evaluate_ranker.py `
  --config GNN_Neural_Network\configs\kure_text_optin_ranker.yaml `
  --split test `
  --model-path GNN_Neural_Network\artifacts\experiments\phase5_c_text_embedding\e5_domain_features_validation_thread18\ranker_model.txt `
  --output GNN_Neural_Network\artifacts\experiments\phase5_c_text_embedding\e5_domain_features_test_thread18\test_metrics.json `
  --experiment-id e5_domain_features_test_thread18 `
  --text-embedding-model-name dragonkue/multilingual-e5-small-ko-v2 `
  --embedding-batch-size 0 `
  --embedding-vram-utilization 0.85 `
  --cpu-thread-count 18 `
  --feature-build-parallelism auto `
  --ranking-build-parallelism auto `
  --skip-v1 `
  --progress-mode on
```

## 주요 Artifact

| 목적 | 경로 |
| --- | --- |
| 현재 SOTA/default 모델 | `artifacts/experiments/phase5_c_text_embedding/e5_domain_features_validation_thread18/ranker_model.txt` |
| 현재 SOTA/default validation 결과 | `artifacts/experiments/phase5_c_text_embedding/e5_domain_features_validation_thread18/validation_metrics.json` |
| 현재 SOTA/default test 결과 | `artifacts/experiments/phase5_c_text_embedding/e5_domain_features_test_thread18/test_metrics.json` |
| 이전 E5 single test 결과 | `artifacts/experiments/phase5_c_text_embedding/e5_small_stage2_single_feature_test_thread18/test_metrics.json` |
| 이전 Snowflake single test 결과 | `artifacts/experiments/phase5_c_text_embedding/snowflake_stage2_single_feature_test_thread18/test_metrics.json` |
| 이전 KURE current-split 비교 | `artifacts/experiments/phase5_c_text_embedding/current_locked_num_leaves31_comparison.json` |
| 실험 의사결정 JSON | `artifacts/experiment_decisions.json` |
| 사람이 읽는 실험 요약 | `artifacts/experiment_run_summary.md` |

## 다음 실험 우선순위

1. E5-domain rank/margin feature
   - `e5_similarity_percentile`
   - `e5_similarity_rank`
   - `e5_similarity_gap_to_top`
   - `e5_similarity_gap_to_mean`
2. Candidate hobby text expansion
   - `name_only`
   - `name_plus_aliases`
   - `name_plus_category`
   - `name_plus_short_description`
3. Cold-start, segment, qualitative review
   - `known_hobbies <= 1`
   - age/sex segment gap
   - top-k 추천 실패 사례 점검

모든 후속 실험은 validation-first로 진행하고, test split은 validation winner에
대해서만 1회 실행합니다. 진행바는 항상 표시해야 하며, cache/model metadata에는
embedding model, revision, preprocessing, masking, feature columns, worker/thread
count를 기록해야 합니다.

## 관련 문서

- `PRD.md`: 요구사항, 현재 SOTA/default 결정, promotion rule
- `TASKS.md`: 실행 가능한 작업 상태
- `DATASET_EXPLAIN.md`: 데이터셋 구조와 leakage 참고
- `artifacts/experiment_decisions.json`: machine-readable 실험 의사결정
- `artifacts/experiment_run_summary.md`: human-readable 실험 기록
