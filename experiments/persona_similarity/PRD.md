# 유사 페르소나 추천 실험 PRD

## 목표

이 문서는 `Person -> Person` 유사 페르소나 추천 실험의 제품/모델 요구사항을 정의한다.

핵심 목표는 공부용 알고리즘 나열이 아니라, 현재 Neo4j에 적재된 5만 페르소나 그래프에서 실제로 더 좋은 유사 페르소나 순위를 만들 수 있는지 관측하는 것이다.

이 실험은 `GNN_Neural_Network/`의 `Person -> Hobby` 취미 추천 실험과 분리한다.

```text
GNN_Neural_Network/
  Person -> Hobby 추천

experiments/persona_similarity/
  Person -> Person 유사 페르소나 추천
```

## 현재 추천 구조

현재 플랫폼의 유사 페르소나는 Neo4j GDS로 생성된다.

```text
Neo4j 이질 그래프
  -> FastRP node embedding
  -> KNN으로 가까운 Person 탐색
  -> (:Person)-[:SIMILAR_TO {score}]->(:Person)
```

여기서 `SIMILAR_TO.score`가 현재 `fastrp_score`이다.

```text
fastrp_score = 그래프 구조상 두 Person embedding이 얼마나 가까운지 나타내는 후보생성 점수
```

이 점수는 유용하지만, 왜 비슷한지 feature별로 분해되지 않는다. 그래서 현재 API는 post-hoc 방식으로 공통 속성 설명을 따로 계산한다.

## 문제 정의

현재 FastRP/KNN 방식은 그래프상 가까운 사람을 빠르게 찾을 수 있지만 다음 문제가 있다.

- 지역, 성별, 혼인상태, community 같은 넓은 속성만 같아도 비슷하게 나올 수 있다.
- "왜 비슷한가"가 사용자에게 충분히 납득되지 않을 수 있다.
- 나이, 학력, 군필, 주거 같은 구조화 속성보다 실제로는 문장형 서술이 더 중요한 경우가 많다.
- 예: 성격, 퇴근 후 활동, 생활패턴, 가치관, 커리어 태도, 가족관, 여가 방식.

따라서 목표는 단순히 그래프상 가까운 사람을 찾는 것이 아니라, 다음 조건을 만족하는 순위를 만드는 것이다.

- 후보는 FastRP/KNN으로 충분히 넓게 확보한다.
- 구조화된 공통점과 문장 의미 유사도를 함께 본다.
- 설명 가능한 이유를 제공한다.
- 너무 일반적인 속성만으로 추천하지 않는다.
- 모델/feature/비용을 비교 가능한 artifact로 남긴다.

## 데이터 현실

현재 로컬 Neo4j DB는 원본 100만 페르소나 전체가 아니다.

```text
원본 데이터: 약 100만 페르소나
현재 Neo4j DB: 10대/20대/30대 중심 5만 페르소나 샘플
```

따라서 이 실험의 결론은 다음 범위로 제한한다.

```text
현재 5만 페르소나 그래프 안에서
유사 페르소나 추천을 어떻게 개선할 수 있는가?
```

전체 100만 페르소나 또는 전체 연령대에 대한 운영 성능을 주장하지 않는다.

## 실험 단위

학습 데이터의 한 row는 한 사람 자체가 아니라 `source -> target` 후보쌍이다.

```text
source_uuid -> target_uuid
```

의미:

```text
source_uuid 사람을 보고 있을 때
target_uuid 사람을 유사 페르소나로 얼마나 높게 보여줄 것인가?
```

예:

```text
source_uuid | target_uuid | label | fastrp_score | same_occupation | all_text_cosine
A           | B           | 0.72  | 0.91         | 1               | 0.84
A           | C           | 0.31  | 0.66         | 0               | 0.41
A           | D           | 0.18  | 0.51         | 0               | 0.22
```

모델은 `source_uuid`, `target_uuid`를 외우는 것이 아니라, 두 사람 사이의 pair feature를 보고 같은 source 안에서 후보 target의 순위를 학습한다.

## 범위

### 포함

- Neo4j의 `SIMILAR_TO` 후보쌍 export.
- `source-target` pair feature 생성.
- 구조화 feature 기반 baseline.
- 문장 embedding similarity feature 실험.
- LightGBM LambdaRank / rank_xendcg 실험.
- FastRP, deterministic score, LightGBM, hybrid, text feature ablation 비교.
- metrics, manual review sample, model, metadata artifact 저장.

### 제외

- 모든 `Person x Person` 조합 학습.
- `uuid`나 이름을 모델 feature로 넣는 방식.
- 문장 원문을 LightGBM에 직접 넣는 방식.
- weak label만으로 production 성능을 주장하는 것.
- 유사 페르소나 실험을 취미 추천 폴더와 섞는 것.

## 핵심 모델 전략

유사 페르소나에서는 문장 서술이 중요할 가능성이 높다.

다만 문장 임베딩을 후보생성기로 바로 쓰는 것은 위험하다. 취미 추천 실험에서 KURE semantic Stage1 provider는 candidate recall을 크게 떨어뜨렸다.

따라서 이 프로젝트에서는 다음 전략을 사용한다.

```text
FastRP/KNN = 후보생성기
LightGBM ranking model = reranker
KURE/text embedding = reranker feature
```

즉 KURE/문장 임베딩은 사람 후보를 새로 찾는 주체가 아니라, 이미 확보한 후보쌍을 더 잘 정렬하기 위한 feature로 사용한다.

## 실험 우선순위

이 프로젝트의 실험은 모델을 많이 나열하는 것이 아니라, 현재 데이터꼴에서 실제 개선 가능성이 높은 순서로 실행한다.

### 필수 실험

1차 의사결정에 반드시 필요한 실험이다.

```text
E0. FastRP/KNN 후보생성 baseline
E1. 구조화 deterministic baseline
E2. 구조화 LightGBM LambdaRank / rank_xendcg
E3. 문장 embedding cosine feature 생성
E4. Text-only ablation
E5. 구조화 + 문장 + FastRP 통합 LightGBM
E6. FastRP score와 model score hybrid
E7. Diversity / novelty final reranking
```

이 단계에서 판단할 질문은 다음과 같다.

- FastRP/KNN 순서를 LightGBM reranker가 실제로 이기는가?
- 구조화 feature만으로 충분한가, 문장 feature가 의미 있는 신호를 주는가?
- 문장 feature가 성능을 올리는가, 아니면 노이즈만 늘리는가?
- 최종 top-k가 같은 직업/지역/커뮤니티에 과도하게 몰리지 않는가?
- 설명 가능한 추천 이유가 baseline보다 좋아지는가?

### 후보생성 확장 실험

필수 실험 이후 candidate recall이 부족하다고 판단되면 실행한다.

```text
E8. Personalized PageRank 후보생성 baseline
E9. Node2Vec 후보생성 baseline
```

역할:

- FastRP/KNN이 놓치는 후보가 있는지 비교한다.
- LightGBM reranker의 입력 후보 pool을 넓히는 용도로만 사용한다.
- PPR/Node2Vec 결과도 그대로 production으로 승격하지 않고, 같은 split/metric/manual review로 비교한다.

### 대체 reranker 검증

LightGBM이 충분히 강하지 않거나 범주형 feature 처리 방식이 유리한지 확인할 때만 실행한다.

```text
E10. CatBoost ranking
```

원칙:

- 기본 reranker는 LightGBM이다.
- CatBoost는 동일 feature, 동일 split, 동일 candidate pool에서만 비교한다.
- 성능 차이가 작고 학습/운영 비용이 크면 LightGBM을 유지한다.

### 장기 후보

현재 5만 샘플과 weak label만으로는 우선순위가 낮다.

```text
GraphSAGE / PinSage
Two-Tower persona encoder
Cross-encoder reranker
```

실행 조건:

- human labeled similar-person pair가 생긴다.
- 실제 클릭/상세보기/선택 로그가 쌓인다.
- 100만 전체 데이터에서 FastRP/KNN refresh 비용이 병목이 된다.
- 신규 persona에 대한 inductive embedding이 필요해진다.

## 실험 계획

### E0. FastRP/KNN 후보생성 baseline

목적:

- 현재 유사 페르소나 추천의 control group 고정.
- reranker가 재정렬할 후보 pool 확보.

실험:

- `topK=5`: smoke 검증용.
- `topK=50`: 1차 실제 실험 기본값.
- `topK=100`: candidate recall/품질 비교용.

중요:

```text
reranker는 export된 후보 안에서만 순서를 바꿀 수 있다.
topK=5로 만든 후보쌍으로는 유의미한 reranker 실험이 어렵다.
```

현재 추천:

```text
5만 Person 전체 + GDS topK=50
```

### E1. 구조화 deterministic baseline

목적:

- ML 없이 이해 가능한 baseline을 만든다.
- LightGBM이 진짜 필요한지 판단한다.

방식:

```text
직업 일치
지역 일치
교육/전공 일치
가족/주거 일치
공유 취미 수
FastRP score
```

를 가중합한다.

이 baseline을 LightGBM이 못 이기면, 학습 모델을 쓸 이유가 약하다.

### E2. 구조화 LightGBM LambdaRank

목적:

- 첫 번째 메인 ranking 모델.

입력 feature:

```text
fastrp_score
age_diff
same_age_group
same_sex
same_province
same_district
same_occupation
same_education
same_field
same_marital
same_family
same_housing
same_community
shared_hobby_count
shared_skill_count
explanation_feature_count
```

학습 방식:

```text
group = source_uuid
objective = lambdarank
```

비교:

- `lambdarank`
- `rank_xendcg`

### E3. 문장 embedding similarity feature

목적:

- 나이/학력/군필/지역보다 실제 유사성에 가까운 문장 의미를 반영한다.

사용할 가능성이 높은 문장 컬럼:

```text
persona
professional_persona
sports_persona
arts_persona
travel_persona
culinary_persona
family_persona
cultural_background
career_goals_and_ambitions
skills_and_expertise
hobbies_and_interests
```

원문을 모델에 직접 넣지 않는다. 각 사람의 문장을 embedding으로 바꾼 뒤 pairwise cosine similarity를 feature로 넣는다.

예상 feature:

```text
all_text_cosine
persona_text_cosine
professional_text_cosine
hobbies_text_cosine
skills_text_cosine
career_text_cosine
family_text_cosine
lifestyle_text_cosine
```

중요:

- 취미추천에서 KURE Stage1 후보생성은 실패했다.
- 하지만 KURE text feature는 일부 실험에서 no-text보다 좋은 신호가 있었다.
- 유사페르소나는 취미 item 정답 맞히기보다 사람의 생활/성향/가치관 유사성이 중요하므로 text feature 실험 가치가 더 크다.

### E4. Text-only ablation

목적:

- 문장 의미만으로도 유사성 신호가 있는지 확인한다.

방식:

```text
구조화 feature 제거
text cosine feature만 사용
```

해석:

- text-only가 강하면 문장형 서술이 핵심 신호라는 뜻이다.
- text-only가 약해도 structured+text가 좋아지면 보조 feature로 가치가 있다.

### E5. 구조화 + 문장 + FastRP 통합 모델

목적:

- 최종 후보 모델.

입력:

```text
FastRP score
구조화 pair feature
문장 embedding cosine feature
```

모델:

```text
LightGBM LambdaRank 또는 rank_xendcg
```

기대:

- FastRP는 그래프 구조를 잡는다.
- 구조화 feature는 명시적 설명 근거를 준다.
- 문장 feature는 성향/생활패턴/가치관을 잡는다.

### E6. Hybrid score

목적:

- LightGBM이 weak label에 과적합하는 것을 줄인다.

방식:

```text
final_score = alpha * normalized_model_score
            + (1 - alpha) * normalized_fastrp_score
```

실험:

```text
alpha = 0.3, 0.5, 0.7, 0.9
```

### E7. Diversity / novelty final reranking

목적:

- 유사 페르소나 top-k가 같은 직업, 같은 지역, 같은 커뮤니티, broad demographic match로만 수축되는 것을 막는다.
- 같은 target persona가 여러 source에서 과도하게 반복되는 hub 현상을 관찰한다.
- ranking metric을 크게 잃지 않는 범위에서 설명가능성과 탐색 다양성을 높인다.

취미추천의 category diversity를 그대로 쓰지는 않는다. 유사페르소나에서는 다음 축으로 바꿔 본다.

```text
occupation diversity
location diversity
community diversity
text-domain similarity diversity
low-information match penalty
hub target / repeated target concentration
```

1차 실험:

```text
base_score = fastrp_score 또는 model_score
final_score = base ranking에서 직업/지역/community 반복을 penalty로 조정한 rerank score
```

기본 penalty 후보:

```text
target_occupation 반복
target_province 반복
target_community_id 반복
low-information-only match
```

실험값:

```text
diversity_lambda = 0.05, 0.1, 0.2
```

판단:

- NDCG@5/10이 크게 떨어지면 reject.
- occupation/location/community diversity가 개선되어야 한다.
- demographic-only recommendation ratio가 낮아져야 한다.
- manual review에서 억지 다양화가 아니라 의미 있는 유사성으로 보여야 한다.

### E8. Personalized PageRank 후보생성 baseline

현재 FastRP/KNN 후보 pool의 recall이 부족하다고 판단될 때 실행한다.

목적:

- 그래프에서 source persona 주변을 random-walk 관점으로 탐색한다.
- FastRP embedding 기반 후보와 다른 후보가 나오는지 확인한다.
- 설명 가능한 구조적 근접 후보를 추가로 확보할 수 있는지 본다.

방식:

```text
source Person
  -> PPR / random walk with restart
  -> topK target Person 후보
  -> 동일 pair feature builder
  -> 동일 reranker/evaluation pipeline
```

판단:

- FastRP/KNN 대비 새로운 strong-reason 후보가 늘어나는가?
- candidate overlap이 너무 높으면 유지할 이유가 약하다.
- 후보생성 시간이 FastRP/KNN 대비 감당 가능한가?
- 최종 성능은 PPR 단독이 아니라 reranker 입력 후보 pool 개선으로 판단한다.

### E9. Node2Vec 후보생성 baseline

현재 FastRP embedding이 이질 그래프 구조를 충분히 담지 못한다고 판단될 때 실행한다.

목적:

- random-walk 기반 node embedding으로 Person 후보를 생성한다.
- FastRP/KNN과 후보 다양성, ranking 성능, 설명가능성을 비교한다.

방식:

```text
Neo4j graph export
  -> Node2Vec embedding
  -> approximate nearest neighbor / topK Person 후보
  -> 동일 pair feature builder
  -> 동일 reranker/evaluation pipeline
```

판단:

- FastRP보다 NDCG/strong-reason/manual review가 좋아야 한다.
- 학습/embedding refresh 비용이 과도하면 reject한다.
- FastRP와 비슷한 결과라면 운영 단순성을 위해 FastRP를 유지한다.

### E10. CatBoost ranking

범주형 feature 처리 방식이 LightGBM보다 유리한지 확인하는 대체 reranker 실험이다.

현재 feature는 대부분 pairwise binary/numeric이므로 우선순위는 LightGBM보다 낮다.

원칙:

- 동일 candidate pair dataset을 사용한다.
- 동일 feature set과 동일 group split을 사용한다.
- `source_uuid` 단위 group ranking으로 비교한다.
- XGBoost는 실험 대상에서 제외한다.

판단:

- LightGBM 대비 ranking metric, 설명가능성, manual review가 동시에 개선되어야 한다.
- categorical handling 이점이 관측되지 않으면 유지하지 않는다.
- 성능 차이가 작거나 학습/운영 비용이 크면 LightGBM을 유지한다.

### E11. GraphSAGE / PinSage / Two-Tower 계열

현재는 후순위다.

이유:

- 현재는 사람-사람 정답 라벨이 없다.
- 5만 규모에서는 FastRP/KNN + reranker가 더 현실적이다.
- GNN은 weak label을 복잡하게 외울 위험이 있다.
- Two-Tower는 100만 전체 운영과 ANN 검색이 필요해질 때 의미가 커진다.

나중에 다음 조건이 생기면 고려한다.

- human labeled similar-person pairs.
- 실제 사용자 클릭/선택 로그.
- 100만 전체에서 FastRP/KNN refresh 비용이 병목.
- 신규 페르소나 inductive embedding이 필요.

## 평가 지표

### Ranking

- `NDCG@5`
- `NDCG@10`
- FastRP baseline 대비 pairwise win-rate
- top-K overlap

### 설명가능성

- explanation coverage@K
- strong reason coverage@K
- average reason count@K
- low-information dominance@K

강한 설명:

```text
직업
세부 지역
교육/전공
공유 취미
공유 스킬
문장 의미 유사도
```

약한 설명:

```text
성별만 같음
혼인상태만 같음
province만 같음
community만 같음
```

### 다양성/안정성

- unique target count
- repeated target concentration
- occupation diversity
- location diversity
- community diversity
- demographic-only recommendation ratio
- hub target rate
- seed 고정 시 순위 안정성

### 효율

- GDS build time
- candidate pair export time
- feature build time
- embedding build/cache time
- train time
- evaluation time
- inference throughput
- model size
- GPU/CPU 사용량

## Promotion 기준

실험 모델이 root 플랫폼 통합 후보가 되려면 최소 조건은 다음과 같다.

- FastRP baseline보다 ranking metric이 나쁘지 않아야 한다.
- deterministic baseline보다 의미 있는 개선이 있어야 한다.
- 설명가능성이 좋아져야 한다.
- broad demographic match로만 추천하지 않아야 한다.
- 문장 feature를 썼다면 manual review에서 실제 의미 유사성이 확인되어야 한다.
- refresh/inference 비용이 감당 가능해야 한다.
- 원래 FastRP order로 rollback 가능해야 한다.

현재 기본 production 동작은 유지한다.

```text
FastRP/KNN SIMILAR_TO + post-hoc explanation API
```

## 현재 권장 실행 순서

```text
1. scripts/build_gds.py --top-k 50
2. export_pairs.py
3. build_features.py
4. evaluate_fastrp_baseline.py
5. evaluate_deterministic_baseline.py
6. train_lambdarank.py
7. evaluate_lambdarank.py
8. train_rank_xendcg.py
9. evaluate_rank_xendcg.py
10. hybrid score 비교
11. diversity/final rerank 비교
12. text embedding feature 실험 추가
13. structured+text 통합 모델 비교
14. 필요 시 PPR 후보생성 baseline 비교
15. 필요 시 Node2Vec 후보생성 baseline 비교
16. 필요 시 train_catboost_ranker.py / evaluate_catboost_ranker.py로 CatBoost ranking 대체 reranker 비교
17. manual review와 decision artifact 갱신
```

## 결론

현재 기준의 1차 모델은 다음이다.

```text
FastRP/KNN candidate generation
  -> structured pair features
  -> LightGBM LambdaRank reranker
```

하지만 유사페르소나 품질을 진짜로 끌어올릴 가능성이 큰 2차 핵심 실험은 다음이다.

```text
FastRP/KNN candidate generation
  -> structured pair features
  -> KURE/text embedding cosine features
  -> LightGBM LambdaRank/rank_xendcg reranker
```
