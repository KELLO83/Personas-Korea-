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
HGT / RGCN relational graph transformer
GraphSAGE / PinSage
Two-Tower persona encoder
Cross-encoder reranker
```

실행 조건:

- human labeled similar-person pair가 생긴다.
- 실제 클릭/상세보기/선택 로그가 쌓인다.
- 100만 전체 데이터에서 FastRP/KNN refresh 비용이 병목이 된다.
- 신규 persona에 대한 inductive embedding이 필요해진다.

관계형 그래프 트랜스포머(HGT/RGCN)는 이 PRD에서 유사 페르소나 `Person -> Person`의 장기 Stage1 후보생성 대체 실험으로만 둔다. 현재 기본 전략은 FastRP/KNN 후보 pool을 고정하고 LightGBM/rank_xendcg reranker와 text feature를 검증하는 것이다.

HGT/RGCN을 열 때의 고정 조건:

```text
same source split
same topK candidate budget
same reranker recipe
same text feature policy
same evaluation metrics
```

승격 판단은 HGT embedding 자체의 직관이 아니라 FastRP/KNN 대비 `candidate_recall@50`, NDCG@5/10, explanation coverage, diversity, refresh cost가 동시에 통과하는지로 한다.

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

## 취미추천 실험에서 가져온 유사페르소나 추천 원칙

`GNN_Neural_Network/`의 취미추천 실험에서 확인한 핵심 교훈은 유사페르소나 추천에도 그대로 적용한다.

```text
임베딩/그래프 기반 후보생성 점수를 바로 최종 추천으로 믿지 않는다.
Stage1은 넓은 후보 pool을 안정적으로 만든다.
Stage2 reranker가 구조 feature, 텍스트 feature, 설명 feature를 함께 보고 최종 순서를 정한다.
```

따라서 유사페르소나 추천의 기본 실험 구조는 다음으로 고정한다.

```text
Stage1 = FastRP/KNN topK >= 50 candidate generation
Stage2 = LightGBM LambdaRank / rank_xendcg reranker
Text embedding = Stage2 pair feature
Final rerank = diversity / explanation-aware rerank, only after accuracy baseline is known
```

중요한 금지 사항:

- KURE/Snowflake 같은 텍스트 임베딩을 곧바로 Stage1 후보생성기로 승격하지 않는다.
- Stage1 후보 pool, split, label, LightGBM 설정, text builder를 동시에 바꾸지 않는다.
- 한 실험에서는 하나의 변수만 바꾼다.
- `source_uuid`, `target_uuid`, `display_name` 같은 식별자는 feature로 사용하지 않는다.

## 루트 플랫폼 기능 연동 경계

`PRD.md`의 Virtual Guild, Life Track, Agent Interaction Playground는 이 실험 PRD의 모델 학습 범위가 아니라 root FastAPI/Next.js 제품 기능이다. 이 PRD는 해당 기능이 소비할 수 있는 유사 페르소나 score, reason, diversity, text-domain feature, model metadata를 정의하고 검증한다.

제품 기능별 연결 방식:

```text
Virtual Guild:
  SIMILAR_TO / reranker score + community_id + shared hobby/skill + PageRank
  root API에서 소모임 후보와 D3 graph schema로 변환한다.

Life Track:
  source persona와 older cohort target persona 간 유사성/reason을 제공한다.
  개인 미래 예측이 아니라 cross-sectional cohort 탐색 근거로만 사용한다.

Agent Interaction Playground:
  source-target pair, reason categories, selected persona text fields를 제공한다.
  LLM 대화 생성과 harmony review는 root platform의 cached/admin 기능으로 제한한다.
```

유사 페르소나 모델이 제품 경로에 연결될 때 adapter는 최소 다음 metadata를 반환해야 한다.

```text
source_uuid
target_uuid
rank
score
score_source
model_version
reason_categories
strong_reason_count
weak_only
text_feature_columns
fallback_used
```

모델 artifact가 promotion 전이면 root API는 `experimental` 또는 `fallback` 상태로 표시해야 하며, 일반 사용자 기본 화면에 promoted 모델처럼 섞지 않는다.

### 유사페르소나 추천 실험 우선순위

| 우선순위 | Track | 실험 | 목적 |
| ---: | --- | --- | --- |
| 1 | Candidate Lock | FastRP/KNN `topK >= 50` 후보 pool 재생성 | reranker가 학습할 충분한 후보 폭을 확보한다. |
| 2 | Structured Baseline | 구조 feature 기반 deterministic / LightGBM baseline | FastRP 순서를 reranker가 실제로 개선하는지 확인한다. |
| 3 | Text Feature Baseline | KURE-v1 pair text cosine feature 추가 | 서술형 persona text가 유사도 판단에 주는 신호를 검증한다. |
| 4 | Track A | Snowflake-ko embedding backbone swap | KURE-v1보다 강한 한국어 임베딩 백본이 있는지 비교한다. |
| 5 | Track D | persona text builder ablation | 어떤 persona text 구성이 유사도 추천에 가장 유효한지 검증한다. |
| 6 | Track B | domain-specific text cosine feature | 직업/취미/가족/생활방식 등 어떤 영역이 유사도를 설명하는지 분리한다. |
| 7 | Final Rerank | diversity / explanation-aware rerank | 너무 뻔한 같은 직업/지역 추천으로 수축되는지 완화한다. |
| 8 | Optional Reranker | CatBoost ranking | LightGBM이 명확히 부족할 때만 동일 split/pool에서 비교한다. |

이 순서는 decision artifact 없이 임의로 바꾸지 않는다.

### Track A: Embedding Backbone Swap

Track A는 텍스트 임베딩 모델만 교체하는 실험이다.

기준 모델:

```text
nlpai-lab/KURE-v1
```

후보 모델:

```text
dragonkue/snowflake-arctic-embed-l-v2.0-ko
dragonkue/multilingual-e5-small-ko-v2
```

고정해야 할 항목:

- FastRP/KNN candidate pool
- `source_uuid` 기준 split
- weak label generation policy
- LightGBM config
- pair feature schema
- persona text builder
- leakage audit

기록해야 할 항목:

- `model_name`, `model_revision`
- embedding dimension
- pooling behavior, 알 수 있는 경우
- device, batch size, runtime
- cache hit/miss
- text preprocessing version
- validation NDCG@5/10, explanation coverage, strong-reason rate

### Track D: Persona Text Builder Ablation

Track D는 임베딩 모델을 고정하고 persona text 구성만 바꾸는 실험이다. 기본 백본은 KURE-v1로 둔다.

실험 후보:

```text
persona_text_structured_only
persona_text_narrative_only
persona_text_structured_plus_narrative
persona_text_domain_tagged_blocks
persona_text_summary_style
```

목적은 나이/지역/직업 같은 구조 feature보다 성격, 지향하는 행동, 생활방식, 가치관 같은 서술적 정보가 유사 페르소나 추천에 얼마나 유효한지 확인하는 것이다.

Track D는 Track A와 동시에 수행하지 않는다. 임베딩 백본과 text builder를 함께 바꾸면 성능 변동 요인을 분해할 수 없다.

### Track B: Domain-Specific Text Cosine

Track B는 단일 `all_text_cosine` 대신 영역별 cosine feature를 만든다.

```text
professional_text_cosine
hobbies_text_cosine
skills_text_cosine
career_text_cosine
family_text_cosine
lifestyle_text_cosine
persona_text_cosine
```

이 실험은 LightGBM이 "더 비슷한지"를 깊이있게 학습하고, API 설명 카드에도 연결 가능한 feature를 만들기 위한 것이다.

### Promotion Gate

어떤 reranker도 다음 조건을 통과하기 전에는 production 기본값으로 승격되지 않는다.

- FastRP/KNN baseline과 같은 candidate pool, 같은 split에서 비교한다.
- validation-first, test는 winner-only로 실행한다.
- NDCG@5/10이 개선되어야 한다.
- explanation coverage 또는 strong-reason rate가 악화되면 수동 검토가 필요하다.
- low-information recommendation rate가 증가하면 promotion 보류한다.
- 같은 직업/지역 커뮤니티로 과도하게 몰리는지 diversity metric을 기록한다.
- rollback 경로는 항상 raw `SIMILAR_TO` ordering이다.

## Phase 8: High-accuracy Similar-Persona Extension and Quality Calibration Plan

본 단계는 기존 FastRP/KNN 및 Baseline LightGBM 랭커가 가지는 한계를 극복하고, 의미론적(Semantic)·인구통계학적(Demographic)·행동양식적(Psychographic) 유사성을 통합 제어하기 위한 고정밀 유사 페르소나 추천 고도화 명세를 다룬다.

### 1. 5대 핵심 고도화 아키텍처 명세

#### [1] Cross-Encoder Reranker

* **목적**: Bi-Encoder(Source-Target 독립 임베딩 간 코사인 유사도)의 구조적 한계인 교차 어텐션(Cross-Attention) 부재를 극복하여 텍스트 상호작용의 심층 특징을 학습한다.
* **아키텍처**:
  * Stage 1 (FastRP/KNN)에서 통과된 상위 $K \le 50$ 후보 페어에 대해 Reranking을 수행한다.
  * 입력 구성: `[CLS] Source Persona Description [SEP] Target Persona Description [SEP]`
  * 사전 학습된 한국어 인코더(dragonkue/snowflake-arctic-embed-l-v2.0-ko 또는 multilingual-e5 계열 등)의 전 레이어(All-layer) self-attention을 통과시켜 두 페르소나 텍스트의 결합 표현(Joint Representation)을 얻고, 이를 MLP 레이어를 거쳐 최종 랭킹 점수 $S_{CE}$를 연산한다.
* **레이턴시 및 비용 제어**:
  * 전체 $N \times N$ 연산은 현실적으로 불가능하므로, Stage 1 후보군을 $50$개 이하로 제한하여 실시간 서비스 오버헤드를 $50\text{ms}$ 이내로 통제한다.

#### [2] MMoE (Multi-Gate Mixture-of-Experts) Multi-Task Learning

* **목적**: "유사함"의 다차원 정의(인구통계학적 배경 일치 vs 성격 및 라이프스타일 지향점 일치)를 단일 랭킹 손실로 학습할 때 발생하는 상충(Task Conflict) 현상을 완화한다.
* **구조**:
  * 공통 입력 피처 공간 $X_{pair}$에 대해 $E$개의 공유 전문가(Shared Experts) 네트워크 $f_i(X)$를 둔다.
  * 복수의 Task 헤드를 정의한다:
    * **Task 1 (Demographic Match)**: 나이 차이, 성별, 지역, 직업 등의 인구통계 유사도 예측 ($y_1$)
    * **Task 2 (Lifestyle/Psychographic Match)**: 가치관, 야망, 커리어 지향성, 여가 활동 등 텍스트 표현 내 의미론적 라이프스타일 일치도 예측 ($y_2$)
  * 각 Task $k$는 고유의 게이팅 네트워크 $g^k(X)$를 가지며, 최종 출력은 다음과 같이 가중 합산된다:
    $$y_k = \sum_{i=1}^{E} g^k_i(X) \cdot f_i(X)$$
  * 최종 리랭킹 점수 $S_{MMoE}$는 Task별 가중합 또는 우선순위 게이트로 조합된다.

#### [3] Contrastive Persona Embedding

* **목적**: 유사도 판별의 기저가 되는 페르소나 표상(Representation) 자체를 고도화하기 위해 대비 학습(Contrastive Learning)을 적용한다.
* **학습 방식 및 Loss**:
  * Mini-batch 내에서 동일한 코어 속성(예: 동일 대분류 직군 및 관심사 카테고리)을 지닌 페르소나들을 Positive Pair($p_i, p_j$)로 설정하고, 나머지를 Negative Pair로 취급한다.
  * **InfoNCE Loss**를 활용해 페르소나 임베딩 공간을 최적화한다:
    $$\mathcal{L}_{InfoNCE} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{B} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k) / \tau)}$$
    (여기서 $z_i$는 페르소나 $i$의 Dense 임베딩, $\tau$는 temperature 하이퍼파라미터, $B$는 배치 크기)
  * 이를 통해 텍스트 원문뿐만 아니라 메타 구조가 정합적으로 캘리브레이션된 고품질 Dense 공간을 확보한다.

#### [4] LLM-as-a-judge 오프라인 평가

* **목적**: 레이블이 존재하지 않는(Weak Label 중심의) 유사도 도메인에서 모델 개선 시 실제 인간이 체감하는 정성적 품질 변화를 통계적으로 자동 계측한다.
* **프로토콜**:
  * 평가 데이터셋에서 무작위로 추출된 200개의 Source Persona와, 각 모델(Baseline vs Candidate)이 추천한 Top-5 Target Persona 쌍을 추출한다.
  * GPT-4o 또는 국산 고성능 LLM을 Judge로 기용하여 3대 축(구조 정합성, 가치관 유사성, 행동 시너지)을 1~5점 척도(Rubrics)로 채점한다.
  * 정량 지표(NDCG)와 LLM Judge 평점 간의 피어슨 상관계수(Pearson Correlation Coefficient $r$)를 측정하여 오프라인 평가 정밀도를 지속 검증한다. 목표 상관계수 $r \ge 0.75$를 달성해야 한다.

#### [5] Explainability-Guided Objective

* **목적**: 단순히 유사도 스코어만 높은 추천이 아니라, 사용자 화면(UI)에 "강력하고 설득력 있는 추천 사유(Strong Reason)" 카드를 띄울 수 있는 후보군을 상위로 인양한다.
* **수식 설계**:
  * 개별 Target 후보가 가지고 있는 강한 매칭 특징 개수(예: 동일 전문 분야, 동일 핵심 취미 등)의 잠재력 $C_{exp}(s, t) \in [0, 1]$을 정의한다.
  * 최종 리랭킹 점수 $S_{final}$은 순수 유사도 예측 점수 $S_{sim}$과 설명 가능성 보너스를 결합하여 연산한다:
    $$S_{final}(s, t) = S_{sim}(s, t) + \beta \cdot C_{exp}(s, t)$$
  * 하이퍼파라미터 $\beta$는 그리드 서치를 통해 오프라인 NDCG의 급격한 훼손 없이 설명 Coverage를 보장하도록 조정한다.

---

### 2. 프로덕션 승격 게이트 (Promotion Gates)

신규 리랭커 모델이 실제 배포용 `SIMILAR_TO` 관계 파이프라인으로 승격되기 위해서는 다음의 하드 필터 게이트를 충족해야 한다.

| 평가 차원 | 평가지표 | 승격 요구사항 (Promotion Threshold) | 비고 |
| :--- | :--- | :--- | :--- |
| **정확도** | `NDCG@10` | $\ge +0.005$ (기존 최선 Baseline 대비) | 검증 셋 기준 통계적 유의성 확보 |
| **설명력** | `Explanation Coverage@5` | $\ge 95\%$ | 상위 5개 결과 중 최소 1개 이상 강한 근거 노출 비율 |
| **다양성** | `Repeated Hub Rate@10` | $\le 10\%$ | 특정 마스터 페르소나가 과도하게 추천 허브로 중복 사용되는 비율 제어 |
| **보안** | `Leakage Audit Gate` | `Leakage Ratio = 0.00%` | 식별자(`uuid`, `name`)의 피처 유출 전수 감사 통과 |

```text
[FastRP/KNN Candidate] -> [Cross-Encoder/MMoE Rerank] -> [Explainability-Guided Objective Filter] -> [Audit Gate] -> [Deploy]
```

---

## 3. Appendix: Pre-implementation 4대 필수 조항

본격적인 대규모 ML 실험 및 유사 페르소나 추천 모델 코드 구축 직전, 데이터의 신뢰성 확보 및 프레임워크 한계 극복을 위해 다음 4대 구현 필수 조항을 준수해야 한다.

### [조항 1] 양방향 데이터 누수(Bidirectional Leakage) 방지 조항
* **배경 및 위험성**: 페르소나 유사도 추천 특성상, 데이터셋 내에 페르소나 $A$와 $B$ 간의 양방향 관계가 존재한다. 만약 단순 무작위 분할을 적용할 경우, 학습 데이터에 $(A \to B)$가 포함되고 검증/테스트 데이터에 $(B \to A)$가 포함되어 모델이 관계 구조를 단순히 암기하여 성능이 왜곡(Data Leakage)되는 현상이 발생한다.
* **구현 제약**:
  * 모델에 입력되는 유사도 페어 데이터셋을 구축할 때, 학습(Train)군과 검증/테스트(Val/Test)군은 개별 **페르소나 UUID 기준**으로 완전히 단절되어야 한다.
  * 즉, 특정 페르소나 $i$가 검증/테스트 셋의 `source_uuid` 또는 `target_uuid` 중 어느 하나라도 포함되어 있다면, 해당 페르소나 $i$와 연관된 모든 페어 데이터는 학습 데이터셋에서 전면 배제되어야 한다.
  * 이를 검증하기 위한 자동화된 Leakage Audit Unit Test(`test_leakage_audit`)를 파이프라인 진입부에서 강제 수행한다.

### [조항 2] 콜드 스타트(Cold-Start) Fallback 하이브리드 정책
* **배경 및 위험성**: Neo4j Graph Data Science(GDS) 기반의 **FastRP 임베딩**은 전형적인 Transductive 모델이다. 즉, 그래프 데이터베이스 상에 존재하지 않는 신규 생성 페르소나(Cold-Start)가 인입될 경우, 노드 임베딩 벡터가 존재하지 않아 후보 생성(Candidate Generation)이 완전히 불가능해지는 시스템 마비가 초래된다.
* **구현 제약**:
  * FastRP 임베딩 또는 GDS 그래프 노드가 존재하지 않는 신규 페르소나에 대해서는 다음과 같은 2단계 결정론적 하이브리드 Fallback 로직을 탑재한다.
  
  ```mermaid
  graph TD
      A[유사 추천 요청 인입] --> B{GDS 노드 존재 여부 검사}
      B -- Yes (Warm) --> C[FastRP KNN + Reranker 파이프라인]
      B -- No (Cold-Start) --> D[Fallback 1: Bi-Encoder KURE 텍스트 임베딩 Cosine 유사도]
      D --> E[Fallback 2: 인구통계학적 가중치 Rule-based 랭킹 합산]
      E --> F[최종 유사 페르소나 Top-N 추천 출력]
  ```
  
  * **Fallback Step 1**: KURE-v1 기반 Bi-Encoder 텍스트 임베딩 공간에서 계산된 Cosine Similarity 점수 산출
  * **Fallback Step 2**: 나이, 직업 대분류, 지역 가중치를 결합한 결정론적(Deterministic) 룰베이스 매칭 스코어링 가중합 계산
  * 해당 하이브리드 Fallback 모듈은 기존 Reranker 및 API 프론트와 완벽히 결합되어야 하며, 시스템 오류 로그 없이 200 OK 응답을 보장해야 한다.

### [조항 3] GroupKFold 랭킹 검증(Ranking Validation) 제약
* **배경 및 위험성**: 유사도 리랭커 모델(예: LambdaRank, Cross-Encoder) 평가 시, 여러 Target 페르소나가 동일한 Source 페르소나 쿼리 하에 랭킹 그룹으로 묶이게 된다. 쿼리가 Train/Val 간에 흩어져 있으면 그룹 구조가 훼손되고 NDCG@5/10 지표가 왜곡되어 테스트 셋 성능 예측이 불가하다.
* **구현 제약**:
  * 랭킹 검증 및 최적화를 위해 **`GroupKFold`** 분할 방식을 강제하며, 그룹 Key는 반드시 **`source_uuid`**로 지정한다.
  * 동일한 `source_uuid`를 공유하는 모든 Target 페어들은 동일한 Fold 내에 함께 묶여 움직여야 하며, 절대 Train Fold와 Validation Fold에 분할 교차되어 할당될 수 없다.
  * NDCG@5 및 NDCG@10은 개별 `source_uuid` 그룹 내에서 랭킹을 매긴 후 평균을 취하는 Group-wise NDCG 방식으로 산출한다.

### [조항 4] 하이퍼파라미터 그리드 서치(Hyperparameter Grid Search) 바운더리
* **배경 및 자원 제한**: 본 프로젝트의 ML 학습 인프라(Intel Core Ultra 7 CPU 18스레드, NVIDIA RTX 4060 8GB VRAM) 하에서 과도한 차원의 그리드 서치는 Out Of Memory(OOM)나 시스템 멈춤을 초래한다. 따라서 하드웨어 스펙에 부합하면서도 성능 개선을 최대화할 수 있는 핵심 탐색 공간을 제한하여 명시한다.
* **구현 제약**:
  * LightGBM / LambdaRank 모델의 그리드 서치 바운더리는 다음과 같이 엄격하게 통제한다.
  
  | 파라미터명 | 권장 탐색 바운더리 | 비고 |
  | :--- | :--- | :--- |
  | `num_leaves` | `[15, 31, 63]` | 과적합 방지 및 VRAM 절약을 위한 제약 |
  | `max_depth` | `[4, 6, 8, -1]` | 트리 깊이 한계 설정 |
  | `learning_rate` | `[0.01, 0.05, 0.1]` | 경사 하강 보폭 제어 |
  | `n_estimators` | `[100, 200, 500]` | `early_stopping_rounds=30`과 병행 사용 필수 |
  | `min_child_samples` | `[10, 20, 50]` | 단말 노드 최소 데이터 수 |

  * 학습 시 Multi-threading 제약: `num_threads=18` 설정을 고수하여 OS 백그라운드 프로세스 및 Docker Neo4j 컨테이너 구동을 위한 CPU 여유분을 보장한다.
