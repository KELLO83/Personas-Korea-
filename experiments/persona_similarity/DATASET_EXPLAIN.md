# 유사 페르소나 추천 데이터 설명

## 가장 중요한 한 줄

유사 페르소나 모델의 학습 데이터는 사람 1명 단위가 아니라, `source_person -> target_person` 후보쌍 단위이다.

```text
source_uuid | target_uuid | label | feature...
```

즉 모델은 다음 질문을 학습한다.

```text
source 사람을 보고 있을 때
target 사람을 유사 페르소나로 얼마나 높게 랭킹할 것인가?
```

## 현재 데이터 범위

현재 Neo4j DB는 원본 100만 페르소나 전체가 아니다.

```text
현재 DB: 50,000 Person
연령대: 10대 / 20대 / 30대 중심
```

현재 Excel export 기준 주요 분포:

```text
Person rows: 50,000

age_group:
  20대: 22,842
  30대: 16,667
  10대: 10,491

sex:
  남자: 26,141
  여자: 23,859

province:
  경기: 13,743
  서울: 10,466
  인천: 2,980
  부산: 2,933
  경상남: 2,669

occupation:
  무직: 20,914
  그 외 long-tail 직업 1,423종

education:
  4년제 대학교: 18,131
  고등학교: 15,824
  2~3년제 전문대학: 11,733

field:
  결측: 29,700
  즉 약 59.4%가 전공 field 없음

skill_count:
  현재 전부 0

hobby_count:
  평균 약 2.79개

similar_to_out_count:
  현재 전부 5
  즉 현재 GDS topK=5로 생성된 상태
```

중요:

```text
현재 topK=5 상태는 reranker 실험에 너무 좁다.
실제 유사페르소나 reranker 실험 전에는 GDS topK=50 이상으로 다시 만들어야 한다.
```

## 원천 그래프 구조

Neo4j에는 Person과 여러 속성 노드가 연결되어 있다.

```text
(:Person)
  -[:LIVES_IN]->(:District)-[:IN_PROVINCE]->(:Province)-[:IN_COUNTRY]->(:Country)
  -[:WORKS_AS]->(:Occupation)
  -[:EDUCATED_AT]->(:EducationLevel)
  -[:MAJORED_IN]->(:Field)
  -[:MARITAL_STATUS]->(:MaritalStatus)
  -[:MILITARY_STATUS]->(:MilitaryStatus)
  -[:LIVES_WITH]->(:FamilyType)
  -[:LIVES_IN_HOUSING]->(:HousingType)
  -[:ENJOYS_HOBBY|LIKES]->(:Hobby)
  --> (:Skill)  # 현재 DB에서는 Skill edge가 사실상 없음
```

GDS 실행 후 Person에는 다음 정보가 추가된다.

```text
(:Person).fastrp_embedding
(:Person).community_id
(:Person)-[:SIMILAR_TO {score}]->(:Person)
```

`SIMILAR_TO.score`는 학습 label이 아니라 후보생성 점수이다.

## 왜 모든 Person x Person을 만들지 않는가?

5만 명만 해도 모든 directed pair는 약 25억 개다.

```text
50,000 * 49,999 ~= 2.5B pairs
```

따라서 모든 사람쌍을 학습하지 않는다.

대신 GDS가 만든 `SIMILAR_TO` 후보만 export한다.

```text
source Person -> topN similar target Persons
```

추천 후보 수:

```text
topK=5: smoke 검증용
topK=50: 1차 실제 실험용
topK=100: recall/품질 비교용
```

## 학습 row의 형태

한 row는 다음 의미를 가진다.

```text
source_uuid 사람이 있을 때
target_uuid 사람을 얼마나 유사한 후보로 볼 것인가?
```

예시:

```text
source_uuid | target_uuid | label | fastrp_score | age_diff | same_occupation | shared_hobby_count
A           | B           | 0.72  | 0.91         | 2        | 1               | 2
A           | C           | 0.31  | 0.66         | 8        | 0               | 0
A           | D           | 0.18  | 0.51         | 12       | 0               | 0
```

모델은 같은 `source_uuid` 안에서 target 후보들의 순서를 학습한다.

```text
group = source_uuid
```

## export되는 원본 후보쌍 컬럼

`export_pairs.py`가 Neo4j에서 export하는 원본 pair 컬럼이다.

| 컬럼 | 타입 | 의미 |
| --- | --- | --- |
| `source_uuid` | string | 기준 사람 |
| `target_uuid` | string | 후보 유사 페르소나 |
| `fastrp_score` | float | `SIMILAR_TO.score` |
| `source_age` | int/null | source 나이 |
| `target_age` | int/null | target 나이 |
| `source_age_group` | string/null | source 연령대 |
| `target_age_group` | string/null | target 연령대 |
| `source_sex` | string/null | source 성별 |
| `target_sex` | string/null | target 성별 |
| `source_province` | string/null | source 시/도 |
| `target_province` | string/null | target 시/도 |
| `source_district` | string/null | source 구/군 |
| `target_district` | string/null | target 구/군 |
| `source_occupation` | string/null | source 직업 |
| `target_occupation` | string/null | target 직업 |
| `source_education` | string/null | source 학력 |
| `target_education` | string/null | target 학력 |
| `source_field` | string/null | source 전공/분야 |
| `target_field` | string/null | target 전공/분야 |
| `source_marital` | string/null | source 혼인 상태 |
| `target_marital` | string/null | target 혼인 상태 |
| `source_family` | string/null | source 가족 형태 |
| `target_family` | string/null | target 가족 형태 |
| `source_housing` | string/null | source 주거 형태 |
| `target_housing` | string/null | target 주거 형태 |
| `source_community_id` | int/null | source Leiden community |
| `target_community_id` | int/null | target Leiden community |
| `shared_hobbies` | JSON list[string] | 두 사람이 공유하는 취미 이름 |
| `shared_skills` | JSON list[string] | 두 사람이 공유하는 스킬 이름 |

이 raw 컬럼은 사람이 읽고 audit하기 위한 값도 포함한다. 모델에는 그대로 넣지 않고 숫자 feature로 변환한다.

## 1차 구조화 학습 feature

현재 코드에서 바로 학습에 넣는 숫자 feature는 다음이다.

| feature | 타입 | 의미 |
| --- | --- | --- |
| `fastrp_score` | float | FastRP/KNN 후보생성 점수 |
| `age_diff` | float | 나이 차이 |
| `same_age_group` | 0/1 | 같은 연령대인지 |
| `same_sex` | 0/1 | 같은 성별인지 |
| `same_province` | 0/1 | 같은 시/도인지 |
| `same_district` | 0/1 | 같은 구/군인지 |
| `same_occupation` | 0/1 | 같은 직업인지 |
| `same_education` | 0/1 | 같은 학력인지 |
| `same_field` | 0/1 | 같은 전공/분야인지 |
| `same_marital` | 0/1 | 같은 혼인 상태인지 |
| `same_family` | 0/1 | 같은 가족 형태인지 |
| `same_housing` | 0/1 | 같은 주거 형태인지 |
| `same_community` | 0/1 | 같은 GDS community인지 |
| `shared_hobby_count` | int | 공유 취미 개수 |
| `shared_skill_count` | int | 공유 스킬 개수 |
| `explanation_feature_count` | int | 설명 가능한 공통 feature 개수 |

모델에 넣지 않는 컬럼:

```text
source_uuid
target_uuid
display_name
uuid
원문 문장 텍스트
```

`uuid`는 split/group/review용이지 feature가 아니다.

## feature 품질 해석

모든 feature의 의미가 같은 것은 아니다.

강한 구조화 신호:

```text
same_occupation
same_district
same_education
same_field
shared_hobby_count
shared_skill_count
```

중간 신호:

```text
same_family
same_housing
same_province
same_community
same_age_group
```

약한 신호:

```text
same_sex
same_marital 단독
same_province 단독
same_community 단독
```

주의:

```text
모델이 성별/혼인상태/도 단위 지역만 보고 추천을 잘한다고 나오면
실제 유사성 품질은 낮을 수 있다.
```

## 문장형 데이터

현재 Person에는 구조화 feature 외에 자연어 서술 컬럼이 있다.

| 컬럼 | 의미 |
| --- | --- |
| `persona` | 전체 페르소나 서술 |
| `professional_persona` | 직업/업무 성향 |
| `sports_persona` | 스포츠/활동 성향 |
| `arts_persona` | 예술/문화 성향 |
| `travel_persona` | 여행 성향 |
| `culinary_persona` | 음식/요리 성향 |
| `family_persona` | 가족/관계 성향 |
| `cultural_background` | 문화적 배경 |
| `career_goals_and_ambitions` | 커리어 목표/태도 |
| `skills_and_expertise` | 스킬/전문성 서술 |
| `hobbies_and_interests` | 취미/관심사 서술 |

이 문장들은 유사 페르소나에서 매우 중요할 수 있다.

예:

```text
퇴근 후 혼자 독서와 산책을 즐긴다.
운동 모임에서 새로운 사람을 만나는 것을 좋아한다.
가족과 보내는 시간을 가장 중요하게 생각한다.
커리어 성장을 위해 자격증 공부를 꾸준히 한다.
```

이런 정보는 나이, 학력, 군필, 주거 형태보다 실제 성향 유사성에 더 가까울 수 있다.

## 문장 원문을 그대로 넣지 않는 이유

LightGBM에는 문장 원문을 직접 넣지 않는다.

이유:

- tree model은 긴 자연어 원문을 바로 처리하지 못한다.
- 문장 패턴을 외우거나 중복 정보를 과하게 사용할 수 있다.
- 설명과 leakage audit이 어려워진다.

대신 문장을 embedding으로 바꾸고, source-target 사이의 cosine similarity를 feature로 넣는다.

```text
source_text -> embedding
target_text -> embedding
cosine(source_embedding, target_embedding) -> feature
```

## 2차 semantic feature 후보

문장 embedding 실험에서 추가할 feature 후보:

| feature | 의미 |
| --- | --- |
| `all_text_cosine` | 모든 문장 서술을 합친 전체 의미 유사도 |
| `persona_text_cosine` | 전체 페르소나 서술 유사도 |
| `professional_text_cosine` | 직업/업무 성향 유사도 |
| `hobbies_text_cosine` | 취미/관심사 문장 유사도 |
| `skills_text_cosine` | 스킬/전문성 문장 유사도 |
| `career_text_cosine` | 커리어 목표 유사도 |
| `family_text_cosine` | 가족/관계 성향 유사도 |
| `lifestyle_text_cosine` | 여가/생활패턴 유사도 |

중요한 설계:

```text
KURE/text embedding은 후보생성기가 아니라 reranker feature로 먼저 사용한다.
```

## 취미추천 KURE 결과를 유사페르소나에 그대로 적용하면 안 되는 이유

취미추천에서는 KURE semantic Stage1 provider가 성능을 떨어뜨렸다.

이유:

```text
취미추천 = held-out hobby item을 맞히는 문제
KURE 후보생성 = 정답 취미를 candidate pool에서 놓칠 수 있음
candidate_recall@50 하락 -> Recall/NDCG 하락
```

하지만 유사페르소나는 다르다.

```text
유사페르소나 = 사람과 사람의 생활패턴/성향/가치관 유사성 문제
문장 서술이 핵심 feature일 가능성이 큼
```

따라서 유사페르소나에서는:

```text
KURE/text embedding candidate generation: 비추천
KURE/text embedding reranker feature: 추천 실험
```

## weak label

현재 사람-사람 유사성에 대한 human label은 없다.

따라서 1차 label은 weak label이다.

현재 구조화 weak label:

```text
label =
  0.25 * same_occupation
  0.15 * same_province
  0.10 * same_district
  0.10 * same_age_group
  0.10 * same_education
  0.10 * same_community
  0.12 * min(shared_hobby_count, 5) / 5
  0.08 * min(shared_skill_count, 5) / 5
  0.10 * normalized_fastrp_score
```

한계:

- 이 label은 진짜 사용자 선호가 아니다.
- 구조화 feature와 같은 정보를 label이 쓰므로 평가가 편향될 수 있다.
- 문장 의미 유사성을 충분히 반영하지 못한다.

따라서 text feature 실험에서는 manual review가 특히 중요하다.

## split 방식

row 단위로 split하면 안 된다.

반드시 `source_uuid` 기준으로 split한다.

```text
train: 80% source_uuid
valid: 10% source_uuid
test: 10% source_uuid
seed: 42
```

이유:

```text
같은 source 사람이 train과 valid/test에 동시에 나오면
query-person 단위 일반화 평가가 깨진다.
```

## 학습 데이터 최종 형태

1차 구조화 모델:

```text
source_uuid
target_uuid
label
split
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
deterministic_score
```

2차 semantic 모델에서는 여기에 다음이 추가된다.

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

모델 feature로 들어가는 것은 `label`, `split`, `uuid`가 아니라 feature 컬럼들이다.

## 평가 방식

평가는 반드시 `source_uuid`별 ranking으로 한다.

```text
source_uuid = A
  target B score
  target C score
  target D score
```

비교 대상:

```text
FastRP score order
deterministic score order
structured LightGBM order
text-only LightGBM order
structured+text LightGBM order
hybrid score order
```

## 필요한 artifact

| 파일 | 의미 |
| --- | --- |
| `candidate_pairs.parquet` | Neo4j에서 export한 source-target 후보쌍 |
| `pair_features.parquet` | 학습용 숫자 feature + weak label |
| `splits.json` | source_uuid 기준 split |
| `*_metrics.json` | 실험별 평가 결과 |
| `*_manual_review.csv` | 사람이 직접 볼 추천 샘플 |
| `*_train_metadata.json` | 모델 학습 설정/feature importance |
| `*.txt` | LightGBM 모델 파일 |

## 현재 데이터에서 가장 중요한 주의점

1. 현재 `SIMILAR_TO`는 topK=5 상태다.
   - 본 실험 전 topK=50 재생성이 필요하다.

2. `Skill` feature는 현재 거의 죽어 있다.
   - `skill_count=0`이므로 `shared_skill_count`는 성능에 기여하기 어렵다.

3. `field`는 결측이 많다.
   - 약 59.4% 결측.

4. `occupation`은 long-tail이고 `무직`이 매우 많다.
   - 같은 직업 feature가 일부 후보에 과도하게 작동할 수 있다.

5. 문장형 feature는 유사페르소나에 중요할 가능성이 높다.
   - 단, 원문 직접 입력이 아니라 embedding cosine으로 넣어야 한다.

## 현재 결론

데이터 형태상 가장 합리적인 실험 순서는 다음이다.

```text
1. FastRP/KNN topK=50으로 후보쌍 확보
2. 구조화 pair feature 생성
3. FastRP baseline 평가
4. deterministic baseline 평가
5. structured LightGBM ranking 모델 평가
6. 문장 embedding cosine feature 추가
7. text-only / structured+text / hybrid 비교
8. manual review로 실제 의미 유사성 확인
```

최종적으로 봐야 할 것은 단순 metric이 아니다.

```text
이 두 사람이 왜 비슷한가?
그 이유가 사용자에게 납득되는가?
구조화 속성만 같은 얕은 추천인가?
문장 의미상 생활패턴/성향/가치관도 비슷한가?
```
