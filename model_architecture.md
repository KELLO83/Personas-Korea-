# 모델 아키텍처 문서

> **대상 독자**: AI 모델 개발 입문자
> **목적**: 데이터 전처리부터 최종 추천까지 전체 파이프라인 설명
> **버전**: v2 LightGBM default 기준

---

## 1. 데이터 이해하기

### 1.1 원천 데이터

Nemotron-Personas-Korea 데이터셋. HuggingFace에 있는 **한국인 100만명의 가상 프로필**.

각 사람은 이런 정보를 가짐:

```
person_uuid: "abc-123-def"        ← 사람 식별자
age: "32"                          ← 나이
age_group: "30s"                   ← 연령대
sex: "남성"                        ← 성별
occupation: "소프트웨어 엔지니어"   ← 직업
province: "서울특별시"             ← 도/광역시
district: "강남구"                 ← 시/군/구
hobbies_and_interests_list: "코딩, 등산, 영화 감상, 요리"  ← 취미 목록
persona_text: "저는 활동적인 라이프스타일을 즐깁니다..."  ← 페르소나 텍스트
...
```

### 1.2 프로젝트에서 사용하는 데이터

100만명 중 **10,000명을 샘플링**하여 CSV 2개로 저장:

| 파일 | 내용 | 크기 |
|------|------|------|
| `person_hobby_edges.csv` | (person_uuid, hobby_name) 쌍 — 누가 무슨 취미를 가지는지 | 44,084개 행 |
| `person_context.csv` | 각 person의 인구통계 + 페르소나 텍스트 (21개 컬럼) | 10,000명 |

**person_context.csv 컬럼 (21개)**:

```
person_uuid | age | age_group | sex | occupation | district | province
family_type | housing_type | education_level
persona_text | professional_text | sports_text | arts_text
travel_text | culinary_text | family_text | hobbies_text
skills_text | career_goals | embedding_text
```

### 1.3 그래프(Graph)로 이해하기

이 데이터는 **그래프 구조**:

```
          ┌───────┐              ┌──────┐
          │ 영희   │              │ 독서  │
          │ (Person) │──[좋아함]──→│ (Hobby)│
          └───────┘              └──────┘
               │                     ↑
               │──[좋아함]──→ 요리     │
               │                ┌──────┐
               │                │      │
          ┌───────┐             │      │
          │ 철수   │──[좋아함]──→│      │
          │ (Person)│──[좋아함]──→ 등산  │
          └───────┘             └──────┘
```

- **Node(노드/정점)**: Person(10,000명) + Hobby(180개) = 총 10,180개 노드
- **Edge(엣지/간선)**: `(Person) → (Hobby)` 관계 = 40,743개 엣지

이 구조가 추천 시스템의 핵심: "영희가 독서를 좋아한다면, 독서를 좋아하는 철수가 가진 등산도 영희가 좋아할까?"

---

## 2. 데이터 전처리 (Preprocessing)

### 2.1 단계별 흐름

```
                       데이터 전처리 흐름도
                           
 CSV 파일 로드 ───→ 취미 이름 정규화 ───→ 별칭 매핑
                            │
                            ↓
               Taxonomy 규칙 적용 (30개 규칙)
                            │
                            ↓
                    희소 취미 제거 (3회 미만 DROP)
                            │
                            ↓
             27,137개 → 2,302개 → 180개 취미
                            │
                            ↓
               문자열 → 숫자 ID 변환 (인덱싱)
                            │
                            ↓
              80/10/10 Train/Validation/Test 분할
                            │
                            ↓
                 Hobby Profile 생성 (통계 정보)
```

### 2.2 상세: 취미 이름 정규화

```python
# normalize_hobby_name() 함수가 하는 일
입력: "  저 녁   산 책  "
  ① NFKC 유니코드 정규화 → "  저 녁   산 책  "
  ② 소문자 변환 → "  저 녁   산 책  "
  ③ 공백 축소 (정규식 \s+ → " ") → "저 녁 산 책"
  출력: "저 녁 산 책"
```

### 2.3 상세: 별칭 매핑 (Alias)

같은 취미인데 표현이 다른 경우를 **수동으로** 매핑:

```json
{
  "코인 노래방 방문": "노래방",
  "달리기": "러닝",
  "볼링장 가기": "볼링"
}
```

최우선 적용: "코인 노래방 방문" → 바로 "노래방"으로 변환

### 2.4 상세: Taxonomy 규칙 (30개 규칙)

키워드 기반 매칭으로 더 다양한 표현을 하나로 통일:

```
규칙 예시: "산책"으로 통일하는 규칙
  
  include_keywords: ["산책", "걷기", "둘레길", "공원"]
  exclude_keywords: ["자전거", "러닝", "트레킹", "등산"]

  → "저녁 산책"       → include("산책") 매칭 → "산책"
  → "한강 공원 걷기"  → include("걷기") 매칭 → "산책"
  → "둘레길 트레킹"   → include("둘레길") 매칭, exclude("트레킹") 없으면 "산책"
  → "자전거 산책"     → include("산책") 매칭 BUT exclude("자전거") 있음 → 변환 안 함
```

### 2.5 상세: 희소 취미 제거 (min_item_degree=3)

```
조건: 3명 미만이 가진 취미는 전부 제거

이유: "우주여행"을 고작 1명만 가졌다면?
  → 이 취미를 다른 사람에게 추천할 근거 부족
  → 모델이 학습할 패턴이 없음
  → 과감히 제거

효과:
  정규화 전: 27,137개 고유 취미
  정규화 후:  2,302개 고유 취미  (같은 취미끼리 합쳐짐)
  제거 후:      180개 고유 취미  (희소한 것들 제거)
  
  엣지 수도 감소:
  44,084개 → 40,743개 (3,341개 손실, 약 7.6%)
```

### 2.6 상세: 문자열 → 숫자 ID 변환 (인덱싱)

컴퓨터는 문자열을 직접 처리 못 함. 전부 숫자 ID로 변환:

```python
# person_to_id: 사람 UUID 문자열 → 정수 ID
{
  "abc-123-def": 0,    # 영희
  "ghi-789-jkl": 1,    # 철수
  ...
}
# 총 10,000명 (0 ~ 9,999)

# hobby_to_id: 취미 이름 문자열 → 정수 ID
{
  "독서": 0,
  "요리": 1,
  "등산": 2,
  "러닝": 3,
  ...
}
# 총 180개 (0 ~ 179)

# edges: (사람_ID, 취미_ID) 쌍의 리스트
[
  (0, 0),   # 영희 - 독서
  (0, 1),   # 영희 - 요리
  (1, 0),   # 철수 - 독서
  (1, 2),   # 철수 - 등산
  ...
]
# 총 40,743개 엣지
```

### 2.7 상세: 데이터 분할

각 사람별로 **취미를 섞어서** 3덩이로 나눔:

```
각 사람의 취미 리스트를 셔플한 후:
  
  ┌──────────────────────────────────────────────────┐
  │  80% → TRAIN  (32,594 edges)  ← 모델이 학습      │
  │  10% → VAL    ( 4,074 edges)  ← 중간 성능 확인    │
  │  10% → TEST   ( 4,075 edges)  ← 최종 성능 평가    │
  └──────────────────────────────────────────────────┘

중요 변수:
  train_known: {person_id → set[train_hobby_ids]}
    → "이 사람이 이미 아는 취미" (추천할 때 제외)
  
  full_known: {person_id → set[all_hobby_ids]}
    → "이 사람의 모든 취미" (부정 샘플링할 때 제외)
```

분할 규칙:
- 취미 3개 이상: 80/10/10 분할 (최소 1개는 holdout)
- 취미 2개: 모두 train (two_hobby_policy = "train_only")
- 취미 1개: train (holdout 불가)

### 2.8 상세: Hobby Profile 생성

각 취미의 **통계 이력서**를 train split만 사용하여 생성:

```python
hobby_profile = {
    "source": "train_split_only",       # train만 사용 (정보 누출 방지)
    "num_hobbies": 180,
    "hobbies": {
        "독서": {
            "hobby_id": 0,
            "train_popularity": 523,    # train에서 523명이 가짐
            
            "distributions": {          # 이 취미를 가진 사람들의 분포
                "age_group": {
                    "10s": 50,
                    "20s": 200,
                    "30s": 180,
                    "40s": 70,
                    "50s": 23
                },
                "occupation": {
                    "학생": 150,
                    "엔지니어": 100,
                    "디자이너": 80,
                    ...
                },
                "province": {
                    "서울특별시": 200,
                    "경기도": 100,
                    ...
                },
                "district": { ... },
                "family_type": { ... },
                "housing_type": { ... },
                "education_level": { ... }
            },
            
            "cooccurring_hobbies": [    # 같이 등장하는 취미 top-20
                {"hobby_name": "영화 감상", "count": 300},
                {"hobby_name": "요리",      "count": 250},
                {"hobby_name": "카페 가기", "count": 200},
                ...
            ]
        },
        "요리": { ... },
        ...
    }
}
```

### 2.9 상세: Leakage Audit (정보 누출 검사)

```
검사 내용:
  VAL/TEST에 있는 취미 이름이 persona text field에 등장하는가?
  
  예: 어떤 사람의 TEST 취미가 "요리"인데, 
      persona_text에 "저는 요리를 좋아합니다"라고 적혀 있다면?
      → 모델이 cheat 할 수 있음 (시험 보면서 정답지 보는 꼴)

결과:
  29.3%의 holdout 엣지에서 누출 발견
  → "persona_text_fit" feature를 아예 사용하지 않기로 결정
```

---

## 3. Stage 1: 후보 생성 (Candidate Generation)

### 3.1 개념

**목표**: 전체 180개 취미 중에서 "이 사람에게 적합할 만한" 후보 50개로 1차 필터링

**방식**: 2개의 통계 기반 제공자(Provider) 사용 (학습 없음)

```
                        Stage 1 흐름도

       ┌────────────────────────────────────────────┐
       │              Stage 1: 후보 생성              │
       │                                            │
       │  ┌────────────────┐  ┌────────────────┐    │
       │  │  인기도 제공자   │  │  동시출현 제공자  │    │
       │  │ (Popularity)   │  │ (Co-occurrence)│    │
       │  └───────┬────────┘  └───────┬────────┘    │
       │          │                    │             │
       │          └──────┬─────────────┘             │
       │                 │                           │
       │      rank_percentile 정규화 [0 ~ 1]         │
       │                 │                           │
       │      merge_candidates_by_hobby()            │
       │      (중복 제거 + 최고 점수 채택)            │
       │                 │                           │
       │         최대 50개 HobbyCandidate             │
       └─────────────────┬──────────────────────────┘
                         │
                         ↓
                    Stage 2 로 전달
```

### 3.2 제공자 1: 인기도 (Popularity)

**아이디어**: "전체적으로 인기 있는 취미를 일단 추천해보자"

**방법**:

```python
# 1. train 데이터에서 각 취미 등장 횟수 카운트
counts = {
    "독서": 523,    # train에서 523명이 가진 취미
    "요리": 480,
    "등산": 410,
    "영화 감상": 380,
    "러닝": 350,
    ...
}

# 2. 횟수 기준 내림차순 정렬
ranked = ["독서", "요리", "등산", "영화 감상", "러닝", ...]

# 3. "이미 아는 취미(train_known)"는 제외
#    영희가 "독서"를 이미 안다면, "독서"는 추천 불가
#    → 영희의 추천 목록: "요리", "등산", "영화 감상", "러닝", ...

# 4. 상위 50개 유지
```

**한계**: 모든 사람에게 똑같은 인기 취미만 추천. 개인화 전혀 없음.

### 3.3 제공자 2: 동시출현 (Co-occurrence)

**아이디어**: "이 사람이 이미 가진 취미와 **자주 함께 등장하는** 취미를 추천하자"

**사전 계산 (동시출현 행렬)**:

```python
# 취미 A와 취미 B가 같은 사람에게 함께 등장한 횟수
cooccurrence = {
    "독서": {"영화 감상": 300, "요리": 200, "카페 가기": 180, ...},
    "요리": {"영화 감상": 250, "배달 음식": 180, "독서": 200, ...},
    "등산": {"러닝": 280, "캠핑": 200, "자전거": 150, ...},
    ...
}
```
```


**추론 (영희가 독서를 좋아할 때)**:

```python
# 영희의 train_known = {"독서", "요리"}
# 각 후보 취미의 점수 계산:

# "영화 감상" 점수 = cooccurrence["독서"]["영화 감상"] + cooccurrence["요리"]["영화 감상"]
#                    = 300 + 250 = 550

# "카페 가기" 점수 = cooccurrence["독서"]["카페 가기"] + cooccurrence["요리"]["카페 가기"]
#                    = 180 + 120 = 300

# 등산 점수 = cooccurrence["독서"]["등산"] + cooccurrence["요리"]["등산"]
#              = 5 + 3 = 8  (거의 없음)

# → 점수 높은 순으로 50개: 영화 감상(550), 카페 가기(300), 배달 음식(250), ...

# train_known("독서", "요리") 제외
```

**한계**: 새로운 취미나 잘 안 알려진 취미는 동시출현 횟수가 적어서 추천되기 어려움.

### 3.4 점수 정규화: rank_percentile

두 제공자의 점수 체계가 다르므로 **동일한 기준**으로 변환:

```python
# 인기도 점수 예시: [523, 480, 410, 380, 350, ...]  (절대 빈도)
# 동시출현 점수 예시: [550, 300, 250, ...]            (상대 빈도)

# rank_percentile 정규화:
#   1등 = 1.0
#   꼴등 = 0.0
#   그 사이는 선형 보간

# 인기도 정규화 결과 (50개 중):
#   1등: 요리 → 1.0
#   2등: 등산 → (49/49) = 1.0 - (1/49) ≈ 0.9796
#   ...
#   50등: ... → 0.0

# 동시출현 정규화 결과 (50개 중):
#   1등: 영화 감상 → 1.0
#   2등: 카페 가기 → 0.9796
#   ...
```

### 3.5 후보 병합: merge_candidates_by_hobby

```python
# 영희의 후보 목록 (정규화 점수):
#   인기도 제공자: [요리(1.0), 등산(0.98), 영화 감상(0.96), ...]
#   동시출현 제공자: [영화 감상(1.0), 카페 가기(0.98), 요리(0.96), ...]

# 같은 취미 = 같은 취미 ID 로 그룹화:
#   "요리":    인기도(1.0),  동시출현(0.96)  → 채택: 1.0 (인기도)
#   "영화 감상": 인기도(0.96), 동시출현(1.0)  → 채택: 1.0 (동시출현)
#   "등산":    인기도(0.98),  동시출현(없음)  → 채택: 0.98
#   "카페 가기": 인기도(없음), 동시출현(0.98)  → 채택: 0.98

# source_scores에 모든 제공자 원점수 기록 (Stage 2가 활용):
#   "요리": source_scores = {"popularity": 1.0, "cooccurrence": 0.96}
#   "영화 감상": source_scores = {"popularity": 0.96, "cooccurrence": 1.0}

# 최종 50개 유지
```

### 3.6 Stage 1의 최종 출력

```python
HobbyCandidate = {
    "hobby_id": 1,          # 취미 ID (0~179)
    "hobby_name": "요리",   # 취미 이름
    "source_scores": {      # 정규화 점수
        "popularity": 1.0,
        "cooccurrence": 0.96
    },
    "raw_source_scores": {  # 원점수
        "popularity_raw": 480,
        "cooccurrence_raw": 200
    },
    "reason_features": {}   # 디버깅 정보
}
```

**Stage 1은 모델 학습 없음**. 순수 통계 계산만으로 50개 후보 선별.

---

## 4. Stage 2: LightGBM Learned Ranker (순위모델)

### 4.1 개념

**목표**: Stage 1의 50개 후보에서 13개 특징(feature)을 계산하고, LightGBM 모델이 각 후보의 점수를 예측하여 **최종 10개** 선별

**Stage 1과의 차이**:

| 구분 | Stage 1 | Stage 2 |
|------|---------|---------|
| 방식 | 통계 기반 규칙 | **머신러닝 모델 (학습됨)** |
| 입력 | 엣지 정보만 | 13개 feature (엣지 + 인구통계 + 분포) |
| 학습 | 없음 | LightGBM이 AUC 최적화 학습 |
| 개인화 | 제한적 | 높음 (13개 feature가 개인 특성 반영) |

```
                      Stage 2 흐름도

    Stage 1의 50개 HobbyCandidate
               │
               ↓
    각 후보마다 13개 Feature 계산
    (PersonContext + HobbyProfile 활용)
               │
               ↓
    13개 숫자 → LightGBM 모델 → 예측 확률 (0~1)
               │
               ↓
    확률 높은 순으로 10개 최종 선별
```

### 4.2 13개 Feature 상세

각 후보 취미마다 13개의 Feature(특징/숫자)를 계산:

```
후보 취미: "요리"
Person: 영희 (30대, 여성, 서울, 디자이너)
```

#### Feature 1: cooccurrence_score (동시출현 점수, 0~1)

```python
# 영희의 train_known = {"독서", "필라테스"}
# cooccurrence["요리"] = {"영화 감상": 250, "독서": 200, "배달 음식": 180, ...}

known_hobby_compatibility = 
  (cooccurrence["요리"]["독서"] + cooccurrence["요리"]["필라테스"]) 
  / sum(cooccurrence["요리"])
  
= (200 + 5) / (250 + 200 + 180 + ...) ≈ 0.35

# 하지만 cooccurrence_score는 source_scores에서 직접 가져옴:
cooccurrence_score = 0.96  (Stage 1에서 정규화된 점수)
```

#### Feature 2: segment_popularity_score (세그먼트 인기도, 0~1)

```python
# "요리"의 분포에서 영희의 인구통계와 일치하는 비율 합산

age_group_fit = 
  distributions["요리"]["age_group"]["30s"] / sum(distributions["요리"]["age_group"])
  = 180 / (50 + 200 + 180 + 70 + 23) ≈ 0.34
  # "요리"를 좋아하는 사람 중 30대 비율 34%

occupation_fit =
  distributions["요리"]["occupation"]["디자이너"] / sum(...)
  = 80 / (150 + 100 + 80 + ...) ≈ 0.12

region_fit = max(province_fit, district_fit)
  = max(0.30, 0.08) = 0.30

# (참고: score_rerank_features에서 age/occupation/region fit은 
#  별도 feature로 들어감 → 위 예시는 설명용)

segment_popularity_score = 0.72
  # source_scores에서 가져온 값
```

#### Feature 3: known_hobby_compatibility (취미 적합도, 0~1)

```python
# "요리"와 함께 등장하는 취미(top-20) 중에
# 영희가 이미 아는 취미("독서", "필라테스")가 얼마나 포함되는가?

cooccurring_hobbies = [
    {"hobby_name": "영화 감상", "count": 250},
    {"hobby_name": "독서",      "count": 200},   # ← 영희가 아는 취미!
    {"hobby_name": "배달 음식", "count": 180},
    ...
]

known_hobby_compatibility = 
  200 / (250 + 200 + 180 + ...) ≈ 0.18

# 영희가 아는 취미("독서") 1개가 포함되어 있음
# "요리"는 "독서"와 자주 함께 등장하므로 적합도가 어느 정도 있음
```

#### Feature 4: age_group_fit (연령대 적합도, 0~1)

```python
# "요리"를 좋아하는 사람들의 연령대 분포
age_dist = {
    "10s": 50,
    "20s": 200,
    "30s": 180,    # ← 영희의 연령대
    "40s": 70,
    "50s": 23
}

total = 50 + 200 + 180 + 70 + 23 = 523

age_group_fit = 180 / 523 ≈ 0.34
# 30대 중 "요리"를 좋아하는 비율 34%
```

#### Feature 5: occupation_fit (직업 적합도, 0~1)

```python
# "요리"를 좋아하는 사람들의 직업 분포

occupation_fit = 0.12
# 디자이너 중 "요리" 비율 12%
```

#### Feature 6: region_fit (지역 적합도, 0~1)

```python
province_fit = 0.30  # 서울 거주자 중 "요리" 비율
district_fit = 0.08  # 강남구 거주자 중 "요리" 비율

region_fit = max(0.30, 0.08) = 0.30
```

#### Feature 7: popularity_prior (인기도 사전확률, 0~1)

```python
# "요리"의 전체 인기도를 0~1로 정규화

train_popularity["요리"] = 480
max_popularity = 523  # 가장 인기 있는 취미

popularity_prior = 480 / 523 ≈ 0.92
# "요리"는 거의 모든 취미 중 가장 인기 있음
```

#### Feature 8: mismatch_penalty (부정합 벌점, 0~0.33)

```python
# 세 가지 항목 검사:
# age_group_fit < 0.05 → 벌점
# occupation_fit < 0.05 → 벌점
# sex 분포 fit < 0.05 → 벌점

# "요리" 분석:
#   age_group_fit = 0.34 → 0.05 이상 → 벌점 없음
#   occupation_fit = 0.12 → 0.05 이상 → 벌점 없음
#   sex 분포에서 "여성" 비율 = 0.55 → 0.05 이상 → 벌점 없음

mismatch_penalty = (0 + 0 + 0) / 3 = 0.0

# counter-example: "등산"이 영희(30대 여성 디자이너)에게 추천될 때
#   age_group_fit = 0.40 → OK
#   occupation_fit = 0.02 → 0.05 미만! → 벌점 1 - 0.02 = 0.98
#   sex 분포 "여성" = 0.25 → OK
#   mismatch_penalty = (0 + 0.98 + 0) / 3 = 0.327
#   → 영희는 디자이너인데 "등산"을 좋아하는 디자이너가 거의 없음 → 벌점
```

#### Feature 9~13: 현재 0 (미사용 feature)

```python
lightgcn_score = 0.0           # GNN 미사용
popularity_penalty = 0.0       # 가중치 0
novelty_bonus = 0.0            # 가중치 0
category_diversity_reward = 0.0  # 가중치 0
is_cold_start = 0.0            # 영희는 취미 2개 보유
```

#### Feature 벡터 (LightGBM 실제 입력)

```python
# "요리"의 13개 Feature 벡터:
features = [
    0.96,   # 1. cooccurrence_score
    0.72,   # 2. segment_popularity_score
    0.18,   # 3. known_hobby_compatibility
    0.34,   # 4. age_group_fit
    0.12,   # 5. occupation_fit
    0.30,   # 6. region_fit
    0.92,   # 7. popularity_prior
    0.0,    # 8. mismatch_penalty (부정합 없음)
    0.0,    # 9. lightgcn_score
    0.0,    # 10. popularity_penalty
    0.0,    # 11. novelty_bonus
    0.0,    # 12. category_diversity_reward
    0.0,    # 13. is_cold_start
]
```

### 4.3 LightGBM 모델 구조

#### 모델이란?

LightGBM은 **결정 트리(Decision Tree)** 여러 개를 연결한 **앙상블 모델**:

```
Tree 1 (단순): 
  "popularity_prior > 0.5 ?"
    ├── Yes → 점수 +0.3
    └── No  → 점수 -0.1

Tree 2 (조금 더 복잡):
  "cooccurrence_score > 0.8 AND age_group_fit > 0.3 ?"
    ├── Yes → 점수 +0.2
    └── No  → 점수 -0.05

Tree 3:
  "known_hobby_compatibility > 0.15 ?"
    ├── Yes → 점수 +0.15
    └── No  → 점수 -0.08

...

최종 점수 = Tree1 결과 + Tree2 결과 + Tree3 결과 + ...
           = (+0.3) + (+0.2) + (+0.15) + ...
           = 0.85

→ sigmoid(0.85) ≈ 0.70  (70% 확률로 "좋아할 것이다")
```

#### 핵심 하이퍼파라미터

| 파라미터 | 값 | 의미 |
|---------|-----|------|
| `objective` | `binary` | 이진 분류 (좋아함 / 안 좋아함) |
| `metric` | `auc` | AUC 점수로 성능 측정 (0.5=랜덤, 1.0=완벽) |
| `num_leaves` | **31** | 각 트리의 최대 리프 개수 (클수록 복잡) |
| `min_data_in_leaf` | 50 | 리프가 가지는 최소 데이터 수 (작을수록 과적합 위험) |
| `learning_rate` | 0.05 | 한 번에 얼마나 배울지 (작을수록 천천히) |
| `reg_alpha` | 0.1 | L1 정규화 (불필요한 feature 페널티) |
| `reg_lambda` | 0.1 | L2 정규화 (큰 가중치 페널티) |
| `num_boost_round` | 500 | 최대 트리 개수 |
| `early_stopping_rounds` | 50 | 50번째 동안 성능 개선 없으면 중단 |

```
num_leaves의 의미:

num_leaves=3인 트리:
       ┌──────────┐
       │  condition  │
       ├────┬───────┤
       │ Y  │   N   │
    ┌──┴──┐    ┌──┴──┐
    │score│    │score│
    └─────┘    └─────┘
    3 leaves → 단순한 규칙

num_leaves=31인 트리 (default):
       더 복잡한 규칙 가능
       예: "인기도 > 0.5 AND 연령대=30대 AND 직업=디자이너 AND ..."
       31 leaves → 충분히 표현력 있음
```

#### 학습 과정 (Training)

```python
# step 1: 학습 데이터 준비
ranker_train_edges = 3,259개 (positive)
  → 각 positive마다 4개의 negative 생성 (MNS)
  → 총 약 16,295개 학습 Row (13 features + label)

ranker_val_edges = 815개 (positive)
  → 동일하게 negative 생성
  → 총 약 4,075개 검증 Row

# step 2: LightGBM 학습
Epoch  1: Train AUC=0.65, Val AUC=0.63  (시작)
Epoch  2: Train AUC=0.70, Val AUC=0.67
Epoch  3: Train AUC=0.74, Val AUC=0.70
...
Epoch 47: Train AUC=0.88, Val AUC=0.85  (best)
Epoch 48: Train AUC=0.89, Val AUC=0.849 (떨어짐)
...
Epoch 97: Train AUC=0.95, Val AUC=0.84  (50번째 개선 없음)
          → EARLY STOPPING! ← 47번째 트리에서 중단

# 47개의 결정 트리가 생성됨
# 각 트리는 13개 feature 중 일부를 사용하여 분기
```

### 4.4 학습 데이터 생성: Mixed Negative Sampling (MNS)

**문제**: 데이터에는 "좋아하는 취미"만 있고 "싫어하는 취미"는 없음.

**해결**: 인위적으로 "negative 예시"를 만들어서 positive와 함께 학습.

```python
# Positive (정답 = 1):
#   실제로 영희의 validation 취미 중 하나
#   → "영희-요리" → label=1 (영희는 요리를 좋아함)

# Negative (정답 = 0): 각 Positive마다 4개 생성

# Hard Negative (80% = 3.2개 → 보통 3~4개):
#   Stage 1의 후보 50개에는 들어있지만, 
#   실제로 영희의 취미는 아닌 것
#   → "영희-등산" (등산은 후보에는 있지만 영희 취미 아님)
#   → "영희-러닝" (후보에는 있지만 취미 아님)
#   → 모델이 헷갈릴 만한 "어려운" 예시

# Easy Negative (20% = 0.8개 → 보통 1개):
#   180개 취미 중 무작위로 뽑되, 영희 취미가 아닌 것
#   → "영희-우주여행" (후보에도 없고 절대 안 할 취미)
#   → 모델이 확실히 구분할 수 있는 "쉬운" 예시

# 최종 학습 Row:
#   (영희, 요리,     label=1, features=[...])  ← positive
#   (영희, 등산,     label=0, features=[...])  ← hard negative
#   (영희, 러닝,     label=0, features=[...])  ← hard negative
#   (영희, 독서,     label=0, features=[...])  ← hard negative
#   (영희, 요트,     label=0, features=[...])  ← easy negative
#   → 1:4 비율로 모델이 "왜 이건 좋아하고 저건 싫어하지?" 학습
```

**하이퍼파라미터**:
```
neg_ratio = 4    → positive 1개당 negative 4개
hard_ratio = 0.8 → negative의 80%는 "어려운" 것 (후보 pool 출신)
                   20%는 "쉬운" 것 (무작위)
```

### 4.5 추론 (Inference) — 실제 예측

```python
# 영희(30대, 여성, 디자이너, 서울)에게 최종 추천

# Step 1: Stage 1이 50개 후보 생성
stage1_candidates = [요리, 등산, 영화 감상, 필라테스, ...]  # 50개

# Step 2: 각 후보마다 13개 Feature 계산
for candidate_hobby in stage1_candidates:
    features = build_features(영희_context, candidate_hobby, hobby_profile, known_hobbies)
    
    # 예: "요리"의 features
    # [0.96, 0.72, 0.18, 0.34, 0.12, 0.30, 0.92, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    # 예: "필라테스"의 features
    # [0.45, 0.85, 0.05, 0.60, 0.35, 0.20, 0.30, 0.10, 0.0, 0.0, 0.0, 0.0, 0.0]

# Step 3: LightGBM이 각각의 확률 예측
    요리     → 0.70  (70% 확률로 좋아함)
    필라테스 → 0.88  (88% 확률로 좋아함)
    등산     → 0.35  (35% 확률로 좋아함)
    영화 감상 → 0.82  (82% 확률로 좋아함)
    ...

# Step 4: 확률 높은 순으로 10개 선별
final_top10 = [
    필라테스 (0.88),
    영화 감상 (0.82),
    요리     (0.70),
    ...
]
```

**LightGBM 예측 내부 동작**:

```
"필라테스"의 13개 feature → 각 트리 통과

Tree 1: "segment_popularity_score > 0.7?" → YES → +0.25
Tree 2: "age_group_fit > 0.5 AND is_cold_start == 0?" → YES → +0.18
Tree 3: "mismatch_penalty < 0.05 AND cooccurrence > 0.3?" → YES → +0.15
Tree 4: "occupation_fit > 0.2?" → YES → +0.10
Tree 5: "popularity_prior > 0.5?" → NO → -0.05
...

raw_score = 0.25 + 0.18 + 0.15 + 0.10 - 0.05 + ... = 1.95

probability = sigmoid(1.95) = 1 / (1 + e^(-1.95)) ≈ 0.88
```

---

## 5. 전체 파이프라인 요약

### 5.1 종합 모식도

```
                ┌──────────────────────────────────────┐
                │         데이터 전처리                  │
                │                                      │
                │   person_hobby_edges.csv (44K edges)  │
                │   person_context.csv (10K persons)    │
                │              │                        │
                │              ↓                        │
                │   • 취미 이름 NFKC 정규화              │
                │   • Alias 매핑 + Taxonomy 규칙         │
                │   • 희소 취미 제거 (3회 미만 DROP)     │
                │   • 문자열 → 숫자 ID 인덱싱            │
                │   • 80/10/10 Train/Val/Test 분할      │
                │   • Hobby Profile 생성 (통계)          │
                └────────────────┬─────────────────────┘
                                │
                                ↓
┌──────────────────────────────────────────────────────────────────┐
│                     STAGE 1: 후보 생성                            │
│                    (통계 기반, 학습 없음)                          │
│                                                                  │
│  ┌─────────────────────┐     ┌─────────────────────┐            │
│  │   인기도 제공자       │     │   동시출현 제공자     │            │
│  │ (Popularity)         │     │ (Co-occurrence)    │            │
│  │                      │     │                     │            │
│  │  train_edges → 카운트  │     │  known + cooccurrence│            │
│  │  → most_common()     │     │  → 합산 점수        │            │
│  │  → known 제외 → 50개  │     │  → known 제외 → 50개 │            │
│  └──────────┬───────────┘     └──────────┬──────────┘            │
│             │                            │                       │
│             └──────────┬─────────────────┘                       │
│                        │                                         │
│               rank_percentile 정규화 [0, 1]                      │
│                        │                                         │
│               merge_candidates_by_hobby()                       │
│               (중복 제거 + 최고 점수 채택)                        │
│                        │                                         │
│               최대 50개 HobbyCandidate                            │
│               (hobby_id, name, source_scores)                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ↓
┌──────────────────────────────────────────────────────────────────┐
│                  STAGE 2: 재순위화                                │
│              (LightGBM Learned Ranker, 학습됨)                    │
│                                                                  │
│  13개 Feature 계산 (각 후보마다):                                 │
│                                                                  │
│  ┌─────────────────────────────────────────────────────┐        │
│  │  1. cooccurrence_score    (동시출현 점수)            │        │
│  │  2. segment_popularity    (인구통계 적합도)          │        │
│  │  3. known_hobby_compat    (취미 일치율)             │        │
│  │  4. age_group_fit         (연령대 적합도)           │        │
│  │  5. occupation_fit        (직업 적합도)             │        │
│  │  6. region_fit            (지역 적합도)             │        │
│  │  7. popularity_prior      (인기도 사전확률)          │        │
│  │  8. mismatch_penalty      (부정합 벌점)             │        │
│  │  9~13. (현재 0)                                     │        │
│  └─────────────────────────────────────────────────────┘        │
│                         │                                         │
│                         ↓                                         │
│  LightGBM 모델 (num_leaves=31, binary classification)            │
│  각 후보의 13개 feature → 예측 확률 (0~1)                        │
│                         │                                         │
│                         ↓                                         │
│  예측 확률 순 내림차순 → 상위 10개 최종 추천                      │
└──────────────────────────────────────────────────────────────────┘
```

### 5.2 최종 성능

| 지표 | 의미 | Validation | Test |
|------|------|:----------:|:----:|
| **Recall@10** | 10개 추천 중 평균 몇 개가 정답인가? | **73.9%** | **71.0%** |
| **NDCG@10** | 순서까지 고려한 정확도 (100%가 완벽) | 45.8% | 44.8% |
| **Coverage@10** | 전체 180개 취미 중 몇 개를 추천에 사용? | 15.6% | 15.6% |

**Stage 1만 사용했을 때보다**: Recall@10 기준 **+1.9% 향상** (69.1% → 71.0%)

### 5.3 키워드 요약

| 단계 | 핵심 동사 | 입력 → 출력 | 방식 |
|------|----------|------------|------|
| 전처리 | **정리한다** | 더러운 CSV → 깨끗한 숫자 데이터 | 규칙 기반 |
| Stage 1 | **뽑는다** | 180개 취미 → 50개 후보 | 통계 (빈도/동시등장) |
| Stage 2 | **고른다** | 50개 후보 → 10개 최종 | 머신러닝 (LightGBM) |

```
Stage 1 = "일단 대충 걸러내자" (통계)
Stage 2 = "자, 이제 제대로 골라보자" (학습된 모델)
```