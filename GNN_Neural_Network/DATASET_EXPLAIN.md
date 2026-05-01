# Nemotron-Personas-Korea Dataset Explain

> **문서 역할**: 이 문서는 데이터셋 특성, 스키마, 그래프 매핑 맥락을 설명하는 보조 문서입니다. 추천 시스템의 기준 요구사항/실행 규칙은 `GNN_Neural_Network/PRD.md`와 `GNN_Neural_Network/TASKS.md`를 우선 적용합니다.
> 
> **문서 우선순위**: 충돌 시 **`GNN_Neural_Network/PRD.md` → `GNN_Neural_Network/TASKS.md` → `README`**를 우선으로 적용하고, 본 문서는 보조 근거자료로만 사용합니다.

## Dataset Shape

`nvidia/Nemotron-Personas-Korea`는 한국어 synthetic persona 데이터셋이다.

- Source: <https://huggingface.co/datasets/nvidia/Nemotron-Personas-Korea>
- Format: Parquet (HuggingFace Datasets library로 로드)
- Split: `train` (단일 split)
- Rows: **1,000,000** (100만 행)
- Unit: 1 row = 1 persona
- Local file: `data/raw/nemotron-personas-korea/train.parquet`
- License: CC-BY-4.0
- Tags: Synthetic, personas, NVIDIA, Korean, datadesigner

### 전체 컬럼 (26개)

아래는 HuggingFace 데이터셋 브라우저에서 확인한 26개 전 컬럼의 상세 스키마다.

| # | Group | Column | Type | Length/Cardinality | Use |
|---|-------|--------|------|-------------------|-----|
| 1 | ID | `uuid` | string(32) | 1M unique | persona primary key |
| 2-8 | Persona text (7) | `persona` | string | 42~174자 | persona 요약 |
| 3 | | `professional_persona` | string | 60~306자 | 직업 맥락 persona |
| 4 | | `sports_persona` | string | 51~293자 | 운동/스포츠 맥락 |
| 5 | | `arts_persona` | string | 54~260자 | 문화/예술 맥락 |
| 6 | | `travel_persona` | string | 51~248자 | 여행 맥락 |
| 7 | | `culinary_persona` | string | 59~277자 | 음식/요리 맥락 |
| 8 | | `family_persona` | string | 58~252자 | 가족 관계 맥락 |
| 9-12 | Context text (4) | `cultural_background` | string | 55~299자 | 성향/배경 |
| 10 | | `skills_and_expertise` | string | 50~247자 | 기술 설명 문단 |
| 11 | | `hobbies_and_interests` | string | 6~305자 | 취미 설명 문단 |
| 12 | | `career_goals_and_ambitions` | string | 47~243자 | 직업 목표 |
| 13-14 | List text (2) | `skills_and_expertise_list` | string | 35~201자 | graph edge source (Skill) |
| 14 | | `hobbies_and_interests_list` | string | 32~184자 | graph edge source (Hobby) |
| 15-22 | Demographics (8) | `sex` | string(2) | 2 classes (남자/여자) | reranking feature |
| 16 | | `age` | int64 | 19~99 | reranking feature |
| 17 | | `marital_status` | string(4) | 4 classes | reranking feature |
| 18 | | `military_status` | string(2) | 2 classes | reranking feature |
| 19 | | `family_type` | string(39) | 39 classes | reranking feature |
| 20 | | `housing_type` | string(6) | 6 classes | reranking feature |
| 21 | | `education_level` | string(7) | 7 classes | reranking feature |
| 22 | | `bachelors_field` | string(11) | 11 classes (해당없음 포함) | reranking feature |
| 23-26 | Work/location (4) | `occupation` | string | 2~40자, 500+ classes | graph node + feature |
| 24 | | `district` | string(252) | 252 classes | graph node |
| 25 | | `province` | string(17) | 17 classes | graph node |
| 26 | | `country` | string(1) | 1 class (`대한민국` 고정) | graph node |

`src.data.preprocessor.preprocess()`는 list-like string을 list로 파싱하고, `district`를 `province_cleaned`/`district_cleaned`로 나누며, `age_group`과 `embedding_text`를 만든다.

### Raw Data vs GNN System: 규모 비교

| 항목 | Raw Dataset (100만 행) | GNN System (retained) | 비고 |
|------|----------------------|----------------------|------|
| Persons | 1,000,000 | 10,000 | GNN PoC는 10K 서브셋 사용 |
| Edges | ~5,000,000 (est.) | 40,743 | person당 평균 5개 hobby 가정 |
| Raw hobbies | ~50,000+ (est.) | 27,137 | canonical 전 raw unique hobby |
| Canonical hobbies | — | 2,302 | taxonomy 기반 병합 후 |
| Retained hobbies | — | 180 | min_item_degree >= 3 필터 후 |
| Persona text fields | 13개 텍스트 필드 | 9개 텍스트 필드 (text_fit 비활성) | leakage 29%로 text feature disabled |

### 컬럼별 카디널리티 및 특이사항

| 컬럼 | Unique 값 수 | 특이사항 |
|------|-------------|---------|
| `country` | 1 | 항상 `대한민국`. Graph node로서 정보량 0 |
| `province` | 17 | 광역시/도 단위. 적절한 granularity |
| `district` | 252 | 시/군/구 단위. `서울-서초구`, `경기-성남시 분당구` 형태 |
| `sex` | 2 | 남자/여자 |
| `age` | 81 (19~99) | 모든 연령대 고르게 분포 |
| `marital_status` | 4 | e.g. 배우자있음, 미혼, 사별, 이혼 |
| `military_status` | 2 | 비현역/현역. 여성은 전원 비현역 |
| `family_type` | 39 | 혼자 거주, 배우자와 거주, 자녀와 거주 등 |
| `housing_type` | 6 | 아파트, 다세대주택, 단독주택 등 |
| `education_level` | 7 | 초등학교 ~ 4년제 대학교, 대학원 |
| `bachelors_field` | 11 | 자연과학, 공학, 인문학 등 + 해당없음 |
| `occupation` | 500+ | 매우 세분화된 직업명 |
| `hobbies_and_interests_list` | 매우 큼 | raw 기준 27K+ unique 취미명 |

### Persona Text 구조: 7개 도메인 × 3~5문단 스토리텔링

각 persona 도메인 텍스트는 LLM이 생성한 일관된 내러티브 구조를 가진다:

```
persona:              [이름]은/는 [지역/직업]에서 [성격/특징]을 가진 [나이]대 [성별]입니다.
professional_persona:  직장에서의 능력, 업무 스타일, 동료와의 관계를 3~5문단으로 서술
sports_persona:        운동 선호도, 활동량, 구체적인 운동/산책 패턴
arts_persona:          문화 생활, 예술 취향, 미디어 소비 패턴
travel_persona:        여행 스타일, 선호하는 여행지, 동행자
culinary_persona:      식사 패턴, 외식 빈도, 선호 음식
family_persona:        가족 관계, 주거 형태, 가족 내 역할
cultural_background:   성장 배경, 가치관, 지역 문화 특성
hobbies_and_interests: 종합 취미/관심사 요약 (hobbies_and_interests_list의 문단형 버전)
```

각 필드는 일관된 화자 시점과 톤을 유지하며, 동일 인물에 대해 7개 도메인이 서로 모순 없이 연결된다. 예를 들어 `sports_persona`에서 "격렬한 운동보다 가벼운 산책"을 선호한다면, `culinary_persona`에서도 "자극적이지 않은 담백한 한식"을 선호하는 식으로 일관성을 유지한다.

### Persona Text Leakage 현황

`embedding_text`는 `persona` + 7개 domain persona + `hobbies_and_interests` + `skills_and_expertise`를 모두 concatenation한 필드다. 이 필드가 held-out hobby name을 직접 포함하는 비율(leakage rate)은 **29.3%** (validation 2,879/9,841 edges)로 높은 수준이다.

| 텍스트 필드 | Leak 건수 | 주요 누수 취미 |
|------------|----------|---------------|
| `embedding_text` | 2,873 | 산책, 배드민턴, 게임, 노래방, 사진 촬영, 낚시 등 |
| `sports_text` | 1,290 | 산책, 배드민턴, 유산소 운동 등 스포츠 계열 |
| `hobbies_text` | 1,093 | 산책, 게임, 배드민턴, 사진 촬영 등 |
| `persona_text` | 292 | 게임, 봉사활동, 낚시 등 |
| `arts_text` | 293 | 노래방, 게임, 사진 촬영 |
| `travel_text` | 102 | 산책, 역사 유적지 여행 |
| `family_text` | 72 | (가족 관련 취미) |
| `professional_text` | 22 | (직업 관련 취미) |
| `culinary_text` | 8 | (요리 관련 취미) |

→ 누수 방지를 위해 기본 평가에서는 모든 text 기반 feature를 비활성화한다. `mask_holdout_hobbies()` + `post_mask_leakage_audit()` 구현은 완료되었으나(text_embedding.py), 아직 평가 파이프라인에 통합되지 않았다.

### Persona Text의 Synthetic 데이터 특성

이 데이터셋은 LLM(NVIDIA NeMo 기반)이 생성한 synthetic persona로, 실제 사용자 데이터가 아니다. 이로 인한 주요 특성:

- **연령/성별/직업별 취미 패턴이 전형적(stereotype)이다.** 70대 남성은 사우나·트로트·산책, 30대 여성은 맛집·사진·넷플릭스 등 LLM이 학습한 "전형적인 한국인" 상(image)이 반영되었다. 이는 실제보다 취미 조합의 다양성이 낮고 예측 가능성이 높은 원인이 된다.
- **취미 이름이 매우 구체적인 자연어 문장 수준이다.** `"송도 센트럴파크 산책로 걷기"`, `"무등산 둘레길 산책"`처럼 장소+활동이 결합된 형태. 이는 canonicalization이 필수적인 이유다.
- **Persona 텍스트가 매우 풍부하다.** 7개 도메인 × 3~5문단의 일관된 스토리텔링으로, 실제 서비스에서 수집하기 어려운 수준의 persona 정보를 제공한다. 따라서 이 데이터셋의 추천 과제는 "부족한 정보에서 추론"이 아니라 "풍부한 정보 속에서 진짜 signal 찾기"에 가깝다.
- **100만 행 모두 동일한 LLM 파이프라인으로 생성되었으므로, 데이터의 노이즈 패턴이 인위적이다.** 실제 사용자 데이터에서 발생하는 결측, 불균형, 비일관성 등이 없어 오프라인 실험은 현실보다 metric이 낙관적으로 나올 가능성이 있다.

### Raw Data vs GNN System: 규모 비교

| 항목 | Raw Dataset (100만 행) | GNN System (retained) | 비고 |
|------|----------------------|----------------------|------|
| Persons | 1,000,000 | 10,000 | GNN PoC는 10K 서브셋 사용 |
| Edges | ~5,000,000 (est.) | 40,743 | person당 평균 5개 hobby 가정 |
| Raw hobbies | ~50,000+ (est.) | 27,137 | canonical 전 raw unique hobby |
| Canonical hobbies | — | 2,302 | taxonomy 기반 병합 후 |
| Retained hobbies | — | 180 | min_item_degree >= 3 필터 후 |
| Persona text fields | 13개 텍스트 필드 | 9개 텍스트 필드 (text_fit 비활성) | leakage 29%로 text feature disabled |

### 컬럼별 카디널리티 및 특이사항

| 컬럼 | Unique 값 수 | 특이사항 |
|------|-------------|---------|
| `country` | 1 | 항상 `대한민국`. Graph node로서 정보량 0 |
| `province` | 17 | 광역시/도 단위. 적절한 granularity |
| `district` | 252 | 시/군/구 단위. `서울-서초구`, `경기-성남시 분당구` 형태 |
| `sex` | 2 | 남자/여자 |
| `age` | 81 (19~99) | 모든 연령대 고르게 분포 |
| `marital_status` | 4 | e.g. 배우자있음, 미혼, 사별, 이혼 |
| `military_status` | 2 | 비현역/현역. 여성은 전원 비현역 |
| `family_type` | 39 | 혼자 거주, 배우자와 거주, 자녀와 거주 등 |
| `housing_type` | 6 | 아파트, 다세대주택, 단독주택 등 |
| `education_level` | 7 | 초등학교 ~ 4년제 대학교, 대학원 |
| `bachelors_field` | 11 | 자연과학, 공학, 인문학 등 + 해당없음 |
| `occupation` | 500+ | 매우 세분화된 직업명 |
| `hobbies_and_interests_list` | 매우 큼 | raw 기준 27K+ unique 취미명 |

### Persona Text 구조: 7개 도메인 × 3~5문단 스토리텔링

각 persona 도메인 텍스트는 LLM이 생성한 일관된 내러티브 구조를 가진다:

```
persona:              [이름]은/는 [지역/직업]에서 [성격/특징]을 가진 [나이]대 [성별]입니다.
professional_persona:  직장에서의 능력, 업무 스타일, 동료와의 관계를 3~5문단으로 서술
sports_persona:        운동 선호도, 활동량, 구체적인 운동/산책 패턴
arts_persona:          문화 생활, 예술 취향, 미디어 소비 패턴
travel_persona:        여행 스타일, 선호하는 여행지, 동행자
culinary_persona:      식사 패턴, 외식 빈도, 선호 음식
family_persona:        가족 관계, 주거 형태, 가족 내 역할
cultural_background:   성장 배경, 가치관, 지역 문화 특성
hobbies_and_interests: 종합 취미/관심사 요약 (hobbies_and_interests_list의 문단형 버전)
```

각 필드는 일관된 화자 시점과 톤을 유지하며, 동일 인물에 대해 7개 도메인이 서로 모순 없이 연결된다. 예를 들어 `sports_persona`에서 "격렬한 운동보다 가벼운 산책"을 선호한다면, `culinary_persona`에서도 "자극적이지 않은 담백한 한식"을 선호하는 식으로 일관성을 유지한다.

### Persona Text Leakage 현황

`embedding_text`는 `persona` + 7개 domain persona + `hobbies_and_interests` + `skills_and_expertise`를 모두 concatenation한 필드다. 이 필드가 held-out hobby name을 직접 포함하는 비율(leakage rate)은 **29.3%** (validation 2,879/9,841 edges)로 높은 수준이다.

| 텍스트 필드 | Leak 건수 | 주요 누수 취미 |
|------------|----------|---------------|
| `embedding_text` | 2,873 | 산책, 배드민턴, 게임, 노래방, 사진 촬영, 낚시 등 |
| `sports_text` | 1,290 | 산책, 배드민턴, 유산소 운동 등 스포츠 계열 |
| `hobbies_text` | 1,093 | 산책, 게임, 배드민턴, 사진 촬영 등 |
| `persona_text` | 292 | 게임, 봉사활동, 낚시 등 |
| `arts_text` | 293 | 노래방, 게임, 사진 촬영 |
| `travel_text` | 102 | 산책, 역사 유적지 여행 |
| `family_text` | 72 | (가족 관련 취미) |
| `professional_text` | 22 | (직업 관련 취미) |
| `culinary_text` | 8 | (요리 관련 취미) |

→ 누수 방지를 위해 기본 평가에서는 모든 text 기반 feature를 비활성화한다. `mask_holdout_hobbies()` + `post_mask_leakage_audit()` 구현은 완료되었으나(text_embedding.py), 아직 평가 파이프라인에 통합되지 않았다.

### Persona Text의 Synthetic 데이터 특성

이 데이터셋은 LLM(NVIDIA NeMo 기반)이 생성한 synthetic persona로, 실제 사용자 데이터가 아니다. 이로 인한 주요 특성:

- **연령/성별/직업별 취미 패턴이 전형적(stereotype)이다.** 70대 남성은 사우나·트로트·산책, 30대 여성은 맛집·사진·넷플릭스 등 LLM이 학습한 "전형적인 한국인" 상(image)이 반영되었다. 이는 실제보다 취미 조합의 다양성이 낮고 예측 가능성이 높은 원인이 된다.
- **취미 이름이 매우 구체적인 자연어 문장 수준이다.** `"송도 센트럴파크 산책로 걷기"`, `"무등산 둘레길 산책"`처럼 장소+활동이 결합된 형태. 이는 canonicalization이 필수적인 이유다.
- **Persona 텍스트가 매우 풍부하다.** 7개 도메인 × 3~5문단의 일관된 스토리텔링으로, 실제 서비스에서 수집하기 어려운 수준의 persona 정보를 제공한다. 따라서 이 데이터셋의 추천 과제는 "부족한 정보에서 추론"이 아니라 "풍부한 정보 속에서 진짜 signal 찾기"에 가깝다.
- **100만 행 모두 동일한 LLM 파이프라인으로 생성되었으므로, 데이터의 노이즈 패턴이 인위적이다.** 실제 사용자 데이터에서 발생하는 결측, 불균형, 비일관성 등이 없어 오프라인 실험은 현실보다 metric이 낙관적으로 나올 가능성이 있다.

## Current Graph Mapping

현재 Neo4j 그래프는 `Person` 중심 heterogeneous graph로 구성된다.

### Person Node

`Person` 노드는 다음 속성을 가진다.

- `uuid`
- `display_name`
- `age`, `age_group`, `sex`
- `persona`
- persona text fields
- `embedding_text`

### Entity Nodes

| Dataset field | Graph label | Cardinality | 비고 |
| --- | --- | --- | --- |
| `country` | `Country` | 1 | 값이 `대한민국` 하나뿐. 그래프 노드로서 정보량 거의 0. 제거 고려 |
| `province` | `Province` | 17 | 적절 |
| `district` / cleaned district | `District` | 252 | 적절 |
| `occupation` | `Occupation` | 500+ | 적절 |
| `skills_and_expertise_list` | `Skill` | 다수 | 적절 |
| `hobbies_and_interests_list` | `Hobby` | 다수 | ✅ **핵심 노드** |
| `education_level` | `EducationLevel` | 7 | 적절 |
| `bachelors_field` | `Field` | 11 | `해당없음` 비중 높을 가능성 |
| `marital_status` | `MaritalStatus` | 4 | 적절 |
| `military_status` | `MilitaryStatus` | 2 | `비현역`이 압도적. feature로 쓸만한 분산이 없을 가능성 있음 |
| `family_type` | `FamilyType` | 39 | 적절 |
| `housing_type` | `HousingType` | 6 | 적절 |

### Relationships

| Relationship | Meaning |
| --- | --- |
| `(Person)-[:LIVES_IN]->(District)` | residence |
| `(District)-[:IN_PROVINCE]->(Province)` | location hierarchy |
| `(Province)-[:IN_COUNTRY]->(Country)` | location hierarchy |
| `(Person)-[:WORKS_AS]->(Occupation)` | job |
| `(Person)-[:HAS_SKILL]->(Skill)` | skill/expertise |
| `(Person)-[:ENJOYS_HOBBY]->(Hobby)` | hobby |
| `(Person)-[:EDUCATED_AT]->(EducationLevel)` | education |
| `(Person)-[:MAJORED_IN]->(Field)` | major field |
| `(Person)-[:MARITAL_STATUS]->(MaritalStatus)` | marital status |
| `(Person)-[:MILITARY_STATUS]->(MilitaryStatus)` | military status |
| `(Person)-[:LIVES_WITH]->(FamilyType)` | family/living arrangement |
| `(Person)-[:LIVES_IN_HOUSING]->(HousingType)` | housing type |

## Recommendation Use

### Stage 1: Candidate Generation

Stage 1 should generate broad hobby candidates. It should prioritize recall over final precision.

Possible providers:

- LightGCN over `Person-Hobby` edges
- co-occurrence baseline
- popularity fallback
- similar-person hobby lookup

LightGCN input is only:

```text
person_uuid,hobby_name
```

After preprocessing:

```text
person_id,hobby_id
```

LightGCN learns from shared hobby graph structure. It does not directly know age, job, personality, lifestyle, or persona text. Therefore it is suitable as a candidate generator, not as the final recommender.

### Stage 2: Persona-aware Reranking

Stage 2 should decide whether a candidate hobby actually fits the target persona.

Useful inputs:

- `embedding_text`
- `persona`
- `professional_persona`
- `sports_persona`
- `arts_persona`
- `travel_persona`
- `culinary_persona`
- `family_persona`
- `hobbies_and_interests`
- `skills_and_expertise`
- `career_goals_and_ambitions`
- `age`, `age_group`, `sex`
- `occupation`
- `district`, `province`
- `family_type`, `housing_type`
- known hobbies

Example final scoring:

```text
final_score =
  graph_candidate_score
+ persona_text_fit
+ known_hobby_compatibility
+ demographic_fit
+ lifestyle_fit
+ region_accessibility_fit
- mismatch_penalty
```

This is necessary because two people may share one hobby but have very different life contexts. For example, a 50s office worker and a 20s woman may both like golf, but their next best hobby recommendations can be very different.

## Data Quality Notes

### Raw Hobby의 형태적 문제

Raw hobby names are highly specific natural-language phrases. Similar hobbies may appear as separate strings.

Examples of raw hobby variance (same canonical `배드민턴`):

- `지역 배드민턴 동호회 활동`
- `지역 배드민턴 클럽 활동`
- `동네 배드민턴 클럽 활동`
- `배드민턴 동호회 활동`

Other real examples from the dataset:

| Raw hobby string | Canonical | 문제 |
|---|---|---|
| `무등산 둘레길 산책` | 산책 | 장소 modifier만 다름 |
| `올림픽공원 숲길 산책` | 산책 | 장소 modifier만 다름 |
| `탄천 산책로 걷기` | 산책 | 동일 활동 |
| `송도 센트럴파크 산책` | 산책 | 장소 modifier만 다름 |
| `경복궁과 창덕궁 고궁 산책` | 산책 | 장소 modifier만 다름 |

### Vocabulary 통계 (현재 GNN System)

| 단계 | Unique Hobbies | Edges | Singleton Ratio |
|------|---------------|-------|-----------------|
| Raw (정규화 전) | 27,137 | 44,083 | 0.834 |
| Canonical (taxonomy 적용 후) | 2,302 | 43,034 | 0.848 |
| Retained (min_item_degree >= 3) | **180** | **40,743** | **0.000** |

Canonicalization으로 hobby 수가 27K → 2.3K로 91.5% 감소했지만, retained 기준 180개는 100만 personas 중 10,000명 retain 기준이다. **180개 hobby 중 degree 분포는 매우 skewed되어 있다** — 상위 취미(산책, 사교/친목 모임, 사우나/목욕 등)가 전체 edge의 대부분을 차지한다.

### Synthetic 데이터의 근본적 한계

이 데이터셋은 LLM이 생성한 synthetic 데이터로, 다음과 같은 한계가 있다:

1. **취미가 연령/성별/직업별로 stereotype에 가깝다.** LLM이 "전형적인 70대 남성"에 대해 생성한 persona는 실제 다양한 70대 남성보다 훨씬 예측 가능하다. 이는 추천 모델이 쉽게 높은 accuracy를 달성하게 하지만, 동시에 diversity를 해치는 ranking collapse의 근본 원인이 된다.
2. **그래프가 본질적으로 sparse하다.** Person당 평균 취미 수가 적고(4~5개), 서로 다른 인구통계 그룹 간 취미 중복이 적어 collaborative signal이 약하다. 이것이 LightGCN이 popularity + cooccurrence를 Stage 1에서 이기지 못한 원인이다.
3. **Noise 패턴이 인위적이다.** 실제 사용자 데이터의 결측, 불균형, 비일관성, measurement error가 없어 오프라인 metric이 현실보다 낙관적일 수 있다.

### Required Gates for Graph Recommendation

For graph recommendation, raw hobby strings should not be used blindly:

- Unicode normalization
- whitespace collapse
- optional alias/canonical taxonomy map
- dedupe after aliasing
- `min_item_degree` filtering (현재 3)
- vocabulary report with raw/canonical/retained counts
- fallback for dropped long-tail items/persons

## Current System Architecture (as of 2026-05-01)

```text
Dataset row (100만 rows, 26 columns)
  -> preprocessing + canonicalization
  -> 10K subsample, 180 canonical hobbies, 40K edges

Stage 1 candidate generation (selected baseline: popularity + cooccurrence)
  -> popularity provider (global train-split popularity)
  -> cooccurrence provider (train-split co-occurrence)
  -> [disabled] LightGCN (auxiliary, did not improve over baseline)
  -> [disabled] segment_popularity (toxic in ablation)
  -> [disabled] BM25/PMI/IDF/Jaccard/pop-capped (below baseline)

Stage 2 persona-aware reranking
  -> [promoted default] LightGBM learned ranker (AUC=0.8890555966387075)
  -> [fallback] v1 deterministic reranker (weighted scoring)
  -> [disabled] persona_text_fit (leakage 29%, no-text mode)

Final Top-K recommendation
  -> canonical_hobby + score + display_examples
  -> [optional] --use-learned-ranker (default: on)
  -> [optional] --use-mmr (default: off, NO-GO: one-hot embedding 한계)
  -> [optional] --explain (SHAP-based reason generation)
  -> [optional] --use-text-embedding (default: off, leakage audit 미통과)
```

### Current Status Summary

| Component | Status | Key Metric |
|-----------|--------|-----------|
| Stage 1 baseline | `popularity + cooccurrence` | recall@10=0.694, candidate_recall@50=0.978 |
| Stage 2 default | LightGBM learned ranker (promoted) | recall@10=0.7391 (val), 0.7097 (test) |
| Stage 2 fallback | v1 deterministic reranker | recall@10=0.710 (val), 0.704 (test) |
| MMR | optional flag only (default off) | NO-GO (category one-hot 한계) |
| LightGCN | auxiliary/analysis only | recall@10=0.677 (+ cooc/pop 0.691) |

**Known issue**: LightGBM ranking collapse — coverage@10 0.1556 (v1: 0.517), novelty@10 4.5843 (v1: 4.732). Feature importance is concentrated on cooccurrence_score + popularity_prior (60.73% + 28.14%).

In short:

- Stage 1은 `popularity + cooccurrence`가 최적 조합으로 확정됐다. LightGCN 및 6개 item-item provider 모두 ablation에서 baseline 미만이었다.
- Stage 2는 LightGBM learned ranker가 promoted default다. accuracy는 개선했지만 coverage/novelty가 감소한 ranking collapse 상태로, regularization tuning과 negative sampling ablation이 필요하다.
- Persona text는 leakage(29%) 문제로 기본 비활성이며, KURE dense embedding 기반 MMR 재평가와 text embedding ablation은 Phase 2.5 이후로 계획되어 있다.
