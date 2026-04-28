# PRD Phase 2: 확장 기능 명세서

> Phase 1 (GraphRAG 인사이트, 유사 매칭, 커뮤니티 탐지, 관계 경로)이 완료된 상태에서,
> 프론트엔드 활용도를 극대화하기 위한 5개 신규 기능을 정의한다.

---

## 1. Phase 2 개요

### 1.1 목표
Phase 1은 "데이터를 그래프로 적재하고 질의하는 백엔드 파이프라인"에 집중했다.
Phase 2는 **"100만 명의 페르소나 데이터를 프론트엔드에서 탐색·비교·시각화할 수 있는 사용자 경험"**을 구축한다.

### 1.2 신규 기능 목록

| # | 기능명 | 엔드포인트 | 핵심 가치 |
|---|--------|-----------|----------|
| F5 | 페르소나 검색/필터 엔진 | `GET /api/search` | 100만 명 중 원하는 사람을 조건으로 찾는 진입점 |
| F6 | 인구통계 대시보드 | `GET /api/stats`, `GET /api/stats/{dimension}` | 데이터 전체를 조망하는 집계 통계 |
| F7 | 페르소나 프로필 상세 뷰 | `GET /api/persona/{uuid}` | 한 사람의 모든 정보를 한 번에 반환 |
| F8 | 크로스 세그먼트 비교 분석 | `POST /api/compare/segments` | 두 그룹 간 취미·직업·학력 분포 비교 |
| F9 | 지식 그래프 서브그래프 시각화 | `GET /api/graph/subgraph/{uuid}` | 특정 페르소나 중심 관계 네트워크 반환 |

### 1.3 의존성

모든 Phase 2 기능은 Phase 1의 Neo4j 그래프 적재가 완료된 상태를 전제한다.
F5~F7은 Neo4j Cypher만으로 구현 가능하며, F8은 Cypher + LLM, F9는 Cypher만 사용한다.

---

## 2. 기능 명세서

### 2.1 기능 F5: 페르소나 검색/필터 엔진

**엔드포인트**: `GET /api/search`

**설계 원칙**:
- 모든 필터는 선택적(optional). 필터 없이 호출하면 전체 페르소나를 페이지네이션으로 반환한다.
- 필터 조건은 AND 결합. 복수 값 필터(예: province 여러 개)는 내부 OR 결합.
- 텍스트 키워드 검색은 Neo4j 전문 검색 또는 CONTAINS 매칭을 사용한다.
- 결과는 항상 페이지네이션되며, 총 건수(`total_count`)를 함께 반환한다.

**쿼리 파라미터**:

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|-------|------|
| `province` | string (comma-separated) | — | 시도 필터. 복수 가능: `서울,경기` |
| `district` | string (comma-separated) | — | 시군구 필터. 복수 가능: `서울-서초구,서울-강남구` |
| `age_min` | int | — | 최소 나이 (포함) |
| `age_max` | int | — | 최대 나이 (포함) |
| `age_group` | string (comma-separated) | — | 연령대 필터: `20대,30대` |
| `sex` | string | — | `남자` 또는 `여자` |
| `occupation` | string | — | 직업 키워드 (CONTAINS 매칭) |
| `education_level` | string (comma-separated) | — | 학력: `4년제 대학교,대학원` |
| `hobby` | string (comma-separated) | — | 취미 필터: `등산,독서` |
| `skill` | string (comma-separated) | — | 스킬 필터: `Python,용접` |
| `keyword` | string | — | 자유 텍스트 검색 (persona 필드 CONTAINS) |
| `sort_by` | string | `age` | 정렬 기준: `age`, `display_name` |
| `sort_order` | string | `asc` | `asc` 또는 `desc` |
| `page` | int | 1 | 페이지 번호 (1-based) |
| `page_size` | int | 20 | 페이지 크기 (최대 100) |

**응답**:
```json
{
    "total_count": 3245,
    "page": 1,
    "page_size": 20,
    "total_pages": 163,
    "results": [
        {
            "uuid": "03b4f36a18e6469386d0286dddd513c8",
            "display_name": "전기태",
            "age": 74,
            "sex": "남자",
            "province": "광주",
            "district": "서초구",
            "occupation": "하역 및 적재 관련 단순 종사원",
            "education_level": "초등학교",
            "persona": "광주 서구에서 평생 하역 일을 하며 살아온 70대 가장으로..."
        }
    ]
}
```

**Cypher 쿼리 전략**:
```cypher
MATCH (p:Person)
// 지역 필터
OPTIONAL MATCH (p)-[:LIVES_IN]->(d:District)-[:IN_PROVINCE]->(prov:Province)
// 취미 필터
OPTIONAL MATCH (p)-[:ENJOYS_HOBBY]->(h:Hobby)
// 스킬 필터
OPTIONAL MATCH (p)-[:HAS_SKILL]->(s:Skill)
// 직업 필터
OPTIONAL MATCH (p)-[:WORKS_AS]->(occ:Occupation)
// 학력 필터
OPTIONAL MATCH (p)-[:EDUCATED_AT]->(edu:EducationLevel)

WHERE
  (prov.name IN $provinces OR $provinces IS NULL)
  AND (p.age >= $age_min OR $age_min IS NULL)
  AND (p.age <= $age_max OR $age_max IS NULL)
  AND (p.sex = $sex OR $sex IS NULL)
  AND (h.name IN $hobbies OR $hobbies IS NULL)
  // ... 추가 조건

RETURN p, d.name AS district, prov.name AS province, occ.name AS occupation
ORDER BY p.age ASC
SKIP $skip LIMIT $limit
```

**구현 모듈**:
- `src/api/routes/search.py` — 엔드포인트
- `src/graph/search_queries.py` — 동적 Cypher 빌더
- `src/api/schemas.py` — `SearchResponse`, `SearchResult` Pydantic 모델 추가

**테스트 항목**:
- 필터 없이 호출 → 전체 결과 페이지네이션
- 단일 필터 (province=서울) → 서울 거주자만 반환
- 복합 필터 (province=서울 + age_group=30대 + sex=여자) → 교차 조건 충족자만 반환
- hobby 필터 (hobby=등산) → 등산 취미 보유자만 반환
- keyword 검색 → persona 필드 내 텍스트 매칭
- page_size 초과 방지 (101 → 400 에러)
- 결과 없음 → `total_count: 0, results: []`

---

### 2.2 기능 F6: 인구통계 대시보드

**엔드포인트 A**: `GET /api/stats`

전체 데이터의 요약 통계를 한 번에 반환한다. 프론트엔드 대시보드 홈 화면에 사용한다.

**응답**:
```json
{
    "total_personas": 1000000,
    "age_distribution": [
        {"age_group": "20대", "count": 142000, "ratio": 0.142},
        {"age_group": "30대", "count": 168000, "ratio": 0.168}
    ],
    "sex_distribution": [
        {"sex": "남자", "count": 502000, "ratio": 0.502},
        {"sex": "여자", "count": 498000, "ratio": 0.498}
    ],
    "province_distribution": [
        {"province": "경기", "count": 280000, "ratio": 0.280},
        {"province": "서울", "count": 195000, "ratio": 0.195}
    ],
    "top_occupations": [
        {"occupation": "사무 행정 사무원", "count": 12500},
        {"occupation": "회계 사무원", "count": 8900}
    ],
    "top_hobbies": [
        {"hobby": "등산", "count": 42300},
        {"hobby": "독서", "count": 38100}
    ],
    "top_skills": [
        {"skill": "문서 작성", "count": 35200},
        {"skill": "데이터 분석", "count": 28400}
    ],
    "education_distribution": [
        {"education_level": "4년제 대학교", "count": 320000, "ratio": 0.320},
        {"education_level": "고등학교", "count": 280000, "ratio": 0.280}
    ],
    "marital_distribution": [
        {"marital_status": "배우자있음", "count": 580000, "ratio": 0.580}
    ]
}
```

**엔드포인트 B**: `GET /api/stats/{dimension}`

특정 차원에 대한 상세 드릴다운 통계를 반환한다.

**경로 파라미터**: `dimension` = `age` | `sex` | `province` | `district` | `occupation` | `hobby` | `skill` | `education` | `marital` | `military` | `family_type` | `housing`

**쿼리 파라미터** (드릴다운 필터):

| 파라미터 | 설명 |
|---------|------|
| `province` | 해당 시도 내로 범위 한정 |
| `age_group` | 해당 연령대 내로 범위 한정 |
| `sex` | 해당 성별 내로 범위 한정 |
| `limit` | 반환 항목 수 (기본 20, 최대 100) |

**예시 요청**: `GET /api/stats/hobby?province=서울&age_group=30대&sex=여자&limit=10`

**응답**:
```json
{
    "dimension": "hobby",
    "filters_applied": {
        "province": "서울",
        "age_group": "30대",
        "sex": "여자"
    },
    "filtered_count": 32450,
    "distribution": [
        {"hobby": "카페투어", "count": 9080, "ratio": 0.280},
        {"hobby": "요가", "count": 7139, "ratio": 0.220},
        {"hobby": "독서", "count": 5516, "ratio": 0.170},
        {"hobby": "등산", "count": 4543, "ratio": 0.140},
        {"hobby": "영화감상", "count": 3894, "ratio": 0.120}
    ]
}
```

**Cypher 쿼리 전략**:
```cypher
// 예시: 취미 분포 드릴다운
MATCH (p:Person)-[:ENJOYS_HOBBY]->(h:Hobby)
OPTIONAL MATCH (p)-[:LIVES_IN]->(:District)-[:IN_PROVINCE]->(prov:Province)
WHERE prov.name = $province
  AND p.age_group = $age_group
  AND p.sex = $sex
RETURN h.name AS hobby, count(p) AS count
ORDER BY count DESC
LIMIT $limit
```

**구현 모듈**:
- `src/api/routes/stats.py` — 엔드포인트 2개
- `src/graph/stats_queries.py` — 집계 Cypher 쿼리 모음
- `src/api/schemas.py` — `StatsResponse`, `DimensionStatsResponse` 모델 추가

**테스트 항목**:
- 전체 통계 응답 구조 검증 (8개 분포 항목 존재)
- ratio 합산 = 1.0 (반올림 허용)
- 드릴다운 필터 적용 시 filtered_count < total_personas
- 존재하지 않는 dimension → 400 에러
- limit 제한 동작 확인

---

### 2.3 기능 F7: 페르소나 프로필 상세 뷰

**엔드포인트**: `GET /api/persona/{uuid}`

한 사람의 **모든 정보**를 단일 API 호출로 반환한다.
프론트엔드에서 프로필 페이지, 검색 결과 클릭, 유사 매칭 결과 클릭 등의 랜딩으로 사용한다.

**응답**:
```json
{
    "uuid": "03b4f36a18e6469386d0286dddd513c8",
    "display_name": "전기태",

    "demographics": {
        "age": 74,
        "age_group": "70대",
        "sex": "남자",
        "marital_status": "배우자있음",
        "military_status": "비현역",
        "family_type": "배우자와 거주",
        "housing_type": "아파트",
        "education_level": "초등학교",
        "bachelors_field": null
    },

    "location": {
        "country": "대한민국",
        "province": "광주",
        "district": "서구"
    },

    "occupation": "하역 및 적재 관련 단순 종사원",

    "personas": {
        "summary": "광주 서구에서 평생 하역 일을 하며 살아온 70대 가장으로...",
        "professional": "광주 서구의 하역 현장에서 수십 년간...",
        "sports": "매일 새벽 5시에 일어나...",
        "arts": "트로트 음악을 즐기며...",
        "travel": "가까운 무등산 등반을 즐기며...",
        "culinary": "된장찌개와 막걸리를 즐기는...",
        "family": "30년 넘게 한 사람과 살아온..."
    },

    "cultural_background": "광주 토박이로서...",
    "career_goals": "은퇴 후에도 현장 경험을 전수하고 싶어하며...",

    "skills": ["중장비 운전", "자재 관리", "안전 관리"],
    "hobbies": ["등산", "낚시", "트로트 노래"],

    "community": {
        "community_id": 42,
        "label": "광주/전남 등산 + 트로트"
    },

    "similar_preview": [
        {
            "uuid": "7a8b9c...",
            "display_name": "김영수",
            "age": 71,
            "similarity": 0.89,
            "shared_hobbies": ["등산", "낚시"]
        }
    ],

    "graph_stats": {
        "total_connections": 18,
        "hobby_count": 3,
        "skill_count": 3
    }
}
```

**Cypher 쿼리 전략**:

단일 UUID에 대해 다음을 병렬/순차 쿼리로 수집한다:

```cypher
// 1. 기본 프로필 + 모든 관계 엔티티
MATCH (p:Person {uuid: $uuid})
OPTIONAL MATCH (p)-[:LIVES_IN]->(d:District)-[:IN_PROVINCE]->(prov:Province)-[:IN_COUNTRY]->(c:Country)
OPTIONAL MATCH (p)-[:WORKS_AS]->(occ:Occupation)
OPTIONAL MATCH (p)-[:EDUCATED_AT]->(edu:EducationLevel)
OPTIONAL MATCH (p)-[:MAJORED_IN]->(field:Field)
OPTIONAL MATCH (p)-[:MARITAL_STATUS]->(ms:MaritalStatus)
OPTIONAL MATCH (p)-[:MILITARY_STATUS]->(mil:MilitaryStatus)
OPTIONAL MATCH (p)-[:LIVES_WITH]->(ft:FamilyType)
OPTIONAL MATCH (p)-[:LIVES_IN_HOUSING]->(ht:HousingType)
OPTIONAL MATCH (p)-[:HAS_SKILL]->(s:Skill)
OPTIONAL MATCH (p)-[:ENJOYS_HOBBY]->(h:Hobby)
RETURN p, d, prov, c, occ, edu, field, ms, mil, ft, ht,
       collect(DISTINCT s.name) AS skills,
       collect(DISTINCT h.name) AS hobbies

// 2. 유사 인물 미리보기 (SIMILAR_TO 관계가 있을 경우)
MATCH (p:Person {uuid: $uuid})-[r:SIMILAR_TO]->(sim:Person)
RETURN sim.uuid, sim.display_name, sim.age, r.score
ORDER BY r.score DESC LIMIT 3

// 3. 커뮤니티 정보
MATCH (p:Person {uuid: $uuid})
WHERE p.community_id IS NOT NULL
RETURN p.community_id
```

**구현 모듈**:
- `src/api/routes/persona.py` — 엔드포인트
- `src/graph/persona_queries.py` — 프로필 전용 Cypher 쿼리
- `src/api/schemas.py` — `PersonaProfileResponse` 모델 추가

**테스트 항목**:
- 존재하는 UUID → 전체 프로필 반환
- 존재하지 않는 UUID → 404 NotFoundException
- skills/hobbies가 빈 리스트일 수 있음 (정상 동작)
- community_id가 null일 수 있음 (GDS 미실행 상태)
- similar_preview가 빈 리스트일 수 있음 (KNN 미실행 상태)

---

### 2.4 기능 F8: 크로스 세그먼트 비교 분석

**엔드포인트**: `POST /api/compare/segments`

두 개의 필터 조건(세그먼트)을 입력받아, 각 세그먼트의 취미·직업·학력·가족 분포를 비교하고, LLM으로 차이점을 한국어로 해석한다.

**입력**:
```json
{
    "segment_a": {
        "label": "서울 30대 남성",
        "filters": {
            "province": "서울",
            "age_group": "30대",
            "sex": "남자"
        }
    },
    "segment_b": {
        "label": "부산 30대 남성",
        "filters": {
            "province": "부산",
            "age_group": "30대",
            "sex": "남자"
        }
    },
    "dimensions": ["hobby", "occupation", "education"],
    "top_k": 10
}
```

**필터 조건 (segment_a, segment_b 공통)**:

| 필드 | 타입 | 설명 |
|------|------|------|
| `province` | string | 시도 |
| `district` | string | 시군구 |
| `age_group` | string | 연령대 |
| `sex` | string | 성별 |
| `education_level` | string | 학력 |
| `hobby` | string | 취미 보유 조건 |
| `skill` | string | 스킬 보유 조건 |

**응답**:
```json
{
    "segment_a": {
        "label": "서울 30대 남성",
        "count": 48200
    },
    "segment_b": {
        "label": "부산 30대 남성",
        "count": 18500
    },
    "comparisons": {
        "hobby": {
            "segment_a": [
                {"name": "헬스", "count": 15424, "ratio": 0.320},
                {"name": "등산", "count": 13496, "ratio": 0.280},
                {"name": "독서", "count": 7230, "ratio": 0.150}
            ],
            "segment_b": [
                {"name": "낚시", "count": 6475, "ratio": 0.350},
                {"name": "등산", "count": 4625, "ratio": 0.250},
                {"name": "서핑", "count": 2220, "ratio": 0.120}
            ],
            "common": ["등산"],
            "only_a": ["헬스", "독서"],
            "only_b": ["낚시", "서핑"]
        },
        "occupation": { ... },
        "education": { ... }
    },
    "ai_analysis": "서울과 부산의 30대 남성은 '등산'이라는 공통 취미를 공유하지만, 서울은 실내 운동(헬스)과 자기계발(독서) 중심인 반면, 부산은 해양 레저(낚시, 서핑) 선호도가 뚜렷합니다. 직업 분포에서는..."
}
```

**처리 흐름**:
1. 각 세그먼트에 대해 F6의 드릴다운 통계 쿼리를 재활용하여 분포를 집계한다.
2. 두 분포를 비교하여 `common`, `only_a`, `only_b` 항목을 산출한다.
3. 비교 결과를 LLM (NVIDIA API DeepSeek-V4-Pro)에 전달하여 한국어 분석 텍스트를 생성한다.

**LLM 프롬프트 전략**:
```
다음은 두 인구 집단의 {dimension} 분포 비교 결과입니다.

[그룹 A: {label_a} ({count_a}명)]
{distribution_a}

[그룹 B: {label_b} ({count_b}명)]
{distribution_b}

두 그룹의 공통점과 차이점을 3~5문장으로 한국어로 분석해주세요.
```

**구현 모듈**:
- `src/api/routes/compare.py` — 엔드포인트
- `src/graph/stats_queries.py` — F6의 집계 쿼리 재활용
- `src/rag/compare_chain.py` — 비교 분석용 LLM 체인
- `src/api/schemas.py` — `SegmentCompareRequest`, `SegmentCompareResponse` 모델 추가

**테스트 항목**:
- 유효한 두 세그먼트 → 비교 결과 + AI 분석 반환
- 동일 필터 두 세그먼트 → 비교 결과에 `only_a`, `only_b` 빈 리스트
- 결과 0건 세그먼트 → `count: 0` + 빈 분포 (에러 아님)
- dimensions에 잘못된 값 → 400 에러
- LLM 호출 실패 시 → ai_analysis를 빈 문자열로 반환 (graceful degradation)

---

### 2.5 기능 F9: 지식 그래프 서브그래프 시각화

**엔드포인트**: `GET /api/graph/subgraph/{uuid}`

특정 Person 노드를 중심으로 연결된 모든 엔티티 노드(취미, 스킬, 지역, 직업 등)와 관계를 **그래프 시각화용 JSON (nodes + edges)**으로 반환한다.

프론트엔드에서 D3.js, vis.js, Cytoscape.js 등으로 인터랙티브 그래프를 렌더링할 수 있다.

**쿼리 파라미터**:

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|-------|------|
| `depth` | int | 1 | 탐색 깊이. 1=직접 연결만, 2=2-hop까지 (최대 2) |
| `include_similar` | bool | false | SIMILAR_TO 관계 포함 여부 |
| `max_nodes` | int | 50 | 반환 최대 노드 수 (그래프 폭발 방지) |

**응답**:
```json
{
    "center_uuid": "03b4f36a18e6469386d0286dddd513c8",
    "center_label": "전기태",
    "node_count": 18,
    "edge_count": 17,
    "nodes": [
        {
            "id": "person_03b4f36a...",
            "label": "전기태",
            "type": "Person",
            "properties": {
                "age": 74,
                "sex": "남자",
                "persona": "광주 서구에서 평생 하역 일을 하며..."
            }
        },
        {
            "id": "hobby_등산",
            "label": "등산",
            "type": "Hobby",
            "properties": {}
        },
        {
            "id": "district_광주-서구",
            "label": "서구",
            "type": "District",
            "properties": {}
        },
        {
            "id": "province_광주",
            "label": "광주",
            "type": "Province",
            "properties": {}
        }
    ],
    "edges": [
        {
            "source": "person_03b4f36a...",
            "target": "hobby_등산",
            "type": "ENJOYS_HOBBY"
        },
        {
            "source": "person_03b4f36a...",
            "target": "district_광주-서구",
            "type": "LIVES_IN"
        },
        {
            "source": "district_광주-서구",
            "target": "province_광주",
            "type": "IN_PROVINCE"
        }
    ]
}
```

**depth=2 동작**:

depth=2일 경우, 중심 Person과 같은 Hobby를 공유하는 **다른 Person 노드**도 포함된다.
이를 통해 "등산을 같이 좋아하는 사람들의 네트워크"를 시각화할 수 있다.

```cypher
// depth=1: 직접 연결
MATCH (p:Person {uuid: $uuid})-[r]-(n)
WHERE NOT type(r) = 'SIMILAR_TO' OR $include_similar = true
RETURN p, r, n

// depth=2: 2-hop (같은 엔티티를 공유하는 다른 Person)
MATCH (p:Person {uuid: $uuid})-[r1]-(entity)-[r2]-(other:Person)
WHERE entity:Hobby OR entity:Skill OR entity:District
  AND other.uuid <> $uuid
RETURN entity, r2, other
LIMIT $max_secondary_nodes
```

**노드 타입별 시각화 힌트** (프론트엔드 참고용):

| 노드 타입 | 추천 색상 | 추천 아이콘/형태 |
|----------|----------|----------------|
| Person | #4A90D9 (파란색) | 원형, 크기=연결 수 비례 |
| Hobby | #7ED321 (초록색) | 둥근 사각형 |
| Skill | #F5A623 (주황색) | 다이아몬드 |
| District | #D0021B (빨간색) | 삼각형 |
| Province | #BD10E0 (보라색) | 큰 삼각형 |
| Occupation | #9013FE (진보라) | 육각형 |
| EducationLevel | #50E3C2 (청록색) | 사각형 |

**구현 모듈**:
- `src/api/routes/graph_viz.py` — 엔드포인트
- `src/graph/subgraph_queries.py` — 서브그래프 추출 Cypher
- `src/api/schemas.py` — `SubgraphResponse`, `GraphNode`, `GraphEdge` 모델 추가

**테스트 항목**:
- 존재하는 UUID, depth=1 → 중심 노드 + 직접 연결 엔티티
- depth=2 → 2-hop Person 노드 포함
- include_similar=true → SIMILAR_TO 엣지 포함
- max_nodes 제한 동작 (50개 초과 시 잘라냄)
- 존재하지 않는 UUID → 404 에러
- depth > 2 → 400 에러 (그래프 폭발 방지)

---

## 3. 확장 아키텍처

Phase 2 기능 추가 후의 전체 아키텍처:

```
[프론트엔드 (Streamlit / React)]
    ├── 대시보드 홈        → GET /api/stats
    ├── 드릴다운 차트       → GET /api/stats/{dimension}
    ├── 검색/필터          → GET /api/search
    ├── 프로필 상세        → GET /api/persona/{uuid}
    ├── 세그먼트 비교       → POST /api/compare/segments
    ├── 그래프 시각화       → GET /api/graph/subgraph/{uuid}
    │
    │   (기존 Phase 1)
    ├── 인사이트 질의       → POST /api/insight
    ├── 유사 매칭          → POST /api/similar/{uuid}
    ├── 커뮤니티 탐지       → GET /api/communities
    └── 관계 경로          → GET /api/path/{u1}/{u2}

[FastAPI]
    ├── src/api/routes/
    │   ├── search.py          (NEW - F5)
    │   ├── stats.py           (NEW - F6)
    │   ├── persona.py         (NEW - F7)
    │   ├── compare.py         (NEW - F8)
    │   ├── graph_viz.py       (NEW - F9)
    │   ├── insight.py         (기존)
    │   ├── similar.py         (기존)
    │   ├── communities.py     (기존)
    │   └── path.py            (기존)
    │
    ├── src/graph/
    │   ├── search_queries.py   (NEW - F5)
    │   ├── stats_queries.py    (NEW - F6, F8 공유)
    │   ├── persona_queries.py  (NEW - F7)
    │   ├── subgraph_queries.py (NEW - F9)
    │   ├── schema.py           (기존)
    │   ├── loader.py           (기존)
    │   └── queries.py          (기존)
    │
    └── src/rag/
        ├── compare_chain.py    (NEW - F8)
        ├── router.py           (기존)
        ├── cypher_chain.py     (기존)
        ├── vector_chain.py     (기존)
        └── llm.py              (기존)
```

---

## 4. 신규 프로젝트 구조 (추가 파일만)

```
src/
├── api/
│   └── routes/
│       ├── search.py           # F5: GET /api/search
│       ├── stats.py            # F6: GET /api/stats, GET /api/stats/{dimension}
│       ├── persona.py          # F7: GET /api/persona/{uuid}
│       ├── compare.py          # F8: POST /api/compare/segments
│       └── graph_viz.py        # F9: GET /api/graph/subgraph/{uuid}
├── graph/
│   ├── search_queries.py       # F5: 동적 Cypher 빌더
│   ├── stats_queries.py        # F6+F8: 집계 쿼리 모음
│   ├── persona_queries.py      # F7: 프로필 Cypher 쿼리
│   └── subgraph_queries.py     # F9: 서브그래프 추출 쿼리
└── rag/
    └── compare_chain.py        # F8: 세그먼트 비교 LLM 체인

tests/
├── test_search.py              # F5 테스트
├── test_stats.py               # F6 테스트
├── test_persona.py             # F7 테스트
├── test_compare.py             # F8 테스트
└── test_graph_viz.py           # F9 테스트
```

---

## 5. 구현 순서 및 의존 관계

```
F7 (프로필 뷰)          ← 의존성 없음, 가장 먼저 구현
  ↓
F6 (인구통계 대시보드)    ← 의존성 없음, F7과 병렬 가능
  ↓
F5 (검색/필터 엔진)      ← F7 프로필 뷰의 결과 스키마 재활용
  ↓
F8 (세그먼트 비교)       ← F6 집계 쿼리 재활용 + LLM
  ↓
F9 (그래프 시각화)       ← 의존성 없음, 언제든 구현 가능
```

**권장 구현 순서**: F7 → F6 → F5 → F9 → F8

이유:
1. **F7** (프로필): 다른 모든 기능의 "클릭하면 도착하는 곳". 가장 먼저 있어야 함.
2. **F6** (대시보드): 홈 화면. 집계 쿼리를 F8에서 재활용하므로 먼저 구현.
3. **F5** (검색): F7 프로필의 결과 스키마를 검색 결과에 재활용.
4. **F9** (그래프): 독립적이며 프론트 시각적 임팩트가 큼.
5. **F8** (비교): F6 쿼리 + LLM 조합. 가장 복잡하므로 마지막.

---

## 6. 성공 기준

| 기능 | 성공 기준 |
|------|----------|
| F5 검색 | 다중 필터 조합 → 정확한 결과 반환, 페이지네이션 정상 동작, 응답 < 2초 (10K 샘플) |
| F6 대시보드 | 전체 통계 8개 분포 정상 반환, 드릴다운 필터 적용 시 정확한 부분 집계 |
| F7 프로필 | UUID → 전체 프로필 (인구통계 + 7개 페르소나 + 취미/스킬 + 커뮤니티 + 유사인물) 반환 |
| F8 세그먼트 비교 | 두 필터 조건 → 차원별 분포 비교 + 공통/차이 항목 + 한국어 AI 분석 |
| F9 그래프 시각화 | UUID → nodes/edges JSON → 프론트에서 인터랙티브 그래프 렌더링 가능 |

---

## 7. 리스크 및 대응

| 리스크 | 대응 |
|--------|------|
| 검색 쿼리 성능 (1M rows 풀스캔) | Neo4j 인덱스 활용 (age, sex, age_group 인덱스 이미 존재), 복합 인덱스 추가 검토 |
| 대시보드 집계 쿼리 느림 | 결과 캐싱 (in-memory 또는 Redis), 데이터 변경 없으므로 TTL 길게 설정 |
| 그래프 시각화 depth=2 노드 폭발 | max_nodes 파라미터로 강제 제한, 프론트에서 lazy loading |
| 세그먼트 비교 LLM 호출 비용 | NVIDIA API LLM 호출 제한 고려, 분포 데이터만으로도 가치 있으므로 ai_analysis 없이도 동작 |
| 프로필 뷰 쿼리 복잡도 | 단일 Cypher로 모든 OPTIONAL MATCH 결합 또는 2~3개 쿼리 분리 실행 |
