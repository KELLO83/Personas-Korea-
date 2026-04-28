# PRD: 한국인 페르소나 지식 그래프 인사이트 플랫폼

## 1. 프로젝트 개요

### 1.1 프로젝트명
**Korean Persona Knowledge Graph Insight Platform** (내부명: `persona-kg`)

### 1.2 한 줄 설명
NVIDIA Nemotron-Personas-Korea의 한국인 페르소나 데이터를 지식 그래프로 구축하고, 자연어 인사이트 질의와 유사 페르소나 매칭을 제공하는 시스템

### 1.3 프로젝트 목표
- 페르소나 데이터를 Neo4j 지식 그래프로 구조화
- 자연어 질문으로 그래프 인사이트를 도출 (GraphRAG)
- Neo4j GDS 그래프 임베딩 기반 페르소나 유사도 매칭
- 커뮤니티 자동 탐지 및 관계 경로 시각화

---

## 2. 데이터 정의서

### 2.1 데이터셋 개요
- **원본 행 수**: 1,000,000 rows (`default` subset, `train` split)
- **개발/검증 권장 단위**: 10K~100K 샘플부터 시작 후 전체 1M 확장
- **열 수**: 26
- **출처**: `nvidia/Nemotron-Personas-Korea` (Hugging Face)
- **형식**: parquet / optimized-parquet
- **언어**: Korean
- **모달리티**: Text + Image
- **라이선스**: CC-BY-4.0
- **저장소 크기**: 약 1.98GB

### 2.2 필드 정의

#### A. 식별자
| 필드 | 타입 | 설명 | 그래프 매핑 |
|---|---|---|---|
| `uuid` | string (32자 hex) | 고유 ID | `Person` 노드 PK |

#### B. 서술형 페르소나 (6개 관점)
각 페르소나는 같은 사람의 다른 측면을 묘사. 내부에 성격의 모순/갈등이 내재.

| 필드 | 관점 | 그래프 매핑 |
|---|---|---|
| `professional_persona` | 직업/업무 성향 | `Person.professional_persona` 속성 |
| `sports_persona` | 신체/운동 습관 | `Person.sports_persona` 속성 |
| `arts_persona` | 문화/예술 활동 | `Person.arts_persona` 속성 |
| `travel_persona` | 여행/이동 스타일 | `Person.travel_persona` 속성 |
| `culinary_persona` | 식문화 | `Person.culinary_persona` 속성 |
| `family_persona` | 가족/관계 | `Person.family_persona` 속성 |

#### C. 요약/종합 서술
| 필드 | 타입 | 그래프 매핑 |
|---|---|---|
| `persona` | string | `Person` 노드 속성 |
| `cultural_background` | string | `Person.cultural_background` 속성 |
| `skills_and_expertise` | string | `Person` 노드 속성 |
| `hobbies_and_interests` | string | `Person` 노드 속성 |
| `career_goals_and_ambitions` | string | `Person.career_goals_and_ambitions` 속성 |

데이터셋에는 별도 `name` 컬럼이 없다. API/UI에서 사람을 표시할 때는 `uuid`와 `persona` 요약을 함께 사용하고, 필요하면 서술형 텍스트의 첫 호칭을 `display_name`으로 파생한다.

#### C-1. 벡터 검색 단위

MVP에서는 각 사람별로 하나의 통합 텍스트를 만들어 `Person.text_embedding`에 저장한다. 통합 텍스트는 다음 필드를 순서대로 결합한다.

1. `persona`
2. `professional_persona`
3. `sports_persona`
4. `arts_persona`
5. `travel_persona`
6. `culinary_persona`
7. `family_persona`
8. `cultural_background`
9. `skills_and_expertise`
10. `hobbies_and_interests`
11. `career_goals_and_ambitions`

후속 버전에서는 6개 페르소나 관점별 별도 임베딩을 추가할 수 있지만, MVP에서는 Neo4j의 단일 `Person` 벡터 인덱스를 기준으로 구현한다.

#### D. 리스트형 속성 (엔티티 추출 대상)
| 필드 | 포맷 | 그래프 매핑 |
|---|---|---|
| `skills_and_expertise_list` | Python list string | `(Person)-[:HAS_SKILL]->(:Skill)` |
| `hobbies_and_interests_list` | Python list string | `(Person)-[:ENJOYS_HOBBY]->(:Hobby)` |

#### E. 범주형 속성
| 필드 | 고유값 수 | 그래프 매핑 |
|---|---|---|
| `sex` | 2 | `Person.sex` 속성 |
| `age` | 19~99 정수 | `Person.age` 속성 + 10단위 연령대 파생 |
| `marital_status` | 4 | `(Person)-[:MARITAL_STATUS]->(:MaritalStatus)` |
| `military_status` | 2 | `(Person)-[:MILITARY_STATUS]->(:MilitaryStatus)` |
| `family_type` | 39 | `(Person)-[:LIVES_WITH]->(:FamilyType)` |
| `housing_type` | 6 | `(Person)-[:LIVES_IN_HOUSING]->(:HousingType)` |
| `education_level` | 7 | `(Person)-[:EDUCATED_AT]->(:EducationLevel)` |
| `bachelors_field` | 11 | `(Person)-[:MAJORED_IN]->(:Field)` |
| `occupation` | 문자열 길이 2~40 | `(Person)-[:WORKS_AS]->(:Occupation)` |
| `district` | 252 | `(Person)-[:LIVES_IN]->(:District)-[:IN_PROVINCE]->(:Province)` |
| `province` | 17 | `(:Province)-[:IN_COUNTRY]->(:Country)` |
| `country` | 1 | `(:Country)` |

### 2.3 데이터 품질 이슈
| 이슈 | 대응 방안 |
|---|---|
| `skills_list` Python list string 파싱 | `ast.literal_eval()` 사용 |
| `hobbies_list` 항목 정규화 | 유사도 기반 엔티티 병합 |
| `occupation` ~240개 고유값 정규화 | 원문 그대로 사용 (KSCO 매핑은 향후) |
| `district` `"서울-서초구"` 포맷 | `Province-District` 분리 파싱 |
| `bachelors_field` "해당없음" 67.4% | "해당없음" 노드 생성 제외 |

### 2.4 데이터 활용 방법

1차 로딩은 Hugging Face `datasets` 라이브러리를 사용한다.

```python
from datasets import load_dataset

dataset = load_dataset("nvidia/Nemotron-Personas-Korea", split="train")
df = dataset.to_pandas()
```

개발 중에는 전체 1M rows를 바로 처리하지 않고 `DATA_SAMPLE_SIZE`로 샘플링한다.

```python
dataset = load_dataset("nvidia/Nemotron-Personas-Korea", split="train[:10000]")
```

로컬 파일을 사용할 경우 `data/raw/personas.parquet` 또는 `data/raw/personas.csv`를 우선 탐색한다. 로컬 파일이 없으면 Hugging Face에서 다운로드한다.

---

## 3. 그래프 스키마

```
(:Person {
    uuid,
    display_name,
    age,
    age_group,
    sex,
    persona,
    professional_persona,
    sports_persona,
    arts_persona,
    travel_persona,
    culinary_persona,
    family_persona,
    cultural_background,
    skills_and_expertise,
    hobbies_and_interests,
    career_goals_and_ambitions,
    text_embedding
})
  -[:LIVES_IN]->(:District {name})-[:IN_PROVINCE]->(:Province {name})-[:IN_COUNTRY]->(:Country {name})
  -[:WORKS_AS]->(:Occupation {name})
  -[:HAS_SKILL]->(:Skill {name})
  -[:ENJOYS_HOBBY]->(:Hobby {name})
  -[:EDUCATED_AT]->(:EducationLevel {name})
  -[:MAJORED_IN]->(:Field {name})
  -[:MARITAL_STATUS]->(:MaritalStatus {name})
  -[:LIVES_WITH]->(:FamilyType {name})
  -[:LIVES_IN_HOUSING]->(:HousingType {name})
```

---

## 4. 기능 명세서

### 4.1 기능 1: 자연어 인사이트 질의 (GraphRAG)

**엔드포인트**: `POST /api/insight`

**입력**:
```json
{
    "question": "광주 30대 남성들이 공통으로 즐기는 취미는?"
}
```

**출력**:
```json
{
    "answer": "광주 지역 30대 남성들의 공통 취미는...",
    "sources": [
        {"type": "cypher", "query": "MATCH (p:Person)..."},
        {"type": "vector", "score": 0.87}
    ],
    "query_type": "aggregation"
}
```

**처리 흐름**:
1. LangGraph Router가 질문 유형 판단 (통계/집계 vs 의미론적)
2. 통계 질의 → GraphCypherQAChain → Cypher 실행 → LLM 한국어 응답
3. 의미론적 질의 → Vector 검색 → LLM 한국어 응답

### 4.2 기능 2: 유사 페르소나 매칭 (그래프 임베딩)

**엔드포인트**: `POST /api/similar/{uuid}`

**입력**:
```json
{
    "top_k": 5,
    "weights": {"graph": 0.6, "text": 0.4}
}
```

**출력**:
```json
{
    "query_persona": {"uuid": "...", "name_summary": "최은지, 71세, 서초구 회계 사무원"},
    "similar_personas": [
        {
            "uuid": "...",
            "summary": "박영희, 69세, 강남구 세무사",
            "similarity": 0.89,
            "shared_traits": ["4년제 대학교", "자연과학·수학"],
            "shared_hobbies": ["고궁 산책", "한식집 탐방"]
        }
    ]
}
```

**처리 흐름**:
1. Neo4j GDS에서 FastRP 그래프 임베딩 조회
2. Neo4j GDS KNN으로 top-k 유사 Person 노드 탐색
3. 공통 속성/취미 추출
4. 필요 시 KURE-v1 텍스트 임베딩 유사도와 가중 결합
5. LLM으로 매칭 사유 한국어 생성

### 4.3 기능 3: 커뮤니티 자동 탐지

**엔드포인트**: `GET /api/communities`

**입력**:
```
?algorithm=leiden&min_size=10
```

**출력**:
```json
{
    "communities": [
        {
            "id": 0,
            "label": "역사 여행 + 트로트 + 전통시장",
            "size": 842,
            "top_traits": {"province": "광주", "hobbies": ["등산", "목욕탕"]},
            "representative_persona_uuid": "03b4f36a..."
        }
    ]
}
```

**처리 흐름**:
1. Neo4j GDS Leiden 알고리즘 실행
2. 각 커뮤니티의 특성 추출 (빈도 기반)
3. LLM으로 커뮤니티 라벨 한국어 생성

### 4.4 기능 4: 관계 경로 탐색

**엔드포인트**: `GET /api/path/{uuid1}/{uuid2}`

**입력**:
```
?max_depth=4
```

**출력**:
```json
{
    "path_found": true,
    "length": 3,
    "path": [
        {"node": "전기태 (Person)", "edge": "ENJOYS_HOBBY"},
        {"node": "역사 유적지 여행 (Hobby)", "edge": "ENJOYS_HOBBY"},
        {"node": "최은지 (Person)"}
    ],
    "shared_nodes": [
        {"type": "Hobby", "name": "역사 유적지 여행"}
    ],
    "summary": "전기태와 최은지는 '역사 유적지 여행' 취미를 공유합니다."
}
```

**처리 흐름**:
1. Neo4j `shortestPath` Cypher 쿼리 실행
2. 경로 상 노드/엣지 추출
3. 공통 노드 식별
4. LLM으로 관계 요약 한국어 생성

---

## 5. 기술 스택

| 구성요소 | 기술 | 버전/상세 |
|---|---|---|
| **임베딩 모델** | KURE-v1 (로컬) | nlpai-lab/KURE-v1, float16, VRAM ~1.6GB |
| **LLM** | NVIDIA API DeepSeek-V4-Pro | deepseek-ai/deepseek-v4-pro |
| **지식 그래프 DB** | Neo4j | Community Edition 5.x |
| **그래프 알고리즘** | Neo4j GDS | FastRP, Leiden, KNN |
| **벡터 검색** | Neo4j Vector Index | KURE-v1 임베딩을 Neo4j 노드 속성으로 저장 |
| **RAG 프레임워크** | LangChain | langchain, langchain-community, langchain-neo4j |
| **라우팅** | LangGraph | StateGraph 기반 질의 분기 |
| **런타임** | Python | 3.11 (`.venv` 표준, 3.14 미사용) |
| **데이터 처리** | Pandas + Python | pandas, pyarrow, datasets |
| **API** | FastAPI | 0.100+ |
| **프론트** | Streamlit | 1.30+ |
| **형태소 분석** | kiwipiepy | 한국어 전처리 |

### 하드웨어 제약
- GPU: NVIDIA RTX 4080 Laptop (8GB VRAM)
- KURE-v1 임베딩: float16 시 ~1.6GB VRAM 사용, 여유 충분

### 로컬 개발 환경

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Python 3.14는 현재 설치되어 있어도 이 프로젝트에서는 사용하지 않는다. ML/임베딩/Neo4j/LangChain 계열 의존성 호환성을 위해 Python 3.11을 기준 런타임으로 고정한다.

---

## 6. 아키텍처

```
[Streamlit UI] ←→ [FastAPI]
                    ├── POST /api/insight       → LangGraph Router → Cypher / Vector
                    ├── POST /api/similar/{uuid} → Neo4j GDS FastRP KNN
                    ├── GET  /api/communities   → Neo4j GDS Leiden
                    └── GET  /api/path/{u1}/{u2} → Neo4j shortestPath
                    
[LangGraph Router]
    ├── 통계/집계 질의 → LangChain GraphCypherQAChain → Neo4j Cypher
    ├── 의미론적 질의 → Neo4j Vector Index → KURE-v1 임베딩
    └── 복합 질의     → Cypher + Vector 결합 → LLM 응답 생성

[Neo4j]
    ├── Person 노드 (개발 샘플 10K~100K, 최종 1M)
    ├── 관계 엣지 (Skills, Hobbies, District 등)
    ├── FastRP 임베딩 (GDS)
    └── Leiden 커뮤니티 (GDS)

[KURE-v1 (로컬 GPU)]
    └── 페르소나 텍스트 임베딩 → Neo4j 벡터 인덱스에 저장
```

---

## 7. 프로젝트 구조

```
persona-kg/
├── PRD.md                          # 본 문서
├── CHECKLIST.md                    # 체크리스트
├── .env.example                    # 환경변수 템플릿
├── requirements.txt                # 의존성
├── data/
│   └── raw/                        # 원시 CSV/Parquet 데이터
├── src/
│   ├── __init__.py
│   ├── config.py                   # 설정 (환경변수, 상수)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py               # CSV/Parquet 로더
│   │   ├── parser.py               # 리스트 파싱, district 분리
│   │   └── preprocessor.py         # 전처리 (결측치, 정규화)
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── schema.py               # 그래프 스키마 정의
│   │   ├── loader.py               # Neo4j 적재
│   │   └── queries.py              # 자주 쓰는 Cypher 쿼리
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── kure_model.py           # KURE-v1 임베딩 모델
│   │   └── vector_index.py          # Neo4j 벡터 인덱스 관리
│   ├── gds/
│   │   ├── __init__.py
│   │   ├── fastrp.py               # FastRP 임베딩 파이프라인
│   │   ├── communities.py          # Leiden 커뮤니티 탐지
│   │   └── similarity.py            # KNN 유사도 매칭
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── router.py                # LangGraph StateGraph 라우터
│   │   ├── cypher_chain.py          # GraphCypherQAChain
│   │   ├── vector_chain.py          # Vector RAG 체인
│   │   └── llm.py                  # NVIDIA API LLM 클라이언트
│   └── api/
│       ├── __init__.py
│       ├── main.py                  # FastAPI 앱
│       ├── routes/
│       │   ├── __init__.py
│       │   ├── insight.py           # POST /api/insight
│       │   ├── similar.py           # POST /api/similar/{uuid}
│       │   ├── communities.py       # GET /api/communities
│       │   └── path.py              # GET /api/path/{uuid1}/{uuid2}
│       └── schemas.py               # Pydantic 모델
├── app/
│   └── streamlit_app.py            # Streamlit 프론트엔드
├── scripts/
│   ├── build_graph.py              # 그래프 구축 스크립트
│   ├── build_embeddings.py         # 임베딩 생성 스크립트
│   └── build_gds.py                # GDS 파이프라인 스크립트
└── tests/
    ├── test_data_parser.py
    ├── test_graph_loader.py
    ├── test_embeddings.py
    └── test_api.py
```

---

## 8. 성공 기준

| 기능 | 성공 기준 |
|---|---|
| 인사이트 질의 | 한국어 자연어 질문 → 정확한 Cypher 쿼리 생성 → 한국어 인사이트 응답 |
| 유사 매칭 | uuid 입력 → top-5 유사 페르소나 + 공통 속성 + 유사도 점수 반환 |
| 커뮤니티 탐지 | 개발 샘플 및 전체 1M 데이터 → Leiden 실행 → 의미 있는 클러스터 + 한국어 라벨 |
| 관계 경로 | 두 uuid → 최단 경로 + 공통 노드 + 한국어 요약 |

---

## 9. 리스크 및 대응

| 리스크 | 대응 |
|---|---|
| NVIDIA API LLM 호출 제한 | LLM 호출 최소화, Cypher 템플릿 캐싱 |
| KURE-v1 임베딩 1M행 시간 | `DATA_SAMPLE_SIZE`로 단계적 처리, 배치 처리 (batch_size=64), 진행률 표시 |
| Neo4j GDS 메모리 | FastRP 임베딩 차원 축소 (1024→256) |
| Cypher 쿼리 생성 오류 | validate_cypher=True + 재시도 로직 |
| 한국어 텍스트 전처리 | kiwipiepy 형태소 분석 적용 |
