# PRD: 한국인 페르소나 지식 그래프 인사이트 플랫폼 v2.0

> **Unified Product Requirements Document**  
> 본 문서는 Phase 1~3의 모든 기능 요구사항을 통합 관리합니다.  
> 과거 버전: `docs/prd-archive/prd-v1.0-phase1.md`, `prd-v1.5-phase2.md`, `prd-v2.0-expansion.md`

---

## 1. 개요 (Executive Summary)

### 1.1 프로젝트명
**Korean Persona Knowledge Graph Insight Platform** (내부명: `persona-kg`)

### 1.2 목표
NVIDIA Nemotron-Personas-Korea의 100만 한국인 페르소나 데이터를 Neo4j 지식 그래프로 구조화하고, 자연어 인사이트 질의·유사 매칭·네트워크 분석·추천·대화형 탐색을 제공하는 시스템

### 1.3 현재 상태 (Status Dashboard)

| Phase | 기능 | 상태 | 문서 |
|:---|:---|:---|:---|
| **Phase 1** | GraphRAG 인사이트, 유사 페르소나 매칭, 커뮤니티 탐지, 관계 경로 | ✅ 완료 | `docs/prd-archive/prd-v1.0-phase1.md` |
| **Phase 2** | 검색/필터(F5), 통계 대시보드(F6), 프로필 상세(F7), 세그먼트 비교(F8), 서브그래프 시각화(F9) | ✅ 완료 | `docs/prd-archive/prd-v1.5-phase2.md` |
| **Phase 3** | 네트워크 영향력(F10), 추천 엔진(F11), 대화형 챗봇(F12) | ✅ 핵심 구현 완료 | 본 문서 §3~5 |
| **Phase 4** | 대규모 운영검증(F13), 확장/고도화(F14), 챗봇 LLM 합성(F15) | 📝 계획/검수 단계 | 본 문서 §11 |

### 1.4 검수 이력 (Review History)

| 날짜 | 검수자 | 결과 | 핵심 조치사항 |
|:---|:---|:---|:---|
| 2026-04-28 | Momus (Plan Critic) | ✅ OKAY | PRD 실행 가능, 구조 양호 |
| 2026-04-28 | Oracle (Architecture) | ⚠️ PASS with Concerns | GDS 실시간 계산 금지 → 배치 전환 (ADR-001) |
| 2026-04-28 | Metis (Pre-planning) | ⚠️ High Risk | Feature 1이 최고 위험, 의존성 명시 필요 |
| 2026-04-28 | 예정 | 📝 PENDING | Phase 4 대규모 운영검증 및 확장/고도화 계획 검수 |

---

## 2. 기능 목록 (Feature Catalog)

### 2.1 전체 기능 맵

| 코드 | 기능명 | 엔드포인트 | Phase | 상태 | 핵심 기술 |
|:---|:---|:---|:---:|:---:|:---|
| F1 | GraphRAG 인사이트 질의 | `POST /api/insight` | 1 | ✅ | LangGraph, Cypher/Vector/Composite 라우팅 |
| F2 | 유사 페르소나 매칭 | `POST /api/similar/{uuid}` | 1 | ✅ | GDS FastRP + KNN, KURE-v1 임베딩 |
| F3 | 커뮤니티 자동 탐지 | `GET /api/communities` | 1 | ✅ | GDS Leiden |
| F4 | 관계 경로 탐색 | `GET /api/path/{u1}/{u2}` | 1 | ✅ | shortestPath + 공유 노드 |
| F5 | 페르소나 검색/필터 | `GET /api/search` | 2 | ✅ | 동적 Cypher 빌더, 페이지네이션 |
| F6 | 인구통계 대시보드 | `GET /api/stats` | 2 | ✅ | 집계 쿼리, 드릴다운 |
| F7 | 프로필 상세 뷰 | `GET /api/persona/{uuid}` | 2 | ✅ | OPTIONAL MATCH, 유사 인물 미리보기 |
| F8 | 세그먼트 비교 분석 | `POST /api/compare/segments` | 2 | ✅ | 분포 비교 + LLM 해석 |
| F9 | 서브그래프 시각화 | `GET /api/graph/subgraph/{uuid}` | 2 | ✅ | 깊이별 추출 (1~3 hop) |
| **F10** | **네트워크 영향력 분석** | `GET /api/influence/top` | **3** | 📝 | **PageRank, Betweenness, Degree** |
| **F10** | **노드 제거 시뮬레이션** | `POST /api/influence/simulate-removal` | **3** | 📝 | **WCC 연결성 변화 측정** |
| **F11** | **페르소나 추천 엔진** | `GET /api/recommend/{uuid}` | **3** | 📝 | **그래프 기반 협업 필터링** |
| **F12** | **대화형 탐색 챗봇** | `POST /api/chat` | **3** | 📝 | **멀티턴 RAG, 필터 누적** |
| **F13** | **대규모 운영검증** | 운영 runbook / smoke / benchmark | **4** | 📝 | **1M 성능, 배치 안정성, QA 게이트** |
| **F14** | **확장 및 고도화 로드맵** | TBD | **4** | 📝 | **챗봇 orchestration, 고급 추천, 관측성** |
| **F15** | **챗봇 LLM 응답 합성** | `POST /api/chat` (확장) | **4** | 📝 | **Rule 우선 + LLM 합성, InsightRouter 흡수** |

---

## 3. Feature 10: 네트워크 영향력 및 핵심 인물 분석

> **검수 결과 반영**: Oracle ADR-001 (GDS 사전 계산), Metis (Betweenness 위험)

### 3.1 목표 및 성공 지표
- **목표**: GDS 알고리즘으로 네트워크 중심성을 측정하고, 핵심 인물 및 브릿지 노드를 식별
- **성공 지표**:
  - `/api/influence/top` 응답 시간 < 100ms (사전 계산 + 인덱스 기반)
  - PageRank/Degree 배치 계산 완료 시간 < 30분 (1M 노드 기준)
  - Betweenness는 샘플링 기법(RA-Brandes) 적용, 완료 시간 < 2시간

### 3.2 아키텍처 결정 (ADR-001 반영)
⚠️ **절대로 API 요청에서 GDS `.stream`을 실시간 실행하지 않음**

```
[배치 작업 (주기적)]          [API (실시간)]
gds.pageRank.write      →     Person.pagerank 속성 저장
  (매일 새벽 2시)              →     CREATE INDEX → ORDER BY LIMIT 10

gds.betweenness.sample  →     Person.betweenness 속성 저장
  (주 1회 RA-Brandes)          →     CREATE INDEX → ORDER BY LIMIT 10
```

### 3.3 API Contract

#### `GET /api/influence/top`
```
Query Params:
  - metric: enum ["pagerank", "betweenness", "degree"]
  - limit: int (기본 10, 최대 100)
  - community_id: int? (선택적 필터)

Response: 200 OK
{
  "metric": "pagerank",
  "results": [
    {
      "uuid": "abc123",
      "display_name": "김OO",
      "score": 0.0421,
      "rank": 1,
      "community_id": 5
    }
  ]
}
```

#### `POST /api/influence/simulate-removal`
```
Request:
{
  "target_uuids": ["abc123", "def456"],
  "max_depth": 3  // 서브그래프 제한 (Metis 권고)
}

Response: 200 OK
{
  "original_connectivity": 0.89,    // 원본 WCC 비율
  "current_connectivity": 0.73,     // 제거 후 WCC 비율
  "fragmentation_increase": 0.18,   // 증가율
  "affected_communities": [5, 12]   // 영향받은 커뮤니티
}
```

Phase 3 P0/P1에서는 동기 API를 제공하지 않으며, P2에서 아래 둘 중 하나를 최종 선택합니다.

- **기본 선택**: sync-with-hard-bounds. 최대 5개 UUID, 3-hop 서브그래프, 10초 timeout을 초과하면 422를 반환합니다.
- **확장 선택**: async job. `202 Accepted { job_id }` 반환 후 `/api/influence/jobs/{job_id}`에서 polling합니다.

MVP 기준은 sync-with-hard-bounds이며, 1M 전체 그래프 WCC를 API 요청에서 직접 실행하지 않습니다.

### 3.4 Backend Logic

#### 사전 계산 파이프라인

- `src/gds/centrality.py`: GDS 중심성 계산 쿼리와 서비스 로직
- `src/jobs/centrality_batch.py`: 배치 실행 오케스트레이션, 상태 기록, 실패 처리
```python
class CentralityBatchJob:
    def compute_pagerank(self):
        # gds.pageRank.write 호출 → Person.pagerank 저장
        pass
    
    def compute_betweenness(self):
        # gds.betweenness.sample 호출 (RA-Brandes)
        # https://neo4j.com/docs/graph-data-science/current/algorithms/betweenness-centrality/#algorithms-betweenness-centrality-approx
        pass
```

#### 시뮬레이션 로직
- **제한사항**: 최대 5개 노드 동시 제거 (rate limiting)
- **서브그래프**: 대상 노드 주변 3-hop만 추출하여 WCC 계산
- **타임아웃**: 10초 초과 시 "계산 복잡도 초과" 응답
- **연결성 정의**: `largest_component_size / total_subgraph_nodes`를 connectivity로 사용합니다.

### 3.5 Frontend Changes
- **"핵심 인물" 탭 추가** (Streamlit)
  - 중심성 지표 라디오 버튼 (PageRank / Betweenness / Degree)
  - 상위 20인 테이블 + Plotly bar chart
  - 마지막 계산 시간 표시 ("마지막 갱신: 2026-04-28 02:00")
  - 노드 클릭 → 제거 시뮬레이션 팝업 (max 5개 선택)

### 3.6 구현 단계
- **P0**: PageRank/Degree 배치 작업 + `/api/influence/top` 엔드포인트
- **P1**: Betweenness 샘플링 + 커뮤니티 대표자 연동
- **P2**: 노드 제거 시뮬레이션 (서브그래프 제한 + rate limiting)

### 3.7 리스크 및 대응
| 리스크 | 대응 |
|:---|:---|
| Betweenness 1M 노드 계산 시 OOM | RA-Brandes 샘플링 (probability=0.1), Neo4j heap 32GB+ |
| GDS Projection 휘발성 | 배치 작업 시작 전 projection check/recreate 수행. 사용자 API 요청 경로에서는 GDS projection 생성 금지 |
| 시뮬레이션 API 타임아웃 | 서브그래프 3-hop 제한, 비동기 작업 큐 고려 |

---

## 4. Feature 11: 페르소나 기반 추천 엔진

> **검수 결과 반영**: Oracle ADR-002 (템플릿 기반 reasoning), Metis (SIMILAR_TO 의존성)

### 4.1 목표 및 성공 지표
- **목표**: 유사 페르소나가 가진 속성 중 대상에게 없는 항목을 추천
- **성공 지표**:
  - API 응답 시간 < 500ms (템플릿 기반 reasoning)
  - 추천 정확도: 유사 페르소나의 70%+가 해당 속성 보유

### 4.2 아키텍처 결정 (ADR-002 반영)
⚠️ **추천 사유(Reasoning)는 LLM 동기 호출 금지 → 템플릿 기반**

```python
# ❌ 금지: 동기 LLM 호출 (API 2초 SLA 위반)
llm.generate("왜 TypeScript를 추천하는가?")  # 1~3초 소요

# ✅ 허용: 템플릿 기반 (즉시 생성)
f"당신과 유사한 {similar_count}명 중 {ratio:.0%}가 '{item_name}'을 가지고 있습니다."
```

### 4.3 선행 조건 (Metis 의존성 분석 반영)
> **주의**: Feature 11은 Feature 2(KNN)의 `SIMILAR_TO` 관계가 존재할 때만 동작
- KNN 미실행 상태 → `503 Service Unavailable` 응답
- 또는 fallback: 벡터 검색으로 즉시 유사 페르소나 찾기 (느리지만 동작)

### 4.4 API Contract

#### `GET /api/recommend/{uuid}`
```
Query Params:
  - category: enum ["hobby", "skill", "occupation", "district"]
  - top_n: int (기본 5, 최대 20)

Response: 200 OK
{
  "uuid": "abc123",
  "category": "hobby",
  "recommendations": [
    {
      "item_name": "클라이밍",
      "reason": "당신과 유사한 128명 중 73%가 이 취미를 가지고 있습니다.",
      "reason_score": 0.73,
      "similar_users_count": 128
    }
  ]
}

// SIMILAR_TO 없을 때
Response: 503 Service Unavailable
{
  "error": "유사도 매칭 데이터가 없습니다. 관리자에게 KNN 파이프라인 실행을 요청하세요."
}
```

### 4.5 Backend Logic

#### 추천 알고리즘 (`src/graph/recommendation.py`)
```python
def recommend(uuid: str, category: str, top_n: int = 5):
    # 1. 유사 페르소나 집합 S 추출 (SIMILAR_TO 사용)
    similar = get_similar_personas(uuid, top_k=50)
    
    # 2. S의 속성 집계 (대상에게 없는 것만)
    target_traits = get_person_traits(uuid, category)
    candidate_traits = aggregate_traits(similar, category)
    candidates = [t for t in candidate_traits if t not in target_traits]
    
    # 3. 스코어링: 빈도 + 유사도 가중치
    ranked = score_by_frequency_and_similarity(candidates, similar)
    
    # 4. 템플릿 기반 reasoning 생성 (LLM 호출 없음)
    return format_recommendations(ranked, top_n)
```

### 4.6 Frontend Changes
- **프로필 상세 페이지에 "추천" 섹션 추가**
  - 카테고리 탭 (취미 / 기술 / 직업 / 지역)
  - 카드형 UI: `st.columns` 활용
  - 각 카드: 아이템명 + 사유 + 비율 + 유사 인물 수
  - 카드 클릭 → 해당 속성 보유 유사 페르소나 목록

### 4.7 구현 단계
- **P0**: 취미/기술 추천 (빈도 기반) + 템플릿 reasoning
- **P1**: FastRP 유사도 가중치 스코어링
- **P2**: 지역/직업 추천 확장

---

## 5. Feature 12: 대화형 탐색 챗봇

> **검수 결과 반영**: Oracle (히스토리 3-5턴 제한), Metis (필터 누적 위험)

### 5.1 목표 및 성공 지표
- **목표**: 단발 질의를 넘어 대화 맥락을 유지하며 데이터 탐색
- **성공 지표**:
  - 3턴 연속 필터 유지 성공률 > 90%
  - API 응답 시간 < 3초
  - 컨텍스트 윈도우 초과 0건 (요약 노드 적용)

### 5.2 아키텍처 결정

#### 히스토리 제한 (Oracle 권고)
```python
# LangGraph state에서 최근 3-5턴만 유지
history = state["history"][-5:]  # 최근 5턴만
```

#### 필터 상태 스키마 (Metis 권고)
```python
class FilterState:
    age_group: str | None      # "20대"
    sex: str | None            # "남자"
    province: str | None       # "서울"
    hobby: str | None          # "등산"
    # ... 명확한 스키마로 충돌 방지
```

#### 리셋 메커니즘 (Metis 권고)
- 사용자가 "처음부터" 또는 "리셋"이라고 하면 필터 초기화
- UI에 "현재 필터 초기화" 버튼 노출

### 5.3 API Contract

#### `POST /api/chat`
```
Request:
{
  "session_id": "user-abc-123",
  "message": "그중에서 개발자만 보여줘",
  "stream": false
}

Response: 200 OK
{
  "response": "서울에 사는 20대 남성 개발자 중 상위 5명입니다...",
  "context_filters": {
    "province": "서울",
    "age_group": "20대",
    "sex": "남자",
    "occupation": "개발자"
  },
  "sources": [
    {"type": "cypher", "query": "MATCH ..."},
    {"type": "vector", "uuids": ["u1", "u2"]}
  ],
  "turn_count": 3
}
```

### 5.4 Backend Logic

#### LangGraph 상태 관리 (`src/rag/chat_graph.py`)
```python
class ChatState(TypedDict):
    history: list[dict]          # 최근 5턴 (자동 잘림)
    current_filters: FilterState  # 현재 누적 필터
    last_intent: str | None      # 마지막 의도 (검색/비교/통계)

def intent_classifier(state: ChatState) -> ChatState:
    """의도 분류: 검색 / 비교 / 통계 / 일반대화 / 리셋"""
    pass

def filter_merge(state: ChatState) -> ChatState:
    """새 필터를 기존 필터와 병합 (AND 조건)"""
    # "그중에서" → 기존 필터 유지 + 새 조건 추가
    # "대신" → 해당 필터만 교체
    pass

def summarize_context(state: ChatState) -> ChatState:
    """5턴 초과 시 요약 노드 실행"""
    pass
```

### 5.5 Frontend Changes
- **메인 화면 Chat Interface**
  - `st.chat_message` + `st.chat_input`
  - 사이드바: 현재 필터 칩 형태 표시
  - "필터 초기화" 버튼
  - 이전 대화 기록 (최근 10턴만 표시)

### 5.6 구현 단계
- **P0**: 기존 `/api/insight` + LangChain Memory 연결 (단순 대화)
- **P1**: 필터 병합 로직 + 명확한 FilterState 스키마
- **P2**: 차트/그래프 채팅창 내 렌더링

---

## 6. 교차 기능 통합

### 6.1 기능 간 의존성 그래프
```
Feature 10 (영향력)
    ↓ (중심성 스코어를 가중치로)
Feature 11 (추천)
    ↓ ("이 사람에게 추천은?" 질문)
Feature 12 (챗봇)
    ↓ (필터 결과)
Feature 5 (검색)
```

### 6.2 통합 시나리오
| 시나리오 | 흐름 |
|:---|:---|
| 챗봇 → 추천 | "나에게 추천할 취미는?" → `/api/recommend/{uuid}` 호출 → 답변 |
| 챗봇 → 영향력 | "내가 속한 커뮤니티에서 누가 중심이야?" → `/api/influence/top?community_id=X` |
| 영향력 → 추천 | 영향력 높은 인물의 특성을 추천 가중치에 반영 (P1 이후) |

---

## 7. 성능 및 인프라

### 7.1 Neo4j 설정 권장사항 (Oracle 권고)
```properties
# neo4j.conf
server.memory.heap.max_size=32G
server.memory.pagecache.size=16G
```

### 7.2 인덱스 계획
```cypher
-- 기존 인덱스 외 추가
CREATE INDEX person_pagerank FOR (p:Person) ON (p.pagerank);
CREATE INDEX person_betweenness FOR (p:Person) ON (p.betweenness);
CREATE INDEX person_degree FOR (p:Person) ON (p.degree);
CREATE INDEX person_community_id FOR (p:Person) ON (p.community_id);
CREATE INDEX person_age_group FOR (p:Person) ON (p.age_group);
CREATE INDEX person_sex FOR (p:Person) ON (p.sex);
CREATE INDEX district_key FOR (d:District) ON (d.key);
```

주의: `province`는 `Person` 속성이 아니라 `(:District)-[:IN_PROVINCE]->(:Province)` 계층으로 저장됩니다. 지역 필터/추천은 `District.key` 또는 `Province.name`을 명시적으로 사용해야 하며, `p.province` 기반 composite index는 사용하지 않습니다.

### 7.3 배치 작업 스케줄
| 작업 | 주기 | 파일 | 설명 |
|:---|:---|:---|:---|
| PageRank 계산 | 매일 02:00 | `src/jobs/centrality_batch.py` | gds.pageRank.write |
| Degree 계산 | 매일 02:00 | `src/jobs/centrality_batch.py` | gds.degree.write |
| Betweenness 계산 | 주 1회 일요일 02:00 | `src/jobs/centrality_batch.py` | gds.betweenness.sample |
| KNN 재계산 | 주 1회 월요일 02:00 | `src/jobs/knn_refresh.py` | SIMILAR_TO 갱신 |

운영 원칙:
- 배치/재계산 작업은 FastAPI request-response 경로에서 실행하지 않습니다.
- Streamlit UI는 GDS 재생성/배치 실행을 직접 트리거하지 않습니다.
- 스케줄러는 앱 프로세스와 분리된 외부 프로세스(Windows Task Scheduler, cron, 또는 독립 실행 스크립트)를 우선합니다.
- UI는 상태 조회와 polling만 수행합니다.

---

## 8. 테스트 전략

### 8.1 자동화 테스트

| 기능 | 테스트 파일 | 시나리오 |
|:---|:---|:---|
| F10 영향력 | `tests/test_api_influence.py` | `/api/influence/top` 응답 < 100ms, metric=pagerank 시 10개 반환 |
| F10 시뮬레이션 | `tests/test_api_influence.py` | 제거 후 connectivity < original |
| F11 추천 | `tests/test_recommendation.py` | 추천 항목이 대상의 기존 속성과 중복되지 않음 |
| F11 응답시간 | `tests/test_recommendation.py` | API 응답 < 500ms |
| F12 챗봇 | `tests/test_chat_graph.py` | 3턴 연속 필터 누적 검증 |
| F12 리셋 | `tests/test_chat_graph.py` | "리셋" 발화 시 필터 초기화 |
| F15 LLM 합성 | `tests/test_chat_graph.py` | stats/search → LLM 합성 호출, LLM 실패 시 템플릿 fallback |
| F15 general fallback | `tests/test_chat_graph.py` | general intent → InsightRouter 호출, 실패 시 fallback |

### 8.2 수동 평가 (Golden Set)
- F12 대화 품질: 50개 시나리오 세트, 90% 이상 정확 필터 유지
- F15 LLM 합성 품질: golden set 10개 이상, 데이터 기반 답변 여부 검증

---

## 9. 리스크 레지스터

| ID | 리스크 | 영향 | 대응 | 담당 |
|:---|:---|:---:|:---|:---|
| R1 | Betweenness 1M 노드 OOM | 🔴 높음 | RA-Brandes 샘플링 + 32GB heap | 백엔드 |
| R2 | GDS Projection 휘발성 | 🟡 중간 | 배치 시작 전 projection check/recreate, 사용자 API에서는 503만 반환 | 백엔드 |
| R3 | SIMILAR_TO 없을 시 F11 실패 | 🟡 중간 | 503 응답 + 선택적 벡터 fallback | 백엔드 |
| R4 | 챗봇 필터 누적 오류 | 🟡 중간 | 명확한 FilterState 스키마 + 리셋 버튼 | 백엔드/프론트 |
| R5 | LLM context window 초과 | 🟢 낮음 | 5턴 자동 잘림 + 요약 노드 | 백엔드 |
| R6 | LLM이 제공 데이터 무시하고 사전학습 지식으로 답변 | 🔴 높음 | 시스템 프롬프트 강화, 응답에 수치 포함 여부 검증 | 백엔드 |
| R7 | Neo4j 결과 대량 시 토큰 비용/지연 급증 | 🟡 중간 | 결과 상위 N건 제한, 집계 요약 후 전달 | 백엔드 |
| R8 | LLM API 장애 시 챗봇 응답 불가 | 🟡 중간 | LLM 실패 시 기존 템플릿 문자열로 graceful fallback | 백엔드 |

---

## 10. 운영, 오류 처리, 롤백 기준

### 10.1 공통 API 오류 계약

신규 Phase 3 API는 기존 FastAPI 예외 처리 패턴을 유지하되, 아래 오류를 명시적으로 반환해야 합니다.

| 상황 | HTTP Status | 응답 예시 | 적용 기능 |
|:---|:---:|:---|:---|
| 존재하지 않는 UUID | 404 | `{ "error": "Persona not found" }` | F10/F11/F12 |
| 잘못된 enum 파라미터 | 400 | `{ "error": "Invalid metric/category" }` | F10/F11 |
| GDS projection 없음 또는 생성 실패 | 503 | `{ "error": "Graph projection is not ready" }` | F10/F11 |
| 중심성 배치 미실행/score 없음 | 503 | `{ "error": "Centrality scores are not available yet" }` | F10 |
| SIMILAR_TO 관계 없음 | 503 | `{ "error": "Similarity data is not available" }` | F11 |
| 시뮬레이션 계산 제한 초과 | 422 | `{ "error": "Simulation scope is too large" }` | F10 |
| 챗봇 세션 만료/손상 | 400 | `{ "error": "Invalid or expired chat session" }` | F12 |

### 10.2 관측성 및 운영 상태

Phase 3 기능은 사용자-facing API와 별도로 운영 상태를 확인할 수 있어야 합니다.

| 상태 항목 | 저장/노출 방식 | 목적 |
|:---|:---|:---|
| 중심성 배치 마지막 성공 시각 | `CentralityJobStatus` 노드 또는 설정 테이블 | UI의 `마지막 갱신` 표시 |
| 중심성 배치 실패 사유 | 로그 + 상태 노드 | 운영자가 재실행 판단 |
| GDS projection 상태 | API 호출 전 check 함수 | Neo4j 재시작 후 자동 복구 |
| KNN/SIMILAR_TO 갱신 시각 | 상태 노드 또는 메타 파일 | 추천 결과 stale 여부 판단 |
| 챗봇 필터 상태 | 응답의 `context_filters` | 사용자에게 현재 조건 투명화 |

권장 상태 노드 예시:

```cypher
MERGE (s:SystemStatus {key: 'centrality_batch'})
SET s.last_success_at = datetime(),
    s.status = 'success',
    s.run_id = 'centrality-20260428-020000',
    s.metrics = ['pagerank', 'degree']
```

#### 배치 결과 공개 규칙

- 배치 작업은 `run_id` 단위로 실행 상태를 기록합니다.
- 배치가 실패하면 기존 성공 결과를 유지하고 `SystemStatus.status = 'failed'`와 실패 사유를 기록합니다.
- API는 `SystemStatus.status = 'success'`인 마지막 `run_id`의 결과만 사용자에게 노출합니다.
- 부분 write 상태를 피하기 위해 구현 시 임시 속성(`pagerank_next`, `degree_next`)에 먼저 기록한 뒤, 성공 후 공개 속성(`pagerank`, `degree`)으로 승격하는 방식을 우선 검토합니다.

### 10.3 데이터 신선도 및 stale data 정책

- `/api/influence/top`은 사전 계산된 값을 반환하므로 응답에 `last_updated_at`을 포함합니다.
- `last_updated_at`이 7일 이상 오래된 경우 UI에 경고 배지를 표시합니다.
- `/api/recommend/{uuid}`는 `SIMILAR_TO` 갱신 시각을 내부적으로 확인합니다.
- KNN 갱신 이후 새로 추가된 UUID가 추천 대상이면 503 또는 벡터 fallback 중 하나를 명확히 선택합니다. Phase 3 P0 기본값은 **503 명시 오류**입니다.

### 10.4 마이그레이션 및 롤백

Phase 3 도입 시 추가되는 속성/인덱스는 기존 기능을 깨지 않도록 선택적으로 적용합니다.

#### 추가 속성
- `Person.pagerank`
- `Person.betweenness`
- `Person.degree`
- `Person.centrality_updated_at`

#### 추가 인덱스
```cypher
CREATE INDEX person_pagerank IF NOT EXISTS FOR (p:Person) ON (p.pagerank);
CREATE INDEX person_betweenness IF NOT EXISTS FOR (p:Person) ON (p.betweenness);
CREATE INDEX person_degree IF NOT EXISTS FOR (p:Person) ON (p.degree);
```

#### 롤백 기준
- 신규 API 라우터를 `src/api/main.py`에서 언마운트하면 Phase 1~2 기능은 계속 동작해야 합니다.
- 중심성 속성은 제거하지 않아도 기존 기능에 영향이 없어야 합니다.
- 배치 작업 실패 시 마지막 성공 값을 유지하고, API 응답에 stale 경고를 포함합니다.

### 10.5 UX 상태 기준

| 기능 | Empty State | Loading State | Error State |
|:---|:---|:---|:---|
| F10 영향력 | "아직 계산된 중심성 점수가 없습니다." | "중심성 점수를 불러오는 중입니다." | "중심성 배치가 아직 준비되지 않았습니다." |
| F10 시뮬레이션 | "선택된 노드가 없습니다." | "서브그래프 영향을 계산 중입니다." | "선택 범위가 커서 계산할 수 없습니다." |
| F11 추천 | "추천할 새 항목이 없습니다." | "유사 페르소나 기반 추천을 계산 중입니다." | "유사도 데이터가 없어 추천할 수 없습니다." |
| F12 챗봇 | "질문을 입력해 데이터를 탐색해 보세요." | "답변을 생성하는 중입니다." | "대화 상태를 처리하지 못했습니다. 필터를 초기화해 주세요." |

추가 UX 규칙:
- F10 시뮬레이션은 세션당 1개 작업만 진행 중일 수 있습니다.
- F10 시뮬레이션 선택 노드는 취소/초기화 버튼으로 모두 해제할 수 있어야 합니다.
- Streamlit rerun 이후에도 선택된 중심성 metric, 선택 노드, 챗봇 필터 chip은 `st.session_state`로 유지합니다.
- 사용자-facing 문구는 한국어를 기본으로 하며, `Last updated` 대신 `마지막 갱신`을 사용합니다.

---

## 11. Phase 4: 대규모 운영검증 및 확장/고도화 계획

> **원칙**: 이 Phase는 코드 구현 전 **검수 먼저** 진행합니다. 운영 성능, 장애 대응, 사용자 흐름, 확장 설계를 문서와 체크리스트로 승인한 뒤에만 구현 작업을 시작합니다.

### 11.1 목표

- 1M 전체 데이터 기준으로 Phase 3 기능의 운영 가능성을 검증합니다.
- 배치 작업, API SLA, Streamlit UX, 장애/롤백 기준을 실제 운영 시나리오로 점검합니다.
- F12 챗봇을 검색/통계 MVP에서 추천·영향력·프로필 orchestration으로 확장할 계획을 수립합니다.
- 구현 전 검수 산출물(PRD, 체크리스트, runbook, QA 시나리오, Go/No-Go 기준)을 확정합니다.

### 11.2 F13: 대규모 운영검증 요구사항

| 영역 | 검증 항목 | 성공 기준 | 산출물 |
|:---|:---|:---|:---|
| 데이터 규모 | 1M Person, SIMILAR_TO, 중심성 속성 존재 여부 | 누락/중복/관계 수 기준 통과 | 데이터 검증 리포트 |
| 중심성 배치 | PageRank/Degree 전체 실행 | 1M 기준 < 30분 또는 병목 원인 기록 | benchmark log |
| Betweenness | RA-Brandes sampling 실행 가능성 | 1M 기준 < 2시간 목표, 실패 시 sampling/heap 조정안 | benchmark log + 튜닝안 |
| API SLA | `/api/influence/top`, `/api/recommend/{uuid}`, `/api/chat` | 각각 < 100ms, < 500ms, < 3초 목표 | smoke 결과표 |
| 스케줄러 | Windows Task Scheduler/cron 운영 | 앱 프로세스와 분리, 실패 재시도/로그 확인 | 운영 runbook |
| 장애 대응 | GDS/KNN/중심성 미준비 상태 | 사용자 API에서 전체 재계산 금지, 503/stale 경고 | 장애 시나리오표 |
| UI QA | Streamlit F10/F11/F12 흐름 | empty/loading/error state 한국어 표시 | QA 체크리스트 |

### 11.3 운영검증 절차

1. **검수 계획 승인**: 본 PRD와 `CHECKLIST.md` Phase 19 항목을 먼저 리뷰합니다.
2. **환경 고정**: Neo4j heap/pagecache, GDS 버전, 데이터 크기, GPU/CPU 환경을 기록합니다.
3. **데이터 준비 검증**: Person/관계/SIMILAR_TO/중심성 속성의 기준 수량을 확인합니다.
4. **배치 벤치마크**: PageRank, Degree, Betweenness를 독립 실행하고 run_id별 결과를 보존합니다.
5. **API/UI smoke**: 신규 API와 Streamlit 주요 흐름을 전체 데이터 기준으로 확인합니다.
6. **장애 주입 검수**: projection 없음, SIMILAR_TO 없음, stale batch, timeout 상황의 응답을 확인합니다.
7. **Go/No-Go 판정**: 모든 P0 기준 통과 시 구현/자동화 단계로 넘어갑니다.

### 11.4 F14: 확장 및 고도화 로드맵

| 우선순위 | 고도화 항목 | 설명 | 선행 조건 |
|:---:|:---|:---|:---|
| P1 | 통합 챗봇 UX | 대화형 탐색을 메인 챗봇으로 유지하고, 인사이트 질의를 챗봇 내부 `고급 분석 모드`로 흡수 | F12 ChatGPT형 UI 안정화, F1 InsightRouter 호환성 |
| P1 | 챗봇 → 추천 orchestration | “이 사람에게 추천할 활동은?” 질문에서 `/api/recommend/{uuid}` 호출 | F12 session 안정화, UUID 선택 UX |
| P1 | 챗봇 → 영향력 orchestration | “이 커뮤니티 핵심 인물은?” 질문에서 `/api/influence/top` 호출 | 중심성 batch stale 정책 |
| P1 | 프로필 intent | UUID/선택 인물을 기준으로 프로필 요약 응답 | F7 profile API 안정성 |
| P1 | 운영 상태 API/화면 | batch run_id, last_success_at, stale 여부를 UI에서 확인 | SystemStatus 저장 정책 |
| P2 | LLM 기반 필터 추출 | regex를 Pydantic structured output으로 보강 | golden set, hallucination guardrail |
| P2 | 고급 추천 스코어링 | 중심성/커뮤니티/유사도 가중치 혼합 | F10/F11 교차 검증 |
| P2 | 비동기 장기 작업 | removal simulation/Betweenness를 job queue로 분리 | job status API 설계 |
| P2 | 관측성 강화 | metrics, structured logs, alerting | 운영 로그 표준 |

### 11.5 구현 전 검수 게이트

Phase 4의 코드는 아래 산출물이 승인되기 전 작성하지 않습니다.

- PRD Phase 4 요구사항 검수 완료
- CHECKLIST Phase 19/20 항목 검수 완료
- 대규모 운영검증 실행 계획 승인
- 성능 목표와 실패 시 fallback 기준 합의
- UI/UX empty/loading/error 문구 승인
- 구현 범위(P1/P2)와 제외 범위 명시

### 11.6 통합 챗봇 UX 전환 계획

현재 `대화형 탐색`과 `인사이트 질의`는 모두 자연어 입력을 받기 때문에 사용자에게 같은 기능처럼 보일 수 있습니다. Phase 4 P1에서는 아래 원칙으로 UX를 정리합니다.

#### 목표 구조

- **기본 모드: 탐색 챗봇**
  - `POST /api/chat` 기반 멀티턴 대화, 필터 누적, 검색/통계 응답을 담당합니다.
  - 사용자는 ChatGPT형 단일 대화창에서 질문을 입력합니다.
- **고급 분석 모드: 인사이트 질의 흡수**
  - 기존 `POST /api/insight`의 Cypher/Vector/Composite 분석 기능을 챗봇 내부 옵션으로 노출합니다.
  - UI에서는 별도 탭보다 `고급 분석` 토글/버튼/명령어로 접근합니다.
- **호환성 유지**
  - `/api/insight`는 기존 API 사용자를 위해 유지하되, Streamlit 주 UX에서는 챗봇 내부 고급 모드로 이동합니다.
  - `/api/chat`은 향후 `mode: "explore" | "analysis"` 또는 intent 기반 내부 라우팅을 지원하도록 확장합니다.

#### 사용자 경험 기준

| 사용자 의도 | 기본 처리 | 고급 분석 전환 기준 |
|:---|:---|:---|
| “서울 20대 여성의 취미는?” | 탐색 챗봇 통계 응답 | 전환 없음 |
| “그중에서 개발자만” | 현재 필터에 조건 누적 | 전환 없음 |
| “왜 이런 차이가 나?” | 고급 분석 모드 제안 또는 자동 전환 | Cypher+Vector+LLM 분석 |
| “서울과 부산의 취미 차이를 분석해줘” | 고급 분석 모드 | `/api/insight` 또는 통합 router |
| “근거를 자세히 보여줘” | 답변 아래 접힌 sources 표시 | 필요 시 고급 분석 |

#### 구현 전 결정 필요 항목

- UI 전환 방식: 토글, 버튼, slash command(`/분석`), 자동 intent 중 하나를 선택합니다.
- API 전환 방식: `/api/chat`에 mode 필드를 추가할지, 내부에서 `/api/insight`를 호출할지 결정합니다.
- 기존 `💡 인사이트 질의` 탭은 즉시 제거하지 않고, P1 검수 후 `고급 분석 모드`로 이동하거나 deprecated 안내를 표시합니다.
- 분석 모드 응답은 기존 `sources`를 유지하되, 챗봇 대화 history에 요약 형태로 저장합니다.

### 11.7 F15: 챗봇 LLM 응답 합성 (Rule 우선 + LLM 합성)

> **핵심 원칙**: 기존 ChatGraph의 rule-based 분류/필터 추출은 그대로 유지하고, 응답 생성 단계에만 LLM 합성 레이어를 추가한다. InsightRouter는 general intent의 fallback 경로로 흡수한다.

#### 11.7.1 현황 및 문제

현재 `POST /api/chat`(ChatGraph)는 Neo4j 쿼리 결과를 하드코딩된 템플릿 문자열로 반환합니다.

```
현재 응답: "제가 이해한 조건은 지역=부산, 연령대=20대, 성별=여자이고, 요청하신 항목은 취미입니다.
            취미 상위 분포는 다음과 같습니다. 1. 독서 (38명)"
```

- 사용자 질문의 맥락("퇴근후", "여가시간에" 등)을 응답에 반영하지 못합니다.
- 데이터 해석이나 경향 설명 없이 raw count 테이블만 제공합니다.
- 반면 `POST /api/insight`(InsightRouter)는 LLM 합성이 있지만 세션/멀티턴이 없습니다.

#### 11.7.2 목표 구조

```
질문 → 기존 규칙 분류/필터 추출 (즉시, 변경 없음)
        │
        ├─ search/stats/reset → Neo4j 직접 실행 → LLM 합성 (자연어 답변)
        │                        (기존 코드 유지)    (1회 LLM 호출 추가)
        │
        └─ general → InsightRouter fallback (기존 Cypher/Vector RAG)
```

#### 11.7.3 아키텍처 결정

| 결정 사항 | 선택 | 근거 |
|:---|:---|:---|
| 의도 분류 방식 | Rule-based 유지 | 80%+ 정확도, 즉시 응답, 결정론적 |
| 필터 추출 방식 | Rule-based 유지 | LLM 추출은 hallucination 위험, 비용/지연 증가 |
| 응답 생성 방식 | **LLM 합성 추가** | 유일한 갭. 템플릿 → 자연어 전환 |
| LLM 합성 위치 | **별도 `synthesize` LangGraph 노드** (Oracle 권고) | `_run_stats`/`_run_search` 내부가 아닌 `respond` 뒤 독립 노드. Neo4j 드라이버 점유 방지, 디버깅/재시도 용이 |
| LLM 호출 횟수 | 턴당 1회 (합성만) | Cypher 생성까지 LLM이면 턴당 2회, 비용/지연 2배 |
| general intent 처리 | InsightRouter fallback | Rule이 못 잡는 ~20% 질문 커버 |
| InsightRouter lifecycle | **module-level singleton 공유** (Oracle 권고) | InsightRouter 초기화 비용 높음 (Neo4jGraph + LLM 클라이언트), 매 요청 생성 금지 |
| InsightRouter 유지 | `/api/insight` 엔드포인트 호환성 유지 | 기존 API 사용자, 단발 질의 용도 |
| general intent 필터 전달 | **현재 필터를 context prefix로 포함** (Oracle 권고) | InsightRouter는 세션 개념이 없으므로, 필터 없이 보내면 맥락 무시 |

> **LangGraph 플로우 변경**:
> ```
> 기존: classify → merge_filters → respond → commit_history → END
> 변경: classify → merge_filters → respond → synthesize → commit_history → END
> ```
> `respond` 노드는 Neo4j 쿼리만 실행하고 raw 결과를 `ChatState`에 저장.
> `synthesize` 노드는 raw 결과 + 사용자 질문을 LLM에 전달하여 자연어 답변 생성.
> Neo4j 드라이버는 `respond`에서 닫히고, LLM 호출은 `synthesize`에서 별도 실행.

#### 11.7.4 구현 범위

**1단계 (P0): LLM 합성 레이어 추가**

- `src/rag/chat_graph.py`에 `synthesize` LangGraph 노드 추가
- LangGraph 플로우: `classify → merge_filters → respond → synthesize → commit_history → END`
- `respond` 노드는 Neo4j 쿼리만 실행하고 raw 결과를 `ChatState`에 저장 (드라이버 즉시 반환)
- `synthesize` 노드는 raw 결과 + 사용자 질문 + 현재 필터를 LLM에 전달하여 자연어 답변 생성
- `_synthesize_response()` 헬퍼 함수: LLM 프롬프트 구성 + 호출 + 결과 반환
- LLM 실패 시 기존 템플릿 문자열로 graceful fallback (raw 결과를 그대로 사용)
- Neo4j 결과가 대량일 경우 상위 20건만 LLM에 전달 (컨텍스트 윈도우 비대화 방지)
- Neo4j 결과 포맷: compact JSON-lines (토큰 효율, Oracle 권고)
- `src/rag/llm.py`의 `create_llm()` 재사용, `thinking: False` 유지 확인

**2단계 (P1): InsightRouter fallback 연결**

- InsightRouter를 module-level singleton으로 공유 (Oracle 권고, 매 요청 생성 금지)
- `classify_intent`에서 `general`로 분류된 질문을 InsightRouter로 라우팅
- 현재 필터를 context prefix로 포함하여 InsightRouter에 전달 (Oracle 권고, 맥락 유지)
- InsightRouter 결과를 ChatGraph 세션 history에 저장
- InsightRouter 실패 시 기존 `_general_response()` 문자열 fallback
- 단일 `/api/chat` 엔드포인트에서 모든 질문을 처리

**제외 범위**

- LLM 기반 필터 추출 (P2 이후 검토)
- 스트리밍 응답 (프론트엔드 Next.js 전환 후 검토)
- `/api/insight` 엔드포인트 삭제 (호환성 유지)

#### 11.7.5 LLM 프롬프트 설계 원칙

```
시스템: 당신은 한국인 페르소나 데이터를 분석하는 어시스턴트입니다.
       아래 제공된 데이터만 근거로 답변하세요.
       데이터에 없는 내용은 추측하지 마세요.
       수치를 인용할 때는 정확한 값을 사용하세요.
       한국어로 자연스럽게 답변하세요.

사용자 질문: {user_message}
현재 필터: {filter_summary}
조회 결과:
{neo4j_results_json}
```

#### 11.7.6 성능 목표

| 항목 | 기준 |
|:---|:---|
| search/stats 응답 | < 5초 (Neo4j < 1초 + LLM 합성 < 4초) |
| general fallback 응답 | < 10초 (InsightRouter 경유) |
| LLM 합성 입력 크기 | Neo4j 결과 상위 20건 이하 |
| LLM hallucination 방지 | 시스템 프롬프트 + 데이터 전용 답변 강제 |

#### 11.7.7 리스크

| ID | 리스크 | 영향 | 대응 |
|:---|:---|:---:|:---|
| R6 | LLM이 제공 데이터 무시하고 사전학습 지식으로 답변 | 🔴 높음 | 시스템 프롬프트 강화, 응답에 수치 포함 여부 검증 |
| R7 | Neo4j 결과 대량 시 토큰 비용/지연 급증 | 🟡 중간 | 결과 상위 N건 제한, 집계 요약 후 전달 |
| R8 | LLM API 장애 시 응답 불가 | 🟡 중간 | LLM 실패 시 기존 템플릿 문자열로 graceful fallback |

---

## 12. 아키텍처 결정 기록 (ADRs)

본 PRD의 중요한 기술적 결정은 별도 문서로 관리됩니다:

| ADR | 제목 | 파일 |
|:---|:---|:---|
| ADR-001 | GDS 중심성 계산은 배치로 사전 계산 | `docs/decisions/ADR-001-gds-precompute.md` |
| ADR-002 | 추천 Reasoning은 템플릿 기반, LLM 동기 호출 금지 | `docs/decisions/ADR-002-recommendation-reasoning.md` |
| ADR-003 | 챗봇 히스토리는 최근 5턴만 유지 | `docs/decisions/ADR-003-chatbot-memory.md` |
| ADR-004 | 챗봇 응답은 Rule 우선 + LLM 합성, 필터 추출에 LLM 사용 금지 | PRD §11.7.3 (인라인) |

---

## 13. 문서 관리 규칙

### 13.1 파일 구조
```
PRD.md                    ← 본 문서 (현재 상태, 단일 진실 공급원)
CHANGELOG.md              ← 변경 이력
CHECKLIST.md              ← 구현 진행 상황
docs/
  prd-archive/            ← 과거 PRD 버전
    prd-v1.0-phase1.md
    prd-v1.5-phase2.md
    prd-v2.0-expansion.md
  decisions/              ← 아키텍처 결정 기록
    ADR-001-gds-precompute.md
    ADR-002-recommendation-reasoning.md
    ADR-003-chatbot-memory.md
```

### 13.2 변경 프로세스
1. PRD 수정 필요 시 **본 파일(PRD.md)만** 수정
2. 변경 사항은 `CHANGELOG.md`에 기록
3. 중대한 아키텍처 결정은 `docs/decisions/ADR-###` 신규 작성
4. Phase 완료 시 해당 Phase의 아카이브 파일은 **수정하지 않음** (읽기 전용)

---

**작성일**: 2026-04-28  
**버전**: v2.3-f15-llm-synthesis  
**상태**: Phase 3 핵심 구현 완료 / Phase 4 계획 검수 대기 / F15 챗봇 LLM 합성 계획 추가  
**다음 단계**: F15 구현 계획 검수 → 1단계(LLM 합성) 구현 → 2단계(InsightRouter 흡수) 구현
