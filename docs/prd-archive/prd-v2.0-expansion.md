# PRD: 한국인 페르소나 지식 그래프 고도화 (Korean Persona Knowledge Graph Expansion)

본 문서는 NVIDIA Nemotron-Personas-Korea 데이터셋 기반의 지식 그래프 시스템에 세 가지 핵심 기능을 추가하기 위한 제품 요구사항 정의서(PRD)입니다.

---

## 1. 개요 (Vision)
본 프로젝트의 목표는 단순히 페르소나 데이터를 조회하는 수준을 넘어, **네트워크 과학 기반의 영향력 분석**, **데이터 기반의 맞춤형 추천**, 그리고 **자연스러운 대화형 탐색**을 통해 데이터의 가치를 극대화하는 것입니다.

---

## 2. Feature 1: 네트워크 영향력 및 핵심 인물 분석 (Key Person Analysis)

### 1. 목표 및 성공 지표
*   **목표:** GDS 알고리즘을 활용하여 커뮤니티 내 가교 역할을 하는 인물과 중심성이 높은 인물을 식별합니다.
*   **성공 지표:**
    *   중심성 지표(PageRank, Betweenness) 계산 시간 5초 이내 완료 (1M 노드 기준).
    *   커뮤니티 요약 정보 내 '대표 페르소나' 자동 할당 정확도 향상.

### 2. 사용자 스토리
*   "분석가로서 특정 지역 사회에서 정보 전달의 핵심이 되는 인물을 찾아 영향력 전파 경로를 파악하고 싶다."
*   "기획자로서 특정 인물이 사라졌을 때 네트워크가 얼마나 파편화되는지 시뮬레이션하여 조직의 복원력을 측정하고 싶다."
*   "사용자로서 내가 속한 그룹에서 가장 전형적이거나 가장 독특한 인물이 누구인지 순위표로 확인하고 싶다."

### 3. API Contract
*   **GET `/api/influence/top`**
    *   Query Params: `metric` (pagerank, betweenness, degree), `limit` (int)
    *   Response: `List[{uuid, display_name, score, rank, community_id}]`
*   **POST `/api/influence/simulate-removal`**
    *   Request: `{target_uuids: List[str]}`
    *   Response: `{original_connectivity: float, current_connectivity: float, fragmentation_increase: float}`

### 4. Backend Logic
*   **Algorithms:**
    *   **PageRank:** 전체적인 인기도 및 권위 측정.
    *   **Betweenness Centrality:** 서로 다른 커뮤니티를 잇는 '브릿지' 역할 식별.
    *   **Degree Centrality:** 단순 연결성 확인.
*   **Cypher Snippet:**
    ```cypher
    // GDS Betweenness Centrality 예시
    CALL gds.betweenness.stream('personaGraph')
    YIELD nodeId, score
    RETURN gds.util.asNode(nodeId).uuid AS uuid, score
    ORDER BY score DESC LIMIT 10
    ```

### 5. Frontend Changes
*   **"핵심 인물" 탭 추가:**
    *   중심성 지표별 상위 20인 리스트 (Plotly bar chart 활용).
    *   Network Graph 시각화 시 노드 크기를 중심성 스코어에 비례하도록 조정.
    *   시뮬레이션 모드: 노드 클릭 시 '제거 후 영향' 팝업 표시.

### 6. 구현 단계
*   **P0:** PageRank 및 Degree Centrality 기반 랭킹 대시보드 구축.
*   **P1:** Betweenness Centrality 도입 및 커뮤니티 대표자 선정 로직 연동.
*   **P2:** 노드 제거 시뮬레이션 및 시각화 고도화.

---

## 3. Feature 2: 페르소나 기반 추천 엔진 (Recommendation Engine)

### 1. 목표 및 성공 지표
*   **목표:** 유사한 페르소나들이 보유한 특징(취미, 기술, 지역 등) 중 대상 페르소나가 가지지 않은 항목을 추천합니다.
*   **성공 지표:**
    *   추천 결과의 관련성(유사 페르소나와의 공유 빈도) 기반 정밀도 향상.
    *   추천 API 응답 속도 2초 이내.

### 2. 사용자 스토리
*   "사용자로서 나와 비슷한 성향의 사람들이 즐기는 취미 중 내가 아직 경험핮지 못한 활동을 제안받고 싶다."
*   "구직자로서 내 기술 스택과 유사한 페르소나들이 공통적으로 보유한 '보완 기술'을 추천받고 싶다."
*   "마케터로서 특정 직업군이 선호하는 거주 지역 데이터를 기반으로 타겟팅 전략을 세우고 싶다."

### 3. API Contract
*   **GET `/api/recommend/{uuid}`**
    *   Query Params: `category` (hobby, skill, occupation, district), `top_n` (int)
    *   Response: `{recommendations: List[{item_name, reason_score, similar_users_count}]}`

### 4. Backend Logic
*   **Collaborative Filtering (Graph-based):**
    1. 대상 페르소나 `P`와 유사한 `Top-K` 페르소나 집합 `S` 추출 (이미 구축된 FastRP/KNN 활용).
    2. `S`가 공통적으로 가진 속성 중 `P`에게 없는 속성 필터링.
    3. 속성별 등장 빈도(Frequency)와 유사도 점수를 가중치로 합산하여 랭킹.
*   **Reasoning:** LLM을 사용하여 "당신과 유사한 80%의 개발자가 'TypeScript'를 보유하고 있습니다"와 같은 문구 생성.

### 5. Frontend Changes
*   **상세 페이지 내 "추천" 섹션:**
    *   Streamlit의 `st.columns`를 활용한 카드형 UI.
    *   각 카드 클릭 시 해당 속성을 가진 다른 유사 펴r소나 리스트로 이동.

### 6. 구현 단계
*   **P0:** 취미 및 기술에 대한 빈도 기반 단순 추천 로직 개발.
*   **P1:** FastRP 유사도 점수를 가중치로 사용하는 정교한 스코어링 모델 적용.
*   **P2:** LLM 기반 추천 사유(Reasoning) 자동 생성 기능 추가.

---

## 4. Feature 3: 대화형 탐색 챗봇 (Multi-turn RAG)

### 1. 목표 및 성공 지표
*   **목표:** 단발성 쿼리를 넘어 대화 문맥(Context)을 유지하며 데이터를 탐색하는 인터페이스를 제공합니다.
*   **성공 지표:**
    *   이전 대화의 필터 조건(나이, 지역 등)을 3턴 이상 유지하며 쿼리 수행 성공률 90% 이상.
    *   사용자 피드백 '유용함' 점수 4.0/5.0 달성.

### 2. 사용자 스토리
*   "분석가로서 '서울 사는 20대'라고 먼저 묻고, 이어서 '그중 남성만 보여줘'라고 자연스럽게 필터링하고 싶다."
*   "일반 사용자로서 복잡한 검색 UI 대신 채팅으로 '나랑 비슷한 취미를 가진 사람들은 주로 어디 사니?'라고 묻고 싶다."
*   "사용자로서 이전 답변 내용에 대해 '좀 더 자세히 설명해줘' 또는 '표로 요약해줘'라고 요청하고 싶다."

### 3. API Contract
*   **POST `/api/chat`**
    *   Request: `{session_id: str, message: str, stream: bool}`
    *   Response: `{response: str, context_filters: dict, sources: List[str]}`

### 4. Backend Logic
*   **LangGraph State Management:**
    *   `state` 객체에 `history` (대화 기록)와 `current_filters` (현재 유지 중인 필터) 저장.
    *   **Intent Classifier Node:** 사용자의 발화가 '검색', '비교', '통계', '일반 대화' 중 무엇인지 판별.
    *   **Context Refiner Node:** "그중에서"와 같은 지시어가 있을 경우 이전 필터에 새 조건을 병합(Merge).
*   **Memory:** `Neo4jChatMessageHistory` 또는 In-memory 저장소를 활용하여 세션별 맥락 유지.

### 5. Frontend Changes
*   **메인 화면의 Chat Interface:**
    *   `st.chat_message` 및 `st.chat_input`을 사용하여 현대적인 UI 제공.
    *   사이드바에 현재 적용 중인 '대화 문맥 필터'를 실시간으로 표시 (칩 형태).

### 6. 구현 단계
*   **P0:** 기존 `/api/insight`를 LangChain Memory와 연결하여 기본 대화 가능하게 구현.
*   **P1:** 지시어 분석 및 필터 누적 로직(Contextual Filter) 적용.
*   **P2:** 차트 및 그래프 시각화 결과물을 채팅창 내에 직접 렌더링.

---

## 5. 공통 사항 및 리스크 관리

### 1. 교차 기능 통합 (Cross-feature Integration)
*   챗봇 대화 중 "이 사람에게 추천할 활동은?"이라고 물으면 Feature 2의 API를 호출하여 답변.
*   핵심 인물 분석 결과(Feature 1)를 추천 엔진의 가중치로 활용 (영향력 있는 인물의 특징을 우선 추천).

### 2. 성능 고려 사항 (1M 노드 대응)
*   **GDS 캐싱:** 중심성 스코어는 실시간 계산 대신 배치(Batch)로 계산하여 노드 프로퍼티에 저장.
*   **Index 최적화:** `Composite Index` (age + sex + province)를 생성하여 챗봇의 다중 필터 쿼리 속도 보장.
*   **Projection 관리:** GDS 그래프 프로젝션을 메모리에 상주시킬 때 필요한 속성만 선택하여 RAM 사용량 최적화.

### 3. 테스트 전략
*   **Unit Tests:** 추천 스코어링 로직 및 필터 병합 로직 검증.
*   **Integration Tests:** FastAPI와 Neo4j 간의 GDS 호출 흐름 테스트.
*   **LLM Evaluation:** RAG 답변의 할루시네이션(Hallucination) 방지를 위한 정답 셋(Golden Set) 평가.

### 4. 리스크 및 대응 (Risk & Mitigation)
*   **리스크:** 1M 노드에서 Betweenness 계산 시 메모리 부족 가능성.
    *   **대응:** `gds.betweenness.stats`로 미리 메모리 추정 후, 필요 시 샘플링 기법(RA-Brandin) 적용.
*   **리스크:** 대화가 길어질수록 LLM의 컨텍스트 윈도우 초과.
    *   **대응:** 요약(Summarization) 노드를 추가하여 핵심 문맥과 필터 정보만 압축 전달.

---

## 6. 파일 구조 및 구현 대상

### Feature 1: 네트워크 영향력 분석
| 파일 경로 | 목적 |
|:---|:---|
| `src/gds/centrality.py` | PageRank, Betweenness, Degree 계산 및 저장 |
| `src/api/routes/influence.py` | `/api/influence/top`, `/api/influence/simulate-removal` 엔드포인트 |
| `app/streamlit_app.py` | "핵심 인물" 탭 추가 |
| `tests/test_centrality.py` | 중심성 계산 로직 단위 테스트 |
| `tests/test_api_influence.py` | API 응답 및 시뮬레이션 테스트 |

### Feature 2: 추천 엔진
| 파일 경로 | 목적 |
|:---|:---|
| `src/graph/recommendation.py` | 추천 알고리즘 (Collaborative Filtering) 구현 |
| `src/api/routes/recommend.py` | `/api/recommend/{uuid}` 엔드포인트 |
| `app/streamlit_app.py` | 프로필 상세 페이지에 "추천" 섹션 추가 |
| `tests/test_recommendation.py` | 추천 스코어링 로직 테스트 |

### Feature 3: 대화형 챗봇
| 파일 경로 | 목적 |
|:---|:---|
| `src/rag/chat_graph.py` | 멀티턴 LangGraph 상태 관리 및 필터 병합 로직 |
| `src/api/routes/chat.py` | `/api/chat` 엔드포인트 |
| `app/streamlit_app.py` | 메인 화면 Chat Interface 추가 |
| `tests/test_chat_graph.py` | 필터 병합 및 대화 흐름 테스트 |

---

## 7. QA 시나리오 (테스트 계획)

### Feature 1 QA
| 시나리오 | 테스트 방법 | 예상 결과 |
|:---|:---|:---|
| PageRank 계산 | `pytest tests/test_centrality.py -v` | 상위 10명의 UUID와 score 반환, 계산 시간 < 5초 |
| 시뮬레이션 | `curl -X POST /api/influence/simulate-removal -d '{"target_uuids": ["xxx"]}'` | original_connectivity > current_connectivity 반환 |
| API 응답 | `pytest tests/test_api_influence.py` | metric=pagerank, limit=10 시 10개 항목 반환 |

### Feature 2 QA
| 시나리오 | 테스트 방법 | 예상 결과 |
|:---|:---|:---|
| 취미 추천 | `curl /api/recommend/{uuid}?category=hobby&top_n=3` | 대상이 없는 취미 3개 반환, reason_score 포함 |
| 스킬 추천 | `pytest tests/test_recommendation.py` | 유사 페르소나가 많이 가진 스킬 상위 노출 |
| 응답 시간 | `time curl /api/recommend/{uuid}` | 응답 시간 < 2초 |

### Feature 3 QA
| 시나리오 | 테스트 방법 | 예상 결과 |
|:---|:---|:---|
| 필터 유지 | `pytest tests/test_chat_graph.py` | 3턴 연속 "서울 20대 남성" 필터 정확히 누적 |
| 문맥 병합 | `curl -X POST /api/chat -d '{"session_id": "s1", "message": "그중에서 개발자만"}'` | 이전 필터에 occupation='개발자' 추가된 상태로 응답 |
| 응답 품질 | Golden Set 평가 (수동) | 90% 이상의 질문에 맥락 유지하며 정확한 응답 |

---

**작성일:** 2026년 4월 28일
**작성자:** Antigravity (Sisyphus-Junior)
**상태:** PRD 완료 / Momus 검수 완료 (1차 REJECT 후 보강)
