# 프로젝트 체크리스트

> 각 항목 완료 시 [x] 표시. 진행 중이면 [~] 표시.

## 문서 역할 및 충돌 우선순위

- `PRD.md`: 본 프로젝트의 요구사항, 범위, 설계 결정 및 품질 게이트를 결정하는 문서입니다.
- `CHECKLIST.md`: PRD 실행 상태와 완료 조건을 추적·검증하는 단일 진행 규칙 문서입니다.
- `README.md`: 실행 방법, 상태 가시화, 참고 자료를 제공하는 실무 안내서입니다.
- 문서 충돌 시 우선순위는 **`PRD.md` → `CHECKLIST.md` → `README.md`** 입니다.
- 보조 산출물(`docs/`, 실험/리뷰 기록)은 본 우선순위를 무시해 대체할 수 없습니다.

---

## Phase 0: 프로젝트 구조 및 환경 설정

- [x] 프로젝트 디렉토리 구조 생성
- [x] Python 3.11 기반 `.venv` 생성
- [x] `requirements.txt` 작성
- [x] `.env.example` 작성 (NEO4J_URI, NEO4J_PASSWORD, NVIDIA_API_KEY 등)
- [x] `src/config.py` 설정 모듈 작성
- [x] Neo4j Community Edition 설치 및 실행 확인
- [x] Python 가상환경 생성 및 의존성 설치
- [x] KURE-v1 모델 다운로드 및 CUDA 로드 테스트 (torch 2.11.0+cu128, RTX 4060 확인)
- [x] NVIDIA API 키 설정 및 DeepSeek-V4-Pro 호출 테스트
- [x] Hugging Face `nvidia/Nemotron-Personas-Korea` 접근 테스트

## Phase 1: 데이터 파이프라인

- [x] `src/data/loader.py` — CSV/Parquet 데이터 로더
- [x] Hugging Face `datasets` 기반 원본 데이터 로더
- [x] `src/data/parser.py` — 리스트 필드 파싱 (`ast.literal_eval`), district 분리
- [x] `src/data/preprocessor.py` — 결측치 처리, age_group 파생, 정규화
- [x] 데이터 로드 → 파싱 → 전처리 파이프라인 통합 테스트
- [x] 샘플 로드 확인 (`DATA_SAMPLE_SIZE=10000`) 후 전처리 검증

## Phase 2: Neo4j 지식 그래프 구축

- [x] `src/graph/schema.py` — 그래프 스키마 정의 (노드/엣지 타입)
- [x] Neo4j 제약조건 및 인덱스 생성 (uuid UNIQUE, name 인덱스)
- [x] `src/graph/loader.py` — Person 노드 적재
- [x] 범주형 관계 적재 (LIVES_IN, WORKS_AS, EDUCATED_AT 등)
- [x] 리스트형 관계 적재 (HAS_SKILL, ENJOYS_HOBBY)
- [x] 페르소나 텍스트를 Person 속성으로 적재 (6개 페르소나 + cultural_background + career_goals)
- [x] 지역 계층 적재 (District → Province → Country)
- [x] 그래프 적재 검증 (Person 10,000 / 전체 노드 71,154 / 관계 317,130)

## Phase 3: 임베딩 생성 및 Neo4j 벡터 인덱스

- [x] `src/embeddings/kure_model.py` — KURE-v1 모델 로더 (float16, CUDA)
- [x] KURE-v1 CPU 추론 방지 (CPU 설정/감지 시 warning 후 중단)
- [x] 샘플 → 전체 1M rows 순서로 페르소나 텍스트 임베딩 배치 생성 (batch_size=64)
- [x] 임베딩 진행률 표시 (tqdm)
- [x] `src/embeddings/vector_index.py` — Neo4j 벡터 인덱스 구축
- [x] 임베딩을 Neo4j 노드 속성으로 저장
- [x] 임베딩 증분 적재 (재실행 방지, `--skip-existing`)
- [x] 임베딩 품질 검증 (CUDA 샘플 1024차원 벡터 생성 및 10,000건 Vector Index 적재)

## Phase 4: Neo4j GDS 파이프라인

- [x] Neo4j GDS 플러그인 설치 및 확인
- [x] `src/gds/fastrp.py` — FastRP 임베딩 파이프라인 (1024→256 차원)
- [x] `src/gds/communities.py` — Leiden 커뮤니티 탐지
- [x] `src/gds/similarity.py` — KNN 유사도 매칭 (top-k)
- [x] FastRP 임베딩 결과 검증 (71,154개 노드 속성 write)
- [x] Leiden 커뮤니티 결과 검증 (communityCount=88, modularity≈0.493)
- [x] KNN 매칭 결과 검증 (SIMILAR_TO 관계 50,000개 생성)

## Phase 5: LangChain/LangGraph RAG 엔진

- [x] `src/rag/llm.py` — NVIDIA API DeepSeek-V4-Pro 클라이언트
- [x] `src/rag/cypher_chain.py` — GraphCypherQAChain 설정 (validate_cypher=True)
- [x] `src/rag/vector_chain.py` — Neo4j Vector Index + KURE-v1 검색 설정
- [x] `src/rag/router.py` — LangGraph StateGraph 라우터
  - [x] 질문 분류 노드 (통계/집계 vs 의미론적 vs 복합)
  - [x] Cypher 실행 노드
  - [x] Vector 검색 노드
  - [x] LLM 응답 생성 노드
  - [x] Cypher 실패 시 재시도 및 fallback
  - [x] Vector 빈 결과 fallback
  - [x] 복합 질의 (Cypher + Vector 결합) 노드
- [~] 라우터 엔드투엔드 테스트 (코드 완료, 실제 LLM/Neo4j 연동 시 검증)

## Phase 6: FastAPI 엔드포인트

- [x] `src/api/main.py` — FastAPI 앱 초기화 (CORS, 라우터 마운트)
- [x] `src/api/schemas.py` — Pydantic 요청/응답 모델
- [x] `POST /api/insight` — 자연어 인사이트 질의
- [x] `POST /api/similar/{uuid}` — 유사 페르소나 매칭 (weights + text/graph 결합)
- [x] `GET /api/communities` — 커뮤니티 자동 탐지 (라벨 자동 생성)
- [x] `GET /api/path/{uuid1}/{uuid2}` — 관계 경로 탐색 (동일 UUID 차단, PRD 스키마 맞춤)
- [x] FastAPI 글로벌 예외 핸들러
- [~] Swagger UI 확인 (/docs) (코드 완료, 서버 실행 후 확인)
- [x] 각 엔드포인트 개별 테스트

## Phase 7: Legacy Streamlit 프론트엔드 이력

- [x] Legacy Streamlit 프론트엔드 구현 완료 (현재 코드는 삭제됨)
- [x] 인사이트 질의 UI (입력창 + 결과 표시)
- [x] 유사 페르소나 매칭 UI (uuid 입력 + 결과 카드)
- [x] 커뮤니티 탐지 UI (목록 + 특성 표시)
- [x] 관계 경로 UI (두 uuid 입력 + 경로 시각화)
- [x] FastAPI 백엔드 연동 확인 이력 기록

## Phase 8: 통합 테스트 및 검증

- [x] `tests/test_data_parser.py` — 데이터 파싱 테스트
- [x] `tests/test_graph_loader.py` — 그래프 적재 테스트
- [x] `tests/test_embeddings.py` — 임베딩 품질 테스트
- [x] `tests/test_api.py` — API 엔드포인트 테스트
- [x] FastAPI 예외 핸들러 테스트
- [x] 로깅 설정
- [~] 엔드투엔드 테스트 (질문 → 응답 전체 흐름) (코드 완료, 실제 서비스 연동 시 검증)
- [ ] 성능 테스트 (개발 샘플 및 전체 1M rows 기준 응답 시간 측정)

---

> Note: Phase 15~18은 신규 Phase 3 구현 우선순위가 높아 Phase 8 바로 뒤에 배치한다. Phase 9~14는 이미 구현된 Phase 2 후속 기록이다.

## Phase 15: 네트워크 영향력 및 핵심 인물 분석 (F10)

> PRD Expansion Feature 1 — 네트워크 과학 기반 영향력 분석

- [x] `src/gds/centrality.py` — 중심성 알고리즘 구현
  - [x] PageRank 계산 (API 실시간 stream 금지, gds.pageRank.write 배치)
  - [x] Betweenness Centrality 계산 (API 실시간 stream 금지, RA-Brandes/sampling 배치)
  - [x] Degree Centrality 계산
  - [x] 결과를 Person 노드 속성에 저장 (pagerank, betweenness, degree)
  - [x] `src/jobs/centrality_batch.py` — 배치 실행 오케스트레이션
  - [x] 배치 계산 스케줄링 운영 문서화 (FastAPI/Streamlit request 경로 밖에서 실행)
  - [x] GDS projection 존재 여부 확인 및 누락 시 배치 시작 전 재생성
  - [x] 배치 실행 상태/마지막 갱신 시각/run_id 기록
  - [x] 부분 write 방지용 publish 규칙 정의 (마지막 성공 run_id만 API 노출)
- [x] `src/api/routes/influence.py` — 영향력 API 엔드포인트
  - [x] `GET /api/influence/top?metric=pagerank&limit=10` — 중심성 지표별 상위 인물
  - [x] `POST /api/influence/simulate-removal` — 노드 제거 시뮬레이션
    - [x] original_connectivity (WCC) 측정
    - [x] target_uuids 제거 후 connectivity 측정
    - [x] fragmentation_increase 계산
    - [x] connectivity = largest_component_size / total_subgraph_nodes
    - [x] 최대 5개 UUID, 3-hop, 10초 timeout 초과 시 422
  - [x] metric 파라미터 검증 (pagerank, betweenness, degree)
  - [x] 존재하지 않는 metric → 400 에러
  - [x] 중심성 score 없음/배치 미실행 → 503 에러
- [x] `src/api/schemas.py` — `InfluenceResponse`, `InfluenceTopItem`, `RemovalSimulationResponse` Pydantic 모델
- [x] Legacy Streamlit UI에 "핵심 인물" 탭 추가 이력 기록
  - [x] 중심성 지표 선택 (radio button: PageRank / Betweenness / Degree)
  - [x] 상위 20인 테이블 (score, rank, community_id)
  - [x] bar chart (상위 10인 시각화)
  - [x] 노드 선택 기반 제거 시뮬레이션
  - [x] `마지막 갱신` 표시 및 미준비 경고
  - [x] 시뮬레이션 선택 노드 취소/초기화 버튼
  - [x] Streamlit rerun 후 선택 metric/노드 상태 유지
- [x] `tests/test_influence.py` — 중심성/영향력 API 테스트
  - [x] PageRank 계산 결과 10개 반환 확인
  - [ ] PageRank/Degree 배치 계산 시간 < 30분 기준 검증 항목 정의
  - [ ] Betweenness 샘플링 배치 계산 시간 < 2시간 기준 검증 항목 정의
  - [x] `/api/influence/top` 조회 시간 < 100ms 기준 검증
- [x] `tests/test_influence.py` — API 테스트
  - [x] `/api/influence/top` 응답 구조 검증
  - [x] `/api/influence/top` 응답에 `last_updated_at` 포함
  - [x] 시뮬레이션 응답 (original > current)
  - [x] 잘못된 metric → 400

---

## Phase 16: 페르소나 기반 추천 엔진 (F11)

> PRD Expansion Feature 2 — 그래프 기반 협업 필터링 추천

- [x] `src/graph/recommendation.py` — 추천 알고리즘 구현
  - [x] 대상 페르소나 `P`의 유사 페르소나 집합 `S` 추출 (SIMILAR_TO 활용)
  - [x] `S`의 속성 중 `P`에게 없는 항목 필터링
  - [x] 속성별 빈도 + 유사도 가중치 스코어링
  - [x] category 파라미터 (hobby, skill, occupation, district)별 로직
- [x] `src/api/routes/recommend.py` — 추천 API 엔드포인트
  - [x] `GET /api/recommend/{uuid}?category=hobby&top_n=5`
  - [x] category 검증 (hobby, skill, occupation, district)
  - [x] 존재하지 않는 UUID → 404
  - [x] SIMILAR_TO 관계 없음 → 503 명시 오류 (Phase 3 P0 기본값)
  - [x] 응답 시간 < 500ms (LLM 동기 호출 금지, 템플릿 기반 reasoning)
- [x] `src/api/schemas.py` — `RecommendResponse`, `RecommendItem` Pydantic 모델
  - [x] item_name, reason_score, similar_users_count
- [x] Legacy Streamlit UI에 프로필 상세 페이지 "추천" 섹션 추가 이력 기록
  - [x] 카드형 UI (st.columns)
  - [x] 카테고리 선택 (취미 / 기술 / 직업 / 지역)
  - [x] 각 카드 expander → 해당 속성 보유 유사 페르소나 목록
  - [x] 추천 결과 없음 empty state 문구 표시
  - [x] 추천 데이터 미준비/503 error state 문구 표시
- [x] `tests/test_recommendation.py` — 단위 테스트
  - [x] 취미 추천 (대상이 없는 취미 상위 노출)
  - [x] 스킬 추천 (유사 페르소나 공통 스킬)
  - [x] 추천 항목이 대상의 기존 속성과 중복되지 않음
  - [x] 응답 시간 < 500ms

---

## Phase 17: 대화형 탐색 챗봇 (F12)

> PRD Expansion Feature 3 — 멀티턴 RAG 챗봇

- [x] `src/rag/chat_graph.py` — 멀티턴 LangGraph 구현
  - [x] `docs/decisions/ADR-004-chatbot-filter-state.md` — FilterState 설계 작성 후 구현 시작
  - [x] `docs/f12-chatbot-implementation-plan.md` — F12 구현 계획 수립
  - [x] ChatState TypedDict (history, current_filters, last_intent)
  - [x] Intent Classifier Node (검색/통계/리셋/일반대화)
  - [x] Context Refiner Node ("그중에서" → 필터 병합)
  - [x] Filter Merge 로직 (age_group, sex, province 누적)
  - [x] 최근 5턴까지만 history 유지
  - [x] "리셋"/"처음부터" 발화 시 current_filters 초기화
  - [x] Memory 관리 (bounded In-memory session store)
- [x] `src/api/routes/chat.py` — 채팅 API 엔드포인트
  - [x] `POST /api/chat` (session_id, message, stream)
  - [x] 세션별 대화 맥락 유지
  - [x] context_filters 응답 (현재 적용 중인 필터)
  - [x] sources 포함 (참조된 데이터 출처)
- [x] `src/api/schemas.py` — `ChatRequest`, `ChatResponse` Pydantic 모델
- [x] Legacy Streamlit UI 메인 화면 Chat Interface 구현 이력 기록
  - [x] `st.chat_message` 기반 UI
  - [x] `st.chat_input` 입력창
  - [x] 현재 필터 표시 (칩 형태)
  - [x] 이전 대화 기록 표시
  - [x] 첫 진입 empty state 문구 표시
  - [x] 답변 생성 중 loading state 표시
  - [x] 필터 초기화 버튼 및 초기화 완료 문구 표시
- [x] `tests/test_chat_graph.py` — 단위 테스트
  - [x] 3턴 연속 필터 유지 (서울 → 20대 → 남성)
  - [x] 필터 병합 검증 (AND 조건 누적)
  - [x] 세션 분리 (session_id별 독립 맥락)
- [x] `tests/test_api_chat.py` — API 테스트
  - [x] `/api/chat` 응답 구조 검증
  - [x] context_filters 포함 확인

---

## Phase 18: 확장 기능 통합 및 검증

- [ ] Phase 15~17 엔드포인트 Swagger UI 일괄 확인
- [ ] 현재 운영 프론트(Next.js)에 신규 3개 기능 화면/흐름 반영
  - [x] 핵심 인물 탭
  - [x] 추천 섹션 (프로필 상세 내)
  - [x] 채팅 인터페이스 (메인 화면)
- [ ] 교차 기능 통합 테스트
  - [ ] 챗봇 → 추천 API 호출 ("이 사람에게 추천할 활동은?")
  - [ ] 핵심 인물 → 추천 가중치 반영
- [ ] 성능 테스트
  - [x] `/api/influence/top` 조회 < 100ms (precomputed score + index)
  - [ ] PageRank/Degree 배치 계산 < 30분 (1M 노드)
  - [x] 추천 API < 500ms
  - [x] 챗봇 응답 < 3초 (TestClient smoke 기준)
- [ ] 운영/상태 테스트
  - [ ] 중심성 배치 마지막 성공 시각/run_id 조회 가능
  - [ ] 중심성 배치 실패 시 마지막 성공 결과 유지 + stale 경고 표시
  - [ ] GDS projection 미준비 상태에서 사용자 API가 전체 재계산을 트리거하지 않음
  - [ ] SIMILAR_TO 미준비 상태에서 추천 API가 503 오류를 명확히 반환
- [ ] 문서화
  - [ ] PRD 업데이트 (구현 완료 항목 반영)
  - [ ] API 문서 업데이트 (신규 엔드포인트)
  - [ ] 사용 가이드 작성

---

## Phase 19: 대규모 운영검증 계획 및 사전 검수 (F13)

> PRD §11.2~11.3 — 코드 구현 전 검수 우선. 이 Phase의 항목은 자동화/코드 작성 전에 계획과 기준을 승인받기 위한 체크리스트다.

- [ ] 검수 계획 승인
  - [ ] 대규모 운영검증 범위 확정 (1M 데이터, F10/F11/F12, Streamlit UI)
  - [ ] 검증 환경 명세 작성 (OS, Neo4j/GDS 버전, heap/pagecache, CPU/GPU, 데이터 크기)
  - [ ] Go/No-Go 판정 기준 문서화
- [ ] 데이터 준비 검수
  - [ ] Person 1M 적재 여부 확인 기준 정의
  - [ ] 핵심 관계 수 기준 정의 (`SIMILAR_TO`, `LIVES_IN`, `WORKS_AS`, `ENJOYS_HOBBY`, `HAS_SKILL`)
  - [ ] 중심성 속성(`pagerank`, `degree`, `betweenness`) 누락률 허용 기준 정의
- [ ] 배치 성능 검수 계획
  - [ ] PageRank/Degree 배치 < 30분 목표 측정 절차 작성
  - [ ] Betweenness RA-Brandes sampling < 2시간 목표 측정 절차 작성
  - [ ] 배치 실패 시 마지막 성공 결과 유지/stale 경고 검증 절차 작성
  - [ ] Windows Task Scheduler 또는 cron 실행/재시도/로그 확인 절차 작성
- [ ] API SLA 검수 계획
  - [ ] `/api/influence/top` < 100ms 측정 절차 작성
  - [ ] `/api/recommend/{uuid}` < 500ms 측정 절차 작성
  - [ ] `/api/chat` < 3초 측정 절차 작성
  - [ ] GDS/KNN/중심성 미준비 시 503/422 응답 검수 시나리오 작성
- [ ] Streamlit 운영 QA 계획
  - [ ] 핵심 인물 탭 empty/loading/error/stale state 검수 시나리오 작성
  - [ ] 추천 섹션 empty/loading/error state 검수 시나리오 작성
  - [ ] 대화형 탐색 탭 세션 분리/필터 누적/리셋 검수 시나리오 작성
  - [ ] 전체 데이터 기준 UI timeout 또는 rerun 문제 확인 절차 작성
- [ ] 검수 리뷰
  - [ ] 운영검증 계획 Momus/Oracle 또는 동등 리뷰 요청
  - [ ] 리뷰 blocker 수정
  - [ ] 구현/자동화 착수 승인 여부 기록

---

## Phase 20: 확장 및 고도화 계획 검수 (F14)

> PRD §11.4~11.5 — F12 MVP 이후 기능 확장을 코드 작성 전 계획/검수한다.

- [ ] 통합 챗봇 UX 전환 계획
  - [ ] 대화형 탐색을 Streamlit 메인 자연어 UX로 유지하는 화면 구조 정의
  - [ ] 인사이트 질의를 별도 주 탭이 아닌 챗봇 내부 `고급 분석 모드`로 흡수하는 전환 방식 정의
  - [ ] 고급 분석 진입 UX 결정 (토글/버튼/slash command/자동 intent 중 선택)
  - [ ] `/api/chat` mode 확장 또는 내부 `/api/insight` 호출 중 API 통합 방식 결정
  - [ ] 기존 `/api/insight` 호환성 유지 및 deprecated 안내 여부 결정
  - [ ] 분석 mode 응답의 sources/query_type/history 저장 방식 정의
  - [ ] 기존 `💡 인사이트 질의` 탭 이동/숨김/유지 정책과 롤백 기준 정의
- [ ] 챗봇 orchestration P1 계획
  - [ ] 챗봇 → 추천 API 호출 UX/API 흐름 정의
  - [ ] 챗봇 → 영향력 API 호출 UX/API 흐름 정의
  - [ ] UUID/선택 인물 기반 프로필 intent 정의
  - [ ] P1 범위와 제외 범위 명시
- [ ] 챗봇 자연어 이해 P2 계획
  - [ ] regex filter extraction 한계 목록화
  - [ ] LLM/Pydantic structured output 적용 기준 정의
  - [ ] hallucination/잘못된 필터 추출 방지 guardrail 정의
  - [ ] golden set 50개 이상 평가 계획 작성
- [ ] 추천/영향력 고도화 계획
  - [ ] 중심성/커뮤니티/유사도 혼합 추천 스코어링 설계
  - [ ] 영향력 높은 인물 특성을 추천 가중치에 반영하는 기준 정의
  - [ ] 추천 결과 설명 가능성(reason) 확장 기준 정의
- [ ] 비동기/운영 고도화 계획
  - [ ] removal simulation async job 전환 필요성 평가
  - [ ] job status API 설계 초안 작성
  - [ ] batch run_id, last_success_at, stale 상태 UI 노출 계획 작성
  - [ ] structured logs/metrics/alerting 계획 작성
- [ ] 고도화 검수 게이트
  - [ ] P1 구현 우선순위 승인
  - [ ] P2 연구/실험 항목 분리
  - [ ] 구현 착수 전 PRD/CHECKLIST 재검수 완료

---

## Phase 21: 챗봇 LLM 응답 합성 (F15)

> PRD §11.7 — Rule 우선 + LLM 합성. 기존 ChatGraph의 rule-based 분류/필터는 유지하고 응답 생성에만 LLM을 추가한다.
> Oracle 권고 반영: LLM 합성은 별도 `synthesize` LangGraph 노드, InsightRouter는 singleton, general intent 시 필터 context prefix 포함.

- [x] 1단계 (P0): LLM 합성 레이어 추가
  - [x] `src/rag/chat_graph.py` — `synthesize` LangGraph 노드 추가
    - [x] LangGraph 플로우 변경: `classify → merge_filters → respond → synthesize → commit_history → END`
    - [x] `respond` 노드는 Neo4j 쿼리만 실행하고 raw 결과를 `ChatState`에 저장 (드라이버 즉시 반환)
    - [x] `ChatState`에 `raw_results` 필드 추가 (Neo4j 결과 임시 저장용)
    - [x] `synthesize` 노드에서 raw 결과 + 사용자 질문 + 현재 필터를 LLM에 전달
  - [x] `_synthesize_response()` 헬퍼 함수 추가
    - [x] LLM 프롬프트 템플릿 작성 (시스템 프롬프트: 데이터 전용 답변 강제)
    - [x] 프롬프트 입력: 사용자 질문 + 현재 필터 요약 + Neo4j 결과 (compact JSON-lines, Oracle 권고)
    - [x] Neo4j 결과 상위 20건 제한 (컨텍스트 윈도우 비대화 방지)
  - [x] `_run_stats()` 수정 — Neo4j 쿼리 실행 + raw 결과 반환 (LLM 호출 제거, synthesize 노드로 이관)
  - [x] `_run_search()` 수정 — Neo4j 쿼리 실행 + raw 결과 반환 (LLM 호출 제거, synthesize 노드로 이관)
  - [x] LLM 실패 시 graceful fallback — 기존 템플릿 문자열로 응답 (raw 결과 그대로 사용)
  - [x] `src/rag/llm.py`의 `create_llm()` 재사용 (신규 LLM 클라이언트 불필요)
  - [x] `thinking: False` 설정 유지 확인 (Oracle 권고, 지연 방지)
- [x] 1단계 테스트
  - [x] `tests/test_chat_graph.py` — LLM 합성 단위 테스트
    - [x] stats 질문 → `synthesize` 노드에서 LLM 호출 확인 (monkeypatch)
    - [x] search 질문 → `synthesize` 노드에서 LLM 호출 확인 (monkeypatch)
    - [x] LLM 실패 시 템플릿 fallback 확인
    - [x] Neo4j 결과 20건 초과 시 상위 20건만 LLM 전달 확인
    - [x] `respond` 노드 완료 시점에 Neo4j 드라이버가 닫혀 있는지 확인
  - [x] `tests/test_api_chat.py` — API 레벨 합성 응답 테스트
  - [ ] 응답 시간 측정: search/stats < 5초 (Neo4j < 1초 + LLM < 4초)
- [x] 2단계 (P1): InsightRouter fallback 연결
  - [x] InsightRouter를 module-level singleton으로 공유 (Oracle 권고, 매 요청 생성 금지)
  - [x] `src/rag/chat_graph.py` — general intent 시 InsightRouter 호출
    - [x] general intent → 현재 필터를 context prefix로 포함하여 InsightRouter.ask() 실행 (Oracle 권고)
    - [x] InsightRouter 결과를 ChatGraph 세션 history에 저장
  - [x] `_respond()` 수정 — general intent 분기에 InsightRouter 연결
  - [x] InsightRouter 실패 시 기존 general_response() 문자열 fallback
- [x] 2단계 테스트
  - [x] `tests/test_chat_graph.py` — general intent → InsightRouter fallback 테스트
    - [x] general 질문 → InsightRouter 호출 확인 (monkeypatch)
    - [x] InsightRouter 실패 시 fallback 확인
    - [x] InsightRouter 결과가 세션 history에 저장되는지 확인
    - [x] 현재 필터가 InsightRouter에 context prefix로 전달되는지 확인
  - [ ] 응답 시간 측정: general fallback < 10초
- [ ] 제외 범위 확인
  - [ ] LLM 기반 필터 추출은 이 Phase에서 구현하지 않음 (P2 이후)
  - [ ] 스트리밍 응답은 이 Phase에서 구현하지 않음 (프론트엔드 전환 후)
  - [ ] `/api/insight` 엔드포인트는 삭제하지 않음 (호환성 유지)
- [ ] 검수 게이트
  - [ ] PRD §11.7 요구사항과 구현 일치 확인
  - [ ] 기존 테스트 전체 통과 (회귀 없음)
  - [ ] LLM hallucination 방지 프롬프트 검증 (golden set 10개 이상)

---

## Phase 9: 페르소나 프로필 상세 뷰 (F7)

> PRD Phase 2 §2.3 — 한 사람의 모든 정보를 단일 API 호출로 반환

- [x] `src/graph/persona_queries.py` — 프로필 전용 Cypher 쿼리 작성
  - [x] 기본 프로필 + 모든 관계 엔티티 (District, Province, Occupation, EducationLevel, Hobby, Skill 등) OPTIONAL MATCH
  - [x] 유사 인물 미리보기 쿼리 (SIMILAR_TO 관계, top-3)
  - [x] 커뮤니티 정보 쿼리 (community_id)
  - [x] 그래프 통계 (total_connections, hobby_count, skill_count)
- [x] `src/api/schemas.py` — `PersonaProfileResponse` Pydantic 모델 추가
- [x] `src/api/routes/persona.py` — `GET /api/persona/{uuid}` 엔드포인트
  - [x] 존재하지 않는 UUID → 404 NotFoundException
  - [x] skills/hobbies 빈 리스트 허용
  - [x] community_id null 허용 (GDS 미실행 상태)
  - [x] similar_preview 빈 리스트 허용 (KNN 미실행 상태)
- [x] `src/api/main.py` — persona 라우터 마운트
- [x] `tests/test_persona.py` — 단위 테스트 (4개 테스트 통과)
- [ ] Swagger UI 응답 구조 확인

## Phase 10: 인구통계 대시보드 (F6)

> PRD Phase 2 §2.2 — 전체 요약 통계 + 차원별 드릴다운

- [x] `src/graph/stats_queries.py` — 집계 Cypher 쿼리 모음
  - [x] 전체 통계: age, sex, province, education, marital 분포 집계
  - [x] top_occupations, top_hobbies, top_skills 랭킹 쿼리
  - [x] 드릴다운: 단일 dimension + 필터(province, age_group, sex) 조합 쿼리
  - [x] ratio 계산 로직 (count / filtered_count)
- [x] `src/api/schemas.py` — `StatsResponse`, `DimensionStatsResponse` Pydantic 모델 추가
- [x] `src/api/routes/stats.py` — `GET /api/stats` 전체 통계 엔드포인트
- [x] `src/api/routes/stats.py` — `GET /api/stats/{dimension}` 드릴다운 엔드포인트
  - [x] 유효 dimension 검증 (age, sex, province, district, occupation, hobby, skill, education, marital, military, family_type, housing)
  - [x] 존재하지 않는 dimension → 400 에러
  - [x] limit 파라미터 (기본 20, 최대 100)
- [x] `src/api/main.py` — stats 라우터 마운트
- [x] `tests/test_stats.py` — 단위 테스트 (5개 테스트 통과)
  - [x] 전체 통계 응답 구조 검증 (8개 분포 항목)
  - [x] ratio 합산 ≈ 1.0 (반올림 허용)
  - [x] 드릴다운 필터 적용 시 filtered_count < total_personas

## Phase 11: 페르소나 검색/필터 엔진 (F5)

> PRD Phase 2 §2.1 — 다중 조건 필터 + 페이지네이션 검색

- [x] `src/graph/search_queries.py` — 동적 Cypher 빌더
  - [x] 필터 조건 AND 결합, 복수 값 내부 OR 결합
  - [x] 지원 필터: province, district, age_min, age_max, age_group, sex, occupation, education_level, hobby, skill, keyword
  - [x] 정렬: sort_by (age, display_name), sort_order (asc, desc)
  - [x] 페이지네이션: SKIP/LIMIT 계산 + 총 건수 COUNT 쿼리
- [x] `src/api/schemas.py` — `SearchResponse`, `SearchResult` Pydantic 모델 추가
- [x] `src/api/routes/search.py` — `GET /api/search` 엔드포인트
  - [x] 필터 없이 호출 → 전체 결과 페이지네이션
  - [x] page_size 최대 100 제한 (초과 시 400 에러)
  - [x] 결과 없음 → `total_count: 0, results: []`
- [x] `src/api/main.py` — search 라우터 마운트
- [x] `tests/test_search.py` — 단위 테스트 (8개 테스트 통과)
  - [x] 단일 필터 (province=서울)
  - [x] 복합 필터 (province + age_group + sex)
  - [x] hobby 필터
  - [x] keyword 텍스트 검색
  - [x] 페이지네이션 정상 동작

## Phase 12: 지식 그래프 서브그래프 시각화 (F9)

> PRD Phase 2 §2.5 — 특정 Person 중심 nodes + edges JSON 반환

- [x] `src/graph/subgraph_queries.py` — 서브그래프 추출 Cypher
  - [x] depth=1: 직접 연결 엔티티 (Hobby, Skill, District, Occupation 등)
  - [x] depth=2: 2-hop (같은 엔티티 공유하는 다른 Person)
  - [x] include_similar 옵션 (SIMILAR_TO 관계 포함/제외)
  - [x] max_nodes 제한 (기본 50, 그래프 폭발 방지)
- [x] `src/api/schemas.py` — `SubgraphResponse`, `GraphNode`, `GraphEdge` Pydantic 모델 추가
- [x] `src/api/routes/graph_viz.py` — `GET /api/graph/subgraph/{uuid}` 엔드포인트
  - [x] 존재하지 않는 UUID → 404 에러
  - [x] depth > 2 → 400 에러
  - [x] 노드 타입별 id 생성 규칙 (person_{uuid}, hobby_{name}, district_{province}-{name})
- [x] `src/api/main.py` — graph_viz 라우터 마운트
- [x] `tests/test_graph_viz.py` — 단위 테스트 (5개 테스트 통과)
  - [x] depth=1 기본 동작
  - [x] depth=2 2-hop 포함
  - [x] include_similar=true SIMILAR_TO 엣지 포함
  - [x] max_nodes 제한 동작

## Phase 13: 크로스 세그먼트 비교 분석 (F8)

> PRD Phase 2 §2.4 — 두 세그먼트 분포 비교 + LLM 해석

- [x] `src/rag/compare_chain.py` — 세그먼트 비교용 LLM 체인
  - [x] 비교 프롬프트 템플릿 (두 그룹 분포 → 한국어 분석 3~5문장)
  - [x] NVIDIA API DeepSeek-V4-Pro 호출
  - [x] LLM 호출 실패 시 graceful degradation (ai_analysis 빈 문자열)
- [x] `src/graph/stats_queries.py` — F6 집계 쿼리 재활용 (세그먼트 필터 적용 버전)
- [x] `src/api/schemas.py` — `SegmentCompareRequest`, `SegmentCompareResponse` Pydantic 모델 추가
  - [x] segment_a, segment_b 필터 구조 (province, district, age_group, sex, education_level, hobby, skill)
  - [x] dimensions 검증 (hobby, occupation, education 등)
  - [x] top_k 파라미터 (기본 10)
- [x] `src/api/routes/compare.py` — `POST /api/compare/segments` 엔드포인트
  - [x] 각 세그먼트 분포 집계
  - [x] common, only_a, only_b 항목 산출
  - [x] LLM 분석 텍스트 생성
  - [x] dimensions에 잘못된 값 → 400 에러
- [x] `src/api/main.py` — compare 라우터 마운트
- [x] `tests/test_compare.py` — 단위 테스트 (3개 테스트 통과)
  - [x] 유효한 두 세그먼트 → 비교 결과 + AI 분석
  - [x] 동일 필터 세그먼트 → only_a, only_b 빈 리스트
  - [x] 결과 0건 세그먼트 → count: 0 + 빈 분포
  - [x] LLM 실패 시 ai_analysis 빈 문자열

## Phase 14: Phase 2 통합 테스트 및 Streamlit 확장

- [x] 5개 신규 엔드포인트 Swagger UI 일괄 확인
- [ ] Streamlit UI에 신규 기능 탭 추가
  - [ ] 검색/필터 UI (다중 조건 입력 + 결과 테이블)
  - [ ] 대시보드 UI (차트/그래프 시각화)
  - [ ] 프로필 상세 UI (클릭 → 전체 프로필 표시)
  - [ ] 세그먼트 비교 UI (두 그룹 선택 → 비교 차트 + AI 해석)
  - [ ] 그래프 시각화 UI (인터랙티브 네트워크 그래프)
- [ ] Phase 2 엔드투엔드 테스트 (검색 → 프로필 클릭 → 그래프 탐색 전체 흐름)
- [ ] 성능 테스트 (검색/필터 응답 < 2초, 대시보드 부하 테스트)
