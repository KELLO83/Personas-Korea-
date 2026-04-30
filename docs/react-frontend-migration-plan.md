# React Frontend Migration Plan

> Streamlit 기반 Python 프론트에서 React 프론트로 전환했던 구현 계획 및 체크리스트입니다.  
> 현재 기준 원칙: **운영 프론트는 React/Next.js이며, 기존 Streamlit 코드는 이미 삭제되었습니다.**

---

## 1. 목표

- 기존 Streamlit UI에서 사용하던 기능을 React 기반 프론트엔드로 단계적으로 이전한다.
- 기존 FastAPI 백엔드(`src/api`)와 Neo4j/RAG/Graph 로직은 유지한다.
- React가 핵심 기능 parity를 달성할 때까지 FastAPI API 계약과 테스트 결과를 기준선으로 사용한다.
- Streamlit rollback 경로는 더 이상 존재하지 않으며, 필요 시 git history와 API 테스트를 기준으로 회귀 검증한다.

---

## 2. 현재 상태 요약

### 2.1 현재 프론트

- 운영 프론트: `frontend/` 기반 Next.js React
- 기준선: FastAPI API 계약, 테스트, 기존 기능 요구사항 문서
- 참고: 과거 Streamlit UI가 제공하던 탭형 워크플로우를 React 화면으로 재구성하는 것이 목표였다.

### 2.2 기존 Streamlit 상태

- 기존 Streamlit 프론트 코드는 현재 저장소에서 삭제되었다.
- 따라서 이 문서의 Streamlit 언급은 과거 UI 구조를 설명하는 역사적 맥락으로만 읽어야 한다.

### 2.3 기존 API

React 전환 시 우선 재사용할 FastAPI 엔드포인트:

| 기능 | 엔드포인트 | React 화면 |
|---|---|---|
| 통계 대시보드 | `GET /api/stats` | Dashboard |
| 차원별 통계 | `GET /api/stats/{dimension}` | Dashboard Drilldown |
| 검색/필터 | `GET /api/search` | Search |
| 프로필 상세 | `GET /api/persona/{uuid}` | Profile |
| 세그먼트 비교 | `POST /api/compare/segments` | Compare |
| 서브그래프 | `GET /api/graph/subgraph/{uuid}` | Graph Explorer |
| 핵심 인물 | `GET /api/influence/top` | Influence |
| 제거 시뮬레이션 | `POST /api/influence/simulate-removal` | Influence |
| 대화형 탐색 | `POST /api/chat` | Chat |
| 인사이트 질의 | `POST /api/insight` | Insight |
| 유사 페르소나 | `POST /api/similar/{uuid}` | Similar |
| 추천 | `GET /api/recommend/{uuid}` | Recommendation |
| 커뮤니티 | `GET /api/communities` | Communities |
| 관계 경로 | `GET /api/path/{uuid1}/{uuid2}` | Path |

---

## 3. 전환 전략

### 3.1 기본 전략

1. `frontend/` 디렉터리를 새로 만든다.
2. React 앱은 FastAPI를 HTTP API로만 호출한다.
3. React 화면을 하나씩 만들고, API 응답/기능 요구사항/테스트 결과와 비교한다.
4. 핵심 워크플로우가 모두 통과한 뒤에만 기본 프론트를 React로 전환한다.

### 3.2 권장 기술 스택

- React + TypeScript
- Next.js 또는 Vite
  - `toeic_whisper`와 유사한 구조를 원하면 Next.js 기반 `frontend/` 권장
  - 단순 SPA와 빠른 개발을 원하면 Vite 기반도 가능
- Tailwind CSS
- React Query 또는 SWR: API 캐싱, loading/error 상태 관리
- Zustand 또는 Context: `selected_uuid`, 현재 탭/필터/채팅 세션 등 전역 상태 관리
- 그래프 시각화 후보:
  - `vis-network`: 과거 UI의 그래프 표현 방식과 가장 가까움
  - `React Flow`: UI 제어와 React 컴포넌트 통합에 유리
  - `Cytoscape.js`: 그래프 탐색/레이아웃 기능이 강함

### 3.3 toeic_whisper에서 가져올 패턴

- `frontend/` 별도 디렉터리
- Python FastAPI 백엔드와 React 프론트의 포트 분리
- Tailwind 기반 UI 구성
- 사이드바/레이아웃 컴포넌트 분리

### 3.4 toeic_whisper에서 그대로 따라가지 않을 점

- API URL 하드코딩 금지: `http://localhost:8000` 직접 사용 대신 환경변수 기반 API client 사용
- `page.tsx` 단일 대형 파일화 금지: 화면/컴포넌트/훅/API client 분리
- API 타입 수동 추측 금지: FastAPI OpenAPI 또는 `src/api/schemas.py` 기준 타입 정리

---

## 4. 제안 디렉터리 구조

```text
frontend/
  app/ or src/
    routes/ or pages/
    components/
      layout/
      dashboard/
      search/
      profile/
      graph/
      chat/
      common/
    lib/
      api-client.ts
      api-types.ts
      constants.ts
      formatters.ts
    stores/
      persona-selection-store.ts
      chat-store.ts
    styles/
  package.json
  tsconfig.json
  .env.example
```

---

## 5. 단계별 구현 계획

## Phase R0: 기준선 고정


- [x] React 전환 전 기존 Streamlit UI 구조 파악 완료
- [x] React 전환 중 FastAPI API 계약을 기준선으로 사용한다는 원칙 문서화
- [ ] 현재 FastAPI 테스트 전체 통과 확인

## Phase R1: API Contract 정리


- [ ] `/openapi.json` 확인
- [x] `src/api/schemas.py` 기준으로 TypeScript 타입 목록 작성
- [x] 공통 API error shape 정리
- [x] React API client 설계
- [x] `API_BASE_URL` 환경변수 설계
- [ ] CORS origin 정책 점검

필요 시 보강할 API:

- [ ] `GET /api/health`
- [ ] `GET /api/options/provinces`
- [ ] `GET /api/options/districts?province=...`
- [ ] `GET /api/options/occupations?keyword=...`
- [ ] `GET /api/options/hobbies?keyword=...`
- [ ] `GET /api/options/skills?keyword=...`

## Phase R2: React Scaffold 구축


- [x] `frontend/` 생성
- [x] React + TypeScript 프로젝트 초기화
- [x] 기본 스타일 설정
- [x] 기본 레이아웃 구성
- [x] 사이드바/탭 내비게이션 구성
- [x] API client 기본 연결
- [x] loading/error/empty 공통 컴포넌트 작성
- [x] Korean UI text 렌더링 확인

## Phase R3: 읽기 전용 화면 우선 이관


### Dashboard

- [x] `GET /api/stats` 연동
- [x] 총 페르소나 수 카드
- [x] 연령대/성별/지역 분포 차트
- [x] 상위 취미/직업/스킬 목록
- [ ] 기존 요구사항/백엔드 응답과 값 비교

### Search

- [x] `GET /api/search` 연동
- [x] 검색 필터 UI
- [x] 정렬/페이지네이션
- [x] 검색 결과 카드
- [x] “이 사람 선택” 상태 연결
- [x] 빈 결과/에러 상태

### Profile

- [x] `GET /api/persona/{uuid}` 연동
- [x] 기본 정보 카드
- [x] 페르소나 텍스트 표시
- [x] 취미/스킬/직업/지역 표시
- [x] 유사 페르소나 preview
- [ ] 추천 섹션 연결 준비

## Phase R4: 분석 화면 이관


### Compare

- [ ] `POST /api/compare/segments` 연동
- [ ] 그룹 A/B 필터 입력
- [ ] 분포 비교 카드/테이블
- [ ] AI 분석 결과 표시

### Similar

- [ ] `POST /api/similar/{uuid}` 연동
- [ ] top_k 설정
- [ ] 유사도 결과 카드
- [ ] 선택 페르소나 상태 연결

### Recommendation

- [ ] `GET /api/recommend/{uuid}` 연동
- [ ] 카테고리 선택
- [ ] 추천 카드
- [ ] 503 데이터 미준비 상태 처리

### Communities

- [ ] `GET /api/communities` 연동
- [ ] 커뮤니티 카드/요약
- [ ] 대표 페르소나 선택 연결

### Path

- [ ] `GET /api/path/{uuid1}/{uuid2}` 연동
- [ ] 두 UUID 입력
- [ ] 경로 결과 시각화/문장화

## Phase R5: 그래프 시각화 이관


- [x] `GET /api/graph/subgraph/{uuid}` 연동
- [ ] 그래프 라이브러리 최종 선택
- [x] 노드 타입별 색상/크기 매핑 이관
- [x] 관계 타입별 라벨/색상 매핑 이관
- [ ] 노드 타입 필터 UI
- [ ] 관계 요약 테이블
- [ ] 관계 문장 생성
- [ ] 공통점 카드
- [ ] 대형 그래프 loading/performance 확인

Streamlit에서 React로 옮길 주요 함수:

- [ ] `NODE_TYPE_LABELS`
- [ ] `NODE_STYLES`
- [ ] `RELATION_LABELS`
- [ ] `filter_graph_by_types`
- [ ] `relationship_sentence_rows`
- [ ] `commonality_cards`
- [ ] `relation_context_label`

## Phase R6: 채팅/인사이트 이관


### Chat

- [x] `POST /api/chat` 연동
- [x] `session_id` 생성/보관
- [x] 메시지 히스토리 표시
- [x] context_filters 표시
- [x] sources 표시
- [x] loading/error 처리
- [ ] 최근 대화 유지 정책 결정
- [ ] 스트리밍 필요 여부 결정

### Insight

- [ ] `POST /api/insight` 연동
- [ ] 질문 입력/응답 표시
- [ ] sources 표시
- [ ] 장시간 응답 상태 처리

## Phase R7: Parity 검증 및 전환 준비


- [ ] 기존 요구사항 기준으로 화면별 React 결과 비교
- [ ] 주요 API 응답값 일치 확인
- [ ] Korean UI text 검수
- [ ] loading/error/empty 상태 검수
- [ ] 모바일/태블릿 반응형 확인
- [ ] 접근성 기본 확인
- [ ] React build 성공
- [ ] 프론트 lint/typecheck 성공
- [ ] FastAPI 테스트 성공
- [ ] 운영 실행 문서 업데이트
- [ ] React를 기본 프론트로 전환할지 최종 결정

---

## 6. 상태 관리 설계

Streamlit `session_state`에서 React 상태로 이동할 항목:

| Streamlit Key | React 위치 후보 | 설명 |
|---|---|---|
| `selected_uuid` | Zustand/URL param | 전역 선택 페르소나 |
| `selected_persona_label` | Zustand | 현재 선택 표시명 |
| `search_filters` | URL query + local state | 검색 조건 |
| `search_results` | React Query cache | 검색 결과 |
| `profile_uuid` | URL param/local state | 프로필 조회 대상 |
| `graph_uuid` | URL param/local state | 그래프 중심 노드 |
| `graph_data` | React Query cache | 서브그래프 데이터 |
| `graph_profile` | React Query cache | 그래프 중심 프로필 |
| `chat_session_id` | localStorage/Zustand | 백엔드 채팅 세션 키 |
| `chat_messages` | Zustand/localStorage | 채팅 히스토리 |
| `insight_messages` | Zustand/localStorage | 인사이트 질의 히스토리 |
| `similar_uuid` | local state | 유사 페르소나 조회 대상 |
| `path_uuid1`, `path_uuid2` | local state | 관계 경로 입력 |

---

## 7. 검증 기준

각 화면은 완료 처리 전에 아래 기준을 통과해야 한다.

- [ ] 기존 기능 요구사항과 핵심 기능 parity 충족
- [ ] 정상 응답 표시
- [ ] 빈 결과 표시
- [ ] API 오류 표시
- [ ] loading 상태 표시
- [ ] 선택 페르소나 상태 연동
- [ ] Korean UI text 유지
- [ ] TypeScript 타입 오류 없음
- [ ] React build 성공
- [ ] 관련 FastAPI 테스트 성공

---

## 8. 리스크 및 대응

| 리스크 | 영향 | 대응 |
|---|---|---|
| Streamlit 단일 파일에 UI 로직이 많음 | 이관 누락 가능 | 화면별 함수 매핑 후 체크리스트 기반 이관 |
| 그래프 시각화 복잡도 | 일정 증가 | 그래프 화면은 후반 Phase로 배치 |
| API 타입 drift | 런타임 오류 | OpenAPI 또는 Pydantic schema 기반 타입 관리 |
| 채팅 세션 유지 실패 | 대화 맥락 손실 | `session_id`를 localStorage/Zustand에 안정 저장 |
| CORS/환경변수 오류 | 프론트-백엔드 연결 실패 | `.env.example`, 명시적 `API_BASE_URL`, health check 추가 |
| 503/미준비 데이터 | 사용자 혼란 | 추천/중심성 미준비 상태 UI 별도 처리 |
| 일괄 전환 실패 | rollback 어려움 | 화면별 parity 후 전환하고, 필요 시 git history·API 테스트·현재 프론트 검증 결과를 기준으로 회귀 검증 |

---

## 9. 삭제/보관 정책

- 과거 Streamlit UI 코드는 더 이상 저장소에 남아 있지 않다.
- 회귀 검증은 API 테스트, PRD, CHECKLIST, 프론트 구현 결과를 기준으로 수행한다.
- React가 parity에 도달한 이후에도 회귀 검증 기준은 PRD, CHECKLIST, API 테스트, 현재 프론트 구현 결과로 유지한다.
- historical UI 맥락이 필요하면 별도 문서 기록으로만 남기고, 삭제된 프론트 경로를 다시 기준선으로 복원하지 않는다.

---

## 10. 완료 정의

React 전환 완료 조건:

- [ ] React 프론트에서 11개 기존 탭의 핵심 워크플로우를 모두 수행할 수 있다.
- [ ] Streamlit 대비 주요 응답값과 사용 흐름이 일치한다.
- [ ] FastAPI 테스트가 통과한다.
- [ ] React lint/typecheck/build가 통과한다.
- [ ] 그래프/채팅/추천/중심성의 오류 및 미준비 상태가 사용자에게 명확히 표시된다.
- [ ] 실행 문서가 React 기준으로 업데이트된다.
- [ ] git history, API 테스트, 현재 프론트 검증 결과를 기준으로 회귀 검증 가능하다.
