# F12 대화형 탐색 챗봇 구현 계획

이 문서는 F12 챗봇 기능을 실제 코드로 구현하기 전, 구현 범위·파일·단계·테스트·검증 기준을 정리한 실행 계획입니다.

F12는 새 분석 알고리즘이 아니라, 기존 검색/통계/프로필/추천/영향력/인사이트 기능을 **대화형으로 조작하는 컨트롤러**입니다.

## 1. 기준 문서

- `PRD.md` §5 Feature 12
- `TASKS.md` Phase 17
- `docs/decisions/ADR-003-chatbot-memory.md`
- `docs/decisions/ADR-004-chatbot-filter-state.md`

## 2. 구현 목표

### MVP 목표

1. 사용자가 채팅으로 검색 조건을 누적할 수 있다.
2. 챗봇은 최근 5턴 히스토리와 별도 `FilterState`를 유지한다.
3. `POST /api/chat`은 항상 `context_filters`, `sources`, `turn_count`를 반환한다.
4. Streamlit은 `st.chat_message`, `st.chat_input`으로 대화 UI를 제공하고 현재 필터를 칩 형태로 보여준다.

### MVP에서 우선 지원할 의도

| Intent | 설명 | 우선순위 |
|---|---|---|
| `search` | 필터 기반 페르소나 검색 | P0 |
| `stats` | 현재 필터 기반 통계/분포 질의 | P0 |
| `reset` | 필터/대화 상태 초기화 | P0 |
| `general` | 범위 밖 일반 안내/도움말 | P0 |
| `profile` | 특정 UUID/선택 페르소나 설명 | P1 |
| `recommend` | 추천 API 연결 | P1 |
| `influence` | 영향력 API 연결 | P1 |
| `compare` | 두 세그먼트 비교 | P2 |

## 3. 아키텍처 결정

### 3.1 ChatState

```python
class FilterState(TypedDict, total=False):
    province: str
    district: str
    age_group: str
    sex: str
    occupation: str
    education_level: str
    hobby: str
    skill: str
    keyword: str


class ChatState(TypedDict):
    session_id: str
    history: list[dict[str, str]]
    current_filters: FilterState
    last_intent: str | None
    turn_count: int
    response: str
    sources: list[dict[str, Any]]
```

### 3.2 히스토리 정책

- 백엔드 state의 `history`는 최근 5턴만 유지한다.
- 프론트엔드 표시용 메시지는 최대 최근 10턴을 표시할 수 있다.
- `current_filters`는 히스토리와 별도로 유지한다.

### 3.3 세션 저장소

MVP는 **in-memory session store**를 사용한다.

이유:

- 현재 프로젝트에 Redis/DB-backed chat session 인프라가 없다.
- 테스트가 쉽고 Phase 3 MVP에 충분하다.
- `session_id`별 상태 분리는 dict 기반으로 검증 가능하다.

후속 단계에서는 Neo4j `:ChatSession` 또는 LangGraph checkpointer를 검토한다.

## 4. 구현 파일 계획

### 4.1 `src/rag/chat_graph.py` 신규

역할:

- `FilterState`, `ChatState` 타입 정의
- `ChatGraph` 클래스 구현
- `classify_intent(message, state)`
- `extract_filters(message)`
- `merge_filters(current, extracted, message)`
- `trim_history(history)`
- `invoke(session_id, message)`

권장 노드 구조:

```text
receive_message
  -> intent_classifier
  -> filter_extractor
  -> filter_merge_or_reset
  -> route_intent
  -> synthesize_response
```

초기 구현은 LLM 의존도를 낮추기 위해 규칙 기반 intent/filter 추출을 우선한다. 이후 필요 시 Pydantic output parser 기반 LLM 추출로 확장한다.

### 4.2 `src/api/schemas.py` 수정

추가 모델:

```python
class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    stream: bool = False


class ChatResponse(BaseModel):
    response: str
    context_filters: dict[str, str] = Field(default_factory=dict)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    turn_count: int = 0
```

### 4.3 `src/api/routes/chat.py` 신규

패턴:

- `src/api/routes/insight.py`처럼 얇은 route wrapper 유지
- `get_chat_graph()` 함수 제공
- `POST /api/chat` 구현
- 예외는 기존 `PersonaKGException` 계열 사용

### 4.4 `src/api/main.py` 수정

- `chat` router import
- `app.include_router(chat.router)` 추가

### 4.5 프론트 연동 메모

현재 저장소에는 Streamlit 프론트가 남아 있지 않으므로, 이 계획의 UI 항목은 **운영 프론트(Next.js) 또는 API 계약 검증 기준**으로 해석한다.

필요 UI 요구사항:

- 대화형 탐색 진입 지점 제공
- 메시지 표시/입력
- 현재 필터 칩 표시
- `필터 초기화` 버튼
- loading state: “답변 생성 중...”
- empty state: “질문을 입력하면 현재 데이터 탐색을 도와드립니다.”

## 5. 의도별 응답 전략

### 5.1 `search` P0

예:

- “서울 사람 보여줘”
- “그중에서 20대만”
- “남성으로 좁혀줘”

동작:

- 필터 추출/병합
- `/api/search`와 동일한 파라미터 구조로 검색 쿼리 실행 또는 기존 검색 query builder 재사용
- 응답은 상위 5명 요약 + `sources`에 검색 조건 포함

### 5.2 `stats` P0

예:

- “이 조건에서 취미 분포 알려줘”
- “서울 20대 남성은 어떤 직업이 많아?”

동작:

- 현재 `current_filters` 사용
- 기존 stats query/service 패턴 재사용
- 분포 top-k를 한국어 문장으로 요약

### 5.3 `reset` P0

예:

- “리셋”
- “처음부터”
- “초기화”

동작:

- `current_filters = {}`
- `last_intent = "reset"`
- 응답: “필터를 초기화했습니다.”

### 5.4 `general` P0

범위 밖 질문에는 프로젝트 기능 안내를 반환한다.

### 5.5 `recommend`, `influence`, `profile` P1

P0가 안정화된 뒤 다음 기능을 연결한다.

- “이 사람에게 추천할 취미는?” → `/api/recommend/{uuid}` 또는 추천 서비스 재사용
- “핵심 인물 보여줘” → `/api/influence/top` 또는 centrality service 재사용
- “이 UUID 설명해줘” → `/api/persona/{uuid}` 또는 persona query 재사용

## 6. 필터 병합 규칙

ADR-004를 그대로 따른다.

1. “그중에서”, “거기서”, “추가로”, “그리고” → 기존 필터 유지 + 새 조건 추가
2. 동일 필드 새 값 → 해당 필드 교체
3. “대신”, “말고”, “아니고” → 명시적 교체
4. “리셋”, “처음부터”, “초기화” → 전체 초기화
5. 한 턴에 같은 필드 다중값이 있고 명확하지 않으면 재질문

## 7. 테스트 계획

### 7.1 `tests/test_chat_graph.py`

필수 테스트:

- `test_filter_retention_across_turns`
  - “서울 사람 보여줘” → “그중에서 20대만” → `province=서울`, `age_group=20대`
- `test_filter_replacement_logic`
  - “서울 사람” → “대신 부산” → `province=부산`
- `test_filter_reset_command`
  - “서울 20대” → “처음부터” → `{}`
- `test_max_history_retention`
  - 6턴 이상 입력 후 history가 최근 5턴만 남는지 확인
- `test_session_isolation`
  - `session_a=서울`, `session_b=부산` 상태 분리 확인

### 7.2 `tests/test_api_chat.py`

필수 테스트:

- `test_chat_endpoint_response_shape`
  - `response`, `context_filters`, `sources`, `turn_count` 포함
- `test_chat_endpoint_keeps_context_by_session`
  - 같은 `session_id`로 연속 호출 시 필터 누적
- `test_chat_endpoint_isolates_sessions`
  - 다른 `session_id`끼리 필터 미공유
- `test_chat_endpoint_reset`
  - reset 발화 시 `context_filters={}`
- `test_chat_endpoint_rejects_empty_message`
  - 빈 message 422

## 8. 검증 명령

모든 Python 명령은 `.venv`를 사용한다.

```powershell
.\.venv\Scripts\python.exe -m py_compile src\rag\chat_graph.py src\api\routes\chat.py src\api\schemas.py
.\.venv\Scripts\python.exe -m pytest tests\test_chat_graph.py tests\test_api_chat.py -q
.\.venv\Scripts\python.exe -m pytest tests -q
```

가능하면 FastAPI 실행 후 smoke test:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

```powershell
curl -X POST http://localhost:8000/api/chat `
  -H "Content-Type: application/json" `
  -d '{"session_id":"demo","message":"서울 20대 남성 보여줘","stream":false}'
```

## 9. 단계별 구현 순서

### Step 1: 스키마 추가

- `ChatRequest`, `ChatResponse` 추가
- 최소 API 계약 테스트 작성

### Step 2: ChatGraph MVP 구현

- in-memory session store
- history 5턴 제한
- FilterState 추출/병합/리셋
- search/stats/general intent 처리

### Step 3: API 라우터 추가

- `src/api/routes/chat.py`
- `src/api/main.py` router mount
- API 테스트 통과

### Step 4: 운영 프론트 UI 연동

- `🤖 대화형 탐색` 탭
- 메시지 표시/input/current filter chips/reset button
- API 연동 및 error/loading/empty state
- 주의: Streamlit 구현은 더 이상 존재하지 않으므로, 이 단계는 운영 프론트 또는 API 계약 검증 기준으로 진행한다.

### Step 5: P1 intent 확장

- profile/recommend/influence 연결
- 각 intent별 source metadata 포함

### Step 6: 최종 검증

- LSP diagnostics
- py_compile
- targeted tests
- full tests
- API smoke
- 필요 시 Oracle 리뷰

## 10. 주요 리스크와 대응

| 리스크 | 대응 |
|---|---|
| intent drift | `last_intent`와 규칙 기반 우선 분류 사용 |
| 필터 추출 오류 | MVP는 명확한 한국어 패턴 우선, 애매하면 재질문 |
| 세션 유실 | MVP는 in-memory임을 명시, 운영 확장 시 persistent store 검토 |
| 기존 API 로직 중복 | search/stats query builder 또는 service 함수 재사용 |
| LLM 지연/장애 | P0는 LLM 없이 규칙 기반 요약 가능하게 설계 |
| history 무한 증가 | `trim_history()` 테스트로 5턴 제한 강제 |

## 11. 완료 기준

- `POST /api/chat`이 PRD 응답 계약을 만족한다.
- 3턴 연속 필터 유지 테스트가 통과한다.
- reset/교체/세션 분리 테스트가 통과한다.
- 운영 프론트 또는 동등한 UI 검증 환경에서 채팅 입력/응답/현재 필터 표시/초기화가 동작한다.
- 전체 테스트가 통과한다.
