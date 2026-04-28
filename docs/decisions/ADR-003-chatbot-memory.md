# ADR-003: 챗봇 히스토리는 최근 5턴만 유지

**상태**: Accepted  
**날짜**: 2026-04-28  
**결정자**: Oracle (Architecture Review) + Metis (Pre-planning) + 개발팀  
**관련 PRD**: Feature 12 (대화형 탐색 챗봇)

---

## 문제 (Context)

Feature 12의 멀티턴 RAG 챗봇은 대화 맥락을 유지하기 위해 이전 대화 기록을 LangGraph state에 저장합니다.

초기 PRD에는 `history` 필드에 모든 대화 기록을 무제한 저장하는 방식이 암묵적으로 제안되었습니다.

## 고려사항 (Considered Options)

### Option A: 무제한 히스토리 (초기 방식)
```python
state["history"] = [...]  # 계속 누적
```
- ✅ 전체 대화 맥락 보존
- ❌ LLM context window 초과 (128K 토큰 한계)
- ❌ 토큰 비용 증가
- ❌ 지연 시간 증가 (매 턴 히스토리 전달)

### Option B: 최근 N턴만 유지 (채택)
```python
state["history"] = state["history"][-5:]  # 최근 5턴만
```
- ✅ context window 안전
- ✅ 토큰 비용 제어
- ✅ 응답 시간 일정
- ❌ 오래된 맥락 유실 ("3턴 전에 뭐라고 했지?")

### Option C: 요약 + 최근 N턴
- 5턴 초과 시 이전 대화를 LLM으로 요약
- 요약문 + 최근 5턴을 함께 전달
- ✅ 맥락 유실 최소화
- ❌ 요약에 추가 LLM 호출 필요 (비용/지연)
- ❌ Phase 3 MVP에는 과함

## 결정 (Decision)

**Option B (최근 5턴)을 채택합니다. 요약 노드는 P1 이후 검토.**

### 근거
1. **핵심 사용 패턴**: "서울 20대 남성 개발자"는 3턴 내에 완료됨
2. **Context Window**: 사용 LLM의 실제 안전마진은 32K 수준으로 가정
3. **응답 시간**: 히스토리가 길어질수록 LLM 처리 시간 증가
4. **필터 상태 별도 관리**: `current_filters`는 히스토리와 별도로 유지되므로 핵심 맥락은 보존

### 예외 처리
- 사용자가 "3턴 전에 뭐라고 했지?"와 같은 질문 → "죄송합니다, 이전 대화는 기억하지 못합니다. 다시 말씀해 주세요."
- 이는 의도적으로 제한한 것이며, 사용자에게 명확히 전달

## 구현 상세 (Implementation)

### LangGraph State
```python
class ChatState(TypedDict):
    history: list[dict]          # 최근 5턴만 (자동 잘림)
    current_filters: FilterState  # 필터는 별도 유지 (히스토리와 무관)
    last_intent: str | None
    turn_count: int              # 전체 턴 수 (히스토리와 별도)
```

### 히스토리 잘림 로직
```python
def add_message(state: ChatState, role: str, content: str):
    state["history"].append({"role": role, "content": content})
    # 최근 5턴만 유지
    if len(state["history"]) > 5:
        state["history"] = state["history"][-5:]
    state["turn_count"] += 1
```

### 프론트엔드 표시
- Streamlit UI에는 최근 10턴을 표시 (백엔드와 별도)
- 백엔드는 5턴만 기억하지만, 프론트는 브라우저 메모리에 더 많이 보관 가능

## 영향 (Consequences)

- Context window 초과: 0건 예상
- LLM 토큰 비용: 턴당 일정 (히스토리 길이에 비례하지 않음)
- 사용자 경험: 5턴 이상의 복잡한 추적은 불가 (의도적 제한)
- 리셋 버튼: "필터 초기화" 버튼으로 언제든 새 대화 시작 가능

## 관련 문서
- PRD v2.0 §5 (Feature 12)
- `src/rag/chat_graph.py` (구현 예정)
- `docs/decisions/ADR-004-chatbot-filter-state.md` (FilterState 스키마 상세, 작성 예정)

## 향후 검토
- Phase 3 P1 이후: 요약 노드 도입 검토 (Option C)
- 사용자 피드백: 5턴 제한이 불편하다는 피드백이 20% 이상 시 N 증가 검토
