# ADR-004: 챗봇 FilterState는 명시적 스키마와 병합 규칙으로 관리

**상태**: Accepted  
**날짜**: 2026-04-28  
**결정자**: Metis (Pre-planning) + 개발팀  
**관련 PRD**: Feature 12 (대화형 탐색 챗봇)

---

## 문제 (Context)

Feature 12 챗봇은 사용자가 여러 턴에 걸쳐 말한 조건을 누적해야 합니다.

예시:

1. "서울 사람 보여줘" → `province=서울`
2. "그중에서 20대만" → `province=서울 AND age_group=20대`
3. "남성으로 좁혀줘" → `province=서울 AND age_group=20대 AND sex=남자`

필터 상태를 자유 텍스트나 LLM 히스토리에만 의존하면 다음 문제가 생깁니다.

- "그중에서" 같은 지시어가 기존 필터를 유지해야 하는지 모호함
- "대신 부산"처럼 일부 필터만 교체해야 하는 경우 충돌 가능
- 검색/통계/비교/추천 API가 기대하는 파라미터와 챗봇 내부 상태가 어긋날 수 있음
- 오래된 대화 히스토리를 5턴으로 제한해도 핵심 검색 조건은 유지되어야 함

---

## 고려사항 (Considered Options)

### Option A: 자연어 히스토리만 사용

```python
history = [{"role": "user", "content": "서울 사람 보여줘"}, ...]
```

- ✅ 구현이 단순함
- ❌ 필터 충돌/교체/초기화 규칙이 불명확함
- ❌ 최근 5턴 제한과 결합하면 오래된 필터가 사라질 수 있음

### Option B: 명시적 `FilterState` 유지 (채택)

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
```

- ✅ API 파라미터와 1:1 매핑 가능
- ✅ 최근 5턴 히스토리 제한과 독립적으로 필터 유지 가능
- ✅ 리셋/교체/누적 규칙을 테스트 가능
- ❌ 필터 추출/정규화 로직이 별도로 필요

### Option C: LLM이 매 턴 전체 Cypher를 직접 생성

- ✅ 복잡한 질의를 한 번에 처리 가능
- ❌ Cypher 안전성/재현성/테스트 난이도 증가
- ❌ 기존 검색·통계 API 재사용성이 낮아짐

---

## 결정 (Decision)

**Option B: 명시적 `FilterState`를 채택합니다.**

챗봇은 LLM 히스토리와 별도로 `current_filters`를 유지하며, 각 턴에서 추출한 필터를 규칙 기반으로 병합합니다.

---

## FilterState 스키마

Phase 3 MVP의 필터 키는 기존 검색/통계 API에서 이미 지원하는 필드만 허용합니다.

| 필드 | 타입 | 예시 | 매핑 대상 |
|---|---|---|---|
| `province` | `str | None` | `서울` | `/api/search?province=서울` |
| `district` | `str | None` | `서초구` | `/api/search?district=서초구` |
| `age_group` | `str | None` | `20대` | `/api/search?age_group=20대` |
| `sex` | `str | None` | `남자` | `/api/search?sex=남자` |
| `occupation` | `str | None` | `개발자` | `/api/search?occupation=개발자` |
| `education_level` | `str | None` | `대졸` | `/api/search?education_level=대졸` |
| `hobby` | `str | None` | `등산` | `/api/search?hobby=등산` |
| `skill` | `str | None` | `Python` | `/api/search?skill=Python` |
| `keyword` | `str | None` | `창업` | `/api/search?keyword=창업` |

MVP에서는 각 필드를 단일 값으로 유지합니다. 다중 선택(`서울 또는 부산`)은 Phase 3 P1 이후 `list[str]` 확장으로 검토합니다.

---

## 병합 규칙

### 1. 누적

사용자가 "그중에서", "거기서", "추가로", "그리고"를 사용하면 기존 필터를 유지하고 새 필터만 추가합니다.

```python
current = {"province": "서울"}
extracted = {"age_group": "20대"}
merged = {"province": "서울", "age_group": "20대"}
```

### 2. 동일 필드 교체

새 턴에서 동일 필드 값이 명확히 등장하면 해당 필드만 교체합니다.

```python
current = {"province": "서울", "age_group": "20대"}
extracted = {"province": "부산"}
merged = {"province": "부산", "age_group": "20대"}
```

### 3. 명시적 대체

"대신", "말고", "아니고"가 있으면 해당 발화에서 언급된 필드를 교체합니다.

예: "서울 말고 부산" → `province=부산`

### 4. 리셋

사용자가 "리셋", "처음부터", "초기화"라고 말하거나 UI의 필터 초기화 버튼을 누르면 `current_filters`를 빈 객체로 되돌립니다.

```python
merged = {}
last_intent = "reset"
```

### 5. 충돌 처리

한 턴에 같은 필드가 여러 값으로 추출되면 다음 순서를 따릅니다.

1. 명시적 대체어가 있으면 마지막 값을 사용
2. 대체어가 없고 다중값 표현이면 MVP에서는 사용자에게 재질문
3. 재질문 문구: "서울과 부산 중 어느 지역으로 좁힐까요?"

---

## ChatState와의 관계

```python
class ChatState(TypedDict):
    history: list[dict[str, str]]      # 최근 5턴만 유지 (ADR-003)
    current_filters: FilterState       # 히스토리와 독립적으로 유지
    last_intent: str | None            # search / compare / stats / recommend / influence / reset / general
    turn_count: int
```

`history`는 LLM 맥락용이고, `current_filters`는 API 호출용입니다. 5턴을 넘어 히스토리가 잘려도 `current_filters`는 유지됩니다.

---

## API 응답 계약 영향

`POST /api/chat` 응답은 항상 현재 필터 상태를 포함해야 합니다.

```json
{
  "response": "서울 20대 남성 페르소나를 찾았습니다.",
  "context_filters": {
    "province": "서울",
    "age_group": "20대",
    "sex": "남자"
  },
  "sources": [],
  "turn_count": 3
}
```

프론트엔드는 `context_filters`를 사이드바나 채팅창 상단에 칩 형태로 표시합니다.

---

## 테스트 기준

F12 코드 구현 시 최소 테스트는 다음을 포함해야 합니다.

- 3턴 연속 필터 유지: `서울` → `20대` → `남자`
- 동일 필드 교체: `서울` → `부산으로 바꿔줘`
- 리셋 발화: `처음부터` → `{}`
- 충돌 재질문: `서울이랑 부산 둘 다` → clarification 필요
- 세션 분리: 같은 발화라도 `session_id`별 `current_filters` 독립

---

## 영향 (Consequences)

- 검색/통계/추천/영향력 API와 챗봇 상태의 매핑이 명확해집니다.
- LLM이 매번 모든 맥락을 기억하지 않아도 필터 조건은 안정적으로 유지됩니다.
- MVP 범위에서는 단일값 필터만 지원하므로 복잡한 OR 조건은 후속 단계로 남습니다.

---

## 관련 문서

- `PRD.md` §5 Feature 12
- `TASKS.md` Phase 17
- `docs/decisions/ADR-003-chatbot-memory.md`
