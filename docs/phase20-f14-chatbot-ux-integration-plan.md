# Phase 20: F14 확장 및 고도화 (P1) - 챗봇 UX/오케스트레이션 계획

> 근거 문서: `PRD.md` §§ 11.4~11.5, `TASKS.md` Phase 20

본 문서는 구현 전 승인용으로, `Phase 20` 체크리스트의 계획 항목을 실행 가능 형태로 정리한다.

## 1. 통합 챗봇 UX 전환 결정 (P1)

- 기본 UX는 **단일 Chat 인터페이스**를 유지한다.
  - 화면 단일 진입점: `POST /api/chat` 기반 대화형 탐색 중심.
  - 사용자 발화는 기본적으로 “필터 누적 + stats/search 응답 + 자연어 요약” 경로로 처리한다.
- 기존 인사이트 질의는 별도 탭 분리 없이 **챗봇 내부 고급 분석 모드**로 흡수한다.
  - 고급 분석 진입은 `mode` 파라미터 또는 `/분석` 명령어로 트리거한다(기본값 `explore`).
  - `mode=analysis`는 기존 Insight 흐름(`general`/복합 분석)으로 라우팅한다.
- 기존 `/api/insight` 호환성은 유지한다.
  - 단일 대화 전용 경로는 `POST /api/chat`를 우선 사용.
  - 외부 호출 호환 사용자/API가 존재하므로 `/api/insight`는 deprecated 안내 없이 유지한다.
- 분석 모드 응답의 `sources/query_type/history` 저장 방식
  - `response.sources`: 분석 질의 유형(`cypher`, `vector`, `template`, `cache`) + `uuid`/필터 메타
  - `query_type`: `analysis`, `search`, `stats`, `recommend`, `influence`, `profile`
  - `history`: 최근 5턴의 `role/message/intent/context_filters`만 저장 후 요약 정책 적용

## 2. `챗봇 → 추천/영향력` 오케스트레이션 방안

- 공통 기준
  - 인텐트 분류는 규칙 기반 우선.
  - UUID/선택 대상이 없는 경우는 사용자에게 **요구사항 회복형 가이드 메시지** 반환.
  - 추천·영향력 API 실패는 `sources`에 오류 타입(`error=missing_uuid`, `not_ready`, `not_found`)을 남기고 안내.
- 추천 오케스트레이션 (`P1 scope`)
  - trigger: "이 사람에게 추천할 활동은?" 형태의 발화
  - required context: 직전 탐색/프로필에서 선택된 `uuid`(또는 메시지에 직접 uuid 언급)
  - fallback: UUID 미확인 시 "사람을 먼저 선택해 달라" 메시지
- 영향력 오케스트레이션 (`P1 scope`)
  - trigger: "핵심 인물" / "커뮤니티에서 영향력 있는 사람" 계열 발화
  - optional metric: 커뮤니티/metric(예: pagerank) 추론 시도, 미지정시 기본값 적용
  - stale/미준비 상태일 때는 기존 `503/stale` 정책을 채팅 응답으로 전달
- 프로필 인텐트 (`P1 scope`)
  - trigger: "이 사람 상세 보여줘"/"이 UUID의 프로필"
  - 응답은 챗봇 메시지 + `sources.profile` 형태로 저장

## 3. P1/P2 범위 분리

- `P1 포함`: 탐색, stats, profile, recommend, influence 오케스트레이션 및 고급 분석 모드 진입 로직.
- `P2 보류`: LLM 기반 필터 추출, structured output 정규화, 정밀 hallucination 가드레일 강화.

## 4. Phase 20 체크리스트 반영 대상 (문서화)

- 통합 챗봇 UX 전환 항목
  - 진입 UX: 단일 채팅 인터페이스 + 고급 분석 mode
  - 고급 분석 진입 방식: `mode` + `/분석` 명령어(둘 중 한 방식, 우선 `mode`)
  - API 통합 방식: `/api/chat` 내부 라우팅(분석은 내부 InsightRouter/호환 `/api/insight` 사용)
  - 탭 정책: 기존 인사이트 탭은 유지(현재 동작 보존), 단계별로 메인 챗봇 고급 모드로 안내
- 오케스트레이션 P1
  - 추천/영향력/프로필 intent flow와 오류 처리 정책, 범위/제외 기준을 PRD 용어로 문서화

## 5. 운영 전환 시 추적 항목

- 세션 상태: `session_id` 단위 필터 상태(필터 누적), 마지막 intent, 마지막 선택 UUID 유지
- 분석 모드 전환 이력: `analysis` intent 진입/종료/실패 건수
- API fallback 지표: `/api/insight`, `/api/recommend`, `/api/influence/top` 503/422 빈도

## 6. 다음 단계

이 문서가 승인되면 `TASKS.md` Phase 20의 체크리스트 일부(UX 통합, 오케스트레이션 P1 범주)를 완료 처리하고,
`PRD.md §11.5`의 검수 게이트(범위/제외 범위 승인)로 진행한다.

## 7. P2 준비(계획): 자연어 추출/가드레일 강화

- `regex filter extraction` 한계 목록
  - 1턴에 `10대 이상/미만`, `~외` 등 범위 표현을 동시에 주면 충돌 위험
  - 지역/구군 동의어(`서울/서울시/서울특별시`)가 혼재할 때 정규식 매칭 우선순위가 뒤섞임
  - `그중/그건 말고` 등 교체 의도와 추가 의도가 혼재한 발화 처리 불명확
  - 부정 표현(`~제외`, `~빼고`)에서 누락/과대 삭제 위험
  - UUID 언급과 프로필 의도 동시 발화 시 선택 우선순위 충돌
- LLM/Pydantic 적용 기준(안) (P2)
  - Rule 기반이 1차이고, LLM structured output은 추론 보강만 사용
  - 스키마는 `FilterState` enum/nullable 필드 + confidence 최소치 + `reask_reason` 필드를 포함
  - 필터 변경값이 기존 스키마 밖이면 즉시 `reask`로 전환
- Guardrail(안)
  - 필터 추출 값은 허용 enum/known vocabulary와 `None`만 허용
  - `profanity`/의미 모호/오버래핑 발화는 clarification message로 되돌림
  - 각 턴에 추출된 필터와 채택된 의도를 history에 남겨 추적성 확보
- Golden set 평가안 (50+)
  - 대상: 필터 누적/교체/리셋, 추천 intent, 영향력 intent, 오해 유도 발화, 경계 케이스
  - 지표: intent 정확도, `context_filters` 일치율, 재확인율(Clarification rate), 회수율(clarification 이후 성공률)
