# 운영/개발 사용 가이드

## 0) 목적

이 문서는 루트 API + 프론트(Next.js) 조합을 기준으로, 사용자에게 공개 가능한 기능 흐름을 빠르게 이해할 수 있도록 정리한다.

## 1) 실행 환경

- Python: `.venv` 가상환경 필수 (`Python 3.11`)
- 백엔드: `.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000`
- 프론트: `cd frontend && npm install && npm run dev`
- API 기본 URL: `NEXT_PUBLIC_API_BASE_URL`

## 2) 핵심 흐름 (사용자 관점)

- **검색/필터**: `GET /api/search` (연령대/성별/지역/직업/취미/기술)
- **통계 확인**: `GET /api/stats` 및 `GET /api/stats/{dimension}`
- **프로필 상세**: `GET /api/persona/{uuid}`
- **채팅 탐색**: `POST /api/chat`
  - 예: "서울 사람 보여줘" → 필터 누적
  - 예: "이 사람에게 추천할 활동은?" → 추천 경로로 전환
  - 예: "이 커뮤니티 핵심 인물은 누구야" → 영향력 경로로 전환
- **추천/영향력/동선 분석**:
  - 추천: `GET /api/recommend/{uuid}`
  - 영향력: `GET /api/influence/top`
  - 경로: `GET /api/path/{uuid1}/{uuid2}`
  - 세그먼트 비교: `POST /api/compare/segments`

## 3) 고급 분석(설계 반영)

- 인사이트 질의는 현재 API 호환성 유지를 위해 `/api/insight`가 남아 있으며,
  챗봇 UX에서는 분석 모드로 흡수되는 방향으로 운영한다.
- 챗봇 응답은 `response`, `context_filters`, `sources`, `turn_count`를 포함한다.
- 분석/오류 이력은 세션 히스토리에 반영되므로, 세션별 맥락은 대화 진행 중 유지된다.

## 4) 상태/예외 처리 규약(요약)

- GDS 미준비: `503` + 사용자 메시지(전체 재계산 안 함)
- SIMILAR_TO 미준비: `503` + 추천 미제공 안내
- 중복 필터/오래된 상태: 마지막 성공 결과+`stale_warning` 안내
- 세션 손상/만료: `400` 메시지와 함께 새 세션 시작 유도

## 5) 점검용 최소 실행 체크리스트

1. 백엔드 실행 후 `/docs`(Swagger)에서 핵심 엔드포인트 확인
2. `tests/test_api_verification.py` 실행으로 응답/시간 Gate 확인
3. 검색/필터 → 선택 UUID → 추천/영향력 흐름 수동 점검
4. 대화형에서 추천/영향력/프로필 intent가 순차적으로 동작하는지 확인
5. 운영 미준비 상태(예: KNN 비활성/중심성 미갱신)에서 `503/stale` 응답이 명시적으로 노출되는지 확인
