# Changelog

프로젝트 PRD와 구현 계획의 변경 이력을 기록합니다.

---

## 2026-04-28 — PRD v2.1 Operational Readiness

### Added
- `PRD.md`에 Phase 3 운영/오류/롤백 기준 추가
  - 공통 API 오류 계약
  - 관측성 및 운영 상태
  - 데이터 신선도/stale data 정책
  - 마이그레이션 및 롤백 기준
  - UX Empty/Loading/Error 상태 기준
  - 배치 결과 공개 규칙(run_id, 마지막 성공 결과만 노출)
  - 시뮬레이션 connectivity 정의 및 sync-with-hard-bounds 기준
- `CHECKLIST.md`에 PRD/ADR 정렬 항목 보강
  - GDS 중심성 계산은 API 실시간 stream 금지, 배치 write 기준
  - 추천 API는 500ms 목표 및 템플릿 기반 reasoning 기준
  - 챗봇 history 최근 5턴 제한 및 reset 발화 처리
  - F10/F11/F12 UX empty/loading/error/reset 검증 항목

### Changed
- `PRD.md` 버전을 `v2.1-operational-readiness`로 갱신
- `CHECKLIST.md`의 Phase 15~18 성능 기준을 Oracle/Metis 검수 결과에 맞게 조정
- `ADR-001`에 projection 소유권과 외부 스케줄러 원칙 추가

---

## 2026-04-28 — PRD v2.0 Unified

### Added
- `PRD.md`를 Phase 1~3 단일 진실 공급원으로 통합
- `docs/prd-archive/`에 과거 PRD 아카이브 생성
- `docs/decisions/`에 ADR 3건 추가
  - ADR-001: GDS 중심성 계산은 배치로 사전 계산
  - ADR-002: 추천 Reasoning은 템플릿 기반, LLM 동기 호출 금지
  - ADR-003: 챗봇 히스토리는 최근 5턴만 유지

### Removed
- 중복 관리 방지를 위해 루트 `PRD_PHASE2.md`와 `.sisyphus/plans/persona-expansion-prd.md` 제거
