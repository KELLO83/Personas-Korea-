# Phase 19: F13 대규모 운영검증 사전 검수 계획

> 기준 문서: `PRD.md` §11.1~11.3, §10.3, `TASKS.md` Phase 19
>
> 목적: 코드 변경 전 운영전환 위험을 최소화하기 위한 검수 산출물을 단일 위치에 정리합니다.

## 1) 검수 승인 범위

- 대상: `F10` 영향력 API + `F11` 추천 API + `F12` 챗봇 API + Next.js 운영 화면 흐름
- 데이터 규모: 1M Person 기준(또는 운영환경 동등 샘플)
- 환경 고정: Neo4j 5.x + GDS, Python/.venv, 운영 OS, CPU/GPU, JVM 메모리
- 제외: F16~F18 신규 기능의 기본 구현(이미 Phase 22에서 연동 범위 확정)

## 2) 운영 검수 환경 명세 (작성 항목)

문서화할 항목:

- OS / CPU / RAM / Disk
- Neo4j 버전, Heap/PageCache (`server.memory.heap.max_size`, `server.memory.pagecache.size`)
- GDS 버전 및 plugin 로드 상태
- Python, CUDA, PyTorch, LangChain, FastAPI 버전
- 샘플 크기 / 데이터셋 식별자 (`HF_DATASET_ID`, `DATA_SAMPLE_SIZE`)
- 스케줄러: cron 또는 Windows Task Scheduler 사용 여부

예시:

```text
OS: Windows 11
GPU: NVIDIA RTX 40xx (CUDA 12.8)
Python: .venv 3.11.x
Neo4j: 5.x, server.memory.heap.max_size=32G, server.memory.pagecache.size=16G
GDS: 2.x (GDS procedure available)
```

## 3) 데이터 준비 검증 체크리스트

다음 커맨드/쿼리로 사전 점검합니다.

### 3.1 핵심 엔티티 수량

```cypher
MATCH (p:Person) RETURN count(p) AS person_count;
MATCH ()-[r:LIVES_IN]-() RETURN count(r) AS lives_in_count;
MATCH ()-[r:WORKS_AS]-() RETURN count(r) AS works_as_count;
MATCH ()-[r:ENJOYS_HOBBY]-() RETURN count(r) AS enjoys_hobby_count;
MATCH ()-[r:HAS_SKILL]-() RETURN count(r) AS has_skill_count;
MATCH ()-[r:SIMILAR_TO]-() RETURN count(r) AS similar_to_count;
```

### 3.2 중심성 속성 커버리지

```cypher
MATCH (p:Person) WITH count(p) AS total,
    count{(p:Person WHERE p.pagerank IS NOT NULL)} AS pagerank,
    count{(p:Person WHERE p.degree IS NOT NULL)} AS degree,
    count{(p:Person WHERE p.betweenness IS NOT NULL)} AS betweenness
RETURN total, pagerank, degree, betweenness;
```

허용 기준(안전 기본값):

- `pagerank / degree` 가용률 >= 95%
- `betweenness` 가용률 >= 90% (샘플링 실행/재시도 허용)

### 3.3 상태 노드/속성 노출 확인

```cypher
MATCH (s:SystemStatus {key: 'centrality_batch'})
RETURN s.last_success_at, s.status, s.run_id, s.metrics, s.fail_reason;

MATCH (s:SystemStatus {key: 'knn_refresh'})
RETURN s.last_success_at, s.status, s.run_id, s.metrics, s.fail_reason;
```

## 4) 배치 성능 검수 절차 (F13.2)

1. 운영 환경 고정 후 `src/jobs/centrality_batch.py` dry-run이 아닌 실제 실행
2. `--metrics pagerank,degree` 실행: 1M 기준 < 30분 달성 목표
3. `--metrics pagerank,degree,betweenness --betweenness-sampling-size <환경 기준>` 실행: < 2시간 목표
4. 실행 전후 `SystemStatus` 타임라인 기록
5. 실패 시 원인 로그(`status=failed`, `fail_reason`) 보존

실행 예시:

```powershell
# PageRank + Degree
.\.venv\Scripts\python.exe -m src.jobs.centrality_batch --metrics pagerank,degree | Tee-Object -FilePath logs\centrality-pagerank-degree-<timestamp>.log

# Betweenness (RA-Brandes sampling)
.\.venv\Scripts\python.exe -m src.jobs.centrality_batch --metrics pagerank,degree,betweenness --betweenness-sampling-size 10000 | Tee-Object -FilePath logs\centrality-betweenness-<timestamp>.log
```

## 5) API SLA 검수 계획 (F13.4)

목표 SLA:

- `GET /api/influence/top` < 100ms
- `GET /api/recommend/{uuid}` < 500ms
- `POST /api/chat` < 3초 (일반 질의)

측정 방법:

- `tests/test_api_verification.py`의 existing timing assertions 보강
- 대규모 데이터 기준 추가 smoke를 별도 실행 프로필로 기록 (`--host` 환경 고정)
- 통합 응답이 유효한지(필드 존재, 빈 값 fallback, 오류 메시지) 동시에 검증

## 6) 장애 주입 검수 시나리오 (F13.5)

| 장애 | 기대 응답 | 확인 포인트 |
|---|---|---|
| GDS projection 없음 | 503 | 사용자 API는 projection 재생성 트리거하지 않음 |
| SIMILAR_TO 없음 | 503 (F11) | 추천 응답 `error` 메시지, 503 고정 |
| stale 중심성 (run_id 오래됨) | `stale_warning=true` + 기존 결과 노출 | 마지막 성공 run_id 유지 |
| 시뮬레이션 과부하 | 422 | 제거 요청 제한(<=5개/3-hop/10s) 적용 |

각 항목은 수동/자동 스크립트 로그로 재현 가능해야 합니다.

## 7) Streamlit/운영 UX QA 체크 (F13.6)

- 상태 메시지: 빈 값/로딩/에러/스테일 상태 한국어 표시
- F10/F11/F12 기본 화면에서 중간 상태 유지성 확인
- 페이지 전환 또는 rerun 시 필터/선택 상태 유지 정책 확인

## 8) Go/No-Go 기준

- **Go**: 위 항목에서 1회 실행 기준치 모두 충족 + 자동화/수동 시나리오 회귀 없음
- **No-Go**: 배치 SLA 또는 API SLA 미충족, 장애 응답 규약 위반, stale/fallback 동작 불일치 발생
- No-Go 발생 시: 원인-영향-완화 계획 + 재실행 일정 + 승인 필요 여부를 회귀 트래커에 기록

## 9) 검수 승인 산출물

최소 산출물:

- `benchmark log` (PageRank/Degree/Betweenness)
- `API smoke 결과표`
- `운영 runbook`
- `장애 시나리오 검수표`
- `Go/No-Go 결정 기록`

현재는 **검수 계획 버전 1.0**으로 작성되었으며, 이해관계자 승인/서명 전 최종 코드 동결 단계로 진행하지 않습니다.
