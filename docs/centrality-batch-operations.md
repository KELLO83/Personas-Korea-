# 중심성 배치 운영 절차

이 문서는 F10 네트워크 영향력 분석에 필요한 PageRank, Degree, Betweenness 중심성 점수를 Neo4j에 사전 계산하는 운영 절차를 정리합니다.

중심성 계산은 **FastAPI/Streamlit 요청 경로에서 실행하지 않습니다.** 반드시 앱 프로세스와 분리된 외부 스케줄러(Windows Task Scheduler, cron, 운영 배치 시스템)에서 실행합니다.

## 1) 실행 엔트리포인트

중심성 배치의 실행 파일은 다음입니다.

```powershell
.\.venv\Scripts\python.exe -m src.jobs.centrality_batch
```

옵션:

- `--metrics <list>`: 쉼표로 구분된 중심성 지표. 기본값은 `pagerank,degree`
- `--recreate-projection`: 기존 GDS projection을 삭제하고 다시 생성
- `--betweenness-sampling-size <N>`: Betweenness 샘플링 크기. 기본값은 `10000`

예시:

```powershell
# 매일 실행 권장: PageRank + Degree
.\.venv\Scripts\python.exe -m src.jobs.centrality_batch --metrics pagerank,degree

# Neo4j 재시작 또는 projection 불일치가 의심될 때
.\.venv\Scripts\python.exe -m src.jobs.centrality_batch --metrics pagerank,degree --recreate-projection

# 주 1회 실행 권장: Betweenness 샘플링 포함
.\.venv\Scripts\python.exe -m src.jobs.centrality_batch --metrics pagerank,degree,betweenness --betweenness-sampling-size 10000
```

## 2) 권장 스케줄

| 작업 | 주기 | 명령 |
|---|---|---|
| PageRank + Degree | 매일 02:00 | `python -m src.jobs.centrality_batch --metrics pagerank,degree` |
| Betweenness 샘플링 | 주 1회 일요일 02:00 | `python -m src.jobs.centrality_batch --metrics pagerank,degree,betweenness --betweenness-sampling-size 10000` |
| Projection 재생성 | Neo4j 재시작/점검 후 | 위 명령에 `--recreate-projection` 추가 |

## 3) Windows Task Scheduler 예시

작업 스케줄러에서 새 작업을 만들고 다음 값을 사용합니다.

| 항목 | 값 |
|---|---|
| Program/script | `C:\Users\Kello\Nemotron-Personas-Korea\.venv\Scripts\python.exe` |
| Add arguments | `-m src.jobs.centrality_batch --metrics pagerank,degree` |
| Start in | `C:\Users\Kello\Nemotron-Personas-Korea` |
| Trigger | Daily, 02:00 |

Betweenness 주간 작업은 별도 작업으로 만들고 `Add arguments`만 다음처럼 바꿉니다.

```text
-m src.jobs.centrality_batch --metrics pagerank,degree,betweenness --betweenness-sampling-size 10000
```

## 4) cron 예시

Linux/WSL 환경에서는 프로젝트 루트 기준으로 다음과 같이 등록할 수 있습니다.

```cron
# PageRank + Degree: daily 02:00
0 2 * * * cd /path/to/Nemotron-Personas-Korea && ./.venv/Scripts/python.exe -m src.jobs.centrality_batch --metrics pagerank,degree

# Betweenness: Sunday 02:00
0 2 * * 0 cd /path/to/Nemotron-Personas-Korea && ./.venv/Scripts/python.exe -m src.jobs.centrality_batch --metrics pagerank,degree,betweenness --betweenness-sampling-size 10000
```

운영 OS가 Linux이면 Python 경로를 `./.venv/bin/python`으로 바꿉니다.

## 5) 실행 후 확인 포인트

Neo4j Browser 또는 Python 스크립트에서 다음을 확인합니다.

```cypher
MATCH (p:Person) WHERE p.pagerank IS NOT NULL RETURN count(p);
MATCH (p:Person) WHERE p.degree IS NOT NULL RETURN count(p);
MATCH (p:Person) WHERE p.betweenness IS NOT NULL RETURN count(p);
MATCH (s:SystemStatus {key: 'centrality_batch'})
RETURN s.status, s.run_id, s.last_success_at, s.metrics;
```

API 확인:

```powershell
# 백엔드 실행 후
curl "http://localhost:8000/api/influence/top?metric=pagerank&limit=10"
```

예상 응답 예시:

```json
{
  "metric": "pagerank",
  "last_updated_at": "2026-04-28T02:00:00+00:00",
  "run_id": "centrality-20260428-020000",
  "stale_warning": false,
  "results": []
}
```

정상 상태에서는 `results`가 비어 있지 않고 `last_updated_at`, `run_id`가 포함됩니다.

## 6) 실패 시 처리

- `ServiceUnavailableException` 또는 503 응답: 중심성 점수 미계산 또는 배치 실패 상태입니다.
- Neo4j 재시작 후 projection 오류: `--recreate-projection`을 붙여 다시 실행합니다.
- Betweenness 시간이 길거나 메모리 사용량이 높음: `--betweenness-sampling-size`를 낮추고 주간 배치로만 실행합니다.
- 사용자 API 요청 중에는 배치를 자동 시작하지 않습니다. 운영자가 외부 스케줄러나 수동 명령으로 재실행합니다.

## 7) 관련 문서

- `PRD.md` §3, §7.3
- `docs/decisions/ADR-001-gds-precompute.md`
- `src/jobs/centrality_batch.py`
