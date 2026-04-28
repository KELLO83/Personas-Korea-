# ADR-001: GDS 중심성 계산은 배치로 사전 계산

**상태**: Accepted  
**날짜**: 2026-04-28  
**결정자**: Oracle (Architecture Review) + 개발팀  
**관련 PRD**: Feature 10 (네트워크 영향력 분석)

---

## 문제 (Context)

Feature 10은 Neo4j GDS의 `pageRank`, `betweenness`, `degree` 중심성 알고리즘을 사용하여 네트워크 내 핵심 인물을 식별해야 합니다.

초기 PRD에는 API 엔드포인트(`/api/influence/top`)에서 `gds.*.stream`을 실시간으로 호출하는 방식이 제안되었습니다.

## 고려사항 (Considered Options)

### Option A: 실시간 GDS 스트림 (초기 PRD 방식)
```cypher
CALL gds.pageRank.stream('personaGraph')
YIELD nodeId, score
RETURN ... ORDER BY score DESC LIMIT 10
```
- ✅ 구현이 단순함
- ❌ 1M 노드에서 응답 시간 5~30초 (SLA 위반)
- ❌ Betweenness의 경우 O(V×E) 복잡도로 타임아웃 확실
- ❌ 동시 API 호출 시 Neo4j CPU/RAM 과부하

### Option B: 사전 계산 + 속성 저장 (채택)
```cypher
// 배치 작업 (매일 새벽)
CALL gds.pageRank.write('personaGraph', {
  writeProperty: 'pagerank'
})

// API에서는 단순 조회
MATCH (p:Person)
WHERE p.pagerank IS NOT NULL
RETURN p.uuid, p.display_name, p.pagerank
ORDER BY p.pagerank DESC LIMIT 10
```
- ✅ API 응답 < 100ms (인덱스 활용)
- ✅ 다중 동시 요청 처리 가능
- ✅ GDS 부하를 API 시간 외로 분리
- ❌ 데이터는 "어제 기준" (지연 허용)

### Option C: 하이브리드 (실시간 + 사전 계산)
- 사전 계산된 값을 먼저 반환
- 요청 시 10분 이내 업데이트된 데이터면 캐시 사용
- 그 이상 지났으면 백그라운드 재계산 트리거
- ❌ 복잡도 높음, Phase 3 MVP에는 과함

## 결정 (Decision)

**Option B (사전 계산)을 채택합니다.**

### 근거
1. **1M 노드 규모**: PageRank는 수 초, Betweenness는 수 분 이상 소요됨
2. **API SLA**: `/api/influence/top`은 < 100ms 응답이 요구됨
3. **네트워크 중심성은 변하지 않음**: 페르소나 데이터가 매일 바뀌지 않음
4. **Neo4j 리소스 보호**: GDS 계산은 단일 스레드 CPU 집약적 작업

## 구현 상세 (Implementation)

### 배치 작업 파일
- `src/gds/centrality.py`: GDS 중심성 계산 서비스와 Cypher 실행 로직
- `src/jobs/centrality_batch.py`: PageRank, Degree, Betweenness 배치 오케스트레이션, 상태 기록, 실패 처리
- 실행: FastAPI/Streamlit 앱 프로세스와 분리된 외부 스케줄러(Windows Task Scheduler, cron, 또는 독립 실행 스크립트)를 우선 사용

### Projection 소유권
- GDS projection 생성/삭제/재생성은 배치 작업이 소유합니다.
- 사용자 API 요청 경로에서는 projection 자동 생성이나 전체 GDS 계산을 수행하지 않습니다.
- API는 마지막 성공 batch 결과만 조회합니다.

### 인덱스
```cypher
CREATE INDEX person_pagerank FOR (p:Person) ON (p.pagerank);
CREATE INDEX person_betweenness FOR (p:Person) ON (p.betweenness);
CREATE INDEX person_degree FOR (p:Person) ON (p.degree);
```

### Betweenness 샘플링
1M 노드에서 정확한 Betweenness는 불가능하므로 RA-Brandes 샘플링 적용:
```cypher
CALL gds.betweenness.sample('personaGraph', {
  samplingSize: 10000,
  samplingSeed: 42,
  writeProperty: 'betweenness'
})
```

## 영향 (Consequences)

- `/api/influence/top` 응답 시간: 5초 → 50ms (100x 개선)
- Neo4j 부하: API 시간대에 GDS 계산 부하 제거
- 데이터 신선도: 최대 24시간 지연 (비즈니스적으로 허용)
- 추가 작업: 배치 스케줄러 구현 필요

## 관련 문서
- PRD v2.0 §3 (Feature 10)
- `src/gds/centrality.py` (구현 예정)
