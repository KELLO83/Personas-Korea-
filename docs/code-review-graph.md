# code-review-graph 사용법

이 문서는 `Nemotron-Personas-Korea` 저장소에서 `code-review-graph` MCP를 실제로 어떻게 써야 하는지 정리한 운영 가이드입니다.

이 프로젝트는 코드 규모가 크고, 특히 `GNN_Neural_Network/artifacts/experiments/` 아래에 대형 실험 산출물이 많습니다. 그래서 broad query를 무심코 호출하면 타임아웃이 날 수 있습니다. 이 문서의 목적은 **그래프를 먼저 사용하되, 이 저장소에서 타임아웃 없이 안정적으로 쓰는 방법**을 정리하는 것입니다.

---

## 기본 원칙

`AGENTS.md` 기준으로 이 저장소에서는 **코드 탐색 전에 항상 `code-review-graph`를 먼저 사용**해야 합니다.

- 코드 탐색: `semantic_search_nodes` 또는 `query_graph`
- 영향 범위 확인: `get_impact_radius`
- 변경 리뷰: `detect_changes`
- 실행 흐름 확인: `get_affected_flows`
- 테스트 연결 확인: `query_graph(pattern="tests_for")`

다만 다음 경우에는 파일 기반 탐색으로 우회해도 됩니다.

- graph tool이 반복적으로 타임아웃 나는 경우
- 이번 작업이 코드 관계 분석보다 문서 상태 확인이 더 중요한 경우
- broad graph query 대신 직접 파일 몇 개를 읽는 편이 더 빠르고 정확한 경우

즉, 원칙은 **graph first**이고, 예외는 **graph가 현재 질문을 실용적으로 처리하지 못할 때**입니다.

---

## 이 저장소에서 타임아웃이 자주 나는 이유

이 저장소는 그래프 규모가 크고, Git diff 범위도 쉽게 커집니다.

- 그래프 크기: 파일 수와 노드 수가 큰 편이라 broad query 비용이 큼
- GNN 실험 산출물: `artifacts/experiments/**` 아래 CSV, 모델 텍스트, 메트릭 JSON이 많음
- Git 변경 범위: `HEAD~1` 기준 diff가 대형 generated file까지 포함하면 `detect_changes`나 `get_minimal_context`가 느려질 수 있음
- 그래프 stale 상태: 자동 갱신이 밀렸으면 현재 변경과 그래프 상태가 어긋날 수 있음

특히 다음 파일들은 review/change analysis 범위를 크게 키울 수 있습니다.

- `GNN_Neural_Network/artifacts/experiments/**/*.csv`
- `GNN_Neural_Network/artifacts/experiments/**/ranker_model.txt`
- 대형 실험 결과 JSON 및 summary 파일

---

## 가장 중요한 운영 규칙

### 1. broad call보다 targeted call을 우선합니다

좋은 예:

```text
detect_changes(base="HEAD~1", changed_files=["src/api/main.py"], detail_level="minimal")
```

나쁜 예:

```text
get_minimal_context(task="review everything changed in GNN", base="HEAD~1")
```

### 2. `changed_files`를 직접 넣어 범위를 줄입니다

이 저장소에서는 `base="HEAD~1"`만 주고 자동 diff를 태우면, 실험 산출물까지 전부 포함되어 느려질 수 있습니다.

가능하면 다음처럼 **실제로 보고 싶은 파일만 명시**합니다.

```text
changed_files=[
  "GNN_Neural_Network/PRD.md",
  "GNN_Neural_Network/TASKS.md",
  "GNN_Neural_Network/artifacts/experiment_decisions.json",
  "GNN_Neural_Network/artifacts/experiment_run_summary.md"
]
```

### 3. detail level은 항상 최소로 시작합니다

기본값은 다음처럼 생각하면 됩니다.

- 1차 호출: `detail_level="minimal"`
- 정말 더 필요할 때만: `detail_level="standard"`
- 특정 함수/파일만 깊게 볼 때만 추가 확장

### 4. generated artifact는 review 범위에서 빼는 것이 좋습니다

특히 아래 경로는 broad graph review에서 제외하는 쪽이 안전합니다.

- `GNN_Neural_Network/artifacts/experiments/**`
- `**/*.csv`
- `**/ranker_model.txt`

이 파일들은 코드 관계를 이해하는 데 비해 토큰과 처리 시간을 크게 잡아먹습니다.

---

## 추천 사용 순서

## 1) 코드 탐색

특정 기능이나 심볼을 찾고 싶을 때:

```text
semantic_search_nodes(query="recommendation ranking", kind="Function", detail_level="minimal")
```

혹은 특정 파일/심볼 관계를 확인할 때:

```text
query_graph(pattern="file_summary", target="GNN_Neural_Network/scripts/evaluate_ranker.py", detail_level="minimal")
query_graph(pattern="callers_of", target="some_function_name", detail_level="minimal")
```

추천 상황:

- 함수가 어디 있는지 찾고 싶을 때
- 어떤 함수가 누구를 호출하는지 보고 싶을 때
- 테스트가 연결되어 있는지 보고 싶을 때

---

## 2) Git 변경 리뷰

이 저장소에서 가장 기본적인 Git 리뷰 흐름은 다음입니다.

### 최소 리뷰 흐름

```text
detect_changes(base="HEAD~1", changed_files=[...], detail_level="minimal")
```

그 다음 필요 시:

```text
get_affected_flows(base="HEAD~1", changed_files=[...])
query_graph(pattern="tests_for", target="<changed file or function>", detail_level="minimal")
```

### 예시: 문서/결정 파일만 검토할 때

```text
detect_changes(
  base="HEAD~1",
  changed_files=[
    "GNN_Neural_Network/PRD.md",
    "GNN_Neural_Network/TASKS.md",
    "GNN_Neural_Network/artifacts/experiment_decisions.json",
    "GNN_Neural_Network/artifacts/experiment_run_summary.md"
  ],
  detail_level="minimal"
)
```

### 예시: 특정 코드 파일 영향 확인

```text
detect_changes(
  base="HEAD~1",
  changed_files=["GNN_Neural_Network/scripts/evaluate_ranker.py"],
  detail_level="minimal"
)

query_graph(
  pattern="tests_for",
  target="GNN_Neural_Network/scripts/evaluate_ranker.py",
  detail_level="minimal"
)
```

---

## 3) 영향 범위 분석

변경이 다른 모듈에 어떤 영향을 주는지 보고 싶을 때:

```text
get_impact_radius(
  changed_files=["src/api/main.py"],
  base="HEAD~1",
  max_depth=2,
  detail_level="minimal"
)
```

실행 흐름까지 보고 싶으면:

```text
get_affected_flows(
  changed_files=["src/api/main.py"],
  base="HEAD~1"
)
```

추천 상황:

- API 변경이 어떤 경로에 영향을 주는지 보고 싶을 때
- 추천 엔진 변경이 어느 실행 흐름에 들어가는지 보고 싶을 때

---

## 4) 테스트 연결 확인

변경한 코드에 연결된 테스트가 있는지 먼저 확인할 수 있습니다.

```text
query_graph(
  pattern="tests_for",
  target="GNN_Neural_Network/scripts/evaluate_ranker.py",
  detail_level="minimal"
)
```

특정 함수의 호출자까지 보고 싶으면:

```text
query_graph(pattern="callers_of", target="evaluate_ranker", detail_level="minimal")
query_graph(pattern="callees_of", target="evaluate_ranker", detail_level="minimal")
```

---

## 타임아웃이 날 때의 대응 순서

이 저장소에서는 아래 순서로 대응하는 것이 가장 안전합니다.

### 1. 같은 broad call을 반복하지 않습니다

예를 들어 `get_minimal_context(task="GNN 전체 리뷰")`가 타임아웃 났다면, 같은 형태로 다시 크게 호출하지 않습니다.

### 2. 범위를 바로 줄입니다

- `base="HEAD~1"` 자동 diff만 쓰지 않기
- `changed_files`를 직접 넣기
- generated artifact 제외하기
- `detail_level="minimal"` 유지하기

### 3. 그래도 느리면 도구를 바꿉니다

예:

- broad review → `detect_changes`로 축소
- change analysis → `query_graph(pattern="tests_for")`처럼 targeted query로 전환
- 문서 상태 점검 → `Read/Grep`로 우회

### 4. 문서 분석 작업이면 graph를 과하게 고집하지 않습니다

예를 들어 다음 작업은 graph보다 직접 읽기가 더 적합할 수 있습니다.

- `PRD.md` 완료 상태 확인
- `TASKS.md` 남은 작업 확인
- `experiment_decisions.json` 결정 상태 확인
- `experiment_run_summary.md` 최근 실험 요약 확인

---

## 그래프 업데이트 방법

이 저장소의 로컬 지침상 그래프는 **파일 변경 시 hook으로 자동 업데이트**됩니다.

즉, 일반적인 경우에는 별도 수동 작업 없이 최신 상태가 유지되는 것이 정상입니다.

하지만 다음 경우에는 수동 업데이트가 필요할 수 있습니다.

- graph 결과가 최근 변경을 반영하지 않는 것 같을 때
- stale 상태로 보일 때
- 대규모 파일 이동/리팩터링 후 결과가 이상할 때
- 다른 환경에서 graph DB를 새로 준비해야 할 때

### 증분 업데이트

가장 먼저 시도할 수 있는 방법입니다.

```text
code-review-graph_build_or_update_graph_tool(
  full_rebuild=false,
  base="HEAD~1",
  postprocess="minimal"
)
```

설명:

- `full_rebuild=false`: 바뀐 부분만 반영
- `base="HEAD~1"`: 최근 변경 기준으로 증분 계산
- `postprocess="minimal"`: 비용을 줄여 빠르게 갱신

### 전체 재빌드

그래프가 많이 꼬였거나, 증분 업데이트로 해결되지 않을 때 사용합니다.

```text
code-review-graph_build_or_update_graph_tool(
  full_rebuild=true,
  postprocess="minimal"
)
```

전체 재빌드는 비용이 크므로 자주 쓰기보다 필요할 때만 사용하는 것이 좋습니다.

---

## 이 저장소에서 추천하는 실전 패턴

### 패턴 A: 일반 코드 리뷰

```text
detect_changes(base="HEAD~1", changed_files=["src/api/main.py"], detail_level="minimal")
get_affected_flows(changed_files=["src/api/main.py"], base="HEAD~1")
query_graph(pattern="tests_for", target="src/api/main.py", detail_level="minimal")
```

### 패턴 B: GNN 문서/실험 상태 점검

```text
detect_changes(
  base="HEAD~1",
  changed_files=[
    "GNN_Neural_Network/PRD.md",
    "GNN_Neural_Network/TASKS.md",
    "GNN_Neural_Network/artifacts/experiment_decisions.json",
    "GNN_Neural_Network/artifacts/experiment_run_summary.md"
  ],
  detail_level="minimal"
)
```

그래도 타임아웃이 나면 바로 `Read/Grep`로 전환합니다.

### 패턴 C: 특정 함수 영향 확인

```text
query_graph(pattern="callers_of", target="evaluate_ranker", detail_level="minimal")
query_graph(pattern="callees_of", target="evaluate_ranker", detail_level="minimal")
query_graph(pattern="tests_for", target="evaluate_ranker", detail_level="minimal")
```

---

## 하지 말아야 할 것

- broad graph call을 반복 재시도하기
- 대형 generated artifact까지 포함한 상태로 `HEAD~1` 전체 리뷰 돌리기
- 처음부터 `detail_level="standard"` 이상으로 크게 시작하기
- 문서 상태 점검 작업인데 graph를 끝까지 고집하기
- graph가 stale한데 결과만 믿고 진행하기

---

## 빠른 치트시트

### 코드 탐색

```text
semantic_search_nodes(query="keyword", kind="Function", detail_level="minimal")
query_graph(pattern="file_summary", target="path/to/file.py", detail_level="minimal")
```

### Git 변경 리뷰

```text
detect_changes(base="HEAD~1", changed_files=[...], detail_level="minimal")
```

### 영향 확인

```text
get_impact_radius(changed_files=[...], base="HEAD~1", detail_level="minimal")
get_affected_flows(changed_files=[...], base="HEAD~1")
```

### 테스트 연결

```text
query_graph(pattern="tests_for", target="...", detail_level="minimal")
```

### 그래프 갱신

```text
code-review-graph_build_or_update_graph_tool(full_rebuild=false, base="HEAD~1", postprocess="minimal")
code-review-graph_build_or_update_graph_tool(full_rebuild=true, postprocess="minimal")
```

---

## 정리

이 저장소에서 `code-review-graph`는 매우 유용하지만, **크게 던지는 broad query보다 작고 명시적인 query가 훨씬 안정적**입니다.

가장 중요한 것은 다음 세 가지입니다.

1. 먼저 graph를 사용한다.
2. 반드시 범위를 줄인다.
3. graph가 반복적으로 실패하면 문서/파일 직접 읽기로 빠르게 전환한다.

이 원칙만 지켜도 대부분의 타임아웃과 불필요한 토큰 낭비를 크게 줄일 수 있습니다.
