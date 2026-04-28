# GNN 취미/여가 추천 시스템 PRD

## 1. 목적

`Nemotron-Personas-Korea` 지식 그래프를 활용해 특정 페르소나에게 어울릴 가능성이 높은 취미/여가활동을 추천하는 GNN 기반 추천 PoC를 구축한다.

초기 목표는 백엔드 API와 분리된 오프라인 학습/추론 스크립트이며, 모델 성능과 운영 가능성을 검증한 뒤 FastAPI inference endpoint로 확장한다.

## 2. 문제 정의

현재 프로젝트는 Neo4j 그래프, GDS 유사도, 통계, 자연어 질의 기반 분석을 제공한다. 하지만 특정 페르소나에게 아직 연결되지 않은 취미를 예측하는 학습 기반 추천은 없다.

GNN PoC는 기존 `Person -> Hobby` 관계를 학습해 다음 질문에 답하는 것을 목표로 한다.

> 이 페르소나가 새롭게 좋아할 가능성이 높은 취미/여가활동은 무엇인가?

## 3. 범위

### In Scope

- 기존 `.venv` Python 환경 사용
- PyTorch 기반 LightGCN PoC
- Neo4j에서 `Person-Hobby` edge export
- train/validation/test split
- negative sampling 기반 학습
- Recall@K, NDCG@K, HitRate@K 평가
- 특정 `persona_uuid`에 대한 취미 Top-K 추천 CLI
- 학습 artifact 저장

### Out of Scope

- 초기 단계 FastAPI endpoint 구현
- 프론트 UI 연결
- GraphSAGE/R-GCN 등 복합 관계 모델
- 실시간 온라인 학습
- LLM 기반 추천 생성
- 사용자 행동 로그 기반 개인화

## 4. 첫 모델 선택

초기 모델은 **LightGCN**으로 한다.

이유:

- `Person-Hobby` bipartite 추천에 적합
- 8GB VRAM 환경에서 PoC 가능성이 높음
- PyTorch만으로 구현 가능해 PyG 설치 리스크가 낮음
- 추천 baseline 대비 성능 비교가 명확함

## 5. 데이터 설계

### 입력 그래프

초기 PoC는 아래 관계만 사용한다.

```text
(Person)-[:ENJOYS_HOBBY]->(Hobby)
```

### Export 데이터

예상 CSV:

```csv
person_uuid,hobby_name
a5ad493e75e74e5cb4a81ac934a1db8f,전국 유명 빵집 투어
a5ad493e75e74e5cb4a81ac934a1db8f,친구들과의 보드게임 모임
```

학습 시 내부적으로 다음 ID mapping을 생성한다.

- `person_uuid -> person_id`
- `hobby_name -> hobby_id`

## 6. 학습 방식

기존 연결 일부를 숨기고 모델이 숨겨진 취미를 맞추도록 학습한다.

```text
positive edge: 실제 Person-Hobby 연결
negative edge: 연결되지 않은 Person-Hobby 조합
objective: BPR loss 또는 BCE loss
```

초기 권장 설정:

```yaml
embedding_dim: 64
num_layers: 2
batch_size: 4096
negative_samples: 1
epochs: 10
top_k: [5, 10, 20]
device: cuda_if_available
```

## 7. 평가 지표

- Recall@10
- NDCG@10
- HitRate@10
- 추천 결과에서 이미 보유한 취미 제외 여부
- 학습 시간 및 GPU 메모리 사용량

## 8. CLI 사용 목표

초기 추론은 CLI로 제공한다.

```powershell
.\.venv\Scripts\python.exe GNN_Neural_Network\scripts\recommend_for_persona.py --uuid a5ad493e75e74e5cb4a81ac934a1db8f --top-k 10
```

예상 출력:

```text
추천 취미 Top 10
1. 성수동 테마 카페 투어 | score=0.842
2. 기구 필라테스 | score=0.817
3. 대학로 소극장 연극 관람 | score=0.791
```

## 9. Artifact

학습 결과는 Git에 포함하지 않는 것을 원칙으로 한다.

예상 산출물:

- `artifacts/lightgcn_hobby.pt`
- `artifacts/person_mapping.json`
- `artifacts/hobby_mapping.json`
- `artifacts/metrics.json`

## 10. 성공 기준

PoC 성공 기준:

- Neo4j에서 Person-Hobby edge export 성공
- LightGCN 학습 스크립트가 `.venv`에서 실행됨
- CUDA 사용 가능 시 GPU 학습 가능
- Recall@10/NDCG@10 산출
- 특정 UUID에 대해 기존 취미를 제외한 추천 Top-K 출력
- FastAPI 없이 독립 CLI로 동작

## 11. 향후 확장

PoC 이후 성능이 유의미하면 다음 단계로 확장한다.

1. Cypher baseline 추천과 성능 비교
2. FastAPI inference endpoint 추가
3. 프로필 상세 화면에 GNN 추천 카드 추가
4. `Person-Skill`, `Person-Occupation`, `Person-Region` 관계를 반영한 GraphSAGE/R-GCN 실험
5. 추천 이유 생성 로직 추가
