# GNN 취미/여가 추천 구현 체크리스트

## Phase G0. 준비

- [ ] 기존 `.venv`에서 PyTorch CUDA 상태 확인
  - [ ] `torch.__version__` 확인
  - [ ] `torch.cuda.is_available()` 확인
  - [ ] GPU 이름 확인
- [ ] `GNN_Neural_Network/` 기본 폴더 구조 생성
  - [ ] `configs/`
  - [ ] `scripts/`
  - [ ] `src/`
  - [ ] `data/`
  - [ ] `artifacts/`
- [ ] artifact/data Git 제외 정책 확인

## Phase G1. 데이터 Export

- [ ] Neo4j 연결 설정 재사용 방식 결정
- [ ] `Person-Hobby` edge export Cypher 작성
- [ ] `scripts/export_person_hobby_edges.py` 구현
- [ ] CSV 저장 경로 결정
- [ ] 중복 edge 제거
- [ ] 최소 연결 수 통계 출력
  - [ ] person 수
  - [ ] hobby 수
  - [ ] edge 수
  - [ ] person당 평균 hobby 수

## Phase G2. 데이터셋 구성

- [ ] `person_uuid -> person_id` mapping 생성
- [ ] `hobby_name -> hobby_id` mapping 생성
- [ ] train/validation/test split 구현
- [ ] user 단위 leakage 방지 검토
- [ ] 이미 보유한 취미 mask 생성
- [ ] negative sampling 구현

## Phase G3. Baseline 추천

- [ ] GNN 전 Cypher/빈도 기반 baseline 정의
- [ ] 유사 페르소나 기반 취미 추천 쿼리 작성
- [ ] baseline Recall@K/NDCG@K 평가 가능하게 구성
- [ ] GNN 성능 비교 기준 저장

## Phase G4. LightGCN 모델

- [ ] PyTorch 기반 LightGCN 모델 구현
- [ ] embedding 초기화 구현
- [ ] graph propagation 구현
- [ ] BPR loss 또는 BCE loss 선택
- [ ] mini-batch 학습 루프 구현
- [ ] CUDA/CPU device 자동 선택
- [ ] seed 고정

## Phase G5. 학습 스크립트

- [ ] `configs/lightgcn_hobby.yaml` 작성
- [ ] `scripts/train_lightgcn.py` 구현
- [ ] epoch별 loss 출력
- [ ] validation metric 출력
- [ ] best checkpoint 저장
- [ ] metrics JSON 저장

## Phase G6. 평가

- [ ] Recall@5/10/20 구현
- [ ] NDCG@5/10/20 구현
- [ ] HitRate@5/10/20 구현
- [ ] 이미 보유한 취미 제외 검증
- [ ] baseline 대비 개선 여부 기록

## Phase G7. CLI 추천

- [ ] `scripts/recommend_for_persona.py` 구현
- [ ] UUID 입력 지원
- [ ] Top-K 입력 지원
- [ ] 추천 취미명/score 출력
- [ ] 알 수 없는 UUID 오류 처리
- [ ] 기존 보유 취미 출력 옵션 검토

## Phase G8. 문서화

- [ ] 실행 방법 README 작성
- [ ] CUDA 확인 명령 추가
- [ ] 데이터 export 명령 추가
- [ ] 학습 명령 추가
- [ ] 추천 명령 추가
- [ ] artifact 설명 추가

## Phase G9. 백엔드 연결 후보

- [ ] CLI 결과가 유효하면 inference API 설계
- [ ] `GET /api/recommendations/gnn/hobbies/{uuid}` 후보 검토
- [ ] 모델 로딩 위치 결정
- [ ] cold start 비용 측정
- [ ] 프론트 프로필 카드 연동 설계

## Acceptance Criteria

- [ ] `.venv` Python으로 모든 GNN 스크립트 실행 가능
- [ ] FastAPI 없이 독립 학습 가능
- [ ] 특정 UUID에 대한 추천 Top-K 출력 가능
- [ ] 학습/평가/추천이 재현 가능
- [ ] 모델 artifact와 mapping 파일 저장 가능
- [ ] 코드 구현 전 PRD와 체크리스트가 최신 상태로 유지됨
