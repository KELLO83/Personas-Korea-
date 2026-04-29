# GNN 취미/여가 추천 구현 체크리스트

## Phase G0. 준비

- [ ] 기존 `.venv`에서 PyTorch CUDA 상태 확인
  - [ ] `torch.__version__` 확인
  - [ ] `torch.cuda.is_available()` 확인
  - [ ] GPU 이름 확인
  - [ ] CUDA 미사용 환경에서는 CPU fallback으로 실행되는지 확인
  - [ ] 필요한 패키지 목록을 `requirements-gnn.txt` 또는 README에 기록
  - [ ] `pyyaml`이 설치되어 있거나 `GNN_Neural_Network/requirements-gnn.txt`에 포함되어 있는지 확인
- [ ] GNN/추천 전용 외부 라이브러리 미사용 정책 확인
  - [ ] `torch_geometric` 사용 금지
  - [ ] DGL 사용 금지
  - [ ] TorchRec 사용 금지
  - [ ] RecBole 등 추천 프레임워크 사용 금지
  - [ ] 외부 LightGCN 구현체 복사/의존 금지
- [ ] `GNN_Neural_Network/` 기본 폴더 구조 생성
  - [ ] `configs/`
  - [ ] `scripts/`
  - [ ] `gnn_recommender/`
  - [ ] `data/`
  - [ ] `artifacts/`
- [ ] 중첩 `src/` 폴더를 만들지 않는지 확인
- [ ] `.gitignore`에 아래 항목 추가 전 구현 금지
  - [ ] `GNN_Neural_Network/data/`
  - [ ] `GNN_Neural_Network/artifacts/`

## Phase G1. 데이터 Export

- [ ] Neo4j 연결은 기존 `src.config.settings` 재사용
  - [ ] `NEO4J_URI`
  - [ ] `NEO4J_USER`
  - [ ] `NEO4J_PASSWORD`
  - [ ] `NEO4J_DATABASE`
- [ ] 별도 `.env` parser 또는 hardcoded connection 사용 금지
- [ ] `Person-Hobby` edge export Cypher 작성
  - [ ] `MATCH (p:Person)-[:ENJOYS_HOBBY]->(h:Hobby)` 사용
  - [ ] `p.uuid IS NOT NULL`
  - [ ] `h.name IS NOT NULL`
  - [ ] `RETURN DISTINCT p.uuid AS person_uuid, h.name AS hobby_name`
- [ ] `scripts/export_person_hobby_edges.py` 구현
- [ ] CSV 저장 경로는 `GNN_Neural_Network/data/person_hobby_edges.csv`
- [ ] CSV schema 고정: `person_uuid,hobby_name`
- [ ] UTF-8 encoding 확인
- [ ] 빈 UUID/hobby 제거
- [ ] 중복 edge 제거
- [ ] 최소 데이터셋 크기 검증
  - [ ] edge 0건이면 명확한 오류
  - [ ] 평가 가능한 person 수 출력
- [ ] 최소 연결 수 통계 출력
  - [ ] person 수
  - [ ] hobby 수
  - [ ] edge 수
  - [ ] person당 평균 hobby 수

## Phase G1.5. Offline Boundary

- [ ] G1 export 이후 G2+ 단계는 로컬 파일만 사용
- [ ] G2+ 단계에서 live Neo4j 연결 금지
- [ ] G2+ 단계에서 FastAPI 호출 금지
- [ ] G2+ 단계에서 Streamlit/Next.js 프론트 파일 변경 금지
- [ ] Stage 2용 offline feature export 구현
  - [ ] `GNN_Neural_Network/data/person_context.csv` 저장
  - [ ] `person_context.csv` schema 검증
- [ ] G1 raw export는 split과 무관한 원천 데이터만 저장
- [ ] `SIMILAR_TO`/FastRP 기반 provider는 G1 export artifact로만 G2+에서 사용
- [ ] `similar_person_hobbies.csv`가 train-gated graph에서 생성됐는지 기록
- [ ] train-gated 생성 근거가 없으면 similar-person provider를 offline metric에서 제외
- [ ] live `RecommendationService`/`GET /api/recommend/{uuid}` 비교는 offline metric과 분리

## Phase G2. 데이터셋 구성

- [ ] Vocabulary quality gate 적용
  - [ ] hobby name Unicode 정규화/공백 collapse
  - [ ] optional JSON alias map 적용
  - [ ] alias 적용 후 `(person_uuid, canonical_hobby_name)` dedupe
  - [ ] `min_item_degree` 미만 hobby filtering
  - [ ] `vocabulary_report.json` 저장
  - [ ] raw/canonical/retained hobby 수와 singleton ratio 기록
- [ ] `person_uuid -> person_id` mapping 생성
- [ ] canonical `hobby_name -> hobby_id` mapping 생성
- [ ] mapping 파일 저장 계약 정의
  - [ ] `GNN_Neural_Network/artifacts/person_mapping.json`
  - [ ] `GNN_Neural_Network/artifacts/hobby_mapping.json`
- [ ] train/validation/test split 구현
  - [ ] person별 holdout split
  - [ ] 취미 3개 이상 person은 train/validation/test 분리
  - [ ] 취미 2개 person 처리 정책 config 명시
  - [ ] 취미 1개 person은 train-only 또는 eval 제외
- [ ] validation/test positive edge가 training graph에 들어가지 않는지 검증
- [ ] train-known mask와 full-known mask를 분리
  - [ ] evaluation ranking은 train-known item mask
  - [ ] final recommendation은 full-known item mask
- [ ] split 파일 저장
  - [ ] train_edges.csv
  - [ ] validation_edges.csv
  - [ ] test_edges.csv
- [ ] G2 split 이후 train-only artifact 생성
  - [ ] `hobby_profile.json`은 train split만 사용
  - [ ] co-occurrence/profile 통계는 train split만 사용
  - [ ] `similar_person_hobbies.csv`는 train-gated 생성이 가능할 때만 저장
  - [ ] train-gated 생성이 불가능하면 similar-person provider를 offline metric에서 제외
  - [ ] `similar_person_hobbies.csv` 저장 시 schema 검증

## Phase G3. Baseline 추천

- [ ] Offline popularity baseline 구현
- [ ] Offline co-occurrence baseline 검토
- [ ] GNN 전 CSV/split 기반 baseline 정의
- [ ] baseline Recall@K/NDCG@K 평가 가능하게 구성
- [ ] train/evaluate 결과에 popularity/co-occurrence baseline metrics 같이 저장/출력
- [ ] GNN 성능 비교 기준 저장
- [ ] baseline feature는 train split 기준으로만 계산
- [ ] 기존 graph baseline 비교 항목 기록
  - [ ] `src/graph/recommendation.py` RecommendationService 확인
  - [ ] 기존 `GET /api/recommend/{uuid}` 결과와 비교 가능성 기록
  - [ ] 단, 기존 graph baseline은 live Neo4j 필요하므로 필수 offline baseline과 분리

## Phase G4. LightGCN 모델

- [ ] PyTorch 기반 LightGCN 모델 구현
- [ ] PyG/DGL/TorchRec 없이 순수 PyTorch tensor/autograd 구현
- [ ] PyTorch 기본 도구 사용 허용 범위 확인
  - [ ] `torch.optim.Adam` 허용
  - [ ] `torch.optim.AdamW` config option 허용
  - [ ] `torch.nn.functional.logsigmoid` 허용
  - [ ] `torch.save` / `torch.load` 허용
- [ ] embedding 초기화 구현
- [ ] user/item embedding indexing 규칙 구현
- [ ] sparse normalized adjacency 구성
  - [ ] degree 계산 직접 구현
  - [ ] symmetric normalization 직접 구현
  - [ ] `torch.sparse.mm` propagation 사용
- [ ] graph propagation 구현
- [ ] BPR loss 직접 구현
  - [ ] `torch.nn` loss module 사용 금지
  - [ ] `-mean(logsigmoid(pos_score - neg_score))` 형태 검증
- [ ] negative sampler 구현
  - [ ] full-known positive hobby는 negative에서 제외
  - [ ] validation/test positive hobby는 train negative에서 제외
  - [ ] 초기 negative ratio = 1
- [ ] optimizer 구현 정책
  - [ ] v1 기본은 `torch.optim.Adam`
  - [ ] `AdamW`는 config option으로 허용
  - [ ] gradient zeroing / backward / step 순서 명확화
- [ ] learning-rate scheduler 정책
  - [ ] v1 기본은 constant LR
  - [ ] step decay 또는 cosine decay는 config option으로만 추가
  - [ ] PyTorch 외부 scheduler 구현체 사용 금지
- [ ] mini-batch 학습 루프 구현
- [ ] inference scoring 구현
  - [ ] 메모리 한도 내에서 `person batch x all hobbies` 또는 `selected persons x hobby chunk` 방식 사용
  - [ ] 현재 데이터 크기에서는 person batch matmul 기본
  - [ ] 더 큰 데이터셋에서는 hobby chunk fallback 가능
  - [ ] top-k buffer 또는 `torch.topk`로 메모리 사용 제한
- [ ] checkpoint save/load 계약 구현
- [ ] CUDA/CPU device 자동 선택
- [ ] seed 고정

## Phase G5. 학습 스크립트

- [ ] `configs/lightgcn_hobby.yaml` 작성
- [ ] `scripts/train_lightgcn.py` 구현
- [ ] epoch별 loss 출력
- [ ] validation metric 출력
- [ ] best checkpoint 저장
- [ ] config snapshot 저장
- [ ] metrics JSON 저장
- [ ] CPU 실행 smoke test
- [ ] CUDA-if-available 실행 smoke test

## Phase G6. 평가

- [ ] Recall@5/10/20 구현
- [ ] NDCG@5/10/20 구현
- [ ] HitRate@5/10/20 구현
- [ ] evaluation에서 train-known 취미 제외 검증
- [ ] final recommendation에서 full-known 취미 제외 검증
- [ ] leakage 방지 테스트 추가
  - [ ] validation/test positive가 training graph에 없는지 확인
  - [ ] validation/test positive가 train negative에 없는지 확인
  - [ ] final recommendation이 full-known positive를 추천하지 않는지 확인
- [ ] persona text leakage audit 추가
  - [ ] validation/test positive hobby명이 `persona_text`에 직접 등장하는지 확인
  - [ ] validation/test positive hobby명이 domain persona text에 직접 등장하는지 확인
  - [ ] validation/test positive hobby명이 `hobbies_text`에 직접 등장하는지 확인
  - [ ] validation/test positive hobby명이 `embedding_text`에 직접 등장하는지 확인
  - [ ] audit 결과를 `leakage_audit.json`에 저장
  - [ ] 누수율이 높으면 masking mode 또는 no-text mode로 평가 전환
- [ ] random ranking baseline 대비 Recall@10 개선 여부 기록
- [ ] baseline 대비 개선 여부 기록
- [ ] `metrics.json` 필수 필드 확인
  - [ ] Recall@10
  - [ ] NDCG@10
  - [ ] HitRate@10
  - [ ] num_persons
  - [ ] num_hobbies
  - [ ] num_edges

## Phase G7. CLI 추천

- [ ] `scripts/recommend_for_persona.py` 구현
- [ ] UUID 입력 지원
- [ ] Top-K 입력 지원
- [ ] 추천 취미명/score 출력
- [ ] 알 수 없는 UUID 오류 처리
- [ ] 알 수 없는 UUID는 popularity fallback으로 처리
- [ ] 후보 부족 시 popularity fallback으로 top-k 보강
- [ ] 기존 보유 취미 출력 옵션 검토
- [ ] 추천 결과가 이미 보유한 취미를 포함하지 않는지 검증
- [ ] sample recommendation JSON 저장

## Phase G7.5. Stage 1 Multi-provider Candidate Generation

- [ ] LightGCN을 최종 추천기가 아니라 candidate provider로 분리
- [ ] 공통 Candidate 데이터 계약 정의
  - [ ] `hobby_name`
  - [ ] `source` (`lightgcn`, `cooccurrence`, `similar_person`, `segment_popularity`, `popularity`)
  - [ ] `score`
  - [ ] `source_scores`
  - [ ] `reason_features`
- [ ] LightGCN provider 구현
  - [ ] checkpoint 없음 처리
  - [ ] unknown UUID 처리
  - [ ] full-known hobby 제외
  - [ ] 후보 부족 시 다음 provider로 fallback
- [x] co-occurrence provider 구현
  - [x] gated train split 기반 co-occurrence 후보 생성
  - [x] known hobby 제외
  - [x] source score normalize
- [ ] similar-person provider 구현
  - [ ] G2 split 이후 train-gated 방식으로 생성한 `similar_person_hobbies.csv`만 offline metric에 사용
  - [ ] G2+에서 live Neo4j/FastAPI/RecommendationService 호출 금지
  - [ ] supporting persona/evidence 보존
  - [ ] train-gated graph export 여부를 provider metadata에 기록
  - [ ] leakage 통제 불명확 시 offline metric에서 제외
- [x] popularity provider 구현
  - [x] segment popularity fallback 후보 생성
    - ablation 이후 **default provider chain에서는 제외**하고, 필요 시 명시적 실험/비교에서만 사용
  - [x] global popularity fallback 후보 생성
  - [x] train split 기준 popularity만 사용
- [ ] offline fallback chain 구현
  - [ ] LightGCN (auxiliary / experimental)
  - [ ] co-occurrence (primary)
  - [ ] offline similar-person
  - [ ] segment popularity (disabled by default after toxic ablation)
  - [x] global popularity (ultimate fallback)
- [x] Candidate merge 정책 구현
  - [x] canonical hobby id/name 기준 dedupe
  - [x] provider별 score를 `source_scores`에 병합
  - [x] 동일 hobby가 여러 provider에서 나올 때 reason feature 병합
- [x] provider score normalization 구현
  - [x] raw score와 normalized score를 모두 보존
  - [x] min-max 또는 percentile normalization 중 하나를 config로 고정
  - [x] missing score 처리 규칙 정의
  - [x] normalization 설정을 `score_normalization.json`에 저장
  - [x] normalization 규칙 변경 시 기존 metric과 직접 비교 금지
- [ ] Stage 1 safety gate
  - [ ] 최소 후보 수 확인
  - [x] source별 후보 contribution 기록
  - [ ] 후보 부족 시 명확한 fallback chain 실행
  - [x] candidate sample JSON 저장
  - [x] fallback 사용률을 `fallback_usage.json`에 저장

## Phase G7.5b. Stage 1 Graph Model A/B: LightGCN vs SimGCL/XSimGCL

- [x] SimGCL/XSimGCL은 Stage 2 대체재가 아니라 Stage 1 candidate provider 후보로 문서화
- [ ] LightGCN-only Stage 1 candidate recall@K 저장
- [ ] SimGCL/XSimGCL-only Stage 1 candidate recall@K 저장
- [ ] 두 Stage 1 모델이 같은 split, same `candidate_k`, same known-hobby masking을 사용하는지 검증
- [ ] Stage 2 reranker weight/feature policy를 고정한 상태에서 `LightGCN + Stage2`와 `SimGCL/XSimGCL + Stage2` 비교
- [ ] Stage 1 swap 실험에서는 Stage 2 코드를 변경하지 않음
- [ ] XSimGCL 도입 전 LightGCN candidate recall 병목 여부 확인
- [ ] SimGCL/XSimGCL hyperparameter 기록
  - [ ] contrastive loss weight
  - [ ] temperature
  - [ ] perturbation/noise epsilon
  - [ ] contrastive layer
- [ ] XSimGCL 추가 학습 비용과 GPU 메모리 사용량 기록
- [ ] XSimGCL 결과가 Stage 2 후 최종 NDCG/Recall 개선으로 이어지는지 별도 기록

## Phase G7.5c. Hobby canonicalization / alias / taxonomy gate

- [x] Stage 1 provider ablation 전에 canonicalization gate 완료
- [x] canonicalization candidate mining / review workflow 구축
  - [x] `GNN_Neural_Network/artifacts/canonicalization_candidates.json` 생성
  - [x] cluster별 `canonical_candidate`, `members`, `support_edges`, `confidence`, `display_examples` 저장
  - [x] confidence를 `high`, `medium`, `low`로 구분
  - [x] generic token cluster(`시청`, `감상`, `모임`, `투어`, `체험`)는 기본 reject 또는 split_required 처리
  - [x] `GNN_Neural_Network/configs/hobby_taxonomy_review.json` 승인 파일 설계
  - [x] `approved_clusters`, `manual_aliases`, `rejected_patterns`, `split_required` 필드 확정
  - [x] approved cluster만 canonicalization rule로 승격
  - [x] rejected/split_required cluster는 raw 유지 또는 후속 분리 규칙 설계
- [x] raw hobby -> canonical_hobby alias 설계
- [ ] canonical_hobby taxonomy 설계 (approved cluster에 taxonomy metadata 아직 미기입)
  - [ ] category
  - [ ] subcategory
  - [ ] location_modifier
  - [ ] intensity
  - [ ] sociality
- [x] Candidate / data contract 정리
  - [x] `raw_hobby_name -> canonical_hobby -> hobby_id`
  - [x] raw hobby는 alias/example/display evidence로 보존
  - [x] canonical_hobby만 item vocabulary/split/train/eval의 기본 단위로 사용
  - [x] Candidate에 `canonical_hobby`, `display_examples`, `taxonomy`를 반영
- [x] 대표 예시
  - [x] `석촌호수 주변 산책` -> `산책`
  - [x] `올림픽공원 숲길 산책` -> `산책`
  - [x] `탄천 산책로 걷기` -> `산책`
  - [x] `수락산 둘레길 산책` -> `산책`
  - [x] `한강공원 산책` -> `산책`
- [x] alias/taxonomy 파일 형식 결정
  - [x] alias JSON 또는 CSV schema 확정
  - [x] canonical_hobby별 raw examples 저장 방식 확정
- [x] canonical_hobby 기준 export 재생성
- [x] canonical_hobby 기준 split/train/eval 재생성
- [x] canonical 기준 `vocabulary_report.json` 재검토
  - [x] raw hobby 수 감소 확인 (27,137 → 2,302 canonical)
  - [x] retained canonical hobby 수 확인 (180 retained)
  - [ ] singleton ratio 감소 확인
  - [ ] canonical singleton ratio < raw singleton ratio 0.834 확인
    - strict taxonomy 현재 pre-filter canonical singleton ratio는 0.8484로 raw 0.8340보다 높음
    - 대신 retained graph는 180 hobbies / 40,743 edges로 충분히 조밀하며 avg hobby degree 226.3 유지
  - [ ] canonical 기준 `candidate_recall@50`이 raw 기준보다 악화되지 않는지 확인
  - [x] 잘못된 과잉 병합 예시가 줄었는지 확인
- [x] canonical 기준 Stage 1 provider ablation 재실행
- [x] canonical 기준 Stage 2 / LightGCN / SimGCL / XSimGCL 재평가 전제 조건 확인

## Phase G7.5d. Stage 1 provider ablation / baseline fixing

- [x] Stage 1 provider 단독 성능 측정
  - [x] popularity only: recall@10=0.6913, ndcg@10=0.4346, cr@50=0.9784
  - [x] cooccurrence only: recall@10=0.6932, ndcg@10=0.4370, cr@50=0.9728
  - [x] segment_popularity only: recall@10=0.0038 — **해로움 확인, 기본 조합에서 제외**
  - [x] LightGCN only: recall@10=0.6770, ndcg@10=0.4280, cr@50=0.9674
- [x] Stage 1 provider 조합 성능 측정
  - [x] popularity + cooccurrence: recall@10=**0.6940**, ndcg@10=0.4355, cr@50=0.9776 ← **SELECTED BASELINE**
  - [x] popularity + cooccurrence + lightgcn: recall@10=0.6914, ndcg@10=0.4344, cr@50=0.9771
  - [x] segment_popularity + popularity: recall@10=0.5325 — segment_popularity toxic
  - [x] segment_popularity + cooccurrence: recall@10=0.5361 — segment_popularity toxic
  - [x] segment_popularity + cooccurrence + popularity: recall@10=0.5366 — segment_popularity toxic
  - [x] segment_popularity + cooccurrence + popularity + LightGCN: recall@10=0.5362 — segment_popularity toxic
- [x] validation 기준으로 `recall@10`, `ndcg@10` 고정 비교
- [ ] test split은 validation에서 선택된 단일 조합에 대해서만 1회 사용
- [x] `segment + cooccurrence + popularity` 대비 `+ LightGCN` 개선 여부 별도 기록
  - segment 포함 조합 자체가 toxic이므로 이 비교는 무의미. **segment 제외 결정 확정**
- [x] LightGCN이 validation 개선에 실패하면 Stage 1 기본 조합에서 제외 또는 보조 provider로 격하
  - LightGCN 추가 시 pop+cooc 0.6940 → 0.6914로 소폭 하락. **보조 provider로 격하**
- [x] Stage 1 기준선 확정: `SELECTED_STAGE1_BASELINE = ("popularity", "cooccurrence")`
- [ ] Stage 1 기준선 확정 전 SimGCL/XSimGCL Stage 1 교체 실험 보류

## Phase G7.6. Persona-aware Reranker

- [ ] PersonaContext 데이터 계약 정의
  - [ ] uuid, age, age_group, sex
  - [ ] occupation, district, province
  - [ ] family_type, housing_type, education_level
  - [ ] persona/professional/sports/arts/travel/culinary/family text
  - [ ] hobbies_text, skills_text, career_goals
  - [ ] embedding_text 또는 text embedding reference
  - [ ] known_hobbies
- [ ] HobbyCandidate 데이터 계약 정의
  - [ ] hobby_name
  - [ ] source_scores
  - [ ] category 또는 metadata placeholder
  - [ ] reason_features
- [x] weighted reranker v1 구현 전 feature 정의
  - [x] `lightgcn_score`
  - [x] `cooccurrence_score`
  - [x] `segment_popularity_score`
    - 구현은 존재하지만 default runtime에서는 비활성/0-weight 권장
  - [ ] `similar_person_score`
  - [ ] `persona_text_fit`
  - [x] `known_hobby_compatibility`
  - [x] `age_group_fit`
  - [x] `occupation_lifestyle_fit`
  - [x] `region_accessibility_fit`
  - [x] `segment_or_global_popularity_prior`
  - [x] `mismatch_penalty`
- [x] train-only Stage 2 feature 생성
  - [x] `hobby_profile.json` 생성
  - [x] hobby별 train popularity 저장
  - [x] age_group/sex/occupation/region 분포 저장
  - [x] train co-occurring hobbies 저장
  - [x] known hobby compatibility 통계 저장
- [ ] mismatch penalty 규칙 정의
  - [ ] 나이대 mismatch penalty
  - [ ] 직업/생활패턴 mismatch penalty
  - [ ] 지역 접근성 mismatch penalty
  - [ ] 가족/주거 형태와 어울리지 않는 후보 penalty
  - [ ] hard block이 아니라 downweight 기본 정책
- [x] rerank score 계산 구현
  - [x] feature normalization
  - [x] weight config화
  - [x] `segment_popularity_score` weight 추가
    - 현재 baseline 전략상 기본값은 0 또는 비활성 권장
  - [x] source별 score missing 처리
  - [x] validation split에서만 weight 조정
  - [x] test split 결과를 보고 weight 재조정 금지
  - [x] 기본 offline metric에서 text fit disabled 유지
- [ ] Stage 2 tuning before Stage 1 model swap
  - [x] `segment_popularity_score`를 Stage 2 feature에 추가
  - [x] `segment_popularity_score`가 반영된 `RerankerWeights` 추가
  - [x] validation-only weight sweep / grid search 추가
  - [ ] test split은 고정 설정 1회 평가로만 사용
  - [ ] `mismatch_penalty` ablation (`0`, `default`, `scaled-down`)
  - [ ] Stage 2 feature ablation report 생성
- [ ] `selected Stage1 baseline` 대비 Stage 2 열세 원인 분석
  - [ ] tuned Stage 2가 validation에서 selected Stage1 baseline보다 열세가 아닌지 검증
- [ ] explanation/reason feature 출력
  - [ ] LightGCN 후보 여부
  - [ ] similar persona 근거
  - [ ] demographic/lifestyle fit 근거
  - [ ] known hobby compatibility 근거
- [ ] reranker fallback
  - [ ] PersonaContext 조회 실패 시 Stage 1 score only fallback
  - [ ] text embedding 없음 처리
  - [ ] 후보 전체가 penalty로 낮아질 때 popularity floor 유지

## Phase G7.7. 2-stage 평가

- [x] Stage 1 candidate recall@K 평가
- [x] Stage 1 provider ablation metrics 저장
  - [x] provider-only metrics table
  - [x] provider-combination metrics table
  - [x] delta vs `popularity + cooccurrence` (selected baseline)
  - [x] `GNN_Neural_Network/artifacts/stage1_ablation_validation.json` 저장
  - [ ] `GNN_Neural_Network/artifacts/stage1_ablation_test.json` 저장
  - [x] providers / recall@10 / ndcg@10 / hit_rate@10 / candidate_recall@50 / delta_vs_selected_baseline 저장
- [x] LightGCN-only vs multi-provider candidate 비교
  - LightGCN 0.6770 < pop+cooc 0.6940. LightGCN은 보조 provider
- [ ] LightGCN-only vs persona-aware reranker 비교
- [ ] tuned Stage2 vs untuned Stage2 비교
- [ ] `segment_popularity_score` 포함/제외 비교
- [ ] `mismatch_penalty` 설정별 비교
- [ ] 동일 split/동일 masking/동일 candidate pool 비교 조건 검증
- [ ] candidate pool이 다르면 Stage 1 candidate recall 차이를 별도 보고
- [ ] 기존 `RecommendationService` baseline 비교는 online comparison으로 분리
- [ ] offline similar-person provider leakage audit
  - [ ] `similar_person_hobbies.csv` 생성 기준 기록
  - [ ] train-gated graph 생성 여부 기록
  - [ ] validation/test positive가 feature에 직접 포함되지 않는지 검증
  - [ ] 누수 통제 불명확 시 해당 provider를 offline metric에서 제외
- [ ] Stage 2 feature leakage audit
  - [ ] popularity/co-occurrence/hobby profile이 train split 기준인지 검증
  - [ ] validation/test positive를 feature 생성에 사용하지 않는지 검증
  - [ ] persona text leakage audit 결과 반영
- [ ] validation/test 프로토콜 고정
  - [ ] validation으로 weight/config 선택
  - [ ] test는 최종 1회 성능 주장에 사용
  - [ ] test 결과 기반 재조정 시 새 holdout split 생성
- [ ] fallback/cold-start metric 분리
  - [ ] fallback 사용률 기록
  - [ ] unknown UUID/cold-start 성능 별도 기록
  - [ ] normal-case 성능 별도 기록
  - [ ] popularity fallback 개선분을 모델 품질 개선으로 해석하지 않음
- [ ] age_group/sex/occupation segment별 metric 분리
- [ ] mismatch case qualitative review set 작성
  - [ ] 50대 직장인 vs 20대 여성처럼 shared hobby는 같지만 맥락이 다른 케이스
  - [ ] 지역 접근성이 중요한 취미 케이스
  - [ ] 가족/주거 형태가 취미 적합도에 영향을 주는 케이스
- [ ] rerank metrics 저장
  - [ ] Recall@K
  - [ ] NDCG@K
  - [ ] provider contribution
  - [ ] catalog coverage
  - [ ] source diversity
  - [ ] leakage audit summary
  - [ ] fallback usage summary
- [ ] 추천 결과가 이미 보유한 취미를 포함하지 않는지 재검증
- [ ] Stage 2 결과의 reason/evidence가 비어 있지 않은지 검증

## Phase G8. 문서화

- [ ] 실행 방법 README 작성
- [ ] CUDA 확인 명령 추가
- [ ] 데이터 export 명령 추가
- [ ] 학습 명령 추가
- [ ] 추천 명령 추가
- [ ] artifact 설명 추가
- [ ] offline-only 실행 경계 설명 추가
- [ ] 기존 `.venv` 사용 명령만 문서화
- [ ] 2-stage architecture 문서화
- [ ] Candidate / PersonaContext / HobbyCandidate 계약 문서화
- [ ] Stage 1 provider fallback chain 문서화
- [ ] Stage 2 reranker feature/weight 문서화
- [ ] LightGCN은 최종 추천기가 아니라 후보 생성기임을 명시
- [ ] offline artifact와 online comparison baseline의 경계 문서화
- [ ] score normalization 규칙 문서화
- [ ] validation/test 사용 원칙 문서화
- [ ] persona text leakage audit/masking/no-text mode 문서화
- [ ] fallback/cold-start/normal-case metric 분리 문서화
- [ ] Stage2 tuning 로드맵 문서화
- [ ] XSimGCL/SimGCL은 Stage2 보정 이후 Stage1 A/B로 진행한다는 원칙 문서화
- [ ] canonical_hobby / alias / taxonomy 설계 문서화
- [ ] canonical item으로 학습하고 raw 예시로 출력하는 정책 문서화
- [ ] canonicalization candidate mining / review workflow 문서화

## Phase G9. 백엔드 연결 후보

- [ ] CLI 결과가 유효하면 inference API 설계
- [ ] 기존 `/api/recommend/{uuid}` 라우트와 호환되는 설계 검토
  - [ ] `method=gnn` query option 후보
  - [ ] `method=hybrid` query option 후보
  - [ ] 별도 route 사용 시 `/api/recommend/*` 네이밍과 일관성 유지
- [ ] 모델 로딩 위치 결정
- [ ] cold start 비용 측정
- [ ] 프론트 프로필 카드 연동 설계
- [ ] offline CLI 품질 승인 전 API 구현 금지
- [ ] Stage 2 reranker 품질 승인 전 API 기본 추천으로 전환 금지

## Acceptance Criteria

- [ ] `.venv` Python으로 모든 GNN 스크립트 실행 가능
- [ ] FastAPI 없이 독립 학습 가능
- [ ] G1 이후 train/eval/recommend가 Neo4j 없이 실행 가능
- [ ] 특정 UUID에 대한 추천 Top-K 출력 가능
- [ ] 학습/평가/추천이 재현 가능
- [ ] 모델 artifact와 mapping 파일 저장 가능
- [ ] random ranking baseline보다 Recall@10 개선
- [x] selected Stage1 baseline 대비 2-stage reranker가 validation NDCG@10 또는 Recall@10에서 열세가 아님 (R@10 +0.0159 개선)
- [x] selected Stage1 baseline 대비 2-stage reranker가 개선될 때만 promotion 검토 (test에서도 +0.0134 개선 확인, 승격 완료)
- [x] candidate_recall@50 낮음 / Stage2 낮음 / delta_vs_stage1 양음 여부로 Stage1 vs Stage2 원인을 분리해 해석 가능
- [x] Stage 1 provider 중 하나 실패 시 fallback으로 Top-K 후보 유지
- [x] 추천 후보마다 source/evidence/reason feature 추적 가능
- [x] 기존 보유 취미가 추천 결과에서 제외됨
- [x] offline evaluation metric은 train-only feature와 leakage audit을 통과함
- [x] score normalization 설정이 artifact로 저장됨
- [x] fallback 사용률과 cold-start 성능이 normal-case 성능과 분리됨
- [x] test metric은 validation으로 config를 고정한 뒤 산출됨
- [ ] 코드 구현 전 PRD와 체크리스트가 최신 상태로 유지됨

## Phase G7.8. Item-Item Collaborative Filtering (BM25/TF-IDF) 고도화

Stage 2 승격 완료 후, popularity bias를 줄이기 위해 Stage 1 `cooccurrence`를 정교화하는 실험을 진행한다.

- [ ] BM25-weighted ItemKNN 또는 TF-IDF weighted cooccurrence 계산 구현
- [ ] BM25 ItemKNN provider 단독 평가 (vs raw cooccurrence)
- [ ] `popularity + BM25 ItemKNN` 조합 평가 (vs 현재 baseline `popularity + cooccurrence`)
- [ ] BM25 기반 향상된 candidate pool 위에서 Stage 2 reranker 재평가
- [ ] XSimGCL/SimGCL은 이 item-item 고도화 실험 이후로 순연
