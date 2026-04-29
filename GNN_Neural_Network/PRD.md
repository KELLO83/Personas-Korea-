# GNN 취미/여가 추천 시스템 PRD

## 1. 목적

`Nemotron-Personas-Korea` 지식 그래프를 활용해 특정 페르소나에게 어울릴 가능성이 높은 취미/여가활동을 추천하는 2-stage persona-aware 추천 PoC를 구축한다.

초기 목표는 백엔드 API와 분리된 오프라인 학습/추론 스크립트이며, 모델 성능과 운영 가능성을 검증한 뒤 FastAPI inference endpoint로 확장한다.

## 2. 문제 정의

현재 프로젝트는 Neo4j 그래프, GDS 유사도, 통계, 자연어 질의 기반 분석을 제공한다. 하지만 특정 페르소나에게 아직 연결되지 않은 취미를 예측하면서도 그 사람의 생활 맥락, 성향, 직업, 지역, 나이대까지 반영하는 추천은 없다.

이 PoC는 `Person -> Hobby` 관계에서 후보를 찾되, 최종 추천은 persona-aware reranker가 판단하도록 다음 질문에 답하는 것을 목표로 한다.

> 이 페르소나가 새롭게 좋아할 가능성이 높은 취미/여가활동은 무엇인가?

단, 이 질문은 단순히 “같은 취미를 가진 다른 사람이 가진 취미”가 아니라 “이 사람의 삶과 성향에서 자연스럽게 이어질 취미”를 의미한다. 예를 들어 50대 직장인과 20대 여성이 모두 골프를 좋아하더라도, 둘의 다음 취미 추천은 나이대, 직업 맥락, 생활패턴, 가족/주거 형태, 성향에 따라 달라져야 한다.

## 3. 범위

### In Scope

- 기존 `.venv` Python 환경 사용
- PyTorch 기반 LightGCN PoC
- LightGCN을 최종 추천기가 아니라 Stage 1 후보 생성기 중 하나로 사용
- popularity, co-occurrence, G1에서 export한 `SIMILAR_TO` 기반 similar-person hobby 후보 생성
- persona/demographic/lifestyle feature 기반 Stage 2 reranking
- 후보 생성 실패 또는 후보 부족 시 fallback provider 사용
- Neo4j에서 `Person-Hobby` edge export
- train/validation/test split
- negative sampling 기반 학습
- Recall@K, NDCG@K, HitRate@K 평가
- 특정 `persona_uuid`에 대한 취미 후보 생성 및 rerank 가능한 Top-K 추천 CLI 후보
- 학습 artifact 저장

### Out of Scope

- 초기 단계 FastAPI endpoint 구현
- 프론트 UI 연결
- GraphSAGE/R-GCN 등 복합 관계 모델
- 실시간 온라인 학습
- LLM 기반 추천 생성
- 사용자 행동 로그 기반 개인화
- 초기 PoC 단계에서 FastAPI, Streamlit, Next.js 프론트 파일 변경

## 4. 구현 경계와 파일 계약

초기 PoC는 **백엔드 없이 오프라인 스크립트로만** 동작해야 한다. 모든 실행 명령은 프로젝트 규칙에 따라 기존 `.venv` 인터프리터를 사용한다.

```powershell
.\.venv\Scripts\python.exe <script-path>
```

구현 시 예상 파일 계약은 다음과 같다.

```text
GNN_Neural_Network/
  README.md
  configs/
    lightgcn_hobby.yaml
  scripts/
    export_person_hobby_edges.py
    train_lightgcn.py
    evaluate_lightgcn.py
    recommend_for_persona.py
  gnn_recommender/
    data.py
    model.py
    train.py
    metrics.py
    recommend.py
  data/
  artifacts/
```

`src/`라는 중첩 패키지명은 루트 `src/`와 혼동될 수 있으므로 사용하지 않는다. GNN 전용 재사용 코드는 `GNN_Neural_Network/gnn_recommender/` 아래에 둔다.

## 5. 추천 아키텍처 선택

최종 추천 구조는 **2-stage persona-aware recommender**로 한다.

```text
Raw Persona Dataset
        |
        v
Preprocessing / Data Quality Gate
- list parse
- raw_hobby_name -> canonical_hobby
- alias / taxonomy
- raw hobby examples 보존
- age_group
- district split
- vocabulary_report 검증
- leakage audit
- embedding/text feature 생성 (optional)
        |
        +------------------------------+
        v                              v
Stage 1 Candidate Generation      Persona Feature Store
- segment_popularity              - demographics
- co-occurrence                   - occupation/region/family/housing
- global popularity fallback      - lifestyle/persona fields
- LightGCN/XSimGCL provider       - text fields only if leakage-safe
- offline similar-person only if train-gated
        |                              |
        +---------------+--------------+
                        v
Stage 2 Persona-aware Reranker
- provider scores
- segment_popularity_score
- persona text fit only if leakage-safe
- lifestyle fit
- demographic fit
- occupation/region fit
- existing hobby compatibility
- mismatch penalty
                        |
                        v
Final Top-K
- canonical_hobby
- raw examples/display suggestions
- source/evidence/reason
```

### Stage 1과 Stage 2의 역할 분리

- **Stage 1 Candidate Generation**은 recall 중심이다. 좋아할 수도 있는 취미 후보를 넓게 뽑는다.
- **Stage 2 Persona-aware Reranking**은 precision/context 중심이다. 후보가 실제로 해당 사람에게 어울리는지 판단한다.

LightGCN은 Stage 1 후보 provider 중 하나다. LightGCN은 `Person-Hobby` edge만 보기 때문에 나이, 성별, 직업, 지역, 가족/주거 형태, persona text, 생활패턴을 직접 판단하지 못한다. 따라서 LightGCN 점수만으로 최종 추천 순위를 결정하지 않는다.

### Stage 1 Candidate Provider

Stage 1은 LightGCN 단일 provider가 아니라 multi-provider ensemble로 구성한다.

```text
Candidate(
    canonical_hobby: str,
    display_examples: list[str],
    source: str,          # lightgcn / cooccurrence / similar_person / segment_popularity / popularity
    score: float,
    source_scores: dict[str, float],
    taxonomy: dict,
    reason_features: dict
)
```

학습/평가 item 계약은 다음처럼 고정한다.

```text
raw_hobby_name -> canonical_hobby -> hobby_id
```

- `raw_hobby_name`은 alias/example/display evidence로 보존한다.
- `canonical_hobby`만 item vocabulary와 split/train/eval의 기본 단위로 사용한다.
- `hobby_id`는 canonical vocabulary에 대해서만 생성한다.

필수 provider:

1. **LightGCN provider**: gated Person-Hobby graph에서 collaborative 후보 생성
2. **co-occurrence provider**: 같은 person에게 함께 나타나는 취미 기반 후보 생성
3. **offline similar-person provider**: G1에서 export한 `SIMILAR_TO`/FastRP 기반 similar-person hobby 후보 생성
4. **segment/global popularity provider**: 후보 부족, unknown UUID, checkpoint 없음 상황의 fallback

G2+ offline 단계에서는 live Neo4j나 FastAPI를 호출하지 않는다. 따라서 `similar-person provider`는 다음 둘 중 하나로만 사용한다.

- **offline evaluation mode**: G2 split 이후 train-gated 방식으로 `similar_person_hobbies.csv`를 생성해 로컬 파일로 사용한다.
- **online comparison mode**: 기존 `RecommendationService`/`GET /api/recommend/{uuid}` 결과와 비교만 수행하며, offline metric에는 포함하지 않는다.

`similar_person_hobbies.csv`가 train-gated graph에서 생성됐다는 근거가 없으면 offline evaluation mode로 사용할 수 없다. 전체 그래프 기준 `SIMILAR_TO`/FastRP는 validation/test hobby 정보를 이미 반영했을 가능성이 있으므로, 이 경우 similar-person 후보는 online comparison 또는 qualitative review에만 사용한다.

Stage 1 안전장치:

- UUID가 LightGCN mapping에 없으면 fallback provider로 전환
- known hobby 제외 후 Top-K 후보가 부족하면 fallback provider로 보강
- 중복 hobby는 canonical name 기준으로 merge
- 각 후보는 source별 score를 보존해 Stage 2에서 feature로 사용
- fallback chain은 `LightGCN -> co-occurrence -> offline similar-person -> segment popularity -> global popularity` 순서를 기본으로 한다.

Provider별 score는 scale이 다르므로 Stage 2 입력 전에 정규화한다. 초기 규칙은 provider 내부 후보 순위 기반 percentile 또는 min-max normalization 중 하나로 config에 고정하며, raw score와 normalized score를 모두 artifact에 남긴다. 정규화 규칙을 바꾸면 기존 metric과 직접 비교하지 않는다.

### Stage 1 graph model A/B: LightGCN vs SimGCL/XSimGCL

SimGCL/XSimGCL은 Stage 2 reranker의 대체재가 아니라 Stage 1 graph candidate provider의 대체/개선 후보로 취급한다. 두 모델은 user-item interaction graph에서 더 강한 embedding을 학습하는 목적이므로, persona text나 나이·직업·지역·생활패턴 적합성 판단은 계속 Stage 2가 담당한다.

따라서 비교 단위는 다음처럼 고정한다.

```text
LightGCN Stage 1 only
SimGCL/XSimGCL Stage 1 only
LightGCN Stage 1 + 동일한 Stage 2 persona reranker
SimGCL/XSimGCL Stage 1 + 동일한 Stage 2 persona reranker
```

Stage 1 모델 비교는 같은 train/validation/test split, 같은 known-hobby masking, 같은 candidate_k, 같은 normalization/fallback 규칙에서 수행한다. Stage 2 weight와 feature policy는 고정한 뒤 Stage 1 후보 생성기만 교체해야 한다. XSimGCL 도입 여부는 LightGCN candidate recall@K가 Stage 2 성능의 병목으로 확인된 뒤 결정한다.

### Data quality gate before Stage 1 ablation: hobby canonicalization

Stage 1 provider ablation은 **canonicalization gate가 끝난 뒤에만** 수행한다. 즉 순서는 다음처럼 고정한다.

```text
raw_hobby_name -> canonical_hobby / taxonomy 설계
-> canonical 기준 export 재생성
-> canonical 기준 split/train/eval 재생성
-> vocabulary_report 재검토
-> Stage 1 provider ablation
-> Stage 2 / LightGCN / SimGCL / XSimGCL 재평가
```

canonicalization gate가 끝나기 전에는 Stage 1 ablation 결과를 최종 기준선으로 승격하지 않는다.

세부 canonicalization 규칙과 성공 기준은 아래의 `Hobby canonicalization / alias / taxonomy design` 및 `Vocabulary Quality Gate` 섹션을 따른다.

### Stage 1 provider ablation before Stage 2 promotion

현재 데이터는 `retained_edges` 대비 `retained_persons`가 적지 않고, person당 retained hobby 수가 매우 작아 collaborative graph 모델보다 통계적/segment 기반 provider가 더 강할 가능성이 높다. 따라서 Stage 2 reranker나 SimGCL/XSimGCL Stage 1 교체 실험보다 먼저, **Stage 1 provider 단독 및 조합 기준선**을 고정한다.

우선 실험군은 다음 8개를 최소 세트로 사용한다.

```text
1. popularity only
2. cooccurrence only
3. segment_popularity only
4. popularity + cooccurrence
5. segment_popularity + popularity
6. segment_popularity + cooccurrence
7. segment_popularity + cooccurrence + popularity
8. segment_popularity + cooccurrence + popularity + LightGCN
```

실험 원칙:

- validation split에서 `recall@10`, `ndcg@10`를 우선 비교한다.
- test split은 validation에서 선택된 단일 조합에 대해서만 최종 1회 사용한다.
- `popularity + cooccurrence` 대비 `+ LightGCN`이 실제 이득을 주는지 별도 기록한다.
- LightGCN이 validation 기준선을 개선하지 못하면, Stage 1 기본 조합에서 제외하거나 보조 provider로만 유지한다.
- Stage 2는 위 Stage 1 기준선이 고정되기 전까지 기본 추천기로 승격하지 않는다.

Stage 1과 Stage 2의 실패 원인은 분리해서 해석한다. 이를 위해 각 Stage 1 조합마다 최소 다음 4개를 함께 기록한다.

```text
Stage1 recall@10 / ndcg@10
candidate_recall@50
Stage2 recall@10 / ndcg@10
delta_vs_stage1
```

판정 규칙은 다음처럼 고정한다.

- `candidate_recall@50`가 낮으면 **Stage 1 후보 생성 실패**로 본다. 이 경우 Stage 2가 낮아도 Stage 2 책임으로 해석하지 않는다.
- `candidate_recall@50`는 충분히 높지만 `delta_vs_stage1`가 음수이면 **Stage 2 reranker 실패**로 본다. 이 경우 Stage 2가 좋은 후보를 위로 올리지 못하거나 오히려 순위를 훼손한 것이다.
- `candidate_recall@50`가 충분히 높고 `delta_vs_stage1`도 양수이면 **2-stage 구조 유효**로 본다.

즉, Stage 2 성능 저하가 관찰되더라도 바로 "Stage 2 아이디어 실패"로 결론 내리지 않고, 먼저 해당 Stage 1 조합의 candidate pool 품질을 확인한 뒤 원인을 분리한다.

이 단계의 성공 기준은 다음과 같다.

- `popularity + cooccurrence` 조합의 validation metric을 기준선으로 고정한다.
- `+ LightGCN` 조합이 validation `recall@10` 또는 `ndcg@10`에서 일관되게 유의미한 개선을 보일 때만 LightGCN을 기본 Stage 1 조합에 유지한다.
- 위 기준선이 확정된 뒤에만 Stage 2 reranker와 SimGCL/XSimGCL Stage 1 A/B를 진행한다.

### Hobby canonicalization / alias / taxonomy design

현재 데이터의 가장 큰 구조적 문제는 동일하거나 매우 유사한 취미가 서로 다른 raw 문장형 hobby로 분리되어 있다는 점이다. 예를 들어 다음 raw hobby들은 collaborative signal 관점에서 사실상 같은 canonical hobby로 묶이는 편이 타당하다.

```text
석촌호수 주변 산책
올림픽공원 숲길 산책
탄천 산책로 걷기
수락산 둘레길 산책
한강공원 산책
=> canonical_hobby: 산책
```

따라서 학습/평가 item은 raw hobby가 아니라 **canonical_hobby**를 기본 단위로 사용한다.

```text
raw_hobby_name
  -> canonical_hobby
  -> taxonomy
       category
       subcategory
       location_modifier
       intensity
       sociality
```

예시:

```text
raw_hobby_name: 석촌호수 주변 산책
canonical_hobby: 산책
category: 야외활동
subcategory: 걷기/산책
location_modifier: 석촌호수
intensity: low
sociality: solo_or_small_group
```

이 설계의 목적은 다음과 같다.

- raw hobby 수 축소
- singleton ratio 감소
- item degree 증가
- LightGCN/SimGCL/XSimGCL collaborative signal 복원
- cooccurrence / segment_popularity 통계 안정화
- 추천 설명성 향상

다만 canonicalization은 "많이 묶을수록 좋다"가 아니다. 잘못된 generic keyword rule은 오히려 취향 의미를 뭉개고 성능을 악화시킬 수 있다. 예를 들어 `시청`만으로 `영화/드라마 감상`, `건강/교양 콘텐츠 시청`, `유튜브/온라인 영상 시청`을 한 canonical로 합치면 잘못된 공통 취향을 학습하게 된다.

따라서 실제 canonicalization 반영은 다음 review workflow를 따른다.

```text
raw hobby 전체
-> candidate cluster 자동 생성
-> high/medium/low confidence 부여
-> 상위/high-impact cluster만 사람 검수
-> approved cluster만 canonicalization rule로 승격
-> rejected / split_required는 raw 유지 또는 분리 규칙 재설계
```

즉 전수 수작업 alias map을 만들지 않고, 자동 후보 생성 + 제한적 human review 방식으로 canonical vocabulary를 점진적으로 안정화한다.

예상 산출물:

- `GNN_Neural_Network/artifacts/canonicalization_candidates.json`
- `GNN_Neural_Network/configs/hobby_taxonomy_review.json`

`canonicalization_candidates.json`에는 최소 다음을 담는다.

```text
canonical_candidate
members
member_count
support_edges
confidence
proposed_rule
proposed_taxonomy
display_examples
status(pending_review)
```

`hobby_taxonomy_review.json`에는 최소 다음을 담는다.

```text
approved_clusters
manual_aliases
rejected_patterns
split_required
```

승인 기준:

- 행동이 동일하고 장소/플랫폼/장르가 modifier 수준이면 approve
- 의미가 넓거나 취향 카테고리가 갈리면 split_required
- `시청`, `감상`, `모임`, `투어`, `체험`처럼 generic token만으로 형성된 cluster는 기본 reject 또는 split_required

출력 정책은 canonical hobby와 raw 예시를 함께 보인다.

```text
추천 취미: 산책
추천 예시: 석촌호수 산책, 한강공원 산책, 동네 둘레길 걷기
```

canonicalization은 Stage 1 ablation의 하위 작업이 아니라 **앞단 데이터 품질 gate**다.

성공 기준:

- canonical singleton ratio는 raw singleton ratio `0.834`보다 명확히 낮아야 한다.
- retained canonical hobby 수는 raw retained hobby 수보다 감소해야 한다.
- canonical 기준 `candidate_recall@50`은 raw 기준보다 악화되지 않아야 한다.
- canonical 기준 Stage 1 baseline은 raw 기준 validation `recall@10` 또는 `ndcg@10`에서 악화되지 않아야 한다.

### Stage 2 Persona-aware Reranker

Stage 2 입력은 target person의 전체 맥락과 Stage 1 후보 취미다.

```text
PersonaContext(
    uuid: str,
    age: int,
    age_group: str,
    sex: str,
    occupation: str,
    district: str,
    province: str,
    family_type: str,
    housing_type: str,
    education_level: str,
    persona_text: str,
    professional_text: str,
    sports_text: str,
    arts_text: str,
    travel_text: str,
    culinary_text: str,
    family_text: str,
    hobbies_text: str,
    skills_text: str,
    embedding_text: str,
    known_hobbies: list[str],
)

HobbyCandidate(
    canonical_hobby: str,
    display_examples: list[str],
    source_scores: dict[str, float],
    taxonomy: dict,
    reason_features: dict,
)
```

초기 reranker는 학습 모델이 아니라 해석 가능한 weighted scoring으로 시작한다.

```text
final_score =
  0.25 * lightgcn_score
+ 0.10 * cooccurrence_score
+ 0.25 * segment_popularity_score
+ 0.10 * similar_person_score
+ 0.00 * persona_text_fit   # no-text mode for offline metric
+ 0.15 * known_hobby_compatibility
+ 0.10 * age_group_fit
+ 0.10 * occupation_lifestyle_fit
+ 0.05 * region_accessibility_fit
+ 0.05 * segment_or_global_popularity_prior
- 0.25 * mismatch_penalty
```

기본 offline metric에서는 `persona_text_fit`을 사용하지 않는다. 현재 leakage audit 기준 `embedding_text`가 held-out hobby를 직접 포함하므로, text 기반 feature는 누수 통제가 끝나기 전까지 validation/test 비교에서 비활성화한다.

이후 validation이 안정되면 LightGBM LambdaRank, Factorization Machine, two-tower retrieval, heterogeneous GraphSAGE/R-GCN 등을 비교한다. 단, 초기 구현에서는 검증 비용과 설명 가능성을 우선해 weighted reranker를 기본으로 한다.

Weighted reranker의 weight는 validation split에서만 조정한다. test split 결과를 보고 weight를 다시 조정하면 해당 test metric은 최종 성능 주장에 사용할 수 없다.

### Stage 2 tuning plan before SimGCL/XSimGCL Stage 1 swap

현재 offline 실행 결과에서 `Stage1 multi-provider`가 `persona-aware reranker`보다 높은 Recall/NDCG를 보였다. 따라서 SimGCL/XSimGCL을 Stage 1에 연결하기 전에, 먼저 Stage 2가 기존 Stage 1 신호를 온전히 사용하도록 보정한다.

우선 보정 대상은 다음 순서로 고정한다.

1. **`segment_popularity_score` 취급 재검토**
   - feature 자체는 보존하되, Stage 1 ablation에서 `segment_popularity`가 toxic이면 default Stage 2 경로에서는 비활성화한다.
   - `RerankerWeights`에는 실험용 weight를 둘 수 있지만, 기본값은 0 또는 명시적 opt-in이 바람직하다.
2. **validation-only weight sweep / grid search**
   - 기본 탐색 축은 `lightgcn_score`, `cooccurrence_score`, `known_hobby_compatibility`, `age_group_fit`, `occupation_fit`, `region_fit`, `popularity_prior`, `mismatch_penalty`로 둔다.
   - `segment_popularity_score`는 별도 ablation/실험군에서만 추가한다.
   - test split은 최종 고정 설정 1회 평가에만 사용한다.
3. **`mismatch_penalty` ablation**
   - sparse demographic distribution 환경에서 penalty가 과도하게 좋은 후보를 밀어내는지 확인한다.
   - 최소 비교군: `penalty=0`, `penalty=default`, `penalty=scaled-down`.
4. **Stage 2 feature ablation report**
   - 기본 Stage 2 feature set에서 `known_hobby_compatibility`, `age_group_fit`, `occupation_fit`, `region_fit`, `popularity_prior`, `mismatch_penalty`를 하나씩 제거해 성능 변화를 기록한다.
   - `segment_popularity_score`는 별도 실험군으로만 비교한다.

이 단계의 성공 기준은 다음과 같다.

- `persona-aware reranker`가 최소한 `selected Stage1 baseline`보다 validation NDCG@10 또는 Recall@10에서 열세가 아니어야 한다.
- `segment_popularity_score`는 default Stage 2 성공 조건이 아니다. 포함 실험군이 기본 Stage 2보다 낫다는 증거가 있을 때만 재도입한다.
- `mismatch_penalty`가 qualitative mismatch case를 줄이면서도 top-K ranking metric을 과도하게 훼손하지 않아야 한다.

이 보정이 끝난 뒤에만 다음 단계로 진행한다.

```text
LightGCN Stage1 only
LightGCN Stage1 + tuned Stage2
XSimGCL/SimGCL Stage1 only
XSimGCL/SimGCL Stage1 + same tuned Stage2
```

즉 SimGCL/XSimGCL 비교는 Stage 2가 안정화된 이후의 **Stage 1 graph model A/B** 단계로 취급한다.

### 왜 LightGCN 단독이 아닌가

LightGCN 단독은 같은 hobby node를 공유하는 사람을 가까운 사람으로 볼 수 있다. 예를 들어 50대 직장인과 20대 여성이 모두 `골프`를 좋아하면 graph propagation상 연결될 수 있다. 하지만 실제 추천에서는 다음 맥락을 봐야 한다.

- 퇴근 후 시간과 피로도
- 혼자/여럿 활동 선호
- 실내/실외 선호
- 나이·성별·직업·지역 맥락
- 가족/주거 형태상 가능한 활동
- 기존 취미와의 semantic compatibility
- persona text에 드러난 성향과 생활패턴

따라서 LightGCN은 “후보군에 넣을 만한가?”를 판단하고, persona-aware reranker가 “이 사람에게 정말 어울리는가?”를 판단한다.

## 6. LightGCN 후보 생성기 선택

Stage 1의 첫 학습 기반 후보 생성기는 **LightGCN**으로 한다.

이유:

- `Person-Hobby` bipartite 후보 생성에 적합
- 8GB VRAM 환경에서 PoC 가능성이 높음
- PyTorch tensor/autograd만으로 구현 가능해 PyG/DGL/TorchRec 설치 리스크가 낮음
- popularity/co-occurrence 후보 provider 대비 성능 비교가 명확함

### 구현 라이브러리 정책

초기 PoC는 **순수 PyTorch 구현**을 원칙으로 한다.

허용:

- `torch` tensor 연산
- `torch.nn.Module`
- `torch.nn.functional.logsigmoid`
- `torch.sparse.mm`
- PyTorch autograd
- `torch.optim.Adam` 또는 `torch.optim.AdamW`
- `torch.save` / `torch.load`
- `numpy`, `pandas`, `pyyaml`, `tqdm` 같은 데이터/설정/로그 보조 라이브러리

금지:

- PyTorch Geometric (`torch_geometric`)
- DGL
- TorchRec
- RecBole 등 추천시스템 프레임워크
- 외부 LightGCN 구현체 복사/의존
- `torch.nn` loss module로 BPR loss 대체
- PyTorch 외부 scheduler 구현체 사용

직접 구현 대상:

- LightGCN propagation
- sparse normalized adjacency 구성
- BPR loss
- negative sampler
- Recall/NDCG/HitRate metric
- learning-rate scheduler 정책 정의
- checkpoint save/load contract

Optimizer는 v1에서 `torch.optim.Adam`을 기본으로 한다. `AdamW`는 config option으로 허용한다. LightGCN propagation, BPR loss, negative sampler, metric은 직접 구현하지만, optimizer/checkpoint/autograd 등 PyTorch 기본 도구는 적극 사용한다.

## 7. 데이터 설계

### 입력 그래프

LightGCN candidate provider는 아래 관계만 사용한다.

```text
(Person)-[:ENJOYS_HOBBY]->(Hobby)
```

Neo4j 연결 정보는 기존 프로젝트 설정을 재사용한다.

- `src.config.settings.NEO4J_URI`
- `src.config.settings.NEO4J_USER`
- `src.config.settings.NEO4J_PASSWORD`
- `src.config.settings.NEO4J_DATABASE`

별도 `.env` parser를 새로 만들거나 연결값을 하드코딩하지 않는다.

### Export Cypher 계약

초기 export 쿼리는 다음 형태를 기준으로 한다.

```cypher
MATCH (p:Person)-[:ENJOYS_HOBBY]->(h:Hobby)
WHERE p.uuid IS NOT NULL AND h.name IS NOT NULL
RETURN DISTINCT p.uuid AS person_uuid, h.name AS hobby_name
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
- `raw_hobby_name -> canonical_hobby`
- `canonical_hobby -> hobby_id`

`raw_hobby_name`은 alias/example/display evidence로 보존하고, `canonical_hobby`만 item vocabulary 기준으로 사용한다.

Export 이후 G2+ 단계(split/train/eval/recommend)는 Neo4j나 FastAPI 없이 로컬 파일만 사용해야 한다.

### Stage 2 Offline Feature / Artifact 계약

2-stage offline evaluation을 하려면 `Person-Hobby` edge 외에도 Stage 2 feature가 필요하다. 단, artifact 생성 순서는 leakage 방지를 위해 분리한다.

G1 raw export 단계에서는 Neo4j에서 split과 무관한 원천 데이터만 export한다.

```text
GNN_Neural_Network/data/person_context.csv
```

G2 split 이후에는 train split만 사용해 train-gated artifact를 생성한다.

```text
GNN_Neural_Network/artifacts/hobby_profile.json           # train split 기반 생성
GNN_Neural_Network/data/similar_person_hobbies.csv        # train-gated 근거가 있을 때만 offline evaluation 사용
```

`person_context.csv` 최소 schema:

```csv
person_uuid,age,age_group,sex,occupation,district,province,family_type,housing_type,education_level,persona_text,professional_text,sports_text,arts_text,travel_text,culinary_text,family_text,hobbies_text,skills_text,career_goals
```

`similar_person_hobbies.csv` 최소 schema:

```csv
person_uuid,similar_person_uuid,similarity_score,hobby_name,evidence
```

`hobby_profile.json`은 validation/test 누수를 막기 위해 **G2 split 이후 train split만** 사용해 생성한다. 포함 정보:

- hobby별 train popularity
- age_group/sex/occupation/region 분포
- train co-occurring hobbies
- known hobby compatibility 계산용 통계

`similar_person_hobbies.csv`를 offline metric에 사용할 경우, held-out validation/test hobby가 provider feature에 새지 않도록 export 기준과 leakage 가능성을 별도 기록한다. train-gated 생성이 구현되지 않았거나 누수 통제가 불명확하면 offline metric에는 포함하지 않고 online comparison baseline으로만 사용한다.

### Persona Text Leakage Audit

Stage 2는 persona text와 hobby text를 feature로 사용할 수 있지만, offline holdout 평가에서는 이 텍스트가 validation/test positive hobby를 직접 언급할 수 있다. 따라서 다음 중 하나를 반드시 적용한다.

1. **audit mode**: validation/test positive hobby명이 `persona_text`, domain persona text, `hobbies_text`, `embedding_text`에 직접 등장하는 비율을 기록하고 metric과 함께 보고한다.
2. **masking mode**: evaluation용 `PersonaContext` 생성 시 held-out hobby명을 텍스트에서 mask 처리한다.
3. **no-text mode**: 누수 통제가 끝날 때까지 `persona_text_fit`과 text embedding 기반 feature를 offline metric에서 제외한다.

초기 구현은 audit mode를 필수로 하고, 누수율이 높으면 masking mode 또는 no-text mode로 전환한다.

### Vocabulary Quality Gate

raw `hobby_name`을 그대로 LightGCN item ID로 사용하면 유사 취미가 여러 노드로 쪼개져 collaborative signal이 약해진다. 따라서 split/index 전에 다음 gate를 통과해야 한다.

- `normalize_hobbies`: Unicode 정규화, 앞뒤 공백 제거, 내부 공백 collapse, 소문자화를 적용한다.
- `alias_map_path`: 선택적 JSON alias map으로 raw/variant hobby명을 canonical hobby명으로 접는다.
- alias 적용 후 `(person_uuid, canonical_hobby_name)` 중복 edge를 제거한다.
- `min_item_degree` 미만 canonical hobby는 학습 item에서 제외한다.
- `vocabulary_report.json`에 raw/canonical/retained person·hobby·edge 수, singleton ratio, dropped count를 저장한다.
- vocabulary gate 후에도 validation/test leakage와 full-known masking을 다시 검증한다.

## 8. Split, Negative Sampling, 학습 방식

기존 연결 일부를 숨기고 모델이 숨겨진 취미를 맞추도록 학습한다.

```text
positive edge: 실제 Person-Hobby 연결
negative edge: 해당 person과 전체 known graph에서 연결되지 않은 hobby
objective: BPR loss
```

BPR loss는 직접 구현한다.

```text
loss = -mean(logsigmoid(pos_score - neg_score))
```

초기 optimizer는 `torch.optim.Adam`을 사용한다. learning-rate scheduler는 v1에서 constant learning rate를 기본으로 하고, step decay 또는 cosine decay가 필요해지면 config로 명시한다. 중요한 구현 대상은 optimizer 자체가 아니라 **누수 없는 split/negative sampling, chunked inference, baseline 대비 평가**다.

### Inference 메모리 정책

전체 `person x hobby` score matrix는 메모리 한도 내에서 `person batch x all hobbies` 또는 `selected persons x hobby chunk` 방식으로 계산한다. 현재 데이터 크기에서는 person batch 단위 matmul이 기본이고, 데이터가 커지면 hobby chunk 방식으로 전환한다.

```text
for hobby_chunk in hobby_chunks:
    score selected persons against hobby_chunk
    update top-k buffer
```

### Split 규칙

평가는 person 단위 holdout을 기본으로 한다.

- 취미가 3개 이상인 person은 최소 1개 train edge를 유지하고 validation/test edge를 분리한다.
- 취미가 2개인 person은 train 1개 + eval 1개 또는 train-only 정책 중 하나를 config로 명시한다.
- 취미가 1개뿐인 person은 train-only로 두거나 평가에서 제외한다.
- validation/test positive edge는 training graph에 포함하지 않는다.
- final recommendation에서는 full known graph의 기존 취미를 모두 제외한다.

### Negative Sampling 규칙

- 초기 구현은 positive edge 1개당 negative 1개를 샘플링한다.
- negative hobby는 해당 person의 full known positive hobby에 포함되면 안 된다.
- validation/test positive hobby를 train negative로 샘플링하면 안 된다.
- seed를 고정해 재현 가능한 샘플링을 보장한다.

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

## 9. 평가 지표

- Recall@10
- NDCG@10
- HitRate@10
- 추천 결과에서 이미 보유한 취미 제외 여부
- Stage 1 후보 recall@K
- Stage 1 graph model A/B: LightGCN candidate recall@K vs SimGCL/XSimGCL candidate recall@K
- Stage 2 rerank 후 Recall@K/NDCG@K 변화
- selected Stage1 baseline 대비 Stage2 reranker 개선 여부
- LightGCN-only 대비 개선 여부(참고 지표)
- 후보 provider별 contribution: lightgcn/cooccurrence/popularity/similar_person
- demographic mismatch penalty 적용 전후 비교
- train-only feature 사용 여부와 leakage audit 결과
- 학습 시간 및 GPU 메모리 사용량
- random ranking baseline 대비 Recall@10 개선 여부
- 기존 `src/graph/recommendation.py` / `GET /api/recommend/{uuid}` 기반 그래프 추천 baseline과 비교 가능성

Metric은 다음 조건을 만족할 때만 유효하다.

- LightGCN-only, multi-provider, persona-aware reranker가 같은 split과 같은 known-hobby masking 규칙을 사용한다.
- Stage 2 reranker 비교는 같은 Stage 1 candidate pool 또는 candidate recall 차이를 별도 보고한 조건에서 수행한다.
- LightGCN과 SimGCL/XSimGCL 비교는 Stage 2 reranker를 동일하게 고정하고 Stage 1 후보 생성기만 교체한 조건에서만 유효하다.
- fallback 사용률, cold-start UUID 성능, normal-case 성능을 분리해 기록한다.
- 운영 안전장치인 popularity fallback으로 개선된 결과를 모델 품질 개선으로 해석하지 않는다.

## 10. Baseline

초기 baseline은 두 종류를 구분한다.

1. **Offline baseline**: export CSV와 split 파일만 사용한 popularity/co-occurrence 추천
2. **Existing graph baseline**: 기존 `src/graph/recommendation.py`의 `RecommendationService` 또는 `/api/recommend/{uuid}` 결과

LightGCN PoC의 필수 비교 대상은 offline baseline이다. 기존 graph baseline은 Neo4j가 필요한 비교 기준으로 별도 기록한다.

학습/평가 결과에는 같은 gated split에서 계산한 popularity와 co-occurrence baseline Recall/NDCG/HitRate를 함께 저장해야 한다.

2-stage 평가에서는 다음 baseline을 추가 비교한다.

1. **LightGCN-only**: Stage 1 후보 점수만 사용
2. **SimGCL/XSimGCL-only**: Stage 1 graph 후보 생성기 교체 실험. LightGCN과 같은 split/candidate_k에서 비교한다.
3. **existing graph recommendation**: `RecommendationService` / `SIMILAR_TO` 기반 추천. live Neo4j가 필요하면 offline metric이 아니라 별도 online comparison으로 기록한다.
4. **multi-provider candidate only**: provider score 단순 merge
5. **persona-aware reranker**: 최종 weighted rerank

최종 성공 기준은 **selected Stage1 baseline**보다 persona-aware reranker가 validation `NDCG@10` 또는 `Recall@10`에서 열세가 아니거나, 개선을 보이는 것이다. LightGCN-only는 너무 약한 기준선일 수 있으므로 Stage2 승격 기준으로 사용하지 않는다.

실험 산출물은 최소 다음을 남긴다.

- `GNN_Neural_Network/artifacts/stage1_ablation_validation.json`
- `GNN_Neural_Network/artifacts/stage1_ablation_test.json`

각 조합별 저장 필드는 최소 다음을 포함한다.

```text
providers
recall@10
ndcg@10
hit_rate@10
candidate_recall@50
delta_vs_selected_baseline
```

평가 시 Stage 2에서 사용하는 popularity, co-occurrence, hobby profile, segment fit feature는 모두 train split에서 계산해야 한다. validation/test positive를 feature 생성에 사용하면 해당 metric은 무효로 본다.

최종 성능 주장은 validation으로 weight/config를 고정한 뒤 test split에서 1회 산출한 metric을 기준으로 한다. 반복적으로 test 결과를 보고 설정을 바꾼 경우, 새 holdout split을 만들어 다시 평가한다.

## 11. CLI 사용 목표

초기 추론은 CLI로 제공한다. 기존 LightGCN CLI는 Stage 1 후보 생성 검증용이며, 2-stage 구현 후에는 candidate generation과 reranking 결과를 구분해 출력해야 한다.

```powershell
.\.venv\Scripts\python.exe GNN_Neural_Network\scripts\recommend_for_persona.py --uuid a5ad493e75e74e5cb4a81ac934a1db8f --top-k 10
```

알 수 없는 UUID나 masked candidate 부족 상황에서는 gated train split 기반 popularity fallback뿐 아니라 co-occurrence, offline similar-person hobby provider를 순차적으로 사용한다. 이 fallback은 LightGCN 품질 검증을 대체하지 않고, long-tail/cold-start 상황에서 빈 결과를 피하기 위한 안전장치다.

예상 출력:

```text
추천 취미 Top 10
1. 성수동 테마 카페 투어 | score=0.842
2. 기구 필라테스 | score=0.817
3. 대학로 소극장 연극 관람 | score=0.791
```

## 12. Artifact

학습 결과는 Git에 포함하지 않는 것을 원칙으로 한다.

예상 산출물:

- `GNN_Neural_Network/artifacts/lightgcn_hobby.pt`
- `GNN_Neural_Network/artifacts/person_mapping.json`
- `GNN_Neural_Network/artifacts/hobby_mapping.json`
- `GNN_Neural_Network/artifacts/metrics.json`
- `GNN_Neural_Network/artifacts/config_snapshot.yaml`
- `GNN_Neural_Network/artifacts/sample_recommendations.json`
- `GNN_Neural_Network/artifacts/candidates_sample.json`
- `GNN_Neural_Network/artifacts/rerank_metrics.json`
- `GNN_Neural_Network/artifacts/reranker_weights.json`
- `GNN_Neural_Network/artifacts/rerank_sample.json`
- `GNN_Neural_Network/data/person_context.csv`
- `GNN_Neural_Network/data/similar_person_hobbies.csv`
- `GNN_Neural_Network/artifacts/hobby_profile.json`
- `GNN_Neural_Network/artifacts/provider_contribution.json`
- `GNN_Neural_Network/artifacts/score_normalization.json`
- `GNN_Neural_Network/artifacts/leakage_audit.json`
- `GNN_Neural_Network/artifacts/fallback_usage.json`

구현 전 `.gitignore`에 다음 항목을 추가해야 한다.

```gitignore
GNN_Neural_Network/data/
GNN_Neural_Network/artifacts/
```

## 13. 성공 기준

PoC 성공 기준:

- Neo4j에서 Person-Hobby edge export 성공
- LightGCN 학습 스크립트가 `.venv`에서 실행됨
- CUDA 사용 가능 시 GPU 학습 가능
- `metrics.json`에 Recall@10, NDCG@10, HitRate@10, num_persons, num_hobbies, num_edges 저장
- 특정 UUID에 대해 기존 취미를 제외한 추천 Top-K 출력
- FastAPI 없이 독립 CLI로 동작
- 추천 결과가 random ranking baseline보다 Recall@10 기준으로 개선됨
- multi-provider Stage 1이 LightGCN 단일 후보보다 candidate recall을 유지하거나 개선
- canonical_hobby 기준 `vocabulary_report.json`이 생성됨
- canonical singleton ratio가 raw singleton ratio `0.834`보다 감소함
- raw hobby examples가 canonical recommendation 출력 근거로 보존됨
- persona-aware Stage 2 reranker가 selected Stage1 baseline 대비 validation과 test에서 모두 개선을 보였으므로 **promoted** 상태로 기록한다. 현재 운영 기본 추천기는 `popularity + cooccurrence` 후보 생성 후 Stage 2 reranking을 적용하는 파이프라인이다.

### Item-Item Collaborative Filtering (BM25/TF-IDF) 고도화

Stage 2가 성공적으로 승격됨에 따라 다음 개선 우선순위는 Stage 1 후보 생성(candidate pool) 품질 향상이다. 

현재 가장 강한 Stage 1 신호는 `cooccurrence`이다. 하지만 단순 cooccurrence는 전역적으로 자주 등장하는 취미(인기 아이템)에 점수가 편향되는 한계가 있다. 이를 극복하고 틈새(niche) 취향의 강한 연관성을 포착하기 위해 다음 실험을 진행한다.

1. **BM25 / TF-IDF ItemKNN Provider 도입**
   - 단순 동시발생 횟수 대신 BM25 또는 TF-IDF 가중치를 적용하여 흔한 아이템의 영향력을 페널티로 줄이고 희귀 아이템 간의 강한 연관성을 부스팅한다.
2. **실험 절차**
   - BM25 ItemKNN provider를 구현하여 단독 recall@10 측정
   - `popularity + BM25 ItemKNN`을 구성하여 현재 selected baseline (`popularity + cooccurrence`)과 비교
   - 새로운 baseline 위에서 Stage 2 reranker 재평가

XSimGCL 등의 그래프 모델은 같은 collaborative 계열로서 popularity bias 한계를 공유할 가능성이 높으므로, BM25/TF-IDF 기반 ItemKNN 고도화를 먼저 완료한 후 실험 트랙으로 돌린다.
- 50대/20대, 직업/지역/생활패턴 mismatch 사례에서 reranker가 부적절 후보를 하향 조정
- 추천 결과마다 source/evidence/reason feature를 추적 가능
- Stage 2 feature가 train split 기준으로 생성되어 leakage audit을 통과
- CPU와 CUDA-if-available 경로 모두 오류 없이 동작

## 14. 향후 확장

PoC 이후 성능이 유의미하면 다음 단계로 확장한다.

1. Cypher baseline 추천과 성능 비교
2. FastAPI inference endpoint 추가
3. 프로필 상세 화면에 GNN 추천 카드 추가
4. `Person-Skill`, `Person-Occupation`, `Person-Region` 관계를 반영한 GraphSAGE/R-GCN 실험
5. 추천 이유 생성 로직 추가

향후 API는 기존 추천 라우트와 충돌하지 않도록 설계한다. 후보는 기존 `GET /api/recommend/{uuid}`에 `method=gnn` 옵션을 추가하거나, 별도 라우트를 만들 경우 기존 `/api/recommend/*` 네이밍과 호환되게 문서화한다.
