# GNN_Neural_Network: 취미 추천 실험 비교

이 폴더는 `Nemotron-Personas-Korea` 데이터에서 `Person -> Hobby` 추천 모델을
비교하는 오프라인 실험 workspace입니다. 현재 문서의 기준일은 2026-05-17입니다.

## 현재 결론

현재 split 기준 최종 선택 모델은 **E5-small-ko-v2 domain-specific Stage2**입니다.

```text
Stage1 = popularity + cooccurrence
Stage2 = LightGBM(num_leaves=31)
Embedding model = dragonkue/multilingual-e5-small-ko-v2
Feature policy = text_embedding_similarity + E5 domain-specific similarities
MMR = false
KURE Stage1 semantic provider = false
```

## 데이터 입력과 전처리

| 입력 | 용도 | 전처리 |
| --- | --- | --- |
| `train_edges.csv` | Stage1 통계와 known hobby 구성 | `(person_id, hobby_id)` edge만 사용하고, person별 known hobby set을 만듦 |
| `validation_edges.csv`, `test_edges.csv` | Stage2 label과 최종 평가 truth | holdout positive로 사용하며, 해당 취미명은 텍스트 feature에서 마스킹 |
| `person_context.csv` | persona profile feature와 embedding 입력 텍스트 | persona/domain 텍스트, age/occupation/region/family 등 구조화 필드 로드 |
| `hobby_profile.json` | hobby별 train-only 통계 feature | train split에서 만든 popularity, segment distribution, cooccurring hobby만 사용 |
| `hobby_aliases.json` | leakage masking | holdout hobby와 alias를 함께 찾아 `[ACT]`로 치환 |

텍스트 feature를 쓰는 실험은 holdout 정답 취미명이 persona 텍스트에 직접 남아 있으면
성능이 과대평가될 수 있습니다. 그래서 `persona_text`, `professional_text`,
`sports_text`, `arts_text`, `travel_text`, `culinary_text`, `family_text`,
`hobbies_text`, `embedding_text`에서 holdout hobby와 alias를 먼저 마스킹하고,
마스킹 후 audit 실패율이 5%를 넘으면 text embedding 실험을 비활성화합니다.

## Stage1 후보 생성

Stage1의 목적은 각 persona마다 Stage2가 다시 정렬할 후보 취미 pool을 만드는
것입니다. 현재 기본 후보 pool 크기는 `candidate_pool_size=50`입니다.

1. `popularity` provider
   - `train_edges`에서 hobby별 등장 횟수를 셉니다.
   - 전체 train popularity가 높은 취미를 우선 후보로 냅니다.
   - 해당 persona가 train에서 이미 가진 known hobby는 제외합니다.

2. `cooccurrence` provider
   - train split에서 같은 persona에게 함께 붙은 취미 쌍의 co-occurrence count를 만듭니다.
   - 현재 persona의 known hobby들과 함께 자주 등장한 다른 취미를 후보로 냅니다.
   - known hobby는 제외하고, 부족하면 popularity 후보가 보완합니다.

3. provider merge
   - provider별 score를 정규화합니다.
   - 같은 hobby가 여러 provider에서 나오면 하나로 합치고 source score를 보존합니다.
   - 최종 Stage1 pool은 person별 최대 50개 후보입니다.

현재 기본 모델에서는 KURE/E5 semantic embedding을 Stage1 후보 생성에 넣지 않습니다.
이 방식은 후보 recall을 낮춰 Stage2가 맞출 수 있는 정답 후보 자체를 줄였기 때문에
비채택했습니다.

## Stage2 학습 방식

Stage2는 Stage1 후보를 그대로 추천하지 않고, LightGBM이 후보별 feature row를 다시
점수화하는 learned ranker입니다.

| 단계 | 처리 방식 |
| --- | --- |
| positive row | holdout split의 `(person_id, hobby_id)`를 label `1`로 사용 |
| negative row | positive 1개당 기본 4개 샘플링 |
| hard negative | negative의 80%는 Stage1 candidate pool 안에서 뽑음 |
| easy negative | negative의 20%는 전체 hobby 중 positive/known/pool이 아닌 곳에서 무작위 샘플링 |
| feature matrix | candidate별 numeric feature를 고정된 column 순서로 `float32` 배열화 |
| model | LightGBM binary ranker, 현재 SOTA 실험은 `num_leaves=31` |

Stage2 기본 feature는 Stage1 source score와 train-only profile 통계입니다.

```text
lightgcn_score
cooccurrence_score
segment_popularity_score
known_hobby_compatibility
age_group_fit
occupation_fit
region_fit
popularity_prior
mismatch_penalty
popularity_penalty
novelty_bonus
category_diversity_reward
is_cold_start
```

현재 SOTA 실험은 여기에 E5-small-ko-v2 embedding similarity를 추가합니다.
전체 persona 텍스트와 hobby 텍스트의 cosine similarity를 `text_embedding_similarity`로
넣고, domain별 persona 텍스트와 hobby 텍스트의 cosine similarity를 추가 feature로
넣습니다.

```text
text_embedding_similarity
e5_professional_similarity
e5_sports_similarity
e5_arts_similarity
e5_travel_similarity
e5_food_similarity
e5_family_similarity
```

즉 Stage1은 "정답이 있을 법한 후보를 넓게 모으는 단계"이고, Stage2는 "person
context, train 통계, embedding similarity를 같이 보고 후보 순서를 다시 정하는 단계"입니다.

## E5-small-ko-v2 모델 구조

`dragonkue/multilingual-e5-small-ko-v2`는
[`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small)
기반의 SentenceTransformer 계열 embedding 모델입니다. Hugging Face 모델 카드 기준으로
Korean retrieval task 성능을 높이기 위해 한국어 query-passage pair로 fine-tuning된
경량 Korean retriever입니다.

| 항목 | 내용 |
| --- | --- |
| 모델 계열 | SentenceTransformer / BERT encoder 계열 |
| base model | `intfloat/multilingual-e5-small` |
| parameter size | 약 118M |
| max sequence length | 512 tokens |
| output dimension | 384-d dense vector |
| similarity | cosine similarity |
| pooling | mean pooling over token embeddings |
| final step | L2 normalize |
| model soup | `dragonkue/multilingual-e5-small-ko` 60% + `intfloat/multilingual-e5-small` 40% |

레이어 흐름은 다음과 같습니다.

```text
입력 텍스트
  -> tokenizer
  -> BERT Transformer encoder
  -> last_hidden_state
  -> attention_mask 기반 mean pooling
  -> 384차원 sentence embedding
  -> L2 normalize
  -> cosine similarity / dot product scoring
```

이 프로젝트에서는 E5-small-ko-v2를 생성 모델처럼 쓰지 않고, persona 텍스트와 hobby
텍스트를 같은 384차원 벡터 공간에 올린 뒤 cosine similarity feature를 만드는 데만
사용합니다.

## 핵심 실험 스펙 비교

모든 실험은 같은 current split, 같은 Stage1 후보풀, 같은 LightGBM recipe를
고정했습니다. 비교 대상은 Stage2에 들어가는 semantic feature 구성입니다.

| 실험 모델 | Stage1 후보 생성 | Stage2 모델 | Embedding / Semantic feature | MMR | 현재 판단 |
| --- | --- | --- | --- | --- | --- |
| no-text LightGBM | popularity + cooccurrence | LightGBM | 없음 | false | baseline |
| KURE-v1 Stage2 single | popularity + cooccurrence | LightGBM | KURE-v1 단일 cosine similarity | false | historical baseline |
| E5-small-ko-v2 Stage2 single | popularity + cooccurrence | LightGBM | E5-small 단일 cosine similarity | false | 이전 production default |
| Snowflake-ko Stage2 single | popularity + cooccurrence | LightGBM | Snowflake-ko 단일 cosine similarity | false | 이전 accuracy reference |
| E5-small-ko-v2 Stage2 domain-specific | popularity + cooccurrence | LightGBM | E5-small 전체 similarity + domain별 similarity | false | 현재 SOTA/default |

## Test 성능 비교

![Model performance comparison](docs/model_performance.svg)

| Model | Test Recall@10 | Test NDCG@10 | Decision |
| --- | ---: | ---: | --- |
| no-text LightGBM | 0.579626 | 0.360270 | baseline |
| KURE-v1 Stage2 single | 0.617482 | 0.386258 | historical baseline |
| E5-small-ko-v2 Stage2 single | 0.623837 | 0.393921 | previous production default |
| Snowflake-ko Stage2 single | 0.637653 | 0.402805 | previous accuracy reference |
| E5-small-ko-v2 Stage2 domain-specific | 0.680943 | 0.436665 | current SOTA/default |

E5 domain-specific 개선폭:

| 비교 대상 | Recall@10 delta | NDCG@10 delta |
| --- | ---: | ---: |
| Stage1/no-text baseline | +0.101317 | +0.076395 |
| E5 single | +0.057106 | +0.042744 |
| Snowflake single | +0.043290 | +0.033860 |
| KURE-v1 single | +0.063461 | +0.050407 |

## E5 Domain-Specific Feature

기존 E5/Snowflake/KURE 단일 feature는 persona 전체 텍스트와 후보 hobby 텍스트의
cosine similarity 하나만 LightGBM에 넣었습니다. 현재 선택 모델은 여기에 더해
persona text를 domain별로 나눈 similarity를 함께 사용합니다.

```text
e5_professional_similarity
e5_sports_similarity
e5_arts_similarity
e5_travel_similarity
e5_food_similarity
e5_family_similarity
```

즉 LightGBM은 "전체적으로 비슷한가"뿐 아니라 "스포츠/예술/여행/음식/가족/직업
맥락 중 어디에서 매칭됐는가"를 함께 학습합니다. 이 구성이 test에서 Snowflake
single-feature보다도 높은 Recall@10과 NDCG@10을 보여 현재 기본값으로 승격했습니다.

## 비채택 실험

| 실험 | 판단 | 이유 |
| --- | --- | --- |
| KURE-v1 Stage1 semantic provider | 거절 | Stage1 후보 품질을 낮춰 Stage2가 맞출 수 있는 정답 후보 자체를 줄임 |
| KURE dense MMR | 거절 | 현재 accuracy 중심 기본 모델에는 불리함 |

현재 증거상 embedding은 Stage1 후보 생성보다 Stage2 ranking feature로 쓰는 것이
맞습니다.
