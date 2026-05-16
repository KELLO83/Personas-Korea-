# GNN Experiment Run Summary

Date: 2026-05-15

This summary records the current offline recommender decision state before any future opt-in KURE/KRUE text feature ablation. The default recommendation path did not change.

## Current default path

```text
Stage 1 = popularity + cooccurrence
Stage 2 = LightGBM learned ranker
include_source_features = false
include_text_embedding_feature = false
MMR = false
```

Stage 1: `popularity + cooccurrence`
Stage 2: v2 LightGBM ranker

The closed Phase 2.5 default remains the comparison baseline for all later KURE dense MMR or text embedding feature work.

## Key Lessons

- Keep the promoted LightGBM regularized default as the accuracy baseline.
- Treat KURE/KRUE text feature and MMR work as gated no-go unless validation, leakage, and cold-start checks pass.

## Phase 2.5 default and cold-start baseline

- Model: `artifacts/experiments/phase2_5_num_leaves_31/ranker_model.txt`
- Config: `num_leaves=31`, `min_data_in_leaf=50`, `learning_rate=0.05`, `reg_alpha=0.1`, `reg_lambda=0.1`, `neg_ratio=4`, `hard_ratio=0.8`
- Default flags: `include_source_features=false`, `include_text_embedding_feature=false`, `MMR=false`

Dedicated sparse-user baseline artifacts were generated for `known_hobbies <= 1`:

| Split | People | V2 Recall@10 | V2 NDCG@10 | V2 Coverage@10 | V2 Novelty@10 | V2 ILD@10 | Stage1 Recall@10 | Stage1 NDCG@10 | Candidate Recall@50 | V2 Fallback |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| validation | 8,563 | 0.592199 | 0.367798 | 0.002802 | 4.570526 | 0.967444 | 0.582389 | 0.363706 | 0.827669 | 0 |
| test | 8,563 | 0.589513 | 0.368271 | 0.002802 | 4.570526 | 0.967444 | 0.578302 | 0.364737 | 0.827208 | 0 |

Artifacts:

- `artifacts/experiments/phase2_5_cold_start_baseline/validation_metrics.json`
- `artifacts/experiments/phase2_5_cold_start_baseline/test_metrics.json`

The test evaluation used process-pool-by-person feature building with `os.cpu_count() - 2` default CPU threads (`22 -> 20`) and completed in about 344 seconds. This is a throughput improvement only; it does not change ranking semantics or defaults.

## Pre-KURE blocker closure

- Status: `closed_with_taxonomy_warning`
- Decision: `ready_for_gated_kure_text_ablation`
- Default change: none

Prepare-only was refreshed for the local 50K slice:

```text
raw_edges: 50000
retained_edges: 48811
rare_item_policy: keep_with_fallback
rare_items_count: 7423
fallback_edges_count: 7457
dropped_edges: 0
```

Phase 5-B2 feature-balance validation artifacts are complete for `feature_fraction=0.7` and `feature_fraction=0.8`. Neither is promoted: `0.8` is slightly better than `0.7`, but both are only marginally above Stage 1 and keep a large fallback count.

Taxonomy over-merge remains a warning, not a default-changing result:

```text
avg_user_top_category_ratio: 0.750987
rare_hobby_count: 49262
warning categories: 기타/다양
```

Artifacts:

- `artifacts/experiments/pre_kure_experiment_summary.json`
- `artifacts/experiments/pre_kure_experiment_summary.md`
- `artifacts/vocabulary_report.json`
- `artifacts/experiments/phase5_b2_feature_balance/feature_fraction_0_7/validation_metrics.json`
- `artifacts/experiments/phase5_b2_feature_balance/feature_fraction_0_8/validation_metrics.json`
- `artifacts/experiments/phase5_taxonomy_overmerge/overmerge_report.json`

## KURE/KRUE semantic feature policy

KURE dense MMR remains `NO-GO` after the completed lambda sweep. KURE text embedding feature runs remain opt-in and non-default. A future text feature run must explicitly enable `include_text_embedding_feature=true`, run masking and post-mask leakage audit, evaluate validation first, and compare both overall metrics and the cold-start baseline above before any default discussion.

The completed Phase 5-C text feature reruns did not produce a validation winner:

- `kure_text_feature_001`: disabled by leakage gate.
- `kure_text_feature_002_context_coverage_gate`: not promoted due low mapped context coverage.
- `kure_text_feature_003_full_ranker_fallback`: rejected because validation Recall@10 regressed and remained below the closed Phase 2.5 default.
- No KURE text test run was selected.

## Phase 5-D KURE semantic Stage1 candidate provider

`kure_stage1_semantic_001_fast_gpu` tested the proposed Stage1 path `popularity + cooccurrence + kure_semantic -> LightGBM` as an opt-in experiment. It used KURE-v1 semantic scoring with visible progress, CUDA embedding when available, and evaluation feature building with the `os.cpu_count() - 2` policy (`22 -> 20` workers).

Validation outcome vs closed Phase 2.5 baseline:

| Metric | Closed Phase 2.5 validation | KURE Stage1 validation | Delta |
| --- | ---: | ---: | ---: |
| V2 Recall@10 | 0.739051 | 0.599705 | -0.139346 |
| V2 NDCG@10 | 0.457970 | 0.370891 | -0.087080 |
| Candidate Recall@50 | 0.977645 | 0.794971 | -0.182674 |
| V2 Fallback | 0 | 0 | 0 |

Decision: rejected on validation; test was skipped. KURE semantic Stage1 increased candidate diversity but removed too many held-out positives from the candidate pool, so it is not a promotion candidate. The default remains unchanged.

Artifacts:

- `artifacts/experiments/phase5_d_stage1_kure_semantic/kure_stage1_semantic_001_fast_gpu/ranker_model.txt`
- `artifacts/experiments/phase5_d_stage1_kure_semantic/kure_stage1_semantic_001_fast_gpu/validation_metrics.json`
- `artifacts/experiments/phase5_d_stage1_kure_semantic/kure_stage1_semantic_001_fast_gpu/validation_metrics.status.json`

## Next follow-up

Before another text embedding ablation, preserve cache provenance by embedding model name/revision and avoid replacing strong cooccurrence candidate recall with semantic-only retrieval. Until then, the default remains `popularity + cooccurrence -> LightGBM learned ranker` with text/source/MMR/KURE Stage1 features disabled by default.

## 2026-05-16 governance hardening and 2K pilot

Implemented text-embedding governance hardening:

- Train/eval text embedding input now uses `build_domain_tagged_persona_text`.
- Empty or missing domain text is treated as a coverage miss, not a leakage failure.
- Text/feature cache provenance includes model name, model revision, and preprocessing version.
- `embedding_model_metadata.json` is persisted for train/eval runs.
- `context_coverage_report.py` records split-aligned context coverage.

Current context coverage artifact:

- `artifacts/experiments/phase5_context_coverage/context_coverage_report.json`
- train/validation/test domain-text coverage: `1.0`

Fast KURE-v1 2K validation pilot:

| Run | Recall@10 | NDCG@10 | Coverage@10 | Novelty@10 |
| --- | ---: | ---: | ---: | ---: |
| Stage1 baseline | 0.576500 | 0.358540 | 0.002935 | 4.538293 |
| No-text LightGBM pilot | 0.620000 | 0.380264 | 0.002802 | 4.603208 |
| KURE text LightGBM pilot | 0.636500 | 0.390696 | 0.004003 | 4.688600 |

Decision: `needs_full_validation_followup`. The same-sample pilot signal is positive, but this is not promotion-grade and ranking-collapse diversity is still unresolved.

## 2026-05-16 KURE domain-tagged full validation and test decision

Historical note: this decision used the older closed Phase 2.5 artifact as the
absolute comparison target. It is now superseded by the later current-data
locked no-text vs KURE Stage2 comparison below.

| Run | Split | Recall@10 | NDCG@10 | Candidate Recall@50 | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| `phase2_5_num_leaves_31` | test | 0.709684 | 0.447713 | 0.977136 | older closed artifact |
| `kure_text_feature_005_domain_tagged_20k_cpu10_test_matrix_retry` | test | 0.617482 | 0.386258 | 0.827208 | historical, superseded |

KURE-v1 domain-tagged text features showed a real Stage2 signal under the matched current candidate pool:

- test delta vs its own Stage1: Recall@10 `+0.047711`, NDCG@10 `+0.029900`
- full validation KURE text vs matched no-text control: Recall@10 `+0.043014`, NDCG@10 `+0.030504`

Historical decision at that point:

- The run was not promoted against the older closed artifact.
- That conclusion must not be used as the current default decision because the
  later locked same-current-data comparison is the valid decision source.

### Final matched current-code validation follow-up

The final validation-only follow-up was executed with `--cpu-thread-count 10` and `--progress-mode on`.

| Run | Validation Recall@10 | Validation NDCG@10 | Candidate Recall@50 |
| --- | ---: | ---: | ---: |
| `control_no_text_current_code_validation_cpu10` | 0.591692 | 0.366055 | 0.827669 |
| `kure_text_feature_005_current_code_validation_cpu10` | 0.634706 | 0.396559 | 0.827669 |

KURE text delta vs matched no-text control:

- Recall@10 `+0.043014`
- NDCG@10 `+0.030504`

Final decision:

- KURE-v1 text feature is useful Stage2 signal.
- This section is superseded by `current_locked_kure_stage2_num_leaves31_cpu10`, which promotes KURE Stage2 on the current data/split.

## 2026-05-16 strict SOTA-pool KURE Stage2 feature attempt

Implemented `scripts/train_eval_sota_pool_kure_feature.py` for the requested strict comparison:

- keep the preserved closed-SOTA candidate feature cache fixed
- append only `text_embedding_similarity`
- train/evaluate no-text and KURE Stage2 LightGBM under the same cached candidate rows
- show progress for reproduction, text masking, KURE embedding, feature build, training, and evaluation
- abort if SOTA candidate-pool reproduction fails before promotion-grade comparison

Result: blocked. The preserved SOTA cache and the current split artifacts do not match.

| Check | Value |
| --- | ---: |
| preserved validation cache persons | 9,841 |
| current `validation_edges.csv` persons | 10,857 |
| reproduced candidate_recall@50 | 0.361702 |
| required candidate_recall@50 guard | 0.950000 |

Decision: the attempted strict SOTA-pool KURE comparison is invalid for default promotion. The default remains `phase2_5_num_leaves_31` with `include_text_embedding_feature=false`. A promotion-grade rerun would require the original SOTA split snapshot or a full rebuild of both baseline and KURE under one newly locked split.

## 2026-05-16 current-data locked no-text vs KURE Stage2 rerun

Because the older closed-SOTA cache is not comparable to the current split, the current data/split was locked and rerun with the same Stage1 candidate pool and SOTA LightGBM recipe (`num_leaves=31`):

- Stage1: `popularity + cooccurrence`
- Stage2 baseline: LightGBM without text embedding
- Stage2 candidate: LightGBM with KURE `text_embedding_similarity`
- CPU thread count: `10`
- Progress mode: `on`
- Direct cached-matrix evaluation script: `scripts/evaluate_cached_ranker_matrix.py`

| Split | Model | Recall@10 | NDCG@10 | Candidate Recall@50 |
| --- | --- | ---: | ---: | ---: |
| validation | no-text | 0.591876 | 0.366105 | 0.827669 |
| validation | KURE Stage2 | 0.634706 | 0.396559 | 0.827669 |
| test | no-text | 0.579626 | 0.360270 | 0.827208 |
| test | KURE Stage2 | 0.617482 | 0.386258 | 0.827208 |

Current-split deltas:

- validation Recall@10 `+0.042830`, NDCG@10 `+0.030454`
- test Recall@10 `+0.037856`, NDCG@10 `+0.025988`

Decision: KURE Stage2 is selected over the current no-text baseline for the current split/candidate pool and is the current SOTA/default candidate.

## 2026-05-16 Stage1 vs Stage2 KURE role decision

KURE should be used in Stage2, not Stage1, under the current evidence.

| Role | Experiment | Validation Recall@10 | Validation NDCG@10 | Candidate Recall@50 | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| Stage1 semantic provider | `kure_stage1_semantic_001_fast_gpu` | 0.599705 | 0.370891 | 0.794971 | rejected |
| Stage2 feature | `current_locked_kure_stage2_num_leaves31_cpu10` | 0.634706 | 0.396559 | 0.827669 | selected |

Reason: Stage1 KURE changes the retrieval pool and drops too many held-out positives before the ranker can see them. Stage2 KURE keeps the stronger `popularity + cooccurrence` candidate pool unchanged and improves ordering inside that pool. Future embedding-model experiments should therefore prioritize Stage2 feature ablations first.

## 2026-05-16 KURE Stage2 feature training-method review

The current KURE Stage2 feature construction is appropriate as the first promoted embedding feature:

- Train/eval both use `build_domain_tagged_persona_text`.
- Held-out hobby names are masked before persona encoding via `mask_holdout_hobbies`.
- Leakage audit passed: `failed_person_count=0`, `passed_person_count=10857`.
- Feature cache metadata records model name, preprocessing version, masking, and text builder.
- The fixed Stage1 candidate pool remains `popularity + cooccurrence`; KURE only adds `text_embedding_similarity` to Stage2.
- LightGBM uses KURE as one numeric feature, so popularity/cooccurrence/context features can still override bad semantic matches.

Current limitation:

- The feature is a single cosine similarity between masked persona text and hobby name text.
- It does not yet expose domain-specific similarities such as sports/art/travel/food separately.
- It does not include per-person KURE confidence or margin features.

Recommended next Stage2 embedding work:

1. Compare other Korean embedding models as the same single `text_embedding_similarity` feature.
2. Add domain-specific KURE features, for example `kure_sports_similarity`, `kure_art_similarity`, `kure_travel_similarity`, while keeping leakage masking.
3. Add rank/margin features inside the fixed candidate pool, for example KURE similarity percentile or gap to the person's top semantic candidate.

Do not reopen Stage1 semantic retrieval until a design explicitly preserves candidate recall.

## 2026-05-16 Snowflake-ko Stage2 single-feature validation attempt

Track A code was patched so Stage2 text embedding experiments can select an explicit SentenceTransformer backbone:

- `train_ranker.py --text-embedding-model-name`
- `train_ranker.py --text-embedding-model-revision`
- `evaluate_ranker.py --text-embedding-model-name`
- `evaluate_ranker.py --text-embedding-model-revision`

Training run completed:

```text
run_id = snowflake_stage2_single_feature_validation_cpu10
embedding_model = dragonkue/snowflake-arctic-embed-l-v2.0-ko
Stage1 = popularity + cooccurrence
Stage2 = LightGBM(num_leaves=31) + text_embedding_similarity
cpu_threads = 10
progress_mode = on
device = cuda
batch_size = 16
```

Training artifact:

```text
artifacts/experiments/phase5_c_text_embedding/snowflake_stage2_single_feature_validation_cpu10/ranker_model.txt
```

Training metadata:

| Item | Value |
| --- | ---: |
| train rows | 43,425 |
| validation rows for internal LightGBM split | 10,860 |
| ranker train persons | 8,685 |
| ranker val persons | 2,172 |
| best iteration | 95 |
| best AUC | 0.873005 |
| runtime seconds | 1457.58 |
| leakage failed persons | 0 |
| leakage passed persons | 10,857 |

Top feature gains:

| Feature | Gain |
| --- | ---: |
| `popularity_prior` | 115785.96 |
| `text_embedding_similarity` | 21505.97 |
| `age_group_fit` | 4149.34 |
| `mismatch_penalty` | 3629.29 |

Validation evaluation status:

- Full validation evaluation was started with the same Snowflake model identity and progress enabled.
- It reached `candidates_done` with `candidate_pool_person_count=10857`.
- The command timed out after 1 hour before `validation_metrics.json` was produced.
- Therefore Snowflake-ko is **not promoted** and **not rejected on Recall/NDCG yet**. It is `trained_needs_validation_resume`.

Next step when resuming:

```powershell
.\.venv\Scripts\python.exe GNN_Neural_Network\scripts\evaluate_ranker.py `
  --config GNN_Neural_Network\configs\kure_text_optin_ranker.yaml `
  --split validation `
  --model-path GNN_Neural_Network\artifacts\experiments\phase5_c_text_embedding\snowflake_stage2_single_feature_validation_cpu10\ranker_model.txt `
  --output GNN_Neural_Network\artifacts\experiments\phase5_c_text_embedding\snowflake_stage2_single_feature_validation_cpu10\validation_metrics.json `
  --experiment-id snowflake_stage2_single_feature_validation_cpu10 `
  --text-embedding-model-name dragonkue/snowflake-arctic-embed-l-v2.0-ko `
  --embedding-batch-size 16 `
  --cpu-thread-count 10 `
  --progress-mode on
```
