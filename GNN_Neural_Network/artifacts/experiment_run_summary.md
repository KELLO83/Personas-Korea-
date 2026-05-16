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

The closed Phase 2.5 default remains the comparison baseline for all later KURE dense MMR or text embedding feature work.

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

Current SOTA remains the closed Phase 2.5 LightGBM ranker:

| Run | Split | Recall@10 | NDCG@10 | Candidate Recall@50 | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| `phase2_5_num_leaves_31` | test | 0.709684 | 0.447713 | 0.977136 | current SOTA/default |
| `kure_text_feature_005_domain_tagged_20k_cpu10_test_matrix_retry` | test | 0.617482 | 0.386258 | 0.827208 | not promoted |

KURE-v1 domain-tagged text features showed a real Stage2 signal under the matched current candidate pool:

- test delta vs its own Stage1: Recall@10 `+0.047711`, NDCG@10 `+0.029900`
- full validation KURE text vs matched no-text control: Recall@10 `+0.043014`, NDCG@10 `+0.030504`

Decision:

- `include_text_embedding_feature=false` remains default.
- Current KURE text path is **not SOTA** and is **not promoted**.
- KURE-v1 remains follow-up-only as an auxiliary Stage2 feature.
- One final validation-only matched-control follow-up is allowed under the same current code/split/candidate pool. It must show progress (`--progress-mode on`) and record CPU/GPU/cache policy. Test runs remain winner-only.

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

- KURE-v1 text feature is useful signal, but it is not the current SOTA.
- Do not promote KURE-v1 into the default recommendation path.
- Keep `phase2_5_num_leaves_31` as the documented SOTA/default reference and move product/default integration forward.
