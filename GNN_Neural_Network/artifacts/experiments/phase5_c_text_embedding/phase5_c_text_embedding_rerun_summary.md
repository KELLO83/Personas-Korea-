# Phase 5-C Text Embedding Feature Rerun Summary

Date: 2026-05-05

## Decision

KURE/KRUE text embedding feature is not promoted.

Default remains:

- Stage 1: popularity + cooccurrence candidate generation
- Stage 2: LightGBM learned ranker
- `include_text_embedding_feature=false`

## What Changed

- Missing persona context is tracked as coverage miss, not leakage failure.
- Leakage audit now uses the same Korean boundary pattern as masking.
- Stage 2 can score missing-context persons with an empty context and `text_embedding_similarity=0`.

## Runs

| Run | Scope | Recall@10 | NDCG@10 | Delta Recall@10 vs Stage1 | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| `kure_text_feature_002_context_coverage_gate` | context-covered only, Stage1 fallback for 19,677 persons | 0.681165 | 0.419520 | +0.000704 | not promoted |
| `kure_text_feature_003_full_ranker_fallback` | all persons, missing context uses text feature 0 | 0.679354 | 0.419567 | -0.001106 | rejected |
| `control_no_text_full_ranker_fallback` | all persons, no text feature | 0.677343 | 0.418952 | -0.003117 | control only |

Stage1 validation baseline in these runs:

- Recall@10: 0.680461
- NDCG@10: 0.419370

Closed Phase 2.5 selected baseline reference:

- Recall@10: 0.739051
- NDCG@10: 0.457970

## Reason

The text feature passed the corrected leakage gate, but current persona context coverage is only 211 audit-eligible validation persons. The context-covered run produced only a tiny validation gain and mostly fell back to Stage 1. The full-ranker run degraded Recall@10 versus Stage 1.

No test run was executed because the KURE text feature did not become the validation winner against the selected default.

## Next Step

Before another KURE feature ablation, regenerate or repair `person_context.csv` coverage so validation/test persons have split-aligned, leakage-masked persona context. Then rerun the same gate and compare again.
