# KURE Text Feature 005 Domain-Tagged Pilot Summary

Date: 2026-05-16

## Scope

This is a fast 2K-person validation pilot, not a promotion-grade full validation run.

Common settings:

- `max_persons=2000`
- Stage 1: `popularity + cooccurrence`
- Stage 2: LightGBM, `num_leaves=31`, `neg_ratio=4`, `hard_ratio=0.8`
- Text input: `build_domain_tagged_persona_text`
- Masking/audit: `mask_holdout_hobbies` before encoding, post-mask audit persisted
- Cache policy: model/preprocessing-aware cache keys, `domain_tagged_masked_v1`

## Governance Result

- Context coverage report: train/validation/test domain text coverage = `1.0`
- Leakage audit: `passed_person_count=10857`, `failed_person_count=0`, `missing_context_person_count=0` for the full validation candidate population used by the train script audit
- KURE cache/device plan: CUDA, auto batch from free VRAM

## Same-2K Validation Comparison

| Run | Recall@10 | NDCG@10 | Coverage@10 | Novelty@10 | Candidate Recall@50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stage1 baseline | 0.576500 | 0.358540 | 0.002935 | 4.538293 | 0.826000 |
| No-text LightGBM pilot | 0.620000 | 0.380264 | 0.002802 | 4.603208 | 0.826000 |
| KURE text LightGBM pilot | 0.636500 | 0.390696 | 0.004003 | 4.688600 | 0.826000 |

Delta KURE text vs no-text pilot:

- Recall@10: `+0.016500`
- NDCG@10: `+0.010432`
- Coverage@10: `+0.001201`
- Novelty@10: `+0.085392`

## Decision

Status: `needs_full_validation_followup`.

The domain-tagged KURE text feature now shows a positive same-sample pilot signal after context coverage and cache governance hardening. It is not promoted because this was a 2K pilot and coverage remains far below the ranking-collapse diversity target.

Next step: run the same optimized lookup path on full validation, then compare against the closed Phase 2.5 baseline and the same-run no-text control before any test split execution.
