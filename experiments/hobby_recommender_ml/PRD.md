# PRD: Hobby Recommender ML

This PRD defines the active experiment contract for the `Person -> Hobby` recommender.

## Scope

The hobby recommender predicts hobbies for a persona. It owns offline training, evaluation, model-selection evidence, and experiment artifacts for `Person -> Hobby` only.

Out of scope:

- Similar-persona recommendation (`Person -> Person`), which belongs in `experiments/persona_similarity/`.
- Root FastAPI/Next.js product behavior, unless a promoted hobby model is explicitly consumed through an adapter.

## Current Architecture

The current documented architecture is a two-stage recommender:

1. Stage 1 builds a candidate pool from train-only popularity and hobby co-occurrence.
2. Stage 2 reranks candidates with LightGBM using structured features and semantic text-similarity features.

Current documented default from `artifacts/experiment_decisions.json`:

- Candidate generation: popularity + co-occurrence.
- Ranker: LightGBM with `num_leaves=31`.
- Semantic features: E5-small-ko-v2 single similarity plus domain-specific similarities.
- Embedding model: `dragonkue/multilingual-e5-small-ko-v2`.
- Current default artifact: `artifacts/experiments/phase5_c_text_embedding/e5_domain_features_validation_thread18/ranker_model.txt`.

## Promotion Requirements

A new default can be promoted only when all conditions are met:

- Validation and test metrics are recorded.
- The comparison uses the same split and candidate-pool policy or explicitly documents any split/pool change.
- Leakage masking and audit results are recorded when persona text or hobby text is used.
- Resource metadata records Python executable, device, thread count, batch size, and cache identity.
- The change is recorded in `artifacts/experiment_decisions.json` and summarized in a human-readable artifact under `artifacts/experiments/`.

## Current Default Metrics

| Metric | Test Value |
|---|---:|
| Recall@10 | 0.6809431703048724 |
| NDCG@10 | 0.4366648134158132 |
| Candidate Recall@50 | 0.8272082527401676 |
| Cold-start Recall@10 | 0.6944995912647437 |
| Cold-start NDCG@10 | 0.4444349889871841 |

## Known Caveat

The Phase 6 alias-based run has stronger stored test metrics, but it is not the documented production default until alias provenance, canonicalization risk, and candidate-text governance are approved.

## Implementation Boundaries

- Reusable experiment code belongs under `hobby_recommender/`.
- Experiment entrypoints belong under `scripts/`.
- Configurations belong under `configs/`.
- Generated metrics and models belong under `artifacts/experiments/<phase>/<run>/`.
- Do not place reusable experiment code under a nested `src/` folder because it conflicts conceptually with the root platform `src/`.

