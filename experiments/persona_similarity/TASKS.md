# Persona Similarity Tasks

## Phase 0 - Workspace Setup

- [x] Create `experiments/persona_similarity/` workspace.
- [x] Add `DATASET_EXPLAIN.md`.
- [x] Add `PRD.md`.
- [x] Add `TASKS.md`.
- [x] Add initial README with command examples.

## Phase 1 - Dataset Export

- [x] Implement `scripts/export_pairs.py`.
- [ ] Export existing `SIMILAR_TO` candidate pairs from Neo4j.
- [x] Include source/target demographic, location, occupation, education, family, housing, community, shared hobbies, shared skills.
- [x] Persist output to `artifacts/datasets/candidate_pairs.parquet`.
- [x] Persist export metadata to `artifacts/metrics/export_status.json`.

## Phase 2 - Feature Builder

- [x] Implement deterministic pair feature builder.
- [x] Convert pair rows to numeric feature matrix.
- [x] Generate weak relevance labels.
- [x] Split by `source_uuid`, not row.
- [x] Persist train/valid/test split column and split metadata.
- [x] Add feature builder unit tests.

## Phase 3 - Baseline Evaluation

- [x] Evaluate raw FastRP/KNN candidate ordering.
- [x] Evaluate deterministic feature-score ordering.
- [x] Split FastRP and deterministic baseline evaluation into independent scripts.
- [x] Compute weak-label NDCG@5/10.
- [x] Compute explanation coverage@5/10.
- [x] Compute strong-reason, low-information, average-reason, diversity metrics.
- [x] Export manual review sample.

## Phase 4 - LightGBM Reranker

- [x] Implement `scripts/train_reranker.py`.
- [x] Train LambdaRank with group by `source_uuid`.
- [x] Persist model to `artifacts/models/`.
- [x] Persist train metadata and feature list.
- [x] Add independent `train_lambdarank.py` and `evaluate_lambdarank.py`.
- [x] Add independent `train_rank_xendcg.py` and `evaluate_rank_xendcg.py`.
- [x] Add independent ablation train/evaluate scripts for FastRP, low-info, location, and hobby features.

## Phase 5 - Evaluation and Decision

- [x] Implement `scripts/evaluate_reranker.py`.
- [x] Compare reranker vs raw FastRP/KNN order.
- [x] Compare hybrid score variants.
- [x] Add independent hybrid-score evaluation script.
- [x] Persist metrics to `artifacts/metrics/`.
- [x] Update `artifacts/experiment_decisions.json`.
- [x] Update `artifacts/experiment_run_summary.md`.

## Phase 6 - Text Embedding Features

- [x] Implement `scripts/export_persona_texts.py`.
- [x] Implement `scripts/audit_text_feature_leakage.py`.
- [x] Implement `scripts/build_text_embeddings.py`.
- [x] Implement `scripts/build_text_features.py`.
- [x] Persist persona text corpus metadata.
- [x] Persist embedding metadata including model, device, batch size, preprocessing version, and runtime.
- [x] Add text feature columns: `all_text_cosine`, `persona_text_cosine`, `professional_text_cosine`, `hobbies_text_cosine`, `skills_text_cosine`, `career_text_cosine`, `family_text_cosine`, `lifestyle_text_cosine`.

## Phase 7 - Text Model Experiments

- [x] Add independent text-only LambdaRank train/evaluate scripts.
- [x] Add independent structured+text LambdaRank train/evaluate scripts.
- [x] Add independent structured+text rank_xendcg train/evaluate scripts.
- [x] Add independent structured+text hybrid evaluation script.
- [ ] Run text feature experiments.
- [ ] Review text feature manual samples before any promotion decision.

## Current Decision

Status: workspace initialized, experiment not run.

Default production behavior remains:

```text
FastRP/KNN -> SIMILAR_TO
```

No LightGBM reranker is promoted or integrated yet.
