# Persona Similarity Tasks

This document breaks the experiment plan in `experiments/persona_similarity/PRD.md` into an executable checklist.

Current status:

- The code scaffold is mostly complete.
- Large-scale experiment runs are still incomplete.
- The default production behavior remains `FastRP/KNN -> SIMILAR_TO`.
- The LightGBM reranker is not yet a promotion/integration target.

## Phase 0 - Workspace Setup

- [x] Create `experiments/persona_similarity/` workspace.
- [x] Add `AGENTS.md`.
- [x] Add `DATASET_EXPLAIN.md`.
- [x] Add `PRD.md`.
- [x] Add `TASKS.md`.
- [x] Add `README.md` with command examples.
- [x] Add `configs/lightgbm_reranker.yaml`.
- [x] Add optional experiment dependencies in `requirements.txt`.
- [x] Add unit tests under `tests/`.

## Phase 1 - Dataset Inspection and Export

- [x] Implement `scripts/export_personas_excel.py` for current Neo4j DB inspection.
- [x] Export current Neo4j persona snapshot to local Excel for data-shape review.
- [x] Add `.gitignore` rules so exported Excel/dataset artifacts are not committed.
- [x] Implement `scripts/export_pairs.py` for `SIMILAR_TO` candidate-pair export.
- [x] Export source/target demographic, location, occupation, education, family, housing, community, shared hobbies, and shared skills.
- [x] Persist candidate-pair output path: `artifacts/datasets/candidate_pairs.parquet`.
- [x] Persist export metadata path: `artifacts/metrics/export_status.json`.
- [ ] Rebuild GDS `SIMILAR_TO` with `topK >= 50` before serious reranker experiments.
- [ ] Export fresh `topK >= 50` candidate pairs from Neo4j.

## Phase 2 - Pair Feature Dataset

- [x] Implement deterministic pair feature builder.
- [x] Convert candidate-pair rows to numeric feature matrix.
- [x] Generate weak relevance labels.
- [x] Split by `source_uuid`, not by candidate-pair row.
- [x] Persist train/valid/test split column and split metadata.
- [x] Add feature builder unit tests.
- [x] Add cache metadata and cache reuse for feature-building stages.
- [ ] Build full `topK >= 50` pair feature dataset.

## Phase 3 - Baseline Evaluation Code

- [x] Implement raw FastRP/KNN baseline evaluation script.
- [x] Implement deterministic feature-score baseline evaluation script.
- [x] Keep FastRP and deterministic baseline evaluation as independent scripts.
- [x] Compute weak-label NDCG@5/10.
- [x] Compute explanation coverage@5/10.
- [x] Compute strong-reason, low-information, average-reason, diversity, and overlap metrics.
- [x] Export manual review samples.
- [x] Add cache reuse for evaluation outputs.

## Phase 4 - Structured LightGBM Reranker Code

- [x] Implement shared LightGBM training utilities.
- [x] Implement legacy aggregate `scripts/train_reranker.py`.
- [x] Implement legacy aggregate `scripts/evaluate_reranker.py`.
- [x] Add independent `train_lambdarank.py` and `evaluate_lambdarank.py`.
- [x] Add independent `train_rank_xendcg.py` and `evaluate_rank_xendcg.py`.
- [x] Train/evaluate by group `source_uuid`.
- [x] Persist model to `artifacts/models/`.
- [x] Persist train metadata and feature list.
- [x] Add cache reuse for model training/evaluation.

## Phase 5 - Structured Ablation and Hybrid Code

- [x] Add independent FastRP-feature ablation train/evaluate scripts.
- [x] Add independent low-information-feature ablation train/evaluate scripts.
- [x] Add independent location-feature ablation train/evaluate scripts.
- [x] Add independent hobby-feature ablation train/evaluate scripts.
- [x] Add independent hybrid-score evaluation script.
- [x] Add independent diversity/final-rerank evaluation script.
- [x] Add occupation/province/community diversity and demographic-only metrics.
- [x] Persist metrics to `artifacts/metrics/`.
- [x] Define decision artifact paths:
  - `artifacts/experiment_decisions.json`
  - `artifacts/experiment_run_summary.md`
- [ ] Run structured baselines and rerankers on full `topK >= 50` candidate data.
- [ ] Compare reranker vs raw FastRP/KNN order.
- [ ] Compare reranker vs deterministic score baseline.
- [ ] Compare hybrid score variants.
- [ ] Compare diversity rerank variants.
- [ ] Update decision artifacts after real experiment runs.

## Phase 6 - Text Embedding Feature Code

- [x] Implement `scripts/export_persona_texts.py`.
- [x] Implement `scripts/audit_text_feature_leakage.py`.
- [x] Implement `scripts/build_text_embeddings.py`.
- [x] Implement `scripts/build_text_features.py`.
- [x] Persist persona text corpus metadata.
- [x] Persist embedding metadata including model, device, batch size, preprocessing version, and runtime.
- [x] Add text cosine feature columns:
  - `all_text_cosine`
  - `persona_text_cosine`
  - `professional_text_cosine`
  - `hobbies_text_cosine`
  - `skills_text_cosine`
  - `career_text_cosine`
  - `family_text_cosine`
  - `lifestyle_text_cosine`
- [x] Add cache reuse for persona text export, text embeddings, leakage audit, and text feature matrix.
- [ ] Build text embeddings for full candidate dataset.
- [ ] Build structured+text feature dataset.
- [ ] Run leakage audit on the actual text feature dataset.

## Phase 7 - Text Model Experiment Code

- [x] Add independent text-only LambdaRank train/evaluate scripts.
- [x] Add independent structured+text LambdaRank train/evaluate scripts.
- [x] Add independent structured+text rank_xendcg train/evaluate scripts.
- [x] Add independent structured+text hybrid evaluation script.
- [ ] Run text-only LambdaRank experiment.
- [ ] Run structured+text LambdaRank experiment.
- [ ] Run structured+text rank_xendcg experiment.
- [ ] Run structured+text hybrid score comparison.
- [ ] Review text feature manual samples before any promotion decision.

## Phase 8 - Candidate Expansion Code

- [ ] Implement PPR candidate-generation comparison only if FastRP/KNN candidate recall is insufficient.
- [ ] Implement Node2Vec candidate-generation comparison only if FastRP/KNN candidate recall is insufficient.

## Phase 8-B - Lessons From Hobby Recommender

Apply the same controlled 2-stage policy learned from `experiments/hobby_recommender_ml/`.

- [ ] Keep Stage1 as `FastRP/KNN topK >= 50` until a decision artifact says candidate recall is insufficient.
- [ ] Treat KURE/Snowflake text embeddings as Stage2 reranker features first, not as a new Stage1 candidate generator.
- [ ] Do not change candidate pool, split, label policy, LightGBM config, embedding backbone, and persona text builder in the same experiment.
- [ ] Record exactly one changed variable in every decision artifact.

### Track A - Embedding Backbone Swap

- [ ] Define KURE-v1 as the reference persona-pair text embedding backbone.
- [ ] Run `dragonkue/snowflake-arctic-embed-l-v2.0-ko` as a validation-only backbone swap with the same candidate pool, split, text builder, labels, and LightGBM config.
- [ ] Run `dragonkue/multilingual-e5-small-ko-v2` only as an optional speed/cost probe after the Snowflake result is known.
- [ ] Persist `model_name`, `model_revision`, embedding dimension, pooling behavior when known, device, batch size, runtime, cache hit/miss, and preprocessing version.
- [ ] Verify KURE-v1, Snowflake-ko, and E5-small-ko embedding caches cannot collide.
- [ ] Promote to test only if validation NDCG@5/10 improves without reducing explanation coverage or strong-reason rate.

### Track D - Persona Text Builder Ablation

- [ ] Keep the embedding backbone fixed to KURE-v1 for Track D.
- [ ] Define and version the persona text builder interface.
- [ ] Evaluate `persona_text_structured_only`.
- [ ] Evaluate `persona_text_narrative_only`.
- [ ] Evaluate `persona_text_structured_plus_narrative`.
- [ ] Evaluate `persona_text_domain_tagged_blocks`.
- [ ] Evaluate `persona_text_summary_style` only if the summary source is reproducible and leakage-audited.
- [ ] Persist 20 source/target text examples for manual review per builder.
- [ ] Run leakage audit for every builder before training.
- [ ] Do not combine Track D with Track A until both isolated effects have validation artifacts.

### Track B - Domain-Specific Text Cosine

- [ ] Keep the candidate pool and embedding backbone fixed.
- [ ] Add or verify domain-specific cosine columns:
  - `professional_text_cosine`
  - `hobbies_text_cosine`
  - `skills_text_cosine`
  - `career_text_cosine`
  - `family_text_cosine`
  - `lifestyle_text_cosine`
  - `persona_text_cosine`
- [ ] Compare domain-specific text features against the single `all_text_cosine` baseline.
- [ ] Persist feature importance and explanation-card coverage for domain text features.

### Final Rerank - Diversity And Explanation

- [ ] Run diversity/final rerank only after the structured+text reranker baseline is known.
- [ ] Track same-occupation, same-region, same-community, and low-information overconcentration.
- [ ] Require manual review samples before any promotion decision.

## Phase 9 - Promotion Gate

- [ ] Confirm candidate generation uses `topK >= 50`, not smoke-test `topK=5`.
- [ ] Confirm no random row-level split leakage.
- [ ] Confirm no raw `uuid`, `display_name`, or raw text identifier features are used.
- [ ] Confirm FastRP baseline, deterministic baseline, structured reranker, text reranker, and hybrid scores are compared on the same split.
- [ ] Confirm final reranking does not trade away too much NDCG/strong-reason coverage for diversity.
- [ ] Confirm manual review quality before promotion.
- [ ] Confirm runtime and memory costs are acceptable.
- [ ] Confirm rollback path to raw FastRP/KNN ordering.
- [ ] Update `artifacts/experiment_decisions.json` only after a real experiment decision.
- [ ] Update `artifacts/experiment_run_summary.md` only after a real experiment decision.
- [ ] Update root platform docs only if API/frontend/production integration changes.

## Current Recommended Execution Order

Run one experiment-purpose script at a time.

```text
1. ops/graph/build_gds.py --top-k 50
2. experiments/persona_similarity/scripts/export_pairs.py
3. experiments/persona_similarity/scripts/build_features.py
4. experiments/persona_similarity/scripts/evaluate_fastrp_baseline.py
5. experiments/persona_similarity/scripts/evaluate_deterministic_baseline.py
6. experiments/persona_similarity/scripts/train_lambdarank.py
7. experiments/persona_similarity/scripts/evaluate_lambdarank.py
8. experiments/persona_similarity/scripts/train_rank_xendcg.py
9. experiments/persona_similarity/scripts/evaluate_rank_xendcg.py
10. experiments/persona_similarity/scripts/evaluate_hybrid_score.py
11. experiments/persona_similarity/scripts/evaluate_diversity_rerank.py
12. experiments/persona_similarity/scripts/export_persona_texts.py
13. experiments/persona_similarity/scripts/audit_text_feature_leakage.py
14. experiments/persona_similarity/scripts/build_text_embeddings.py
15. experiments/persona_similarity/scripts/build_text_features.py
16. text-only / structured+text / hybrid text experiments
17. Snowflake-ko backbone swap, validation-only, same pool/split/text builder
18. persona text builder ablation with KURE-v1 fixed
19. domain-specific text cosine ablation
20. diversity / explanation-aware final rerank
21. optional PPR/Node2Vec candidate-generation comparison only if needed
22. manual review
23. decision artifact update
```

## Current Decision

Status: the code scaffold is implemented, but the full experiment suite has not been run.

Default production behavior remains:

```text
FastRP/KNN -> SIMILAR_TO
```

No LightGBM reranker is promoted or integrated yet.
