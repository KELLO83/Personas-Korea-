# GNN Experiment Map

This file is the navigation index for offline recommender experiments.
Use it when deciding which script, artifact, or document is authoritative.

## Current Default

```text
Stage 1 = popularity + cooccurrence
Stage 2 = LightGBM learned ranker
MMR     = false
```

Closed Phase 2.5 baseline:

```text
model = artifacts/experiments/phase2_5_num_leaves_31/ranker_model.txt
num_leaves=31
min_data_in_leaf=50
learning_rate=0.05
reg_alpha=0.1
reg_lambda=0.1
neg_ratio=4
hard_ratio=0.8
include_source_features=false
include_text_embedding_feature=false
```

Decision sources:

- `artifacts/experiment_decisions.json`: machine-readable accepted/rejected decisions
- `artifacts/experiment_run_summary.md`: human-readable current status summary
- `TASKS.md`: executable checklist and gates
- `PRD.md`: requirements, gates, and policy

## Active Entry Points

Use these for normal training/evaluation work.

| Purpose | Script | Notes |
| --- | --- | --- |
| Export graph edges/context | `scripts/export_person_hobby_edges.py` | Neo4j access point |
| Prepare splits / train LightGCN | `scripts/train_lightgcn.py` | `--prepare-only` is safe preprocessing |
| Evaluate LightGCN | `scripts/evaluate_lightgcn.py` | Historical/auxiliary provider |
| Train LightGBM ranker | `scripts/train_ranker.py` | Current Stage 2 model training |
| Evaluate LightGBM ranker | `scripts/evaluate_ranker.py` | Main validation/test evaluator |
| Recommend for one persona | `scripts/recommend_for_persona.py` | Use `--use-learned-ranker` for current default |
| Stage 1 ablation | `scripts/evaluate_stage1_ablation.py` | Provider comparison |
| Deterministic reranker eval | `scripts/evaluate_reranker.py` | Legacy v1 fallback comparison |

## Phase 2.5 Experiment Scripts

These scripts are part of default-decision closure and should be treated as
controlled experiments, not the default path.

| Experiment | Script | Status |
| --- | --- | --- |
| Regularization tuning | `scripts/tune_ranker_regularization.py` | Closed; `num_leaves=31` selected |
| Negative sampling ablation | `scripts/ablation_neg_sampling.py` | Closed; default remains `neg_ratio=4`, `hard_ratio=0.8` |
| Source one-hot ablation | `scripts/source_onehot_ablation.py` | Rejected; default remains `include_source_features=false` |
| MMR lambda sweep | `scripts/sweep_mmr_lambda.py` | Category one-hot MMR no-go |
| Reranker weight sweep | `scripts/sweep_reranker_weights.py` | Historical v1 path |

## Phase 5 / KURE-Related Scripts

KURE work must remain gated against the closed Phase 2.5 baseline.

| Experiment | Script | Status |
| --- | --- | --- |
| KURE MMR validation | `scripts/evaluate_ranker.py --use-mmr --mmr-embedding-method kure` | Phase 5-A no-go |
| Text embedding leakage check | `scripts/leakage_check.py` | Pre-KURE safety utility |
| Vocabulary report | `scripts/generate_vocabulary_report.py` | Data quality utility |
| Taxonomy overmerge check | `scripts/taxonomy_overmerge.py` | Data quality utility |
| Phase 5B simulation | `scripts/phase5b_simulation.py` | Probe / smoke-test only |

## Legacy And Compatibility Code

- `gnn_recommender/legacy/` exists only to support older import paths.
- New code should import from `GNN_Neural_Network.gnn_recommender.<module>`.
- Do not add new implementation logic under `gnn_recommender/legacy/`.
- Deprecated standalone probes live under `scripts/deprecated/`.
- Backup module copies should not live beside importable package modules.

## Artifact Layout

```text
artifacts/
  experiment_decisions.json
  experiment_run_summary.md
  experiments/
    phase2_5_num_leaves_31/
    phase2_5_neg_ratio_*/
    phase2_5_source_onehot/
    phase5_kure_mmr_lambda_*/
```

Rules:

- Every major validation run should write `validation_metrics.json`.
- Every gated run should write `validation_metrics.status.json`.
- Test metrics are winner-only unless the experiment explicitly says otherwise.
- Any default-path change must update `experiment_decisions.json`, `experiment_run_summary.md`, `README.md`, `PRD.md`, and `TASKS.md`.
- Artifact cleanup must preserve all metrics, status files, params, feature importance, summaries, model weights, and ranker datasets.
- Only generated caches (`cache/`, embedding cache directories, `.npy`, `.npz`, `__pycache__/`) are safe cleanup targets without explicit confirmation.
- If deleted experiment models or ranker datasets must be rebuilt, use `scripts/recover_deleted_experiment_artifacts.py` and follow `artifacts/experiments/RECOVER_DELETED_EXPERIMENTS.md`.
