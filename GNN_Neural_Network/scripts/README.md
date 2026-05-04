# GNN Scripts Index

Run every command from the repository root with the project venv:

```powershell
.\.venv\Scripts\python.exe GNN_Neural_Network\scripts\<script>.py
```

## Default Pipeline Scripts

- `export_person_hobby_edges.py`: export Neo4j `Person-Hobby` edges and persona context.
- `train_lightgcn.py`: prepare splits and optionally train LightGCN.
- `train_ranker.py`: train the current LightGBM Stage 2 ranker.
- `evaluate_ranker.py`: evaluate the current default Stage 1 + Stage 2 path.
- `recommend_for_persona.py`: inspect one persona recommendation path.

## Evaluation And Ablation Scripts

- `evaluate_lightgcn.py`: evaluate LightGCN alone.
- `evaluate_reranker.py`: evaluate the legacy deterministic v1 reranker.
- `evaluate_stage1_ablation.py`: compare Stage 1 providers and combinations.
- `recommendation_quality_audit.py`: audit coverage, popularity bias, and qualitative samples.
- `sweep_reranker_weights.py`: historical deterministic reranker weight sweep.

## Phase 2.5 Closure Scripts

- `tune_ranker_regularization.py`: LightGBM regularization tuning.
- `ablation_neg_sampling.py`: negative sampling ratio/hardness ablation.
- `source_onehot_ablation.py`: provider source feature ablation.
- `sweep_mmr_lambda.py`: historical category one-hot MMR sweep.

## Data Quality / KURE Prep Scripts

- `generate_vocabulary_report.py`: vocabulary/canonicalization report.
- `build_canonicalization_candidates.py`: suggest canonicalization candidates.
- `auto_approve_candidates.py`: apply safe canonicalization approvals.
- `create_strict_taxonomy.py`: generate strict taxonomy mapping.
- `taxonomy_overmerge.py`: detect taxonomy/category over-merge risk.
- `leakage_check.py`: detect text leakage before embedding experiments.
- `phase5b_simulation.py`: KURE/text-embedding smoke probe only.

## Deprecated Or Probe Scripts

- `deprecated/phase25_baseline_trainer.py`: standalone/probe Phase 2.5 trainer.
- `deprecated/simple_phase25_trainer.py`: simplified standalone trainer.
- `deprecated/self_contained_trainer.py`: self-contained exploratory trainer.

Do not use deprecated/probe scripts as the default path unless an experiment
decision artifact explicitly promotes them.
