# Hobby Recommender ML

This experiment owns the `Person -> Hobby` recommender. It is separate from the root FastAPI/Next.js platform and from `experiments/persona_similarity/`.

## Current Documented Default

Based on `artifacts/experiment_decisions.json`, the current documented default is:

- Stage 1: popularity + co-occurrence candidate generation.
- Stage 2: LightGBM ranker with E5-small-ko-v2 single and domain-specific text similarity features.
- Embedding model: `dragonkue/multilingual-e5-small-ko-v2`.
- Default model path: `artifacts/experiments/phase5_c_text_embedding/e5_domain_features_validation_thread18/ranker_model.txt`.
- Test metrics path: `artifacts/experiments/phase5_c_text_embedding/e5_domain_features_test_thread18/test_metrics.json`.

Latest recorded test metrics for the documented default:

| Metric | Value |
|---|---:|
| Recall@10 | 0.6809431703048724 |
| NDCG@10 | 0.4366648134158132 |
| Candidate Recall@50 | 0.8272082527401676 |
| Cold-start Recall@10 | 0.6944995912647437 |
| Cold-start NDCG@10 | 0.4444349889871841 |

Phase 6 has stronger stored test artifacts, but the alias-based candidate text path needs provenance and governance approval before it can replace the documented default.

## Folder Map

- `hobby_recommender/`: reusable experiment code.
- `scripts/`: training, evaluation, ablation, audit, and export entrypoints.
- `configs/`: experiment configuration files.
- `tests/`: experiment-specific tests.
- `artifacts/`: metrics, model files, decision logs, and historical run summaries.
- `data/`: local split/export data when present.

## Runtime

Use the project runtime described by root `AGENTS.md`. For experiment acceleration, `.venv314t` is allowed only for paths that have been verified under free-threaded Python.

Install experiment-only dependencies when needed:

```powershell
.\.venv314\Scripts\python.exe -m pip install -r experiments\hobby_recommender_ml\requirements-hobby-recommender.txt
```

Typical commands:

```powershell
.\.venv314\Scripts\python.exe experiments\hobby_recommender_ml\scripts\train_ranker.py --config experiments\hobby_recommender_ml\configs\lightgbm_ranker.yaml
.\.venv314\Scripts\python.exe experiments\hobby_recommender_ml\scripts\evaluate_ranker.py --config experiments\hobby_recommender_ml\configs\lightgbm_ranker.yaml
.\.venv314\Scripts\python.exe experiments\hobby_recommender_ml\scripts\recommend_for_persona.py --help
```

## Decision Policy

- Keep Stage 1 and Stage 2 decisions in this folder.
- Do not update root API behavior unless a promoted artifact is recorded here.
- Keep `artifacts/experiment_decisions.json` as the machine-readable source for model decisions.
- Keep run summaries under `artifacts/experiments/` as historical records.

