# TASKS: Hobby Recommender ML

This task list tracks the active `Person -> Hobby` experiment state.

## Current State

- [x] Folder moved to `experiments/hobby_recommender_ml/`.
- [x] Package renamed around `hobby_recommender/` instead of the older GNN naming.
- [x] Stage 1 default documented as popularity + co-occurrence.
- [x] Stage 2 documented default recorded as E5-small-ko-v2 domain-specific LightGBM.
- [x] `artifacts/experiment_decisions.json` contains machine-readable model decisions.
- [x] Root platform docs no longer own model-training decisions.

## Active Follow-Ups

- [ ] Re-run the experiment test suite after the folder move is fully staged.
- [ ] Confirm every CLI entrypoint works from the new `experiments/hobby_recommender_ml/` path.
- [ ] Replace stale historical path strings in old artifact summaries when those summaries are used again.
- [ ] Audit Phase 6 alias-based candidate text before any default promotion.
- [ ] Keep README/PRD synchronized with `artifacts/experiment_decisions.json` when a default changes.

## Useful Validation Commands

```powershell
.\.venv314\Scripts\python.exe -m pytest experiments\hobby_recommender_ml\tests -q
.\.venv314\Scripts\python.exe experiments\hobby_recommender_ml\scripts\evaluate_ranker.py --help
.\.venv314\Scripts\python.exe experiments\hobby_recommender_ml\scripts\recommend_for_persona.py --help
```

## Do Not Mix

- Do not mix `Person -> Hobby` metrics with `Person -> Person` metrics.
- Do not write hobby model decisions into root platform docs.
- Do not promote alias, category, or description-based candidate text without leakage and provenance review.

