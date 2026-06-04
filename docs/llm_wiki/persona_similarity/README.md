# Person -> Person LLM Wiki Track

## Scope

This folder is only for the similar-persona recommendation ML experiment track.

```text
Task: Person -> Person
Project folder: experiments/persona_similarity/
Training unit: directed source_uuid -> target_uuid pair
Current production: FastRP/KNN SIMILAR_TO + post-hoc explanation API
```

Do not put hobby (`Person -> Hobby`) experiment decisions here.

## Required Local Context

- `experiments/persona_similarity/AGENTS.md`
- `experiments/persona_similarity/DATASET_EXPLAIN.md`
- `experiments/persona_similarity/PRD.md`
- `experiments/persona_similarity/TASKS.md`
- `experiments/persona_similarity/README.md`
- `experiments/persona_similarity/artifacts/experiment_run_summary.md`
- `experiments/persona_similarity/artifacts/experiment_decisions.json`

## Track Pages

- Current findings: `current_findings.md`
- Existing results: `results_summary.md`
- Code inventory: `code_inventory.md`
- Document inventory: `document_inventory.md`
- Experiment plan: `experiment_plan.md`

## Boundary Rule

Metrics here use similar-persona gates such as NDCG@K, explanation coverage, strong-reason coverage, low-information dominance, diversity, runtime, model size, and manual review. Do not compare them directly with hobby Recall/Coverage/Novelty decisions.
