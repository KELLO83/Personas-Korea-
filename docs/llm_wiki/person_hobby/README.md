# Person -> Hobby LLM Wiki Track

## Scope

This folder is only for the hobby recommendation ML experiment track.

```text
Task: Person -> Hobby
Project folder: GNN_Neural_Network/
Primary data: person_hobby_edges.csv, person_context.csv
Current default: popularity + cooccurrence candidates + LightGBM ranker
```

Do not put similar-persona (`Person -> Person`) experiment decisions here.

## Required Local Context

- `GNN_Neural_Network/AGENTS.md`
- `GNN_Neural_Network/DATASET_EXPLAIN.md`
- `GNN_Neural_Network/EXPERIMENTS.md`
- `GNN_Neural_Network/PRD.md`
- `GNN_Neural_Network/TASKS.md`
- `GNN_Neural_Network/README.md`
- `GNN_Neural_Network/artifacts/ranker_eval_metrics.json`

## Track Pages

- Current findings: `current_findings.md`
- Existing results: `results_summary.md`
- Code inventory: `code_inventory.md`
- Document inventory: `document_inventory.md`
- Experiment plan: `experiment_plan.md`

## Boundary Rule

Metrics here use hobby recommendation gates such as Recall@K, NDCG@K, candidate recall, coverage, novelty, runtime, and qualitative hobby quality. Do not compare them directly with similar-persona NDCG/manual-review metrics.
