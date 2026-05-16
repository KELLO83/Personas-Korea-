# GNN Hobby Recommender Instructions

## Scope

This folder is only for the hobby recommender:

- Recommendation target: `Person -> Hobby`
- Main docs: `PRD.md`, `TASKS.md`, `DATASET_EXPLAIN.md`
- Reranker checklist: `CHECKLIST_GNN_Reranker_v2.md`
- Artifacts: `artifacts/`

Do not put similar-persona (`Person -> Person`) experiments in this folder.

## Models and Evaluation

- Current PoC direction: LightGCN/XSimGCL candidate generation plus LightGBM
  reranker.
- Preserve deterministic evaluation: fixed split, fixed seed, same candidate
  pool, and same known-hobby masking rules.
- Primary metrics: Recall@K, NDCG@K, candidate recall, runtime, and qualitative
  hobby quality.
- Do not promote a model unless it beats the selected baseline on the agreed
  validation/test protocol.

## Performance Rules

- Use CUDA automatically when available, with CPU fallback.
- Keep batch size, chunk size, candidate pool size, worker count, and device
  configurable via YAML or CLI.
- Cache reused LightGCN/XSimGCL embeddings, adjacency tensors, popularity counts,
  and co-occurrence counts.
- Avoid repeated graph propagation or embedding recomputation in tight loops.
- Prefer batched GPU scoring/masking/top-k over per-person model calls.
- Avoid duplicate LightGCN propagation for positive and negative scores in the
  same batch.

## Progress and Cache

- Long-running training, evaluation, candidate generation, embedding computation,
  feature building, and batch scoring must show progress.
- Cache expensive stages and make scripts restartable from the latest valid
  artifact.
- Record cache hits, rebuild reasons, device, worker count, batch/chunk size,
  runtime, and peak GPU memory when practical.

## Experiment Artifacts

Persist raw metrics plus decision artifacts:

- `artifacts/experiment_decisions.json`
- `artifacts/experiment_run_summary.md`

For each major experiment, record:

- tested model/provider/feature/taxonomy change
- selected baseline
- Recall/NDCG/candidate recall deltas
- status: `accepted`, `rejected`, `promoted`, `disabled`, `experimental`, or
  `needs_followup`
- short decision reason

If default recommendation behavior changes, update `README.md`, `PRD.md`, and
`TASKS.md` in the same task.
