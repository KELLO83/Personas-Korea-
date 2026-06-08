# GNN Hobby Recommender Instructions

## Scope

This folder is only for the hobby recommender:

- Recommendation target: `Person -> Hobby`
- Main docs: `PRD.md`, `TASKS.md`, `DATASET_EXPLAIN.md`
- Reranker checklist: `CHECKLIST_HOBBY_Reranker_v2.md`
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

- Default runtime is the root `.venv314` Python 3.14 environment.
- `.venv314t` may be used only for explicit local ML acceleration experiments
  after recording the package versions and Python executable in the artifact.
  Do not use `.venv314t` for production/backend integration.
- In `.venv314t`, use already-exported local artifacts and Polars/NumPy-based
  processing only. Do not require pandas, pyarrow, openpyxl, Neo4j, CatBoost,
  SHAP, kiwipiepy, or datasets for the default GNN recommender ML path.
- Inherit the root default CPU policy: use `18` threads/workers unless an
  experiment explicitly records a safer override.
- Use CUDA automatically when available, with CPU fallback.
- Keep batch size, chunk size, candidate pool size, worker count, and device
  configurable via YAML or CLI.
- Python-heavy feature row builds, candidate feature transforms, cache
  post-processing, and evaluation transforms should use `ThreadPoolExecutor`.
  This policy is applied consistently under both `.venv314` Python 3.14 and
  `.venv314t` Python 3.14t to avoid duplicated worker memory and OOM shutdowns.
- Under verified `.venv314t` runs, the local verified stack is
  `torch 2.11.0+cu128`, `sentence-transformers 5.5.0`,
  `transformers 5.8.1`, `tokenizers 0.23.0-rc0`, `polars 1.37.1` with
  `polars-runtime-32-ft`, and `lightgbm 4.6.0`.
- `shap` is not required for current GNN recommender training/evaluation.
  Explanation reasons use LightGBM `pred_contrib=True`; keep external SHAP
  optional unless code is explicitly changed to import it.
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

