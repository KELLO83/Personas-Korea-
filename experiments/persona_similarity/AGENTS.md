# Similar-Persona Recommender Instructions

## Scope

This folder is only for the similar-persona recommender:

- Recommendation target: `Person -> Person`
- Training/evaluation unit: directed pair `source_uuid -> target_uuid`
- Main docs: `PRD.md`, `TASKS.md`, `DATASET_EXPLAIN.md`
- Artifacts: `artifacts/`

Do not put hobby recommendation (`Person -> Hobby`) experiments in this folder.

## Candidate and Reranking Policy

- Use Neo4j GDS FastRP/KNN `SIMILAR_TO` as the first candidate-generation
  baseline.
- A reranker should reorder exported candidates, not silently change candidate
  generation.
- For serious reranker experiments, export candidates with GDS `topK >= 50`.
  `topK=5` is smoke-test only.
- Compare learned rerankers against both raw `fastrp_score` ordering and the
  deterministic feature-score baseline.

## Data and Feature Rules

- Group ranking data by `source_uuid`; do not randomly split candidate-pair rows
  across train/validation/test.
- Do not use `source_uuid`, `target_uuid`, `display_name`, or raw text
  identifiers as model features.
- Raw Korean persona text should not be passed directly into LightGBM/tree
  models.
- Convert text into embedding cosine features such as `all_text_cosine`,
  `hobbies_text_cosine`, `career_text_cosine`, and `family_text_cosine`.
- Keep structured-only, text-only, structured+text, and hybrid experiments
  separated.
- Audit leakage when text fields restate hobbies, skills, occupations, or other
  structured graph attributes.

## Evaluation Policy

- Primary metrics: NDCG@K, explanation coverage, strong-reason coverage,
  low-information dominance, diversity, runtime, and model size.
- Do not promote a model from weak-label metrics alone; manual review is
  required.
- Treat text feature wins as `experimental` until manual review confirms
  semantically meaningful recommendations.

## Progress, Cache, and Artifacts

- Use one script per experiment purpose.
- Long-running export, feature building, embedding, training, and evaluation
  scripts must show progress.
- Inherit the root `.venv` Python 3.11 runtime.
- `.venv314t` may be used only for explicit local ML acceleration experiments
  after recording the Python executable, package versions, and cache identity.
  Backend/API/frontend paths must remain on `.venv` Python 3.11.
- In `.venv314t`, use already-exported local parquet/csv/npz artifacts and
  Polars-based processing only. Neo4j export, Excel export, pandas/openpyxl,
  and pyarrow utility paths belong to `.venv` Python 3.11.
- Inherit the root default CPU policy: use `18` threads/workers unless an
  experiment explicitly records a safer override.
- Python-heavy feature/evaluation loops should use `ThreadPoolExecutor`. This
  policy is applied consistently under both `.venv` Python 3.11 and `.venv314t`
  Python 3.14t to avoid duplicated worker memory and OOM shutdowns. Keep native
  ML libraries such as LightGBM on their own `num_threads` settings.
- Under verified `.venv314t` runs, the verified local stack for the default
  LightGBM/text path is `polars 1.37.1` with
  `polars-runtime-32-ft`, `lightgbm 4.6.0`, `torch 2.11.0+cu128`,
  `sentence-transformers 5.5.0`, `transformers 5.8.1`, and
  `tokenizers 0.23.0-rc0`.
- `shap` is not required for persona-similarity training/evaluation.
- Cache candidate pairs, pair features, text embeddings, text cosine features,
  splits, and trained models when metadata matches.
- Persist metrics, manual review samples, model metadata, cache metadata, and
  train/evaluation status under `artifacts/`.
