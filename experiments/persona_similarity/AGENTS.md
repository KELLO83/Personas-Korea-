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
- Python-heavy CPU feature/evaluation loops should use multiprocessing/process
  pools by default because the GIL limits thread-level CPU parallelism. Keep
  native ML libraries such as LightGBM/CatBoost on their own
  `num_threads`/`thread_count` settings.
- Cache candidate pairs, pair features, text embeddings, text cosine features,
  splits, and trained models when metadata matches.
- Persist metrics, manual review samples, model metadata, cache metadata, and
  train/evaluation status under `artifacts/`.
