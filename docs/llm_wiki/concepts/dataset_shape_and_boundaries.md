# Dataset Shape And Boundaries

## Boundary Rule

Do not merge experiment decisions across recommender systems.

| Workspace | Task | Training unit | Main label |
|---|---|---|---|
| `GNN_Neural_Network/` | `Person -> Hobby` | person-hobby candidate row | held-out hobby edge |
| `experiments/persona_similarity/` | `Person -> Person` | directed `source_uuid -> target_uuid` pair | weak pair label/manual review |

## Current Local Shapes

| Artifact | Rows | Columns | Notes |
|---|---:|---:|---|
| `GNN_Neural_Network/data/person_hobby_edges.csv` | 50,000 | 2 | `person_uuid,hobby_name` |
| `GNN_Neural_Network/data/person_context.csv` | 50,000 | 21 | context and persona text fields |
| `experiments/persona_similarity/artifacts/datasets/candidate_pairs.parquet` | 2,500,000 | 29 | `SIMILAR_TO` topK=50 candidates |
| `experiments/persona_similarity/artifacts/datasets/pair_features.parquet` | 2,500,000 | 45 | structured pair features |
| `experiments/persona_similarity/artifacts/datasets/pair_features_with_text.parquet` | 2,500,000 | 53 | structured plus text cosine features |
| `experiments/persona_similarity/artifacts/datasets/persona_texts.parquet` | 50,000 | 12 | source persona text table |

## Synthetic Data Caveat

`nvidia/Nemotron-Personas-Korea` is synthetic. Offline metrics can be optimistic because persona text and structured attributes can encode stereotyped, internally consistent patterns that are easier than real user behavior.

## Text Leakage Caveat

Hobby text fields can restate held-out hobby names. Any hobby text experiment must use masking plus leakage audit before it can be compared to structured-only baselines.

For similar-persona recommendation, raw text is not used directly as tree-model input. Text must become embedding cosine features such as all-text or domain-specific cosine columns.
