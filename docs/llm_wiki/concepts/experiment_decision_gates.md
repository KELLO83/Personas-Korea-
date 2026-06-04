# Experiment Decision Gates

## Global Gates

- Use `.venv` Python 3.11 for backend, Neo4j export, Excel, pandas/openpyxl/pyarrow utility scripts.
- Use `.venv314t` only for explicitly recorded local ML acceleration over already-exported parquet/csv/npz artifacts.
- Record candidate width, split policy, worker/thread count, device, runtime, and cache status.
- Keep one script per experiment purpose.
- Do not promote from weak labels alone.
- Do not change production behavior without folder-specific docs and manual-review requirements.

## Hobby Recommendation Gates

Primary metrics:

- Recall@K
- NDCG@K
- candidate recall
- coverage
- novelty
- runtime
- qualitative hobby quality

Promotion requirement:

```text
beats selected baseline on agreed validation/test protocol
does not regress coverage/novelty beyond accepted tolerance
passes known-hobby masking and text-leakage rules
```

## Similar-Persona Gates

Primary metrics:

- NDCG@K
- explanation coverage
- strong-reason coverage
- low-information dominance
- diversity
- runtime
- model size
- manual review

Promotion requirement:

```text
candidate width topK >= 50
source-disjoint train/valid/test split
no raw UUID/display_name/raw text identifier features
manual review approved
rollback path remains FastRP/KNN ordering via fastrp_score
```

## No-Go Conditions

- topK=5 smoke candidates used for serious reranker comparison.
- Raw Korean text passed directly into LightGBM/tree models.
- Session, sequence, bandit, or reinforcement-learning method promoted without real interaction logs.
- Direct metric comparison between `Person -> Hobby` and `Person -> Person`.
