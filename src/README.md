# Runtime Source Map

`src/` is for reusable runtime code used by the API, graph, RAG, and platform
jobs. CLI entrypoints belong in `ops/`; model experiments belong in
`experiments/`.

## Packages

- `api/`: FastAPI application wiring, route modules, and API schemas.
- `config/`: environment-backed settings and configuration helpers.
- `data/`: dataset loading, cleaning, sampling, and preprocessing.
- `embeddings/`: KURE embedding model wrappers and vector index integrations.
- `gds/`: Neo4j GDS projections, FastRP, KNN, and community services.
- `graph/`: Neo4j loader, query helpers, analytics readers, and graph adapters.
- `jobs/`: reusable batch job orchestration used by platform workflows.
- `rag/`: LangChain/LangGraph chatbot and retrieval components.

## Placement Rules

- Add API-facing behavior under `api/`, keeping route handlers thin.
- Add graph database access under `graph/` or `gds/`, not inside route modules.
- Add reusable preprocessing under `data/`, not inside one-off scripts.
- Add experiment-only training, evaluation, or artifact code under
  `experiments/<experiment_name>/`.
- Add command-line orchestration under `ops/<domain>/` and import reusable
  logic from `src/`.
