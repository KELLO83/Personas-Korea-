# PRD: Korean Persona Knowledge Graph Insight Platform

This document is the current source of truth for the platform runtime. It supersedes older Korean planning documents and aligns the product scope with `requirements.txt`, `frontend/package.json`, and the live source tree.

## 1. Product Scope

The platform provides a knowledge-graph and RAG interface over the NVIDIA `Nemotron-Personas-Korea` persona dataset.

Current maintained product surfaces:

- FastAPI backend under `src/api/`.
- Neo4j graph access under `src/graph/`.
- Neo4j GDS services under `src/gds/`.
- LangChain/LangGraph RAG and chat under `src/rag/`.
- KURE-based embedding utilities under `src/embeddings/`.
- Next.js frontend under `frontend/`.
- Operational entrypoints under `ops/`.

Out-of-scope for the root platform contract:

- Model-training decisions for `Person -> Hobby`; those belong in `experiments/hobby_recommender_ml/`.
- Model-training decisions for `Person -> Person`; those belong in `experiments/persona_similarity/`.
- Historical Streamlit UI work; Streamlit is a legacy dependency, not the maintained production frontend.

## 2. Runtime Requirements

Backend/runtime Python:

- Python 3.14 through `.venv314`.
- Install with `./.venv314/Scripts/python.exe -m pip install -r requirements.txt` on Windows PowerShell.
- Run FastAPI with `./.venv314/Scripts/python.exe -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000`.

Frontend runtime:

- Node/npm project in `frontend/`.
- Next.js 16 and React 19 from `frontend/package.json`.
- Run with `npm run dev`; default port is `4000`.
- The frontend reads `NEXT_PUBLIC_API_BASE_URL`, defaulting to the FastAPI backend at `http://localhost:8000`.

Infrastructure:

- Neo4j container/service: `neo4j-personas`.
- Neo4j HTTP: `7474`.
- Neo4j Bolt: `7687`.
- Optional PostgreSQL/pgvector workflow through `psycopg[binary]` and `ops/vector/`.

## 3. Dependency Contract

The active Python package contract is `requirements.txt`.

Required runtime families:

- Data processing: `pandas`, `polars`, `pyarrow`, `openpyxl`, `datasets`.
- Graph database and analytics: `neo4j`, `graphdatascience`.
- Embeddings: `sentence-transformers`, `torch`.
- RAG and LLM orchestration: `langchain`, `langchain-community`, `langchain-neo4j`, `langchain-openai`, `langgraph`.
- Observability, opt-in: `langsmith`, `arize-phoenix-otel`, `arize-phoenix-client`, `opentelemetry-sdk`.
- API: `fastapi`, `uvicorn`.
- PostgreSQL/pgvector: `psycopg[binary]`.
- Testing and HTTP utilities: `pytest`, `httpx`, `tqdm`.

`streamlit` currently remains in `requirements.txt` for legacy compatibility. It must not be treated as the active frontend architecture.

## 4. API Contract

The FastAPI app registers route modules from `src/api/main.py`.

Required product capabilities:

- Search personas with structured filters.
- Return aggregate statistics and drilldowns.
- Return persona profile details by UUID.
- Return similar personas and graph/rule-based recommendation fallback data.
- Return community, influence, path, subgraph, graph quality, and graph insight data.
- Support chat/RAG workflows through LangGraph and LangChain integrations.
- Expose operations, readiness, warnings, and RAG trace surfaces for local debugging.

Model-backed recommendation output may be added only through a stable adapter contract after the relevant experiment folder records a promotion decision.

## 5. Frontend Contract

The maintained UI is the Next.js app under `frontend/`.

Required frontend capabilities:

- Call FastAPI only through HTTP API clients in `frontend/lib/`.
- Keep API response types aligned with `src/api/schemas.py` and frontend API types.
- Render dashboard, search, persona profile, graph, chat, operations, recommendation, and graph-insight workflows.
- Provide loading, empty, and error states for every remote call.
- Keep Korean end-user text acceptable in the UI where product copy targets Korean persona data; repository documentation should remain English.

## 6. Data and Graph Contract

Graph build and GDS operations must run outside FastAPI request handlers.

Current script entrypoints:

- `ops/graph/build_graph.py`
- `ops/graph/build_gds.py`
- `ops/graph/backfill_display_names.py`
- `ops/vector/build_embeddings.py`
- `ops/vector/load_pgvector_embeddings.py`
- `ops/data/download_dataset.py`
- `ops/data/preview_dataset.py`

Long-running operations must expose progress, preserve reusable artifacts where practical, and document cache/rebuild behavior in the relevant script or experiment folder.

## 7. Recommendation Boundaries

The repository contains two separate recommender tracks:

| Track | Location | Target | Root Platform Role |
|---|---|---|---|
| Hobby recommender | `experiments/hobby_recommender_ml/` | `Person -> Hobby` | Consume only promoted model artifacts through adapters. |
| Similar-persona recommender | `experiments/persona_similarity/` | `Person -> Person` | Consume only promoted model artifacts through adapters. |

Root API behavior must keep a graph/rule fallback when model artifacts are missing or not yet promoted.

## 8. Observability and Operations

The platform supports local-first observability:

- Internal trace interfaces live under `src/rag/tracing.py` and related API schemas.
- Phoenix and LangSmith packages are opt-in dependencies, not mandatory hosted services.
- Traces must avoid storing secrets, API keys, database credentials, or unnecessary personally identifying text.
- Operations endpoints should distinguish dependency failures, stale graph artifacts, missing GDS outputs, and LLM failures.

## 9. Documentation Policy

Active documentation must be English and current with the live repository.

- Do not translate obsolete plans without marking them as historical.
- Do not use archive PRDs as the implementation contract.
- When package or framework references conflict with `requirements.txt` or `frontend/package.json`, update the document or mark the item as legacy.
- Keep experiment documentation under its experiment folder.
