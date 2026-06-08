# Korean Persona Knowledge Graph Insight Platform

This project turns the NVIDIA `Nemotron-Personas-Korea` dataset into a Neo4j knowledge graph and exposes search, analytics, recommendation, and RAG workflows through a FastAPI backend and a Next.js frontend.

The repository currently separates the production platform from recommender experiments:

- Platform runtime: `src/`, `ops/`, `frontend/`, root `PRD.md`, and `README.md`.
- Hobby recommendation experiment (`Person -> Hobby`): `experiments/hobby_recommender_ml/`.
- Similar-persona recommendation experiment (`Person -> Person`): `experiments/persona_similarity/`.

## Current Runtime Baseline

Use `.venv314` with Python 3.14 for the backend, graph, RAG, PostgreSQL, and platform scripts.

```powershell
.\.venv314\Scripts\python.exe -m pip install -r requirements.txt
.\.venv314\Scripts\python.exe -m pytest tests -q
.\.venv314\Scripts\python.exe -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

The frontend is the active Next.js application in `frontend/`.

```powershell
cd frontend
npm install
npm run dev
```

The default frontend port is `4000`; the backend defaults to `8000`.

## Local Infrastructure

- Neo4j runs in Docker as `neo4j-personas`.
  - HTTP: `http://localhost:7474`
  - Bolt: `bolt://localhost:7687`
- PostgreSQL/pgvector runs as the local PostgreSQL service, not a project Docker
  container.
  - URI: `postgresql://postgres:1234@localhost:5432/persona_vector`
  - Table: `persona_vectors`
- The old `pgvector-toeic` Docker container is not part of this project runtime.

## Dependency Baseline

The Python runtime is defined by `requirements.txt`:

- Core config: `python-dotenv`, `pydantic`, `pydantic-settings`.
- Data: `pandas`, `polars`, `pyarrow`, `openpyxl`, `datasets`.
- Graph: `neo4j`, `graphdatascience`.
- Embeddings: `sentence-transformers`, `torch`.
- RAG: `langchain`, `langchain-community`, `langchain-neo4j`, `langchain-openai`, `langgraph`.
- Observability, opt-in: `langsmith`, `arize-phoenix-otel`, `arize-phoenix-client`, `opentelemetry-sdk`.
- API: `fastapi`, `uvicorn`.
- PostgreSQL/vector storage: `psycopg[binary]`.
- Utilities: `tqdm`, `httpx`, `pytest`.

The frontend runtime is defined by `frontend/package.json`:

- Next.js `16.x`
- React `19.x`
- TypeScript `5.x`
- D3 / d3-force
- ECharts
- ESLint with Next config

## Current Source Layout

- `src/api/`: FastAPI app, route registration, request/response schemas, exception handling.
- `src/graph/`: Neo4j loading, query, search, stats, recommendation, and subgraph helpers.
- `src/gds/`: Neo4j GDS services for centrality, FastRP, similarity, and communities.
- `src/rag/`: LangGraph/LangChain chat, routing, Cypher/vector chains, LLM integration, tracing.
- `src/data/`: dataset loading, parsing, preprocessing, sampling, and parallel preprocessing.
- `src/embeddings/`: KURE embedding wrappers and Neo4j vector-index helpers.
- `src/jobs/`: reusable batch jobs, currently including centrality batch orchestration.
- `ops/data/`: dataset download and preview entrypoints.
- `ops/graph/`: graph build, display-name backfill, and GDS build entrypoints.
- `ops/vector/`: embedding build, pgvector schema, and pgvector load entrypoints.
- `ops/dev/`: local environment verification helpers.
- `frontend/`: maintained Next.js UI.

## Main API Surfaces

The FastAPI app registers these route groups from `src/api/main.py`:

- Insight and RAG: `/api/insight`, `/api/chat`, `/api/rag/*` trace routes.
- Persona lookup: `/api/persona/{uuid}`.
- Search and stats: `/api/search`, `/api/stats`.
- Similarity and recommendations: `/api/similar/{uuid}`, `/api/recommend/{uuid}`.
- Graph exploration: `/api/graph/subgraph/{uuid}`, `/api/path/{uuid1}/{uuid2}`.
- Communities and influence: `/api/communities`, `/api/influence/*`.
- Segment and advanced analytics: `/api/compare/segments`, target persona, lifestyle map, career transition, graph insights, graph quality.
- Operations: health, readiness, warnings, and operational status routes.

## Data and Graph Operations

Build or refresh graph data through `ops/graph/`:

```powershell
.\.venv314\Scripts\python.exe ops\graph\build_graph.py --help
.\.venv314\Scripts\python.exe ops\graph\build_gds.py --top-k 50
.\.venv314\Scripts\python.exe ops\graph\backfill_display_names.py --help
```

Embedding and pgvector workflows live under `ops/vector/`:

```powershell
.\.venv314\Scripts\python.exe ops\vector\build_embeddings.py --help
.\.venv314\Scripts\python.exe ops\vector\load_pgvector_embeddings.py --help
```

## Documentation Routing

- Root `PRD.md`: current platform product scope and runtime contracts.
- `README.md`: setup, source map, and operational entrypoints.
- `ops/docs/`: focused design notes and operations notes.
- `ops/decisions/`: architecture decision records.
- `experiments/persona_similarity/`: similar-persona recommender experiment documentation.
- `experiments/hobby_recommender_ml/`: hobby recommender experiment documentation.

