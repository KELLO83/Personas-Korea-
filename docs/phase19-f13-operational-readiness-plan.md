# Phase 19: Operational Readiness Plan

This document is a current English readiness checklist for the platform runtime. It is not a deployment SLA.

## Scope

Validate the following surfaces before treating the local platform as ready for end-to-end demos:

- F10 influence APIs.
- F11 recommendation fallback APIs.
- F12 chat APIs.
- Next.js frontend flows.
- Neo4j/GDS graph artifacts.

## Environment Facts to Record

- OS and shell.
- Python executable and package versions from `.venv314`.
- Neo4j version, GDS version, heap, and page-cache settings.
- Dataset identity and sample size.
- CPU/GPU device information when embeddings or ML inference are used.
- Frontend Node/npm versions.

## Graph Readiness Checks

Check entity counts, relationship counts, `SIMILAR_TO` availability, centrality properties, and community properties before running user-facing recommendation or influence flows.

GDS-heavy jobs must be run from ops entrypoints or scheduled jobs, not from FastAPI request handlers.

## API Smoke Targets

- `GET /api/stats`
- `GET /api/search`
- `GET /api/persona/{uuid}`
- `GET /api/recommend/{uuid}`
- `GET /api/influence/top`
- `POST /api/chat`
- Operations/readiness routes

## Failure Scenarios

| Failure | Expected Behavior |
|---|---|
| Missing GDS projection | Return a not-ready response; do not rebuild inside the request. |
| Missing similarity relationships | Return a clear recommendation not-ready state. |
| Stale centrality properties | Surface stale metadata while keeping the last valid result when available. |
| Invalid UUID | Return validation or not-found errors through the FastAPI exception layer. |
| LLM unavailable | Return deterministic fallback or explicit LLM failure metadata. |

## Validation Commands

```powershell
.\.venv314\Scripts\python.exe -m pytest tests -q
cd frontend
npm run typecheck
npm run lint
npm run build
```
