# User and Operator Guide

This guide reflects the current root FastAPI backend and the maintained Next.js frontend.

## 1. Runtime

- Python backend: `.venv314` with Python 3.14.
- Backend command: `./.venv314/Scripts/python.exe -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000`.
- Frontend command: `cd frontend && npm install && npm run dev`.
- Frontend API setting: `NEXT_PUBLIC_API_BASE_URL`.

## 2. Main User Flows

- Search/filter personas: `GET /api/search`.
- View aggregate statistics: `GET /api/stats` and dimension-specific stats routes.
- View persona details: `GET /api/persona/{uuid}`.
- Explore chat/RAG: `POST /api/chat`.
- View recommendations: `GET /api/recommend/{uuid}`.
- View influence data: `GET /api/influence/top`.
- View relationship paths: `GET /api/path/{uuid1}/{uuid2}`.
- Compare segments: `POST /api/compare/segments`.
- Inspect operations and readiness: operations route group under `src/api/routes/operations.py`.

## 3. Advanced Analysis Notes

- `/api/insight` remains for API compatibility.
- Chat is the preferred user-facing natural-language workflow.
- GDS-heavy work must be precomputed through scripts, not inside request handlers.
- Missing GDS outputs, missing model artifacts, and stale graph data should be returned explicitly instead of silently recomputing expensive jobs.

## 4. Minimal Smoke Checklist

1. Start FastAPI and open `/docs`.
2. Run the API verification tests.
3. Start the Next.js frontend.
4. Verify search -> profile -> graph -> recommendation flow.
5. Verify chat can route search/stat/recommendation-style requests.
6. Verify operations/readiness responses expose missing dependency states clearly.

## 5. Validation Commands

```powershell
.\.venv314\Scripts\python.exe -m pytest tests -q
cd frontend
npm run typecheck
npm run lint
npm run build
```
