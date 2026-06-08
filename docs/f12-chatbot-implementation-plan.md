# F12 Chatbot Implementation Plan

This document reflects the current FastAPI + LangGraph chatbot direction.

## Goal

Provide a single conversational exploration surface through `POST /api/chat` while keeping `/api/insight` available for compatibility.

## Current Architecture

- Route: `src/api/routes/chat.py`.
- Chat graph: `src/rag/chat_graph.py`.
- RAG routing: `src/rag/router.py`, `src/rag/cypher_chain.py`, `src/rag/vector_chain.py`.
- LLM client: `src/rag/llm.py`.
- Trace support: `src/rag/tracing.py` and `src/api/routes/rag_traces.py`.

## Required Behavior

- Maintain session-level context filters.
- Support search/stat/profile/recommendation/influence-style intents.
- Keep expensive GDS work outside the request path.
- Return clear not-ready or stale states when graph artifacts are missing.
- Keep request and response contracts aligned with `src/api/schemas.py`.

## Example English Test Utterances

- "Show people who live in Seoul."
- "Filter that result to people in their twenties."
- "Narrow it to men."
- "Show the hobby distribution for this group."
- "Reset the filters."
- "What hobbies would you recommend for this person?"
- "Show influential people in this community."
- "Explain this UUID."

## Frontend Expectations

- Chat UI belongs in the maintained Next.js frontend.
- The UI should expose loading, empty, error, and stale states.
- Filter reset should clear the active session filter state.
- Recommendation and influence intents should require a selected UUID or return a recovery prompt.

## Validation

```powershell
.\.venv314\Scripts\python.exe -m pytest tests\test_chat_graph.py tests\test_api_chat.py -q
```

Manual smoke:

```powershell
curl -X POST http://localhost:8000/api/chat `
  -H "Content-Type: application/json" `
  -d '{"session_id":"demo","message":"Show men in their twenties who live in Seoul","stream":false}'
```
