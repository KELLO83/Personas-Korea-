# Phase 20: Chatbot UX and Orchestration Plan

This plan reflects the current Next.js + FastAPI chat direction.

## Decision

Use a single chat-first UX for natural-language exploration. Keep `/api/insight` for compatibility, but prefer `POST /api/chat` for user-facing workflows.

## Orchestration Rules

- Use rule-based intent handling first.
- Preserve session context filters unless the user explicitly resets or replaces them.
- Require a selected UUID for profile, recommendation, and person-specific influence requests.
- Record not-ready and error states in response metadata.
- Keep long-running graph/GDS work outside the request path.

## Supported Intent Families

- Search and filter exploration.
- Statistics and aggregate summaries.
- Persona profile lookup.
- Recommendation fallback lookup.
- Influence and community lookup.
- Advanced analysis compatibility through the insight route.

## P1 Scope

- Chat entrypoint in the Next.js frontend.
- Filter accumulation and reset behavior.
- Profile/recommendation/influence orchestration.
- Explicit missing-context recovery prompts.

## P2 Scope

- LLM-assisted structured filter extraction.
- Stronger ambiguity handling.
- Golden-set evaluation for intent and filter accuracy.
- More detailed hallucination guardrails.

## Metrics to Track

- Intent accuracy.
- Context-filter match rate.
- Clarification rate.
- Recovery success after clarification.
- API fallback and not-ready rates.
