# Phase 22: Advanced Analysis Implementation Scope

This document records the product-level scope for advanced analysis features in the current FastAPI + Next.js platform.

## Scope

- Target persona generation should use deterministic summaries, evidence UUIDs, guardrails, optional LLM synthesis, and KURE semantic filtering.
- Lifestyle map analysis should expose read-only aggregate relationships and evidence-backed summaries.
- Career transition analysis should expose read-only graph/statistical signals with clear source metadata.
- The frontend surface is the advanced analysis area in the maintained Next.js UI.

## Boundaries

- These are product features, not recommender-training experiments.
- Expensive graph, embedding, and LLM work should be precomputed or bounded.
- Responses should expose missing-data and not-ready states rather than silently substituting unrelated results.
