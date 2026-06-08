# ADR-003: Chatbot Memory

## Status

Accepted for the current chat architecture.

## Decision

Keep short session memory for the common conversational pattern where a user narrows filters over a few turns, such as region -> age group -> gender -> occupation.

## Consequences

- The chat graph must maintain session-scoped context filters.
- Reset and replacement instructions must be handled explicitly.
- Long histories should be summarized or truncated to keep latency bounded.
