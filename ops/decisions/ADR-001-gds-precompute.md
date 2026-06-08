# ADR-001: Precompute GDS Centrality in Batch Jobs

**Status**: Accepted
**Date**: 2026-04-28
**Decision Makers**: Oracle (Architecture Review) + development team
**Related PRD**: Feature 10 (network influence analysis)

---

## Context

Feature 10 must identify key people in the network using Neo4j GDS centrality
algorithms: `pageRank`, `betweenness`, and `degree`.

The initial PRD proposed calling `gds.*.stream` in real time from the API
endpoint (`/api/influence/top`).

## Decision

Centrality calculations must be precomputed by batch jobs and written back to
Neo4j node properties. FastAPI request handlers must only read the latest
published values.

## Consequences

- API latency stays predictable.
- Neo4j GDS projection and long-running centrality work stay outside the request
  path.
- Batch jobs need status metadata so the API can report stale or missing
  centrality data.
- Failed batch runs must not erase the last successful centrality values.

## Operational Rule

Use `ops/docs/centrality-batch-operations.md` for runbook details.
