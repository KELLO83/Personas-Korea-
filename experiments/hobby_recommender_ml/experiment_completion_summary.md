# Experiment Completion Summary

This summary is retained as a historical closure note. The current decision source is `artifacts/experiment_decisions.json`.

## Current Interpretation

The documented default is a two-stage hobby recommender:

- Stage 1: popularity + co-occurrence.
- Stage 2: E5-small-ko-v2 domain-specific LightGBM ranker.

Phase 6 contains stronger stored test artifacts, but those results require alias provenance and governance review before default promotion.

## Current Follow-Up

- Re-run tests and key CLIs after the folder move.
- Keep stale historical paths out of active docs.
- Update the decision JSON before changing any runtime-facing default.

