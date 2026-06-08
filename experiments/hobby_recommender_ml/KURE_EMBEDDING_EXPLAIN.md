# KURE and Text Embedding Notes

This document records text-embedding policy for the hobby recommender.

## Current Default

The current documented default uses E5-small-ko-v2 domain-specific Stage 2 features, not KURE-v1 as the default text feature backbone.

KURE-v1 remains a historical baseline and may still be useful for controlled ablations, cache tests, and comparison runs.

## Required Controls

- Record the embedding model name and revision when available.
- Keep cache identities separate by model, text builder, split, and masking policy.
- Mask holdout hobbies and aliases before embedding persona text.
- Run post-mask leakage audits for text-feature experiments.
- Record device, batch size, thread count, and runtime metadata.

## Promotion Rule

A text-embedding variant can be promoted only after validation and test metrics are recorded and the leakage/provenance review passes.

