# Hobby Recommender Dataset Notes

This document summarizes the dataset assumptions for the `Person -> Hobby` recommender.

## Inputs

Expected local files, when present:

- `data/person_hobby_edges.csv`: persona-hobby interaction edges.
- `data/person_context.csv`: persona metadata and text context.
- Split files such as `train_edges.csv`, `validation_edges.csv`, and `test_edges.csv` when an experiment creates them.
- `artifacts/hobby_profile.json`: train-only hobby profile statistics.

## Split Policy

- Training features must use train-only statistics.
- Validation and test edges are holdout positives.
- Known train hobbies for each persona must be excluded from recommendation output.
- Holdout hobby names and aliases must be masked from persona text before text-embedding features are computed.

## Leakage Policy

Text-feature experiments are disabled or rejected when post-mask leakage is above the configured threshold or when a metric jump cannot be explained without leakage risk.

## Current Candidate Policy

The documented default candidate pool uses popularity + co-occurrence from the train split. Semantic providers are evaluated as controlled experiments and are not the default Stage 1 policy unless promoted in `artifacts/experiment_decisions.json`.

