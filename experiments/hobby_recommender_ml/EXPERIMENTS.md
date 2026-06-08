# Hobby Recommender Experiments

This document summarizes offline experiment families under `artifacts/experiments/`.

## Selected Default

- Stage 1: popularity + co-occurrence.
- Stage 2: LightGBM with E5-small-ko-v2 single and domain-specific text similarities.
- Decision source: `artifacts/experiment_decisions.json`.

## Experiment Families

- Phase 2.5: LightGBM regularization, negative sampling, source-feature ablations.
- Phase 5: KURE/MMR and text-embedding experiments.
- Phase 5C: E5/Snowflake/KURE text-feature comparisons.
- Phase 6: domain text, alias, cross-feature, and topic-calibrated follow-ups.
- Stage1 quota: similar-person and semantic quota tests.

## Reading Order

1. `artifacts/experiment_decisions.json`
2. `README.md`
3. `PRD.md`
4. Specific run metrics under `artifacts/experiments/<phase>/<run>/`

