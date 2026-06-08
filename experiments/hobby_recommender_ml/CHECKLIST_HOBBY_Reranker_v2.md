# Historical Reranker v2 Checklist

This file used to contain a long phase checklist for an older reranker program. The active source of truth is now:

- `PRD.md` for experiment requirements and promotion policy.
- `TASKS.md` for current executable follow-ups.
- `artifacts/experiment_decisions.json` for machine-readable model decisions.
- `README.md` for the current documented default and run commands.

## Historical Takeaways

- Stage 1 candidate recall was not the only bottleneck; ranking collapse and feature balance mattered.
- LightGBM became the promoted Stage 2 family.
- Text features require leakage masking and audit gates.
- MMR and semantic Stage 1 providers remained experimental unless explicitly promoted.

