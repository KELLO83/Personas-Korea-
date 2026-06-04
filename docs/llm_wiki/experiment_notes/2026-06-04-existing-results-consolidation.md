# Experiment Note: Existing Results Consolidation

## Run Metadata

- Date: 2026-06-04
- Command: documentation consolidation only
- Config: none
- Recommender boundary: both, kept separate
- Dataset/artifact:
  - `GNN_Neural_Network/artifacts/experiment_run_summary.md`
  - `GNN_Neural_Network/EXPERIMENTS.md`
  - `experiments/persona_similarity/artifacts/experiment_run_summary.md`
  - `experiments/persona_similarity/artifacts/experiment_decisions.json`
- Model/method: existing-result wiki synthesis
- Split: existing artifact splits
- Device: not applicable
- Results artifact:
  - `docs/llm_wiki/person_hobby/results_summary.md`
  - `docs/llm_wiki/persona_similarity/results_summary.md`

## Metrics

| Track | Current key result |
|---|---|
| `Person -> Hobby` | Latest artifact summary records E5-small-ko-v2 domain-specific Stage2 as current default; Phase 6 alias/domain-text run is strongest stored test artifact but has validation/provenance caveats. |
| `Person -> Person` | Structured LambdaRank is best weak-label reranker with NDCG@5 `0.993136` and NDCG@10 `0.993145`; production remains FastRP/KNN because manual review is not approved. |

## Interpretation

The two recommender tracks have different blockers.

For hobby recommendation, the result history moved beyond the old Phase 2.5 no-text baseline into E5-domain Stage2 and Phase 6 alias/domain-text experiments. The next work should respect the latest artifact summary while keeping governance caveats around candidate aliases and text provenance.

For similar-persona recommendation, the automatic metrics are already very high for structured LambdaRank, so the next blocker is manual review and semantic trust rather than another automatic weak-label metric run.

## Claim Boundary

This note does not create a new result. It only consolidates existing artifacts into the LLM Wiki.

## Next Action

Use `person_hobby/results_summary.md` and `persona_similarity/results_summary.md` as the first wiki pages to read before proposing new recommendation experiments.

Current continuation state:

- `Person -> Hobby`: continue from E5-domain Stage2 and Phase 6 artifacts; do not start from the older Phase 2.5 summary alone.
- `Person -> Person`: continue from structured LambdaRank, E5 text, diversity, and manual-review artifacts; do not start from raw FastRP alone.
- Code and document routing are recorded in `person_hobby/code_inventory.md`, `person_hobby/document_inventory.md`, `persona_similarity/code_inventory.md`, and `persona_similarity/document_inventory.md`.
- The executable follow-up plan is `.omo/plans/recommender-continuation-experiment-plan.md`.
