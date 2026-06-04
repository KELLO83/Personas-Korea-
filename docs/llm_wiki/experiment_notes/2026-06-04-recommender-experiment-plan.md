# Experiment Note: Alternative Recommender Experiment Plan

## Run Metadata

- Date: 2026-06-04
- Command: `.omo/plans/recommender-alternative-methods-investigation.md`
- Config: planning artifact
- Recommender boundary: `Person -> Hobby` and `Person -> Person`, separate tracks
- Dataset/artifact: hobby CSVs and similar-persona parquet artifacts
- Model/method: alternative recommendation method investigation
- Split: use existing folder-specific split policies
- Device: `.venv` Python 3.11 by default; `.venv314t` only for recorded artifact-only acceleration
- Results artifact: `.omo/evidence/recommender-methods/`

## Planned Experiment Tracks

| Track | Boundary | First Question | First Artifact |
|---|---|---|---|
| Dataset shape audit | both | What data shape do we actually have? | `.omo/evidence/recommender-methods/dataset-shape-report.md` |
| Baseline inventory | both | What is already accepted/rejected? | `.omo/evidence/recommender-methods/current-baseline-inventory.md` |
| Method research | both | Which method families fit local shape? | `.omo/evidence/recommender-methods/external-method-research.md` |
| Feasibility matrix | both | Which candidates are realistic now? | `.omo/evidence/recommender-methods/feasibility-matrix.md` |
| Benchmark design | both | What commands and gates prove a result? | `.omo/evidence/recommender-methods/benchmark-design.md` |
| Final recommendation | both | What should be tried first? | `.omo/evidence/recommender-methods/final-recommendation.md` |

## Metrics

| Boundary | Metrics |
|---|---|
| `Person -> Hobby` | Recall@K, NDCG@K, candidate recall, coverage, novelty, runtime, qualitative hobby quality |
| `Person -> Person` | NDCG@K, explanation coverage, strong-reason coverage, low-information dominance, diversity, runtime, model size, manual review |

## Interpretation

The expected first experiment recommendation is not another candidate retriever for hobby recommendation. Candidate recall is already high, so the first practical research branch should focus on ranker feature interactions and diversity-aware tradeoffs.

For similar-persona recommendation, the first practical branch is manual review of existing structured/text recommendations. Automatic weak-label NDCG is already high; trust and semantic quality are the blocker.

## Claim Boundary

This plan is not a model promotion. It defines the evidence needed before any promotion.

## Next Action

Run the tasks in `.omo/plans/recommender-alternative-methods-investigation.md`, then add a dated experiment note for the completed investigation results.
