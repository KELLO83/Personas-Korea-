# Person -> Person Document Inventory

## Scope

This document inventory is only for the directed `Person -> Person` similar-persona recommender track.

## Primary Documents

| Path | Role |
|---|---|
| `experiments/persona_similarity/AGENTS.md` | Folder rules, candidate/reranking policy, data/feature/evaluation rules |
| `experiments/persona_similarity/DATASET_EXPLAIN.md` | Pair-row dataset shape and feature definitions |
| `experiments/persona_similarity/PRD.md` | Track product/experiment requirements |
| `experiments/persona_similarity/TASKS.md` | Track task list |
| `experiments/persona_similarity/README.md` | Commands and pipeline |
| `experiments/persona_similarity/configs/lightgbm_reranker.yaml` | Main config |

## Artifact Documents

| Path | Role |
|---|---|
| `experiments/persona_similarity/artifacts/experiment_run_summary.md` | Current run summary and interpretation |
| `experiments/persona_similarity/artifacts/experiment_decisions.json` | Machine-readable decisions |
| `experiments/persona_similarity/artifacts/metrics/promotion_gate_status.json` | Automatic gate status |
| `experiments/persona_similarity/artifacts/metrics/e5_text_manual_review_status.json` | Manual review status for E5 text samples |
| `experiments/persona_similarity/artifacts/metrics/text_feature_status.json` | Text feature build status |
| `experiments/persona_similarity/artifacts/metrics/text_leakage_audit.json` | Text leakage audit |

## LLM Wiki Documents

| Path | Role |
|---|---|
| `docs/llm_wiki/persona_similarity/results_summary.md` | Consolidated completed results |
| `docs/llm_wiki/persona_similarity/current_findings.md` | Current state and caveats |
| `docs/llm_wiki/persona_similarity/code_inventory.md` | Code inventory for the track |
| `docs/llm_wiki/persona_similarity/experiment_plan.md` | Track experiment plan |
| `docs/llm_wiki/concepts/experiment_decision_gates.md` | Shared gates |
| `docs/llm_wiki/source_cards/recommender_methods/` | Shared method references |

## Already Completed Results

Do not ignore the completed FastRP, deterministic, structured LambdaRank, rank_xendcg, E5 text, text-builder, hybrid, and diversity rerank metrics. The current blocker is manual review and semantic trust, not lack of an automatic weak-label score.

## Next Experiment Plan

Before new model work, preserve this sequence:

```text
FastRP baseline -> deterministic baseline -> structured LambdaRank -> E5 text-only / structured+text -> diversity/text-builder ablations -> manual review and failure taxonomy -> next continuation experiment
```
