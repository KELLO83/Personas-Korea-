# Person -> Hobby Document Inventory

## Scope

This document inventory is only for the `Person -> Hobby` hobby recommender track.

## Primary Documents

| Path | Role |
|---|---|
| `GNN_Neural_Network/AGENTS.md` | Folder rules, model/evaluation policy, artifact requirements |
| `GNN_Neural_Network/DATASET_EXPLAIN.md` | Dataset shape, text leakage caveats, graph mapping |
| `GNN_Neural_Network/EXPERIMENTS.md` | Phase 2.5-era experiment decision summary |
| `GNN_Neural_Network/PRD.md` | Hobby recommender product/experiment requirements |
| `GNN_Neural_Network/TASKS.md` | Track tasks |
| `GNN_Neural_Network/README.md` | Track run commands and summary |
| `GNN_Neural_Network/CHECKLIST_GNN_Reranker_v2.md` | Reranker checklist |
| `GNN_Neural_Network/KURE_EMBEDDING_EXPLAIN.md` | KURE embedding context |

## Artifact Documents

| Path | Role |
|---|---|
| `GNN_Neural_Network/artifacts/experiment_run_summary.md` | Latest E5-domain and Phase 6 result summary |
| `GNN_Neural_Network/artifacts/experiment_decisions.json` | Machine-readable decision state |
| `GNN_Neural_Network/artifacts/experiments/phase6_hobby_validation_summary.md` | Phase 6 validation summary |
| `GNN_Neural_Network/artifacts/experiments/phase2_5_default_decision_closure.md` | Phase 2.5 closure |
| `GNN_Neural_Network/artifacts/experiments/phase2_5_negative_sampling_summary.md` | Negative sampling ablation |
| `GNN_Neural_Network/artifacts/experiments/phase2_5_source_onehot_summary.md` | Source one-hot ablation |
| `GNN_Neural_Network/artifacts/experiments/phase5_kure_mmr_summary.md` | KURE dense MMR no-go |
| `GNN_Neural_Network/artifacts/experiments/pre_kure_experiment_summary.md` | Pre-KURE readiness |

## LLM Wiki Documents

| Path | Role |
|---|---|
| `docs/llm_wiki/person_hobby/results_summary.md` | Consolidated completed results |
| `docs/llm_wiki/person_hobby/current_findings.md` | Current state and caveats |
| `docs/llm_wiki/person_hobby/code_inventory.md` | Code inventory for the track |
| `docs/llm_wiki/person_hobby/experiment_plan.md` | Track experiment plan |
| `docs/llm_wiki/concepts/experiment_decision_gates.md` | Shared gates |
| `docs/llm_wiki/source_cards/recommender_methods/` | Shared method references |

## Already Completed Results

Do not ignore `artifacts/experiment_run_summary.md`. It supersedes the older Phase 2.5-only view for latest result state, while `EXPERIMENTS.md` remains useful as a baseline history.

## Next Experiment Plan

Before any new training run, update the relevant artifact summary and this wiki if the baseline changes. New documents should preserve the sequence:

```text
Phase 2.5 -> E5-domain Stage2 -> Phase 6 alias/domain-text follow-up -> next continuation experiment
```
