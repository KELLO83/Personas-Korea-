# LLM Wiki Index

## Scope

This wiki tracks source-grounded knowledge for the Korean persona recommender project. It separates local experiment evidence from external recommendation-system methods.

The current research direction is dataset-shape-first recommender benchmarking:

```text
keep Person -> Hobby and Person -> Person separate
inspect local artifact shape before proposing models
compare alternatives only against the correct baseline and metric gate
record promotion blockers before changing production behavior
```

Chronological wiki changes are recorded in `LOG.md`.

## Required Local Context

- Root project requirements: `PRD.md`, `TASKS.md`
- Root data/model architecture: `docs/model_architecture.md`
- GDS precompute decision: `docs/decisions/ADR-001-gds-precompute.md`
- Recommendation reasoning decision: `docs/decisions/ADR-002-recommendation-reasoning.md`
- Hobby recommender scope: `GNN_Neural_Network/AGENTS.md`
- Hobby recommender dataset: `GNN_Neural_Network/DATASET_EXPLAIN.md`
- Hobby recommender decisions: `GNN_Neural_Network/EXPERIMENTS.md`
- Hobby recommender current docs: `GNN_Neural_Network/PRD.md`, `GNN_Neural_Network/TASKS.md`, `GNN_Neural_Network/README.md`
- Similar-persona recommender scope: `experiments/persona_similarity/AGENTS.md`
- Similar-persona dataset: `experiments/persona_similarity/DATASET_EXPLAIN.md`
- Similar-persona decisions: `experiments/persona_similarity/artifacts/experiment_run_summary.md`
- Similar-persona current docs: `experiments/persona_similarity/PRD.md`, `experiments/persona_similarity/TASKS.md`, `experiments/persona_similarity/README.md`
- Active investigation plan: `.omo/plans/recommender-alternative-methods-investigation.md`
- Continuation experiment plan: `.omo/plans/recommender-continuation-experiment-plan.md`

## Source Cards

Source cards are grouped by research role.

- `source_cards/recommender_methods/`: recommendation model families and benchmark tooling.

| Source | Type | Compatibility | Status | Card |
|---|---|---:|---|---|
| RecBole | Benchmark framework | Tooling/reference | Track | `source_cards/recommender_methods/recbole.md` |
| LightGCN / XSimGCL | Graph collaborative filtering | Hobby candidate-generation comparison | Track | `source_cards/recommender_methods/graph_cf_lightgcn_xsimgcl.md` |
| KGAT | Knowledge-graph recommendation | Offline comparison only | Track | `source_cards/recommender_methods/kgat.md` |
| NCF / Wide&Deep / two-tower | Neural ranking and retrieval | Benchmark if artifact shape fits | Track | `source_cards/recommender_methods/neural_retrieval_and_ranking.md` |
| Factorization Machines | Tabular feature interactions | High-priority ranker alternative | Track | `source_cards/recommender_methods/factorization_machines.md` |
| Text embedding retrieval | Content/text similarity | Manual-review gated | Track | `source_cards/recommender_methods/text_embedding_retrieval.md` |
| Diversity reranking | Post-rank tradeoff method | Accuracy-gated only | Track | `source_cards/recommender_methods/diversity_reranking.md` |

## Experiment Track Folders

The two ML recommender experiments are separate. Keep their experiment notes, baseline decisions, metrics, and promotion gates in separate folders.

| Track | Task | Folder | Upstream project folder |
|---|---|---|---|
| Hobby recommendation | `Person -> Hobby` | `person_hobby/` | `GNN_Neural_Network/` |
| Similar-persona recommendation | `Person -> Person` | `persona_similarity/` | `experiments/persona_similarity/` |

Track inventories:

- Hobby code inventory: `person_hobby/code_inventory.md`
- Hobby document inventory: `person_hobby/document_inventory.md`
- Similar-persona code inventory: `persona_similarity/code_inventory.md`
- Similar-persona document inventory: `persona_similarity/document_inventory.md`

## Concept Notes

| Concept | Note |
|---|---|
| Current recommender findings | `concepts/current_recommender_findings.md` |
| Dataset shape and boundaries | `concepts/dataset_shape_and_boundaries.md` |
| Experiment decision gates | `concepts/experiment_decision_gates.md` |
| Recommender method shortlist | `concepts/recommender_method_shortlist.md` |

## Experiment Notes

| Date | Run | Note |
|---|---|---|
| 2026-06-04 | Initial wiki setup | `experiment_notes/2026-06-04-initial-wiki-setup.md` |
| 2026-06-04 | Alternative recommender experiment plan | `experiment_notes/2026-06-04-recommender-experiment-plan.md` |
| 2026-06-04 | Existing result consolidation | `experiment_notes/2026-06-04-existing-results-consolidation.md` |
| 2026-06-04 | Hobby recommender track setup | `person_hobby/experiment_plan.md` |
| 2026-06-04 | Hobby existing results | `person_hobby/results_summary.md` |
| 2026-06-04 | Hobby code/document inventory | `person_hobby/code_inventory.md`, `person_hobby/document_inventory.md` |
| 2026-06-04 | Similar-persona track setup | `persona_similarity/experiment_plan.md` |
| 2026-06-04 | Similar-persona existing results | `persona_similarity/results_summary.md` |
| 2026-06-04 | Similar-persona code/document inventory | `persona_similarity/code_inventory.md`, `persona_similarity/document_inventory.md` |
| 2026-06-04 | Continuation experiment plan | `.omo/plans/recommender-continuation-experiment-plan.md` |

## Reproducibility Snapshots

Store lightweight copied command outputs under:

```text
docs/llm_wiki/experiment_notes/artifacts/
```

Do not store raw datasets, model checkpoints, credentials, or full copied papers in this wiki.
