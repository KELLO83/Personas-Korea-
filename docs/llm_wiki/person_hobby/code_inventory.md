# Person -> Hobby Code Inventory

## Scope

This inventory covers the hobby recommender ML experiment track only.

```text
Task: Person -> Hobby
Project folder: GNN_Neural_Network/
Do not put Person -> Person similar-persona experiments here.
```

## Core Package

| Path | Role | Current use |
|---|---|---|
| `GNN_Neural_Network/gnn_recommender/data.py` | Dataset and split structures | Person-hobby edge loading, train/validation/test split support |
| `GNN_Neural_Network/gnn_recommender/baseline.py` | Stage 1 candidate and baseline logic | Popularity/cooccurrence and related candidate provider logic |
| `GNN_Neural_Network/gnn_recommender/ranker.py` | LightGBM ranker and ranking dataset | Current Stage 2 ranker family and feature rows |
| `GNN_Neural_Network/gnn_recommender/rerank.py` | Deterministic/diversity reranking | Fallback and diversity tradeoff experiments |
| `GNN_Neural_Network/gnn_recommender/text_embedding.py` | Text embedding feature governance | KURE/E5/Snowflake text features and leakage controls |
| `GNN_Neural_Network/gnn_recommender/phase6.py` | Phase 6 feature/ranking helpers | Domain-text hard-negative alias follow-up |
| `GNN_Neural_Network/gnn_recommender/diversity.py` | Diversity metrics/helpers | ILD/diversity-aware analysis |
| `GNN_Neural_Network/gnn_recommender/metrics.py` | Ranking metrics | Recall/NDCG/Coverage/Novelty evaluation |

## Experiment Scripts

| Path | Role | Use next |
|---|---|---|
| `GNN_Neural_Network/scripts/train_ranker.py` | Train Stage 2 ranker | Baseline reproduction and candidate ranker variants |
| `GNN_Neural_Network/scripts/evaluate_ranker.py` | Evaluate Stage 2 ranker | Main validation/test gate |
| `GNN_Neural_Network/scripts/evaluate_topic_calibrated_ranker.py` | Topic calibration post-ranker | Optional Phase 6 post-ranker verification |
| `GNN_Neural_Network/scripts/build_phase6_experiment_manifest.py` | Phase 6 manifest | Preserve Phase 6 continuation metadata |
| `GNN_Neural_Network/scripts/evaluate_reranker.py` | Reranker evaluation | Deterministic/diversity comparison |
| `GNN_Neural_Network/scripts/sweep_mmr_lambda.py` | MMR sweep | Historical no-go reference, not first next run |
| `GNN_Neural_Network/scripts/evaluate_lightgcn.py` | LightGCN evaluation | Candidate-generation comparison only |
| `GNN_Neural_Network/scripts/train_lightgcn.py` | LightGCN training | Defer unless candidate recall becomes bottleneck |
| `GNN_Neural_Network/scripts/leakage_check.py` | Leakage audit | Required for text/alias feature promotion |
| `GNN_Neural_Network/scripts/taxonomy_overmerge.py` | Taxonomy quality audit | Required for alias/category text governance |
| `GNN_Neural_Network/scripts/compare_ranker_runs.py` | Existing run comparison | Compare E5/Phase 6 artifacts without retraining |
| `GNN_Neural_Network/scripts/analyze_ranker_slices.py` | Slice/error analysis | Identify segment-level recall gaps from existing metrics |
| `GNN_Neural_Network/scripts/audit_phase6_alias_features.py` | Phase 6 alias audit | Keep alias/domain-text improvements on hold until provenance review passes |
| `GNN_Neural_Network/scripts/build_ranker_feature_ablation_manifest.py` | Feature ablation manifest | Define one-feature-group-at-a-time ranker ablations |

## Test Surface

| Path | Covers |
|---|---|
| `GNN_Neural_Network/tests/test_ranker.py` | Ranker dataset, LightGBM features, ranking behavior |
| `GNN_Neural_Network/tests/test_phase6.py` | Phase 6 feature/manifest behavior |
| `GNN_Neural_Network/tests/test_text_embedding_governance.py` | Text embedding governance |
| `GNN_Neural_Network/tests/test_text_embedding.py` | Text embedding utilities |
| `GNN_Neural_Network/tests/test_experimental_guardrails.py` | Promotion/experiment guardrails |
| `GNN_Neural_Network/tests/test_eval_resource_policy.py` | Thread/resource policy |
| `GNN_Neural_Network/tests/test_diversity.py` | Diversity calculations |
| `GNN_Neural_Network/tests/test_experiment_analysis.py` | Continuation analysis, slice gaps, alias audit, ablation manifest |

## Already Completed Results

Read `results_summary.md` before proposing any new run. Do not restart from Phase 2.5 as if later E5-domain and Phase 6 results do not exist.

## Next Experiment Plan

The next code-facing work should continue from the latest stored artifacts:

1. lock the comparison baseline from `artifacts/experiment_run_summary.md`,
2. run slice/error analysis over the current E5-domain and Phase 6 candidates,
3. test ranker-side feature interaction alternatives before new retrievers,
4. run alias/text provenance and leakage audits before any text/alias promotion,
5. only revisit LightGCN/XSimGCL/KURE Stage1 if candidate recall becomes the bottleneck again.

Generated continuation reports:

- `GNN_Neural_Network/artifacts/experiments/continuation_analysis/phase6_validation_compare.json`
- `GNN_Neural_Network/artifacts/experiments/continuation_analysis/phase6_alias_slice_gaps.json`
- `GNN_Neural_Network/artifacts/experiments/continuation_analysis/phase6_alias_audit.json`
- `GNN_Neural_Network/artifacts/experiments/continuation_analysis/ranker_feature_ablation_manifest.json`
