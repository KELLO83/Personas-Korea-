# Person -> Person Code Inventory

## Scope

This inventory covers the similar-persona ML experiment track only.

```text
Task: Person -> Person
Project folder: experiments/persona_similarity/
Training unit: directed source_uuid -> target_uuid pair
Do not put Person -> Hobby experiments here.
```

## Core Scripts

| Path | Role | Current use |
|---|---|---|
| `experiments/persona_similarity/scripts/export_pairs.py` | Export Neo4j `SIMILAR_TO` pairs | Candidate pair generation artifact |
| `experiments/persona_similarity/scripts/build_features.py` | Build structured features and weak labels | Main pair feature artifact |
| `experiments/persona_similarity/scripts/feature_builder.py` | Feature computation functions | Structured similarity features |
| `experiments/persona_similarity/scripts/train_lambdarank.py` | Train structured LambdaRank | Current best weak-label reranker |
| `experiments/persona_similarity/scripts/evaluate_lambdarank.py` | Evaluate LambdaRank | Main metric output |
| `experiments/persona_similarity/scripts/train_rank_xendcg.py` | Train rank_xendcg | Strong-reason/low-info tradeoff comparison |
| `experiments/persona_similarity/scripts/evaluate_rank_xendcg.py` | Evaluate rank_xendcg | Comparison metrics |
| `experiments/persona_similarity/scripts/evaluate_fastrp_baseline.py` | FastRP baseline | Production baseline comparison |
| `experiments/persona_similarity/scripts/evaluate_deterministic_baseline.py` | Deterministic baseline | Strong weak-label baseline |
| `experiments/persona_similarity/scripts/build_text_embeddings.py` | Text embedding generation | E5 text feature pipeline |
| `experiments/persona_similarity/scripts/build_text_features.py` | Text cosine feature generation | Domain-specific text cosine columns |
| `experiments/persona_similarity/scripts/audit_text_feature_leakage.py` | Text leakage audit | Required text feature gate |
| `experiments/persona_similarity/scripts/build_text_manual_review.py` | Manual review sample builder | Required promotion gate |
| `experiments/persona_similarity/scripts/evaluate_diversity_rerank.py` | Diversity rerank evaluation | Optional tradeoff experiment |
| `experiments/persona_similarity/scripts/check_promotion_gate.py` | Promotion gate checks | Automatic governance check |
| `experiments/persona_similarity/scripts/relational_stage1.py` | Relational Stage1 candidate experiment | New/adjacent candidate-generation track |
| `experiments/persona_similarity/scripts/review_analysis.py` | Manual-review analysis functions | Failure taxonomy and metric comparison core |
| `experiments/persona_similarity/scripts/build_failure_taxonomy.py` | Failure taxonomy report | Label structured/text/manual-review samples by failure mode |
| `experiments/persona_similarity/scripts/analyze_text_feature_value.py` | Text feature value report | Quantify text-only and structured+text deltas against structured LambdaRank |

## Config And Utilities

| Path | Role |
|---|---|
| `experiments/persona_similarity/configs/lightgbm_reranker.yaml` | Main experiment config |
| `experiments/persona_similarity/scripts/common.py` | Shared config/cache/file utilities |
| `experiments/persona_similarity/scripts/training_utils.py` | Training support |
| `experiments/persona_similarity/scripts/evaluation_utils.py` | Evaluation support |
| `experiments/persona_similarity/scripts/text_feature_builder.py` | Text feature implementation |
| `experiments/persona_similarity/scripts/experiment_specs.py` | Experiment specs |

## Test Surface

| Path | Covers |
|---|---|
| `experiments/persona_similarity/tests/test_feature_builder.py` | Pair feature logic |
| `experiments/persona_similarity/tests/test_experiment_utils.py` | Utility behavior |
| `experiments/persona_similarity/tests/test_relational_stage1.py` | Relational Stage1 behavior |
| `experiments/persona_similarity/tests/test_review_analysis.py` | Failure taxonomy and text feature metric comparison |

## Already Completed Results

Read `results_summary.md` before proposing any new model. Do not restart from raw FastRP as if structured LambdaRank, E5 text, text-builder, and diversity rerank experiments do not exist.

## Next Experiment Plan

The next code-facing work should continue from current artifacts:

1. inspect existing manual review samples,
2. label failure modes for structured LambdaRank and text-driven examples,
3. only then run KURE/Snowflake-ko/two-tower/backbone swaps,
4. preserve source-disjoint splits and topK=50 candidate width,
5. keep FastRP/KNN rollback path unchanged.

Generated continuation reports:

- `experiments/persona_similarity/artifacts/metrics/text_feature_value_report.json`
- `experiments/persona_similarity/artifacts/metrics/failure_taxonomy_summary.json`
- `experiments/persona_similarity/artifacts/metrics/failure_taxonomy_labeled_samples.csv`
