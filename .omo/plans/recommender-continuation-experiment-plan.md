# Recommender Continuation Experiment Plan

## TL;DR

This plan continues from existing `Person -> Hobby` and `Person -> Person` ML experiment results. It does not restart either experiment track from scratch.

## Scope Boundaries

| Track | Folder | Task | Boundary |
|---|---|---|---|
| Hobby recommender | `GNN_Neural_Network/` | `Person -> Hobby` | Do not merge with similar-persona experiments |
| Similar-persona recommender | `experiments/persona_similarity/` | `Person -> Person` | Do not merge with hobby experiments |

## Already Completed Results

### Person -> Hobby

The completed results are summarized in `docs/llm_wiki/person_hobby/results_summary.md`.

Continuation baseline layers:

```text
Phase 2.5 baseline
-> E5-small-ko-v2 domain-specific Stage2
-> Phase 6 domain-text hard-negative alias follow-up
```

Do not restart from Phase 2.5 alone. The latest artifact summary records E5-domain Stage2 as current default and Phase 6 alias/domain-text as strongest stored test artifact with validation/provenance caveats.

### Person -> Person

The completed results are summarized in `docs/llm_wiki/persona_similarity/results_summary.md`.

Continuation baseline layers:

```text
FastRP baseline
-> deterministic baseline
-> structured LambdaRank
-> E5 text-only / structured+text
-> diversity and text-builder ablations
-> manual-review gate
```

Do not restart from raw FastRP alone. Structured LambdaRank is the best weak-label reranker, but production remains FastRP/KNN because manual review is not approved.

## Code Inventory

### Person -> Hobby

Primary code inventory:

```text
docs/llm_wiki/person_hobby/code_inventory.md
```

Key scripts for continuation:

- `GNN_Neural_Network/scripts/train_ranker.py`
- `GNN_Neural_Network/scripts/evaluate_ranker.py`
- `GNN_Neural_Network/scripts/evaluate_topic_calibrated_ranker.py`
- `GNN_Neural_Network/scripts/build_phase6_experiment_manifest.py`
- `GNN_Neural_Network/scripts/leakage_check.py`
- `GNN_Neural_Network/scripts/taxonomy_overmerge.py`
- `GNN_Neural_Network/scripts/compare_ranker_runs.py`
- `GNN_Neural_Network/scripts/analyze_ranker_slices.py`
- `GNN_Neural_Network/scripts/audit_phase6_alias_features.py`
- `GNN_Neural_Network/scripts/build_ranker_feature_ablation_manifest.py`

### Person -> Person

Primary code inventory:

```text
docs/llm_wiki/persona_similarity/code_inventory.md
```

Key scripts for continuation:

- `experiments/persona_similarity/scripts/evaluate_lambdarank.py`
- `experiments/persona_similarity/scripts/build_text_manual_review.py`
- `experiments/persona_similarity/scripts/check_promotion_gate.py`
- `experiments/persona_similarity/scripts/evaluate_diversity_rerank.py`
- `experiments/persona_similarity/scripts/audit_text_feature_leakage.py`
- `experiments/persona_similarity/scripts/relational_stage1.py`
- `experiments/persona_similarity/scripts/build_failure_taxonomy.py`
- `experiments/persona_similarity/scripts/analyze_text_feature_value.py`

## Document Inventory

### Person -> Hobby

Primary document inventory:

```text
docs/llm_wiki/person_hobby/document_inventory.md
```

Read order:

```text
GNN_Neural_Network/AGENTS.md
GNN_Neural_Network/artifacts/experiment_run_summary.md
docs/llm_wiki/person_hobby/results_summary.md
docs/llm_wiki/person_hobby/code_inventory.md
docs/llm_wiki/person_hobby/experiment_plan.md
```

### Person -> Person

Primary document inventory:

```text
docs/llm_wiki/persona_similarity/document_inventory.md
```

Read order:

```text
experiments/persona_similarity/AGENTS.md
experiments/persona_similarity/artifacts/experiment_run_summary.md
docs/llm_wiki/persona_similarity/results_summary.md
docs/llm_wiki/persona_similarity/code_inventory.md
docs/llm_wiki/persona_similarity/experiment_plan.md
```

## Next Experiment Plan

### Person -> Hobby

Next work continues from E5-domain and Phase 6 artifacts.

Priority order:

1. Lock the comparison baseline in a short decision note: E5-domain default vs Phase 6 optional post-ranker.
2. Run slice/error analysis for cold-start, long-tail hobbies, age group, occupation, region, and taxonomy category.
3. Run ranker-side feature interaction alternatives only after baseline lock.
4. Run alias/text provenance and leakage audits before any `name_plus_aliases` promotion.
5. Run diversity Pareto analysis only with strict Recall/NDCG tolerance.

Continuation analysis artifacts already generated:

- `GNN_Neural_Network/artifacts/experiments/continuation_analysis/phase6_validation_compare.json`
- `GNN_Neural_Network/artifacts/experiments/continuation_analysis/phase6_alias_slice_gaps.json`
- `GNN_Neural_Network/artifacts/experiments/continuation_analysis/phase6_alias_audit.json`
- `GNN_Neural_Network/artifacts/experiments/continuation_analysis/ranker_feature_ablation_manifest.json`

Do not start with new semantic Stage1 retrieval. KURE semantic Stage1 already reduced candidate recall in the recorded validation result.

### Person -> Person

Next work continues from structured LambdaRank, E5 text, and manual-review artifacts.

Priority order:

1. Build or inspect manual review samples for structured LambdaRank, E5 text-only, structured+text, and diversity rerank outputs.
2. Add failure taxonomy labels: `too_same`, `occupation_only`, `location_only`, `hobby_overlap_only`, `low_information`, `text_semantic_good`, `text_semantic_bad`.
3. Summarize manual review outcomes before any backbone swap.
4. Only after review, compare KURE/Snowflake-ko/two-tower retrieval if the review identifies a semantic failure mode.
5. Keep production rollback as FastRP/KNN `fastrp_score`.

Continuation analysis artifacts already generated:

- `experiments/persona_similarity/artifacts/metrics/text_feature_value_report.json`
- `experiments/persona_similarity/artifacts/metrics/failure_taxonomy_summary.json`
- `experiments/persona_similarity/artifacts/metrics/failure_taxonomy_labeled_samples.csv`

Do not promote from weak-label NDCG alone.

## Verification Commands

### Documentation Coverage

```powershell
rg -n "Code Inventory|Document Inventory|Next Experiment Plan|Already Completed Results" docs/llm_wiki/person_hobby docs/llm_wiki/persona_similarity .omo/plans/recommender-continuation-experiment-plan.md
```

### Boundary Check

```powershell
rg -n "Person -> Hobby|Person -> Person|Do not put|Do not merge|continue|not restart|not start from scratch" docs/llm_wiki/person_hobby docs/llm_wiki/persona_similarity .omo/plans/recommender-continuation-experiment-plan.md
```

### Regression Check

```powershell
Test-Path docs/llm_wiki/person_hobby/results_summary.md
Test-Path docs/llm_wiki/persona_similarity/results_summary.md
Test-Path .omo/plans/recommender-alternative-methods-investigation.md
rg -n "results_summary|recommender-continuation-experiment-plan|E5-domain|structured LambdaRank" docs/llm_wiki .omo/plans
```

## Stop Conditions

- Stop if a proposed experiment ignores existing result summaries.
- Stop if a plan mixes hobby and similar-persona metrics.
- Stop if a production promotion is proposed without the track-specific gate.
- Stop if text or alias features are promoted without provenance/leakage/manual-review evidence.
