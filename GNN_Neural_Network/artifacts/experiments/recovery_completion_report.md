# Deleted Experiment Artifact Recovery Completion

Date: 2026-05-05T00:21:44+09:00

## Command

```powershell
.\.venv\Scripts\python.exe GNN_Neural_Network\scripts\recover_deleted_experiment_artifacts.py --execute-models --execute-datasets
```

## Result

- Recovered model artifacts: 13 / 13
- Recovered dataset artifacts: 4 / 4
- Missing artifacts after recovery: 0
- Input training data source: `GNN_Neural_Network\data\person_hobby_edges.csv` and `GNN_Neural_Network\data\person_context.csv` through `GNN_Neural_Network\configs\lightgcn_hobby.yaml` / `gnn_recommender.config.PathConfig`.
- Existing metrics, status files, params, feature importance, and summary logs were preserved. The recovery command copied regenerated model files and regenerated deleted CSV datasets only.
- These recovered files are regenerated artifacts, not byte-identical originals. Existing performance logs remain the source of truth for historical metric values.

## Recovered Files

| Relative path | Exists | Size bytes |
|---|---:|---:|
| feature_balance_probe\ranker_dataset_with_kure.csv | True | 3980626 |
| phase2_5_neg_ratio_1\ranker_model.txt | True | 3511 |
| phase2_5_neg_ratio_2\ranker_model.txt | True | 48647 |
| phase2_5_neg_ratio_4_hard_0_5\ranker_model.txt | True | 36014 |
| phase2_5_neg_ratio_4_hard_1_0\ranker_model.txt | True | 8183 |
| phase2_5_neg_ratio_8\ranker_model.txt | True | 55416 |
| phase2_5_source_onehot\ranker_model.txt | True | 34804 |
| phase5_b1_listwise\smoke_lambdarank_f07\ranker_model.txt | True | 9257 |
| phase5_b1_listwise\smoke_lambdarank_f08\ranker_model.txt | True | 11421 |
| phase5_b2_feature_balance\feature_fraction_0_7\ranker_model.txt | True | 32839 |
| phase5_b2_feature_balance\ranker_dataset_with_kure.csv | True | 3980626 |
| phase5_pre_50k_baseline\feature_fraction_0.8\ranker_model.txt | True | 3212 |
| phase5_pre_50k_baseline\feature_fraction_0.85\ranker_model.txt | True | 3214 |
| phase5_pre_50k_baseline\ranker_dataset_phase25.csv | True | 38880542 |
| phase5_pre_50k_baseline\ranker_model.txt | True | 3214 |
| probe_lgbm_balance\ranker_dataset_with_kure.csv | True | 3980626 |
| probe_lgbm_balance\ranker_model.txt | True | 3214 |

## Follow-up Policy

- `GNN_Neural_Network/artifacts/experiments/**` is now unignored for git tracking except generated caches and binary embedding arrays.
- Future cleanup must not remove metrics, params, summaries, datasets, or model weights unless the user explicitly confirms that exact artifact class.
