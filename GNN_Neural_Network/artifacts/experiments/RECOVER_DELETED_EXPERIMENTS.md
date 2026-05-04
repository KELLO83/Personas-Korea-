# Recover Deleted Experiment Artifacts

Use this only to rebuild artifacts listed in
`cleanup_removed_generated_files.json`.

## What Is Preserved

The recovery script does not overwrite preserved experiment logs:

- `validation_metrics.json`
- `test_metrics.json`
- `*.status.json`
- `ranker_params.json`
- `ranker_feature_importance.json`
- summary Markdown files

For `train_ranker.py` based experiments, it trains in `_recovery_scratch/` and
copies only `ranker_model.txt` back to the original experiment directory.

## Dry Run

```powershell
.\.venv\Scripts\python.exe GNN_Neural_Network\scripts\recover_deleted_experiment_artifacts.py
```

This writes:

```text
GNN_Neural_Network/artifacts/experiments/deleted_artifact_recovery_plan.json
```

## Rebuild Deleted Models

```powershell
.\.venv\Scripts\python.exe GNN_Neural_Network\scripts\recover_deleted_experiment_artifacts.py --execute-models
```

## Rebuild Deleted Ranker Dataset CSVs

```powershell
.\.venv\Scripts\python.exe GNN_Neural_Network\scripts\recover_deleted_experiment_artifacts.py --execute-datasets
```

## Rebuild Both

```powershell
.\.venv\Scripts\python.exe GNN_Neural_Network\scripts\recover_deleted_experiment_artifacts.py --execute-models --execute-datasets
```

## Notes

- Existing targets are skipped unless `--force` is provided.
- `_recovery_scratch/` is deleted after each recovered model unless
  `--keep-scratch` is provided.
- KURE embedding caches are intentionally not restored.
- Rebuilt standalone probe datasets are reproducibility aids, not replacements
  for preserved performance logs.
