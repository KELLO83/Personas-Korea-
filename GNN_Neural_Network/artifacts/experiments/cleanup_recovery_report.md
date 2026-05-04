# Experiment Artifact Cleanup Recovery Report

## Status

The cleanup command removed generated caches and also removed some experiment
artifacts that should have been preserved for reproducibility.

## Preserved

Performance and decision logs remain present:

- `validation_metrics.json`
- `validation_metrics.status.json`
- `test_metrics.json`
- `test_metrics.status.json`
- `ranker_params.json`
- `ranker_train.status.json`
- `ranker_feature_importance.json`
- `feature_importance.csv`
- experiment summary `.md` files
- `phase2_5_num_leaves_31/ranker_model.txt`
- root `artifacts/ranker_model.txt`

## Removed In Error

These are listed in `cleanup_removed_generated_files.json`:

- non-default `ranker_model.txt` files under experiment directories
- `ranker_dataset*.csv` files under experiment directories

These files were not Git-tracked at cleanup time, so they cannot be restored via
`git restore`.

## Correct Cleanup Policy Going Forward

Safe to delete:

- `cache/` directories
- generated embedding cache directories
- `.npy` / `.npz` cache matrices
- Python `__pycache__/`

Do not delete without explicit confirmation:

- `validation_metrics.json`
- `test_metrics.json`
- `*.status.json`
- `ranker_params.json`
- `ranker_feature_importance.json`
- `feature_importance.csv`
- summary Markdown files
- any `ranker_model.txt`
- any `ranker_dataset*.csv`

## Recovery Path

If model weights or ranker datasets are needed again, regenerate them from the
corresponding experiment script and preserved params/metrics artifacts. Do not
overwrite preserved logs during recovery unless explicitly requested.
