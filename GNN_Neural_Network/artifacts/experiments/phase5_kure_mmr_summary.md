# Phase 5-A KURE MMR Re-Evaluation Summary

- Date: 2026-05-01
- Objective: re-test MMR diversity reordering with dense KURE-v1 hobby embeddings on fixed Phase 2.5 default baseline.

## Baseline (closed Phase 2.5 default)

- Model path: `artifacts/experiments/phase2_5_num_leaves_31/ranker_model.txt`
- Validation: `recall@10=0.7390509094604207`, `ndcg@10=0.45797028878684237`, `coverage@10=0.15555555555555556`, `novelty@10=4.584286633989583`

## Validation sweeps

| lambda | command signature | status file | recall@10 | ndcg@10 | coverage@10 | novelty@10 | candidate_recall@50 | gate result |
|:---:|:---|:---|---:|---:|---:|---:|---:|:---|
| 0.5 | `evaluate_ranker.py --split validation --use-mmr --mmr-embedding-method kure --phase5-kure-mmr --mmr-lambda 0.5` | `artifacts/experiments/phase5_kure_mmr_lambda_0.5/validation_metrics.status.json` | 0.7025708769434 | 0.44405002047689657 | 0.17222222222222222 | 4.600134502809017 | 0.9776445483182603 | `blocked` |
| 0.7 | `evaluate_ranker.py --split validation --use-mmr --mmr-embedding-method kure --phase5-kure-mmr --mmr-lambda 0.7` | `artifacts/experiments/phase5_kure_mmr_lambda_0.7/validation_metrics.status.json` | 0.7230972462148155 | 0.45215806458894897 | 0.14444444444444443 | 4.574651938789798 | 0.9776445483182603 | `blocked` |
| 0.8 | `evaluate_ranker.py --split validation --use-mmr --mmr-embedding-method kure --phase5-kure-mmr --mmr-lambda 0.8` | `artifacts/experiments/phase5_kure_mmr_lambda_0.8/validation_metrics.status.json` | 0.7293974189614877 | 0.4544781022168803 | 0.14444444444444443 | 4.572329701397024 | 0.9776445483182603 | `blocked` |
| 0.9 | `evaluate_ranker.py --split validation --use-mmr --mmr-embedding-method kure --phase5-kure-mmr --mmr-lambda 0.9` | `artifacts/experiments/phase5_kure_mmr_lambda_0.9/validation_metrics.status.json` | 0.7288893405141754 | 0.45464407336870966 | 0.15000000000000000 | 4.574799041368609 | 0.9776445483182603 | `blocked` |

## Decision

- **Final status**: `NO-GO`
- Reason: all candidates failed accuracy gate (`delta_recall@10 >= -0.002` and `delta_ndcg@10 >= -0.002`); no winner selected.
- `v2_fallback_count = 0` for all sweeps; `candidate_recall@50` unchanged.
- Test run: skipped for all lambdas.

## Artifact outputs

- `artifacts/experiments/phase5_kure_mmr_lambda_0.5/validation_metrics.json`
- `artifacts/experiments/phase5_kure_mmr_lambda_0.5/validation_metrics.status.json`
- `artifacts/experiments/phase5_kure_mmr_lambda_0.7/validation_metrics.json`
- `artifacts/experiments/phase5_kure_mmr_lambda_0.7/validation_metrics.status.json`
- `artifacts/experiments/phase5_kure_mmr_lambda_0.8/validation_metrics.json`
- `artifacts/experiments/phase5_kure_mmr_lambda_0.8/validation_metrics.status.json`
- `artifacts/experiments/phase5_kure_mmr_lambda_0.9/validation_metrics.json`
- `artifacts/experiments/phase5_kure_mmr_lambda_0.9/validation_metrics.status.json`
