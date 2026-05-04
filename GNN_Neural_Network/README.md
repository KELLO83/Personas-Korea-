# GNN Neural Network PoC

Offline LightGCN hobby/leisure recommendation PoC for `Nemotron-Personas-Korea`.

## Documentation hierarchy

- This README is a **single source of truth for execution commands and current recommendation status**.
- For requirements and implementation decisions, follow:
  - `PRD.md` (scope, architecture, goals)
  - `TASKS.md` (execution gate and completion status)
- v2 experimental docs are supplementary:
  - `PRD_GNN_Reranker_v2.md`
  - `CHECKLIST_GNN_Reranker_v2.md`
- Dataset and schema description is contextual reference:
  - `DATASET_EXPLAIN.md`
- Experiment navigation and script ownership:
  - `EXPERIMENTS.md`
  - `scripts/README.md`
- Conflict rule: if a conflict appears, prioritize requirements/gates above and then align this README.

## Scope

- Runs as Python scripts only.
- Uses the existing project `.venv` interpreter.
- Initial Neo4j access is limited to edge export.
- Training/evaluation/recommendation after export use local files only.
- No FastAPI, Streamlit, or Next.js integration in the initial PoC.

## Dependency policy

Allowed:

- PyTorch core APIs (`torch`, `torch.nn.Module`, `torch.optim.Adam`, `torch.sparse.mm`)
- Small helper packages in `requirements-gnn.txt`

Not allowed for the initial PoC:

- PyTorch Geometric
- DGL
- TorchRec
- RecBole
- External LightGCN implementations

## CUDA check

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Install helper dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r GNN_Neural_Network\requirements-gnn.txt
```

## Export Person-Hobby edges

```powershell
.\.venv\Scripts\python.exe GNN_Neural_Network\scripts\export_person_hobby_edges.py --output GNN_Neural_Network\data\person_hobby_edges.csv
```

By default this also writes raw persona context for later offline reranking:

```text
GNN_Neural_Network/data/person_context.csv
```

Use `--skip-context` only when you intentionally want edge export without context export.

## Vocabulary quality gate

Training preparation canonicalizes hobby names before indexing them as LightGCN items.

- `data.normalize_hobbies`: trims, Unicode-normalizes, lowercases, and collapses whitespace.
- `data.alias_map_path`: optional JSON map from raw/variant hobby names to canonical hobby names.
- `data.min_item_degree`: drops canonical hobbies below the configured support threshold.
- `paths.vocabulary_report`: stores raw/canonical/retained edge, person, hobby, singleton, and dropped counts.

Run `--prepare-only` and inspect `vocabulary_report.json` before any actual training.

## Prepare splits without training

```powershell
.\.venv\Scripts\python.exe GNN_Neural_Network\scripts\train_lightgcn.py --prepare-only
```

Preparation writes train/validation/test splits plus leakage-safe offline artifacts:

- `hobby_profile.json`: train-split-only popularity, distributions, and co-occurrence profile
- `leakage_audit.json`: validation/test held-out hobby mentions in persona text fields
- `score_normalization.json`: initial provider score normalization contract
- `fallback_usage.json`: placeholder updated by recommendation fallback execution
- `experiment_decisions.json`: current accepted/rejected provider decisions and final recommendation status snapshot
- `experiment_run_summary.md`: human-readable engineering summary of the selected baseline, promoted reranker, model architecture, data input pipeline, feature schema, current metrics, and failed ablations

## Train

Only run when training is explicitly intended.

```powershell
.\.venv\Scripts\python.exe GNN_Neural_Network\scripts\train_lightgcn.py
```

## Evaluate

```powershell
.\.venv\Scripts\python.exe GNN_Neural_Network\scripts\evaluate_lightgcn.py --split test
```

## Recommend

```powershell
.\.venv\Scripts\python.exe GNN_Neural_Network\scripts\recommend_for_persona.py --uuid a5ad493e75e74e5cb4a81ac934a1db8f --top-k 10
```

For the legacy v1 comparison path, add `--rerank` to apply the deterministic Stage 2 reranker to the Stage 1 candidate pool before printing final recommendations.

For the current promoted ranking path, run the same CLI with `--use-learned-ranker` so the Stage 1 candidate pool is scored by the LightGBM ranker.

If the UUID is unknown to the trained canonical vocabulary, the CLI falls back to global popularity recommendations from the gated train split. If LightGCN returns fewer than `top-k` candidates after known-hobby masking, popularity fills the remaining slots.

The recommendation CLI now records Stage 1 provider observability:

- `candidates_sample.json`: Stage 1 provider candidate samples and selected candidates
- `provider_contribution.json`: raw/normalized candidates per provider and selected source counts
- `fallback_usage.json`: unknown UUID and underfilled-candidate fallback usage

Selected Stage 1 baseline is currently `popularity + cooccurrence`.

- Primary/default providers: train-split global popularity, train-split co-occurrence
- Auxiliary/experimental provider: LightGCN
- Disabled by default after ablation: `segment_popularity`, `bm25_itemknn`, `pmi_itemknn`
- Not selected (slightly below baseline): `idf_cooccurrence`, `pop_capped_cooccurrence`, `jaccard_itemknn`
- Similar-person candidates remain excluded from offline metrics until a train-gated `similar_person_hobbies.csv` is available.

`segment_popularity`, `bm25_itemknn`, and `pmi_itemknn` are still available for explicit ablation/history comparisons, but they are not part of the default Stage 1 recommendation path because they consistently degraded validation recall.

## Evaluate Stage 2 ranking paths

The repo currently keeps two Stage 2 paths:

- **v1 deterministic reranker**: legacy fallback and historical comparison baseline
- **v2 LightGBM learned ranker**: promoted default ranking strategy

The deterministic evaluator still runs after Stage 1 candidate generation and does not train a new model or call Neo4j/FastAPI.

```powershell
.\.venv\Scripts\python.exe GNN_Neural_Network\scripts\evaluate_reranker.py --split validation
```

The evaluator treats the selected Stage 1 baseline (`popularity + cooccurrence`) as the promotion gate for Stage 2. Candidate recall is reported against the full `candidate_k` pool, and the output includes reranker weights plus Stage 2 fallback counts. Text fit is disabled by default for leakage safety.

Evaluate the promoted LightGBM ranking path with:

```powershell
.\.venv\Scripts\python.exe GNN_Neural_Network\scripts\evaluate_ranker.py --split validation
```

**Status:** The Stage 2 reranker has evolved through two phases:

1. **v1 Deterministic reranker** — weighted scoring, passed initial promotion gate.
2. **v2 LightGBM learned ranker** — binary classifier (AUC=0.8890555966387075, best_iteration=84), **PASSED** the promotion gate on both validation and test splits. **Now the promoted default recommendation strategy.**

   - Validation: recall@10=0.7391 (+0.029), ndcg@10=0.4580 (+0.0156) vs v1
   - Test: recall@10=0.7097 (+0.0054), ndcg@10=0.4477 (+0.0074) vs v1
   - Note: coverage@10=0.1556 and novelty@10=4.5843 are lower than v1 (0.517, 4.732) — diversity improvement is deferred to KURE dense embeddings.
   - Negative sampling ablation: completed. `neg_ratio=4, hard_ratio=1.0` won validation but underperformed the current `hard_ratio=0.8` default on final test Recall/NDCG, so the default remains `neg_ratio=4, hard_ratio=0.8`.
   - Source one-hot ablation: completed and rejected. `include_source_features=true` lowered validation Recall/NDCG and coverage, so the default remains `include_source_features=false`.
   - Phase 2.5 default decision closure: completed. This config is the fixed baseline for KURE dense embedding MMR and leakage-safe text embedding experiments.

3. **MMR diversity reordering** — **NO-GO**. Category one-hot embedding produces binary cosine similarity (0 or 1), making MMR a no-op within same-category items. All lambda values (0.1–0.9) produced effectively identical results to the baseline. MMR remains available as `--use-mmr` flag (default: false).
4. **Phase 5-A KURE dense embedding MMR** — **NO-GO**. Lambda `0.5`, `0.7`, `0.8`, `0.9` 모두 validation 게이트에서 recall/ndcg accuracy 기준 미달해 모두 `blocked`. test는 winner 부재로 생략, 기본은 `MMR=false` 유지.

**Current default pipeline:**

```
Stage 1 = popularity + cooccurrence (candidate generation)
Stage 2 = LightGBM learned ranker (relevance scoring)
MMR     = off (optional flag only)
```

**Next priorities:**

1. Phase 5-A 정리: KURE dense embedding MMR 재평가는 `phase5_kure_mmr_summary.md`로 마무리되어 `NO-GO`; 기본 경로는 유지.
2. Leakage-safe text embedding feature ablation after audit passes
3. Maintain fixed comparison baseline from closed Phase 2.5 (`num_leaves=31`, `neg_ratio=4`, `hard_ratio=0.8`, `include_source_features=false`, `MMR=false`)

## Evaluate Stage 1 ablation (including item-item providers)

```powershell
.\.venv\Scripts\python.exe GNN_Neural_Network\scripts\evaluate_stage1_ablation.py --split validation
```

This evaluates all Stage 1 providers (single and combination) against the selected baseline. Output includes delta vs baseline for each provider/combination.

## Persistent experiment logs

Key experiment outputs are persisted under `GNN_Neural_Network/artifacts/` rather than only printed to the console.

- `metrics.json`: LightGCN training history and best checkpoint metrics
- `stage1_ablation_validation.json`, `stage1_ablation_test.json`: provider-only and provider-combination comparisons
- `rerank_metrics.json`: selected Stage 1 baseline vs Stage 2 metrics and promotion decision
- `artifacts/experiments/phase5_kure_mmr_summary.md`: Phase 5-A KURE dense embedding MMR validation sweep summary and NO-GO ruling
- `recommendation_quality_audit.json`: coverage, entropy, popularity-bias, and segment audit
- `sample_recommendations_review.json`: qualitative sample review payload
- `experiment_decisions.json`: explicit accepted/rejected component decisions
- `experiment_run_summary.md`: detailed narrative summary for later review, including architecture, row schema, negative sampling, feature policy, cache policy, metrics, and current diagnosis

If a provider or model is rejected, record the reason in `experiment_decisions.json` and reflect the current default path in this README.
