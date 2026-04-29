# GNN Neural Network PoC

Offline LightGCN hobby/leisure recommendation PoC for `Nemotron-Personas-Korea`.

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

Add `--rerank` to apply the deterministic Stage 2 reranker to the Stage 1 candidate pool before printing final recommendations.

If the UUID is unknown to the trained canonical vocabulary, the CLI falls back to global popularity recommendations from the gated train split. If LightGCN returns fewer than `top-k` candidates after known-hobby masking, popularity fills the remaining slots.

The recommendation CLI now records Stage 1 provider observability:

- `candidates_sample.json`: Stage 1 provider candidate samples and selected candidates
- `provider_contribution.json`: raw/normalized candidates per provider and selected source counts
- `fallback_usage.json`: unknown UUID and underfilled-candidate fallback usage

Selected Stage 1 baseline is currently `popularity + cooccurrence`.

- Primary/default providers: train-split global popularity, train-split co-occurrence
- Auxiliary/experimental provider: LightGCN
- Disabled by default after ablation: `segment_popularity`
- Similar-person candidates remain excluded from offline metrics until a train-gated `similar_person_hobbies.csv` is available.

`segment_popularity` is still available for explicit ablation/history comparisons, but it is not part of the default Stage 1 recommendation path because it consistently degraded validation recall.

## Evaluate Stage 2 reranker

Stage 2 is a deterministic weighted reranker that runs after Stage 1 candidate generation. It does not train a new model and does not call Neo4j/FastAPI.

```powershell
.\.venv\Scripts\python.exe GNN_Neural_Network\scripts\evaluate_reranker.py --split validation
```

The evaluator treats the selected Stage 1 baseline (`popularity + cooccurrence`) as the promotion gate for Stage 2. Candidate recall is reported against the full `candidate_k` pool, and the output includes reranker weights plus Stage 2 fallback counts. Text fit is disabled by default for leakage safety.

**Status:** The Stage 2 persona-aware reranker has passed the promotion gate (beating the selected baseline on both validation and test splits) and is now the **promoted default recommendation strategy**. The next planned experiment is improving the Stage 1 candidate pool using BM25/TF-IDF ItemKNN collaborative filtering to reduce popularity bias before reranking.
