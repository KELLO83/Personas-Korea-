# Persona Similarity Experiments

Experimental workspace for `Person -> Person` similar-persona reranking.

This workspace is separate from `GNN_Neural_Network/`, which owns `Person -> Hobby` recommendation.

## Pipeline

```text
Neo4j SIMILAR_TO candidates
  -> export pair rows
  -> build pair features, deterministic score, and weak labels
  -> optionally export persona texts and build text cosine features
  -> run one experiment script at a time
  -> write experiment-specific metrics and manual review artifacts
```

## Commands

Do not run these automatically during setup. They are documented for future reproduction.
Most long-running scripts reuse valid artifacts by default. Pass `--force` only when an artifact must be rebuilt.

```powershell
# Recommended before a serious reranker run: rebuild wider candidates.
.\.venv\Scripts\python.exe scripts\build_gds.py --top-k 50

.\.venv\Scripts\python.exe experiments\persona_similarity\scripts\export_pairs.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml
.\.venv\Scripts\python.exe experiments\persona_similarity\scripts\build_features.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml

# Baselines
.\.venv\Scripts\python.exe experiments\persona_similarity\scripts\evaluate_fastrp_baseline.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml
.\.venv\Scripts\python.exe experiments\persona_similarity\scripts\evaluate_deterministic_baseline.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml

# Main ranking models
.\.venv\Scripts\python.exe experiments\persona_similarity\scripts\train_lambdarank.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml
.\.venv\Scripts\python.exe experiments\persona_similarity\scripts\evaluate_lambdarank.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml
.\.venv\Scripts\python.exe experiments\persona_similarity\scripts\train_rank_xendcg.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml
.\.venv\Scripts\python.exe experiments\persona_similarity\scripts\evaluate_rank_xendcg.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml

# Hybrid score based on a trained model
.\.venv\Scripts\python.exe experiments\persona_similarity\scripts\evaluate_hybrid_score.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml --source-experiment lambdarank

# Final reranking / diversity probes
.\.venv\Scripts\python.exe experiments\persona_similarity\scripts\evaluate_diversity_rerank.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml --base-score fastrp_score --experiment-name diversity_rerank_fastrp

# Ablations
.\.venv\Scripts\python.exe experiments\persona_similarity\scripts\ablation_without_fastrp.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml
.\.venv\Scripts\python.exe experiments\persona_similarity\scripts\evaluate_ablation_without_fastrp.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml
.\.venv\Scripts\python.exe experiments\persona_similarity\scripts\ablation_without_low_info.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml
.\.venv\Scripts\python.exe experiments\persona_similarity\scripts\evaluate_ablation_without_low_info.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml
.\.venv\Scripts\python.exe experiments\persona_similarity\scripts\ablation_without_location.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml
.\.venv\Scripts\python.exe experiments\persona_similarity\scripts\evaluate_ablation_without_location.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml
.\.venv\Scripts\python.exe experiments\persona_similarity\scripts\ablation_without_hobby.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml
.\.venv\Scripts\python.exe experiments\persona_similarity\scripts\evaluate_ablation_without_hobby.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml

# Text embedding feature pipeline
.\.venv\Scripts\python.exe experiments\persona_similarity\scripts\export_persona_texts.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml
.\.venv\Scripts\python.exe experiments\persona_similarity\scripts\audit_text_feature_leakage.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml
.\.venv\Scripts\python.exe experiments\persona_similarity\scripts\build_text_embeddings.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml
.\.venv\Scripts\python.exe experiments\persona_similarity\scripts\build_text_features.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml

# Text-only and structured+text experiments
.\.venv\Scripts\python.exe experiments\persona_similarity\scripts\train_text_only_lambdarank.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml
.\.venv\Scripts\python.exe experiments\persona_similarity\scripts\evaluate_text_only_lambdarank.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml
.\.venv\Scripts\python.exe experiments\persona_similarity\scripts\train_structured_text_lambdarank.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml
.\.venv\Scripts\python.exe experiments\persona_similarity\scripts\evaluate_structured_text_lambdarank.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml
.\.venv\Scripts\python.exe experiments\persona_similarity\scripts\train_structured_text_rank_xendcg.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml
.\.venv\Scripts\python.exe experiments\persona_similarity\scripts\evaluate_structured_text_rank_xendcg.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml
.\.venv\Scripts\python.exe experiments\persona_similarity\scripts\evaluate_structured_text_hybrid_score.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml --source-experiment structured_text_lambdarank
```

## Current Status

Workspace and experiment code are initialized. No experiment has been executed and no reranker is promoted.
