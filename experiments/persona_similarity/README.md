# Persona Similarity Experiments

Experimental workspace for `Person -> Person` similar-persona reranking.

This workspace is separate from `experiments/hobby_recommender_ml/`, which owns `Person -> Hobby` recommendation.

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
# Optional experiment dependencies
.\.venv314\Scripts\python.exe -m pip install -r experiments\persona_similarity\requirements.txt

# Recommended before a serious reranker run: rebuild wider candidates.
.\.venv314\Scripts\python.exe ops\graph\build_gds.py --top-k 50

.\.venv314\Scripts\python.exe experiments\persona_similarity\scripts\export_pairs.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml
.\.venv314\Scripts\python.exe experiments\persona_similarity\scripts\build_features.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml

# Baselines
.\.venv314\Scripts\python.exe experiments\persona_similarity\scripts\evaluate_fastrp_baseline.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml
.\.venv314\Scripts\python.exe experiments\persona_similarity\scripts\evaluate_deterministic_baseline.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml

# Main ranking models
.\.venv314\Scripts\python.exe experiments\persona_similarity\scripts\train_lambdarank.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml
.\.venv314\Scripts\python.exe experiments\persona_similarity\scripts\evaluate_lambdarank.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml
.\.venv314\Scripts\python.exe experiments\persona_similarity\scripts\train_rank_xendcg.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml
.\.venv314\Scripts\python.exe experiments\persona_similarity\scripts\evaluate_rank_xendcg.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml

# Hybrid score based on a trained model
.\.venv314\Scripts\python.exe experiments\persona_similarity\scripts\evaluate_hybrid_score.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml --source-experiment lambdarank

# Final reranking / diversity probes
.\.venv314\Scripts\python.exe experiments\persona_similarity\scripts\evaluate_diversity_rerank.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml --base-score fastrp_score --experiment-name diversity_rerank_fastrp

# Ablations
.\.venv314\Scripts\python.exe experiments\persona_similarity\scripts\ablation_without_fastrp.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml
.\.venv314\Scripts\python.exe experiments\persona_similarity\scripts\evaluate_ablation_without_fastrp.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml
.\.venv314\Scripts\python.exe experiments\persona_similarity\scripts\ablation_without_low_info.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml
.\.venv314\Scripts\python.exe experiments\persona_similarity\scripts\evaluate_ablation_without_low_info.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml
.\.venv314\Scripts\python.exe experiments\persona_similarity\scripts\ablation_without_location.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml
.\.venv314\Scripts\python.exe experiments\persona_similarity\scripts\evaluate_ablation_without_location.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml
.\.venv314\Scripts\python.exe experiments\persona_similarity\scripts\ablation_without_hobby.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml
.\.venv314\Scripts\python.exe experiments\persona_similarity\scripts\evaluate_ablation_without_hobby.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml

# Text embedding feature pipeline
.\.venv314\Scripts\python.exe experiments\persona_similarity\scripts\export_persona_texts.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml
.\.venv314\Scripts\python.exe experiments\persona_similarity\scripts\audit_text_feature_leakage.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml
.\.venv314\Scripts\python.exe experiments\persona_similarity\scripts\build_text_embeddings.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml
.\.venv314\Scripts\python.exe experiments\persona_similarity\scripts\build_text_features.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml

# Text-only and structured+text experiments
.\.venv314\Scripts\python.exe experiments\persona_similarity\scripts\train_text_only_lambdarank.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml
.\.venv314\Scripts\python.exe experiments\persona_similarity\scripts\evaluate_text_only_lambdarank.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml
.\.venv314\Scripts\python.exe experiments\persona_similarity\scripts\train_structured_text_lambdarank.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml
.\.venv314\Scripts\python.exe experiments\persona_similarity\scripts\evaluate_structured_text_lambdarank.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml
.\.venv314\Scripts\python.exe experiments\persona_similarity\scripts\train_structured_text_rank_xendcg.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml
.\.venv314\Scripts\python.exe experiments\persona_similarity\scripts\evaluate_structured_text_rank_xendcg.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml
.\.venv314\Scripts\python.exe experiments\persona_similarity\scripts\evaluate_structured_text_hybrid_score.py --config experiments\persona_similarity\configs\lightgbm_reranker.yaml --source-experiment structured_text_lambdarank
```

## Current Status

Workspace and experiment code are initialized. No experiment has been executed and no reranker is promoted.
