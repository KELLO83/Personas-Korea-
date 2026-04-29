import sys
from pathlib import Path

import pytest

from GNN_Neural_Network.scripts import sweep_reranker_weights


def test_sweep_parser_blocks_test_split(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["sweep_reranker_weights.py", "--split", "test"])

    with pytest.raises(SystemExit):
        sweep_reranker_weights.parse_args()


def test_sweep_sorts_by_ndcg_then_recall(tmp_path: Path, monkeypatch) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
train:
  device: cpu
eval:
  top_k: [5, 10, 20]
  score_chunk_size: 2
rerank:
  candidate_pool_size: 50
  use_text_fit: false
  weights:
    lightgcn_score: 0.25
paths:
  train_edges: {(artifact_dir / 'train_edges.csv').as_posix()}
  validation_edges: {(artifact_dir / 'validation_edges.csv').as_posix()}
  test_edges: {(artifact_dir / 'test_edges.csv').as_posix()}
  checkpoint: {(artifact_dir / 'lightgcn_hobby.pt').as_posix()}
  person_context_csv: {(artifact_dir / 'person_context.csv').as_posix()}
  hobby_profile: {(artifact_dir / 'hobby_profile.json').as_posix()}
  score_normalization: {(artifact_dir / 'score_normalization.json').as_posix()}
  artifact_dir: {artifact_dir.as_posix()}
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(sweep_reranker_weights, "_safe_torch_load", lambda path: {"person_to_id": {"p1": 0}, "hobby_to_id": {"h1": 0}, "num_persons": 1, "num_hobbies": 1, "embedding_dim": 1, "num_layers": 0, "state_dict": {}})
    monkeypatch.setattr(sweep_reranker_weights, "_read_indexed_edges", lambda path: [(0, 0)])
    monkeypatch.setattr(sweep_reranker_weights, "_known_from_edges", lambda edges: {0: {0}})
    monkeypatch.setattr(sweep_reranker_weights, "load_person_contexts", lambda path: {})
    monkeypatch.setattr(sweep_reranker_weights, "load_json", lambda path: {"source": "train_split_only", "hobbies": {}})
    monkeypatch.setattr(sweep_reranker_weights, "_normalization_method", lambda path: "rank_percentile")
    monkeypatch.setattr(sweep_reranker_weights, "choose_device", lambda configured: "cpu")

    class _DummyModel:
        def __init__(self, **kwargs):
            self.num_persons = 1
            self.num_hobbies = 1

        def to(self, device):
            return self

        def load_state_dict(self, state_dict):
            return None

    monkeypatch.setattr(sweep_reranker_weights, "LightGCN", _DummyModel)
    monkeypatch.setattr(sweep_reranker_weights, "build_normalized_adjacency", lambda *args, **kwargs: None)
    monkeypatch.setattr(sweep_reranker_weights, "compute_lightgcn_embeddings", lambda model, adjacency: (None, None))
    monkeypatch.setattr(sweep_reranker_weights, "build_popularity_counts", lambda train_edges: {})
    monkeypatch.setattr(sweep_reranker_weights, "build_cooccurrence_counts", lambda train_edges: {})

    current_weights: dict[str, float] = {}

    def _fake_rerank_rankings(**kwargs):
        current_weights.clear()
        current_weights.update(kwargs["weights"])
        return {0: [0]}

    def _fake_metrics(truth, rankings, top_k_values):
        if "segment_popularity_score" not in current_weights:
            return {"ndcg@10": 0.0, "recall@10": 0.0}
        segment = current_weights["segment_popularity_score"]
        mismatch = current_weights["mismatch_penalty"]
        return {"ndcg@10": segment, "recall@10": 1.0 - mismatch}

    saved: dict[str, object] = {}

    monkeypatch.setattr(
        sweep_reranker_weights,
        "_precompute_stage2_features",
        lambda **kwargs: [{"person_id": 0, "candidate_features": []}],
    )
    monkeypatch.setattr(sweep_reranker_weights, "_rerank_rankings", _fake_rerank_rankings)
    monkeypatch.setattr(sweep_reranker_weights, "summarize_ranking_metrics", _fake_metrics)
    monkeypatch.setattr(sweep_reranker_weights, "save_json", lambda path, value: saved.update({"path": path, "value": value}))
    monkeypatch.setattr(sys, "argv", ["sweep_reranker_weights.py", "--config", str(config_path), "--output", str(artifact_dir / "rerank_sweep_validation.json"), "--top-n", "2"])

    sweep_reranker_weights.main()

    payload = saved["value"]
    assert isinstance(payload, dict)
    top_results = payload["top_results"]
    assert isinstance(top_results, list)
    assert len(top_results) == 2
    first = top_results[0]
    second = top_results[1]
    assert first["metrics"]["ndcg@10"] >= second["metrics"]["ndcg@10"]
