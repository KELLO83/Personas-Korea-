import json
import sys
from pathlib import Path

import torch

from GNN_Neural_Network.gnn_recommender.model import LightGCN
from GNN_Neural_Network.gnn_recommender.recommend import Candidate
from GNN_Neural_Network.scripts import evaluate_reranker


def _write_edges(path: Path, edges: list[tuple[int, int]]) -> None:
    path.write_text(
        "person_id,hobby_id\n" + "".join(f"{person_id},{hobby_id}\n" for person_id, hobby_id in edges),
        encoding="utf-8",
    )


def test_evaluate_reranker_uses_full_candidate_pool_for_candidate_recall(tmp_path: Path, monkeypatch) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    train_edges = artifact_dir / "train_edges.csv"
    validation_edges = artifact_dir / "validation_edges.csv"
    test_edges = artifact_dir / "test_edges.csv"
    _write_edges(train_edges, [(0, 0)])
    _write_edges(validation_edges, [(0, 3)])
    _write_edges(test_edges, [])
    person_context = tmp_path / "person_context.csv"
    person_context.write_text(
        "person_uuid,age,age_group,sex,occupation,district,province,family_type,housing_type,education_level,persona_text,professional_text,sports_text,arts_text,travel_text,culinary_text,family_text,hobbies_text,skills_text,career_goals,embedding_text\n"
        "p1,30,30대,여자,개발자,강남구,서울특별시,1인 가구,아파트,학사,,,,,,,,,,,\n",
        encoding="utf-8",
    )
    hobby_profile = artifact_dir / "hobby_profile.json"
    hobby_profile.write_text(
        json.dumps(
            {
                "source": "train_split_only",
                "hobbies": {
                    "known": {"hobby_id": 0, "train_popularity": 1, "distributions": {}, "cooccurring_hobbies": []},
                    "h1": {"hobby_id": 1, "train_popularity": 3, "distributions": {}, "cooccurring_hobbies": []},
                    "h2": {"hobby_id": 2, "train_popularity": 2, "distributions": {}, "cooccurring_hobbies": []},
                    "target": {"hobby_id": 3, "train_popularity": 1, "distributions": {}, "cooccurring_hobbies": []},
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    leakage_audit = artifact_dir / "leakage_audit.json"
    leakage_audit.write_text('{"status":"completed"}', encoding="utf-8")
    checkpoint_path = artifact_dir / "lightgcn_hobby.pt"
    model = LightGCN(num_persons=1, num_hobbies=4, embedding_dim=4, num_layers=0)
    torch.save(
        {
            "person_to_id": {"p1": 0},
            "hobby_to_id": {"known": 0, "h1": 1, "h2": 2, "target": 3},
            "num_persons": 1,
            "num_hobbies": 4,
            "embedding_dim": 4,
            "num_layers": 0,
            "state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    output_path = artifact_dir / "rerank_metrics.json"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
train:
  device: cpu
eval:
  top_k: [1, 2]
  score_chunk_size: 2
rerank:
  candidate_pool_size: 3
  use_text_fit: false
  weights:
    lightgcn_score: 0.7
paths:
  train_edges: {train_edges.as_posix()}
  validation_edges: {validation_edges.as_posix()}
  test_edges: {test_edges.as_posix()}
  person_context_csv: {person_context.as_posix()}
  checkpoint: {checkpoint_path.as_posix()}
  hobby_profile: {hobby_profile.as_posix()}
  leakage_audit: {leakage_audit.as_posix()}
  score_normalization: {(artifact_dir / 'missing_score_normalization.json').as_posix()}
  rerank_metrics: {output_path.as_posix()}
  reranker_weights: {(artifact_dir / 'reranker_weights.json').as_posix()}
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        evaluate_reranker,
        "lightgcn_candidate_provider",
        lambda *args, **kwargs: [
            Candidate(1, "lightgcn", 3.0, source_scores={"lightgcn": 3.0}),
            Candidate(2, "lightgcn", 2.0, source_scores={"lightgcn": 2.0}),
            Candidate(3, "lightgcn", 1.0, source_scores={"lightgcn": 1.0}),
        ],
    )
    monkeypatch.setattr(evaluate_reranker, "cooccurrence_candidate_provider", lambda *args, **kwargs: [])
    monkeypatch.setattr(evaluate_reranker, "popularity_candidate_provider", lambda *args, **kwargs: [])
    monkeypatch.setattr(evaluate_reranker, "segment_popularity_candidate_provider", lambda *args, **kwargs: [])
    monkeypatch.setattr(sys, "argv", ["evaluate_reranker.py", "--config", str(config_path), "--split", "validation"])

    evaluate_reranker.main()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["candidate_recall"]["recall@3"] == 1.0
    assert payload["stage1_multi_provider"]["recall@2"] == 0.0
    assert payload["reranker_weights"]["lightgcn_score"] == 0.7
    assert payload["reranker_weights"]["known_hobby_compatibility"] == 0.15
    assert payload["stage2_fallback_count"] == 0
    weights_payload = json.loads((artifact_dir / "reranker_weights.json").read_text(encoding="utf-8"))
    assert weights_payload["configured_weights"] == {"lightgcn_score": 0.7}
    assert weights_payload["effective_weights"]["lightgcn_score"] == 0.7
    assert weights_payload["candidate_k"] == 3
