import sys
import json
from pathlib import Path

import torch

from GNN_Neural_Network.gnn_recommender.baseline import popularity_candidate_provider
from GNN_Neural_Network.gnn_recommender.model import LightGCN
from GNN_Neural_Network.gnn_recommender.recommend import Candidate
from GNN_Neural_Network.scripts import recommend_for_persona


def _write_edges(path: Path, edges: list[tuple[int, int]]) -> None:
    path.write_text(
        "person_id,hobby_id\n" + "".join(f"{person_id},{hobby_id}\n" for person_id, hobby_id in edges),
        encoding="utf-8",
    )


def test_unknown_uuid_uses_popularity_fallback(tmp_path: Path, monkeypatch, capsys) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    train_edges = artifact_dir / "train_edges.csv"
    validation_edges = artifact_dir / "validation_edges.csv"
    test_edges = artifact_dir / "test_edges.csv"
    _write_edges(train_edges, [(1, 10), (2, 10), (3, 11)])
    _write_edges(validation_edges, [])
    _write_edges(test_edges, [])
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
train:
  device: cpu
eval:
  top_k: [1, 2]
  score_chunk_size: 2
paths:
  train_edges: {train_edges.as_posix()}
  validation_edges: {validation_edges.as_posix()}
  test_edges: {test_edges.as_posix()}
  checkpoint: {(artifact_dir / 'lightgcn_hobby.pt').as_posix()}
  sample_recommendations: {(artifact_dir / 'sample_recommendations.json').as_posix()}
  fallback_usage: {(artifact_dir / 'fallback_usage.json').as_posix()}
  score_normalization: {(artifact_dir / 'score_normalization.json').as_posix()}
  provider_contribution: {(artifact_dir / 'provider_contribution.json').as_posix()}
""".strip(),
        encoding="utf-8",
    )
    checkpoint = {
        "person_to_id": {"known": 1},
        "hobby_to_id": {"등산": 10, "요가": 11},
        "num_persons": 4,
        "num_hobbies": 12,
        "embedding_dim": 4,
        "num_layers": 1,
        "state_dict": {},
    }
    monkeypatch.setattr(recommend_for_persona, "_safe_torch_load", lambda path: checkpoint)
    monkeypatch.setattr(sys, "argv", ["recommend_for_persona.py", "--config", str(config_path), "--uuid", "unknown", "--top-k", "2"])

    recommend_for_persona.main()

    output = capsys.readouterr().out
    assert "Falling back to global popularity" in output
    assert "등산" in output
    assert "요가" in output
    fallback_usage = (artifact_dir / "fallback_usage.json").read_text(encoding="utf-8")
    assert "unknown_uuid_popularity" in fallback_usage
    assert "unknown_uuid" in fallback_usage
    fallback_payload = json.loads(fallback_usage)
    assert fallback_payload["fallback_provider_counts"] == {"popularity": 2}
    assert fallback_payload["popularity_count"] == 2
    assert fallback_payload["fallback_count"] == 2
    provider_contribution = (artifact_dir / "provider_contribution.json").read_text(encoding="utf-8")
    assert "popularity" in provider_contribution


def test_popularity_provider_excludes_known_items() -> None:
    candidates = popularity_candidate_provider(
        train_edges=[(1, 10), (2, 10), (3, 11), (4, 12)],
        person_id=1,
        top_k=2,
        known_hobbies={10},
    )

    assert [candidate.hobby_id for candidate in candidates] == [11, 12]


def test_known_uuid_can_write_stage2_rerank_sample(tmp_path: Path, monkeypatch, capsys) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    train_edges = artifact_dir / "train_edges.csv"
    validation_edges = artifact_dir / "validation_edges.csv"
    test_edges = artifact_dir / "test_edges.csv"
    _write_edges(train_edges, [(0, 0)])
    _write_edges(validation_edges, [])
    _write_edges(test_edges, [])
    person_context = tmp_path / "person_context.csv"
    person_context.write_text(
        "person_uuid,age,age_group,sex,occupation,district,province,family_type,housing_type,education_level,persona_text,professional_text,sports_text,arts_text,travel_text,culinary_text,family_text,hobbies_text,skills_text,career_goals,embedding_text\n"
        "known,30,30대,여자,개발자,강남구,서울특별시,1인 가구,아파트,학사,요가,,,,,,,,,,요가\n",
        encoding="utf-8",
    )
    hobby_profile = artifact_dir / "hobby_profile.json"
    hobby_profile.write_text(
        json.dumps(
            {
                "source": "train_split_only",
                "hobbies": {
                    "등산": {"hobby_id": 0, "train_popularity": 1, "distributions": {}, "cooccurring_hobbies": []},
                    "요가": {"hobby_id": 1, "train_popularity": 3, "distributions": {"age_group": {"30대": 3}}, "cooccurring_hobbies": []},
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    checkpoint_path = artifact_dir / "lightgcn_hobby.pt"
    model = LightGCN(num_persons=1, num_hobbies=2, embedding_dim=4, num_layers=0)
    torch.save(
        {
            "person_to_id": {"known": 0},
            "hobby_to_id": {"등산": 0, "요가": 1},
            "num_persons": 1,
            "num_hobbies": 2,
            "embedding_dim": 4,
            "num_layers": 0,
            "state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
train:
  device: cpu
eval:
  top_k: [1]
  score_chunk_size: 2
rerank:
  candidate_pool_size: 2
  use_text_fit: false
  weights:
    age_group_fit: 1.0
paths:
  train_edges: {train_edges.as_posix()}
  validation_edges: {validation_edges.as_posix()}
  test_edges: {test_edges.as_posix()}
  person_context_csv: {person_context.as_posix()}
  checkpoint: {checkpoint_path.as_posix()}
  sample_recommendations: {(artifact_dir / 'sample_recommendations.json').as_posix()}
  candidates_sample: {(artifact_dir / 'candidates_sample.json').as_posix()}
  rerank_sample: {(artifact_dir / 'rerank_sample.json').as_posix()}
  fallback_usage: {(artifact_dir / 'fallback_usage.json').as_posix()}
  score_normalization: {(artifact_dir / 'score_normalization.json').as_posix()}
  provider_contribution: {(artifact_dir / 'provider_contribution.json').as_posix()}
  hobby_profile: {hobby_profile.as_posix()}
""".strip(),
        encoding="utf-8",
    )
    captured_top_k: list[int] = []

    def _lightgcn_candidates(*args, **kwargs):
        captured_top_k.append(int(kwargs["top_k"]))
        return [Candidate(1, "lightgcn", 0.5, source_scores={"lightgcn": 0.5})]

    monkeypatch.setattr(
        recommend_for_persona,
        "lightgcn_candidate_provider",
        _lightgcn_candidates,
    )
    monkeypatch.setattr(sys, "argv", ["recommend_for_persona.py", "--config", str(config_path), "--uuid", "known", "--top-k", "1", "--rerank"])

    recommend_for_persona.main()

    output = capsys.readouterr().out
    assert "요가" in output
    rerank_sample = json.loads((artifact_dir / "rerank_sample.json").read_text(encoding="utf-8"))
    assert rerank_sample["recommendations"][0]["source"] == "stage2_reranker"
    assert rerank_sample["final_stage1_candidates"][0]["hobby_id"] == 1
    candidates_sample = json.loads((artifact_dir / "candidates_sample.json").read_text(encoding="utf-8"))
    assert candidates_sample["final_recommendations"][0]["source"] == "stage2_reranker"
    assert captured_top_k == [2]
    fallback_usage = json.loads((artifact_dir / "fallback_usage.json").read_text(encoding="utf-8"))
    assert fallback_usage["primary_fallback_reason"] == "lightgcn_underfilled"
    assert fallback_usage["candidate_pool_size"] == 2
    assert fallback_usage["lightgcn_underfilled_count"] == 1
    assert fallback_usage["fallback_provider_counts"] == {}
    assert fallback_usage["popularity_count"] == 0
    assert fallback_usage["segment_popularity_count"] == 0
    assert fallback_usage["cooccurrence_count"] == 0
