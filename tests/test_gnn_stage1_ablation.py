import json
import sys
from pathlib import Path

import torch

from GNN_Neural_Network.gnn_recommender.model import LightGCN
from GNN_Neural_Network.gnn_recommender.recommend import Candidate
from GNN_Neural_Network.scripts import evaluate_stage1_ablation


def _write_edges(path: Path, edges: list[tuple[int, int]]) -> None:
    path.write_text(
        "person_id,hobby_id\n" + "".join(f"{person_id},{hobby_id}\n" for person_id, hobby_id in edges),
        encoding="utf-8",
    )


def test_stage1_ablation_writes_validation_artifact(tmp_path: Path, monkeypatch) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    train_edges = artifact_dir / "train_edges.csv"
    validation_edges = artifact_dir / "validation_edges.csv"
    test_edges = artifact_dir / "test_edges.csv"
    _write_edges(train_edges, [(0, 0)])
    _write_edges(validation_edges, [(0, 2)])
    _write_edges(test_edges, [])
    person_context = tmp_path / "person_context.csv"
    person_context.write_text(
        "person_uuid,age,age_group,sex,occupation,district,province,family_type,housing_type,education_level,persona_text,professional_text,sports_text,arts_text,travel_text,culinary_text,family_text,hobbies_text,skills_text,career_goals,embedding_text\n"
        "p1,30,30대,여자,개발자,강남구,서울특별시,1인 가구,아파트,학사,,,,,,,,,,,\n",
        encoding="utf-8",
    )
    hobby_profile = artifact_dir / "hobby_profile.json"
    hobby_profile.write_text(json.dumps({"source": "train_split_only", "hobbies": {}}, ensure_ascii=False), encoding="utf-8")
    leakage_audit = artifact_dir / "leakage_audit.json"
    leakage_audit.write_text('{"status":"completed"}', encoding="utf-8")
    checkpoint_path = artifact_dir / "lightgcn_hobby.pt"
    model = LightGCN(num_persons=1, num_hobbies=3, embedding_dim=4, num_layers=0)
    torch.save(
        {
            "person_to_id": {"p1": 0},
            "hobby_to_id": {"known": 0, "h1": 1, "target": 2},
            "num_persons": 1,
            "num_hobbies": 3,
            "embedding_dim": 4,
            "num_layers": 0,
            "state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    output_path = artifact_dir / "stage1_ablation_validation.json"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
train:
  device: cpu
eval:
  top_k: [5, 10, 20]
  score_chunk_size: 2
rerank:
  candidate_pool_size: 3
  use_text_fit: false
paths:
  train_edges: {train_edges.as_posix()}
  validation_edges: {validation_edges.as_posix()}
  test_edges: {test_edges.as_posix()}
  person_context_csv: {person_context.as_posix()}
  checkpoint: {checkpoint_path.as_posix()}
  hobby_profile: {hobby_profile.as_posix()}
  leakage_audit: {leakage_audit.as_posix()}
  score_normalization: {(artifact_dir / 'missing_score_normalization.json').as_posix()}
  artifact_dir: {artifact_dir.as_posix()}
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        evaluate_stage1_ablation,
        "_provider_candidates",
        lambda **kwargs: {
            "popularity": [Candidate(2, "popularity", 3.0, source_scores={"popularity": 3.0})],
            "cooccurrence": [Candidate(1, "cooccurrence", 2.0, source_scores={"cooccurrence": 2.0})],
            "segment_popularity": [Candidate(2, "segment_popularity", 4.0, source_scores={"segment_popularity": 4.0})],
            "lightgcn": [Candidate(1, "lightgcn", 1.0, source_scores={"lightgcn": 1.0})],
        },
    )
    monkeypatch.setattr(sys, "argv", ["evaluate_stage1_ablation.py", "--config", str(config_path), "--split", "validation", "--output", str(output_path)])

    evaluate_stage1_ablation.main()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["split"] == "validation"
    assert payload["selected_stage1_baseline"] == ["popularity", "cooccurrence"]
    assert len(payload["provider_only"]) >= 4
    assert len(payload["provider_combinations"]) >= 1
    baseline = next(item for item in payload["provider_combinations"] if item["combo_name"] == "popularity+cooccurrence")
    assert "delta_vs_selected_baseline" in baseline
