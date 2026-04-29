import json
import sys
from pathlib import Path

from GNN_Neural_Network.scripts import train_lightgcn


def test_prepare_only_writes_vocabulary_report_and_baseline_ready_splits(tmp_path: Path, monkeypatch) -> None:
    edge_csv = tmp_path / "edges.csv"
    edge_csv.write_text(
        "person_uuid,hobby_name\n"
        "p1,등산하기\n"
        "p2,등산\n"
        "p3,등산\n"
        "p1,요가\n"
        "p2,요가\n"
        "p3,요가\n"
        "p4,요가\n"
        "p1,독서\n"
        "p2,독서\n"
        "p3,독서\n"
        "p4,독서\n",
        encoding="utf-8",
    )
    artifact_dir = tmp_path / "artifacts"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
data:
  normalize_hobbies: true
  alias_map_path:
  hobby_taxonomy_path: {(tmp_path / 'taxonomy.json').as_posix()}
  min_item_degree: 2
  rare_item_policy: drop
split:
  validation_ratio: 0.25
  test_ratio: 0.25
  min_eval_hobbies: 3
  two_hobby_policy: train_only
train:
  embedding_dim: 8
  num_layers: 1
  batch_size: 4
  negative_samples: 1
  epochs: 1
  learning_rate: 0.001
  optimizer: adam
  weight_decay: 0.0
  seed: 42
  device: cpu
  scheduler: constant
eval:
  top_k: [1, 2]
  score_chunk_size: 2
paths:
  edge_csv: {edge_csv.as_posix()}
  person_context_csv: {(tmp_path / 'missing_person_context.csv').as_posix()}
  artifact_dir: {artifact_dir.as_posix()}
  train_edges: {(artifact_dir / 'train_edges.csv').as_posix()}
  validation_edges: {(artifact_dir / 'validation_edges.csv').as_posix()}
  test_edges: {(artifact_dir / 'test_edges.csv').as_posix()}
  person_mapping: {(artifact_dir / 'person_mapping.json').as_posix()}
  hobby_mapping: {(artifact_dir / 'hobby_mapping.json').as_posix()}
  checkpoint: {(artifact_dir / 'lightgcn_hobby.pt').as_posix()}
  metrics: {(artifact_dir / 'metrics.json').as_posix()}
  vocabulary_report: {(artifact_dir / 'vocabulary_report.json').as_posix()}
  config_snapshot: {(artifact_dir / 'config_snapshot.yaml').as_posix()}
  sample_recommendations: {(artifact_dir / 'sample_recommendations.json').as_posix()}
  hobby_profile: {(artifact_dir / 'hobby_profile.json').as_posix()}
  leakage_audit: {(artifact_dir / 'leakage_audit.json').as_posix()}
  fallback_usage: {(artifact_dir / 'fallback_usage.json').as_posix()}
  score_normalization: {(artifact_dir / 'score_normalization.json').as_posix()}
  provider_contribution: {(artifact_dir / 'provider_contribution.json').as_posix()}
  hobby_aliases: {(artifact_dir / 'hobby_aliases.json').as_posix()}
  hobby_taxonomy: {(artifact_dir / 'hobby_taxonomy.json').as_posix()}
  canonical_hobby_examples: {(artifact_dir / 'canonical_hobby_examples.json').as_posix()}
""".strip(),
        encoding="utf-8",
    )
    (tmp_path / "taxonomy.json").write_text(
        json.dumps(
            {
                "version": 1,
                "manual_aliases": {"등산하기": "등산"},
                "taxonomy": {"등산": {"category": "야외활동"}},
                "display_examples": {"등산": ["등산하기", "등산"]},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["train_lightgcn.py", "--prepare-only", "--config", str(config_path)])

    train_lightgcn.main()

    report = json.loads((artifact_dir / "vocabulary_report.json").read_text(encoding="utf-8"))
    assert isinstance(report, dict)
    assert report["retained_hobbies"] == 3
    assert (artifact_dir / "train_edges.csv").exists()
    assert (artifact_dir / "validation_edges.csv").exists()
    assert (artifact_dir / "test_edges.csv").exists()
    assert (artifact_dir / "hobby_profile.json").exists()
    assert (artifact_dir / "leakage_audit.json").exists()
    assert (artifact_dir / "fallback_usage.json").exists()
    assert (artifact_dir / "score_normalization.json").exists()
    assert (artifact_dir / "hobby_aliases.json").exists()
    assert (artifact_dir / "hobby_taxonomy.json").exists()
    assert (artifact_dir / "canonical_hobby_examples.json").exists()
    train_edge_count = len((artifact_dir / "train_edges.csv").read_text(encoding="utf-8").strip().splitlines()) - 1
    profile = json.loads((artifact_dir / "hobby_profile.json").read_text(encoding="utf-8"))
    assert isinstance(profile, dict)
    assert profile["num_train_edges"] == train_edge_count
    assert profile["source"] == "train_split_only"
    audit = json.loads((artifact_dir / "leakage_audit.json").read_text(encoding="utf-8"))
    assert isinstance(audit, dict)
    assert audit["status"] == "skipped"
    aliases = json.loads((artifact_dir / "hobby_aliases.json").read_text(encoding="utf-8"))
    assert isinstance(aliases, dict)
    assert aliases["등산하기"] == "등산"
    taxonomy = json.loads((artifact_dir / "hobby_taxonomy.json").read_text(encoding="utf-8"))
    assert isinstance(taxonomy, dict)
    assert "rules" in taxonomy
