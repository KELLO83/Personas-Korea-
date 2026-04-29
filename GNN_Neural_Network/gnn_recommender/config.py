from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    normalize_hobbies: bool = True
    alias_map_path: Path | None = None
    hobby_taxonomy_path: Path | None = None
    hobby_taxonomy_review_path: Path | None = None
    min_item_degree: int = 3
    rare_item_policy: str = "drop"


@dataclass(frozen=True)
class SplitConfig:
    validation_ratio: float = 0.1
    test_ratio: float = 0.1
    min_eval_hobbies: int = 3
    two_hobby_policy: str = "train_only"


@dataclass(frozen=True)
class TrainConfig:
    embedding_dim: int = 64
    num_layers: int = 2
    batch_size: int = 4096
    negative_samples: int = 1
    epochs: int = 10
    learning_rate: float = 0.001
    optimizer: str = "adam"
    weight_decay: float = 0.0
    seed: int = 42
    device: str = "cuda_if_available"
    scheduler: str = "constant"


@dataclass(frozen=True)
class EvalConfig:
    top_k: tuple[int, ...] = (5, 10, 20)
    score_chunk_size: int = 4096


@dataclass(frozen=True)
class RerankConfig:
    candidate_pool_size: int = 50
    use_text_fit: bool = False
    weights: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PathConfig:
    edge_csv: Path = Path("GNN_Neural_Network/data/person_hobby_edges.csv")
    person_context_csv: Path = Path("GNN_Neural_Network/data/person_context.csv")
    artifact_dir: Path = Path("GNN_Neural_Network/artifacts")
    train_edges: Path = Path("GNN_Neural_Network/artifacts/train_edges.csv")
    validation_edges: Path = Path("GNN_Neural_Network/artifacts/validation_edges.csv")
    test_edges: Path = Path("GNN_Neural_Network/artifacts/test_edges.csv")
    person_mapping: Path = Path("GNN_Neural_Network/artifacts/person_mapping.json")
    hobby_mapping: Path = Path("GNN_Neural_Network/artifacts/hobby_mapping.json")
    checkpoint: Path = Path("GNN_Neural_Network/artifacts/lightgcn_hobby.pt")
    metrics: Path = Path("GNN_Neural_Network/artifacts/metrics.json")
    vocabulary_report: Path = Path("GNN_Neural_Network/artifacts/vocabulary_report.json")
    config_snapshot: Path = Path("GNN_Neural_Network/artifacts/config_snapshot.yaml")
    sample_recommendations: Path = Path("GNN_Neural_Network/artifacts/sample_recommendations.json")
    candidates_sample: Path = Path("GNN_Neural_Network/artifacts/candidates_sample.json")
    hobby_profile: Path = Path("GNN_Neural_Network/artifacts/hobby_profile.json")
    leakage_audit: Path = Path("GNN_Neural_Network/artifacts/leakage_audit.json")
    fallback_usage: Path = Path("GNN_Neural_Network/artifacts/fallback_usage.json")
    score_normalization: Path = Path("GNN_Neural_Network/artifacts/score_normalization.json")
    provider_contribution: Path = Path("GNN_Neural_Network/artifacts/provider_contribution.json")
    hobby_aliases: Path = Path("GNN_Neural_Network/artifacts/hobby_aliases.json")
    hobby_taxonomy: Path = Path("GNN_Neural_Network/artifacts/hobby_taxonomy.json")
    canonical_hobby_examples: Path = Path("GNN_Neural_Network/artifacts/canonical_hobby_examples.json")
    rerank_metrics: Path = Path("GNN_Neural_Network/artifacts/rerank_metrics.json")
    reranker_weights: Path = Path("GNN_Neural_Network/artifacts/reranker_weights.json")
    rerank_sample: Path = Path("GNN_Neural_Network/artifacts/rerank_sample.json")


@dataclass(frozen=True)
class LightGCNConfig:
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    paths: PathConfig = field(default_factory=PathConfig)


def load_config(path: Path) -> LightGCNConfig:
    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping")
    return LightGCNConfig(
        data=DataConfig(**_normalize_data(_section(raw, "data"), path.parent)),
        split=SplitConfig(**_section(raw, "split")),
        train=TrainConfig(**_section(raw, "train")),
        eval=EvalConfig(**_normalize_eval(_section(raw, "eval"))),
        rerank=RerankConfig(**_section(raw, "rerank")),
        paths=PathConfig(**_normalize_paths(_section(raw, "paths"))),
    )


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    value = raw.get(key, {})
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{key}' must be a mapping")
    return value


def _normalize_eval(raw: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(raw)
    if "top_k" in normalized:
        normalized["top_k"] = tuple(int(item) for item in normalized["top_k"])
    return normalized


def _normalize_data(raw: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    normalized = dict(raw)
    for key in ("alias_map_path", "hobby_taxonomy_path", "hobby_taxonomy_review_path"):
        if normalized.get(key) in ("", None):
            normalized[key] = None
        elif key in normalized:
            path = Path(normalized[key])
            normalized[key] = path if path.is_absolute() else (base_dir / path).resolve()
    return normalized


def _normalize_paths(raw: dict[str, Any]) -> dict[str, Any]:
    return {key: Path(value) for key, value in raw.items()}
