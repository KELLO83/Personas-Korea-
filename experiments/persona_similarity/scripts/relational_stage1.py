from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


SUPPORTED_RELATIONAL_PROVIDERS = ("hgt", "rgcn")
BASELINE_STAGE1_PROVIDER = "fastrp_knn_topk50"
BASELINE_RERANKER = "lightgbm_lambdarank_or_rank_xendcg"


@dataclass(frozen=True)
class RelationalStage1Spec:
    experiment_id: str
    provider: str
    top_k: int = 50
    source_split: str = "source_uuid_group_split"
    reranker_recipe: str = BASELINE_RERANKER
    text_feature_policy: str = "stage2_pair_cosine_only"
    validation_first: bool = True
    winner_only_test: bool = True


def validate_relational_stage1_spec(spec: RelationalStage1Spec) -> None:
    if not spec.experiment_id.strip():
        raise ValueError("experiment_id must not be empty")
    if spec.provider not in SUPPORTED_RELATIONAL_PROVIDERS:
        allowed = ", ".join(SUPPORTED_RELATIONAL_PROVIDERS)
        raise ValueError(f"provider must be one of: {allowed}")
    if spec.top_k <= 0:
        raise ValueError("top_k must be positive")
    if spec.reranker_recipe != BASELINE_RERANKER:
        raise ValueError("relational Stage1 experiments must keep the reranker recipe fixed")
    if spec.text_feature_policy != "stage2_pair_cosine_only":
        raise ValueError("relational Stage1 experiments must keep text features as Stage2 pair cosine features")
    if not spec.validation_first or not spec.winner_only_test:
        raise ValueError("relational Stage1 experiments must be validation-first and winner-only-test")


def build_graph_schema_manifest(
    *,
    node_labels: list[str] | None = None,
    relationship_types: list[str] | None = None,
) -> dict[str, Any]:
    labels = node_labels or [
        "Person",
        "Hobby",
        "Skill",
        "Occupation",
        "District",
        "Province",
        "EducationLevel",
        "Field",
        "FamilyType",
        "HousingType",
        "Community",
    ]
    relationships = relationship_types or [
        "ENJOYS_HOBBY",
        "HAS_SKILL",
        "WORKS_AS",
        "LIVES_IN",
        "IN_PROVINCE",
        "HAS_EDUCATION",
        "STUDIED_FIELD",
        "HAS_FAMILY_TYPE",
        "HAS_HOUSING_TYPE",
        "IN_COMMUNITY",
    ]
    return {
        "node_labels": labels,
        "relationship_types": relationships,
        "edge_type_count": len(relationships),
        "purpose": "relational_persona_similarity_stage1_candidate_generation",
    }


def build_experiment_manifest(spec: RelationalStage1Spec) -> dict[str, Any]:
    validate_relational_stage1_spec(spec)
    return {
        "phase": "persona_similarity_relational_stage1",
        "status": "manifest_only",
        "baseline_stage1": BASELINE_STAGE1_PROVIDER,
        "provider": spec.provider,
        "experiment_id": spec.experiment_id,
        "fixed_controls": {
            "top_k": spec.top_k,
            "source_split": spec.source_split,
            "reranker_recipe": spec.reranker_recipe,
            "text_feature_policy": spec.text_feature_policy,
            "validation_first": spec.validation_first,
            "winner_only_test": spec.winner_only_test,
        },
        "graph_schema": build_graph_schema_manifest(),
        "promotion_gate": {
            "candidate_recall@50": ">= baseline",
            "ndcg@5": ">= baseline",
            "ndcg@10": ">= baseline",
            "explanation_coverage": ">= baseline",
            "diversity": "no regression",
            "refresh_cost": "acceptable",
        },
    }


def relational_stage1_promotion_passed(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    *,
    max_refresh_cost_ratio: float = 2.0,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for metric in ("candidate_recall@50", "ndcg@5", "ndcg@10", "explanation_coverage"):
        if float(candidate.get(metric, 0.0)) < float(baseline.get(metric, 0.0)):
            failures.append(f"{metric}_regressed")
    if float(candidate.get("diversity", baseline.get("diversity", 0.0))) < float(baseline.get("diversity", 0.0)):
        failures.append("diversity_regressed")
    baseline_cost = max(float(baseline.get("refresh_cost_seconds", 0.0)), 1.0)
    candidate_cost = float(candidate.get("refresh_cost_seconds", 0.0))
    if candidate_cost / baseline_cost > max_refresh_cost_ratio:
        failures.append("refresh_cost_too_high")
    return not failures, failures
