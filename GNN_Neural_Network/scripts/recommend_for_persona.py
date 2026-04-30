from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GNN_Neural_Network.gnn_recommender.baseline import (  # noqa: E402
    build_cooccurrence_counts,
    build_popularity_counts,
    cooccurrence_candidate_provider,
    popularity_candidate_provider,
    segment_popularity_candidate_provider,
)
from GNN_Neural_Network.gnn_recommender.config import load_config  # noqa: E402
from GNN_Neural_Network.gnn_recommender.data import load_json, load_person_contexts, save_json  # noqa: E402
from GNN_Neural_Network.gnn_recommender.model import (  # noqa: E402
    LightGCN,
    build_normalized_adjacency,
    choose_device,
)
from GNN_Neural_Network.gnn_recommender.recommend import (  # noqa: E402
    Candidate,
    build_provider_contribution_artifact,
    candidate_to_dict,
    compute_lightgcn_embeddings,
    lightgcn_candidate_provider,
    merge_candidates_by_hobby,
    normalize_candidate_scores,
)
from GNN_Neural_Network.gnn_recommender.ranker import LightGBMRanker  # noqa: E402
from GNN_Neural_Network.gnn_recommender.ranker_explain import batch_generate_reasons  # noqa: E402
from GNN_Neural_Network.gnn_recommender.text_embedding import batch_compute_embedding_similarity  # noqa: E402
from GNN_Neural_Network.gnn_recommender.rerank import (  # noqa: E402
    HobbyCandidate,
    RerankedCandidate,
    build_rerank_features,
    build_reranker_config,
    merge_stage1_candidates,
    rerank_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recommend hobbies for a persona from a trained LightGCN checkpoint.")
    parser.add_argument("--config", type=Path, default=Path("GNN_Neural_Network/configs/lightgcn_hobby.yaml"))
    parser.add_argument("--uuid", required=True, help="Persona UUID")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--save-sample", action="store_true")
    parser.add_argument("--rerank", action="store_true", help="Apply deterministic Stage 2 reranker after Stage 1 candidate generation.")
    parser.add_argument("--use-learned-ranker", action="store_true", help="Apply LightGBM learned ranker (v2) instead of deterministic reranker.")
    parser.add_argument("--explain", action="store_true", help="Generate SHAP-based explanation reasons (requires --use-learned-ranker).")
    parser.add_argument("--use-text-embedding", action="store_true", help="Enable leakage-safe text embedding similarity feature.")
    parser.add_argument("--ranker-model", type=Path, default=Path("GNN_Neural_Network/artifacts/ranker_model.txt"), help="Path to trained LightGBM ranker model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.use_text_embedding:
        raise ValueError(
            "--use-text-embedding is not yet enabled for CLI inference. "
            "The split-aware masking/audit and train/eval feature contract are not fully wired."
        )
    checkpoint = _safe_torch_load(config.paths.checkpoint)
    person_to_id = _expect_mapping(checkpoint.get("person_to_id"), "person_to_id")
    hobby_to_id = _expect_mapping(checkpoint.get("hobby_to_id"), "hobby_to_id")
    id_to_hobby = {value: key for key, value in hobby_to_id.items()}
    train_edges = _read_indexed_edges(config.paths.train_edges)
    full_known = _known_from_edges(_read_indexed_edges(config.paths.train_edges) + _read_indexed_edges(config.paths.validation_edges) + _read_indexed_edges(config.paths.test_edges))
    contexts = load_person_contexts(config.paths.person_context_csv) if config.paths.person_context_csv.exists() else {}
    hobby_profile = load_json(config.paths.hobby_profile) if config.paths.hobby_profile.exists() else None
    resolved_hobby_profile: dict[str, object] = hobby_profile if isinstance(hobby_profile, dict) else {}
    hobby_taxonomy = _load_hobby_taxonomy(config.paths.hobby_taxonomy, config.paths.artifact_dir)
    if args.uuid not in person_to_id:
        normalization_method = _normalization_method(config.paths.score_normalization)
        popularity_candidates = normalize_candidate_scores(
            popularity_candidate_provider(train_edges, -1, set(), args.top_k),
            normalization_method,
        )
        payload = _payload_from_candidates(popularity_candidates, id_to_hobby)
        print("Unknown persona UUID. Falling back to global popularity recommendations.")
        _print_payload(payload)
        provider_candidates = {"popularity": popularity_candidates}
        save_json(
            config.paths.provider_contribution,
            build_provider_contribution_artifact(provider_candidates, popularity_candidates, args.top_k, "unknown_uuid_popularity"),
        )
        save_json(config.paths.candidates_sample, _candidates_sample(provider_candidates, popularity_candidates))
        save_json(
            config.paths.fallback_usage,
            _fallback_usage_record(
                uuid=args.uuid,
                requested_top_k=args.top_k,
                fallback_type="unknown_uuid_popularity",
                fallback_count=len(payload),
                used_popularity=True,
                known_hobby_count=0,
            ),
        )
        if args.save_sample:
            save_json(config.paths.sample_recommendations, {"uuid": args.uuid, "fallback": "popularity", "recommendations": payload})
        return
    device = choose_device(config.train.device)
    model = LightGCN(
        num_persons=int(checkpoint["num_persons"]),
        num_hobbies=int(checkpoint["num_hobbies"]),
        embedding_dim=int(checkpoint["embedding_dim"]),
        num_layers=int(checkpoint["num_layers"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    adjacency = build_normalized_adjacency(model.num_persons, model.num_hobbies, train_edges, device)
    person_embeddings, hobby_embeddings = compute_lightgcn_embeddings(model, adjacency)
    popularity_counts = build_popularity_counts(train_edges)
    cooccurrence_counts = build_cooccurrence_counts(train_edges)
    person_id = int(person_to_id[args.uuid])
    known = full_known.get(person_id, set())
    normalization_method = _normalization_method(config.paths.score_normalization)
    pool_k = max(args.top_k, config.rerank.candidate_pool_size) if (args.rerank or args.use_learned_ranker) else args.top_k
    if args.use_learned_ranker:
        provider_candidates = {
            "cooccurrence": normalize_candidate_scores(
                cooccurrence_candidate_provider(train_edges, person_id, known, pool_k, cooccurrence_counts=cooccurrence_counts),
                normalization_method,
            ),
            "popularity": normalize_candidate_scores(
                popularity_candidate_provider(train_edges, person_id, known, pool_k, popularity_counts=popularity_counts),
                normalization_method,
            ),
        }
    else:
        provider_candidates = {
            "lightgcn": normalize_candidate_scores(
                lightgcn_candidate_provider(
                    model=model,
                    adjacency=adjacency,
                    person_id=person_id,
                    known_hobbies=known,
                    top_k=pool_k,
                    chunk_size=config.eval.score_chunk_size,
                    device=device,
                    person_embeddings=person_embeddings,
                    hobby_embeddings=hobby_embeddings,
                ),
                normalization_method,
            ),
            "cooccurrence": normalize_candidate_scores(
                cooccurrence_candidate_provider(train_edges, person_id, known, pool_k, cooccurrence_counts=cooccurrence_counts),
                normalization_method,
            ),
            "popularity": normalize_candidate_scores(
                popularity_candidate_provider(train_edges, person_id, known, pool_k, popularity_counts=popularity_counts),
                normalization_method,
            ),
            "segment_popularity": normalize_candidate_scores(
                segment_popularity_candidate_provider(
                    resolved_hobby_profile,
                    contexts.get(args.uuid),
                    known,
                    pool_k,
                ),
                normalization_method,
            ),
        }
    selected_candidates = merge_candidates_by_hobby(provider_candidates, pool_k)
    final_stage1_candidates: list[Any]
    payload: list[dict[str, Any]]

    if args.use_learned_ranker:
        if not isinstance(hobby_profile, dict):
            final_stage1_candidates = selected_candidates[: args.top_k]
            payload = _payload_from_candidates(final_stage1_candidates, id_to_hobby)
            _print_payload(payload)
            save_json(config.paths.fallback_usage, {
                "uuid": args.uuid,
                "requested_top_k": args.top_k,
                "candidate_pool_size": pool_k,
                "primary_fallback_reason": "learned_ranker_profile_missing",
                "fallback_type": "stage1_only",
                "fallback_events": {"stage2_profile_missing": True},
            })
            return
        hobby_candidates = merge_stage1_candidates(selected_candidates, id_to_hobby)
        known_names = {id_to_hobby[hobby_id] for hobby_id in known if hobby_id in id_to_hobby}
        if not args.ranker_model.exists():
            raise FileNotFoundError(f"Ranker model not found: {args.ranker_model}. Run train_ranker.py first.")
        ranker = LightGBMRanker.load(args.ranker_model)
        model_feature_columns = ranker.feature_columns()

        person_context = contexts.get(args.uuid)
        if person_context is None:
            final_stage1_candidates = selected_candidates[: args.top_k]
            payload = _payload_from_candidates(final_stage1_candidates, id_to_hobby)
            _print_payload(payload)
            save_json(config.paths.fallback_usage, {
                "uuid": args.uuid,
                "requested_top_k": args.top_k,
                "candidate_pool_size": pool_k,
                "primary_fallback_reason": "learned_ranker_context_missing",
                "fallback_type": "stage1_only",
                "fallback_events": {"missing_context": True},
            })
            return
        text_sim_map: dict[int, float] = {}

        features_list: list[list[float]] = []
        hobby_ids_list: list[int] = []
        for candidate in hobby_candidates:
            features = build_rerank_features(
                person_context, candidate,
                resolved_hobby_profile,
                known_names, build_reranker_config(config.rerank.use_text_fit, config.rerank.weights),
                text_embedding_similarity=text_sim_map.get(candidate.hobby_id, 0.0),
            )
            features.pop("similar_person_score", None)
            features.pop("persona_text_fit", None)
            if not args.use_text_embedding:
                features["text_embedding_similarity"] = 0.0
            features_list.append([features.get(col, 0.0) for col in model_feature_columns])
            hobby_ids_list.append(candidate.hobby_id)

        X = np.array(features_list, dtype=np.float32)
        scores = ranker.predict(X)
        sorted_indices = np.argsort(-scores)
        top_indices = [int(i) for i in sorted_indices[: args.top_k]]

        reasons: list[str] = []
        if args.explain:
            try:
                _, reasons = batch_generate_reasons(ranker, X[top_indices], model_feature_columns, top_k=3)
            except Exception:
                reasons = [""] * len(top_indices)
        else:
            reasons = [""] * len(top_indices)

        payload = _payload_from_learned_ranker(hobby_candidates, top_indices, scores, reasons, id_to_hobby)
        final_stage1_candidates = [
            selected_candidates[hobby_ids_list.index(hobby_candidates[idx].hobby_id)]
            for idx in top_indices if hobby_candidates[idx].hobby_id in hobby_ids_list
        ] or selected_candidates[: args.top_k]
        save_json(
            config.paths.rerank_sample,
            {
                "uuid": args.uuid,
                "recommendations": payload,
                "stage1_candidate_pool_size": len(selected_candidates),
                "model_feature_columns": model_feature_columns,
            },
        )
    elif args.rerank:
        hobby_candidates = merge_stage1_candidates(selected_candidates, id_to_hobby)
        known_names = {id_to_hobby[hobby_id] for hobby_id in known if hobby_id in id_to_hobby}
        reranked = rerank_candidates(
            contexts.get(args.uuid),
            hobby_candidates,
            resolved_hobby_profile,
            known_names,
            build_reranker_config(config.rerank.use_text_fit, config.rerank.weights),
            hobby_taxonomy=hobby_taxonomy,
        )
        final_reranked = reranked[: args.top_k]
        final_stage1_candidates = _stage1_candidates_for_reranked(final_reranked, selected_candidates)
        payload = _payload_from_reranked(final_reranked)
        save_json(
            config.paths.rerank_sample,
            {
                "uuid": args.uuid,
                "recommendations": payload,
                "stage1_candidate_pool_size": len(selected_candidates),
                "final_stage1_candidates": [candidate_to_dict(candidate) for candidate in final_stage1_candidates],
            },
        )
    else:
        final_stage1_candidates = selected_candidates[: args.top_k]
        payload = _payload_from_candidates(final_stage1_candidates, id_to_hobby)
    lightgcn_count = len(provider_candidates.get("lightgcn", []))
    fallback_usage = _fallback_usage_record(
        uuid=args.uuid,
        requested_top_k=args.top_k,
        candidate_pool_size=pool_k,
        lightgcn_count=lightgcn_count,
        selected_candidates=final_stage1_candidates,
        has_context=args.uuid in contexts,
        has_train_profile=isinstance(hobby_profile, dict) and hobby_profile.get("source") == "train_split_only",
        stage2_enabled=args.rerank or args.use_learned_ranker,
    )
    _print_payload(payload)
    save_json(
        config.paths.provider_contribution,
        build_provider_contribution_artifact(provider_candidates, final_stage1_candidates, args.top_k, str(fallback_usage["primary_fallback_reason"])),
    )
    save_json(config.paths.candidates_sample, _candidates_sample(provider_candidates, final_stage1_candidates, payload))
    save_json(config.paths.fallback_usage, fallback_usage)
    if args.save_sample:
        save_json(
            config.paths.sample_recommendations,
            {"uuid": args.uuid, "rerank": args.rerank, "recommendations": payload, "candidates": [candidate_to_dict(candidate) for candidate in selected_candidates]},
        )


def _print_payload(payload: list[dict[str, float | str]]) -> None:
    for index, item in enumerate(payload, start=1):
        print(f"{index}. {item['hobby']} | score={float(item['score']):.6f}")


def _payload_from_candidates(candidates: list[Candidate], id_to_hobby: dict[int, str]) -> list[dict[str, float | str]]:
    return [
        {"hobby": id_to_hobby[candidate.hobby_id], "score": candidate.score, "source": candidate.provider}
        for candidate in candidates
    ]


def _payload_from_reranked(candidates: list[RerankedCandidate]) -> list[dict[str, float | str]]:
    return [
        {"hobby": candidate.hobby_name, "score": candidate.final_score, "source": "stage2_reranker"}
        for candidate in candidates
    ]


def _payload_from_learned_ranker(
    hobby_candidates: list[HobbyCandidate],
    top_indices: list[int],
    scores: np.ndarray,
    reasons: list[str],
    id_to_hobby: dict[int, str],
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for rank, idx in enumerate(top_indices, start=1):
        candidate = hobby_candidates[idx]
        reason = reasons[rank - 1] if rank - 1 < len(reasons) else ""
        payload.append({
            "hobby": id_to_hobby.get(candidate.hobby_id, candidate.hobby_name),
            "score": float(scores[idx]),
            "source": "learned_ranker",
            "reason": reason,
        })
    return payload


def _stage1_candidates_for_reranked(reranked: list[RerankedCandidate], stage1_candidates: list[Candidate]) -> list[Candidate]:
    by_hobby_id = {candidate.hobby_id: candidate for candidate in stage1_candidates}
    return [by_hobby_id[candidate.hobby_id] for candidate in reranked if candidate.hobby_id in by_hobby_id]


def _normalization_method(path: Path) -> str:
    if not path.exists():
        return "rank_percentile"
    value = load_json(path)
    if not isinstance(value, dict):
        return "rank_percentile"
    method = value.get("method", "rank_percentile")
    return str(method)


def _load_hobby_taxonomy(configured_path: Path, artifact_dir: Path) -> dict[str, object] | None:
    for path in (configured_path, artifact_dir / "hobby_taxonomy.json"):
        if path.exists():
            value = load_json(path)
            if isinstance(value, dict):
                return value
    return None


def _candidates_sample(provider_candidates: dict[str, list[Candidate]], selected_candidates: list[Candidate], final_recommendations: list[dict[str, float | str]] | None = None) -> dict[str, object]:
    return {
        "providers": {
            provider: [candidate_to_dict(candidate) for candidate in candidates[:10]]
            for provider, candidates in provider_candidates.items()
        },
        "selected_candidates": [candidate_to_dict(candidate) for candidate in selected_candidates],
        "final_recommendations": final_recommendations or [candidate_to_dict(candidate) for candidate in selected_candidates],
    }


def _fallback_usage_record(
    *,
    uuid: str,
    requested_top_k: int,
    candidate_pool_size: int = 0,
    lightgcn_count: int = 0,
    selected_candidates: list[Candidate] | None = None,
    has_context: bool = False,
    has_train_profile: bool = False,
    fallback_type: str | None = None,
    fallback_count: int = 0,
    used_popularity: bool = False,
    known_hobby_count: int = 0,
    stage2_enabled: bool = False,
) -> dict[str, object]:
    events: dict[str, int | bool] = {}
    if fallback_type == "unknown_uuid_popularity":
        events["unknown_uuid"] = True
    expected_lightgcn_count = candidate_pool_size or requested_top_k
    if fallback_type != "unknown_uuid_popularity" and lightgcn_count < expected_lightgcn_count:
        events["lightgcn_underfilled"] = expected_lightgcn_count - lightgcn_count
    if stage2_enabled and not has_context and fallback_type != "unknown_uuid_popularity":
        events["missing_context"] = True
    if stage2_enabled and not has_train_profile and fallback_type != "unknown_uuid_popularity":
        events["stage2_profile_missing"] = True
    selected = selected_candidates or []
    provider_counts: dict[str, int] = {}
    for candidate in selected[:requested_top_k]:
        provider_counts[candidate.provider] = provider_counts.get(candidate.provider, 0) + 1
    lightgcn_underfilled_count = _safe_int(events.get("lightgcn_underfilled", 0))
    fallback_provider_counts = {provider: count for provider, count in provider_counts.items() if provider != "lightgcn"}
    if fallback_type == "unknown_uuid_popularity" and not fallback_provider_counts:
        fallback_provider_counts = {"popularity": fallback_count}
    popularity_count = fallback_provider_counts.get("popularity", 0)
    segment_popularity_count = fallback_provider_counts.get("segment_popularity", 0)
    cooccurrence_count = fallback_provider_counts.get("cooccurrence", 0)
    primary = next(iter(events), "none")
    return {
        "uuid": uuid,
        "requested_top_k": requested_top_k,
        "candidate_pool_size": candidate_pool_size,
        "primary_fallback_reason": primary,
        "fallback_type": fallback_type or primary,
        "fallback_events": events,
        "lightgcn_underfilled_count": lightgcn_underfilled_count,
        "fallback_provider_counts": fallback_provider_counts,
        "fallback_count": fallback_count or sum(fallback_provider_counts.values()),
        "popularity_count": popularity_count,
        "segment_popularity_count": segment_popularity_count,
        "cooccurrence_count": cooccurrence_count,
        "used_popularity": used_popularity or popularity_count > 0,
        "selected_provider_counts": provider_counts,
        "known_hobby_count": known_hobby_count,
        "cold_start": primary == "unknown_uuid",
    }


def _safe_int(value: object) -> int:
    return int(value) if isinstance(value, int | float | str) else 0


def _safe_torch_load(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint {path} must contain a dictionary")
    return value


def _expect_mapping(value: object, name: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint missing mapping: {name}")
    return {str(key): int(item) for key, item in value.items()}


def _read_indexed_edges(path: Path) -> list[tuple[int, int]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return [(int(row["person_id"]), int(row["hobby_id"])) for row in reader]


def _known_from_edges(edges: list[tuple[int, int]]) -> dict[int, set[int]]:
    known: dict[int, set[int]] = {}
    for person_id, hobby_id in edges:
        known.setdefault(person_id, set()).add(hobby_id)
    return known


if __name__ == "__main__":
    main()
