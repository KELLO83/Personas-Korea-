from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GNN_Neural_Network.gnn_recommender.config import load_config  # noqa: E402
from GNN_Neural_Network.gnn_recommender.data import (  # noqa: E402
    build_hobby_profile,
    build_initial_fallback_usage,
    build_leakage_audit,
    build_score_normalization_config,
    index_edges,
    load_alias_map,
    load_hobby_taxonomy,
    load_taxonomy_review,
    load_person_contexts,
    load_person_hobby_edges,
    merge_review_into_taxonomy,
    prepare_hobby_edges,
    save_json,
    split_edges_by_person,
    write_edges,
)
from GNN_Neural_Network.gnn_recommender.train import train_lightgcn  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train offline LightGCN hobby recommender. Do not run unless training is intended.")
    parser.add_argument("--config", type=Path, default=Path("GNN_Neural_Network/configs/lightgcn_hobby.yaml"))
    parser.add_argument("--prepare-only", action="store_true", help="Build mappings/splits but skip model training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if config.train.negative_samples != 1:
        raise ValueError("v1 supports negative_samples=1")
    edges = load_person_hobby_edges(config.paths.edge_csv)
    alias_map = load_alias_map(config.data.alias_map_path)
    hobby_taxonomy = load_hobby_taxonomy(config.data.hobby_taxonomy_path)
    review = load_taxonomy_review(config.data.hobby_taxonomy_review_path)
    hobby_taxonomy = merge_review_into_taxonomy(hobby_taxonomy, review)
    prepared = prepare_hobby_edges(
        edges,
        normalize_hobbies=config.data.normalize_hobbies,
        alias_map=alias_map,
        hobby_taxonomy=hobby_taxonomy,
        min_item_degree=config.data.min_item_degree,
        rare_item_policy=config.data.rare_item_policy,
    )
    indexed = index_edges(prepared.edges)
    split = split_edges_by_person(
        indexed.edges,
        validation_ratio=config.split.validation_ratio,
        test_ratio=config.split.test_ratio,
        min_eval_hobbies=config.split.min_eval_hobbies,
        two_hobby_policy=config.split.two_hobby_policy,
        seed=config.train.seed,
    )
    config.paths.artifact_dir.mkdir(parents=True, exist_ok=True)
    save_json(config.paths.vocabulary_report, prepared.report)
    save_json(config.paths.person_mapping, indexed.person_to_id)
    save_json(config.paths.hobby_mapping, indexed.hobby_to_id)
    write_edges(config.paths.train_edges, split.train)
    write_edges(config.paths.validation_edges, split.validation)
    write_edges(config.paths.test_edges, split.test)
    contexts = load_person_contexts(config.paths.person_context_csv) if config.paths.person_context_csv.exists() else None
    save_json(config.paths.hobby_profile, build_hobby_profile(split.train, indexed.person_to_id, indexed.hobby_to_id, contexts))
    save_json(config.paths.leakage_audit, build_leakage_audit(split, indexed.person_to_id, indexed.hobby_to_id, contexts))
    save_json(config.paths.fallback_usage, build_initial_fallback_usage(split))
    save_json(config.paths.score_normalization, build_score_normalization_config())
    save_json(config.paths.hobby_aliases, prepared.canonicalization.get("manual_aliases", {}))
    save_json(
        config.paths.hobby_taxonomy,
        {
            "rules": prepared.canonicalization.get("rules", []),
            "taxonomy": prepared.canonicalization.get("taxonomy", {}),
            "display_examples": prepared.canonicalization.get("display_examples", {}),
        },
    )
    save_json(config.paths.canonical_hobby_examples, prepared.canonicalization.get("observed_examples", {}))
    shutil.copyfile(args.config, config.paths.config_snapshot)
    if args.prepare_only:
        print("Prepared mappings and split files. Training skipped because --prepare-only was set.")
        return
    metrics = train_lightgcn(indexed=indexed, split=split, config=config)
    print(f"Training complete. best_recall@10={metrics.get('best_recall@10', 0.0):.4f}")


if __name__ == "__main__":
    main()
