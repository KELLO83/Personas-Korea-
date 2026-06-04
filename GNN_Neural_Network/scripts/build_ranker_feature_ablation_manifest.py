from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GNN_Neural_Network.gnn_recommender.experiment_analysis import (  # noqa: E402
    FeatureAblationGroup,
    JsonValue,
    build_feature_ablation_manifest,
)
from GNN_Neural_Network.gnn_recommender.ranker import (  # noqa: E402
    RANKER_DOMAIN_TEXT_FEATURE_COLUMNS,
    RANKER_PHASE6_CROSS_FEATURE_COLUMNS,
    RANKER_SOURCE_FEATURE_COLUMNS,
    RANKER_TEXT_RANK_MARGIN_FEATURE_COLUMNS,
    get_ranker_feature_columns,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a one-variable hobby ranker feature ablation manifest.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-source-features", action="store_true")
    parser.add_argument("--include-domain-text-features", action="store_true")
    parser.add_argument("--include-rank-margin-features", action="store_true")
    parser.add_argument("--include-phase6-cross-features", action="store_true")
    args = parser.parse_args()

    baseline = get_ranker_feature_columns(
        include_source_features=bool(args.include_source_features),
        include_text_embedding_feature=True,
        include_domain_text_embedding_features=bool(args.include_domain_text_features),
        include_text_rank_margin_features=bool(args.include_rank_margin_features),
        include_phase6_cross_features=bool(args.include_phase6_cross_features),
    )
    manifest = build_feature_ablation_manifest(
        baseline_feature_columns=baseline,
        groups=_groups_for_columns(baseline),
    )
    _write_json(args.output, {"stage": "build_ranker_feature_ablation_manifest", **manifest})


def _groups_for_columns(columns: list[str]) -> tuple[FeatureAblationGroup, ...]:
    groups = [
        FeatureAblationGroup("without_text_embedding", ("text_embedding_similarity",)),
        FeatureAblationGroup("without_domain_text", tuple(RANKER_DOMAIN_TEXT_FEATURE_COLUMNS)),
        FeatureAblationGroup("without_text_rank_margin", tuple(RANKER_TEXT_RANK_MARGIN_FEATURE_COLUMNS)),
        FeatureAblationGroup("without_source_features", tuple(RANKER_SOURCE_FEATURE_COLUMNS)),
        FeatureAblationGroup("without_phase6_cross", tuple(RANKER_PHASE6_CROSS_FEATURE_COLUMNS)),
    ]
    present = set(columns)
    return tuple(group for group in groups if any(column in present for column in group.remove_columns))


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
