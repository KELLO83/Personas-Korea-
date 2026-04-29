import copy
import json
from pathlib import Path

from GNN_Neural_Network.gnn_recommender.data import load_taxonomy_review, merge_review_into_taxonomy


def test_load_taxonomy_review_none_returns_empty_structure() -> None:
    assert load_taxonomy_review(None) == {
        "version": 1,
        "approved_clusters": [],
        "manual_aliases": {},
        "rejected_patterns": [],
        "split_required": [],
    }


def test_load_taxonomy_review_non_existent_path_returns_empty_structure(tmp_path: Path) -> None:
    assert load_taxonomy_review(tmp_path / "missing_review.json") == {
        "version": 1,
        "approved_clusters": [],
        "manual_aliases": {},
        "rejected_patterns": [],
        "split_required": [],
    }


def test_load_taxonomy_review_valid_review_file_parses_sections(tmp_path: Path) -> None:
    review_path = tmp_path / "taxonomy_review.json"
    _ = review_path.write_text(
        json.dumps(
            {
                "version": 2,
                "approved_clusters": [
                    {
                        "canonical_hobby": "산책",
                        "include_keywords": ["산책", "걷기"],
                        "exclude_keywords": ["반려견"],
                        "taxonomy": {"category": "야외활동"},
                        "source_cluster_id": "cluster-1",
                    }
                ],
                "manual_aliases": {"코인 노래방 방문": "노래방"},
                "rejected_patterns": ["시청"],
                "split_required": [{"original_suffix": "운동", "note": "too broad"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    review = load_taxonomy_review(review_path)

    assert review == {
        "version": 2,
        "approved_clusters": [
            {
                "canonical_hobby": "산책",
                "include_keywords": ["산책", "걷기"],
                "exclude_keywords": ["반려견"],
                "taxonomy": {"category": "야외활동"},
                "source_cluster_id": "cluster-1",
            }
        ],
        "manual_aliases": {"코인 노래방 방문": "노래방"},
        "rejected_patterns": ["시청"],
        "split_required": [{"original_suffix": "운동", "note": "too broad"}],
    }


def test_load_taxonomy_review_normalizes_hobby_names(tmp_path: Path) -> None:
    review_path = tmp_path / "taxonomy_review.json"
    _ = review_path.write_text(
        json.dumps(
            {
                "approved_clusters": [
                    {
                        "canonical_hobby": "  ＹＯＧＡ   Flow  ",
                        "include_keywords": ["  Morning   Yoga  "],
                        "exclude_keywords": [" Hot   ＹＯＧＡ "],
                        "taxonomy": {},
                    }
                ],
                "manual_aliases": {"  ＹＯＧＡ   Class ": " Yoga   Flow "},
                "rejected_patterns": ["  Hot   ＹＯＧＡ "],
                "split_required": [{"original_suffix": "  Group   ＹＯＧＡ  ", "note": " normalize note "}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    review = load_taxonomy_review(review_path)

    assert review["approved_clusters"] == [
        {
            "canonical_hobby": "yoga flow",
            "include_keywords": ["morning yoga"],
            "exclude_keywords": ["hot yoga"],
            "taxonomy": {},
        }
    ]
    assert review["manual_aliases"] == {"yoga class": "yoga flow"}
    assert review["rejected_patterns"] == ["hot yoga"]
    assert review["split_required"] == [{"original_suffix": "group yoga", "note": "normalize note"}]


def test_merge_review_into_taxonomy_returns_new_dict_without_mutating_inputs() -> None:
    taxonomy: dict[str, object] = {"rules": [], "manual_aliases": {"등산하기": "등산"}, "taxonomy": {}, "display_examples": {}}
    review: dict[str, object] = {
        "approved_clusters": [
            {"canonical_hobby": "산책", "include_keywords": ["산책"], "exclude_keywords": [], "taxonomy": {"category": "야외활동"}}
        ],
        "manual_aliases": {"걷기": "산책"},
    }
    original_taxonomy = copy.deepcopy(taxonomy)
    original_review = copy.deepcopy(review)

    merged = merge_review_into_taxonomy(taxonomy, review)

    assert merged is not taxonomy
    assert merged == {
        "rules": [
            {"canonical_hobby": "산책", "include_keywords": ["산책"], "exclude_keywords": [], "taxonomy": {"category": "야외활동"}}
        ],
        "manual_aliases": {"등산하기": "등산", "걷기": "산책"},
        "taxonomy": {"산책": {"category": "야외활동"}},
        "display_examples": {},
    }
    assert taxonomy == original_taxonomy
    assert review == original_review


def test_merge_review_into_taxonomy_appends_approved_clusters_to_rules() -> None:
    taxonomy: dict[str, object] = {
        "rules": [{"canonical_hobby": "등산", "include_keywords": ["등산"], "exclude_keywords": [], "taxonomy": {}}],
        "manual_aliases": {},
        "taxonomy": {},
        "display_examples": {},
    }
    review: dict[str, object] = {
        "approved_clusters": [
            {"canonical_hobby": "산책", "include_keywords": ["산책"], "exclude_keywords": [], "taxonomy": {}}
        ]
    }

    merged = merge_review_into_taxonomy(taxonomy, review)

    assert merged["rules"] == [
        {"canonical_hobby": "등산", "include_keywords": ["등산"], "exclude_keywords": [], "taxonomy": {}},
        {"canonical_hobby": "산책", "include_keywords": ["산책"], "exclude_keywords": [], "taxonomy": {}},
    ]


def test_merge_review_into_taxonomy_merges_manual_aliases_last_write_wins() -> None:
    taxonomy: dict[str, object] = {"rules": [], "manual_aliases": {"걷기": "걷기", "등산하기": "등산"}, "taxonomy": {}, "display_examples": {}}
    review: dict[str, object] = {"manual_aliases": {"걷기": "산책", "코인 노래방 방문": "노래방"}}

    merged = merge_review_into_taxonomy(taxonomy, review)

    assert merged["manual_aliases"] == {"걷기": "산책", "등산하기": "등산", "코인 노래방 방문": "노래방"}


def test_merge_review_into_taxonomy_adds_taxonomy_metadata_from_approved_clusters() -> None:
    taxonomy: dict[str, object] = {
        "rules": [],
        "manual_aliases": {},
        "taxonomy": {"등산": {"category": "스포츠"}},
        "display_examples": {},
    }
    review: dict[str, object] = {
        "approved_clusters": [
            {
                "canonical_hobby": "산책",
                "include_keywords": ["산책"],
                "exclude_keywords": [],
                "taxonomy": {"category": "야외활동", "intensity": "low"},
            }
        ]
    }

    merged = merge_review_into_taxonomy(taxonomy, review)

    assert merged["taxonomy"] == {
        "등산": {"category": "스포츠"},
        "산책": {"category": "야외활동", "intensity": "low"},
    }


def test_merge_review_into_taxonomy_empty_review_returns_copy_of_original_taxonomy() -> None:
    taxonomy: dict[str, object] = {
        "version": 1,
        "rules": [{"canonical_hobby": "등산", "include_keywords": ["등산"], "exclude_keywords": [], "taxonomy": {}}],
        "manual_aliases": {"등산하기": "등산"},
        "taxonomy": {"등산": {"category": "스포츠"}},
        "display_examples": {"등산": ["북한산 등산"]},
    }

    merged = merge_review_into_taxonomy(taxonomy, load_taxonomy_review(None))

    assert merged == taxonomy
    assert merged is not taxonomy
    assert merged["rules"] is not taxonomy["rules"]
    assert merged["manual_aliases"] is not taxonomy["manual_aliases"]
