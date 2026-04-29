import json
import random
from pathlib import Path

import pytest

from GNN_Neural_Network.gnn_recommender.data import (
    EdgeSplit,
    HobbyEdge,
    PersonContext,
    build_hobby_profile,
    build_leakage_audit,
    index_edges,
    load_alias_map,
    load_hobby_taxonomy,
    load_person_hobby_edges,
    normalize_hobby_name,
    prepare_hobby_edges,
    sample_negative,
    split_edges_by_person,
)


def test_load_person_hobby_edges_accepts_utf8_bom_header(tmp_path: Path) -> None:
    csv_path = tmp_path / "edges.csv"
    csv_path.write_text("person_uuid,hobby_name\np1, 등산 \n", encoding="utf-8-sig")

    edges = load_person_hobby_edges(csv_path)

    assert edges == [HobbyEdge(person_uuid="p1", hobby_name="등산")]


def test_normalize_hobby_name_trims_and_collapses_whitespace() -> None:
    assert normalize_hobby_name("  고궁   산책  ") == "고궁 산책"


def test_load_alias_map_normalizes_keys_and_values(tmp_path: Path) -> None:
    alias_path = tmp_path / "aliases.json"
    alias_path.write_text(json.dumps({" 등산하기 ": " 등산 "}, ensure_ascii=False), encoding="utf-8")

    aliases = load_alias_map(alias_path)

    assert aliases == {"등산하기": "등산"}


def test_load_alias_map_rejects_conflicting_normalized_keys(tmp_path: Path) -> None:
    alias_path = tmp_path / "aliases.json"
    alias_path.write_text(json.dumps({"A": "등산", "Ａ": "요가"}, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="conflicting"):
        load_alias_map(alias_path)


def test_load_hobby_taxonomy_supports_rules_and_manual_aliases(tmp_path: Path) -> None:
    taxonomy_path = tmp_path / "taxonomy.json"
    taxonomy_path.write_text(
        json.dumps(
            {
                "version": 1,
                "rules": [
                    {
                        "canonical_hobby": "산책",
                        "include_keywords": ["산책", "걷기"],
                        "exclude_keywords": [],
                        "taxonomy": {"category": "야외활동"},
                    }
                ],
                "manual_aliases": {"코인 노래방 방문": "노래방"},
                "display_examples": {"산책": ["석촌호수 주변 산책"]},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    taxonomy = load_hobby_taxonomy(taxonomy_path)
    manual_aliases = taxonomy["manual_aliases"]
    rules = taxonomy["rules"]
    display_examples = taxonomy["display_examples"]

    assert isinstance(manual_aliases, dict)
    assert isinstance(rules, list)
    assert isinstance(display_examples, dict)
    assert manual_aliases["코인 노래방 방문"] == "노래방"
    assert isinstance(rules[0], dict)
    assert rules[0]["canonical_hobby"] == "산책"
    assert display_examples["산책"] == ["석촌호수 주변 산책"]


def test_prepare_hobby_edges_aliases_before_degree_filtering() -> None:
    edges = [
        HobbyEdge("p1", "등산하기"),
        HobbyEdge("p2", " 등산 "),
        HobbyEdge("p3", "혼자만의 희귀 취미"),
    ]

    prepared = prepare_hobby_edges(
        edges,
        normalize_hobbies=True,
        alias_map={"등산하기": "등산"},
        min_item_degree=2,
        rare_item_policy="drop",
    )

    assert prepared.edges == [HobbyEdge("p1", "등산"), HobbyEdge("p2", "등산")]
    assert prepared.report["raw_hobbies"] == 3
    assert prepared.report["raw_singleton_hobbies"] == 3
    assert prepared.report["raw_singleton_ratio"] == 1.0
    assert prepared.report["canonical_hobbies"] == 2
    assert prepared.report["retained_hobbies"] == 1
    assert prepared.report["dropped_hobbies"] == 1


def test_prepare_hobby_edges_uses_taxonomy_rules_and_preserves_examples() -> None:
    edges = [
        HobbyEdge("p1", "석촌호수 주변 산책"),
        HobbyEdge("p2", "탄천 산책로 걷기"),
        HobbyEdge("p3", "코인 노래방 방문"),
    ]

    prepared = prepare_hobby_edges(
        edges,
        normalize_hobbies=True,
        alias_map={},
        hobby_taxonomy={
            "rules": [
                {
                    "canonical_hobby": "산책",
                    "include_keywords": ["산책", "걷기"],
                    "exclude_keywords": [],
                    "taxonomy": {"category": "야외활동"},
                }
            ],
            "manual_aliases": {"코인 노래방 방문": "노래방"},
            "taxonomy": {"산책": {"category": "야외활동"}, "노래방": {"category": "문화콘텐츠"}},
            "display_examples": {"산책": ["석촌호수 주변 산책"]},
        },
        min_item_degree=1,
        rare_item_policy="drop",
    )

    assert prepared.edges == [HobbyEdge("p1", "산책"), HobbyEdge("p2", "산책"), HobbyEdge("p3", "노래방")]
    taxonomy = prepared.canonicalization["taxonomy"]
    observed_examples = prepared.canonicalization["observed_examples"]
    assert isinstance(taxonomy, dict)
    assert isinstance(observed_examples, dict)
    assert isinstance(taxonomy["산책"], dict)
    assert taxonomy["산책"]["category"] == "야외활동"
    assert isinstance(observed_examples["산책"], list)
    assert "석촌호수 주변 산책" in observed_examples["산책"]
    assert "탄천 산책로 걷기" in observed_examples["산책"]


def test_prepare_hobby_edges_does_not_map_generic_viewing_to_movie_drama() -> None:
    prepared = prepare_hobby_edges(
        [HobbyEdge("p1", "유튜브 시사 및 교양 콘텐츠 시청")],
        normalize_hobbies=True,
        alias_map={},
        hobby_taxonomy={
            "rules": [
                {
                    "canonical_hobby": "영화/드라마 감상",
                    "include_keywords": ["영화", "드라마", "넷플릭스", "왓챠"],
                    "exclude_keywords": ["유튜브", "교양"],
                    "taxonomy": {},
                },
                {
                    "canonical_hobby": "유튜브/온라인 영상 시청",
                    "include_keywords": ["유튜브", "콘텐츠 시청"],
                    "exclude_keywords": ["영화", "드라마"],
                    "taxonomy": {},
                },
            ],
            "manual_aliases": {},
            "taxonomy": {},
            "display_examples": {},
        },
        min_item_degree=1,
        rare_item_policy="drop",
    )

    assert prepared.edges == [HobbyEdge("p1", "유튜브/온라인 영상 시청")]


def test_prepare_hobby_edges_deduplicates_after_aliasing() -> None:
    edges = [HobbyEdge("p1", "등산하기"), HobbyEdge("p1", "등산")]

    prepared = prepare_hobby_edges(
        edges,
        normalize_hobbies=True,
        alias_map={"등산하기": "등산"},
        min_item_degree=1,
        rare_item_policy="drop",
    )

    assert prepared.edges == [HobbyEdge("p1", "등산")]
    assert prepared.report["canonical_edges"] == 1


def test_prepare_hobby_edges_rejects_non_positive_min_degree() -> None:
    with pytest.raises(ValueError, match="min_item_degree"):
        prepare_hobby_edges(
            [HobbyEdge("p1", "등산")],
            normalize_hobbies=True,
            alias_map={},
            min_item_degree=0,
            rare_item_policy="drop",
        )


def test_split_edges_by_person_keeps_holdout_out_of_train() -> None:
    indexed = index_edges(
        [
            HobbyEdge("p1", "h1"),
            HobbyEdge("p1", "h2"),
            HobbyEdge("p1", "h3"),
            HobbyEdge("p1", "h4"),
        ]
    )

    split = split_edges_by_person(indexed.edges, 0.25, 0.25, 3, "train_only", seed=7)

    assert set(split.train).isdisjoint(split.validation)
    assert set(split.train).isdisjoint(split.test)
    assert set(split.validation).isdisjoint(split.test)


def test_sample_negative_never_samples_full_known_positive() -> None:
    full_known = {1: {0, 1, 2}}
    rng = random.Random(42)

    samples = {sample_negative(1, 5, full_known, rng) for _ in range(20)}

    assert samples <= {3, 4}


def test_build_hobby_profile_uses_train_split_only() -> None:
    profile = build_hobby_profile(
        train_edges=[(1, 10)],
        person_to_id={"p1": 1},
        hobby_to_id={"등산": 10, "요가": 11},
        contexts=None,
    )
    assert isinstance(profile, dict)

    assert profile["source"] == "train_split_only"
    assert profile["num_train_edges"] == 1
    hobbies = profile["hobbies"]
    assert isinstance(hobbies, dict)
    assert set(hobbies) == {"등산"}


def test_build_leakage_audit_detects_holdout_hobby_in_text() -> None:
    split = EdgeSplit(
        train=[(1, 10)],
        validation=[(1, 11)],
        test=[],
        full_known={1: {10, 11}},
        train_known={1: {10}},
    )
    context = PersonContext(
        person_uuid="p1",
        age="30",
        age_group="30대",
        sex="여자",
        occupation="개발자",
        district="강남구",
        province="서울특별시",
        family_type="1인 가구",
        housing_type="아파트",
        education_level="학사",
        persona_text="요가를 즐기는 사람",
        professional_text="",
        sports_text="",
        arts_text="",
        travel_text="",
        culinary_text="",
        family_text="",
        hobbies_text="요가와 산책",
        skills_text="",
        career_goals="",
        embedding_text="요가",
    )

    audit = build_leakage_audit(
        split,
        person_to_id={"p1": 1},
        hobby_to_id={"등산": 10, "요가": 11},
        contexts={"p1": context},
    )
    assert isinstance(audit, dict)
    validation = audit["validation"]
    assert isinstance(validation, dict)

    assert audit["status"] == "completed"
    assert validation["leaked_edges"] == 1
    assert validation["leakage_rate"] == 1.0
    field_mentions = validation["field_mentions"]
    assert isinstance(field_mentions, dict)
    assert field_mentions["hobbies_text"] == 1
