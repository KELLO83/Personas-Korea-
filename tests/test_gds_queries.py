from src.gds.communities import COMMUNITY_SUMMARY_QUERY, LEIDEN_WRITE_QUERY, _build_top_traits
from src.gds.fastrp import FASTRP_WRITE_QUERY, PROJECT_GRAPH_QUERY, PROJECT_GRAPH_WITH_FASTRP_QUERY
from src.gds.similarity import KNN_WRITE_QUERY, QUERY_PERSONA_QUERY, SHARED_TRAITS_QUERY, SIMILAR_PERSONAS_QUERY


def test_fastrp_queries_target_persona_graph() -> None:
    assert "gds.graph.project" in PROJECT_GRAPH_QUERY
    assert "gds.fastRP.write" in FASTRP_WRITE_QUERY
    assert "fastrp_embedding" in FASTRP_WRITE_QUERY
    assert "nodePropertiesWritten" in FASTRP_WRITE_QUERY
    assert "ranIterations" not in FASTRP_WRITE_QUERY
    assert "fastrp_embedding" in PROJECT_GRAPH_WITH_FASTRP_QUERY


def test_similarity_queries_use_person_nodes() -> None:
    assert "gds.knn.write" in KNN_WRITE_QUERY
    assert "nodeLabels: ['Person']" in KNN_WRITE_QUERY
    assert "SIMILAR_TO" in SIMILAR_PERSONAS_QUERY
    assert "shared.name" in SHARED_TRAITS_QUERY
    assert "district.name AS district_name" in QUERY_PERSONA_QUERY


def test_community_queries_write_and_summarize_leiden() -> None:
    assert "gds.leiden.write" in LEIDEN_WRITE_QUERY
    assert "community_id" in LEIDEN_WRITE_QUERY
    assert "top_hobbies" in COMMUNITY_SUMMARY_QUERY
    assert "top_provinces" in COMMUNITY_SUMMARY_QUERY


def test_build_top_traits_uses_primary_province_and_hobbies() -> None:
    traits = _build_top_traits(["등산", "산책"], ["서울", "경기"])

    assert traits["province"] == "서울"
    assert traits["hobbies"] == ["등산", "산책"]
