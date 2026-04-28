import pandas as pd

from src.graph.loader import _to_graph_rows
from src.graph.schema import schema_queries


def test_schema_queries_include_person_uuid_constraint() -> None:
    queries = schema_queries()

    assert any("person_uuid_unique" in query for query in queries)
    assert any("person_age" in query for query in queries)


def test_to_graph_rows_builds_person_and_relationship_payload() -> None:
    df = pd.DataFrame(
        [
            {
                "uuid": "73f75d42a3934626b0d9a4bff062715a",
                "display_name": "최은지",
                "age": 71,
                "age_group": "70대",
                "sex": "여자",
                "persona": "최은지 씨는 회계 사무원입니다.",
                "professional_persona": "회계 업무를 잘합니다.",
                "sports_persona": "산책합니다.",
                "arts_persona": "서예를 합니다.",
                "travel_persona": "역사 여행을 좋아합니다.",
                "culinary_persona": "청국장을 즐깁니다.",
                "family_persona": "가족을 아낍니다.",
                "cultural_background": "서울에서 살았습니다.",
                "skills_and_expertise": "부동산 세금 계산",
                "hobbies_and_interests": "고궁 산책",
                "career_goals_and_ambitions": "안정적인 근무",
                "embedding_text": "통합 텍스트",
                "province_cleaned": "서울",
                "district_cleaned": "서초구",
                "country": "대한민국",
                "occupation": "회계 사무원",
                "education_level": "4년제 대학교",
                "bachelors_field": None,
                "marital_status": "배우자있음",
                "military_status": "비현역",
                "family_type": "배우자·자녀와 거주",
                "housing_type": "다세대주택",
                "skills_and_expertise_list": ["부동산 취득세 산출"],
                "hobbies_and_interests_list": ["고궁 산책"],
            }
        ]
    )

    row = _to_graph_rows(df)[0]

    assert row["uuid"] == "73f75d42a3934626b0d9a4bff062715a"
    assert row["person_properties"]["display_name"] == "최은지"
    assert row["person_properties"]["embedding_text"] == "통합 텍스트"
    assert row["province"] == "서울"
    assert row["district_name"] == "서초구"
    assert row["district_key"] == "서울-서초구"
    assert row["skills"] == ["부동산 취득세 산출"]
    assert row["hobbies"] == ["고궁 산책"]
    assert row["bachelors_field"] is None
