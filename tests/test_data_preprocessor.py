import pandas as pd
import pytest

from src.data.preprocessor import preprocess


def _sample_row() -> dict[str, object]:
    return {
        "uuid": "73f75d42a3934626b0d9a4bff062715a",
        "professional_persona": "최은지 씨는 회계 업무를 잘합니다.",
        "sports_persona": "산책을 즐깁니다.",
        "arts_persona": "서예를 배웁니다.",
        "travel_persona": "역사 유적지를 좋아합니다.",
        "culinary_persona": "청국장을 즐깁니다.",
        "family_persona": "가족을 아낍니다.",
        "persona": "최은지 씨는 서초구의 회계 사무원입니다.",
        "cultural_background": "서울 서초구에서 오래 지냈습니다.",
        "skills_and_expertise": "부동산 세금 계산에 능숙합니다.",
        "skills_and_expertise_list": "['부동산 취득세 산출', '복식부기 장부 작성']",
        "hobbies_and_interests": "고궁 산책을 즐깁니다.",
        "hobbies_and_interests_list": "['고궁 산책', '트로트 시청']",
        "career_goals_and_ambitions": "안정적인 직장 생활을 원합니다.",
        "sex": "여자",
        "age": 71,
        "marital_status": "배우자있음",
        "military_status": "비현역",
        "family_type": "배우자·자녀와 거주",
        "housing_type": "다세대주택",
        "education_level": "4년제 대학교",
        "bachelors_field": "해당없음",
        "occupation": "회계 사무원",
        "district": "서울-서초구",
        "province": "서울",
        "country": "대한민국",
    }


def test_preprocess_derives_graph_ready_columns() -> None:
    df = pd.DataFrame([_sample_row()])

    result = preprocess(df)

    row = result.iloc[0]
    assert row["skills_and_expertise_list"] == ["부동산 취득세 산출", "복식부기 장부 작성"]
    assert row["hobbies_and_interests_list"] == ["고궁 산책", "트로트 시청"]
    assert row["province_cleaned"] == "서울"
    assert row["district_cleaned"] == "서초구"
    assert row["age_group"] == "70대"
    assert row["bachelors_field"] is None
    assert row["display_name"] == "최은지"
    assert "최은지" in row["embedding_text"]
    assert "서초구" in row["embedding_text"]
    assert "회계 사무원" in row["embedding_text"]
    assert "부동산 취득세 산출" in row["embedding_text"]
    assert "고궁 산책" in row["embedding_text"]


def test_preprocess_rejects_duplicate_uuid() -> None:
    row = _sample_row()
    df = pd.DataFrame([row, row])

    with pytest.raises(ValueError, match="UUID values must be unique"):
        preprocess(df)
