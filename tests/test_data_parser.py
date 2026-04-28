from src.data.parser import parse_age_group, parse_district, parse_list_field


def test_parse_list_field_handles_python_list_string() -> None:
    result = parse_list_field("['가계부 정밀 기록', ' 서류 보관 ', '']")

    assert result == ["가계부 정밀 기록", "서류 보관"]


def test_parse_list_field_returns_empty_list_for_bad_input() -> None:
    assert parse_list_field("not a list") == []
    assert parse_list_field("") == []
    assert parse_list_field(None) == []


def test_parse_district_splits_province_and_district() -> None:
    assert parse_district("서울-서초구") == ("서울", "서초구")


def test_parse_district_handles_missing_separator() -> None:
    assert parse_district("서초구") == ("", "서초구")


def test_parse_age_group_returns_decade_label() -> None:
    assert parse_age_group(31) == "30대"
    assert parse_age_group("74") == "70대"
    assert parse_age_group(None) == ""
