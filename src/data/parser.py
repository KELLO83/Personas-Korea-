import ast

MAX_LIST_FIELD_LENGTH = 10_000


def parse_list_field(text: str) -> list[str]:
    if not isinstance(text, str):
        return []

    normalized_text = text.strip()
    if not normalized_text or len(normalized_text) > MAX_LIST_FIELD_LENGTH:
        return []

    try:
        parsed = ast.literal_eval(normalized_text)
    except (SyntaxError, ValueError):
        return []

    if not isinstance(parsed, (list, tuple)):
        return []

    return [str(item).strip() for item in parsed if str(item).strip()]


def parse_district(district: str) -> tuple[str, str]:
    if not isinstance(district, str):
        return "", ""

    normalized_district = district.strip()
    if not normalized_district:
        return "", ""

    province, separator, district_name = normalized_district.partition("-")
    if not separator:
        return "", normalized_district

    if not province.strip() and not district_name.strip():
        return "", ""
    if not province.strip():
        return "", district_name.strip()
    if not district_name.strip():
        return province.strip(), ""

    return province.strip(), district_name.strip()


def parse_age_group(age: int) -> str:
    if isinstance(age, bool):
        return ""

    try:
        age_text = str(age).strip()
        if not age_text.isdigit():
            return ""
        age_value = int(age_text)
    except (TypeError, ValueError):
        return ""

    if age_value < 0:
        return ""

    return f"{(age_value // 10) * 10}대"
