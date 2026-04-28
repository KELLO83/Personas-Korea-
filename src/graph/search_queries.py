import re


VALID_SORT_FIELDS = {"age": "p.age", "display_name": "p.display_name"}
VALID_SORT_ORDERS = {"asc", "desc"}


def _compact_occupation_term(term: str) -> str:
    lowered = term.lower().strip()
    return re.sub(r"[\s\-_/.]", "", lowered)


def build_search_query(
    province: list[str] | None = None,
    district: list[str] | None = None,
    age_min: int | None = None,
    age_max: int | None = None,
    age_group: list[str] | None = None,
    sex: str | None = None,
    occupation: str | list[str] | None = None,
    education_level: list[str] | None = None,
    hobby: list[str] | None = None,
    skill: list[str] | None = None,
    keyword: str | None = None,
    semantic_persona_uuids: list[str] | None = None,
    sort_by: str = "age",
    sort_order: str = "asc",
    page: int = 1,
    page_size: int = 20,
) -> tuple[str, str, dict[str, object]]:
    required_matches: list[str] = ["MATCH (p:Person)"]
    where_clauses: list[str] = []
    params: dict[str, object] = {}

    need_location_filter = province is not None or district is not None
    need_occupation_filter = occupation is not None
    need_education_filter = education_level is not None
    need_hobby_filter = hobby is not None
    need_skill_filter = skill is not None

    if need_location_filter:
        required_matches.append(
            "MATCH (p)-[:LIVES_IN]->(d:District)-[:IN_PROVINCE]->(prov:Province)"
        )
    if need_hobby_filter:
        required_matches.append("MATCH (p)-[:ENJOYS_HOBBY]->(h:Hobby)")
    if need_skill_filter:
        required_matches.append("MATCH (p)-[:HAS_SKILL]->(s:Skill)")
    if need_occupation_filter:
        required_matches.append("MATCH (p)-[:WORKS_AS]->(occ:Occupation)")
    if need_education_filter:
        required_matches.append("MATCH (p)-[:EDUCATED_AT]->(edu:EducationLevel)")

    if province is not None:
        where_clauses.append("prov.name IN $provinces")
        params["provinces"] = province
    if district is not None:
        where_clauses.append("(d.key IN $districts OR d.name IN $districts)")
        params["districts"] = district
    if age_min is not None:
        where_clauses.append("p.age >= $age_min")
        params["age_min"] = age_min
    if age_max is not None:
        where_clauses.append("p.age <= $age_max")
        params["age_max"] = age_max
    if age_group is not None:
        where_clauses.append("p.age_group IN $age_groups")
        params["age_groups"] = age_group
    if sex is not None:
        where_clauses.append("p.sex = $sex")
        params["sex"] = sex
    if occupation is not None:
        if isinstance(occupation, list):
            occupation_terms = [term for term in occupation if term]
        else:
            occupation_terms = [occupation]

        occupation_clauses = []
        for idx, term in enumerate(occupation_terms):
            compact_key = f"occupation_{idx}_compact"
            raw_key = f"occupation_{idx}_raw"
            compact_term = _compact_occupation_term(str(term))

            clause = (
                f"(toLower(occ.name) CONTAINS ${raw_key} OR "
                f"toLower(replace(replace(replace(replace(occ.name, ' ', ''), '-', ''), '_', ''), '/', '')) CONTAINS ${compact_key})"
            )
            occupation_clauses.append(clause)

            params[raw_key] = str(term)
            params[compact_key] = compact_term

        where_clauses.append("(" + " OR ".join(occupation_clauses) + ")")
    if education_level is not None:
        where_clauses.append("edu.name IN $education_levels")
        params["education_levels"] = education_level
    if hobby is not None:
        where_clauses.append("h.name IN $hobbies")
        params["hobbies"] = hobby
    if skill is not None:
        where_clauses.append("s.name IN $skills")
        params["skills"] = skill
    if semantic_persona_uuids is not None:
        where_clauses.append("p.uuid IN $persona_uuids")
        params["persona_uuids"] = semantic_persona_uuids
    if keyword is not None:
        where_clauses.append("p.persona CONTAINS $keyword")
        params["keyword"] = keyword

    where_str = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    count_parts = required_matches.copy()
    if where_str:
        count_parts.append(where_str)
    count_parts.append("RETURN count(DISTINCT p) AS total_count")
    count_query = "\n".join(count_parts)

    data_parts = required_matches.copy()
    if where_str:
        data_parts.append(where_str)

    if not need_location_filter:
        data_parts.append(
            "OPTIONAL MATCH (p)-[:LIVES_IN]->(d:District)-[:IN_PROVINCE]->(prov:Province)"
        )
    if not need_occupation_filter:
        data_parts.append("OPTIONAL MATCH (p)-[:WORKS_AS]->(occ:Occupation)")
    if not need_education_filter:
        data_parts.append("OPTIONAL MATCH (p)-[:EDUCATED_AT]->(edu:EducationLevel)")

    data_parts.append(
        "RETURN DISTINCT p.uuid AS uuid, p.display_name AS display_name, "
        "p.age AS age, p.sex AS sex, p.persona AS persona, "
        "prov.name AS province, d.name AS district, "
        "occ.name AS occupation, edu.name AS education_level"
    )

    sort_field = VALID_SORT_FIELDS.get(sort_by, "p.age")
    order = sort_order.upper() if sort_order.lower() in VALID_SORT_ORDERS else "ASC"
    skip = (page - 1) * page_size

    data_parts.append(f"ORDER BY {sort_field} {order}")
    data_parts.append("SKIP $skip LIMIT $page_size")
    params["skip"] = skip
    params["page_size"] = page_size

    data_query = "\n".join(data_parts)

    return data_query, count_query, params
