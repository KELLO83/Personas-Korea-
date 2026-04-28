TOTAL_COUNT_QUERY = """
MATCH (p:Person) RETURN count(p) AS total
"""

AGE_DISTRIBUTION_QUERY = """
MATCH (p:Person)
WHERE p.age_group IS NOT NULL
RETURN p.age_group AS age_group, count(p) AS count
ORDER BY count DESC
"""

SEX_DISTRIBUTION_QUERY = """
MATCH (p:Person)
WHERE p.sex IS NOT NULL
RETURN p.sex AS sex, count(p) AS count
ORDER BY count DESC
"""

PROVINCE_DISTRIBUTION_QUERY = """
MATCH (p:Person)-[:LIVES_IN]->(:District)-[:IN_PROVINCE]->(prov:Province)
RETURN prov.name AS province, count(p) AS count
ORDER BY count DESC
"""

EDUCATION_DISTRIBUTION_QUERY = """
MATCH (p:Person)-[:EDUCATED_AT]->(edu:EducationLevel)
RETURN edu.name AS education_level, count(p) AS count
ORDER BY count DESC
"""

MARITAL_DISTRIBUTION_QUERY = """
MATCH (p:Person)-[:MARITAL_STATUS]->(ms:MaritalStatus)
RETURN ms.name AS marital_status, count(p) AS count
ORDER BY count DESC
"""

TOP_OCCUPATIONS_QUERY = """
MATCH (p:Person)-[:WORKS_AS]->(occ:Occupation)
RETURN occ.name AS occupation, count(p) AS count
ORDER BY count DESC LIMIT $limit
"""

TOP_HOBBIES_QUERY = """
MATCH (p:Person)-[:ENJOYS_HOBBY]->(h:Hobby)
RETURN h.name AS hobby, count(p) AS count
ORDER BY count DESC LIMIT $limit
"""

TOP_SKILLS_QUERY = """
MATCH (p:Person)-[:HAS_SKILL]->(s:Skill)
RETURN s.name AS skill, count(p) AS count
ORDER BY count DESC LIMIT $limit
"""

VALID_DIMENSIONS = {
    "age",
    "sex",
    "province",
    "district",
    "occupation",
    "hobby",
    "skill",
    "education",
    "marital",
    "military",
    "family_type",
    "housing",
}

_DIMENSION_CONFIG: dict[str, dict[str, str | None]] = {
    "age": {
        "match": None,
        "return_field": "p.age_group",
        "alias": "label",
    },
    "sex": {
        "match": None,
        "return_field": "p.sex",
        "alias": "label",
    },
    "province": {
        "match": "(p)-[:LIVES_IN]->(:District)-[:IN_PROVINCE]->(target:Province)",
        "return_field": "target.name",
        "alias": "label",
    },
    "district": {
        "match": "(p)-[:LIVES_IN]->(target:District)",
        "return_field": "target.name",
        "alias": "label",
    },
    "occupation": {
        "match": "(p)-[:WORKS_AS]->(target:Occupation)",
        "return_field": "target.name",
        "alias": "label",
    },
    "hobby": {
        "match": "(p)-[:ENJOYS_HOBBY]->(target:Hobby)",
        "return_field": "target.name",
        "alias": "label",
    },
    "skill": {
        "match": "(p)-[:HAS_SKILL]->(target:Skill)",
        "return_field": "target.name",
        "alias": "label",
    },
    "education": {
        "match": "(p)-[:EDUCATED_AT]->(target:EducationLevel)",
        "return_field": "target.name",
        "alias": "label",
    },
    "marital": {
        "match": "(p)-[:MARITAL_STATUS]->(target:MaritalStatus)",
        "return_field": "target.name",
        "alias": "label",
    },
    "military": {
        "match": "(p)-[:MILITARY_STATUS]->(target:MilitaryStatus)",
        "return_field": "target.name",
        "alias": "label",
    },
    "family_type": {
        "match": "(p)-[:LIVES_WITH]->(target:FamilyType)",
        "return_field": "target.name",
        "alias": "label",
    },
    "housing": {
        "match": "(p)-[:LIVES_IN_HOUSING]->(target:HousingType)",
        "return_field": "target.name",
        "alias": "label",
    },
}


def build_dimension_query(
    dimension: str,
    province: str | None = None,
    age_group: str | None = None,
    sex: str | None = None,
    occupation: str | None = None,
    keyword: str | None = None,
) -> tuple[str, dict[str, object]]:
    config = _DIMENSION_CONFIG[dimension]
    params: dict[str, object] = {}

    parts: list[str] = ["MATCH (p:Person)"]

    if config["match"] is not None:
        parts.append(f"MATCH {config['match']}")

    if province is not None:
        parts.append(
            "MATCH (p)-[:LIVES_IN]->(:District)-[:IN_PROVINCE]->(filt_prov:Province)"
        )

    if occupation is not None:
        parts.append("MATCH (p)-[:WORKS_AS]->(filt_occ:Occupation)")

    where_clauses: list[str] = []

    if province is not None:
        where_clauses.append("filt_prov.name = $province")
        params["province"] = province

    if age_group is not None:
        where_clauses.append("p.age_group = $age_group")
        params["age_group"] = age_group

    if sex is not None:
        where_clauses.append("p.sex = $sex")
        params["sex"] = sex

    if occupation is not None:
        where_clauses.append("filt_occ.name CONTAINS $occupation")
        params["occupation"] = occupation

    if keyword is not None:
        where_clauses.append("p.persona CONTAINS $keyword")
        params["keyword"] = keyword

    return_field = config["return_field"]

    if dimension in ("age", "sex") and return_field is not None:
        where_clauses.append(f"{return_field} IS NOT NULL")

    if where_clauses:
        parts.append("WHERE " + " AND ".join(where_clauses))

    parts.append(f"RETURN {return_field} AS label, count(p) AS count")
    parts.append("ORDER BY count DESC LIMIT $limit")
    params["limit"] = 20

    query = "\n".join(parts)
    return query, params
