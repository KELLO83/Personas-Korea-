CREATE_PERSON_GRAPH_QUERY = """
UNWIND $rows AS row
MERGE (p:Person {uuid: row.uuid})
SET p += row.person_properties

WITH p, row
MERGE (country:Country {name: row.country})
MERGE (province:Province {name: row.province})
MERGE (district:District {key: row.district_key})
SET district.name = row.district_name,
    district.province = row.province
MERGE (province)-[:IN_COUNTRY]->(country)
MERGE (district)-[:IN_PROVINCE]->(province)
MERGE (p)-[:LIVES_IN]->(district)

WITH p, row
FOREACH (_ IN CASE WHEN row.occupation IS NULL OR row.occupation = '' THEN [] ELSE [1] END |
    MERGE (occupation:Occupation {name: row.occupation})
    MERGE (p)-[:WORKS_AS]->(occupation)
)
FOREACH (_ IN CASE WHEN row.education_level IS NULL OR row.education_level = '' THEN [] ELSE [1] END |
    MERGE (education:EducationLevel {name: row.education_level})
    MERGE (p)-[:EDUCATED_AT]->(education)
)
FOREACH (_ IN CASE WHEN row.bachelors_field IS NULL OR row.bachelors_field = '' THEN [] ELSE [1] END |
    MERGE (field:Field {name: row.bachelors_field})
    MERGE (p)-[:MAJORED_IN]->(field)
)
FOREACH (_ IN CASE WHEN row.marital_status IS NULL OR row.marital_status = '' THEN [] ELSE [1] END |
    MERGE (marital:MaritalStatus {name: row.marital_status})
    MERGE (p)-[:MARITAL_STATUS]->(marital)
)
FOREACH (_ IN CASE WHEN row.military_status IS NULL OR row.military_status = '' THEN [] ELSE [1] END |
    MERGE (military:MilitaryStatus {name: row.military_status})
    MERGE (p)-[:MILITARY_STATUS]->(military)
)
FOREACH (_ IN CASE WHEN row.family_type IS NULL OR row.family_type = '' THEN [] ELSE [1] END |
    MERGE (family:FamilyType {name: row.family_type})
    MERGE (p)-[:LIVES_WITH]->(family)
)
FOREACH (_ IN CASE WHEN row.housing_type IS NULL OR row.housing_type = '' THEN [] ELSE [1] END |
    MERGE (housing:HousingType {name: row.housing_type})
    MERGE (p)-[:LIVES_IN_HOUSING]->(housing)
)

WITH p, row
FOREACH (skill_name IN row.skills |
    MERGE (skill:Skill {name: skill_name})
    MERGE (p)-[:HAS_SKILL]->(skill)
)
FOREACH (hobby_name IN row.hobbies |
    MERGE (hobby:Hobby {name: hobby_name})
    MERGE (p)-[:ENJOYS_HOBBY]->(hobby)
)
"""

COUNT_GRAPH_QUERY = """
MATCH (n)
RETURN labels(n)[0] AS label, count(n) AS count
ORDER BY label
"""
