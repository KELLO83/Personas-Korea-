PROFILE_QUERY = """
MATCH (p:Person {uuid: $uuid})
OPTIONAL MATCH (p)-[:LIVES_IN]->(d:District)-[:IN_PROVINCE]->(prov:Province)-[:IN_COUNTRY]->(c:Country)
OPTIONAL MATCH (p)-[:WORKS_AS]->(occ:Occupation)
OPTIONAL MATCH (p)-[:EDUCATED_AT]->(edu:EducationLevel)
OPTIONAL MATCH (p)-[:MAJORED_IN]->(field:Field)
OPTIONAL MATCH (p)-[:MARITAL_STATUS]->(ms:MaritalStatus)
OPTIONAL MATCH (p)-[:MILITARY_STATUS]->(mil:MilitaryStatus)
OPTIONAL MATCH (p)-[:LIVES_WITH]->(ft:FamilyType)
OPTIONAL MATCH (p)-[:LIVES_IN_HOUSING]->(ht:HousingType)
OPTIONAL MATCH (p)-[:HAS_SKILL]->(s:Skill)
OPTIONAL MATCH (p)-[:ENJOYS_HOBBY]->(h:Hobby)
RETURN p,
       d.name AS district_name, d.key AS district_key,
       prov.name AS province_name,
       c.name AS country_name,
       occ.name AS occupation_name,
       edu.name AS education_level,
       field.name AS bachelors_field,
       ms.name AS marital_status,
       mil.name AS military_status,
       ft.name AS family_type,
       ht.name AS housing_type,
       collect(DISTINCT s.name) AS skills,
       collect(DISTINCT h.name) AS hobbies
"""

SIMILAR_PREVIEW_QUERY = """
MATCH (p:Person {uuid: $uuid})-[r:SIMILAR_TO]->(sim:Person)
OPTIONAL MATCH (p)-[:ENJOYS_HOBBY]->(h1:Hobby)<-[:ENJOYS_HOBBY]-(sim)
RETURN sim.uuid AS uuid, sim.display_name AS display_name, sim.age AS age,
       r.score AS similarity, collect(DISTINCT h1.name) AS shared_hobbies
ORDER BY r.score DESC LIMIT 3
"""

GRAPH_STATS_QUERY = """
MATCH (p:Person {uuid: $uuid})
OPTIONAL MATCH (p)-[r]-()
OPTIONAL MATCH (p)-[:ENJOYS_HOBBY]->(h:Hobby)
OPTIONAL MATCH (p)-[:HAS_SKILL]->(s:Skill)
RETURN count(DISTINCT r) AS total_connections,
       count(DISTINCT h) AS hobby_count,
       count(DISTINCT s) AS skill_count
"""
