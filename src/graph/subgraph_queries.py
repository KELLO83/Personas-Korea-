SUBGRAPH_DEPTH1_QUERY = """
MATCH (p:Person {uuid: $uuid})-[r]-(n)
WHERE NOT type(r) = 'SIMILAR_TO' OR $include_similar = true
RETURN p, r, n, type(r) AS rel_type, labels(n) AS node_labels,
       p.uuid AS center_uuid, p.display_name AS center_label,
       n.uuid AS n_uuid, n.name AS n_name, n.display_name AS n_display_name,
       n.age AS n_age, n.sex AS n_sex, n.persona AS n_persona,
       n.key AS n_key, n.province AS n_province
"""

SUBGRAPH_DEPTH2_QUERY = """
MATCH (p:Person {uuid: $uuid})-[r1]-(entity)
WHERE (entity:Hobby OR entity:Skill OR entity:District)
WITH DISTINCT p, entity
CALL {
    WITH p, entity
    MATCH (entity)-[r2]-(other:Person)
    WHERE other.uuid <> p.uuid
    WITH DISTINCT other, type(r2) AS rel2_type
    ORDER BY coalesce(other.display_name, other.uuid)
    LIMIT $max_per_entity
    RETURN collect({other: other, rel2_type: rel2_type}) AS capped_people
}
UNWIND capped_people AS capped_person
WITH entity, capped_person.other AS other, capped_person.rel2_type AS rel2_type
RETURN DISTINCT entity, other, rel2_type,
       labels(entity) AS entity_labels, entity.name AS entity_name, entity.key AS entity_key,
       other.uuid AS other_uuid, other.display_name AS other_display_name,
       other.age AS other_age, other.sex AS other_sex
ORDER BY CASE
           WHEN 'Hobby' IN labels(entity) THEN 0
           WHEN 'Skill' IN labels(entity) THEN 1
           WHEN 'District' IN labels(entity) THEN 2
           ELSE 3
         END,
         coalesce(entity.name, entity.key),
         other.uuid
LIMIT $max_secondary
"""

SUBGRAPH_DEPTH3_QUERY = """
MATCH (p:Person {uuid: $uuid})-[r1]-(entity)-[r2]-(other:Person)-[r3]-(next_entity)
WHERE (entity:Hobby OR entity:Skill OR entity:District)
  AND (next_entity:Hobby OR next_entity:Skill OR next_entity:District OR next_entity:Occupation)
  AND other.uuid <> $uuid
  AND NOT next_entity = entity
RETURN other.uuid AS other_uuid, other.display_name AS other_display_name,
       labels(next_entity) AS next_entity_labels,
       next_entity.name AS next_entity_name,
       next_entity.key AS next_entity_key,
       type(r3) AS rel3_type
LIMIT $max_tertiary
"""

PERSON_EXISTS_QUERY = """
MATCH (p:Person {uuid: $uuid}) RETURN p.uuid AS uuid LIMIT 1
"""
