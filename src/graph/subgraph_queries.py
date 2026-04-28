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
MATCH (p:Person {uuid: $uuid})-[r1]-(entity)-[r2]-(other:Person)
WHERE (entity:Hobby OR entity:Skill OR entity:District)
  AND other.uuid <> $uuid
RETURN entity, other, type(r1) AS rel1_type, type(r2) AS rel2_type,
       labels(entity) AS entity_labels, entity.name AS entity_name,
       other.uuid AS other_uuid, other.display_name AS other_display_name,
       other.age AS other_age, other.sex AS other_sex
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
