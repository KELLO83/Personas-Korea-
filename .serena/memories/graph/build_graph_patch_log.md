# Build Graph Patch Log - 2026-05-03

## Issue
The `build_graph.py` script was querying Person nodes for `hobbies_text` property which did not exist on Person nodes, causing 0 Person-Hobby relationships to be created.

## Root Cause
- Person nodes have `hobbies_and_interests` property (from PERSON_PROPERTY_FIELDS) containing hobby text
- The `_batch_create_hobby_relationships` function in `scripts/build_graph.py` was querying for `p.hobbies_text` 
- Since this property didn't exist, the query returned 0 results

## Fix Applied
Changed the query in `scripts/build_graph.py` (lines 131-133):

**Before:**
```cypher
MATCH (p:Person)
WHERE p.hobbies_text IS NOT NULL AND trim(p.hobbies_text) <> ""
RETURN p.uuid AS uuid, p.hobbies_text AS hobbies
```

**After:**
```cypher
MATCH (p:Person)
WHERE p.hobbies_and_interests IS NOT NULL AND trim(p.hobbies_and_interests) <> ""
RETURN p.uuid AS uuid, p.hobbies_and_interests AS hobbies
```

## Verification Results
- ✅ Ran: `.venv\Scripts\python.exe scripts\build_graph.py --reset --target-persons 10000 --workers 2`
- ✅ Generated: 10,000 Person nodes
- ✅ Created: 27,895 Person-Hobby (LIKES) relationships
- ✅ Exported: 27,895 edges to `GNN_Neural_Network/data/person_hobby_edges.csv`
- ✅ Exported: 10,000 person contexts to `GNN_Neural_Network/data/person_context.csv`

## Files Modified
- `scripts/build_graph.py` - Line 132-133: Changed property name from `hobbies_text` to `hobbies_and_interests`

## Notes
- The `_create_hobby_relationships` function itself was already correctly patched
- During initial load, `CREATE_PERSON_GRAPH_QUERY` already creates `[:ENJOYS_HOBBY]` relationships from the `hobbies` list
- The batch process creates additional `[:LIKES]` relationships by reparsing the `hobbies_and_interests` text
- Both relationship types coexist in the graph