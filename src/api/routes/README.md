# API Routes

Route files are grouped by API surface. Keep business logic in `src/` services
and use route handlers for request parsing, response shaping, and dependency
wiring.

- Persona lookup and search: `persona.py`, `search.py`, `target_persona.py`.
- Recommendations and similarity: `recommend.py`, `similar.py`, `compare.py`.
- Graph exploration: `graph_viz.py`, `path.py`, `communities.py`, `influence.py`.
- Insights and quality: `insight.py`, `graph_insights.py`, `graph_quality.py`,
  `stats.py`, `lifestyle_map.py`, `career_transition.py`.
- Operations and observability: `operations.py`, `rag_traces.py`, `chat.py`.
