# React Frontend Migration Plan

> Implementation plan and checklist for the migration from the Streamlit-based Python frontend to the React frontend.  
> Current principle: **The production frontend is React/Next.js, and the previous Streamlit code has already been removed.**

---

## 1. Goals

- Gradually migrate the features used in the previous Streamlit UI to a React-based frontend.
- Keep the existing FastAPI backend (`src/api`) and Neo4j/RAG/Graph logic.
- Use the FastAPI API contract and test results as the baseline until React reaches core feature parity.
- There is no longer a Streamlit rollback path; when needed, verify regressions against git history and API tests.

---

## 2. Current Status Summary

### 2.1 Current Frontend

- Production frontend: Next.js React based in `frontend/`
- Baseline: FastAPI API contract, tests, and existing feature requirements documents
- Note: The goal was to reconstruct the tab-based workflow previously provided by the Streamlit UI as React screens.

### 2.2 Previous Streamlit Status

- The previous Streamlit frontend code has been removed from the current repository.
- Therefore, Streamlit references in this document should be read only as historical context describing the previous UI structure.

### 2.3 Existing API

FastAPI endpoints to prioritize for reuse during the React migration:

| Feature | Endpoint | React Screen |
|---|---|---|
| Statistics dashboard | `GET /api/stats` | Dashboard |
| Dimension-specific statistics | `GET /api/stats/{dimension}` | Dashboard Drilldown |
| Search/filter | `GET /api/search` | Search |
| Profile details | `GET /api/persona/{uuid}` | Profile |
| Segment comparison | `POST /api/compare/segments` | Compare |
| Subgraph | `GET /api/graph/subgraph/{uuid}` | Graph Explorer |
| Key people | `GET /api/influence/top` | Influence |
| Removal simulation | `POST /api/influence/simulate-removal` | Influence |
| Conversational exploration | `POST /api/chat` | Chat |
| Insight query | `POST /api/insight` | Insight |
| Similar personas | `POST /api/similar/{uuid}` | Similar |
| Recommendations | `GET /api/recommend/{uuid}` | Recommendation |
| Communities | `GET /api/communities` | Communities |
| Relationship path | `GET /api/path/{uuid1}/{uuid2}` | Path |

---

## 3. Migration Strategy

### 3.1 Basic Strategy

1. Create a new `frontend/` directory.
2. The React app calls FastAPI only through the HTTP API.
3. Build React screens one by one and compare them against API responses, feature requirements, and test results.
4. Switch the default frontend to React only after all core workflows pass.

### 3.2 Recommended Tech Stack

- React + TypeScript
- Next.js or Vite
  - If a structure similar to `toeic_whisper` is desired, prefer a Next.js-based `frontend/`
  - If a simple SPA and fast development are desired, a Vite-based setup is also possible
- Tailwind CSS
- React Query or SWR: API caching and loading/error state management
- Zustand or Context: global state management for `selected_uuid`, current tab/filter/chat session, and similar state
- Graph visualization candidates:
  - `vis-network`: closest to the graph presentation style of the previous UI
  - `React Flow`: better for UI controls and React component integration
  - `Cytoscape.js`: strong graph exploration and layout capabilities

### 3.3 Patterns to Bring from toeic_whisper

- Separate `frontend/` directory
- Separate ports for the Python FastAPI backend and React frontend
- Tailwind-based UI composition
- Separate sidebar/layout components

### 3.4 Patterns Not to Copy Directly from toeic_whisper

- Do not hardcode API URLs: use an environment-variable-based API client instead of directly using `http://localhost:8000`
- Do not turn `page.tsx` into one large file: separate screens, components, hooks, and the API client
- Do not manually guess API types: organize types based on FastAPI OpenAPI or `src/api/schemas.py`

---

## 4. Proposed Directory Structure

```text
frontend/
  app/ or src/
    routes/ or pages/
    components/
      layout/
      dashboard/
      search/
      profile/
      graph/
      chat/
      common/
    lib/
      api-client.ts
      api-types.ts
      constants.ts
      formatters.ts
    stores/
      persona-selection-store.ts
      chat-store.ts
    styles/
  package.json
  tsconfig.json
  .env.example
```

---

## 5. Phased Implementation Plan

## Phase R0: Baseline Freeze


- [x] Completed review of the previous Streamlit UI structure before the React migration
- [x] Documented the principle that the FastAPI API contract is used as the baseline during the React migration
- [ ] Confirm that the full current FastAPI test suite passes

## Phase R1: API Contract Cleanup


- [ ] Check `/openapi.json`
- [x] Create a TypeScript type list based on `src/api/schemas.py`
- [x] Define the common API error shape
- [x] Design the React API client
- [x] Design the `API_BASE_URL` environment variable
- [ ] Review the CORS origin policy

APIs to add if needed:

- [ ] `GET /api/health`
- [ ] `GET /api/options/provinces`
- [ ] `GET /api/options/districts?province=...`
- [ ] `GET /api/options/occupations?keyword=...`
- [ ] `GET /api/options/hobbies?keyword=...`
- [ ] `GET /api/options/skills?keyword=...`

## Phase R2: Build the React Scaffold


- [x] Create `frontend/`
- [x] Initialize the React + TypeScript project
- [x] Set up base styles
- [x] Build the base layout
- [x] Build sidebar/tab navigation
- [x] Connect the basic API client
- [x] Create common loading/error/empty components
- [x] Verify Korean UI text rendering

## Phase R3: Migrate Read-Only Screens First


### Dashboard

- [x] Connect `GET /api/stats`
- [x] Total persona count card
- [x] Age/gender/region distribution charts
- [x] Top hobbies/occupations/skills lists
- [ ] Compare values against existing requirements and backend responses

### Search

- [x] Connect `GET /api/search`
- [x] Search filter UI
- [x] Sorting/pagination
- [x] Search result cards
- [x] Connect "Select this person" state
- [x] Empty result/error states

### Profile

- [x] Connect `GET /api/persona/{uuid}`
- [x] Basic information card
- [x] Persona text display
- [x] Hobby/skill/occupation/region display
- [x] Similar persona preview
- [ ] Prepare to connect the recommendation section

## Phase R4: Migrate Analysis Screens


### Compare

- [ ] Connect `POST /api/compare/segments`
- [ ] Group A/B filter inputs
- [ ] Distribution comparison cards/tables
- [ ] AI analysis result display

### Similar

- [ ] Connect `POST /api/similar/{uuid}`
- [ ] `top_k` setting
- [ ] Similarity result cards
- [ ] Connect selected persona state

### Recommendation

- [ ] Connect `GET /api/recommend/{uuid}`
- [ ] Category selection
- [ ] Recommendation cards
- [ ] Handle 503 data-not-ready state

### Communities

- [ ] Connect `GET /api/communities`
- [ ] Community cards/summary
- [ ] Connect representative persona selection

### Path

- [ ] Connect `GET /api/path/{uuid1}/{uuid2}`
- [ ] Two UUID inputs
- [ ] Visualize/verbalize path results

## Phase R5: Migrate Graph Visualization


- [x] Connect `GET /api/graph/subgraph/{uuid}`
- [ ] Finalize graph library selection
- [x] Migrate color/size mappings by node type
- [x] Migrate label/color mappings by relationship type
- [ ] Node type filter UI
- [ ] Relationship summary table
- [ ] Relationship sentence generation
- [ ] Commonality cards
- [ ] Verify loading/performance for large graphs

Key functions to move from Streamlit to React:

- [ ] `NODE_TYPE_LABELS`
- [ ] `NODE_STYLES`
- [ ] `RELATION_LABELS`
- [ ] `filter_graph_by_types`
- [ ] `relationship_sentence_rows`
- [ ] `commonality_cards`
- [ ] `relation_context_label`

## Phase R6: Migrate Chat/Insight


### Chat

- [x] Connect `POST /api/chat`
- [x] Generate/store `session_id`
- [x] Display message history
- [x] Display context_filters
- [x] Display sources
- [x] Handle loading/error states
- [ ] Decide the recent conversation retention policy
- [ ] Decide whether streaming is needed

### Insight

- [ ] Connect `POST /api/insight`
- [ ] Question input/response display
- [ ] Display sources
- [ ] Handle long-running response state

## Phase R7: Parity Verification and Migration Preparation


- [ ] Compare React results by screen against existing requirements
- [ ] Confirm major API response values match
- [ ] Review Korean UI text
- [ ] Review loading/error/empty states
- [ ] Check mobile/tablet responsiveness
- [ ] Basic accessibility check
- [ ] React build succeeds
- [ ] Frontend lint/typecheck succeeds
- [ ] FastAPI tests succeed
- [ ] Update production run documentation
- [ ] Make the final decision on whether to switch React to the default frontend

---

## 6. State Management Design

Items to move from Streamlit `session_state` to React state:

| Streamlit Key | Candidate React Location | Description |
|---|---|---|
| `selected_uuid` | Zustand/URL param | Globally selected persona |
| `selected_persona_label` | Zustand | Current selected display name |
| `search_filters` | URL query + local state | Search conditions |
| `search_results` | React Query cache | Search results |
| `profile_uuid` | URL param/local state | Profile lookup target |
| `graph_uuid` | URL param/local state | Graph center node |
| `graph_data` | React Query cache | Subgraph data |
| `graph_profile` | React Query cache | Graph center profile |
| `chat_session_id` | localStorage/Zustand | Backend chat session key |
| `chat_messages` | Zustand/localStorage | Chat history |
| `insight_messages` | Zustand/localStorage | Insight query history |
| `similar_uuid` | local state | Similar persona lookup target |
| `path_uuid1`, `path_uuid2` | local state | Relationship path inputs |

---

## 7. Verification Criteria

Each screen must pass the criteria below before being marked complete.

- [ ] Meets core feature parity with existing feature requirements
- [ ] Displays successful responses
- [ ] Displays empty results
- [ ] Displays API errors
- [ ] Displays loading states
- [ ] Connects selected persona state
- [ ] Maintains Korean UI text
- [ ] No TypeScript type errors
- [ ] React build succeeds
- [ ] Related FastAPI tests pass

---

## 8. Risks and Responses

| Risk | Impact | Response |
|---|---|---|
| Many UI behaviors live in a single Streamlit file | Possible migration omissions | Map functions by screen and migrate from the checklist |
| Graph visualization complexity | Schedule increase | Place the graph screen in a later phase |
| API type drift | Runtime errors | Manage types based on OpenAPI or Pydantic schemas |
| Chat session retention failure | Lost conversation context | Store `session_id` reliably in localStorage/Zustand |
| CORS/environment variable errors | Frontend-backend connection failure | Add `.env.example`, explicit `API_BASE_URL`, and a health check |
| 503/data not ready | User confusion | Handle recommendation/centrality not-ready states in separate UI |
| Full migration failure | Difficult rollback | Switch after screen-by-screen parity, and when needed verify regressions against git history, API tests, and current frontend verification results |

---

## 9. Deletion/Retention Policy

- The previous Streamlit UI code is no longer present in the repository.
- Regression verification is performed against API tests, PRD, TASKS, and frontend implementation results.
- Even after React reaches parity, the regression verification baseline remains PRD, TASKS, API tests, and current frontend implementation results.
- If historical UI context is needed, keep it only as separate document records and do not restore deleted frontend paths as the baseline.

---

## 10. Definition of Done

React migration completion criteria:

- [ ] All core workflows from the 11 previous tabs can be performed in the React frontend.
- [ ] Major response values and user flows match Streamlit.
- [ ] FastAPI tests pass.
- [ ] React lint/typecheck/build passes.
- [ ] Error and not-ready states for graph/chat/recommendation/centrality are clearly shown to users.
- [ ] Run documentation is updated for React.
- [ ] Regression verification is possible against git history, API tests, and current frontend verification results.
