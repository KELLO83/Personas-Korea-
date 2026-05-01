# Phase 22 F16-F18 Implementation Scope

## Purpose

This document records the implementation scope, exclusions, and execution order for PRD Phase 4 / TASKS Phase 22.

## Implemented scope

- F16 Target Persona Generator
  - Backend: `GET /api/target-persona`
  - Frontend: `확장 분석` tab
  - Supports deterministic summary, evidence UUIDs, guardrails, optional LLM synthesis, and KURE semantic filtering.
- F17 Cross-domain Lifestyle Map
  - Backend: `GET /api/lifestyle-map`
  - Frontend: `확장 분석` tab
  - Supports persona text field pairs, source keyword, candidate keyword list, automatic keyword extraction, segment filters, and conditional ratio output.
- F18 Career Transition Map
  - Backend: `GET /api/career-transition-map`
  - Frontend: `확장 분석` tab
  - Supports occupation-centered goal, skill, adjacent occupation, and segment distribution analysis.
- Graph quality review
  - Backend: `GET /api/graph-quality`
  - Frontend: `확장 분석` tab and profile/search exposure policy
  - Supports Country, MilitaryStatus, and bachelors_field distribution checks with action/severity flags.

## Explicit exclusions

- No destructive Country node migration is executed by the API.
- No GNN Phase 2.5 integration is included until experiment artifacts are finalized.
- No default LLM synthesis is enabled for F16; `use_llm=true` is opt-in and falls back to deterministic output.
- F18 is analysis/comparison only. It does not replace the existing recommendation API.

## Execution order

1. Use `GET /api/graph-quality` to verify Country, MilitaryStatus, and bachelors_field distributions.
2. Keep low-information profile/search fields hidden or low-priority in the frontend.
3. Validate F16 with `configs/target_persona_golden_set.json` before promoting LLM synthesis beyond opt-in.
4. Validate F17 keyword policies with selected persona text field pairs.
5. Validate F18 occupation query quality before adding personalized recommendation flows.
6. Revisit Phase 23 only after GNN Phase 2.5 experiment artifacts are complete.

## Verification commands

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_graph_quality.py tests/test_target_persona.py tests/test_lifestyle_map.py tests/test_career_transition.py tests/test_target_persona_golden_set.py -q
cd frontend
npm run typecheck
```
