# Project: Korean Persona Knowledge Graph

## Python Environment (CRITICAL)

**This project uses a `.venv` virtual environment with Python 3.11. NEVER use the global/system Python.**

All Python commands MUST be executed via the venv interpreter:

```powershell
# CORRECT - always use .venv Python
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app

# WRONG - global Python lacks project dependencies (langchain, neo4j, etc.)
python -m pytest tests
pytest tests
```

**Why**: The global Python does not have `langchain_core`, `langgraph`, `neo4j`, and other project dependencies installed. Using it will cause `ModuleNotFoundError`.

## Project Structure

- `frontend/` - Next.js frontend
- `src/api/` - FastAPI backend (routes, main)
- `src/rag/` - LangChain/LangGraph RAG engine
- `src/graph/` - Neo4j graph operations, GDS algorithms
- `src/gds/` - Neo4j Graph Data Science pipelines (FastRP, KNN, Leiden, PageRank)
- `src/embeddings/` - KURE-v1 Korean text embedding model + vector index
- `src/data/` - Data loader, parser, preprocessor
- `src/jobs/` - Batch jobs (centrality computation)
- `GNN_Neural_Network/` - GNN hobby recommender PoC (LightGCN + LightGBM)
- `configs/` - YAML/JSON config files
- `scripts/` - Data pipeline, test scripts
- `tests/` - PyTest test suite (212+ tests)

## Key Documents

- `PRD.md` - Root product requirements for the full Korean Persona Knowledge Graph platform (FastAPI, Neo4j, RAG, frontend)
- `TASKS.md` - Root implementation checklist for the platform features
- `GNN_Neural_Network/PRD.md` - GNN offline hobby recommender system requirements (LightGCN + LightGBM ranker)
- `GNN_Neural_Network/TASKS.md` - GNN offline model implementation checklist
- `GNN_Neural_Network/CHECKLIST_GNN_Reranker_v2.md` - GNN Stage 2.5 reranker experiment and reranker checklist
- `GNN_Neural_Network/DATASET_EXPLAIN.md` - 26개 컬럼, 1M row, 그래프 매핑 등 전체 데이터셋 구조 설명

**Important**: `PRD.md` + `TASKS.md` are the root platform source of truth for API/web product scope.
- `GNN_Neural_Network/PRD.md` and `GNN_Neural_Network/TASKS.md` are the source of truth for offline recommender experiments and implementation state.
- `GNN_Neural_Network/CHECKLIST_GNN_Reranker_v2.md` tracks Stage 2.5 reranker-specific experiment checkpoints.

When root `TASKS.md` references root tasks that depend on GNN results (for example, F11), gate completion should be documented based on the latest `GNN_Neural_Network` experiment decision artifacts.

## Document Routing Rules

- Root `PRD.md` and `TASKS.md` are only for platform/product scope:
  - FastAPI APIs
  - Neo4j graph operations
  - RAG/chatbot
  - frontend
  - root-level integration gates such as F11
- All GNN recommender model training/evaluation work MUST be documented under `GNN_Neural_Network/`:
  - `GNN_Neural_Network/PRD.md` for requirements, experiment policy, promotion gates, and baseline decisions
  - `GNN_Neural_Network/TASKS.md` for executable experiment tasks and checklists
  - `GNN_Neural_Network/CHECKLIST_GNN_Reranker_v2.md` for reranker-specific historical/sub-checklists
- Do NOT add detailed GNN experiment plans, metric tables, model hyperparameter plans, training/evaluation gates, or artifact schemas to root `PRD.md` or root `TASKS.md`.
- Root `TASKS.md` may reference GNN work only as a platform integration gate, for example:
  - check latest GNN decision artifacts
  - decide whether GNN should integrate with F11
  - link to `GNN_Neural_Network/PRD.md` and `GNN_Neural_Network/TASKS.md`
- If a task changes GNN default recommendation behavior, update:
  - `GNN_Neural_Network/PRD.md`
  - `GNN_Neural_Network/TASKS.md`
  - `GNN_Neural_Network/artifacts/experiment_decisions.json`
  - `GNN_Neural_Network/artifacts/experiment_run_summary.md`
  - `GNN_Neural_Network/README.md` if user-facing run instructions or current status changed
- Update root `PRD.md`/`TASKS.md` only when the change affects platform integration, API behavior, frontend behavior, or root F11 acceptance state.
 
## Build, Test, and Run Commands

```powershell
# Install dependencies
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Run tests
.\.venv\Scripts\python.exe -m pytest tests -q

# Start backend (FastAPI on :8000)
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# GNN: train LightGCN
.\.venv\Scripts\python.exe GNN_Neural_Network\scripts\train_lightgcn.py

# GNN: train/evaluate LightGBM ranker
.\.venv\Scripts\python.exe GNN_Neural_Network\scripts\train_ranker.py
.\.venv\Scripts\python.exe GNN_Neural_Network\scripts\evaluate_ranker.py --split test

# GNN: run Phase 2.5 tuning experiments
.\.venv\Scripts\python.exe GNN_Neural_Network\scripts\tune_ranker_regularization.py
.\.venv\Scripts\python.exe GNN_Neural_Network\scripts\ablation_neg_sampling.py
.\.venv\Scripts\python.exe GNN_Neural_Network\scripts\source_onehot_ablation.py
```

## Infrastructure

- **Neo4j**: Docker container `neo4j-personas` (neo4j:5), bolt on 7687, HTTP on 7474
- **Python**: 3.11 via `.venv` (NOT 3.14, NOT global)
- **OS**: Windows (PowerShell)

## Git Safety

- Agents MUST NOT run `git push` under any circumstance.
- Agents may inspect git status/diff/log and may prepare local commits only when explicitly requested.
- Pushing to any remote repository is reserved for the human user.

## GNN Training/Evaluation Performance Policy

When writing or modifying code under `GNN_Neural_Network/`:

- Use CUDA automatically when available, with CPU fallback.
- Use available GPU/CPU resources efficiently, but do not intentionally allocate all VRAM or memory.
- Keep batch size, chunk size, and candidate pool size configurable via YAML or CLI.
- Avoid repeated graph propagation or embedding recomputation inside tight loops.
- Cache reused LightGCN/XSimGCL embeddings, adjacency tensors, popularity counts, and co-occurrence counts.
- Prefer batched GPU tensor operations for dense scoring, masking, and top-k when practical.
- Avoid per-person model forward calls when batch scoring is possible.
- Training code should avoid duplicate LightGCN propagation for positive and negative scores in the same batch.
- Use CPU loops only when the data structure is small or vectorization would add unnecessary complexity.
- Record training/evaluation time and GPU memory usage for major experiments when practical.
- Preserve deterministic evaluation: fixed split, fixed seed, same candidate pool, and same known-hobby masking rules.

## GNN Experiment Logging Policy

When running or modifying training/evaluation code under `GNN_Neural_Network/`:

- Persist machine-readable metrics to `GNN_Neural_Network/artifacts/` instead of relying only on console output.
- For major experiments, maintain these two decision artifacts in addition to raw metric files:
  - `GNN_Neural_Network/artifacts/experiment_decisions.json`
    - machine-readable decision log
    - record the selected baseline, experiment status such as `accepted`, `rejected`, `promoted`, `disabled`, `experimental`, or `needs_followup`, key metric deltas, and short decision reasons
  - `GNN_Neural_Network/artifacts/experiment_run_summary.md`
    - human-readable run summary
    - record what was tested, the main outcome, what changed in the default recommendation path, and the next recommended step
- For each major experiment, record:
  - what was tested,
  - the selected baseline,
  - the main metrics such as Recall, NDCG, and candidate recall,
  - whether the change was accepted, rejected, promoted, disabled, or left experimental,
  - and the short reason for that decision.
- When a provider, model, feature, or taxonomy rule is rejected, record the failure reason concisely, for example:
  - degraded validation recall,
  - toxic in ablation,
  - below selected baseline,
  - leakage risk,
  - or qualitative recommendation quality regression.
- Keep a human-readable experiment summary artifact in addition to raw JSON metrics when the decision changes the default recommendation path.
- Update both `experiment_decisions.json` and `experiment_run_summary.md` whenever a training/evaluation run changes the default recommendation path or closes an experiment with a clear decision.
- If a result is inconclusive, record that explicitly as `experimental` or `needs_followup` instead of leaving the decision implicit.
- If the default Stage 1 baseline, Stage 2 promotion status, provider policy, or taxonomy policy changes, update the relevant `README.md`, `PRD.md`, and `TASKS.md` entries in the same task.
- Prefer short decision logs over long narrative notes. The goal is reproducibility and later review.

## Coding Conventions

- Python 4-space indentation, `snake_case` file/function names, `PascalCase` classes
- Type hints for all new/edited functions
- Korean UI text, English code/comments
- **Do NOT add unnecessary comments or docstrings unless explicitly requested.**

<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, cheaper (fewer tokens), and gives
you structural context (callers, dependents, test coverage) that file
scanning cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes` or `query_graph` instead of Grep
- **Understanding impact**: `get_impact_radius` instead of manually tracing imports
- **Code review**: `detect_changes` + `get_review_context` instead of reading entire files
- **Finding relationships**: `query_graph` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview` + `list_communities`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.

### Key Tools

| Tool | Use when |
|------|----------|
| `detect_changes` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context` | Need source snippets for review — token-efficient |
| `get_impact_radius` | Understanding blast radius of a change |
| `get_affected_flows` | Finding which execution paths are impacted |
| `query_graph` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes` | Finding functions/classes by name or keyword |
| `get_architecture_overview` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. The graph auto-updates on file changes (via hooks).
2. Use `detect_changes` for code review.
3. Use `get_affected_flows` to understand impact.
4. Use `query_graph` pattern="tests_for" to check coverage.
