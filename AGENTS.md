# Project: Korean Persona Knowledge Graph

## Python Environment (CRITICAL)

**This project uses a `.venv` virtual environment with Python 3.11. NEVER use the global/system Python.**

All Python commands MUST be executed via the venv interpreter:

```powershell
# CORRECT - always use .venv Python
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe -m py_compile app\streamlit_app.py
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py

# WRONG - global Python lacks project dependencies (langchain, neo4j, etc.)
python -m pytest tests
pytest tests
python -m py_compile app\streamlit_app.py
```

**Why**: The global Python does not have `langchain_core`, `langgraph`, `neo4j`, `streamlit`, and other project dependencies installed. Using it will cause `ModuleNotFoundError`.

## Project Structure

- `app/streamlit_app.py` - Streamlit frontend (single file, ~1200 lines)
- `src/api/` - FastAPI backend (routes, main)
- `src/rag/` - LangChain/LangGraph RAG engine
- `src/graph/` - Neo4j graph operations, GDS algorithms
- `src/data/` - Data loader, parser, preprocessor
- `tests/` - PyTest test suite (67+ tests)

## Build, Test, and Run Commands

```powershell
# Install dependencies
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Run tests
.\.venv\Scripts\python.exe -m pytest tests -q

# Compile check
.\.venv\Scripts\python.exe -m py_compile app\streamlit_app.py

# Start backend (FastAPI on :8000)
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Start frontend (Streamlit on :8501)
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py --server.port 8501
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

## Coding Conventions

- Python 4-space indentation, `snake_case` file/function names, `PascalCase` classes
- Type hints for all new/edited functions
- Korean UI text, English code/comments
- Streamlit session state keys: `selected_uuid`, `graph_uuid`, `profile_uuid`, `similar_uuid`, `path_uuid1`, etc.
