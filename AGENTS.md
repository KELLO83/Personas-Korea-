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

## Coding Conventions

- Python 4-space indentation, `snake_case` file/function names, `PascalCase` classes
- Type hints for all new/edited functions
- Korean UI text, English code/comments
- Streamlit session state keys: `selected_uuid`, `graph_uuid`, `profile_uuid`, `similar_uuid`, `path_uuid1`, etc.
