# Project: Korean Persona Knowledge Graph

## Python Environment

This project uses `.venv` with Python 3.11 as the default Python runtime. Never
use global/system Python.

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app
```

## Project Map

- `frontend/` - Next.js frontend
- `src/api/` - FastAPI backend
- `src/rag/` - LangChain/LangGraph RAG
- `src/graph/`, `src/gds/` - Neo4j graph and GDS operations
- `src/embeddings/` - KURE-v1 Korean text embeddings
- `GNN_Neural_Network/` - `Person -> Hobby` recommender
- `experiments/persona_similarity/` - `Person -> Person` recommender
- `scripts/`, `configs/`, `tests/` - shared pipeline/config/test code

## Recommender Boundaries

This project has two recommender systems. Keep their datasets, labels, metrics,
artifacts, and model decisions separate.

- Hobby recommender: `GNN_Neural_Network/AGENTS.md`
  - Task: recommend hobbies for a persona.
  - Scope: `Person -> Hobby`.
- Similar-persona recommender: `experiments/persona_similarity/AGENTS.md`
  - Task: recommend similar personas for a source persona.
  - Scope: `Person -> Person`.

## Document Routing

- Root `PRD.md` and `TASKS.md` are for platform/product scope only:
  FastAPI, Neo4j graph operations, RAG/chatbot, frontend, and integration gates.
- Put hobby recommender experiment plans, metrics, and decisions under
  `GNN_Neural_Network/`.
- Put similar-persona experiment plans, metrics, and decisions under
  `experiments/persona_similarity/`.
- Update root docs only when API behavior, frontend behavior, graph build
  behavior, or production integration changes.

## Run Commands

```powershell
# Install dependencies
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Run tests
.\.venv\Scripts\python.exe -m pytest tests -q

# Start backend
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

## Windows Encoding

Before running scripts that print Korean text in PowerShell:

```powershell
chcp 65001
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
```

## Infrastructure

- Neo4j Docker container: `neo4j-personas`
- Neo4j ports: bolt `7687`, HTTP `7474`
- OS: Windows PowerShell
- Python: 3.11 via `.venv`

## Resource Policy

CPU-heavy ML code should default to this laptop profile:

- CPU: Intel Core Ultra 7 155H class, 16 cores / 22 logical processors.
- Default CPU threads/workers: `18`.
- Do not default to all logical processors.
- Recommended fallback: `min(max(os.cpu_count() - 4, 1), 18)`.
- Apply this to LightGBM `num_threads`, joblib `n_jobs`, process pools, PyTorch
  CPU threads, NumPy/BLAS env vars, feature builders, and evaluation.
- Python 3.11 CPU-bound Python loops are constrained by the GIL. For Python-heavy
  work such as feature row construction, candidate feature building, text/cache
  post-processing, and pure-Python evaluation transforms, default to
  multiprocessing/process pools rather than thread pools. Threads are acceptable
  for I/O-bound work or native libraries that release the GIL.

```powershell
$env:OMP_NUM_THREADS = "18"
$env:MKL_NUM_THREADS = "18"
$env:OPENBLAS_NUM_THREADS = "18"
$env:NUMEXPR_NUM_THREADS = "18"
```

GPU-capable code should use this profile:

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8GB VRAM.
- Use CUDA automatically when available, with CPU fallback.
- Use available VRAM efficiently through configurable batch/chunk sizes.
- Do not intentionally allocate 100% of VRAM; leave headroom for Windows,
  browser, Docker/Neo4j tools, and CUDA allocator fragmentation.
- Record device, batch/chunk size, runtime, and peak GPU memory for major runs.

## Experiment UX Policy

For model training, evaluation, embedding, candidate generation, and feature
building:

- Show progress with `tqdm` or equivalent for long-running jobs.
- Persist reusable cache/artifacts and reuse them when data/config/model/split
  metadata matches.
- Expose `--force`, `--rebuild`, or `--no-cache` when recomputation is needed.
- Persist machine-readable metrics/status under the relevant experiment folder.
- Use one script per experiment purpose; do not silently run many experiments
  from one script.

## Git Safety

- Never run `git push`.
- Do not revert user changes unless explicitly requested.
- Local commits are allowed only when explicitly requested.

## Coding Conventions

- Python: 4-space indentation, `snake_case` functions/files, `PascalCase` classes.
- Add type hints for new or edited functions.
- Korean UI text, English code/comments.
- Do not add unnecessary comments or docstrings.

## Code Exploration

If code-review graph MCP tools are available, use them before broad text search
for architecture, impact, and review context. Fall back to `rg` when graph tools
are unavailable or insufficient.
