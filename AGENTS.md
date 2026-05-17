# Project: Korean Persona Knowledge Graph

## Python Environment

This project uses `.venv` with Python 3.11 as the default Python runtime. Never
use global/system Python.

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app
```

### Python 3.14t ML Experiment Environment

`.venv314t` is allowed only for local ML experiment acceleration, not for the
backend, frontend, Neo4j integration, RAG, or production API path.

- Default runtime remains `.venv` Python 3.11.
- Backend/FastAPI must run with `.venv` Python 3.11.
- Use `.venv314t` only when the specific training/evaluation path has been
  verified under free-threaded Python 3.14t.
- Do not mix `.venv` and `.venv314t` artifacts unless the experiment metadata
  records the Python executable, package versions, device, and cache identity.
- `shap` is not required by the current recommender training/evaluation path.
  Current explanation code uses LightGBM `pred_contrib=True`, not the external
  SHAP package.

Use `.venv` Python 3.11 for:

- Backend/FastAPI/frontend integration.
- Neo4j connections and graph exports.
- Excel exports.
- pandas/openpyxl/pyarrow-based utility scripts.
- Dataset download and platform/RAG work.

Use `.venv314t` Python 3.14t only for:

- Reading already-exported local parquet/csv/npz/artifacts.
- Polars-based feature building.
- KURE/Snowflake text embedding generation.
- LightGBM training/evaluation.
- Python-heavy CPU feature/evaluation loops where free-threaded
  multithreading reduces memory pressure.


Verified `.venv314t` local ML package set:

```text
torch 2.11.0+cu128
sentence-transformers 5.5.0
transformers 5.8.1
tokenizers 0.23.0-rc0
polars 1.37.1 + polars-runtime-32-ft 1.37.1
lightgbm 4.6.0
numpy/scipy/scikit-learn/tqdm/pyyaml
psutil
```

Important caveats:

- `tokenizers 0.23.0-rc0` is a release candidate. It currently keeps the GIL
  disabled without `-X gil=0` and satisfies `transformers 5.8.1`, but treat this
  as experimental until a stable `transformers` release accepts
  `tokenizers 0.23.1`.
- `polars-runtime-32-ft` is an unofficial wheel and is only for Polars `1.37.1`.

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
- Apply this to LightGBM `num_threads`, joblib `n_jobs`, thread pools, PyTorch
  CPU threads, NumPy/BLAS env vars, feature builders, and evaluation.
- For recommender ML experiments, use `ThreadPoolExecutor` for Python-heavy
  feature row construction, candidate feature building, text/cache
  post-processing, and pure-Python evaluation transforms. The thread-pool
  policy is used consistently across `.venv` Python 3.11 and `.venv314t` Python
  3.14t to avoid duplicated worker memory and OOM shutdowns. Record
  worker/thread counts in experiment metadata.

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
