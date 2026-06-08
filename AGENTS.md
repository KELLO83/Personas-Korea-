# Project LLM Instructions: Korean Persona Knowledge Graph

This is the top-level instruction document for LLM coding agents working in
this repository. Follow it before local README/PRD notes unless a more specific
`AGENTS.md` exists in the target subdirectory.

Primary rule: keep platform code, recommender experiments, RAG, Neo4j, and
pgvector work separated by folder and runtime boundary. Do not restore deleted
planning files or archive documents unless the user explicitly asks.

## Python Environment

This project uses `.venv314` with Python 3.14 as the default
backend, platform/RAG, and general ML runtime. Never use global/system Python.
Do not use or recreate `.venv` Python 3.11; that runtime has been retired for
this repository.

```powershell
.\.venv314\Scripts\python.exe -m pytest tests -q
.\.venv314\Scripts\python.exe -m uvicorn src.api.main:app
```

### Python 3.14 Service and General ML Environment

Use `.venv314` Python 3.14 for:

- Backend/FastAPI/frontend integration.
- Neo4j connections and graph exports.
- LangChain/LangGraph RAG and production API path.
- pandas/openpyxl/pyarrow-based utility scripts.
- Dataset download and platform/RAG work.
- General ML experiments, including PyTorch and sentence-transformers.

### Python 3.14t Free-Threaded ML Experiment Environment

`.venv314t` is allowed only for local ML experiment acceleration, not for the
backend, frontend, Neo4j integration, RAG, or production API path.

- Default runtime is `.venv314` Python 3.14.
- Backend/FastAPI must run with `.venv314` Python 3.14.
- Use `.venv314t` only when the specific training/evaluation path has been
  verified under free-threaded Python 3.14t.
- Do not mix `.venv314` and `.venv314t` artifacts unless the experiment metadata
  records the Python executable, package versions, device, and cache identity.
- `shap` is not required by the current recommender training/evaluation path.
  Current explanation code uses LightGBM `pred_contrib=True`, not the external
  SHAP package.

Use `.venv314t` Python 3.14t only for:

- Reading already-exported local parquet/csv/npz/artifacts.
- LightGBM-heavy training/evaluation paths where free-threaded execution has
  been measured to help.
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
- `ops/` - platform operational entrypoints for data, graph, vector, and dev checks
- `experiments/hobby_recommender_ml/` - `Person -> Hobby` recommender
- `experiments/persona_similarity/` - `Person -> Person` recommender
- `experiments/persona_segmentation/` - persona clustering and segment discovery
- `experiments/persona_quality_model/` - persona data-quality and usefulness scoring
- `configs/`, `tests/` - shared config and test code

There is no root-level `scripts/` contract. Platform entrypoints belong under
`ops/`; experiment-only entrypoints may live under
`experiments/<experiment_name>/scripts/`.

## Experiment Boundaries

Keep datasets, labels, metrics, artifacts, and model decisions separate across
experiments. Do not merge experiment outputs into platform behavior until the
user explicitly asks for integration.

- Hobby recommender: `experiments/hobby_recommender_ml/AGENTS.md`
  - Task: recommend hobbies for a persona.
  - Scope: `Person -> Hobby`.
- Similar-persona recommender: `experiments/persona_similarity/AGENTS.md`
  - Task: recommend similar personas for a source persona.
  - Scope: `Person -> Person`.
- Persona segmentation: `experiments/persona_segmentation/`
  - Task: discover persona groups and explain segment traits.
  - Scope: `Person -> Segment`.
- Persona quality model: `experiments/persona_quality_model/`
  - Task: score persona record completeness, consistency, and downstream usefulness.
  - Scope: `Person -> QualityScore`.

## Document Routing

- Root `PRD.md` and `README.md` are for platform/product scope only:
  FastAPI, Neo4j graph operations, RAG/chatbot, frontend, and integration gates.
- Put hobby recommender experiment plans, metrics, and decisions under
  `experiments/hobby_recommender_ml/`.
- Put similar-persona experiment plans, metrics, and decisions under
  `experiments/persona_similarity/`.
- Put persona segmentation plans, metrics, and decisions under
  `experiments/persona_segmentation/`.
- Put persona quality model plans, metrics, and decisions under
  `experiments/persona_quality_model/`.
- Put experiment-specific LLM wiki notes under each experiment folder's
  `llm_wiki/` directory.
- Duplicate shared LLM wiki source cards, templates, and
  cross-experiment concepts into each relevant experiment folder's `llm_wiki/`
  directory so each experiment remains self-contained.
- Update root documents only when API behavior, frontend behavior, graph build
  behavior, or production integration changes.
- Do not recreate deleted `TASKS.md`, `docs/`, or archive documents unless the
  user explicitly asks.

## Run Commands

```powershell
# Install dependencies
.\.venv314\Scripts\python.exe -m pip install -r requirements.txt

# Run tests
.\.venv314\Scripts\python.exe -m pytest tests -q

# Start backend
.\.venv314\Scripts\python.exe -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
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
- PostgreSQL/pgvector runtime: local PostgreSQL service, not Docker.
- PostgreSQL/pgvector URI: `postgresql://postgres:1234@localhost:5432/persona_vector`
- Do not use the old TOEIC pgvector Docker container for this project.
- OS: Windows PowerShell
- Python: 3.14 via `.venv314`

## Resource Policy

CPU-heavy ML code should default to this laptop profile:

- CPU: Intel Core Ultra 7 155H class, 16 cores / 22 logical processors.
- Default CPU threads/workers: `12`.
- Do not default to all logical processors.
- Recommended fallback: `min(max(os.cpu_count() - 4, 1), 12)`.
- Apply this to LightGBM `num_threads`, joblib `n_jobs`, thread pools, PyTorch
  CPU threads, NumPy/BLAS env vars, feature builders, and evaluation.
- For recommender ML experiments, use `ThreadPoolExecutor` for Python-heavy
  feature row construction, candidate feature building, text/cache
  post-processing, and pure-Python evaluation transforms. The thread-pool
  policy is used consistently across `.venv314` Python 3.14 and `.venv314t`
  Python 3.14t to avoid duplicated worker memory and OOM shutdowns. Record
  worker/thread counts in experiment metadata.

```powershell
$env:OMP_NUM_THREADS = "12"
$env:MKL_NUM_THREADS = "12"
$env:OPENBLAS_NUM_THREADS = "12"
$env:NUMEXPR_NUM_THREADS = "12"
```

GPU-capable code should use this profile:

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8GB VRAM.
- Use CUDA automatically when available, with CPU fallback.
- When GPU is available, target up to 90% of usable VRAM for ML workloads while
  keeping 10% headroom for Windows, browser, Docker/Neo4j tools, and CUDA
  allocator fragmentation.
- Do not hard-code a fixed batch size as the default. Use auto batch sizing that
  probes or adapts batch/chunk size to the available VRAM budget, with explicit
  override flags only when needed for reproducibility or debugging.
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
