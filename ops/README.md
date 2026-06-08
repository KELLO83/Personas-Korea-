# Platform Ops

This directory contains shared operational entrypoints for the platform runtime.
Experiment-specific scripts belong under `experiments/<experiment_name>/`.

## Folders

- `data/`: dataset download, inspection, and local preview helpers.
- `graph/`: Neo4j graph build, graph backfill, and GDS pipeline entrypoints.
- `vector/`: embedding generation and pgvector schema/load utilities.
- `dev/`: local environment checks and developer-only helpers.

## Runtime

Use the project Python 3.14 environment unless an experiment document explicitly
requires another runtime:

```powershell
.\.venv314\Scripts\python.exe ops\graph\build_graph.py --help
.\.venv314\Scripts\python.exe ops\vector\load_pgvector_embeddings.py --help
```

Keep one-off smoke checks out of this directory. If a check is worth keeping,
turn it into a focused test under `tests/` or document it in the relevant
experiment folder.
