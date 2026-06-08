# Centrality Batch Operations

This document describes the operating procedure for precomputing the PageRank, Degree, and Betweenness centrality scores required for F10 network influence analysis in Neo4j.

Centrality calculations **must not run on the FastAPI request path.** They must run from an external scheduler that is separate from the application process, such as Windows Task Scheduler, cron, or an operations batch system.

## 1) Execution Entry Point

The executable for the centrality batch is:

```powershell
.\.venv314\Scripts\python.exe -m src.jobs.centrality_batch
```

Options:

- `--metrics <list>`: Comma-separated centrality metrics. The default is `pagerank,degree`
- `--recreate-projection`: Drop and recreate the existing GDS projection
- `--betweenness-sampling-size <N>`: Betweenness sampling size. The default is `10000`

Examples:

```powershell
# Recommended daily run: PageRank + Degree
.\.venv314\Scripts\python.exe -m src.jobs.centrality_batch --metrics pagerank,degree

# Use after Neo4j restart or when projection mismatch is suspected
.\.venv314\Scripts\python.exe -m src.jobs.centrality_batch --metrics pagerank,degree --recreate-projection

# Recommended weekly run: include sampled Betweenness
.\.venv314\Scripts\python.exe -m src.jobs.centrality_batch --metrics pagerank,degree,betweenness --betweenness-sampling-size 10000
```

## 2) Recommended Schedule

| Job | Frequency | Command |
|---|---|---|
| PageRank + Degree | Daily, 02:00 | `python -m src.jobs.centrality_batch --metrics pagerank,degree` |
| Betweenness sampling | Weekly, Sunday 02:00 | `python -m src.jobs.centrality_batch --metrics pagerank,degree,betweenness --betweenness-sampling-size 10000` |
| Projection recreation | After Neo4j restart/maintenance | Add `--recreate-projection` to the command above |

## 3) Windows Task Scheduler Example

Create a new task in Task Scheduler and use these values.

| Item | Value |
|---|---|
| Program/script | `C:\Users\Kello\Nemotron-Personas-Korea\.venv314\Scripts\python.exe` |
| Add arguments | `-m src.jobs.centrality_batch --metrics pagerank,degree` |
| Start in | `C:\Users\Kello\Nemotron-Personas-Korea` |
| Trigger | Daily, 02:00 |

Create the weekly Betweenness job as a separate task and change only `Add arguments` as follows.

```text
-m src.jobs.centrality_batch --metrics pagerank,degree,betweenness --betweenness-sampling-size 10000
```

## 4) cron Example

In Linux/WSL environments, register the jobs from the project root as follows.

```cron
# PageRank + Degree: daily 02:00
0 2 * * * cd /path/to/Nemotron-Personas-Korea && ./.venv314/Scripts/python.exe -m src.jobs.centrality_batch --metrics pagerank,degree

# Betweenness: Sunday 02:00
0 2 * * 0 cd /path/to/Nemotron-Personas-Korea && ./.venv314/Scripts/python.exe -m src.jobs.centrality_batch --metrics pagerank,degree,betweenness --betweenness-sampling-size 10000
```

If the operations OS is Linux, change the Python path to `./.venv314/bin/python`.

## 5) Post-Run Checks

Check the following in Neo4j Browser or from a Python script.

```cypher
MATCH (p:Person) WHERE p.pagerank IS NOT NULL RETURN count(p);
MATCH (p:Person) WHERE p.degree IS NOT NULL RETURN count(p);
MATCH (p:Person) WHERE p.betweenness IS NOT NULL RETURN count(p);
MATCH (s:SystemStatus {key: 'centrality_batch'})
RETURN s.status, s.run_id, s.last_success_at, s.metrics;
```

API check:

```powershell
# After starting the backend
curl "http://localhost:8000/api/influence/top?metric=pagerank&limit=10"
```

Expected response example:

```json
{
  "metric": "pagerank",
  "last_updated_at": "2026-04-28T02:00:00+00:00",
  "run_id": "centrality-20260428-020000",
  "stale_warning": false,
  "results": []
}
```

In a healthy state, `results` is not empty and the response includes `last_updated_at` and `run_id`.

## 6) Failure Handling

- `ServiceUnavailableException` or a 503 response: centrality scores have not been computed, or the batch is in a failed state.
- Projection errors after a Neo4j restart: rerun with `--recreate-projection`.
- Long Betweenness runtime or high memory usage: lower `--betweenness-sampling-size` and run it only as a weekly batch.
- Do not start the batch automatically during user API requests. Operators should rerun it through the external scheduler or a manual command.

## 7) Related Documents

- `PRD.md` §3, §7.3
- `ops/decisions/ADR-001-gds-precompute.md`
- `src/jobs/centrality_batch.py`


