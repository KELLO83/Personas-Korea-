# code-review-graph Usage

This operating guide explains how to use the `code-review-graph` MCP effectively in the `Nemotron-Personas-Korea` repository.

This project has a large codebase, and it contains many large experiment artifacts under `experiments/hobby_recommender_ml/artifacts/experiments/`. Because of that, broad queries can time out if they are called casually. The purpose of this document is to explain **how to use the graph first while keeping usage stable and timeout-free in this repository**.

---

## Basic Principles

Based on `AGENTS.md`, this repository should **always use `code-review-graph` before code exploration**.

- Code exploration: `semantic_search_nodes` or `query_graph`
- Impact scope checks: `get_impact_radius`
- Change review: `detect_changes`
- Execution flow checks: `get_affected_flows`
- Test linkage checks: `query_graph(pattern="tests_for")`

However, file-based exploration is acceptable in these cases.

- The graph tool repeatedly times out
- The task is more about checking document state than analyzing code relationships
- Reading a few files directly is faster and more accurate than using a broad graph query

In short, the principle is **graph first**, and the exception is **when the graph cannot practically answer the current question**.

---

## Why Timeouts Are Common in This Repository

This repository has a large graph, and the Git diff scope can easily become large as well.

- Graph size: the repository has many files and nodes, making broad queries expensive
- GNN experiment artifacts: there are many CSV files, model text files, and metrics JSON files under `artifacts/experiments/**`
- Git change scope: if a `HEAD~1` diff includes large generated files, `detect_changes` or `get_minimal_context` can become slow
- Stale graph state: if automatic updates lag behind, the graph state may diverge from the current changes

The following files can especially expand the review/change-analysis scope.

- `experiments/hobby_recommender_ml/artifacts/experiments/**/*.csv`
- `experiments/hobby_recommender_ml/artifacts/experiments/**/ranker_model.txt`
- Large experiment result JSON and summary files

---

## Most Important Operating Rules

### 1. Prefer targeted calls over broad calls

Good example:

```text
detect_changes(base="HEAD~1", changed_files=["src/api/main.py"], detail_level="minimal")
```

Bad example:

```text
get_minimal_context(task="review everything changed in GNN", base="HEAD~1")
```

### 2. Reduce scope by passing `changed_files` directly

In this repository, passing only `base="HEAD~1"` and relying on automatic diff detection can include all experiment artifacts and become slow.

Whenever possible, **explicitly list only the files you actually want to inspect**.

```text
changed_files=[
  "experiments/hobby_recommender_ml/PRD.md",
  "experiments/hobby_recommender_ml/TASKS.md",
  "experiments/hobby_recommender_ml/artifacts/experiment_decisions.json",
  "experiments/hobby_recommender_ml/artifacts/experiment_run_summary.md"
]
```

### 3. Always start with the minimum detail level

Use this as the default pattern.

- First call: `detail_level="minimal"`
- Only when truly needed: `detail_level="standard"`
- Expand further only for a specific function or file

### 4. Exclude generated artifacts from review scope

For broad graph reviews, it is safer to exclude the paths below.

- `experiments/hobby_recommender_ml/artifacts/experiments/**`
- `**/*.csv`
- `**/ranker_model.txt`

These files consume much more token budget and processing time than they contribute to understanding code relationships.

---

## Recommended Usage Order

## 1) Code Exploration

When you want to find a specific feature or symbol:

```text
semantic_search_nodes(query="recommendation ranking", kind="Function", detail_level="minimal")
```

Or when you want to inspect relationships for a specific file or symbol:

```text
query_graph(pattern="file_summary", target="experiments/hobby_recommender_ml/scripts/evaluate_ranker.py", detail_level="minimal")
query_graph(pattern="callers_of", target="some_function_name", detail_level="minimal")
```

Recommended situations:

- You want to find where a function is located
- You want to see what a function calls or what calls it
- You want to see whether tests are connected

---

## 2) Git Change Review

The most basic Git review flow in this repository is:

### Minimal Review Flow

```text
detect_changes(base="HEAD~1", changed_files=[...], detail_level="minimal")
```

Then, if needed:

```text
get_affected_flows(base="HEAD~1", changed_files=[...])
query_graph(pattern="tests_for", target="<changed file or function>", detail_level="minimal")
```

### Example: Reviewing Only Document/Decision Files

```text
detect_changes(
  base="HEAD~1",
  changed_files=[
    "experiments/hobby_recommender_ml/PRD.md",
    "experiments/hobby_recommender_ml/TASKS.md",
    "experiments/hobby_recommender_ml/artifacts/experiment_decisions.json",
    "experiments/hobby_recommender_ml/artifacts/experiment_run_summary.md"
  ],
  detail_level="minimal"
)
```

### Example: Checking the Impact of a Specific Code File

```text
detect_changes(
  base="HEAD~1",
  changed_files=["experiments/hobby_recommender_ml/scripts/evaluate_ranker.py"],
  detail_level="minimal"
)

query_graph(
  pattern="tests_for",
  target="experiments/hobby_recommender_ml/scripts/evaluate_ranker.py",
  detail_level="minimal"
)
```

---

## 3) Impact Scope Analysis

When you want to see how a change affects other modules:

```text
get_impact_radius(
  changed_files=["src/api/main.py"],
  base="HEAD~1",
  max_depth=2,
  detail_level="minimal"
)
```

When you also want execution flows:

```text
get_affected_flows(
  changed_files=["src/api/main.py"],
  base="HEAD~1"
)
```

Recommended situations:

- You want to see which paths are affected by an API change
- You want to see which execution flow includes a recommender engine change

---

## 4) Test Linkage Checks

You can first check whether tests are connected to the code you changed.

```text
query_graph(
  pattern="tests_for",
  target="experiments/hobby_recommender_ml/scripts/evaluate_ranker.py",
  detail_level="minimal"
)
```

If you also want callers of a specific function:

```text
query_graph(pattern="callers_of", target="evaluate_ranker", detail_level="minimal")
query_graph(pattern="callees_of", target="evaluate_ranker", detail_level="minimal")
```

---

## Timeout Response Order

In this repository, the safest response order is:

### 1. Do not repeat the same broad call

For example, if `get_minimal_context(task="full GNN review")` times out, do not retry the same large call shape.

### 2. Reduce the scope immediately

- Do not rely only on automatic diff detection with `base="HEAD~1"`
- Pass `changed_files` directly
- Exclude generated artifacts
- Keep `detail_level="minimal"`

### 3. If it is still slow, switch tools

Examples:

- Broad review -> reduce to `detect_changes`
- Change analysis -> switch to a targeted query such as `query_graph(pattern="tests_for")`
- Document state check -> bypass to `Read/Grep`

### 4. Do not overuse the graph for document analysis tasks

For example, direct reading may be more appropriate than the graph for these tasks.

- Check the relevant `PRD.md` completion state
- Check remaining work in the owning experiment `TASKS.md` when the change is experiment-scoped
- Check decision state in `experiment_decisions.json`
- Check the latest experiment summary in `experiment_run_summary.md`

---

## Graph Update Method

According to this repository's local instructions, the graph is **automatically updated by a hook when files change**.

That means that, in normal cases, it should stay current without a separate manual action.

However, a manual update may be needed in these cases.

- Graph results do not seem to reflect recent changes
- The graph appears stale
- Results look wrong after large file moves or refactors
- You need to prepare a new graph DB in another environment

### Incremental Update

This is the first method to try.

```text
code-review-graph_build_or_update_graph_tool(
  full_rebuild=false,
  base="HEAD~1",
  postprocess="minimal"
)
```

Explanation:

- `full_rebuild=false`: reflect only changed parts
- `base="HEAD~1"`: compute incrementally from recent changes
- `postprocess="minimal"`: reduce cost and update quickly

### Full Rebuild

Use this when the graph is badly inconsistent or an incremental update does not fix the issue.

```text
code-review-graph_build_or_update_graph_tool(
  full_rebuild=true,
  postprocess="minimal"
)
```

A full rebuild is expensive, so it is better to use it only when needed instead of running it frequently.

---

## Recommended Practical Patterns for This Repository

### Pattern A: General Code Review

```text
detect_changes(base="HEAD~1", changed_files=["src/api/main.py"], detail_level="minimal")
get_affected_flows(changed_files=["src/api/main.py"], base="HEAD~1")
query_graph(pattern="tests_for", target="src/api/main.py", detail_level="minimal")
```

### Pattern B: GNN Document/Experiment State Check

```text
detect_changes(
  base="HEAD~1",
  changed_files=[
    "experiments/hobby_recommender_ml/PRD.md",
    "experiments/hobby_recommender_ml/TASKS.md",
    "experiments/hobby_recommender_ml/artifacts/experiment_decisions.json",
    "experiments/hobby_recommender_ml/artifacts/experiment_run_summary.md"
  ],
  detail_level="minimal"
)
```

If that still times out, switch immediately to `Read/Grep`.

### Pattern C: Specific Function Impact Check

```text
query_graph(pattern="callers_of", target="evaluate_ranker", detail_level="minimal")
query_graph(pattern="callees_of", target="evaluate_ranker", detail_level="minimal")
query_graph(pattern="tests_for", target="evaluate_ranker", detail_level="minimal")
```

---

## What Not To Do

- Repeatedly retry broad graph calls
- Run a full `HEAD~1` review while large generated artifacts are included
- Start with `detail_level="standard"` or higher
- Insist on using the graph all the way through a document state check
- Trust results blindly when the graph is stale

---

## Quick Cheatsheet

### Code Exploration

```text
semantic_search_nodes(query="keyword", kind="Function", detail_level="minimal")
query_graph(pattern="file_summary", target="path/to/file.py", detail_level="minimal")
```

### Git Change Review

```text
detect_changes(base="HEAD~1", changed_files=[...], detail_level="minimal")
```

### Impact Check

```text
get_impact_radius(changed_files=[...], base="HEAD~1", detail_level="minimal")
get_affected_flows(changed_files=[...], base="HEAD~1")
```

### Test Linkage

```text
query_graph(pattern="tests_for", target="...", detail_level="minimal")
```

### Graph Refresh

```text
code-review-graph_build_or_update_graph_tool(full_rebuild=false, base="HEAD~1", postprocess="minimal")
code-review-graph_build_or_update_graph_tool(full_rebuild=true, postprocess="minimal")
```

---

## Summary

`code-review-graph` is very useful in this repository, but **small, explicit queries are much more stable than large broad queries**.

The three most important points are:

1. Use the graph first.
2. Always reduce the scope.
3. If the graph repeatedly fails, switch quickly to direct document/file reads.

Following only these principles will significantly reduce most timeouts and unnecessary token usage.
