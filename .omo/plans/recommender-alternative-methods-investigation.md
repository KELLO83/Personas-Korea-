# Recommender Alternative Methods Investigation

## TL;DR
> **Summary**: 현재 `Person -> Hobby`와 `Person -> Person` 추천 시스템을 분리해 데이터셋 shape, 기존 baseline, 대체 추천 방법의 적합성을 조사하고, 실행 가능한 benchmark 후보를 순위화한다.
> **Deliverables**:
> - `.omo/evidence/recommender-methods/dataset-shape-report.md`
> - `.omo/evidence/recommender-methods/current-baseline-inventory.md`
> - `.omo/evidence/recommender-methods/alternative-method-shortlist.md`
> - `.omo/evidence/recommender-methods/benchmark-design.md`
> - `.omo/evidence/recommender-methods/final-recommendation.md`
> - `docs/llm_wiki/` recommender research wiki updates
> **Effort**: Short
> **Parallel**: YES - 4 waves
> **Critical Path**: Task 1 -> Task 4 -> Task 7 -> Task 9 -> Final Verification

## Context
### Original Request
The user asked, in Korean, to continue planning research for the current recommender system and similar-persona recommendation model, asking whether there are other recommendation methods and requiring dataset-shape understanding.

### Interview Summary
- Scope is research/planning, not immediate production promotion.
- Keep the two recommender boundaries separate:
  - `GNN_Neural_Network/`: `Person -> Hobby`
  - `experiments/persona_similarity/`: directed `source_uuid -> target_uuid`
- No source-code changes are required for this plan unless a later worker explicitly executes benchmark prototypes.

### Gap Review (gaps addressed)
- Metis agent request was issued, but the agent did not return substantive output after repeated waits; this plan uses the planner's explicit gap audit instead.
- Guardrail applied: do not compare hobby metrics directly with similar-persona metrics.
- Guardrail applied: do not treat weak-label NDCG as production readiness for similar-persona reranking.
- Guardrail applied: text features must be evaluated with leakage and manual-review gates before promotion.
- Guardrail applied: topK/candidate width must be recorded for every comparison.

## Work Objectives
### Core Objective
Create a decision-ready investigation report that ranks alternative recommendation methods by fit for the current Korean persona dataset shape, existing artifacts, evaluation policy, and implementation cost.

### Deliverables
- Dataset-shape report for both recommender systems.
- Current baseline inventory with metrics and artifact paths.
- Alternative-method shortlist with fit/no-go decisions.
- Benchmark design with exact commands, metrics, artifacts, and pass/fail gates.
- Final recommendation that names which method family to try first, which to defer, and which to reject.
- LLM Wiki pages that preserve durable source cards, concept notes, and experiment notes for future agents.
- Separate LLM Wiki experiment-track folders for `Person -> Hobby` and `Person -> Person`.

### Definition of Done
- `Test-Path .omo/evidence/recommender-methods/final-recommendation.md` returns `True`.
- Dataset-shape report includes row counts, columns, candidate width, label unit, split policy, and leakage risks for both systems.
- Alternative-method shortlist includes at least 8 method families and each has `fit`, `required_shape`, `expected_benefit`, `risk`, `evaluation_gate`, and `decision`.
- Benchmark design includes exact commands for `.venv` Python 3.11 and explicitly marks `.venv314t` as optional only for already-exported artifact acceleration.
- `docs/llm_wiki/INDEX.md` links source cards, concept notes, and the experiment-plan note for this recommender investigation.
- `docs/llm_wiki/person_hobby/` and `docs/llm_wiki/persona_similarity/` exist and keep ML experiment plans separate.
- No production/backend/frontend behavior is changed.

### Must Have
- Use `.venv` Python 3.11 for all artifact inspection commands.
- Preserve folder-specific scope rules from `AGENTS.md`.
- Separate `Person -> Hobby` and `Person -> Person` labels, metrics, datasets, and decisions.
- Include manual-review gates for text-driven or weak-label-driven recommendations.
- Include Korean persona synthetic-data caveats.

### Must NOT Have
- Do not merge the two recommender systems into one benchmark.
- Do not promote any model.
- Do not run Neo4j rebuilds or long GPU training during investigation unless the executor explicitly chooses to execute a later benchmark plan.
- Do not use raw `uuid`, `display_name`, or raw text identifiers as model features.
- Do not compare topK=5 smoke artifacts against topK=50 serious reranker artifacts.

## Verification Strategy
> ZERO HUMAN INTERVENTION - all verification is agent-executed.
- Test decision: none for source code; this is a research/report artifact plan. Use artifact schema checks and command-output evidence.
- QA policy: Every task has agent-executed scenarios with concrete commands and evidence paths.
- Evidence root: `.omo/evidence/recommender-methods/`

## Execution Strategy
### Parallel Execution Waves
Wave 1: Tasks 1, 2, 3 can run in parallel.
Wave 2: Tasks 4, 5, 6 can run after Wave 1.
Wave 3: Task 7 consolidates and ranks final recommendations.
Wave 4: Tasks 8 and 9 update the LLM Wiki and lock the experiment-plan addendum.

### Dependency Matrix
| Task | Depends On | Blocks |
| --- | --- | --- |
| 1. Dataset Shape Audit | none | 4, 7 |
| 2. Current Baseline Inventory | none | 4, 7 |
| 3. External Method Research | none | 5, 7 |
| 4. Feasibility Matrix | 1, 2 | 6, 7 |
| 5. Method Shortlist | 3 | 6, 7 |
| 6. Benchmark Design | 4, 5 | 7 |
| 7. Final Recommendation Report | 1, 2, 4, 5, 6 | 8, 9 |
| 8. LLM Wiki Knowledge Base Update | 3, 5, 7 | 9 |
| 9. Experiment Plan Addendum | 6, 7, 8 | Final Verification |

## TODOs
- [ ] 1. Dataset Shape Audit

  **What to do**: Create `.omo/evidence/recommender-methods/dataset-shape-report.md`. Inspect actual local artifacts, not only docs. Record row counts, columns, label unit, split unit, candidate width, feature groups, and leakage risks for both recommender systems.
  **Must NOT do**: Do not mutate data, rebuild graph, or run training.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 4, 7 | Blocked By: none

  **References**:
  - `GNN_Neural_Network/AGENTS.md` - hobby recommender scope and evaluation rules.
  - `experiments/persona_similarity/AGENTS.md` - similar-persona scope, candidate width, feature, and manual-review rules.
  - `GNN_Neural_Network/DATASET_EXPLAIN.md` - raw 1M dataset and hobby leakage caveats.
  - `experiments/persona_similarity/DATASET_EXPLAIN.md` - directed pair row shape and current topK guidance.
  - `GNN_Neural_Network/data/person_hobby_edges.csv` - local hobby edge artifact.
  - `GNN_Neural_Network/data/person_context.csv` - local hobby context artifact.
  - `experiments/persona_similarity/artifacts/datasets/candidate_pairs.parquet` - similar-persona candidate rows.
  - `experiments/persona_similarity/artifacts/datasets/pair_features.parquet` - structured pair features.
  - `experiments/persona_similarity/artifacts/datasets/pair_features_with_text.parquet` - text-augmented pair features.

  **Acceptance Criteria**:
  - [ ] Run:
    ```powershell
    .\.venv\Scripts\python.exe -c "from pathlib import Path; import csv; paths=[Path('GNN_Neural_Network/data/person_hobby_edges.csv'),Path('GNN_Neural_Network/data/person_context.csv')]; [print(f'{p}: rows={sum(1 for _ in p.open(encoding=\"utf-8-sig\"))-1}') for p in paths]"
    ```
  - [ ] Run:
    ```powershell
    .\.venv\Scripts\python.exe -c "from pathlib import Path; import pyarrow.parquet as pq; paths=['experiments/persona_similarity/artifacts/datasets/candidate_pairs.parquet','experiments/persona_similarity/artifacts/datasets/pair_features.parquet','experiments/persona_similarity/artifacts/datasets/pair_features_with_text.parquet','experiments/persona_similarity/artifacts/datasets/persona_texts.parquet']; [print(f'{p}: rows={pq.ParquetFile(p).metadata.num_rows}, cols={len(pq.ParquetFile(p).schema_arrow.names)}') for p in paths if Path(p).exists()]"
    ```
  - [ ] Report includes the known actual shapes: hobby CSVs 50,000 rows each; similar-persona `candidate_pairs`, `pair_features`, and `pair_features_with_text` 2,500,000 rows each; `persona_texts` 50,000 rows.

  **QA Scenarios**:
  ```text
  Scenario: Happy path dataset shape reproduction
    Tool: tmux
    Steps: tmux new-session -d -s ulw-qa-shape; tmux send-keys -t ulw-qa-shape "<the two PowerShell python commands above>" Enter; tmux capture-pane -pS -200 -E - -t ulw-qa-shape > .omo/evidence/recommender-methods/task-1-shape-transcript.txt
    Expected: transcript contains 50,000 hobby rows and 2,500,000 similar-persona pair rows.
    Evidence: .omo/evidence/recommender-methods/task-1-shape-transcript.txt

  Scenario: Missing artifact boundary check
    Tool: bash
    Steps: test -f GNN_Neural_Network/data/person_hobby_edges.csv && test -f experiments/persona_similarity/artifacts/datasets/candidate_pairs.parquet
    Expected: exit code 0.
    Evidence: .omo/evidence/recommender-methods/task-1-artifact-check.txt
  ```

  **Commit**: NO | Message: `docs(recommender): document dataset shape investigation` | Files: `.omo/evidence/recommender-methods/dataset-shape-report.md`

- [ ] 2. Current Baseline Inventory

  **What to do**: Create `.omo/evidence/recommender-methods/current-baseline-inventory.md`. Summarize current default behavior, offline experimental winners, rejected experiments, metrics, and promotion blockers.
  **Must NOT do**: Do not reinterpret promotion decisions beyond recorded artifacts.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 4, 7 | Blocked By: none

  **References**:
  - `GNN_Neural_Network/EXPERIMENTS.md` - Phase 2.5 hobby baseline and rejected experiments.
  - `GNN_Neural_Network/artifacts/ranker_eval_metrics.json` - current hobby ranker metrics.
  - `experiments/persona_similarity/artifacts/experiment_run_summary.md` - current similar-persona decision summary.
  - `experiments/persona_similarity/artifacts/experiment_decisions.json` - current experimental status and promotion gate.

  **Acceptance Criteria**:
  - [ ] Inventory states hobby default as `popularity + cooccurrence` candidate generation plus LightGBM ranker, not LightGCN production default.
  - [ ] Inventory states hobby unresolved issue as ranking/diversity/coverage, because candidate recall@50 is already high.
  - [ ] Inventory states similar-persona production behavior as FastRP/KNN `SIMILAR_TO` and offline structured LambdaRank as not promoted due manual review.
  - [ ] Inventory lists at least 3 rejected/deferred prior paths: KURE MMR, DPP diversity rerank, LambdaRank smoke for hobby; structured+text and diversity rerank not promoted for similar-persona.

  **QA Scenarios**:
  ```text
  Scenario: Baseline facts present
    Tool: bash
    Steps: rg -n "popularity \\+ cooccurrence|candidate recall|FastRP/KNN|manual review|structured LambdaRank" .omo/evidence/recommender-methods/current-baseline-inventory.md
    Expected: all required terms are found.
    Evidence: .omo/evidence/recommender-methods/task-2-baseline-rg.txt

  Scenario: No cross-scope metric merge
    Tool: bash
    Steps: rg -n "Hobby.*NDCG@5|similar.*Recall@10" .omo/evidence/recommender-methods/current-baseline-inventory.md; test $? -ne 0
    Expected: no direct cross-system metric comparison pattern is found.
    Evidence: .omo/evidence/recommender-methods/task-2-scope-check.txt
  ```

  **Commit**: NO | Message: `docs(recommender): inventory current baselines` | Files: `.omo/evidence/recommender-methods/current-baseline-inventory.md`

- [ ] 3. External Method Research

  **What to do**: Create `.omo/evidence/recommender-methods/external-method-research.md`. Research credible recommendation method families and map each to the project data shape. Use primary papers or official docs where possible.
  **Must NOT do**: Do not chase methods that require unavailable real user interaction logs unless marked `reject/defer`.

  **Parallelization**: Can Parallel: YES | Wave 1 | Blocks: 5, 7 | Blocked By: none

  **References**:
  - RecBole official docs/model list: `https://recbole.io/docs/`, `https://recbole.io/model_list.html`
  - LightGCN paper: `https://arxiv.org/abs/2002.02126`
  - XSimGCL paper: `https://arxiv.org/abs/2209.02544`
  - KGAT paper: `https://arxiv.org/abs/1905.07854`
  - Neural Collaborative Filtering paper: `https://arxiv.org/abs/1708.05031`
  - Wide & Deep paper: `https://arxiv.org/abs/1606.07792`
  - YouTube two-stage DNN recommendation paper: `https://research.google.com/pubs/archive/45530.pdf`
  - Factorization Machines paper reference: `https://www.gabormelli.com/RKB/2010_FactorizationMachines`

  **Acceptance Criteria**:
  - [ ] Research covers at least: graph CF, graph contrastive learning, KG-aware recommendation, neural CF, FM/FFM/DeepFM-style tabular interactions, two-tower retrieval, Wide&Deep ranking, content/text embedding retrieval, and diversity reranking.
  - [ ] Each method states required dataset shape and whether this project has it now.
  - [ ] Each method is assigned one of: `try-first`, `benchmark`, `manual-review-only`, `defer`, `reject`.

  **QA Scenarios**:
  ```text
  Scenario: Source-backed method list
    Tool: bash
    Steps: rg -n "LightGCN|XSimGCL|KGAT|Neural Collaborative Filtering|Wide & Deep|two-tower|Factorization Machines|RecBole" .omo/evidence/recommender-methods/external-method-research.md
    Expected: every named method family is present.
    Evidence: .omo/evidence/recommender-methods/task-3-method-rg.txt

  Scenario: Dataset fit classification
    Tool: bash
    Steps: rg -n "required_shape|decision|try-first|defer|reject" .omo/evidence/recommender-methods/external-method-research.md
    Expected: required classification fields are present.
    Evidence: .omo/evidence/recommender-methods/task-3-fit-rg.txt
  ```

  **Commit**: NO | Message: `docs(recommender): research alternative methods` | Files: `.omo/evidence/recommender-methods/external-method-research.md`

- [ ] 4. Feasibility Matrix

  **What to do**: Create `.omo/evidence/recommender-methods/feasibility-matrix.md`. Map current dataset shape and baseline gaps to method feasibility separately for hobby and similar-persona.
  **Must NOT do**: Do not rank by paper popularity; rank by local fit, metric target, leakage risk, and artifact readiness.

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 6, 7 | Blocked By: 1, 2

  **References**:
  - `.omo/evidence/recommender-methods/dataset-shape-report.md`
  - `.omo/evidence/recommender-methods/current-baseline-inventory.md`
  - `GNN_Neural_Network/AGENTS.md`
  - `experiments/persona_similarity/AGENTS.md`

  **Acceptance Criteria**:
  - [ ] Hobby matrix includes at least 5 candidates and identifies the top gap as ranking/diversity, not candidate recall.
  - [ ] Similar-persona matrix includes at least 5 candidates and identifies manual-review/weak-label dependence as the promotion blocker.
  - [ ] Matrix includes `data_ready`, `requires_new_export`, `requires_text_leakage_audit`, `expected_metric_target`, `cost`, and `decision`.

  **QA Scenarios**:
  ```text
  Scenario: Matrix has required fields
    Tool: bash
    Steps: rg -n "data_ready|requires_new_export|requires_text_leakage_audit|expected_metric_target|cost|decision" .omo/evidence/recommender-methods/feasibility-matrix.md
    Expected: all required fields are found.
    Evidence: .omo/evidence/recommender-methods/task-4-fields.txt

  Scenario: System separation check
    Tool: bash
    Steps: rg -n "Person -> Hobby|Person -> Person" .omo/evidence/recommender-methods/feasibility-matrix.md
    Expected: both sections exist.
    Evidence: .omo/evidence/recommender-methods/task-4-boundaries.txt
  ```

  **Commit**: NO | Message: `docs(recommender): build method feasibility matrix` | Files: `.omo/evidence/recommender-methods/feasibility-matrix.md`

- [ ] 5. Method Shortlist

  **What to do**: Create `.omo/evidence/recommender-methods/alternative-method-shortlist.md`. Convert external research into a practical shortlist with explicit recommendations.
  **Must NOT do**: Do not recommend reinforcement learning, sequential session models, or online bandits as primary work because the dataset lacks real temporal/interactive logs.

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 6, 7 | Blocked By: 3

  **References**:
  - `.omo/evidence/recommender-methods/external-method-research.md`
  - `GNN_Neural_Network/EXPERIMENTS.md`
  - `experiments/persona_similarity/artifacts/experiment_run_summary.md`

  **Acceptance Criteria**:
  - [ ] Shortlist recommends `try-first` methods separately:
    - Hobby: feature interaction/tabular reranker alternatives and diversity-aware objective/rerank only if accuracy gates are preserved.
    - Similar-persona: manual-review-first text-driven reranking and possibly KG-aware/embedding retrieval only as offline comparison.
  - [ ] Shortlist rejects or defers methods requiring missing session/order/feedback logs.
  - [ ] Shortlist includes explicit `why not now` for at least 3 tempting but risky methods.

  **QA Scenarios**:
  ```text
  Scenario: Try-first decisions exist
    Tool: bash
    Steps: rg -n "try-first|why not now|defer|reject" .omo/evidence/recommender-methods/alternative-method-shortlist.md
    Expected: all decision categories are present.
    Evidence: .omo/evidence/recommender-methods/task-5-decisions.txt

  Scenario: No missing-log method promoted
    Tool: bash
    Steps: rg -n "reinforcement learning.*try-first|session.*try-first|bandit.*try-first" .omo/evidence/recommender-methods/alternative-method-shortlist.md; test $? -ne 0
    Expected: no missing-log method is marked try-first.
    Evidence: .omo/evidence/recommender-methods/task-5-no-bad-promotion.txt
  ```

  **Commit**: NO | Message: `docs(recommender): shortlist alternative recommender methods` | Files: `.omo/evidence/recommender-methods/alternative-method-shortlist.md`

- [ ] 6. Benchmark Design

  **What to do**: Create `.omo/evidence/recommender-methods/benchmark-design.md`. Define exact benchmark tracks, commands, metrics, artifacts, cache rules, and promotion gates for the shortlisted methods.
  **Must NOT do**: Do not prescribe production integration.

  **Parallelization**: Can Parallel: YES | Wave 2 | Blocks: 7 | Blocked By: 4, 5

  **References**:
  - `GNN_Neural_Network/scripts/evaluate_ranker.py`
  - `GNN_Neural_Network/scripts/train_ranker.py`
  - `GNN_Neural_Network/scripts/evaluate_reranker.py`
  - `experiments/persona_similarity/scripts/evaluate_fastrp_baseline.py`
  - `experiments/persona_similarity/scripts/evaluate_deterministic_baseline.py`
  - `experiments/persona_similarity/scripts/train_lambdarank.py`
  - `experiments/persona_similarity/scripts/evaluate_lambdarank.py`
  - `experiments/persona_similarity/scripts/evaluate_diversity_rerank.py`

  **Acceptance Criteria**:
  - [ ] Benchmark design includes exact commands for existing baseline reproduction before any new method comparison.
  - [ ] Hobby gates include Recall@K, NDCG@K, candidate recall, coverage, novelty, runtime, and qualitative hobby quality.
  - [ ] Similar-persona gates include NDCG@K, explanation coverage, strong-reason coverage, low-information dominance, diversity, runtime, model size, and manual review.
  - [ ] Benchmark design states that `.venv314t` is optional and only for already-exported parquet/npz acceleration with metadata.

  **QA Scenarios**:
  ```text
  Scenario: Benchmark command presence
    Tool: bash
    Steps: rg -n "\\.\\\\\\.venv\\\\Scripts\\\\python\\.exe|evaluate_ranker|train_lambdarank|evaluate_lambdarank" .omo/evidence/recommender-methods/benchmark-design.md
    Expected: baseline and reranker commands are present.
    Evidence: .omo/evidence/recommender-methods/task-6-commands.txt

  Scenario: Promotion gate coverage
    Tool: bash
    Steps: rg -n "Recall@K|NDCG@K|candidate recall|manual review|low-information|model size|runtime" .omo/evidence/recommender-methods/benchmark-design.md
    Expected: all major metric gates are present.
    Evidence: .omo/evidence/recommender-methods/task-6-gates.txt
  ```

  **Commit**: NO | Message: `docs(recommender): design alternative method benchmarks` | Files: `.omo/evidence/recommender-methods/benchmark-design.md`

- [ ] 7. Final Recommendation Report

  **What to do**: Create `.omo/evidence/recommender-methods/final-recommendation.md`. Synthesize all reports into a concise Korean recommendation: what to try first, what to keep as baseline, what to defer/reject, and what evidence is required before model promotion.
  **Must NOT do**: Do not claim a method is better without benchmark evidence.

  **Parallelization**: Can Parallel: NO | Wave 3 | Blocks: Final Verification | Blocked By: 1, 2, 4, 5, 6

  **References**:
  - `.omo/evidence/recommender-methods/dataset-shape-report.md`
  - `.omo/evidence/recommender-methods/current-baseline-inventory.md`
  - `.omo/evidence/recommender-methods/feasibility-matrix.md`
  - `.omo/evidence/recommender-methods/alternative-method-shortlist.md`
  - `.omo/evidence/recommender-methods/benchmark-design.md`

  **Acceptance Criteria**:
  - [ ] Report includes a ranked list for hobby methods and a separate ranked list for similar-persona methods.
  - [ ] Report includes a `No-Go / Defer` section.
  - [ ] Report includes a one-page decision table suitable for the user to choose next execution.
  - [ ] Report is in Korean, with technical terms preserved where useful.

  **QA Scenarios**:
  ```text
  Scenario: Final report sections
    Tool: bash
    Steps: rg -n "취미 추천|유사 페르소나|No-Go|Defer|다음 실행" .omo/evidence/recommender-methods/final-recommendation.md
    Expected: all final report sections are present.
    Evidence: .omo/evidence/recommender-methods/task-7-sections.txt

  Scenario: No unproven promotion claim
    Tool: bash
    Steps: rg -n "프로덕션.*승격|production.*promoted" .omo/evidence/recommender-methods/final-recommendation.md; test $? -ne 0
    Expected: no unproven promotion language is present.
    Evidence: .omo/evidence/recommender-methods/task-7-no-promotion.txt
  ```

  **Commit**: NO | Message: `docs(recommender): recommend next benchmark tracks` | Files: `.omo/evidence/recommender-methods/final-recommendation.md`

- [ ] 8. LLM Wiki Knowledge Base Update

  **What to do**: Update `docs/llm_wiki/` after the investigation deliverables are produced. Source cards must capture external methods, concept notes must capture local boundary/gate decisions, and experiment notes must capture the final investigation result.
  **Must NOT do**: Do not store raw datasets, checkpoints, credentials, full copied papers, or Neo4j secrets in the wiki.

  **Parallelization**: Can Parallel: YES | Wave 4 | Blocks: 9 | Blocked By: 3, 5, 7

  **References**:
  - `docs/llm_wiki/INDEX.md` - wiki index and required local context.
  - `docs/llm_wiki/raw_sources/2026-06-04-recommender-method-sources.md` - source manifest.
  - `docs/llm_wiki/source_cards/recommender_methods/` - external method cards.
  - `docs/llm_wiki/person_hobby/` - `Person -> Hobby` track.
  - `docs/llm_wiki/persona_similarity/` - `Person -> Person` track.
  - `docs/llm_wiki/concepts/current_recommender_findings.md` - current durable findings.
  - `docs/llm_wiki/concepts/experiment_decision_gates.md` - promotion and no-go gates.
  - `docs/llm_wiki/templates/experiment_note_template.md` - note structure.

  **Acceptance Criteria**:
  - [ ] `docs/llm_wiki/INDEX.md` links required local context, source cards, concept notes, and experiment notes.
  - [ ] `docs/llm_wiki/INDEX.md` links separate `person_hobby/` and `persona_similarity/` experiment track folders.
  - [ ] Source cards exist for RecBole, LightGCN/XSimGCL, KGAT, neural retrieval/ranking, Factorization Machines, text embedding retrieval, and diversity reranking.
  - [ ] Concept notes explicitly separate `Person -> Hobby` and `Person -> Person`.
  - [ ] `LOG.md` records the wiki update date and scope.

  **QA Scenarios**:
  ```text
  Scenario: Wiki index coverage
    Tool: bash
    Steps: rg -n "Required Local Context|Source Cards|Concept Notes|Experiment Notes|Experiment Track Folders|person_hobby|persona_similarity|Person -> Hobby|Person -> Person" docs/llm_wiki/INDEX.md
    Expected: all wiki index sections and recommender boundaries are present.
    Evidence: .omo/evidence/recommender-methods/task-8-wiki-index.txt

  Scenario: Source-card coverage
    Tool: bash
    Steps: rg -n "RecBole|LightGCN|XSimGCL|KGAT|Factorization Machines|Text Embedding|Diversity" docs/llm_wiki/source_cards/recommender_methods docs/llm_wiki/concepts
    Expected: all tracked method families are present in wiki pages.
    Evidence: .omo/evidence/recommender-methods/task-8-source-cards.txt
  ```

  **Commit**: NO | Message: `docs(llm-wiki): add recommender research wiki` | Files: `docs/llm_wiki/**`

- [ ] 9. Experiment Plan Addendum

  **What to do**: Add a durable experiment-plan note that turns the final recommendation into executable tracks. The addendum must identify the first experiment branch for hobby recommendation and the first validation branch for similar-persona recommendation.
  **Must NOT do**: Do not start training or execute long-running benchmarks in the planning task.

  **Parallelization**: Can Parallel: NO | Wave 4 | Blocks: Final Verification | Blocked By: 6, 7, 8

  **References**:
  - `docs/llm_wiki/experiment_notes/2026-06-04-recommender-experiment-plan.md`
  - `.omo/evidence/recommender-methods/benchmark-design.md`
  - `.omo/evidence/recommender-methods/final-recommendation.md`
  - `docs/llm_wiki/concepts/recommender_method_shortlist.md`

  **Acceptance Criteria**:
  - [ ] The experiment-plan note names the first hobby branch as ranker feature-interaction or diversity-aware ranking, not a new retriever by default.
  - [ ] The experiment-plan note names the first similar-persona branch as manual-review-first validation of current structured/text results.
  - [ ] The note lists evidence artifacts for dataset-shape, baseline inventory, method shortlist, benchmark design, and final recommendation.
  - [ ] The note says no production promotion is allowed from this plan alone.

  **QA Scenarios**:
  ```text
  Scenario: Experiment branches are explicit
    Tool: bash
    Steps: rg -n "ranker|diversity|manual review|FastRP|production" docs/llm_wiki/person_hobby/experiment_plan.md docs/llm_wiki/persona_similarity/experiment_plan.md docs/llm_wiki/experiment_notes/2026-06-04-recommender-experiment-plan.md
    Expected: first experiment branches and production boundary are present.
    Evidence: .omo/evidence/recommender-methods/task-9-experiment-note.txt

  Scenario: Plan links wiki addendum
    Tool: bash
    Steps: rg -n "LLM Wiki|docs/llm_wiki|Experiment Plan Addendum|recommender-experiment-plan" .omo/plans/recommender-alternative-methods-investigation.md
    Expected: the investigation plan links the wiki addendum.
    Evidence: .omo/evidence/recommender-methods/task-9-plan-link.txt
  ```

  **Commit**: NO | Message: `docs(recommender): add experiment plan wiki addendum` | Files: `.omo/plans/recommender-alternative-methods-investigation.md`, `docs/llm_wiki/experiment_notes/2026-06-04-recommender-experiment-plan.md`

## Final Verification Wave
> ALL must APPROVE. Present consolidated results to user and get explicit okay before completing any later implementation/promotion work.
- [ ] F1. Plan Compliance Audit
  - Verify every task has references, acceptance criteria, QA scenarios, and commit decision.
  - Command:
    ```powershell
    .\.venv\Scripts\python.exe -c "from pathlib import Path; p=Path('.omo/plans/recommender-alternative-methods-investigation.md'); t=p.read_text(encoding='utf-8'); required=['References','Acceptance Criteria','QA Scenarios','Commit','Final Verification Wave']; print({k: t.count(k) for k in required})"
    ```
- [ ] F2. Scope Fidelity Check
  - Confirm no source-code files were changed by the investigation execution unless explicitly authorized later.
  - Command: `git diff --stat -- . ':!.omo'`
- [ ] F3. Evidence Artifact Check
  - Confirm all deliverables exist.
  - Command:
    ```powershell
    @(
      ".omo/evidence/recommender-methods/dataset-shape-report.md",
      ".omo/evidence/recommender-methods/current-baseline-inventory.md",
      ".omo/evidence/recommender-methods/alternative-method-shortlist.md",
      ".omo/evidence/recommender-methods/benchmark-design.md",
      ".omo/evidence/recommender-methods/final-recommendation.md",
      "docs/llm_wiki/INDEX.md",
      "docs/llm_wiki/experiment_notes/2026-06-04-recommender-experiment-plan.md",
      "docs/llm_wiki/person_hobby/experiment_plan.md",
      "docs/llm_wiki/persona_similarity/experiment_plan.md"
    ) | ForEach-Object { if (-not (Test-Path $_)) { throw "missing $_" } }
    ```
- [ ] F4. Real Manual QA
  - Run the Task 1 dataset-shape tmux scenario and save transcript.
  - Run `rg` checks from Tasks 2-9 and save outputs.
  - Confirm final report has no production-promotion claim.

## Commit Strategy
- Default: no commit. The investigation deliverables live under `.omo/evidence/recommender-methods/`.
- If the user asks for a commit after execution, create one conventional commit:
  - `docs(recommender): investigate alternative recommendation methods`
- Do not commit source-code changes unless a later execution plan explicitly implements benchmarks.

## Success Criteria
- The user can open `.omo/evidence/recommender-methods/final-recommendation.md` and see a ranked, dataset-shape-grounded recommendation of alternative methods.
- The user can open `docs/llm_wiki/INDEX.md` and navigate the recommender research source cards, concept notes, and experiment-plan note.
- Every recommendation is traceable to local dataset shape and current baseline constraints.
- Every proposed benchmark has exact commands, metrics, and promotion blockers.
- No production behavior changes.
