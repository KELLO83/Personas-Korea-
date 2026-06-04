# GNN Recommender Tasks

This file tracks executable tasks for `GNN_Neural_Network/` experiments. For requirements and design decisions, use `PRD.md`. For historical v2 reranker checklist details, use `CHECKLIST_GNN_Reranker_v2.md`.

## Phase 6 Closure Status (2026-05-20)

Current best Phase 6 candidate:

```text
run_id = phase6_domain_text_hard1_aliases_full_validation
Stage1 = popularity + cooccurrence
Stage2 = LightGBM(num_leaves=31)
negative_sampling = neg_ratio=4, hard_ratio=1.0
candidate_text_builder = name_plus_aliases
include_text_embedding_feature = true
include_domain_text_embedding_features = true
KURE semantic Stage1 provider = false
topic_calibration = optional lambda=0.02 post-ranker
```

### Completed Phase 6 Runs

- [x] Implement Phase 6 experiment scaffolding and manifest validation.
  - [x] `gnn_recommender/phase6.py`
  - [x] `scripts/build_phase6_experiment_manifest.py`
  - [x] `tests/test_phase6.py`
- [x] Implement Stage2 cross features as opt-in LightGBM columns.
  - [x] `age_group_region_cross_fit`
  - [x] `occupation_region_cross_fit`
  - [x] `demographic_text_cross_fit`
  - [x] `--include-phase6-cross-features`
  - [x] Result: rejected as standalone signal; full and pilot importances stayed `0.0`.
- [x] Implement topic-calibrated cached-ranker evaluator.
  - [x] `scripts/evaluate_topic_calibrated_ranker.py`
  - [x] `lambda=0.02` validation/test runs completed for the best alias model.
  - [x] Result: optional post-ranker only; slight metric gain, not a core model replacement.
- [x] Run no-text baseline controls.
  - [x] `phase6_baseline_m500`
  - [x] `phase6_baseline_m2000`
- [x] Run domain text feature pilots.
  - [x] `phase6_domain_text_cross_m500`
  - [x] `phase6_domain_text_cross_full_validation`
  - [x] Result: large NDCG improvement, recall below validation recall leader.
- [x] Combine domain text with old hard-negative sampling strength.
  - [x] `phase6_domain_text_hard1_full_validation`
  - [x] Result: improved over `phase6_domain_text_cross_full_validation`.
- [x] Combine domain text, hard negatives, and alias candidate text.
  - [x] `phase6_domain_text_hard1_aliases_full_validation`
  - [x] validation Recall@10 `0.732523`, NDCG@10 `0.480773`
  - [x] test Recall@10 `0.710786`, NDCG@10 `0.464645`
  - [x] topic-calibrated test Recall@10 `0.711338`, NDCG@10 `0.464943`
  - [x] Result: current best tested Phase 6 candidate; keep validation-recall caveat.
- [x] Test KURE semantic Stage1 with the best alias/domain-text recipe.
  - [x] `phase6_kure_stage1_domain_text_hard1_aliases_full_validation`
  - [x] Result: rejected for this recipe. Validation oracle_recall@10 fell to `0.794326`, and final Recall@10 fell to `0.725338`.

### Phase 6 Decision Checklist

- [x] Keep Stage1 default as `popularity + cooccurrence`.
- [x] Do not promote KURE semantic Stage1 for the current recipe.
- [x] Keep `candidate_k=50`; do not use the KURE semantic full-provider pool as default.
- [x] Keep domain text scalar similarity features; they are the main effective Phase 6 signal.
- [x] Do not claim validation Recall@10 absolute SOTA.
- [x] Record that the Phase 6 alias/domain-text model is the strongest stored test artifact for both Recall@10 and NDCG@10 in the scanned artifacts.
- [x] Record experiment results in `artifacts/experiments/phase6_hobby_validation_summary.md`.
- [x] Update `PRD.md` with the 2026-05-20 Phase 6 result and caveats.
- [x] Update `TASKS.md` with completed/rejected/follow-up Phase 6 status.

### Remaining Follow-Up Tasks

- [ ] Decide whether `candidate_text_builder=name_plus_aliases` is acceptable for production governance.
  - Alias text improved metrics, but older Track D notes flagged metadata provenance and taxonomy/canonicalization bias risk.
  - Do not wire this into production API until the source-field policy is approved.
- [ ] If production governance rejects aliases, rerun the best hard-negative/domain-text recipe with `name_only` and treat it as the deployable candidate.
- [ ] Add a machine-readable Phase 6 decision record if the project keeps using `experiment_decisions.json` as the canonical default registry.
- [ ] Optionally run cold-start/segment audits for the best alias/domain-text candidate before API integration.
- [ ] Update README or deployment notes only after a production default decision is made.

## Current Data And KURE Preconditions (2026-05-05)

This section is the executable blocker list before any `include_text_embedding_feature=true` KURE-v1 feature experiment. Use `KURE-v1` as the canonical model name; older `KRUE` wording in historical artifacts means KURE-v1.

### Local Data Reality Lock

- [x] Local edge file inspected: `GNN_Neural_Network/data/person_hobby_edges.csv`
- [x] Local context file inspected: `GNN_Neural_Network/data/person_context.csv`
- [x] Current local edge rows recorded: `50,000`
- [x] Current local context rows recorded: `50,000`
- [x] Current local persons with hobby edges recorded: `17,907`
- [x] Current local unique raw hobby strings recorded: `49,558`
- [x] Current local average hobbies per person recorded: `2.79`
- [x] Decision recorded: raw hobby phrases are not stable item IDs and must not be used directly for promotion-grade GNN/LightGCN item training.
- [x] Decision recorded: GNN/LightGCN is auxiliary/analysis provider under this data shape; current default remains `popularity + cooccurrence -> LightGBM`.

### Mandatory Blockers Before KURE Text Feature Ablation

- [x] **50K canonical/fallback baseline closure**
  - [x] rebuild or verify `raw_hobby_phrase -> canonical/fallback item` mapping for local 50K data
  - [x] verify `rare_item_policy=keep_with_fallback` candidate_recall@50 drift is within `-0.01` (closed by prepare-only artifact refresh; no dropped edges, fallback preserved)
  - [x] run closed Phase 2.5 config on the local 50K baseline
  - [x] record validation metrics and status artifact
  - [x] update `artifacts/experiment_decisions.json`
  - [x] update `artifacts/experiment_run_summary.md`

- [x] **Phase 5-B2 feature-balance closure**
  - [x] complete `phase5_b2_feature_balance/feature_fraction_0_7/validation_metrics.json`
  - [x] replace any `candidates_done`-only status with final gated validation status
  - [x] run `feature_fraction=0.8` probe only if still required under the same baseline
  - [x] compare against the closed Phase 2.5 default
  - [x] record accept/reject/blocked decision in experiment decision artifacts

- [x] **Taxonomy over-merge risk closure**
  - [x] inspect whether over-merged canonical/category mappings concentrate top-k recommendations
  - [x] record data-quality decision in artifacts even if no default changes
  - [x] document whether taxonomy work is a blocker or follow-up for KURE text feature experiments

- [x] **Cold-start baseline closure**
  - [x] define cold-start subset as `known_hobbies <= 1`
  - [x] compute closed Phase 2.5 default cold-start Recall@10 and NDCG@10
  - [x] compute closed Phase 2.5 default cold-start Coverage@10, Novelty@10, and ILD@10
  - [x] persist cold-start metrics in validation/test artifacts or a dedicated baseline artifact
  - [x] use these metrics as comparison baseline for KURE text feature ablation

### KURE Text Feature Ablation Scope

Do not confuse this with the completed KURE dense MMR sweep.

- [x] KURE dense MMR sweep status: completed, NO-GO, default `MMR=false` unchanged
- [x] KURE text embedding feature ablation status: early runs were rejected/blocked after corrected audit and fallback reruns, but the later current-locked same-split comparison promoted KURE Stage2. `kure_text_feature_001` was disabled by leakage gate, `kure_text_feature_002_context_coverage_gate` was not promoted due low mapped context coverage, and `kure_text_feature_003_full_ranker_fallback` regressed on validation Recall@10 below the closed Phase 2.5 default.
- [x] KURE text feature run must set `include_text_embedding_feature=true`
- [x] KURE text feature run must record `persona_hobby_semantic_sim` or `text_embedding_similarity` in the ranker feature policy
- [x] `mask_holdout_hobbies()` must run before encoding persona text
- [x] `post_mask_leakage_audit()` must be persisted
- [x] leakage-audit failure must mark the run `disabled` and exclude it from metric comparison
- [x] train/eval feature construction must be identical
- [x] missing persona context counted as coverage miss, not leakage failure
- [x] missing-context Stage 2 fallback design evaluated
- [x] test split may run only for a validation-selected winner

### KURE Semantic Stage1 Candidate Provider Scope

> Goal: test the user's proposed next path, `popularity + cooccurrence + KURE semantic candidates -> LightGBM`, as a gated opt-in Stage1 candidate-generation experiment.

- [x] Scope approved as a new gated experiment; it is not part of the current default path.
- [x] Default remains `popularity + cooccurrence -> LightGBM`, `MMR=false`, `include_text_embedding_feature=false`.
- [x] Add explicit guardrail config/CLI opt-in for `allow_stage1_kure_provider=true`.
- [x] Implement Stage1 `kure_semantic` provider without changing existing provider defaults.
- [x] Ensure candidate-pool cache keys include provider list and KURE model/revision/fingerprint metadata.
- [x] Use leakage-safe masked persona text before semantic candidate scoring.
- [x] Use CUDA automatically when available and CPU fallback otherwise.
- [x] Keep evaluation CPU default at `max(1, os.cpu_count() - 2)` and show progress during embedding/scoring/features/ranking.
- [x] Train a separate opt-in LightGBM artifact for the KURE Stage1 candidate pool.
- [x] Run validation first and compare against closed Phase 2.5 SOTA.
- [x] Run test only if validation passes the documented promotion gate. Result: validation failed, so test was skipped.
- [x] Record final decision in `experiment_decisions.json` and `experiment_run_summary.md`.

### Embedding Follow-Up Priority

- [x] KURE-v1 embedding decision is split by role:
  - KURE dense MMR: NO-GO.
  - KURE Stage2 text feature: PROMOTED on the current data/split, then SUPERSEDED by Snowflake-ko Stage2.
  - KURE Stage1 semantic provider: validation failed because candidate_recall@50 regressed materially.
- [x] Current production default and accuracy SOTA is `popularity + cooccurrence -> LightGBM(num_leaves=31) + E5-small-ko-v2 text_embedding_similarity + E5-small domain similarities`, with `MMR=false` and `kure_semantic=false`.
- [x] Snowflake-ko single-feature Stage2 is now the previous accuracy reference, not the current top model.
- [x] Other embedding models are worth testing as Stage2 features because KURE Stage2 improved the no-text baseline and Snowflake-ko further improved KURE. They remain lower priority for Stage1 candidate generation.
- [x] `kure_text_feature_005_domain_tagged_20k_cpu10_test_matrix_retry` completed on test with progress enabled and CPU thread count 10.
  - [x] Test Recall@10 `0.617482`, NDCG@10 `0.386258`.
  - [x] Delta vs its Stage1 baseline: Recall@10 `+0.047711`, NDCG@10 `+0.029900`.
  - [x] Decision was superseded by the locked same-current-data comparison below; KURE Stage2 is now selected for the current split.
- [x] Final KURE follow-up: run a validation-only matched-control experiment under the same current code/split/candidate pool:
  - [x] no-text control: validation Recall@10 `0.591692`, NDCG@10 `0.366055`
  - [x] `include_text_embedding_feature=true`: validation Recall@10 `0.634706`, NDCG@10 `0.396559`
  - [x] CPU thread count explicitly set to `10` and progress visible (`--progress-mode on`)
  - [x] KURE selected vs matched no-text control. The later locked same-current-data comparison below is the promotion-grade decision source.
- [x] Build the requested strict comparison script: existing SOTA candidate feature cache + Stage2 KURE feature only.
  - [x] Script: `scripts/train_eval_sota_pool_kure_feature.py`
  - [x] Progress is always shown for reproduction/evaluation/text-prep/embedding/feature-build/training when `--progress-mode on`.
  - [x] CPU thread count was run with `--cpu-thread-count 10`.
  - [x] The script now aborts if SOTA candidate-pool reproduction fails (`candidate_recall@50 < 0.95`) before KURE training/evaluation is allowed.
  - [x] Current repo state is blocked for this strict SOTA-pool comparison: preserved `features_ac22205dddbdfaba.npz` has `9,841` persons, but current `validation_edges.csv` has `10,857` persons; reproduction candidate_recall@50 is only `0.361702`.
  - [x] Decision: strict "closed SOTA candidate pool + Stage2 KURE" promotion-grade evaluation is not valid with the current split artifacts. Do not use the attempted run for default promotion.
- [x] Rerun current-data locked baseline vs KURE Stage2 comparison with SOTA LightGBM recipe (`num_leaves=31`).
  - [x] no-text model: `artifacts/experiments/phase5_c_text_embedding/current_locked_no_text_num_leaves31_cpu10/ranker_model.txt`
  - [x] KURE model: `artifacts/experiments/phase5_c_text_embedding/current_locked_kure_stage2_num_leaves31_cpu10/ranker_model.txt`
  - [x] validation no-text: Recall@10 `0.591876`, NDCG@10 `0.366105`, candidate_recall@50 `0.827669`
  - [x] validation KURE: Recall@10 `0.634706`, NDCG@10 `0.396559`, candidate_recall@50 `0.827669`
  - [x] test no-text: Recall@10 `0.579626`, NDCG@10 `0.360270`, candidate_recall@50 `0.827208`
  - [x] test KURE: Recall@10 `0.617482`, NDCG@10 `0.386258`, candidate_recall@50 `0.827208`
  - [x] Decision: on the current split/candidate pool, KURE Stage2 is selected over the current no-text baseline, but is now superseded by the Snowflake-ko Stage2 winner-only test below.
- [x] Next Stage2 embedding probe: `dragonkue/snowflake-arctic-embed-l-v2.0-ko` as a single validation-only Stage2 text feature ablation against the KURE Stage2 SOTA.
  - [x] Validation completed with `--cpu-thread-count 2` and visible progress; current code now uses the thread backend for feature/evaluation workers.
  - [x] Snowflake validation Recall@10 `0.655430`, NDCG@10 `0.410234`, candidate_recall@50 `0.827669`.
  - [x] Delta vs promoted KURE Stage2 validation: Recall@10 `+0.020724`, NDCG@10 `+0.013675`, candidate_recall@50 `+0.000000`.
  - [x] Decision: Snowflake-ko passed the validation gate and was eligible for winner-only test.
- [x] Snowflake-ko winner-only test executed under the current split/candidate pool.
  - [x] Test artifact: `artifacts/experiments/phase5_c_text_embedding/snowflake_stage2_single_feature_test_thread18/test_metrics.json`
  - [x] Test Snowflake Recall@10 `0.637653`, NDCG@10 `0.402805`, candidate_recall@50 `0.827208`.
  - [x] Delta vs KURE Stage2 test: Recall@10 `+0.020171`, NDCG@10 `+0.016547`, candidate_recall@50 `+0.000000`.
  - [x] Decision: Snowflake-ko Stage2 is promoted as the current accuracy SOTA/reference.
- [x] Final planned Stage2 embedding probe: `dragonkue/multilingual-e5-small-ko-v2`.
  - [x] Kept Stage1, LightGBM params, masking, candidate text, and feature slot identical to KURE/Snowflake runs.
  - [x] Validation E5-small Recall@10 `0.645390`, NDCG@10 `0.404545`, candidate_recall@50 `0.827669`.
  - [x] Test E5-small Recall@10 `0.623837`, NDCG@10 `0.393921`, candidate_recall@50 `0.827208`.
  - [x] Delta vs Snowflake test: Recall@10 `-0.013816`, NDCG@10 `-0.008884`, candidate_recall@50 `+0.000000`.
  - [x] Delta vs KURE test: Recall@10 `+0.006355`, NDCG@10 `+0.007663`, candidate_recall@50 `+0.000000`.
  - [x] Decision: E5-small-ko-v2 single-feature is promoted as the production/efficiency default at that point; it is later superseded by E5-small domain-specific Stage2 below.
- [x] E5-small Stage2 feature-shape probe: split the single cosine feature into domain-specific masked text blocks (`sports`, `arts`, `travel`, `food`, etc.) and compare against the promoted single-feature E5-small default. Also report delta vs Snowflake accuracy reference.
  - [x] Training completed: `e5_domain_features_validation_thread18`, best AUC `0.881034`, best iteration `105`.
  - [x] Validation E5-domain Recall@10 `0.699180`, NDCG@10 `0.448862`, candidate_recall@50 `0.827669`.
  - [x] Test E5-domain Recall@10 `0.680943`, NDCG@10 `0.436665`, candidate_recall@50 `0.827208`.
  - [x] Delta vs E5-small single test: Recall@10 `+0.057106`, NDCG@10 `+0.042744`, candidate_recall@50 `+0.000000`.
  - [x] Delta vs Snowflake single test: Recall@10 `+0.043290`, NDCG@10 `+0.033860`, candidate_recall@50 `+0.000000`.
  - [x] Delta vs Stage1 test: Recall@10 `+0.111173`, NDCG@10 `+0.080306`.
  - [x] Decision: E5-small-ko-v2 domain-specific Stage2 is promoted as the current production default and accuracy SOTA.
- [ ] Next candidate hobby text probe: evaluate candidate text builders (`name_only`, `name_plus_aliases`, `name_plus_category`, `name_plus_short_description`) as a separate Track D validation-only ablation. Do not change the embedding backbone in the same run.
- [ ] Optional E5-small Stage2 rank/margin probe: add within-candidate-pool E5-small percentile/gap features without changing Stage1 candidate generation.
- [ ] Do not run another Stage1 semantic candidate generator experiment without a new PRD/TASKS reopening note, because KURE Stage1 reduced candidate_recall@50 from `0.977645` to `0.794971`.
- [ ] Stop any future Stage2 embedding run at validation if Recall@10/NDCG@10 miss the promoted E5-domain production default gate or candidate_recall@50 regresses materially. Also report deltas against E5-single and Snowflake-single references when relevant.

### Stage2 Embedding Improvement Plan

- [x] Establish the fixed comparison baseline for every follow-up:
  - baseline artifact: `artifacts/experiments/phase5_c_text_embedding/current_locked_num_leaves31_comparison.json`
  - current default model: `artifacts/experiments/phase5_c_text_embedding/e5_domain_features_validation_thread18/ranker_model.txt`
  - previous E5 single-feature model: `artifacts/experiments/phase5_c_text_embedding/e5_small_stage2_single_feature_validation_thread18/ranker_model.txt`
  - previous accuracy reference model: `artifacts/experiments/phase5_c_text_embedding/snowflake_stage2_single_feature_validation_cpu10/ranker_model.txt`
  - Stage1 must remain `popularity + cooccurrence`
  - Stage2 LightGBM recipe must remain `num_leaves=31` unless a separate tuning task is opened
  - Stage2 text feature contract must remain `text_embedding_similarity = cosine(masked persona domain text embedding, candidate hobby text embedding)`
  - persona text path must remain `mask_holdout_hobbies -> post_mask_leakage_audit -> build_domain_tagged_persona_text`
  - CPU thread count should follow the current root resource policy. Full validation now uses thread workers on this laptop to reduce worker memory duplication.
  - progress must be visible for embedding, feature building, training, and evaluation
- [ ] Track A - embedding backbone swap:
  - [x] define Track A control contract: only the embedding model name/revision may change
  - [x] make the Stage2 text feature path accept an explicit embedding model name/revision and write it into cache metadata
    - [x] `train_ranker.py` supports `--text-embedding-model-name`
    - [x] `train_ranker.py` supports `--text-embedding-model-revision`
    - [x] `evaluate_ranker.py` supports `--text-embedding-model-name`
    - [x] `evaluate_ranker.py` supports `--text-embedding-model-revision`
    - [x] `PersonEmbeddingCache` and `HobbyEmbeddingCache` pass model revision to SentenceTransformer loading
    - [x] cache metadata and ranker metadata record model name/revision/preprocessing
  - [x] verify Track A runs keep candidate hobby text builder, masking policy, LightGBM params, split, and candidate pool unchanged
  - [x] record embedding model metadata, device, batch size, cache hit/miss, runtime, and cache fingerprint
  - [x] run `dragonkue/snowflake-arctic-embed-l-v2.0-ko` validation-only with the same candidate pool and same feature slot
    - [x] training completed: `snowflake_stage2_single_feature_validation_cpu10`, best AUC `0.873005`, best iteration `95`
    - [x] leakage audit passed: failed `0`, passed `10857`
    - [x] validation evaluation completed after resuming with `--cpu-thread-count 2`
    - [x] validation Recall@10 `0.655430`, NDCG@10 `0.410234`, candidate_recall@50 `0.827669`
    - [x] Snowflake-ko beat the promoted KURE Stage2 validation gate and was eligible for winner-only test
    - [x] winner-only test completed: Recall@10 `0.637653`, NDCG@10 `0.402805`, candidate_recall@50 `0.827208`
    - [x] Snowflake-ko is promoted as the current accuracy SOTA/reference
  - [x] run `dragonkue/multilingual-e5-small-ko-v2` validation/test to close the planned KURE/Snowflake/E5 backbone comparison set
    - [x] training completed: `e5_small_stage2_single_feature_validation_thread18`, best AUC `0.863216`, best iteration `90`
    - [x] validation Recall@10 `0.645390`, NDCG@10 `0.404545`, candidate_recall@50 `0.827669`
    - [x] test Recall@10 `0.623837`, NDCG@10 `0.393921`, candidate_recall@50 `0.827208`
    - [x] E5-small-ko-v2 is promoted as the current production/efficiency default
  - [x] promote to test only if validation Recall@10 and NDCG@10 beat KURE-v1 Stage2
    - [x] Gate passed for Snowflake-ko validation; winner-only test executed and accepted
- [x] Track B - domain-specific E5-small feature split:
  - [x] define masked domain text builders for sports, arts, travel, food, family, and professional context
  - [x] add feature columns such as `e5_sports_similarity`, `e5_arts_similarity`, `e5_travel_similarity`, and `e5_food_similarity`
  - [x] keep `text_embedding_similarity` as the single-cosine control feature in the same model
  - [x] do not feed raw 384-dimensional E5 vectors into LightGBM; only add scalar cosine features
  - [x] ensure train/eval parity by deriving domain features from persisted `feature_columns`
  - [x] ensure feature/cache metadata separates single-feature and domain-feature runs
  - [x] keep progress visible during domain embedding prewarm, feature building, training, and evaluation
  - [x] smoke test `e5_domain_features_smoke_100_thread18` train/eval completed with 20 feature columns
  - [x] keep the single E5-small Stage2 baseline available as an ablation control
  - [x] promote to test only if validation beats the single-cosine E5-small Stage2 default and candidate_recall@50 is unchanged
  - [x] validation/test completed and promoted as current SOTA/default
- [x] Track C - candidate-pool E5-small rank/margin features:
  - [x] derive `e5_similarity_percentile`, `e5_similarity_rank`, `e5_similarity_gap_to_top`, and `e5_similarity_gap_to_mean` inside each person's fixed candidate pool
  - [x] confirm these features do not add or remove candidates
  - [x] smoke test `e5_domain_rank_margin_smoke_100_thread18` train/eval completed with 24 feature columns
  - [x] full validation completed: `e5_domain_rank_margin_validation_thread18`
    - [x] validation Recall@10 `0.702404`, NDCG@10 `0.449661`, candidate_recall@50 `0.827669`
    - [x] delta vs E5-domain default validation: Recall@10 `+0.003224`, NDCG@10 `+0.000799`
  - [x] winner-only test completed because validation improved both Recall@10 and NDCG@10
    - [x] test Recall@10 `0.682509`, NDCG@10 `0.436354`, candidate_recall@50 `0.827208`
    - [x] delta vs E5-domain default test: Recall@10 `+0.001566`, NDCG@10 `-0.000311`
    - [x] Decision: mixed test result; do not replace the current E5-domain default yet. Keep as `needs_followup`/optional feature candidate.
- [ ] Track D - candidate hobby text expansion:
  - [x] define the candidate hobby text builder interface and persist `candidate_text_builder`
    - [x] supported builders: `name_only`, `name_plus_aliases`, `name_plus_category`, `name_plus_short_description`
    - [x] feature cache key/metadata include `candidate_text_builder`
    - [x] cache miss fallback encodes missing candidate texts so similarity pair coverage stays complete
  - [x] evaluate `hobby_text_name_only` as the explicit control builder through the current E5-domain default
  - [x] evaluate `hobby_text_name_plus_aliases`
    - [x] validation Recall@10 `0.711615`, NDCG@10 `0.461267`, candidate_recall@50 `0.827669`
    - [x] test Recall@10 `0.694207`, NDCG@10 `0.445550`, candidate_recall@50 `0.827208`
    - [x] Decision: metric-positive but not promoted. Excluded from default promotion because alias/name metadata provenance and taxonomy/canonicalization bias risk are not acceptable for the locked default.
  - [x] evaluate `hobby_text_name_plus_category`
    - [x] smoke test completed: `e5_domain_candidate_text_category_smoke_100_thread18`
    - [x] full validation completed: `e5_domain_candidate_text_category_validation_thread18`
    - [x] validation Recall@10 `0.676062`, NDCG@10 `0.424624`, candidate_recall@50 `0.827669`
    - [x] Decision: rejected on validation; test skipped because it is below E5-domain default.
  - [x] evaluate `hobby_text_name_plus_short_description`
    - [x] validation Recall@10 `0.674035`, NDCG@10 `0.426106`, candidate_recall@50 `0.827669`
    - [x] Decision: rejected on validation; local profile has no short description coverage.
  - [x] persist coverage finding: local alias artifact has only 4 aliases / 2 canonical targets, and `hobby_profile.json` has no `short_description` or `description` fields.
  - [x] verify expanded hobby text policy: alias/category/description builders are excluded from default promotion due metadata provenance, leakage, and taxonomy/canonicalization bias risk.
  - [ ] compare validation Recall@10/NDCG@10 against the promoted E5-small-ko-v2 Stage2 default before any test run; also report Snowflake accuracy reference delta
- [ ] Governance for all tracks:
  - [ ] one script per experiment purpose; do not silently batch unrelated experiments
  - [ ] cache keys must include data split, embedding model/revision, preprocessing version, masking policy, and feature columns
  - [ ] cache keys must include candidate text builder version and source-field policy for Track D
  - [ ] Track A and Track D must not be combined until each isolated effect has a recorded validation artifact
  - [ ] decision artifacts must state exactly which variable changed: embedding backbone, domain feature split, rank/margin feature, or candidate text builder
  - [ ] persist `validation_metrics.status.json`, runtime, device, batch/chunk size, cache hit/miss, and peak GPU memory when available
  - [ ] update `experiment_decisions.json`, `experiment_run_summary.md`, `PRD.md`, `TASKS.md`, and `README.md` if a default decision changes

## Phase 6: High-accuracy Hybrid Extension Skeleton

> PRD Phase 6 — This is a validation-first follow-up skeleton. It must not replace the current E5-small-ko-v2 domain-specific Stage2 default unless the promotion gates pass against the fixed baseline.

### Phase 6 Baseline Lock

- [ ] Lock the current production/default baseline before any Phase 6 run:
  - [ ] Stage1: `popularity + cooccurrence`
  - [ ] Stage2: LightGBM `num_leaves=31`
  - [ ] text feature: E5-small-ko-v2 single + domain-specific scalar cosine features
  - [ ] split, candidate pool, masking policy, candidate text builder, feature columns
- [ ] Record baseline artifact paths and metrics in a Phase 6 decision stub.
- [ ] Define one changed variable per Phase 6 experiment.
- [ ] Run validation first; run test only for a validation-selected winner.

### Phase 6 Track 1 - Similar-Person CF Candidate Injection

- [ ] Define a train-split-only similar-person neighbor source.
- [ ] Ensure validation/test holdout hobbies cannot enter neighbor hobby aggregation.
- [ ] Add quota/merge rules for CF-injected candidates without removing existing Stage1 candidates.
- [ ] Persist provider list, quota, candidate counts, and candidate_recall@50.
- [ ] Compare against the locked Stage1 candidate pool before Stage2 training.

### Phase 6 Track 2 - Demographic/Lifestyle Cross Features

- [ ] Define smoothed cross-feature families:
  - [ ] age group x sex hobby fit
  - [ ] occupation x province/district hobby fit
  - [ ] demographic x lifestyle/domain text interaction
- [ ] Apply rare-combination fallback or smoothing before training.
- [ ] Add cross-feature columns as scalar LightGBM features only.
- [ ] Persist feature importance and subgroup metric deltas.
- [ ] Verify cross features do not encode holdout hobby labels.

### Phase 6 Track 3 - Two-Tower Dynamic Scoring

- [ ] Define persona tower input fields from leakage-safe masked persona/domain text.
- [ ] Define hobby tower input fields from canonical hobby text only.
- [ ] Implement or prototype InfoNCE/contrastive pretraining as an opt-in artifact.
- [ ] Export only scalar similarity scores or rank/margin features into Stage2 LightGBM.
- [ ] Do not feed raw dense vectors directly into the default LightGBM feature set.
- [ ] Compare against E5-domain features with identical candidate pool and split.

### Phase 6 Track 4 - User-level Topic Calibration

- [ ] Define user topic distribution source from train-visible hobbies/domain text.
- [ ] Define calibration objective and rerank penalty/bonus.
- [ ] Evaluate accuracy, coverage, novelty, ILD, and category concentration.
- [ ] Reject if Recall@10/NDCG@10 regression exceeds the default promotion tolerance.
- [ ] Persist before/after top-k samples for manual review.

### Phase 6 Pre-implementation 4대 필수 조항

- [ ] Transductive graph split leakage lockout
  - [ ] Build a global blacklist from train/validation/test positive edges for negative sampling.
  - [ ] Add or verify `test_negative_sampling_lockout`.
  - [ ] Persist blacklist/hash metadata in graph-model artifacts.
- [ ] Cold user & cold hobby dual fallback
  - [ ] Define fallback chain: graph model -> E5-domain LightGBM -> popularity/cooccurrence -> taxonomy/category fallback.
  - [ ] Mark `fallback_used` and fallback stage in output metadata.
  - [ ] Test cold-user and cold-hobby paths.
- [ ] Person-wise ranking validation
  - [ ] Use person/group-based validation for ranking experiments.
  - [ ] Ensure all candidate rows for the same person stay in the same split/fold.
  - [ ] Persist split and group metadata.
- [ ] Resource isolation and OOM limits
  - [ ] Cap CPU threads at the project policy limit, default 18.
  - [ ] Cap GNN batch size at or below the PRD limit.
  - [ ] Record device, batch size, runtime, peak memory when available.
  - [ ] Add explicit cleanup for GPU/CPU-heavy loops where applicable.

### Phase 6 Promotion Gate

- [ ] Compare every Phase 6 candidate against the current E5-domain production default.
- [ ] Require validation and winner-only test Recall@10/NDCG@10 improvement before default discussion.
- [ ] Require no candidate_recall@50 regression for candidate-provider changes.
- [ ] Require leakage audit pass and cold-start fallback pass.
- [ ] Update `experiment_decisions.json`, `experiment_run_summary.md`, `PRD.md`, `TASKS.md`, and `README.md` if a default changes.

## Global Execution Policy

- [x] All post-`Phase2.5` default promotion candidates use an accuracy-first hard gate.
- [x] Default promotion hard accuracy gate: `delta_recall@10 >= -0.002`, `delta_ndcg@10 >= -0.002` (vs closed `phase2_5_default`).
- [x] Ranking-collapse exploration may additionally record a non-promoting `diversity_probe` status.
- [x] Diversity probe accuracy gate: `delta_recall@10 >= -0.010`, `delta_ndcg@10 >= -0.010` (vs closed `phase2_5_default`).
- [x] `diversity_probe` status cannot change the default path without a later default-promotion pass.
- [x] Diversity is secondary: at least 2 of `coverage@10`, `novelty@10`, `intra_list_diversity@10` must satisfy minimum gains.
- [x] Diversity minimum gain thresholds
  - `coverage@10`: `+0.025`
  - `novelty@10`: `+0.10`
  - `intra_list_diversity@10`: `+0.02`
- [x] Stability gate baseline: `v2_fallback_count=0`, `candidate_recall@50` drift within tolerance.
- [x] Validation-first + winner-only test for any metric tie-break change.
- [x] Default experiment scope: 10K offline pilot before any full-scale follow-up.
- [x] Phase 5+ evaluations record cold-start subset metrics (`known_hobbies <= 1`) separately from overall metrics.
- [x] Keep artifact governance fixed:
  - `GNN_Neural_Network/artifacts/experiments/<phase>/<run>/...`
  - `validation_metrics.json` + `validation_metrics.status.json` must exist every trial.
  - `test_metrics.json` / `test_metrics.status.json` only for selected winner.

## Phase 2.5: Default Decision Closure

- [x] Regularization tuning completed
  - [x] `num_leaves=31` selected as current best LightGBM setting
  - [x] validation/test metrics recorded
- [x] Negative sampling ablation completed
  - [x] `neg_ratio=4`, `hard_ratio=1.0` selected by validation
  - [x] final test underperformed current default
  - [x] default remains `neg_ratio=4`, `hard_ratio=0.8`
- [x] Source one-hot ablation completed
  - [x] `include_source_features=true` evaluated on validation
  - [x] validation recall/ndcg/coverage below current default
  - [x] default remains `include_source_features=false`
- [x] Category one-hot MMR recorded as NO-GO
  - [x] binary category similarity made lambda sweep ineffective
  - [x] default remains `MMR=false`
- [x] Phase 2.5 default decision closure recorded
  - [x] `artifacts/experiments/phase2_5_default_decision_closure.md`
  - [x] `artifacts/experiment_decisions.json`
  - [x] `artifacts/experiment_run_summary.md`

Closed Phase 2.5 default:

```text
Stage 1 = popularity + cooccurrence
Stage 2 = LightGBM learned ranker
model = artifacts/experiments/phase2_5_num_leaves_31/ranker_model.txt
num_leaves=31
min_data_in_leaf=50
learning_rate=0.05
reg_alpha=0.1
reg_lambda=0.1
neg_ratio=4
hard_ratio=0.8
include_source_features=false
include_text_embedding_feature=false
MMR=false
```

## Phase 5-A: KURE Dense Embedding MMR Re-Evaluation

> Goal: Use the closed Phase 2.5 default as a fixed baseline and test whether KURE-v1 dense hobby embeddings make MMR useful for ranking-collapse mitigation.

### Baseline Lock

- [x] Phase 2.5 closed default confirmed
- [x] validation baseline recorded
  - [x] `Recall@10=0.7390509094604207`
  - [x] `NDCG@10=0.45797028878684237`
  - [x] `coverage@10=0.15555555555555556`
  - [x] `novelty@10=4.584286633989583`
  - [x] `candidate_recall@50=0.9776445483182603`
- [x] test baseline recorded
  - [x] `Recall@10=0.7096839752057718`
  - [x] `NDCG@10=0.447712669317698`
  - [x] `coverage@10=0.15555555555555556`
  - [x] `novelty@10=4.584286633989583`
  - [x] `candidate_recall@50=0.977136469870948`
- [x] baseline artifact paths verified
  - [x] `artifacts/experiments/phase2_5_num_leaves_31/validation_metrics.json`
  - [x] `artifacts/experiments/phase2_5_num_leaves_31/test_metrics.json`

### Implementation Design Before Code

- [x] KURE hobby embedding generation path designed
  - [x] model: `nlpai-lab/KURE-v1`
  - [x] output: L2-normalized dense embedding matrix
- [x] `HobbyEmbeddingCache` reuse policy defined
  - [x] cache directory layout
  - [x] metadata fields: model name, hobby list/hash, embedding dimension, created timestamp
- [x] `evaluate_ranker.py` option plan finalized
  - [x] `--mmr-embedding-method category_onehot|kure`
  - [x] `--embedding-cache-dir <path>`
  - [x] `--embedding-batch-size <int>`
- [x] MMR application scope fixed
  - [x] apply MMR to full `candidate_k=50` pool
  - [x] do not apply MMR only after truncating to top-k
- [x] category one-hot fallback behavior preserved
- [x] KURE load/device/batch policy defined
  - [x] CUDA when available
  - [x] CPU fallback
  - [x] configurable batch size
- [x] existing `sweep_mmr_lambda.py` disposition decided
  - [x] keep as legacy, or
  - [x] refactor to use full candidate pool and KURE method

### Validation Sweep Plan

- [x] lambda candidates confirmed
  - [x] `0.5`
  - [x] `0.7`
  - [x] `0.8`
  - [x] `0.9`
  - [ ] optional `0.3` only if accuracy/diversity curve needs a low-lambda point
- [x] each lambda executed as one validation run
- [x] no test execution until validation winner is selected
- [x] validation failure means test is skipped

#### Validation outcome summary

- [x] `0.5` validation complete (`blocked`)
- [x] `0.7` validation complete (`blocked`)
- [x] `0.8` validation complete (`blocked`)
- [x] `0.9` validation complete (`blocked`)

  - no validation winner passed the promotion gate
  - test runs were skipped intentionally

### Promotion Gates

- [x] accuracy gate finalized
  - [x] `delta_recall@10 >= -0.002` vs closed default
  - [x] `delta_ndcg@10 >= -0.002` vs closed default
- [x] diversity gate finalized
  - [x] at least 2 of these improve: `coverage@10`, `novelty@10`, `intra_list_diversity@10`
  - [x] decide whether `coverage@10` improvement is mandatory for default promotion
- [x] stability gate finalized
  - [x] `v2_fallback_count=0`
  - [x] `candidate_recall@50` remains effectively unchanged
  - [x] KURE embedding cache is reusable

- [x] Result

  - [x] all candidates failed accuracy gate (`recall@10`, `ndcg@10`) and then skipped test
  - [x] diversity/stability gates were secondary after gate fail
  - [x] Phase 5-A final status: `NO-GO`

### Artifacts

- [x] lambda validation output paths finalized
  - [x] `artifacts/experiments/phase5_kure_mmr_lambda_<lambda>/validation_metrics.json`
  - [x] `artifacts/experiments/phase5_kure_mmr_lambda_<lambda>/validation_metrics.status.json`
 - [x] selected lambda test output policy finalized
  - [x] `artifacts/experiments/phase5_kure_mmr_lambda_<lambda>/test_metrics.json` intentionally absent (no winner selected)
  - [x] `artifacts/experiments/phase5_kure_mmr_lambda_<lambda>/test_metrics.status.json` intentionally absent (no winner selected)
- [x] summary artifact planned
  - [x] `artifacts/experiments/phase5_kure_mmr_summary.md`
- [x] decision artifact schema planned
  - [x] `artifacts/experiment_decisions.json` key: `phase5_kure_mmr`
- [x] run summary update planned
  - [x] `artifacts/experiment_run_summary.md`

### Implementation Gate

- [x] `PRD.md` and this `TASKS.md` KURE MMR plan reviewed for consistency
- [x] code-change scope approved
- [x] implementation starts only after the above planning items are accepted

## Phase 5-B: Ranking Collapse Mitigation

> Goal: reduce `v2 LightGBM` top-k popularity concentration while preserving the closed Phase 2.5 baseline.

### Baseline Lock

- [x] `artifacts/experiments/phase2_5_num_leaves_31/validation_metrics.json`
  - `Recall@10=0.7390509094604207`
  - `NDCG@10=0.45797028878684237`
  - `candidate_recall@50=0.9776445483182603`
  - `coverage@10=0.15555555555555556`
  - `novelty@10=4.584286633989583`
- [x] stability reference
  - `v2_fallback_count=0`
  - `candidate_recall@50` drift tolerance for single-run checks

### Execution Policy

- [x] follows single-config policy (validation-first, winner-only testing)
- [x] no blind multi-config ablation without prior Phase 2.5-locked baseline comparison
- [x] one run path per subtask

### Step 1: listwise objective probe (Closed)

- [x] add/enable listwise objective experiment path in ranking pipeline (예: `LambdaRank` 단일 설정)
- [x] run validation passes
  - `smoke_lambdarank_f07`: completed, blocked
  - **결과:** Listwise objective(LambdaRank) 적용 시 ranking collapse(coverage@10 미달) 완화에 실패. 정형 feature 한계 확인.

### Step 2: Text-Embedding Feature Integration Status

The KURE-v1 text embedding feature path is implemented enough for gated ablation, but the completed Phase 5-C experiment did **not** produce a default-promotion winner.

Implementation status:

- [x] `include_text_embedding_feature=true` can be passed through the ranker train/eval path.
- [x] `text_embedding_similarity` can be computed during ranker dataset construction.
- [x] `mask_holdout_hobbies()` runs before encoding persona text.
- [x] `post_mask_leakage_audit()` is persisted and failed runs are excluded from metric comparison.
- [x] Training/evaluation feature construction is aligned.
- [x] Missing persona context is counted as coverage miss, not leakage failure.
- [x] `[ACT]` masking and domain-tagged persona text scaffolding exist in code.

Closed experiment status:

- [x] `kure_text_feature_001`: disabled by leakage gate.
- [x] `kure_text_feature_002_context_coverage_gate`: not promoted due low mapped context coverage.
- [x] `kure_text_feature_003_full_ranker_fallback`: rejected because validation Recall@10 regressed and stayed below the closed Phase 2.5 default.
- [x] No validation winner selected; test artifacts are intentionally absent for KURE text feature runs.
- [x] Default remains `include_text_embedding_feature=false`.

Remaining implementation work before any future text embedding ablation:

- [x] Repair or regenerate split-aligned `person_context.csv` coverage for the current person mapping.
  - [x] `phase5_context_coverage/context_coverage_report.json` records train/validation/test domain-text coverage as `1.0`.
- [x] Add/verify embedding model selection in train/eval config or CLI without changing the Stage2 feature contract.
- [x] Ensure text embedding cache and feature cache keys include embedding model name and revision, not just model family.
- [x] Persist `embedding_model_metadata.json` per run.
- [x] Verify KURE-v1, `dragonkue/snowflake-arctic-embed-l-v2.0-ko`, and `dragonkue/multilingual-e5-small-ko-v2` caches cannot collide.
- [x] Add cold-start metrics to the ranker evaluation artifacts before using text embedding results for follow-up decisions.
  - [x] closed Phase 2.5 validation/test artifacts recorded under `artifacts/experiments/phase2_5_cold_start_baseline/`

Active Stage2 backbone probes:

- [x] Run `dragonkue/snowflake-arctic-embed-l-v2.0-ko` as a validation-only Stage2 feature ablation against the current KURE Stage2 SOTA.
  - [x] Validation Recall@10 `0.655430`, NDCG@10 `0.410234`, candidate_recall@50 `0.827669`.
  - [x] Winner-only test completed: Recall@10 `0.637653`, NDCG@10 `0.402805`, candidate_recall@50 `0.827208`.
  - [x] Status: accuracy SOTA/reference.
- [x] Run `dragonkue/multilingual-e5-small-ko-v2` as the lightweight Stage2 feature ablation.
  - [x] Validation Recall@10 `0.645390`, NDCG@10 `0.404545`, candidate_recall@50 `0.827669`.
  - [x] Test Recall@10 `0.623837`, NDCG@10 `0.393921`, candidate_recall@50 `0.827208`.
  - [x] Status: production/efficiency default.
- [ ] Compare any future backbone against E5-small-ko-v2 production default and Snowflake-ko accuracy reference on overall and cold-start metrics with the same current candidate pool.
- [ ] Record each future result as `rejected`, `experimental`, `needs_followup`, `production_default`, or `accuracy_reference`.
- [x] Run fast KURE-v1 2K pilot after governance hardening.
  - [x] `kure_text_feature_005_domain_tagged_fast_gpu_pilot_2k` showed same-sample validation Recall@10/NDCG@10 gains over the no-text pilot.
  - [x] Decision recorded as `needs_full_validation_followup`, not promoted.
- [x] Run full validation/test follow-up for KURE-v1 domain-tagged text feature after governance hardening.
  - [x] Full validation KURE text Recall@10 `0.634706`, NDCG@10 `0.396559`.
  - [x] Full validation matched no-text control Recall@10 `0.591692`, NDCG@10 `0.366055`.
  - [x] Test KURE text Recall@10 `0.617482`, NDCG@10 `0.386258`.
  - [x] Historical decision was below the older closed Phase 2.5 SOTA, but the later locked same-current-data comparison superseded it and promoted KURE Stage2. Snowflake-ko later superseded KURE Stage2.

### Step 3: Remaining Ranking-Collapse Implementation Plan

Do not run experiments as part of this planning step. Implement or verify the following code paths first:

- [x] Dedicated cold-start metric reporting for `known_hobbies <= 1` in `evaluate_ranker.py` artifacts.
- [x] Phase 5-B2 feature-balance artifact completion path so `candidates_done` cannot be mistaken for final validation status.
- [x] Taxonomy over-merge decision artifact that records whether canonical/category mappings contribute to top-k concentration.
- [ ] XSimGCL train/eval wiring only as a non-default Stage 1 provider experiment path; the existing `XSimGCL` class alone is not a completed experiment.
- [x] Guardrails ensuring KURE, MMR, XSimGCL, and text embedding features are opt-in only and cannot become default without decision artifacts.

### Artifacts

- [x] `artifacts/experiments/phase5_b1_listwise/*/validation_metrics.json`
- [x] `artifacts/experiments/phase5_b1_listwise/*/validation_metrics.status.json`
- [ ] `artifacts/experiments/phase5_b1_listwise/*/test_metrics.json` (winner only; intentionally absent when no validation winner)
- [ ] `artifacts/experiments/phase5_b1_listwise/*/test_metrics.status.json` (winner only; intentionally absent when no validation winner)
- [x] `artifacts/experiments/phase5_b2_feature_balance/*/validation_metrics.json`
- [x] `artifacts/experiments/phase5_b2_feature_balance/*/validation_metrics.status.json`
- [x] `artifacts/experiments/phase5_b2_feature_balance/*/test_metrics.json` (winner only; intentionally absent because no validation winner)
- [x] `artifacts/experiments/phase5_b2_feature_balance/*/test_metrics.status.json` (winner only; intentionally absent because no validation winner)
- [x] `artifacts/experiments/phase5_b3_diversity_rerank/*/validation_metrics.json`
- [x] `artifacts/experiments/phase5_b3_diversity_rerank/*/validation_metrics.status.json`
- [x] `artifacts/experiments/phase5_c_text_embedding/phase5_c_text_embedding_rerun_summary.md`
- [x] `artifacts/experiments/phase5_c_text_embedding/phase5_c_text_embedding_rerun_summary.json`
- [x] `artifacts/experiment_decisions.json`
  - [x] `phase5_text_embedding_ablation` 항목 갱신
  - [x] `phase5_ranking_collapse_mitigation` 항목 갱신 via `phase5_pre_kure_closure` decision artifact
- [x] `artifacts/experiment_run_summary.md`
  - [x] Phase 5-C 결과 요약 반영
  - [x] Phase 5-B 결과 요약 반영

## Phase 5-Pre: Rare Hobby Fallback Policy (KURE-v1 도입 전 필수)

> Goal: 모델의 편향(랭킹 붕괴) 해소 및 long-tail 추천 품질 확보. 희귀 취미(raw hobby)를 삭제(drop)하지 않고, parent canonical 또는 category로 백오프(fallback)하여 학습/추천에 활용한다.

### Policy Lock

- [x] `rare_item_policy=drop` 제거. 기본값을 `keep_with_fallback` 또는 `backoff_to_canonical_or_category`로 변경한다.
- [x] `raw_hobby`는 원본으로 반드시 보존하며, `canonical_hobby` 및 `category`로 백오프 매핑 테이블을 구축한다.
- [x] 학습(Backbone)과 추천(Display/Expansion)은 분리한다. LightGCN 등은 안정적인 백오프 item으로 학습하고, 최종 Top-K에는 raw hobby가 복원되어 노출되어야 한다.

### Implementation Tasks

- [x] `configs/lightgcn_hobby.yaml`에 `rare_item_policy: keep_with_fallback` 및 `fallback_order` 추가.
- [x] `gnn_recommender/data.py` 내 `prepare_hobby_edges()` 수정. `rare_item_policy != "drop"`에 대한 예외 처리 제거 및 fallback 로직 구축.
- [x] `raw_hobby_to_fallback_item.json` artifact 생성 (raw hobby -> canonical -> category 매핑 저장).
- [x] `vocabulary_report.json`에 `dropped_hobbies` 대신 `fallback_hobbies` 및 `fallback_edges` 통계 추가.

### Experiment Plan

- [x] 기존 `drop` 정책(`min_item_degree=3` 적용) Baseline 재측정.
- [x] 신규 `fallback` 정책 적용 후 지표 비교.
- [x] 비교 지표: `raw_hobbies`, `canonical_hobbies`, `retained_edges`, `candidate_recall@50`, `coverage@10`, `novelty@10`, `cold_start_recall@10`.

### Gate

- [x] fallback 적용 시 `candidate_recall@50` 하락폭이 `-0.01` 이내로 유지되어야 함.
- [x] `coverage@10` 또는 `novelty@10` 지표가 기존 drop 정책 대비 **반드시 개선**되어야 함.

### Artifacts

- [x] `artifacts/vocabulary_report.json`
- [x] `artifacts/raw_hobby_to_fallback_item.json`
- [x] `artifacts/fallback_policy_report.json`

## Phase 5-D: Cold-Start Evaluation Slice

> Goal: determine whether ranking-collapse mitigation helps users with sparse known hobbies, where persona text may carry the most value.

- [x] define cold-start subset as `known_hobbies <= 1`
- [x] add subset metric computation to ranker evaluation artifacts
- [x] report cold-start recall@10 and ndcg@10
- [x] report cold-start coverage@10, novelty@10, intra_list_diversity@10
- [x] closed Phase 2.5 validation baseline recorded: Recall@10=0.592199, NDCG@10=0.367798, Coverage@10=0.002802, Novelty@10=4.570526, ILD@10=0.967444
- [x] closed Phase 2.5 test baseline recorded: Recall@10=0.589513, NDCG@10=0.368271, Coverage@10=0.002802, Novelty@10=4.570526, ILD@10=0.967444
- [x] dedicated artifacts persisted:
  - `artifacts/experiments/phase2_5_cold_start_baseline/validation_metrics.json`
  - `artifacts/experiments/phase2_5_cold_start_baseline/test_metrics.json`
- [x] compare cold-start results for closed Phase 2.5 default, Phase 5-B candidates, and Phase 5-C text embedding ablation
   - [x] record whether cold-start gains justify follow-up even if overall default-promotion gate fails

## Phase 5-E: Text Embedding Preprocessing Hardening

> Goal: keep future text-embedding ablations leakage-safe and reproducible before any experiment is run.

Current implementation evidence shows `[ACT]` masking and domain-tagged persona text scaffolding already exist. The remaining work is governance and hardening, not another immediate experiment run.

- [x] Verify `build_domain_tagged_persona_text()` is the single train/eval input builder for text embedding features.
- [ ] Strengthen `mask_holdout_hobbies()` only if tests reveal grammar loss after `[ACT]` replacement; do not change masking semantics without leakage-audit tests.
- [x] Persist masking policy metadata in every future text-embedding run artifact.
- [x] Persist embedding model metadata including model name, revision/hash when available, device, batch size, embedding dimension, and cache key.
- [x] Add tests proving feature/cache keys differ across KURE-v1, Snowflake-ko, and E5-small-ko candidates.
- [ ] Define and test candidate hobby text builder metadata before Track D runs.
- [ ] Add a source-field and leakage audit for expanded candidate hobby text fields before using aliases, category, or descriptions.
- [ ] Keep candidate hobby text expansion isolated from embedding backbone swaps until both have separate validation artifacts.
- [x] Keep all text embedding, KURE, MMR, and XSimGCL paths opt-in and non-default until a validation winner plus decision artifact exists. Snowflake-ko has now completed validation and winner-only test artifacts for the current split.
