# LLM Wiki Change Log

## 2026-06-04

- Added the initial LLM Wiki structure for the Korean persona recommender project.
- Added source-card, raw-source, concept, and experiment-note folders following the reference wiki pattern from `C:\Users\Kello\robot\docs\llm_wiki`.
- Added source cards for RecBole, LightGCN/XSimGCL, KGAT, neural retrieval/ranking, Factorization Machines, text embedding retrieval, and diversity reranking.
- Added current recommender findings:
  - Hobby recommendation is `Person -> Hobby`.
  - Similar-persona recommendation is directed `Person -> Person`.
  - Hobby candidate recall is already high, so ranking/diversity is the main follow-up.
  - Similar-persona structured LambdaRank is strong offline but blocked from promotion by weak-label/manual-review constraints.
- Added an experiment-note page that turns the recommender alternative-method investigation into concrete experiment tracks.
- Split the wiki into two explicit ML experiment track folders:
  - `docs/llm_wiki/person_hobby/` for `Person -> Hobby` hobby recommendation.
  - `docs/llm_wiki/persona_similarity/` for directed `Person -> Person` similar-persona recommendation.
- Kept shared external method cards under `source_cards/recommender_methods/`, but moved track-specific decisions and experiment plans into the separate folders.
- Consolidated already-completed experiment results into:
  - `docs/llm_wiki/person_hobby/results_summary.md`
  - `docs/llm_wiki/persona_similarity/results_summary.md`
  - `docs/llm_wiki/experiment_notes/2026-06-04-existing-results-consolidation.md`
- Recorded the hobby result-source caveat: `GNN_Neural_Network/EXPERIMENTS.md` is a Phase 2.5-era summary, while `GNN_Neural_Network/artifacts/experiment_run_summary.md` contains later E5-domain and Phase 6 results.
- Added code and document inventories for each ML experiment track:
  - `docs/llm_wiki/person_hobby/code_inventory.md`
  - `docs/llm_wiki/person_hobby/document_inventory.md`
  - `docs/llm_wiki/persona_similarity/code_inventory.md`
  - `docs/llm_wiki/persona_similarity/document_inventory.md`
- Added `.omo/plans/recommender-continuation-experiment-plan.md` so future work continues from existing results instead of restarting either recommender experiment.
- Added continuation-analysis code and generated first reports:
  - Hobby: run comparison, slice gap report, Phase 6 alias audit, feature ablation manifest.
  - Similar-persona: text feature value report and failure taxonomy labels for manual-review samples.
