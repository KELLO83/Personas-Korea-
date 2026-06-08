# Similar-Persona Recommendation Experiment PRD

## Goal

This document defines the product and model requirements for the `Person -> Person` similar-persona recommendation experiment.

The core goal is not to list algorithms for study, but to observe whether we can produce a genuinely better similar-persona ranking from the 50,000-persona graph currently loaded in Neo4j.

This experiment is separate from the `Person -> Hobby` hobby recommendation experiment in `experiments/hobby_recommender_ml/`.

```text
experiments/hobby_recommender_ml/
  Person -> Hobby recommendation

experiments/persona_similarity/
  Person -> Person similar-persona recommendation
```

## Current Recommendation Structure

The platform's current similar-persona recommendations are generated with Neo4j GDS.

```text
Neo4j heterogeneous graph
  -> FastRP node embedding
  -> find nearby Person nodes with KNN
  -> (:Person)-[:SIMILAR_TO {score}]->(:Person)
```

Here, `SIMILAR_TO.score` is the current `fastrp_score`.

```text
fastrp_score = candidate-generation score indicating how close two Person embeddings are in graph structure
```

This score is useful, but it is not decomposed by feature to explain why two personas are similar. The current API therefore calculates common-attribute explanations separately in a post-hoc way.

## Problem Definition

The current FastRP/KNN method can quickly find people who are close in the graph, but it has the following problems.

- People can appear similar even when they only share broad attributes such as region, sex, marital status, or community.
- The reason "why they are similar" may not be sufficiently convincing to users.
- In many cases, sentence-style descriptions matter more than structured attributes such as age, education, military status, or housing.
- Examples: personality, after-work activities, lifestyle patterns, values, career attitude, family orientation, leisure style.

Therefore, the goal is not simply to find people who are close in the graph, but to create a ranking that satisfies the following conditions.

- Secure a sufficiently broad candidate set with FastRP/KNN.
- Use both structured commonalities and semantic similarity in text.
- Provide explainable reasons.
- Avoid recommending only from overly generic attributes.
- Leave comparable artifacts for models, features, and costs.

## Data Reality

The current local Neo4j DB is not the full original 1 million personas.

```text
Source data: about 1 million personas
Current Neo4j DB: 50,000-persona sample centered on teens, 20s, and 30s
```

Therefore, conclusions from this experiment are limited to the following scope.

```text
Within the current 50,000-persona graph,
how can similar-persona recommendation be improved?
```

Do not claim operational performance for the full 1 million personas or all age groups.

## Evaluation Reliability and Split Policy

The current similar-persona experiment has no human-labeled ground-truth pairs. Therefore, `label` is a weak/proxy label derived from FastRP/KNN, common structured features, explanation features, and similar signals. These metrics are offline proxies for comparing model candidates, and they do not independently prove user-perceived quality or production performance.

Splits are divided into two grades by purpose.

```text
development split:
  group = source_uuid
  purpose = fast feature/model comparison
  limitation = the same target_uuid can appear in both train and validation

promotion-grade split:
  group = persona_uuid disjoint
  purpose = validation for production integration candidates
  limitation = personas included in validation/test source_uuid or target_uuid
               cannot appear in either source_uuid or target_uuid in train pairs
```

Every experiment artifact must record the split grade used. A model that wins on the development `source_uuid` group split cannot immediately become a production candidate; it must pass a promotion-grade person-disjoint split and manual review.

## Experiment Unit

One training row is not a person. It is a `source -> target` candidate pair.

```text
source_uuid -> target_uuid
```

Meaning:

```text
When viewing the source_uuid person,
how highly should target_uuid be shown as a similar persona?
```

Example:

```text
source_uuid | target_uuid | label | fastrp_score | same_occupation | all_text_cosine
A           | B           | 0.72  | 0.91         | 1               | 0.84
A           | C           | 0.31  | 0.66         | 0               | 0.41
A           | D           | 0.18  | 0.51         | 0               | 0.22
```

The model does not memorize `source_uuid` or `target_uuid`; it learns the order of candidate targets within the same source by looking at pair features between two people.

Do not use `source_uuid`, `target_uuid`, `display_name`, or raw identifiers as features. Also, if features that are identical or nearly identical to the scores used for weak-label generation dominate model performance, interpret the result as reproducing the existing FastRP order and do not treat it as an improvement without separate manual review.

## Scope

### In Scope

- Exporting Neo4j `SIMILAR_TO` candidate pairs.
- Creating `source-target` pair features.
- Structured-feature baseline.
- Sentence embedding similarity feature experiments.
- LightGBM LambdaRank / rank_xendcg experiments.
- Comparing FastRP, deterministic score, LightGBM, hybrid, and text-feature ablations.
- Storing metrics, manual review samples, models, and metadata artifacts.

### Out of Scope

- Training on all `Person x Person` combinations.
- Using `uuid` or names as model features.
- Feeding raw sentence text directly into LightGBM.
- Claiming production performance from weak labels alone.
- Mixing similar-persona experiments with the hobby recommendation folder.

## Core Model Strategy

Sentence descriptions are likely important for similar-persona recommendation.

However, using sentence embeddings directly as a candidate generator is risky. In the hobby recommendation experiment, the KURE semantic Stage1 provider significantly reduced candidate recall.

Therefore, this project uses the following strategy.

```text
FastRP/KNN = candidate generator
LightGBM ranking model = reranker
KURE/text embedding = reranker feature
```

In other words, KURE/sentence embeddings are not the component that finds new person candidates. They are features for better ordering already secured candidate pairs.

## Experiment Priorities

This project does not list many models for their own sake. Experiments are executed in the order most likely to reveal real improvement for the current data shape.

### Required Experiments

These experiments are required for the first decision.

```text
E0. FastRP/KNN candidate-generation baseline
E1. structured deterministic baseline
E2. structured LightGBM LambdaRank / rank_xendcg
E3. sentence embedding cosine feature generation
E4. text-only ablation
E5. integrated structured + sentence + FastRP LightGBM
E6. hybrid FastRP score and model score
E7. diversity / novelty final reranking
```

Questions to answer at this stage:

- Does a LightGBM reranker actually beat the FastRP/KNN order?
- Are structured features enough, or do sentence features provide meaningful signal?
- Do sentence features improve performance, or only add noise?
- Does final top-k overconcentrate in the same occupation, region, or community?
- Are explainable recommendation reasons better than the baseline?

### Candidate-Generation Expansion Experiments

Run these after required experiments only if candidate recall appears insufficient.

```text
E8. Personalized PageRank candidate-generation baseline
E9. Node2Vec candidate-generation baseline
```

Role:

- Compare whether FastRP/KNN misses candidates.
- Use them only to widen the input candidate pool for the LightGBM reranker.
- Do not promote PPR/Node2Vec results directly to production; compare them with the same split, metrics, and manual review.

### Alternative Reranker Validation

Run only when checking whether LightGBM is not strong enough or whether categorical-feature handling is advantageous.

```text
E10. CatBoost ranking
```

Principles:

- The default reranker is LightGBM.
- Compare CatBoost only on the same features, same split, and same candidate pool.
- If the performance difference is small and training/operation cost is higher, keep LightGBM.

### Long-Term Candidates

These are low priority with the current 50,000-sample dataset and weak labels.

```text
HGT / RGCN relational graph transformer
GraphSAGE / PinSage
Two-Tower persona encoder
Cross-encoder reranker
```

Execution conditions:

- Human-labeled similar-person pairs become available.
- Real click/detail-view/selection logs accumulate.
- FastRP/KNN refresh cost becomes a bottleneck on the full 1 million dataset.
- Inductive embeddings for new personas become necessary.

Relational graph transformers (HGT/RGCN) remain in this PRD only as long-term Stage1 candidate-generation replacement experiments for similar-persona `Person -> Person`. The current default strategy is to fix the FastRP/KNN candidate pool and validate the LightGBM/rank_xendcg reranker plus text features.

Fixed conditions when opening HGT/RGCN:

```text
same source split
same topK candidate budget
same reranker recipe
same text feature policy
same evaluation metrics
```

Promotion judgment is not based on the intuition of HGT embeddings themselves, but on whether `candidate_recall@50`, NDCG@5/10, explanation coverage, diversity, and refresh cost all pass compared with FastRP/KNN.

## Experiment Plan

### E0. FastRP/KNN Candidate-Generation Baseline

Purpose:

- Fix the control group for the current similar-persona recommendation.
- Secure the candidate pool that the reranker will reorder.

Experiment:

- `topK=5`: for smoke validation.
- `topK=50`: first real experiment default.
- `topK=100`: for candidate recall/quality comparison.

Important:

```text
The reranker can only change order within exported candidates.
Meaningful reranker experiments are difficult with candidate pairs built from topK=5.
```

Current recommendation:

```text
all 50,000 Person nodes + GDS topK=50
```

### E1. Structured Deterministic Baseline

Purpose:

- Build an understandable baseline without ML.
- Decide whether LightGBM is really needed.

Method:

```text
occupation match
region match
education/field match
family/housing match
shared hobby count
FastRP score
```

Combine them as a weighted sum.

If LightGBM cannot beat this baseline, the reason to use a learned model is weak.

### E2. Structured LightGBM LambdaRank

Purpose:

- First main ranking model.

Input features:

```text
fastrp_score
age_diff
same_age_group
same_sex
same_province
same_district
same_occupation
same_education
same_field
same_marital
same_family
same_housing
same_community
shared_hobby_count
shared_skill_count
explanation_feature_count
```

Training method:

```text
group = source_uuid
objective = lambdarank
```

Comparison:

- `lambdarank`
- `rank_xendcg`

### E3. Sentence Embedding Similarity Feature

Purpose:

- Reflect sentence meaning that may be closer to real similarity than age, education, military status, or region.

Likely sentence columns:

```text
persona
professional_persona
sports_persona
arts_persona
travel_persona
culinary_persona
family_persona
cultural_background
career_goals_and_ambitions
skills_and_expertise
hobbies_and_interests
```

Do not feed raw text into the model. Convert each person's text into embeddings, then add pairwise cosine similarity as features.

Expected features:

```text
all_text_cosine
persona_text_cosine
professional_text_cosine
hobbies_text_cosine
skills_text_cosine
career_text_cosine
family_text_cosine
lifestyle_text_cosine
```

Important:

- In hobby recommendation, KURE Stage1 candidate generation failed.
- However, KURE text features showed better signals than no-text in some experiments.
- Similar-persona recommendation values similarity in people's lifestyles, tendencies, and values more than matching a correct hobby item, so text feature experiments are more valuable here.

### E4. Text-Only Ablation

Purpose:

- Check whether sentence meaning alone contains similarity signal.

Method:

```text
remove structured features
use only text cosine features
```

Interpretation:

- If text-only is strong, sentence descriptions are a core signal.
- If text-only is weak but structured+text improves, text still has value as an auxiliary feature.

### E5. Integrated Structured + Sentence + FastRP Model

Purpose:

- Final candidate model.

Input:

```text
FastRP score
structured pair features
sentence embedding cosine features
```

Model:

```text
LightGBM LambdaRank or rank_xendcg
```

Expectations:

- FastRP captures graph structure.
- Structured features provide explicit explanation evidence.
- Sentence features capture tendencies, lifestyle patterns, and values.

### E6. Hybrid Score

Purpose:

- Reduce LightGBM overfitting to weak labels.

Method:

```text
final_score = alpha * normalized_model_score
            + (1 - alpha) * normalized_fastrp_score
```

Experiment:

```text
alpha = 0.3, 0.5, 0.7, 0.9
```

### E7. Diversity / Novelty Final Reranking

Purpose:

- Prevent similar-persona top-k from collapsing only into the same occupation, same region, same community, or broad demographic matches.
- Observe hub effects where the same target persona is repeated excessively across many sources.
- Increase explainability and exploration diversity without losing too much ranking quality.

Do not reuse hobby recommendation category diversity as-is. For similar personas, test the following axes instead.

```text
occupation diversity
location diversity
community diversity
text-domain similarity diversity
low-information match penalty
hub target / repeated target concentration
```

Initial experiment:

```text
base_score = fastrp_score or model_score
final_score = rerank score adjusted from base ranking with penalties for repeated occupation/region/community
```

Candidate default penalties:

```text
repeated target_occupation
repeated target_province
repeated target_community_id
low-information-only match
```

Experiment values:

```text
diversity_lambda = 0.05, 0.1, 0.2
```

Judgment:

- Reject if NDCG@5/10 drops significantly.
- Occupation/location/community diversity should improve.
- Demographic-only recommendation ratio should decrease.
- Manual review should show meaningful similarity, not forced diversification.

### E8. Personalized PageRank Candidate-Generation Baseline

Run this when the current FastRP/KNN candidate pool appears to have insufficient recall.

Purpose:

- Explore around the source persona in the graph from a random-walk perspective.
- Check whether it produces candidates different from FastRP embedding candidates.
- See whether it can secure additional structurally close and explainable candidates.

Method:

```text
source Person
  -> PPR / random walk with restart
  -> topK target Person candidates
  -> same pair feature builder
  -> same reranker/evaluation pipeline
```

Judgment:

- Does it increase new strong-reason candidates versus FastRP/KNN?
- If candidate overlap is too high, there is little reason to keep it.
- Is candidate-generation time manageable compared with FastRP/KNN?
- Judge final performance as candidate pool improvement for reranker input, not PPR alone.

### E9. Node2Vec Candidate-Generation Baseline

Run this when FastRP embeddings appear not to capture heterogeneous graph structure sufficiently.

Purpose:

- Generate Person candidates with random-walk-based node embeddings.
- Compare candidate diversity, ranking performance, and explainability against FastRP/KNN.

Method:

```text
Neo4j graph export
  -> Node2Vec embedding
  -> approximate nearest neighbor / topK Person candidates
  -> same pair feature builder
  -> same reranker/evaluation pipeline
```

Judgment:

- NDCG/strong-reason/manual review must improve versus FastRP.
- Reject if training/embedding refresh cost is excessive.
- If results are similar to FastRP, keep FastRP for operational simplicity.

### E10. CatBoost Ranking

This is an alternative reranker experiment to check whether categorical-feature handling is better than LightGBM.

The current features are mostly pairwise binary/numeric, so priority is lower than LightGBM.

Principles:

- Use the same candidate pair dataset.
- Use the same feature set and same group split.
- Compare with `source_uuid`-level group ranking.
- Exclude XGBoost from the experiment targets.

Judgment:

- Ranking metrics, explainability, and manual review must all improve versus LightGBM.
- Do not keep it if no categorical-handling advantage is observed.
- Keep LightGBM if the performance difference is small or training/operation cost is high.

### E11. GraphSAGE / PinSage / Two-Tower Family

These are lower priority for now.

Reasons:

- There are currently no person-person ground-truth labels.
- At the 50,000 scale, FastRP/KNN + reranker is more realistic.
- GNNs risk memorizing weak labels in a complex way.
- Two-Tower becomes more meaningful when full 1 million-scale operation and ANN search are needed.

Consider later if the following conditions appear.

- Human-labeled similar-person pairs.
- Real user click/selection logs.
- FastRP/KNN refresh cost becomes a bottleneck on the full 1 million.
- New-persona inductive embeddings are needed.

## Evaluation Metrics

### Ranking

- `NDCG@5`
- `NDCG@10`
- pairwise win-rate versus FastRP baseline
- top-K overlap

### Explainability

- explanation coverage@K
- strong reason coverage@K
- average reason count@K
- low-information dominance@K

Strong explanations:

```text
occupation
detailed region
education/field
shared hobbies
shared skills
sentence semantic similarity
```

Weak explanations:

```text
same sex only
same marital status only
same province only
same community only
```

### Diversity/Stability

- unique target count
- repeated target concentration
- occupation diversity
- location diversity
- community diversity
- demographic-only recommendation ratio
- hub target rate
- ranking stability with fixed seed

### Efficiency

- GDS build time
- candidate pair export time
- feature build time
- embedding build/cache time
- train time
- evaluation time
- inference throughput
- model size
- GPU/CPU usage

## Promotion Criteria

An experiment model must satisfy at least the following conditions before becoming a root platform integration candidate.

- Ranking metrics must be no worse than the FastRP baseline.
- There must be meaningful improvement over the deterministic baseline.
- It must pass validation-first, winner-only testing on a promotion-grade person-disjoint split.
- Explainability must improve.
- It must not recommend only through broad demographic matches.
- If sentence features are used, real semantic similarity must be confirmed in manual review.
- Do not claim production performance from weak-label NDCG improvement alone.
- Refresh/inference cost must be manageable.
- Rollback to the original FastRP order must be possible.

Current default production behavior remains unchanged.

```text
FastRP/KNN SIMILAR_TO + post-hoc explanation API
```

## Current Recommended Execution Order

```text
1. ops/graph/build_gds.py --top-k 50
2. export_pairs.py
3. build_features.py
4. evaluate_fastrp_baseline.py
5. evaluate_deterministic_baseline.py
6. train_lambdarank.py
7. evaluate_lambdarank.py
8. train_rank_xendcg.py
9. evaluate_rank_xendcg.py
10. compare hybrid scores
11. compare diversity/final rerank
12. add text embedding feature experiment
13. compare integrated structured+text model
14. compare PPR candidate-generation baseline if needed
15. compare Node2Vec candidate-generation baseline if needed
16. compare CatBoost ranking alternative reranker with train_catboost_ranker.py / evaluate_catboost_ranker.py if needed
17. update manual review and decision artifacts
```

## Conclusion

The current first model is:

```text
FastRP/KNN candidate generation
  -> structured pair features
  -> LightGBM LambdaRank reranker
```

However, the secondary core experiment with the highest chance of truly improving similar-persona quality is:

```text
FastRP/KNN candidate generation
  -> structured pair features
  -> KURE/text embedding cosine features
  -> LightGBM LambdaRank/rank_xendcg reranker
```

## Similar-Persona Recommendation Principles From Hobby Recommendation Experiments

The core lessons learned from the hobby recommendation experiment in `experiments/hobby_recommender_ml/` also apply to similar-persona recommendation.

```text
Do not trust embedding/graph-based candidate-generation scores directly as final recommendations.
Stage1 creates a broad and stable candidate pool.
The Stage2 reranker looks at structured features, text features, and explanation features together to decide final order.
```

Therefore, the default experiment structure for similar-persona recommendation is fixed as follows.

```text
Stage1 = FastRP/KNN topK >= 50 candidate generation
Stage2 = LightGBM LambdaRank / rank_xendcg reranker
Text embedding = Stage2 pair feature
Final rerank = diversity / explanation-aware rerank, only after accuracy baseline is known
```

Important prohibitions:

- Do not immediately promote text embeddings such as KURE/Snowflake to a Stage1 candidate generator.
- Do not change the Stage1 candidate pool, split, label, LightGBM settings, and text builder in the same experiment.
- Change only one variable per experiment.
- Do not use identifiers such as `source_uuid`, `target_uuid`, or `display_name` as features.

## Root Platform Feature Integration Boundary

Virtual Guild, Life Track, and Agent Interaction Playground in `PRD.md` are root FastAPI/Next.js product features, not model-training scope for this experiment PRD. This PRD defines and validates similar-persona scores, reasons, diversity, text-domain features, and model metadata that those features can consume.

Integration by product feature:

```text
Virtual Guild:
  SIMILAR_TO / reranker score + community_id + shared hobby/skill + PageRank
  The root API converts these into small-group candidates and the D3 graph schema.

Life Track:
  Provides similarity/reason between a source persona and older-cohort target personas.
  Use only as evidence for cross-sectional cohort exploration, not as an individual future prediction.

Agent Interaction Playground:
  Provides source-target pairs, reason categories, and selected persona text fields.
  LLM conversation generation and harmony review are limited to cached/admin root-platform features.
```

When the similar-persona model is connected to a product path, the adapter must return at least the following metadata.

```text
source_uuid
target_uuid
rank
score
score_source
model_version
reason_categories
strong_reason_count
weak_only
text_feature_columns
fallback_used
```

If a model artifact is not yet promoted, the root API must mark it as `experimental` or `fallback` and must not mix it into the default user-facing screen as if it were a promoted model.

### Similar-Persona Recommendation Experiment Priorities

| Priority | Track | Experiment | Purpose |
| ---: | --- | --- | --- |
| 1 | Candidate Lock | Rebuild FastRP/KNN `topK >= 50` candidate pool | Secure enough candidate breadth for the reranker to learn. |
| 2 | Structured Baseline | deterministic / LightGBM baseline based on structured features | Check whether the reranker actually improves the FastRP order. |
| 3 | Text Feature Baseline | Add KURE-v1 pair text cosine feature | Validate the signal that narrative persona text provides for similarity judgment. |
| 4 | Track A | Snowflake-ko embedding backbone swap | Compare whether there is a stronger Korean embedding backbone than KURE-v1. |
| 5 | Track D | persona text builder ablation | Validate which persona text composition is most effective for similar-persona recommendation. |
| 6 | Track B | domain-specific text cosine feature | Separate which domains, such as occupation/hobby/family/lifestyle, explain similarity. |
| 7 | Final Rerank | diversity / explanation-aware rerank | Reduce collapse into obvious same-occupation/same-region recommendations. |
| 8 | Optional Reranker | CatBoost ranking | Compare on the same split/pool only when LightGBM is clearly insufficient. |

Do not change this order arbitrarily without a decision artifact.

### Track A: Embedding Backbone Swap

Track A changes only the text embedding model.

Reference model:

```text
nlpai-lab/KURE-v1
```

Candidate models:

```text
dragonkue/snowflake-arctic-embed-l-v2.0-ko
dragonkue/multilingual-e5-small-ko-v2
```

Items to keep fixed:

- FastRP/KNN candidate pool
- split grade and split file. Promotion candidates use the person-disjoint split.
- weak label generation policy
- LightGBM config
- pair feature schema
- persona text builder
- leakage audit

Items to record:

- `model_name`, `model_revision`
- embedding dimension
- pooling behavior, when known
- device, batch size, runtime
- cache hit/miss
- text preprocessing version
- validation NDCG@5/10, explanation coverage, strong-reason rate

### Track D: Persona Text Builder Ablation

Track D fixes the embedding model and changes only the persona text composition. The default backbone remains KURE-v1.

Experiment candidates:

```text
persona_text_structured_only
persona_text_narrative_only
persona_text_structured_plus_narrative
persona_text_domain_tagged_blocks
persona_text_summary_style
```

The purpose is to check how much narrative information such as personality, intended behavior, lifestyle, and values contributes to similar-persona recommendation compared with structured features such as age, region, and occupation.

Do not run Track D at the same time as Track A. If the embedding backbone and text builder change together, performance drivers cannot be decomposed.

### Track B: Domain-Specific Text Cosine

Track B creates domain-specific cosine features instead of a single `all_text_cosine`.

```text
professional_text_cosine
hobbies_text_cosine
skills_text_cosine
career_text_cosine
family_text_cosine
lifestyle_text_cosine
persona_text_cosine
```

This experiment is intended to help LightGBM learn "more similar" more deeply and to create features that can also connect to API explanation cards.

### Promotion Gate

No reranker is promoted as a production default before passing the following conditions.

- Compare against the FastRP/KNN baseline on the same candidate pool and same split.
- Production promotion candidates must be validated on a person-disjoint promotion-grade split.
- Run validation-first, and run test only for the winner.
- NDCG@5/10 must improve.
- Manual review is required if explanation coverage or strong-reason rate gets worse.
- Hold promotion if low-information recommendation rate increases.
- Interpret weak-label metric improvements together with manual review.
- Record diversity metrics to detect excessive collapse into the same occupation/region community.
- The rollback path is always raw `SIMILAR_TO` ordering.

## Phase 8: High-accuracy Similar-Persona Extension and Quality Calibration Plan

This phase is a long-term candidate research list. For the current 50,000-sample, weak-label-centered data, do not open the models below as the default path. Open them only as separate validation-first ablations after the FastRP/KNN candidate pool, LightGBM/rank_xendcg reranker, text cosine features, and diversity/manual-review gates are completed, and after candidate recall or manual review shows clear limitations.

### 1. Specification for 5 Core Advanced Architectures

#### [1] Cross-Encoder Reranker

* **Purpose**: Learn deep text-interaction features by overcoming the structural limitation of a Bi-Encoder, which uses cosine similarity between independent Source-Target embeddings and lacks Cross-Attention.
* **Architecture**:
  * Rerank the top $K \le 50$ candidate pairs that passed Stage 1 (FastRP/KNN).
  * Input format: `[CLS] Source Persona Description [SEP] Target Persona Description [SEP]`
  * Pass the concatenated texts through all-layer self-attention in a pretrained Korean encoder such as `dragonkue/snowflake-arctic-embed-l-v2.0-ko` or a multilingual-e5-family model, obtain a joint representation of the two persona texts, and compute the final ranking score $S_{CE}$ through an MLP layer.
* **Latency and cost control**:
  * Full $N \times N$ computation is infeasible, so limit Stage 1 candidates to $50$ or fewer.
  * Real-time response within `50ms` is recorded only as an aspirational target before validation. Before real promotion, measure latency, throughput, and GPU/CPU memory in local/server environments.

#### [2] MMoE (Multi-Gate Mixture-of-Experts) Multi-Task Learning

* **Purpose**: Reduce task conflict that appears when a multidimensional definition of "similarity" (demographic background match vs personality/lifestyle orientation match) is learned with a single ranking loss.
* **Structure**:
  * Use $E$ shared expert networks $f_i(X)$ over the shared input feature space $X_{pair}$.
  * Define multiple task heads:
    * **Task 1 (Demographic Match)**: predict demographic similarity such as age difference, sex, region, and occupation ($y_1$)
    * **Task 2 (Lifestyle/Psychographic Match)**: predict semantic lifestyle match in text, such as values, ambitions, career orientation, and leisure activities ($y_2$)
  * Each task $k$ has its own gating network $g^k(X)$, and the final output is weighted as follows:
    $$y_k = \sum_{i=1}^{E} g^k_i(X) \cdot f_i(X)$$
  * The final reranking score $S_{MMoE}$ is combined by task-weighted sum or priority gates.

#### [3] Contrastive Persona Embedding

* **Purpose**: Apply Contrastive Learning to improve the persona representation itself, which is the basis for similarity judgment.
* **Training method and loss**:
  * In a mini-batch, set personas with the same core attributes, such as the same broad occupation group and interest category, as Positive Pair($p_i, p_j$), and treat the rest as Negative Pairs.
  * Use **InfoNCE Loss** to optimize the persona embedding space:
    $$\mathcal{L}_{InfoNCE} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{B} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k) / \tau)}$$
    where $z_i$ is persona $i$'s dense embedding, $\tau$ is the temperature hyperparameter, and $B$ is batch size.
  * This produces a high-quality dense space where not only raw text but also metadata structure is coherently calibrated.

#### [4] LLM-as-a-Judge Offline Evaluation

* **Purpose**: Statistically measure qualitative quality changes that real humans may feel when models improve in a weak-label-centered similarity domain with no labels.
* **Protocol**:
  * Randomly sample 200 Source Personas from the evaluation dataset and extract Top-5 Target Persona pairs recommended by each model (Baseline vs Candidate).
  * If using an LLM Judge, record model name, prompt version, sampling policy, cost, and reproducibility limits.
  * LLM Judge scores do not replace human/manual review. The correlation between quantitative metrics (NDCG) and Judge ratings is a reference metric, and `r >= 0.75` is a long-term quality goal, not a hard promotion gate.

#### [5] Explainability-Guided Objective

* **Purpose**: Lift candidates that can show strong and persuasive recommendation-reason cards in the user UI, rather than recommending only high similarity scores.
* **Formula design**:
  * Define the potential $C_{exp}(s, t) \in [0, 1]$ for the number of strong matching features a Target candidate has, such as the same specialized field or same core hobby.
  * Compute the final reranking score $S_{final}$ by combining pure similarity prediction score $S_{sim}$ with an explainability bonus:
    $$S_{final}(s, t) = S_{sim}(s, t) + \beta \cdot C_{exp}(s, t)$$
  * Tune hyperparameter $\beta$ by grid search so explanation coverage is guaranteed without sharply damaging offline NDCG.

---

### 2. Production Promotion Gates

A new reranker model must satisfy the following hard filter gates before being promoted into the actual deployment `SIMILAR_TO` relationship pipeline. Phase 8 models remain research candidates, not production defaults, until they pass this gate.

| Evaluation Dimension | Metric | Promotion Threshold | Notes |
| :--- | :--- | :--- | :--- |
| **Accuracy** | `NDCG@10` | $\ge +0.005$ versus previous best baseline | secure statistical significance on validation set |
| **Explainability** | `Explanation Coverage@5` | $\ge 95\%$ | share of top-5 results exposing at least one strong reason |
| **Diversity** | `Repeated Hub Rate@10` | $\le 10\%$ | control excessive reuse of a specific master persona as a recommendation hub |
| **Security** | `Leakage Audit Gate` | `Leakage Ratio = 0.00%` | pass full audit for identifier (`uuid`, `name`) feature leakage |

```text
[FastRP/KNN Candidate] -> [Cross-Encoder/MMoE Rerank] -> [Explainability-Guided Objective Filter] -> [Audit Gate] -> [Deploy]
```

---

## 3. Appendix: Four Required Clauses Before Implementation

Before full-scale ML experiments and similar-persona recommendation model code are built, follow the following four implementation requirements to secure data reliability and overcome framework limitations.

### [Clause 1] Bidirectional Leakage Prevention
* **Background and risk**: Similar-persona recommendation naturally contains bidirectional relationships between personas $A$ and $B$ in the dataset. If a simple random split is applied, $(A \to B)$ can appear in training and $(B \to A)$ can appear in validation/test, causing the model to memorize relationship structure and distort performance through data leakage.
* **Implementation constraints**:
  * Fast development comparisons may use a `source_uuid` group split, but this result is not promotion evidence.
  * When building a promotion-grade dataset, Train and Val/Test groups must be completely separated by individual **persona UUID**.
  * In other words, if a persona $i$ appears in either `source_uuid` or `target_uuid` in validation/test, all pair data related to persona $i$ must be completely excluded from training.
  * Split metadata must specify `development_source_group` or `promotion_person_disjoint`.
  * An automated Leakage Audit Unit Test (`test_leakage_audit`) must be enforced at the pipeline entry point.

### [Clause 2] Cold-Start Fallback Hybrid Policy
* **Background and risk**: **FastRP embeddings** based on Neo4j Graph Data Science (GDS) are a typical transductive model. If a newly generated persona that does not exist in the graph database enters the system, no node embedding vector exists and candidate generation can become completely impossible.
* **Implementation constraints**:
  * For new personas without FastRP embeddings or GDS graph nodes, provide the following two-step deterministic hybrid fallback logic.

  ```mermaid
  graph TD
      A[Similar recommendation request arrives] --> B{Check whether GDS node exists}
      B -- Yes (Warm) --> C[FastRP KNN + reranker pipeline]
      B -- No (Cold-Start) --> D[Fallback 1: Bi-Encoder KURE text-embedding cosine similarity]
      D --> E[Fallback 2: demographic-weighted rule-based ranking sum]
      E --> F[Return final similar-persona Top-N recommendations]
  ```

  * **Fallback Step 1**: Compute cosine similarity scores in a KURE-v1-based Bi-Encoder text embedding space.
  * **Fallback Step 2**: Compute a deterministic rule-based matching score as a weighted sum combining age, broad occupation group, and region weights.
  * This hybrid fallback module must integrate cleanly with the existing reranker and API frontend, and must guarantee 200 OK responses without system error logs.

### [Clause 3] GroupKFold Ranking Validation Constraint
* **Background and risk**: When evaluating a similarity reranker model such as LambdaRank or Cross-Encoder, multiple Target Personas are grouped under the same Source Persona query. If queries are scattered across Train/Val, the group structure is damaged and NDCG@5/10 metrics are distorted, making test-set performance prediction impossible.
* **Implementation constraints**:
  * **`GroupKFold`** may be used for development ranking validation and optimization, and the group key must be **`source_uuid`**.
  * All Target pairs sharing the same `source_uuid` must move together within the same fold and must never be split across Train Fold and Validation Fold.
  * A production promotion candidate is not sufficiently validated by a source-group split alone. Final validation/test must be repeated with the person-disjoint split from Clause 1.
  * NDCG@5 and NDCG@10 are calculated as Group-wise NDCG by ranking within each `source_uuid` group and then averaging.

### [Clause 4] Hyperparameter Grid Search Boundary
* **Background and resource limits**: Under this project's ML training infrastructure (Intel Core Ultra 7 CPU, 18 threads, NVIDIA RTX 4060 8GB VRAM), overly high-dimensional grid search can cause Out Of Memory (OOM) or system stalls. Therefore, the core search space must be limited to match hardware constraints while maximizing possible performance improvement.
* **Implementation constraints**:
  * Grid search boundaries for LightGBM / LambdaRank models are strictly controlled as follows.

  | Parameter | Recommended Search Boundary | Notes |
  | :--- | :--- | :--- |
  | `num_leaves` | `[15, 31, 63]` | constraint for overfitting prevention and VRAM savings |
  | `max_depth` | `[4, 6, 8, -1]` | tree depth limit |
  | `learning_rate` | `[0.01, 0.05, 0.1]` | gradient descent step-size control |
  | `n_estimators` | `[100, 200, 500]` | must be used with `early_stopping_rounds=30` |
  | `min_child_samples` | `[10, 20, 50]` | minimum data count in terminal nodes |

  * Multi-threading constraint during training: keep `num_threads=18` to preserve CPU headroom for OS background processes and the Docker Neo4j container.
