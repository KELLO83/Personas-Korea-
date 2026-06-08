# Persona Quality Model Experiment PRD

## Goal

This document defines the product and model requirements for the `Persona -> Quality / Consistency Score` persona quality model experiment.

The core goal is to score and filter whether a synthetic persona can be used as evidence before it is used as context for RAG, pgvector search, Neo4j exploration, or recommendation models.

This project is not a recommendation model. It is a supporting ML/evaluation project for managing input quality for RAG and recommendation systems.

```text
experiments/persona_quality_model/
  Persona -> quality_score
  Persona -> consistency_score
  Persona -> retrieval_eligibility
```

## Problem Definition

The current dataset is NVIDIA synthetic persona data. It has rich text fields and many structured attributes, but the following risks exist.

- Stereotype patterns specific to LLM synthetic data.
- Possible contradictions between persona text and structured metadata.
- Possible mismatches among occupation, skills, and career goals.
- Possible mismatches among family_type, housing_type, and family_persona.
- Hobby leakage and overly direct exposure of hobby names.
- Risk of generating plausible but inaccurate explanations when retrieved as RAG context.

Therefore, this experiment aims to assign each persona the following scores and fields.

```text
quality_score
consistency_score
metadata_text_alignment_score
stereotype_risk_score
retrieval_eligible
quality_flags
quality_reason
```

## Role in RAG/LangGraph

The quality model acts as a context filter for RAG.

Target LangGraph flow:

```text
user question
  -> segment-aware retrieval
  -> persona retrieval candidates
  -> quality filter
  -> rerank / context compression
  -> answer generation
```

Example:

```text
retrieved personas
  -> exclude quality_score < threshold
  -> exclude or lower priority when contradiction_flags exist
  -> use conservatively as answer evidence when stereotype_risk_score is high
```

This model does not generate answers directly. It is used to select context candidates to pass into the LLM.

## Data Basis

The following fields can be used for quality validation.

```text
uuid
age, age_group
sex
province, district
occupation
marital_status, military_status
family_type, housing_type
education_level, bachelors_field
skills_and_expertise_list
hobbies_and_interests_list
persona
professional_persona
sports_persona
arts_persona
travel_persona
culinary_persona
family_persona
cultural_background
skills_and_expertise
hobbies_and_interests
career_goals_and_ambitions
```

Examples of verifiable mismatches:

```text
family_type indicates living alone, but family_persona strongly describes living with children
occupation is developer, but skills contain only unrelated technologies
sports_persona says active, but hobbies are all indoor sedentary activities
age_group and career_goals_and_ambitions form an extremely unnatural combination
district/province repeatedly conflict with travel/lifestyle descriptions
```

## In Scope

- Rule-based quality/consistency scores.
- Metadata-text alignment feature generation.
- Embedding-based outlier score.
- Duplicate/near-duplicate persona detection.
- Stereotype risk heuristic.
- RAG retrieval eligibility flag generation.
- Quality score artifact storage.
- Manual review sample generation.

## Out of Scope

- Finalizing quality labels using only an LLM judge.
- Making social conclusions about real users or population groups.
- Claiming recommendation model performance improvement as this project's standalone result.
- Deleting personas with low quality scores from the source data.
- Fixing thresholds before automatic production RAG blocking without manual review.

## Key Deliverables

Minimum artifacts:

```text
persona_quality_scores.parquet
quality_rules_report.json
quality_flag_examples.csv
outlier_personas.csv
manual_review_samples.csv
metrics.json
run_metadata.json
```

Minimum schema for `persona_quality_scores.parquet`:

```text
person_uuid
quality_score
consistency_score
metadata_text_alignment_score
stereotype_risk_score
duplicate_risk_score
retrieval_eligible
quality_flags
quality_reason
model_version
scored_at
```

## Experiment Priorities

### E0. Quality Rule Inventory

Purpose:

- Define a list of explainable rules before ML.
- Check each rule's hit rate and false-positive risk.

Candidate rules:

```text
missing_required_text
empty_hobby_or_skill_list
metadata_text_family_conflict
metadata_text_occupation_conflict
age_lifestyle_extreme_mismatch
province_district_parse_issue
near_duplicate_text
low_information_persona
overly_generic_persona
```

### E1. Rule-Based Quality Baseline

Purpose:

- Create an initial quality score that can immediately be used as a RAG filter.

Example scoring:

```text
quality_score = 1.0
  - missing_penalty
  - contradiction_penalty
  - duplicate_penalty
  - low_information_penalty
  - stereotype_risk_penalty
```

Principles:

- Do not start with a complex supervised model.
- Without labels, a rule baseline is operationally safer.

### E2. Embedding Outlier Detection

Purpose:

- Find personas that are abnormal outliers in the full persona embedding space.

Candidate methods:

```text
KNN distance outlier
IsolationForest
LocalOutlierFactor
cluster distance to segment centroid
```

Cautions:

- An outlier is not always a bad persona.
- Manual review is needed to distinguish rare personas from low-quality personas.

### E3. Metadata-Text Alignment

Purpose:

- Score whether structured attributes and text descriptions agree with each other.

Candidate methods:

```text
occupation text vs occupation label similarity
family_persona vs family_type consistency
hobbies text vs hobbies list consistency
skills text vs skills list consistency
career text vs occupation/education consistency
```

Implementation principles:

- Start with deterministic keywords/rules first.
- Later expand to embedding similarity or weak-label models.

### E4. Weak Label Quality Model

Purpose:

- Build a quality score model based on rule scores and manual review samples.

Candidate models:

```text
LightGBM binary/ranking/regression
CatBoost
small MLP over embedding + structured features
```

Cautions:

- A weak label is a proxy for quality.
- Use ablation to check whether the model simply memorizes the rules.
- The production gate must include manual review.

### E5. RAG Retrieval Filter Evaluation

Purpose:

- Check whether quality scores actually help RAG context selection.

Evaluation:

```text
baseline retrieval
quality-filtered retrieval
quality-weighted reranking
```

Judgment:

- Does context relevance improve?
- Are contradictory/low-quality personas reduced?
- Does removing too many candidates hurt recall?

## Evaluation Metrics

### Quality/Consistency

- rule hit rate
- contradiction flag rate
- duplicate/near-duplicate rate
- low-information persona rate
- outlier rate

### Manual Review

- sampled persona pass rate
- false positive rate by rule
- false negative examples
- reviewer agreement, when possible

### RAG Utility

- quality-filtered retrieval pass rate
- answer grounding quality sample score
- noisy context reduction rate
- retrieval candidate drop rate
- segment coverage loss

### Operational Stability

- scoring runtime
- memory usage
- reproducibility with fixed seed
- versioned threshold behavior

## Promotion Criteria

The quality model or quality filter must satisfy the following conditions before entering the default RAG path.

- Manual precision for samples with `retrieval_eligible=false` must be sufficiently high.
- It must not remove useful personas excessively through false positives.
- Noisy/contradictory context must decrease in RAG retrieval samples.
- Segment-level coverage must not be significantly damaged.
- Rule/threshold/model versions must be recorded in artifacts.
- The filter must always support being turned off or rolling back thresholds.

Initial production connection should prefer soft reranking over a hard filter.

```text
hard filter:
  exclude only clear missing/parse/duplicate/contradiction cases

soft rerank:
  use quality_score as a retrieval reranking feature
```

## Operational Integration Boundary

This experiment's outputs may be consumed by the following systems.

```text
pgvector:
  metadata.quality_score
  metadata.retrieval_eligible
  quality-aware vector retrieval

Neo4j:
  Person.quality_score
  Person.retrieval_eligible
  quality flag relationship or property

LangGraph/RAG:
  retrieved context quality filter
  noisy context suppression
  answer grounding guardrail

Frontend/Admin:
  persona quality review table
  low-quality persona inspection
```

This PRD defines quality artifacts and threshold policy. Actual application in the root API or UI is handled as a separate product task.

## Current Recommended Execution Order

```text
1. write quality rule inventory
2. generate rule-based baseline score
3. generate manual review sample
4. add embedding outlier score
5. add metadata-text alignment score
6. RAG retrieval filter smoke evaluation
7. decide whether a weak label model is needed
8. record promotion decision
```

## Conclusion

`persona_quality_model` does not start with large supervised ML from the beginning.

The recommended starting point is:

```text
rule-based quality score
+ embedding outlier score
+ metadata-text alignment score
-> RAG context filter / reranking feature
```

The value of this project is not recommendation accuracy itself, but increasing trust in the context used by RAG and recommendation systems.

Therefore, if `persona_segmentation` is the map that defines the search area, `persona_quality_model` is the filter that decides whether retrieved evidence can be used.
