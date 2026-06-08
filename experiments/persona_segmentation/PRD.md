# Persona Segmentation Experiment PRD

## Goal

This document defines the product and model requirements for the `Person -> Segment` persona segmentation experiment.

The core goal is not to simply search or recommend individual records from 1 million synthetic personas, or from the currently loaded 50,000 personas. Instead, it is to group personas into lifestyle/customer-group units that the service and RAG can understand and reuse.

The current project's recommendation axes are separated as follows.

```text
experiments/hobby_recommender_ml/
  Person -> Hobby recommendation

experiments/persona_similarity/
  Person -> Person similar-persona recommendation

experiments/persona_segmentation/
  Person -> Segment lifestyle/customer-group segment
```

## Problem Definition

Individual persona search and recommendation still leave the following problems.

- pgvector similarity search results can spread too broadly.
- Even when Neo4j graph communities exist, they lack human-readable names and descriptions.
- If RAG directly searches all 50,000 or 1 million personas for every question, context selection becomes unstable.
- When explaining hobby recommendation or similar-persona recommendation results, there is no intermediate explanation layer for "what type of person this is."

Therefore, this experiment aims to assign each persona the following segment information.

```text
segment_id
segment_name
segment_description
segment_embedding
dominant_age_groups
dominant_regions
top_occupations
top_hobbies
top_skills
representative_persona_uuids
```

## Role in RAG/LangGraph

Segments become the search unit, filter unit, routing unit, and explanation unit for RAG.

Target LangGraph flow:

```text
user question
  -> intent / constraint extraction
  -> segment routing
  -> segment-aware pgvector retrieval
  -> Neo4j relationship expansion
  -> quality filter
  -> answer generation
```

Example:

```text
Question: Recommend hobbies for working women in their thirties in Seoul

Segment candidates:
  Seoul-metro cultural consumer workers in their thirties
  Self-development, exhibition, and cafe segment in their thirties
  Office-worker segment preferring indoor hobbies

Search:
  search within the relevant segment, centered on persona/hobby/skill/location
```

Without segments, RAG depends only on vector similarity. With segments, it can first narrow the search space and then retrieve more precise persona/hobby/skill context within that space.

## Data Basis

This dataset has both structured and unstructured fields suitable for segmentation.

Major available columns:

```text
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
career_goals_and_ambitions
```

The following axes in particular can be used directly for real segment naming and RAG filters.

```text
demographics: age_group, sex
region: province, district
occupation/skills: occupation, skills
lifestyle: family_type, housing_type, hobbies
text tendency: persona domain text embeddings
```

## In Scope

- Persona feature matrix generation.
- Baseline clustering based on structured features.
- Clustering based on text embeddings.
- Integrated clustering using structured features + text embeddings.
- Comparison between Neo4j community detection results and segments.
- Segment label/summary generation.
- Representative persona extraction.
- Segment metadata artifact generation for pgvector/RAG.
- Segment quality metrics and manual review sample storage.

## Out of Scope

- Retraining the hobby recommendation model in this folder.
- Retraining the similar-persona pair ranking model in this folder.
- Treating a segment as a human demographic stereotype.
- Judging quality only from LLM-generated segment names.
- Immediately promoting the result as the default production API behavior.

## Key Deliverables

Minimum artifacts:

```text
segments.parquet
persona_segments.parquet
segment_profiles.json
segment_centroids.npy or parquet
representative_personas.parquet
metrics.json
manual_review_samples.csv
run_metadata.json
```

Minimum schema for `persona_segments.parquet`:

```text
person_uuid
segment_id
segment_score
segment_version
model_version
assigned_at
```

Minimum schema for `segment_profiles.json`:

```text
segment_id
segment_name
segment_description
size
top_age_groups
top_regions
top_occupations
top_hobbies
top_skills
representative_persona_uuids
risk_notes
```

## Experiment Priorities

### E0. Feature Profiling

Purpose:

- Check missingness, cardinality, distribution, and skew in features used for segments.
- Observe whether synthetic stereotypes create overly easy clusters.

Required report:

```text
row_count
feature_count
category_cardinality
hobby/skill top frequency
age/region/occupation distribution
```

### E1. Structured Baseline Clustering

Purpose:

- Build an understandable baseline using only structured features, without text embeddings.

Candidate features:

```text
age_group
sex
province
occupation group
family_type
housing_type
education_level
top hobbies
top skills
```

Candidate models:

```text
KMeans
MiniBatchKMeans
Use HDBSCAN or DBSCAN-family methods only after validating sample/memory conditions
```

### E2. Text Embedding Clustering

Purpose:

- Check whether persona narratives separate lifestyle segments better.

Input:

```text
embedding_text or domain-tagged persona text
```

Cautions:

- Do not interpret this as recommendation performance because direct hobby-name leakage exists.
- The purpose of this experiment is RAG search/explanation segment generation, not held-out hobby prediction.

### E3. Structured + Text Hybrid Clustering

Purpose:

- Use both the explainability of structured features and the semantic similarity of text embeddings.

Method:

```text
structured feature embedding
+ normalized text embedding
-> clustering
```

Comparison:

```text
structured only
text only
structured + text
```

### E4. Neo4j Community Comparison

Purpose:

- Compare how much Neo4j GDS community detection results overlap with ML segments.

Questions:

- Are graph communities centered on hobbies/skills or demographics?
- Are ML segments more suitable for RAG explanations than graph communities?
- Does using both improve segment explanations?

### E5. Segment Labeling and Summary

Purpose:

- Create human-readable segment names and descriptions.

Principles:

- LLM labels are an auxiliary tool.
- Validate segment names with top features and representative personas.
- Excessive gender/age stereotype labels are prohibited.

### E6. RAG Integration Artifact

Purpose:

- Create metadata that the LangGraph router and retriever can use directly.

Required fields:

```text
segment_id
segment_name
segment_description
segment_embedding
filter_metadata
representative_persona_uuids
top_hobbies
top_skills
```

## Evaluation Metrics

### Cluster Quality

- silhouette score
- Davies-Bouldin index
- Calinski-Harabasz score
- cluster size distribution
- singleton/small cluster ratio
- seed stability

### Explainability

- top feature concentration
- representative persona coherence
- segment naming confidence
- manual review pass rate

### Service/RAG Utility

- segment-aware retrieval hit quality
- segment filter selectivity
- average persona count per segment
- segment coverage over active persona set
- duplicate/near-duplicate segment ratio

### Safety

- stereotype risk notes
- demographic-only segment ratio
- low-information segment ratio
- extremely small protected-like group exposure risk

## Promotion Criteria

The segment model must satisfy the following conditions before connecting to the root platform or default RAG path.

- At least 95% of all personas must be assigned to a segment.
- A single segment must not occupy more than half of the full population.
- Segment names/summaries must be acceptable in manual review.
- Segment-aware retrieval must improve context quality over unfiltered pgvector search.
- Do not promote if there are too many demographic-only segments.
- Segment artifacts must be versioned and support rollback.

## Operational Integration Boundary

This experiment's outputs may be consumed by the following systems.

```text
pgvector:
  persona_vectors.segment_id metadata filter
  segment centroid vector search

Neo4j:
  (:Person)-[:BELONGS_TO]->(:Segment)
  (:Segment)-[:HAS_TOP_HOBBY]->(:Hobby)
  (:Segment)-[:HAS_TOP_SKILL]->(:Skill)

LangGraph/RAG:
  segment router
  segment-aware retriever
  segment summary context

Frontend:
  segment explorer
  segment dashboard
  persona detail segment badge
```

This PRD defines artifact requirements for the integrations above, but root FastAPI/Next.js integration is handled as a separate product task.

## Current Recommended Execution Order

```text
1. feature profiling
2. structured baseline clustering
3. text embedding clustering
4. structured + text hybrid clustering
5. generate segment profile artifact
6. representative persona manual review
7. compare with Neo4j community
8. RAG retrieval smoke evaluation
9. record promotion decision
```

## Conclusion

`persona_segmentation` is not a single recommendation model. It is an intermediate platform layer.

If this experiment succeeds, existing features improve as follows.

```text
Person -> Person:
  used as a similar-persona recommendation reason and diversity correction feature

Person -> Hobby:
  used as a segment-level popularity/compatibility feature

RAG:
  used as a search-scope filter, router, and context summary

Neo4j:
  used as a community explanation and graph exploration unit
```

Therefore, this project is the ML experiment most worth separating first after hobby recommendation and similar-persona recommendation.

