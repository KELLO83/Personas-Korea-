# Similar-Persona Recommendation Dataset Explanation

## The Most Important One-Line Summary

The training data for the similar-persona model is not one row per person. It is one row per `source_person -> target_person` candidate pair.

```text
source_uuid | target_uuid | label | feature...
```

In other words, the model learns the following question.

```text
When viewing the source person,
how highly should the target person be ranked as a similar persona?
```

## Current Data Scope

The current Neo4j DB is not the full original 1 million personas.

```text
Current DB: 50,000 Person
Age groups: centered on teens / twenties / thirties
```

Major distributions from the current Excel export:

```text
Person rows: 50,000

age_group:
  twenties: 22,842
  thirties: 16,667
  teens: 10,491

sex:
  male: 26,141
  female: 23,859

province:
  Gyeonggi: 13,743
  Seoul: 10,466
  Incheon: 2,980
  Busan: 2,933
  Gyeongsangnam: 2,669

occupation:
  unemployed: 20,914
  other long-tail occupations: 1,423 types

education:
  four-year university: 18,131
  high school: 15,824
  two-to-three-year college: 11,733

field:
  missing: 29,700
  about 59.4% have no major field

skill_count:
  currently all 0

hobby_count:
  average about 2.79

similar_to_out_count:
  currently all 5
  this means the current state was generated with GDS topK=5
```

Important:

```text
The current topK=5 state is too narrow for reranker experiments.
Before real similar-persona reranker experiments, GDS must be rebuilt with topK=50 or higher.
```

## Source Graph Structure

In Neo4j, Person nodes are connected to several attribute nodes.

```text
(:Person)
  -[:LIVES_IN]->(:District)-[:IN_PROVINCE]->(:Province)-[:IN_COUNTRY]->(:Country)
  -[:WORKS_AS]->(:Occupation)
  -[:EDUCATED_AT]->(:EducationLevel)
  -[:MAJORED_IN]->(:Field)
  -[:MARITAL_STATUS]->(:MaritalStatus)
  -[:MILITARY_STATUS]->(:MilitaryStatus)
  -[:LIVES_WITH]->(:FamilyType)
  -[:LIVES_IN_HOUSING]->(:HousingType)
  -[:ENJOYS_HOBBY|LIKES]->(:Hobby)
  --> (:Skill)  # Skill edges are effectively absent in the current DB
```

After GDS runs, the following information is added to Person nodes.

```text
(:Person).fastrp_embedding
(:Person).community_id
(:Person)-[:SIMILAR_TO {score}]->(:Person)
```

`SIMILAR_TO.score` is a candidate-generation score, not a training label.

## Why Not Build Every Person x Person Pair?

Even with only 50,000 people, all directed pairs total about 2.5 billion.

```text
50,000 * 49,999 ~= 2.5B pairs
```

Therefore, the experiment does not train on every pair of people.

Instead, it exports only the `SIMILAR_TO` candidates created by GDS.

```text
source Person -> topN similar target Persons
```

Recommendation candidate counts:

```text
topK=5: for smoke validation
topK=50: first real experiment setting
topK=100: for recall/quality comparison
```

## Shape of a Training Row

One row has the following meaning.

```text
Given source_uuid person,
how similar should target_uuid person be considered?
```

Example:

```text
source_uuid | target_uuid | label | fastrp_score | age_diff | same_occupation | shared_hobby_count
A           | B           | 0.72  | 0.91         | 2        | 1               | 2
A           | C           | 0.31  | 0.66         | 8        | 0               | 0
A           | D           | 0.18  | 0.51         | 12       | 0               | 0
```

The model learns the order of target candidates within the same `source_uuid`.

```text
group = source_uuid
```

## Exported Raw Candidate-Pair Columns

These are the raw pair columns exported from Neo4j by `export_pairs.py`.

| Column | Type | Meaning |
| --- | --- | --- |
| `source_uuid` | string | reference person |
| `target_uuid` | string | candidate similar persona |
| `fastrp_score` | float | `SIMILAR_TO.score` |
| `source_age` | int/null | source age |
| `target_age` | int/null | target age |
| `source_age_group` | string/null | source age group |
| `target_age_group` | string/null | target age group |
| `source_sex` | string/null | source sex |
| `target_sex` | string/null | target sex |
| `source_province` | string/null | source province/city |
| `target_province` | string/null | target province/city |
| `source_district` | string/null | source district/county |
| `target_district` | string/null | target district/county |
| `source_occupation` | string/null | source occupation |
| `target_occupation` | string/null | target occupation |
| `source_education` | string/null | source education level |
| `target_education` | string/null | target education level |
| `source_field` | string/null | source major/field |
| `target_field` | string/null | target major/field |
| `source_marital` | string/null | source marital status |
| `target_marital` | string/null | target marital status |
| `source_family` | string/null | source family type |
| `target_family` | string/null | target family type |
| `source_housing` | string/null | source housing type |
| `target_housing` | string/null | target housing type |
| `source_community_id` | int/null | source Leiden community |
| `target_community_id` | int/null | target Leiden community |
| `shared_hobbies` | JSON list[string] | hobby names shared by both people |
| `shared_skills` | JSON list[string] | skill names shared by both people |

These raw columns include values for human reading and audits. They are not fed directly into the model, but converted into numeric features.

## Initial Structured Training Features

The numeric features currently fed directly into training code are:

| Feature | Type | Meaning |
| --- | --- | --- |
| `fastrp_score` | float | FastRP/KNN candidate-generation score |
| `age_diff` | float | age difference |
| `same_age_group` | 0/1 | whether the age group is the same |
| `same_sex` | 0/1 | whether sex is the same |
| `same_province` | 0/1 | whether province/city is the same |
| `same_district` | 0/1 | whether district/county is the same |
| `same_occupation` | 0/1 | whether occupation is the same |
| `same_education` | 0/1 | whether education level is the same |
| `same_field` | 0/1 | whether major/field is the same |
| `same_marital` | 0/1 | whether marital status is the same |
| `same_family` | 0/1 | whether family type is the same |
| `same_housing` | 0/1 | whether housing type is the same |
| `same_community` | 0/1 | whether GDS community is the same |
| `shared_hobby_count` | int | number of shared hobbies |
| `shared_skill_count` | int | number of shared skills |
| `explanation_feature_count` | int | number of explainable common features |

Columns not used by the model:

```text
source_uuid
target_uuid
display_name
uuid
raw sentence text
```

`uuid` is for split/group/review, not a feature.

## Feature Quality Interpretation

Not every feature has the same meaning.

Strong structured signals:

```text
same_occupation
same_district
same_education
same_field
shared_hobby_count
shared_skill_count
```

Medium signals:

```text
same_family
same_housing
same_province
same_community
same_age_group
```

Weak signals:

```text
same_sex
same_marital alone
same_province alone
same_community alone
```

Caution:

```text
If the model appears to recommend well by looking only at sex, marital status, or province-level region,
actual similarity quality may be low.
```

## Sentence-Style Data

Current Person records contain natural-language description columns in addition to structured features.

| Column | Meaning |
| --- | --- |
| `persona` | full persona description |
| `professional_persona` | occupation/work tendency |
| `sports_persona` | sports/activity tendency |
| `arts_persona` | arts/culture tendency |
| `travel_persona` | travel tendency |
| `culinary_persona` | food/cooking tendency |
| `family_persona` | family/relationship tendency |
| `cultural_background` | cultural background |
| `career_goals_and_ambitions` | career goals/attitude |
| `skills_and_expertise` | skill/expertise description |
| `hobbies_and_interests` | hobby/interest description |

These sentences can be very important for similar-persona recommendation.

Example:

```text
Enjoys reading and walking alone after work.
Likes meeting new people in exercise groups.
Values time spent with family the most.
Studies consistently for certifications to grow in their career.
```

This kind of information can be closer to actual tendency similarity than age, education, military service, or housing type.

## Why Not Feed Raw Sentences Directly?

Raw sentences are not passed directly into LightGBM.

Reasons:

- Tree models cannot directly process long natural-language text.
- They can memorize sentence patterns or overuse duplicate information.
- Explanation and leakage audits become difficult.

Instead, sentences are converted into embeddings, and cosine similarity between source and target is added as a feature.

```text
source_text -> embedding
target_text -> embedding
cosine(source_embedding, target_embedding) -> feature
```

## Secondary Semantic Feature Candidates

Candidate features to add in sentence embedding experiments:

| Feature | Meaning |
| --- | --- |
| `all_text_cosine` | overall semantic similarity after combining all text descriptions |
| `persona_text_cosine` | full persona description similarity |
| `professional_text_cosine` | occupation/work tendency similarity |
| `hobbies_text_cosine` | hobby/interest sentence similarity |
| `skills_text_cosine` | skill/expertise sentence similarity |
| `career_text_cosine` | career-goal similarity |
| `family_text_cosine` | family/relationship tendency similarity |
| `lifestyle_text_cosine` | leisure/lifestyle-pattern similarity |

Important design:

```text
KURE/text embeddings are used first as reranker features, not as candidate generators.
```

## Why Hobby-Recommender KURE Results Cannot Be Applied Directly to Similar Personas

In hobby recommendation, the KURE semantic Stage1 provider reduced performance.

Reason:

```text
hobby recommendation = predicting a held-out hobby item
KURE candidate generation = can miss the correct hobby in the candidate pool
candidate_recall@50 drop -> Recall/NDCG drop
```

But similar-persona recommendation is different.

```text
similar-persona recommendation = similarity problem over lifestyle patterns, tendencies, and values between people
sentence descriptions may be a core feature
```

Therefore, for similar-persona recommendation:

```text
KURE/text embedding candidate generation: not recommended
KURE/text embedding reranker feature: recommended experiment
```

## Weak Label

There are currently no human labels for person-person similarity.

Therefore, the initial label is a weak label.

Current structured weak label:

```text
label =
  0.25 * same_occupation
  0.15 * same_province
  0.10 * same_district
  0.10 * same_age_group
  0.10 * same_education
  0.10 * same_community
  0.12 * min(shared_hobby_count, 5) / 5
  0.08 * min(shared_skill_count, 5) / 5
  0.10 * normalized_fastrp_score
```

Limitations:

- This label is not real user preference.
- Evaluation can be biased because the label uses the same information as structured features.
- It does not sufficiently reflect sentence semantic similarity.

Therefore, manual review is especially important for text feature experiments.

## Split Method

Do not split by row.

Always split by `source_uuid`.

```text
train: 80% source_uuid
valid: 10% source_uuid
test: 10% source_uuid
seed: 42
```

Reason:

```text
If the same source person appears in train and valid/test at the same time,
query-person-level generalization evaluation breaks.
```

## Final Training Data Shape

Initial structured model:

```text
source_uuid
target_uuid
label
split
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
deterministic_score
```

The secondary semantic model adds the following:

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

The model features are the feature columns, not `label`, `split`, or `uuid`.

## Evaluation Method

Evaluation must rank within each `source_uuid`.

```text
source_uuid = A
  target B score
  target C score
  target D score
```

Comparison targets:

```text
FastRP score order
deterministic score order
structured LightGBM order
text-only LightGBM order
structured+text LightGBM order
hybrid score order
```

## Required Artifacts

| File | Meaning |
| --- | --- |
| `candidate_pairs.parquet` | source-target candidate pairs exported from Neo4j |
| `pair_features.parquet` | numeric training features + weak labels |
| `splits.json` | split by source_uuid |
| `*_metrics.json` | per-experiment evaluation results |
| `*_manual_review.csv` | recommendation samples for human review |
| `*_train_metadata.json` | model training settings/feature importance |
| `*.txt` | LightGBM model files |

## Most Important Caveats in the Current Data

1. Current `SIMILAR_TO` is in the topK=5 state.
   - topK=50 regeneration is required before this experiment.

2. The `Skill` feature is currently almost dead.
   - Since `skill_count=0`, `shared_skill_count` is unlikely to contribute to performance.

3. `field` has many missing values.
   - About 59.4% missing.

4. `occupation` is long-tailed, and `unemployed` is very common.
   - Same-occupation features can act too strongly for some candidates.

5. Sentence-style features are likely important for similar personas.
   - They must be added as embedding cosine features, not as raw text input.

## Current Conclusion

Given the data shape, the most reasonable experiment order is:

```text
1. Secure candidate pairs with FastRP/KNN topK=50
2. Build structured pair features
3. Evaluate FastRP baseline
4. Evaluate deterministic baseline
5. Evaluate structured LightGBM ranking model
6. Add sentence embedding cosine features
7. Compare text-only / structured+text / hybrid
8. Confirm real semantic similarity through manual review
```

The final thing to inspect is not just a metric.

```text
Why are these two people similar?
Is the reason acceptable to a user?
Is it a shallow recommendation based only on matching structured attributes?
Are their lifestyle patterns, tendencies, and values also semantically similar?
```

