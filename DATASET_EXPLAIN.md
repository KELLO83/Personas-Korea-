# Nemotron-Personas-Korea Dataset Explain

## Dataset Shape

`nvidia/Nemotron-Personas-Korea`는 한국어 synthetic persona 데이터셋이다.

- Source: <https://huggingface.co/datasets/nvidia/Nemotron-Personas-Korea>
- Format: Parquet
- Split: `train`
- Rows: 1,000,000
- Unit: 1 row = 1 persona
- Local file: `data/raw/nemotron-personas-korea/train.parquet`

핵심 컬럼은 다음과 같다.

| Group | Columns | Use |
| --- | --- | --- |
| ID | `uuid` | persona primary key |
| Persona text | `persona`, `professional_persona`, `sports_persona`, `arts_persona`, `travel_persona`, `culinary_persona`, `family_persona` | 생활/직업/취미/가족 맥락 |
| Context text | `cultural_background`, `skills_and_expertise`, `hobbies_and_interests`, `career_goals_and_ambitions` | 성향, 기술, 목표, 취향 |
| List-like text | `skills_and_expertise_list`, `hobbies_and_interests_list` | graph edge source |
| Demographics | `sex`, `age`, `marital_status`, `military_status`, `family_type`, `housing_type`, `education_level`, `bachelors_field` | reranking/context features |
| Work/location | `occupation`, `district`, `province`, `country` | graph nodes and context features |

`src.data.preprocessor.preprocess()`는 list-like string을 list로 파싱하고, `district`를 `province_cleaned`/`district_cleaned`로 나누며, `age_group`과 `embedding_text`를 만든다.

## Current Graph Mapping

현재 Neo4j 그래프는 `Person` 중심 heterogeneous graph로 구성된다.

### Person Node

`Person` 노드는 다음 속성을 가진다.

- `uuid`
- `display_name`
- `age`, `age_group`, `sex`
- `persona`
- persona text fields
- `embedding_text`

### Entity Nodes

| Dataset field | Graph label |
| --- | --- |
| `country` | `Country` |
| `province` | `Province` |
| `district` / cleaned district | `District` |
| `occupation` | `Occupation` |
| `skills_and_expertise_list` | `Skill` |
| `hobbies_and_interests_list` | `Hobby` |
| `education_level` | `EducationLevel` |
| `bachelors_field` | `Field` |
| `marital_status` | `MaritalStatus` |
| `military_status` | `MilitaryStatus` |
| `family_type` | `FamilyType` |
| `housing_type` | `HousingType` |

### Relationships

| Relationship | Meaning |
| --- | --- |
| `(Person)-[:LIVES_IN]->(District)` | residence |
| `(District)-[:IN_PROVINCE]->(Province)` | location hierarchy |
| `(Province)-[:IN_COUNTRY]->(Country)` | location hierarchy |
| `(Person)-[:WORKS_AS]->(Occupation)` | job |
| `(Person)-[:HAS_SKILL]->(Skill)` | skill/expertise |
| `(Person)-[:ENJOYS_HOBBY]->(Hobby)` | hobby |
| `(Person)-[:EDUCATED_AT]->(EducationLevel)` | education |
| `(Person)-[:MAJORED_IN]->(Field)` | major field |
| `(Person)-[:MARITAL_STATUS]->(MaritalStatus)` | marital status |
| `(Person)-[:MILITARY_STATUS]->(MilitaryStatus)` | military status |
| `(Person)-[:LIVES_WITH]->(FamilyType)` | family/living arrangement |
| `(Person)-[:LIVES_IN_HOUSING]->(HousingType)` | housing type |

## Recommendation Use

### Stage 1: Candidate Generation

Stage 1 should generate broad hobby candidates. It should prioritize recall over final precision.

Possible providers:

- LightGCN over `Person-Hobby` edges
- co-occurrence baseline
- popularity fallback
- similar-person hobby lookup

LightGCN input is only:

```text
person_uuid,hobby_name
```

After preprocessing:

```text
person_id,hobby_id
```

LightGCN learns from shared hobby graph structure. It does not directly know age, job, personality, lifestyle, or persona text. Therefore it is suitable as a candidate generator, not as the final recommender.

### Stage 2: Persona-aware Reranking

Stage 2 should decide whether a candidate hobby actually fits the target persona.

Useful inputs:

- `embedding_text`
- `persona`
- `professional_persona`
- `sports_persona`
- `arts_persona`
- `travel_persona`
- `culinary_persona`
- `family_persona`
- `hobbies_and_interests`
- `skills_and_expertise`
- `career_goals_and_ambitions`
- `age`, `age_group`, `sex`
- `occupation`
- `district`, `province`
- `family_type`, `housing_type`
- known hobbies

Example final scoring:

```text
final_score =
  graph_candidate_score
+ persona_text_fit
+ known_hobby_compatibility
+ demographic_fit
+ lifestyle_fit
+ region_accessibility_fit
- mismatch_penalty
```

This is necessary because two people may share one hobby but have very different life contexts. For example, a 50s office worker and a 20s woman may both like golf, but their next best hobby recommendations can be very different.

## Data Quality Notes

Raw hobby names are highly specific natural-language phrases. Similar hobbies may appear as separate strings.

Examples:

- `지역 배드민턴 동호회 활동`
- `지역 배드민턴 클럽 활동`
- `동네 배드민턴 클럽 활동`
- `배드민턴 동호회 활동`

For graph recommendation, raw hobby strings should not be used blindly. Required gates:

- Unicode normalization
- whitespace collapse
- optional alias/canonical taxonomy map
- dedupe after aliasing
- `min_item_degree` filtering
- vocabulary report with raw/canonical/retained counts
- fallback for dropped long-tail items/persons

## Recommended Architecture

```text
Dataset row
  -> preprocessing
  -> graph edges + persona feature store

Stage 1 candidate generation
  -> LightGCN / co-occurrence / popularity / similar-person providers

Stage 2 persona-aware reranking
  -> text fit + demographic fit + lifestyle fit + mismatch penalty

Final Top-K recommendation
  -> hobby name + score + reason
```

In short:

- Use `Person-Hobby` LightGCN for candidate generation.
- Use persona text and structured demographics for final ranking.
- Keep robust fallbacks for cold-start, rare hobbies, and insufficient LightGCN candidates.
