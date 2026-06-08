# ADR-004: Chatbot Filter State

## Status

Accepted.

## Context

The chatbot must preserve structured filters across turns while allowing explicit replacement and reset operations.

Example flow:

1. "Show people in Seoul." -> `province=Seoul`
2. "Only people in their twenties." -> `province=Seoul AND age_group=20s`
3. "Narrow it to men." -> `province=Seoul AND age_group=20s AND sex=male`

## Decision

Represent chat filters as a structured state object with one value per field for the MVP.

| Field | Type | Example | API Mapping |
|---|---|---|---|
| `province` | `str | None` | `Seoul` | `/api/search?province=Seoul` |
| `district` | `str | None` | `Seocho-gu` | `/api/search?district=Seocho-gu` |
| `age_group` | `str | None` | `20s` | `/api/search?age_group=20s` |
| `sex` | `str | None` | `male` | `/api/search?sex=male` |
| `occupation` | `str | None` | `developer` | `/api/search?occupation=developer` |
| `education_level` | `str | None` | `bachelor` | `/api/search?education_level=bachelor` |
| `hobby` | `str | None` | `hiking` | `/api/search?hobby=hiking` |
| `keyword` | `str | None` | `startup` | `/api/search?keyword=startup` |

## Merge Rules

- Additive phrasing preserves existing filters and adds new fields.
- Same-field replacement overwrites only the mentioned field.
- Reset phrasing clears the entire filter state.
- Ambiguous multi-value requests should ask for clarification.

Example merge:

```python
current = {"province": "Seoul"}
extracted = {"age_group": "20s"}
merged = {"province": "Seoul", "age_group": "20s"}
```

Example replacement:

```python
current = {"province": "Seoul", "age_group": "20s"}
extracted = {"province": "Busan"}
merged = {"province": "Busan", "age_group": "20s"}
```

## Response Contract

```json
{
  "response": "Found personas matching Seoul, twenties, and male filters.",
  "context_filters": {
    "province": "Seoul",
    "age_group": "20s",
    "sex": "male"
  }
}
```

## Tests

- Preserve filters across three consecutive turns.
- Replace a same-field value.
- Reset to an empty object.
- Ask for clarification when the user provides conflicting values.
