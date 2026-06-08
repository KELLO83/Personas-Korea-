# ADR-002: Recommendation Reasoning

## Status

Accepted as a runtime explanation policy.

## Context

Recommendation responses must explain why an item was recommended without requiring an LLM call on every request.

## Decision

Use deterministic explanation templates backed by graph evidence, similarity evidence, model metadata, or fallback metadata.

## Template Examples

```json
{
  "hobby": "Among {similar_count} people similar to you, {ratio:.0%} have '{item_name}' as a hobby.",
  "skill": "Among {similar_count} people similar to you, {ratio:.0%} have the '{item_name}' skill.",
  "occupation": "Among {similar_count} people similar to you, {ratio:.0%} have the '{item_name}' occupation.",
  "district": "Among {similar_count} people similar to you, {ratio:.0%} live in '{item_name}'."
}
```

Example response fragment:

```json
{
  "item_name": "climbing",
  "reason": "Among 128 people similar to you, 73% have 'climbing' as a hobby."
}
```

## Consequences

- Explanations are reproducible and testable.
- LLM usage remains optional.
- Model-backed recommendations must expose enough metadata for the same explanation contract.
