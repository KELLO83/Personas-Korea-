# Source Card: Text Embedding Retrieval

## Metadata

- Type: Content/text embedding retrieval and reranking
- URL: local E5 artifacts plus future KURE/Snowflake-ko comparisons
- Authors: local experiment track
- Year: 2026
- Local status: E5 text features already built for similar-persona experiments
- Compatibility label: Manual-review gated

## Summary

Text embedding retrieval uses persona/domain text embeddings and cosine features to find or rerank semantically similar candidates. In this project, E5 domain-specific text cosine features already showed useful signal for similar-persona ranking, but did not beat structured LambdaRank under weak-label metrics.

## Relevance To This Project

For similar-persona recommendation, this is a meaningful offline track because text-only LambdaRank beats raw FastRP by a wide margin. However, structured+text does not beat structured LambdaRank and manual review is not approved.

For hobby recommendation, text features carry leakage risk because persona text can restate held-out hobby names. Text must be masked and audited before being used.

## Protocol Match

| Item | Source | This Project | Match |
|---|---|---|---|
| Task | semantic retrieval/reranking | similar-persona, optional hobby | Good |
| Data unit | text embedding pair | persona text pair or masked hobby text | Good |
| Candidate pool | embedding neighbors | `SIMILAR_TO` candidates or text neighbors | Good |
| Label | semantic or weak label | weak pair label / held-out hobby | Partial |
| Metric | ranking and qualitative review | NDCG, strong reason, low-information, manual review | Good |

## Adopt / Defer / Avoid

- Adopt: Manual-review-first for similar-persona text-driven examples.
- Defer: Backbone swaps to KURE-v1 or Snowflake-ko until E5 review is complete.
- Avoid: Raw text into LightGBM or unmasked hobby text features.

## Claim Boundary Impact

Text features are promising but cannot support production promotion without manual review and leakage controls.
