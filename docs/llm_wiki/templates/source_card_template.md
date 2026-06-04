# Source Card: TITLE

## Metadata

- Type:
- URL:
- Authors:
- Year:
- Local status:
- Compatibility label:

## Summary

Short source-grounded summary.

## Relevance To This Project

How this source affects `Person -> Hobby`, `Person -> Person`, candidate generation, reranking, text features, graph features, or benchmark design.

## Protocol Match

| Item | Source | This Project | Match |
|---|---|---|---|
| Task | | `Person -> Hobby` or `Person -> Person` | |
| Data unit | | edge, source-target pair, or persona text | |
| Candidate pool | | hobby candidate set or `SIMILAR_TO` topK | |
| Label | | held-out hobby or weak pair label | |
| Metric | | Recall/NDCG, coverage, novelty, strong-reason, low-information, runtime | |

## Adopt / Defer / Avoid

- Adopt:
- Defer:
- Avoid:

## Claim Boundary Impact

State whether this changes the experiment plan, default recommendation behavior, or only supports related work.
