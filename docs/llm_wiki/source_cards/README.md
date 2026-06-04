# Source Cards

Source cards are grouped by research role so related-work review stays readable as the wiki grows.

## Folders

- `recommender_methods/`: recommendation model families, benchmark frameworks, and reranking strategies.

## Rule

Do not compare reported numbers across cards unless the task, dataset split, candidate pool, label construction, feature policy, and metric match.

For this project, never compare `Person -> Hobby` Recall/NDCG directly with `Person -> Person` NDCG or manual-review metrics. Treat them as separate recommender systems.
