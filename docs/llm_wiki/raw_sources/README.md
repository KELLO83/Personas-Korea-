# Raw Sources

Store source metadata here before writing synthesized wiki notes.

This directory is the raw evidence layer of the LLM Wiki. Treat files here as append-only records.

Allowed:

- URLs
- BibTeX entries
- short abstracts or short excerpts
- local notes about where a PDF, repository, dataset, or artifact can be found
- dated source manifests
- short notes describing how an external source was accessed

Do not store:

- raw datasets
- model checkpoints
- full copied papers
- API keys or private credentials
- Neo4j credentials

## Immutable Source Rule

- Do not silently rewrite existing raw source files.
- If source metadata changes, add a new dated file or append a correction note with the date and reason.
- Synthesized interpretation belongs in `../source_cards/`, `../concepts/`, or `../experiment_notes/`, not here.
- Raw files are the evidence anchor for later wiki pages.
