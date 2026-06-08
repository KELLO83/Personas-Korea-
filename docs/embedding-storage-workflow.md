# Embedding Vector Storage Operations

This document describes the execution procedure for loading persona embeddings into `Neo4j`.

## 1) Execution Entry Point

The main script that generates and stores embeddings is:

- `python ops/vector/build_embeddings.py`

Options:

- `--batch-size <N>`: Batch size for Neo4j writes. The default is `500`
- `--skip-existing`: Skip UUIDs that already have `text_embedding`
- `--sample-size <N>`: Limit Hugging Face loading to N records
- `--full`: Currently kept for backward compatibility; full loading without sampling is the default behavior

The default behavior is **no sample specified -> full loading**.

Examples:

```powershell
python ops/vector/build_embeddings.py
python ops/vector/build_embeddings.py --sample-size 10000
python ops/vector/build_embeddings.py --skip-existing --batch-size 500
```

## 2) Pipeline Stages

1. **Data loading**
   - Call `load_dataset()` from `src/data/loader.py`
   - First check whether `data/raw/personas.parquet` or `data/raw/personas.csv` exists
   - If neither exists, load from Hugging Face (`nvidia/Nemotron-Personas-Korea`, `train`)

2. **Preprocessing and embedding text construction**
   - Generate `embedding_text` in `preprocess()` from `src/data/preprocessor.py`
   - Combine multiple persona text fields, extracted keywords, and list fields

3. **Embedding generation**
   - Generate vectors with `nlpai-lab/KURE-v1` in `KureEmbedder.encode()` from `src/embeddings/kure_model.py`

4. **Neo4j storage**
   - Use `UNWIND` batch upserts in `Neo4jVectorIndex.set_embeddings()` from `src/embeddings/vector_index.py`
   - Store `text_embedding` on `Person` nodes
   - Configure `person_text_embedding_index` as a cosine index

## 3) Environment Variables

Required/key settings (`src/config.py`, `.env`):

- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`
- `HF_DATASET_ID`, `HF_DATASET_SPLIT`
- `DATA_SAMPLE_SIZE`: if empty, full loading is used
- `EMBEDDING_MODEL_NAME`, `EMBEDDING_DEVICE`, `EMBEDDING_BATCH_SIZE`, `EMBEDDING_DIMENSION`

## 4) Post-Run Checks

- Confirm the `Stored <N> persona embeddings in Neo4j` log message
- Confirm the count of `Person.text_embedding` properties in Neo4j
- Confirm that the vector index exists
