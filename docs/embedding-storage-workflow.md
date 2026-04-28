# 임베딩 벡터 저장 운영 절차

이 문서는 `Neo4j`에 페르소나 임베딩을 적재하는 실행 절차를 정리합니다.

## 1) 실행 엔트리포인트

임베딩 생성/저장을 수행하는 주요 스크립트는 다음입니다.

- `python scripts/build_embeddings.py`

옵션:

- `--batch-size <N>`: Neo4j 반영 시 배치 크기 (기본값: `500`)
- `--skip-existing`: `text_embedding`이 이미 있는 UUID는 건너뜀
- `--sample-size <N>`: Hugging Face 로딩을 N개로 제한
- `--full`: 현재는 과거 호환용(샘플 없이 전체 로딩이 기본 동작)

기본 동작은 **샘플 미지정 → 전체 로딩**입니다.

예시:

```powershell
python scripts/build_embeddings.py
python scripts/build_embeddings.py --sample-size 10000
python scripts/build_embeddings.py --skip-existing --batch-size 500
```

## 2) 파이프라인 단계

1. **데이터 로딩**
   - `src/data/loader.py`의 `load_dataset()` 호출
   - 우선 `data/raw/personas.parquet` 또는 `data/raw/personas.csv` 존재 여부 확인
   - 없으면 Hugging Face(`nvidia/Nemotron-Personas-Korea`, `train`)에서 로드

2. **전처리 및 임베딩 텍스트 구성**
   - `src/data/preprocessor.py`의 `preprocess()`에서 `embedding_text` 생성
   - 여러 persona 텍스트 필드 + 키워드 추출 + 리스트 필드 결합

3. **임베딩 생성**
   - `src/embeddings/kure_model.py`의 `KureEmbedder.encode()`에서 `nlpai-lab/KURE-v1`로 벡터 생성

4. **Neo4j 저장**
   - `src/embeddings/vector_index.py`의 `Neo4jVectorIndex.set_embeddings()`에서 `UNWIND` 배치 업서트
   - `text_embedding`은 `Person` 노드에 저장
   - `person_text_embedding_index`는 cosine 인덱스로 구성

## 3) 환경 변수

필수/주요 설정 (`src/config.py`, `.env`):

- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`
- `HF_DATASET_ID`, `HF_DATASET_SPLIT`
- `DATA_SAMPLE_SIZE`: 값이 비어 있으면 전체 로딩
- `EMBEDDING_MODEL_NAME`, `EMBEDDING_DEVICE`, `EMBEDDING_BATCH_SIZE`, `EMBEDDING_DIMENSION`

## 4) 실행 후 확인 포인트

- 로그에서 `Stored <N> persona embeddings in Neo4j` 메시지 확인
- Neo4j에서 `Person.text_embedding` 속성 존재 건수 확인
- 벡터 인덱스 존재 여부 확인
