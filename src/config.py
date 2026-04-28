from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from pydantic import field_validator
from typing import ClassVar


class Settings(BaseSettings):
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8"
    )

    DATA_DIR: Path = Path("./data/raw")
    DATA_FILE: str = "personas.parquet"
    HF_DATASET_ID: str = "nvidia/Nemotron-Personas-Korea"
    HF_DATASET_SPLIT: str = "train"
    DATA_SAMPLE_SIZE: int | None = None

    @field_validator("DATA_SAMPLE_SIZE", mode="before")
    @classmethod
    def _empty_sample_size_to_none(cls, value: object) -> int | None:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        if isinstance(value, str):
            return int(value)
        if isinstance(value, int):
            return value
        raise TypeError("DATA_SAMPLE_SIZE must be an int or empty")

    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "neo4j_password"
    NEO4J_DATABASE: str = "neo4j"
    
    NVIDIA_API_KEY: str = ""
    NVIDIA_BASE_URL: str = "https://integrate.api.nvidia.com/v1"
    LLM_MODEL: str = "deepseek-ai/deepseek-v4-pro"
    
    EMBEDDING_MODEL_NAME: str = "nlpai-lab/KURE-v1"
    EMBEDDING_DEVICE: str = "cuda"
    EMBEDDING_BATCH_SIZE: int = 64
    EMBEDDING_DIMENSION: int = 1024

    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    GDS_FASTRP_DIMENSION: int = 256
    GDS_LEIDEN_MIN_COMMUNITY_SIZE: int = 10
    GDS_KNN_TOP_K: int = 5

settings = Settings()
