import logging
from typing import Any

import numpy as np

from src.config import settings

logger = logging.getLogger(__name__)


class KureEmbedder:
    def __init__(
        self,
        model_name: str = settings.EMBEDDING_MODEL_NAME,
        device: str = settings.EMBEDDING_DEVICE,
        batch_size: int = settings.EMBEDDING_BATCH_SIZE,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model: Any | None = None

    @property
    def model(self) -> Any:
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return _to_float_vectors(embeddings)

    def encode_one(self, text: str) -> list[float]:
        vectors = self.encode([text])
        return vectors[0] if vectors else []

    def _load_model(self) -> Any:
        normalized_device = self.device.lower()
        if normalized_device == "cpu":
            message = (
                "CPU embedding inference is disabled. Install CUDA-enabled PyTorch and set "
                "EMBEDDING_DEVICE=cuda before running KURE-v1 embeddings."
            )
            logger.warning(message)
            raise RuntimeError(message)

        if normalized_device.startswith("cuda"):
            try:
                import torch
            except ImportError as exc:
                message = "CUDA embedding requires PyTorch, but torch is not installed."
                logger.warning(message)
                raise RuntimeError(message) from exc

            if not torch.cuda.is_available():
                message = (
                    "CUDA embedding requested but torch.cuda.is_available() is false. "
                    "Install a CUDA-enabled PyTorch build such as torch with cu128 support."
                )
                logger.warning(message)
                raise RuntimeError(message)

        from sentence_transformers import SentenceTransformer

        try:
            return SentenceTransformer(
                self.model_name,
                device=self.device,
                model_kwargs={"torch_dtype": "float16"},
            )
        except (RuntimeError, OSError) as exc:
            message = f"Failed to load KURE-v1 on device '{self.device}'. CPU fallback is disabled."
            logger.warning("%s Original error: %s", message, exc)
            raise RuntimeError(message) from exc


def _to_float_vectors(embeddings: Any) -> list[list[float]]:
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return array.tolist()
