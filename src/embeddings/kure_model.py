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
        batch_size: int | None = settings.EMBEDDING_BATCH_SIZE,  # None 허용
    ) -> None:
        self.model_name = model_name
        self.device = device
        # 배치 사이즈는 _load_model() 호출 시점에 결정됨 (지연 평가)
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

    def _get_optimal_batch_size(self) -> int:
        """
        사용 가능한 GPU VRAM을 기반으로 최적의 배치 사이즈를 동적 계산합니다.
        CPU 환경에서는 전체 처리를 위한 Full Back(배치 1)로 전환합니다.
        """
        # CPU 환경: 로그 출력 및 안전한 배치 사이즈 1 반환
        if not torch.cuda.is_available():
            logger.info(
                "⚠️ CUDA가 사용 불가능하여 CPU에서 임베딩을 생성합니다 (Full Back 모드). "
                "이는 상당한 시간이 소요될 수 있습니다."
            )
            return 1

        # GPU 환경: 여유 VRAM 계산
        device = torch.device("cuda")
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        
        # KURE-v1 fp16 모델 로드 시 예상 메모리 (~2.5GB) 및 안전 마진 (80%) 적용
        estimated_model_memory = 2.5 * (1024 ** 3)
        free_memory = (total_memory * 0.8) - allocated_memory
        available_for_batches = free_memory - estimated_model_memory

        # 최소 요구 VRAM이 부족할 경우 최소 배치 사이즈 강제
        if available_for_batches <= 0:
            logger.warning("VRAM이 부족합니다. 최소 배치 사이즈(128)로 강제 설정합니다.")
            return 128

        # 배치 당 예상 메모리 소모량 계산 (시퀀스 길이 64, Overhead 2배 가정)
        memory_per_element = self.batch_size * 1024 * 2 
        avg_seq_len = 64
        memory_per_batch = 2 * memory_per_element * avg_seq_len
        
        # 동적 배치 사이즈 산출 (최소 128 보장)
        calculated_batch = int(available_for_batches // memory_per_batch)
        optimal_batch_size = max(128, calculated_batch)

        logger.info(
            f"[Auto Batch] 총 VRAM: {total_memory/1e9:.1f}GB | "
            f"동적 배치 사이즈 결정: {optimal_batch_size}"
        )
        return optimal_batch_size

    def _load_model(self) -> Any:
        # ✅ [수정] 동적 배치 사이즈 결정 로직 실행
        if self.batch_size is None:
            import torch
            self.batch_size = self._get_optimal_batch_size()

        normalized_device = self.device.lower()
        if normalized_device == "cpu":
            message = (
                "CPU embedding inference is disabled. Install CUDA-enabled PyTorch and set "
                "EMBEDDING_DEVICE=cuda before running KURE-v1 embeddings."
            )
            logger.warning(message)
            raise RuntimeError(message)

        if normalized_device.startswith("cuda"):
            # 이미 위에서 torch 유무를 확인했으므로 재확인 생략 가능하지만 기존 로직 유지
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
