from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

KURE_MODEL_NAME = "nlpai-lab/KURE-v1"
CACHE_DIR = "GNN_Neural_Network/artifacts/embeddings_cache"
DEFAULT_ATTENTION_IMPLEMENTATION = "sdpa"
DEFAULT_TORCH_DTYPE = "float16"
DEFAULT_TORCH_COMPILE = False
DEFAULT_TORCH_COMPILE_MODE = "reduce-overhead"
_KURE_MODEL_CACHE: dict[str, Any] = {}
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingBackendConfig:
    model_name: str
    device: str
    attention_implementation: str = DEFAULT_ATTENTION_IMPLEMENTATION
    torch_dtype: str = DEFAULT_TORCH_DTYPE
    torch_compile: bool = DEFAULT_TORCH_COMPILE
    torch_compile_mode: str = DEFAULT_TORCH_COMPILE_MODE

    def cache_parts(self) -> tuple[str, ...]:
        return (
            self.model_name,
            self.device or "default",
            self.attention_implementation,
            self.torch_dtype,
            str(bool(self.torch_compile)),
            self.torch_compile_mode,
        )


class HobbyEmbeddingCache:
    def __init__(self, cache_file: str | Path) -> None:
        self.cache_file = str(cache_file)
        self._embeddings: dict[str, np.ndarray] = {}
        cache_dir = os.path.dirname(self.cache_file)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        self._load()

    def _load(self) -> None:
        npy_path = self.cache_file.replace(".txt", ".npy")
        if os.path.exists(npy_path):
            try:
                loaded = np.load(npy_path, allow_pickle=True).item()
                if isinstance(loaded, dict):
                    self._embeddings = {
                        str(key): np.asarray(value, dtype=np.float32)
                        for key, value in loaded.items()
                    }
                    return
            except (OSError, ValueError, TypeError):
                pass
        try:
            with open(self.cache_file, "r", encoding="utf-8") as file:
                for line in file:
                    hobby, vector = line.strip().split("\t", 1)
                    self._embeddings[hobby] = np.fromstring(vector, sep=" ", dtype=np.float32)
        except FileNotFoundError:
            return

    def get(self, hobby: str) -> np.ndarray | None:
        return self._embeddings.get(hobby)

    def set(self, hobby: str, embedding: np.ndarray) -> None:
        self._embeddings[hobby] = np.asarray(embedding, dtype=np.float32)

    def save(self) -> None:
        cache_dir = os.path.dirname(self.cache_file)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        np.save(self.cache_file.replace(".txt", ".npy"), self._embeddings)

    def load_cache_np(self) -> None:
        self._load()


def _compile_alias_patterns(alias_map: dict[str, list[str]]) -> dict[str, list[re.Pattern[str]]]:
    patterns: dict[str, list[re.Pattern[str]]] = {}
    for canonical, aliases in alias_map.items():
        names = [canonical, *aliases]
        patterns[canonical] = [_compile_hobby_pattern(name) for name in names if name]
    return patterns


_KOREAN_BOUNDARY_SUFFIXES = (
    "을",
    "를",
    "으로",
    "은",
    "는",
    "이",
    "가",
    "과",
    "와",
    "도",
    "에",
    "에서",
    "만",
    "부터",
    "까지",
    "의",
    "처럼",
    "하고",
    "하며",
    "하면서",
    "한다",
    "하기",
    "하는",
    "한",
    "할",
    "해",
)


def _compile_hobby_pattern(hobby: str) -> re.Pattern[str]:
    escaped = re.escape(hobby)
    suffixes = "|".join(re.escape(suffix) for suffix in sorted(_KOREAN_BOUNDARY_SUFFIXES, key=len, reverse=True))
    return re.compile(
        rf"(?<![\w가-힣]){escaped}(?=(?:{suffixes})|[^\w가-힣]|$)",
        flags=re.IGNORECASE,
    )


def mask_holdout_hobbies(
    text: str,
    holdout_hobbies: set[str] | list[str] | tuple[str, ...],
    alias_map: dict[str, list[str]] | None = None,
    mask_token: str = "[ACT]",
) -> str:
    if not text or not holdout_hobbies:
        return text

    alias_patterns = _compile_alias_patterns(alias_map) if alias_map else {}
    masked = text
    for hobby in sorted(set(holdout_hobbies), key=len, reverse=True):
        masked = _compile_hobby_pattern(hobby).sub(mask_token, masked)
        for pattern in alias_patterns.get(hobby, []):
            masked = pattern.sub(mask_token, masked)
    return masked


def post_mask_leakage_audit(
    masked_text: str,
    holdout_hobbies: set[str] | list[str] | tuple[str, ...],
    alias_map: dict[str, list[str]] | None = None,
) -> bool:
    if not masked_text or not holdout_hobbies:
        return True

    normalized = _normalize_for_audit(masked_text)
    alias_patterns = _compile_alias_patterns(alias_map) if alias_map else {}
    for hobby in holdout_hobbies:
        normalized_hobby = _normalize_for_audit(hobby)
        if normalized_hobby and _compile_hobby_pattern(normalized_hobby).search(normalized):
            return False
        if alias_map:
            for pattern in alias_patterns.get(hobby, []):
                if pattern.search(normalized):
                    return False
    return True


def _normalize_for_audit(text: str) -> str:
    return " ".join(text.lower().split())


def _load_kure_model(
    device: str | None = None,
    *,
    model_name: str = KURE_MODEL_NAME,
    model_revision: str = "",
    attention_implementation: str = DEFAULT_ATTENTION_IMPLEMENTATION,
    torch_dtype: str = DEFAULT_TORCH_DTYPE,
    torch_compile: bool = DEFAULT_TORCH_COMPILE,
    torch_compile_mode: str = DEFAULT_TORCH_COMPILE_MODE,
) -> Any:
    backend_config = EmbeddingBackendConfig(
        model_name=model_name,
        device=device or "default",
        attention_implementation=attention_implementation,
        torch_dtype=torch_dtype,
        torch_compile=torch_compile,
        torch_compile_mode=torch_compile_mode,
    )
    cache_key = "|".join((model_revision, *backend_config.cache_parts()))
    cached = _KURE_MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError("sentence-transformers is required for KURE text embeddings") from exc

    kwargs: dict[str, Any] = {}
    if device:
        kwargs["device"] = device
    if model_revision:
        kwargs["revision"] = model_revision
    resolved_dtype = resolve_torch_dtype(backend_config.torch_dtype, backend_config.device)
    model_kwargs: dict[str, Any] = {}
    if backend_config.attention_implementation:
        model_kwargs["attn_implementation"] = backend_config.attention_implementation
    if resolved_dtype is not None:
        model_kwargs["torch_dtype"] = resolved_dtype
    if model_kwargs:
        kwargs["model_kwargs"] = model_kwargs
    log_embedding_backend_policy(
        LOGGER,
        backend_config,
        "Loading SentenceTransformer",
    )
    model = SentenceTransformer(model_name, **kwargs)
    if backend_config.torch_compile:
        model = compile_sentence_transformer(model, backend_config.torch_compile_mode)
    if hasattr(model, "max_seq_length"):
        model.max_seq_length = 512
    _KURE_MODEL_CACHE[cache_key] = model
    return model


def resolve_torch_dtype(dtype_name: str, device: str | None = None) -> Any:
    normalized = str(dtype_name or "").strip().lower()
    if normalized in {"", "none", "float32", "fp32"}:
        return None
    if normalized == "auto":
        normalized = "float16" if str(device or "").startswith("cuda") else "float32"
    if normalized in {"float16", "fp16", "half"}:
        import torch

        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        import torch

        return torch.bfloat16
    raise ValueError(f"Unsupported torch dtype for text embeddings: {dtype_name}")


def compile_sentence_transformer(model: Any, mode: str = DEFAULT_TORCH_COMPILE_MODE) -> Any:
    import torch

    first_module = model._first_module() if hasattr(model, "_first_module") else None
    auto_model = getattr(first_module, "auto_model", None)
    if auto_model is None:
        LOGGER.warning("torch.compile requested but SentenceTransformer auto_model was not found; continuing without compile")
        return model
    try:
        first_module.auto_model = torch.compile(auto_model, mode=mode)
        LOGGER.info("torch.compile enabled for SentenceTransformer auto_model: mode=%s", mode)
    except Exception as exc:
        LOGGER.warning("torch.compile failed for SentenceTransformer auto_model; continuing eager. error=%s", exc)
    return model


def embedding_backend_policy(config: EmbeddingBackendConfig) -> dict[str, Any]:
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        cuda_flags = {
            "flash_sdp_enabled": getattr(torch.backends.cuda, "flash_sdp_enabled", lambda: None)(),
            "mem_efficient_sdp_enabled": getattr(torch.backends.cuda, "mem_efficient_sdp_enabled", lambda: None)(),
            "math_sdp_enabled": getattr(torch.backends.cuda, "math_sdp_enabled", lambda: None)(),
            "cudnn_sdp_enabled": getattr(torch.backends.cuda, "cudnn_sdp_enabled", lambda: None)(),
        }
        priority_order = [_sdp_backend_name(item) for item in getattr(torch._C, "_get_sdp_priority_order", lambda: [])()]
    except Exception as exc:
        cuda_available = False
        cuda_flags = {"error": repr(exc)}
        priority_order = []
    return {
        "model_name": config.model_name,
        "device": config.device,
        "attention_implementation": config.attention_implementation,
        "torch_dtype": config.torch_dtype,
        "torch_compile": bool(config.torch_compile),
        "torch_compile_mode": config.torch_compile_mode if config.torch_compile else "",
        "cuda_available": cuda_available,
        "sdpa_backend_selection": "auto_by_pytorch_dispatcher",
        "sdpa_actual_kernel_visibility": "not_exposed_by_public_api_per_call",
        "sdpa_enabled_backends": cuda_flags,
        "sdpa_priority_order": priority_order,
    }


def log_embedding_backend_policy(
    logger: logging.Logger,
    config: EmbeddingBackendConfig,
    prefix: str,
) -> dict[str, Any]:
    policy = embedding_backend_policy(config)
    logger.info(
        "%s backend policy: model=%s device=%s attn_implementation=%s dtype=%s compile=%s compile_mode=%s "
        "sdpa_selection=%s enabled_backends=%s priority=%s",
        prefix,
        policy["model_name"],
        policy["device"],
        policy["attention_implementation"],
        policy["torch_dtype"],
        policy["torch_compile"],
        policy["torch_compile_mode"],
        policy["sdpa_backend_selection"],
        policy["sdpa_enabled_backends"],
        policy["sdpa_priority_order"],
    )
    return policy


def _sdp_backend_name(value: Any) -> str:
    name = getattr(value, "name", "")
    if name:
        return str(name)
    try:
        from torch.nn.attention import SDPBackend

        for candidate in (
            "FLASH_ATTENTION",
            "EFFICIENT_ATTENTION",
            "MATH",
            "CUDNN_ATTENTION",
            "OVERRIDEABLE",
            "ERROR",
        ):
            backend = getattr(SDPBackend, candidate)
            if value == backend or int(value) == int(backend):
                return candidate
    except Exception:
        pass
    return str(value)


def compute_text_embedding_similarity(*args: Any, **kwargs: Any) -> float | np.ndarray:
    if args and not isinstance(args[0], str) and hasattr(args[0], "encode"):
        return _compute_similarity_matrix(*args, **kwargs)
    return _compute_similarity_scalar(*args, **kwargs)


def _compute_similarity_scalar(persona_text: str, hobby_name: str) -> float:
    if not persona_text or not hobby_name:
        return 0.0
    return _lexical_similarity(persona_text, hobby_name)


def _compute_similarity_matrix(
    model: Any,
    persona_texts: list[str] | tuple[str, ...],
    hobby_names: list[str] | tuple[str, ...],
    cache: HobbyEmbeddingCache | None = None,
) -> np.ndarray:
    if not persona_texts or not hobby_names:
        return np.zeros((len(persona_texts), len(hobby_names)), dtype=np.float32)

    persona_embeddings = _encode_texts(model, list(persona_texts), batch_size=32)
    hobby_embeddings: list[np.ndarray] = []
    missing_hobbies: list[str] = []
    missing_indices: list[int] = []

    for index, hobby in enumerate(hobby_names):
        cached = cache.get(hobby) if cache else None
        if cached is None:
            missing_hobbies.append(hobby)
            missing_indices.append(index)
            hobby_embeddings.append(np.empty((0,), dtype=np.float32))
        else:
            hobby_embeddings.append(_normalize_vector(cached))

    if missing_hobbies:
        encoded = _encode_texts(model, missing_hobbies, batch_size=32)
        for hobby, index, embedding in zip(missing_hobbies, missing_indices, encoded, strict=False):
            normalized = _normalize_vector(embedding)
            hobby_embeddings[index] = normalized
            if cache:
                cache.set(hobby, normalized)

    if cache:
        cache.save()

    hobby_matrix = np.vstack(hobby_embeddings)
    return np.matmul(persona_embeddings, hobby_matrix.T).clip(0.0, 1.0)


def _encode_texts(model: Any, texts: list[str], batch_size: int) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    matrix = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-8)


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm <= 0.0:
        return array
    return array / norm


def _lexical_similarity(persona_text: str, hobby_name: str) -> float:
    if hobby_name in persona_text:
        return 1.0
    persona_chars = {char for char in persona_text.lower() if not char.isspace()}
    hobby_chars = {char for char in hobby_name.lower() if not char.isspace()}
    if not persona_chars or not hobby_chars:
        return 0.0
    return float(len(persona_chars & hobby_chars) / len(hobby_chars))


def batch_compute_embedding_similarity(
    persona_texts: list[str],
    hobby_names: list[str],
) -> list[float]:
    if len(persona_texts) != len(hobby_names):
        return []
    return [
        float(_compute_similarity_scalar(persona_text, hobby_name))
        for persona_text, hobby_name in zip(persona_texts, hobby_names, strict=False)
    ]
