from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import polars as pl

from experiments.hobby_recommender_ml.hobby_recommender.text_embedding import (
    DEFAULT_TORCH_COMPILE_MODE,
    EmbeddingBackendConfig,
    compile_sentence_transformer,
    log_embedding_backend_policy,
    resolve_torch_dtype,
)
from experiments.persona_similarity.scripts.common import (
    ensure_parent,
    file_sha256,
    load_config,
    mark_cache_hit,
    should_use_cache,
    stable_json_hash,
    write_json,
)
from experiments.persona_similarity.scripts.text_feature_builder import TEXT_DOMAINS, build_domain_text, embedding_key, text_hash

LOGGER = logging.getLogger(__name__)
CPU_TEXT_EMBEDDING_BATCH_SIZE = 32
MAX_AUTO_TEXT_EMBEDDING_BATCH_SIZE = 1024
MIN_AUTO_TEXT_EMBEDDING_BATCH_SIZE = 64
TEXT_EMBEDDING_MB_PER_TEXT = 18
DEFAULT_TEXT_EMBEDDING_VRAM_FRACTION = 0.9


def default_cpu_threads() -> int:
    return min(max((os.cpu_count() or 1) - 4, 1), 18)


def iter_with_progress(items: list[dict[str, Any]], enabled: bool) -> Any:
    if not enabled:
        return items
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return items
    return tqdm(items, desc="building text corpus", unit="text")


def build_text_records(personas: pl.DataFrame, progress: bool) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in iter_with_progress(personas.to_dicts(), progress):
        uuid = str(row["uuid"])
        for domain in TEXT_DOMAINS:
            text = build_domain_text(row, domain)
            rows.append(
                {
                    "key": embedding_key(uuid, domain),
                    "uuid": uuid,
                    "domain": domain,
                    "text": text,
                    "text_hash": text_hash(text),
                }
            )
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/persona_similarity/configs/lightgbm_reranker.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    text_config = config["text_embedding"]
    cache_metadata = {
        "stage": "build_text_embeddings",
        "input_path": config["paths"]["persona_texts"],
        "input_hash": file_sha256(config["paths"]["persona_texts"]),
        "config_hash": stable_json_hash(
            {
                "text_embedding": text_config,
                "domains": list(TEXT_DOMAINS.keys()),
            }
        ),
        "model_name": text_config["model_name"],
        "preprocessing_version": text_config.get("preprocessing_version", "persona_similarity_text_v1"),
    }
    use_cache, cache_reason = should_use_cache(config["paths"]["text_embeddings"], config["paths"]["text_embedding_metadata"], cache_metadata, args.force)
    if use_cache:
        mark_cache_hit(config["paths"]["text_embedding_metadata"], cache_metadata, config["paths"]["text_embeddings"])
        return

    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit("sentence-transformers and torch are required to build text embeddings.") from exc

    os.environ.setdefault("OMP_NUM_THREADS", str(default_cpu_threads()))
    os.environ.setdefault("MKL_NUM_THREADS", str(default_cpu_threads()))
    personas = pl.read_parquet(PROJECT_ROOT / config["paths"]["persona_texts"])
    records = build_text_records(personas, bool(text_config.get("progress", True)))
    texts = [record["text"] for record in records]
    empty_mask = np.array([not text.strip() for text in texts], dtype=bool)

    device = str(text_config.get("device", "auto"))
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    requested_batch_size = int(text_config.get("batch_size") or 0)
    vram_fraction = float(text_config.get("vram_fraction", DEFAULT_TEXT_EMBEDDING_VRAM_FRACTION))
    batch_size = resolve_text_embedding_batch_size(
        requested=requested_batch_size,
        device=device,
        vram_fraction=vram_fraction,
        torch_module=torch,
    )
    start_time = time.perf_counter()
    attention_implementation = str(text_config.get("attention_implementation", "sdpa") or "").strip()
    torch_dtype_name = str(text_config.get("torch_dtype", "float16") or "").strip()
    torch_compile = bool(text_config.get("torch_compile", False))
    torch_compile_mode = str(text_config.get("torch_compile_mode", DEFAULT_TORCH_COMPILE_MODE) or DEFAULT_TORCH_COMPILE_MODE)
    model_kwargs: dict[str, Any] = {}
    if attention_implementation:
        model_kwargs["attn_implementation"] = attention_implementation
    resolved_dtype = resolve_torch_dtype(torch_dtype_name, device)
    if resolved_dtype is not None:
        model_kwargs["torch_dtype"] = resolved_dtype
    backend_policy = log_embedding_backend_policy(
        LOGGER,
        EmbeddingBackendConfig(
            model_name=str(text_config["model_name"]),
            device=device,
            attention_implementation=attention_implementation,
            torch_dtype=torch_dtype_name,
            torch_compile=torch_compile,
            torch_compile_mode=torch_compile_mode,
        ),
        "Persona-similarity text embedding",
    )
    model = SentenceTransformer(str(text_config["model_name"]), device=device, model_kwargs=model_kwargs or None)
    if torch_compile:
        model = compile_sentence_transformer(model, torch_compile_mode)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=bool(text_config.get("progress", True)),
        convert_to_numpy=True,
    ).astype(np.float32)
    embeddings[empty_mask] = 0.0
    elapsed = time.perf_counter() - start_time

    output_path = ensure_parent(config["paths"]["text_embeddings"])
    np.savez_compressed(
        output_path,
        keys=np.array([record["key"] for record in records], dtype=object),
        uuids=np.array([record["uuid"] for record in records], dtype=object),
        domains=np.array([record["domain"] for record in records], dtype=object),
        text_hashes=np.array([record["text_hash"] for record in records], dtype=object),
        empty_mask=empty_mask,
        embeddings=embeddings,
    )
    write_json(
        config["paths"]["text_embedding_metadata"],
        {
            **cache_metadata,
            "cache_hit": False,
            "cache_reason": cache_reason,
            "model_name": text_config["model_name"],
            "attention_implementation": attention_implementation,
            "torch_dtype": torch_dtype_name,
            "torch_compile": torch_compile,
            "torch_compile_mode": torch_compile_mode if torch_compile else "",
            "backend_policy": backend_policy,
            "device": device,
            "batch_size": batch_size,
            "batch_mode": "auto" if requested_batch_size <= 0 else "manual",
            "vram_fraction": vram_fraction,
            "embedding_rows": int(len(records)),
            "persona_rows": int(personas["uuid"].n_unique()),
            "domains": list(TEXT_DOMAINS.keys()),
            "empty_text_rows": int(empty_mask.sum()),
            "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
            "runtime_seconds": elapsed,
        },
    )


def resolve_text_embedding_batch_size(
    *,
    requested: int,
    device: str,
    vram_fraction: float,
    torch_module: Any,
) -> int:
    if requested > 0:
        return requested
    if device != "cuda":
        return CPU_TEXT_EMBEDDING_BATCH_SIZE

    free_mb = query_free_gpu_memory_mb(torch_module)
    usable_mb = int(free_mb * max(0.1, min(vram_fraction, 0.9)))
    batch_size = usable_mb // TEXT_EMBEDDING_MB_PER_TEXT
    return max(MIN_AUTO_TEXT_EMBEDDING_BATCH_SIZE, min(MAX_AUTO_TEXT_EMBEDDING_BATCH_SIZE, int(batch_size)))


def query_free_gpu_memory_mb(torch_module: Any) -> int:
    try:
        free_bytes, _total_bytes = torch_module.cuda.mem_get_info()
        return int(free_bytes / 1024 / 1024)
    except Exception:
        return query_free_gpu_memory_with_nvidia_smi()


def query_free_gpu_memory_with_nvidia_smi() -> int:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return 1024

    first_value = result.stdout.strip().splitlines()[0]
    return max(1, int(first_value))


if __name__ == "__main__":
    main()
