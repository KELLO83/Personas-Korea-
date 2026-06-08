from __future__ import annotations

import hashlib
import json

import numpy as np

from experiments.hobby_recommender_ml.hobby_recommender.embedding_cache import HobbyEmbeddingCache, PersonEmbeddingCache
from experiments.hobby_recommender_ml.hobby_recommender import embedding_cache


def test_person_embedding_cache_rejects_revision_mismatch(tmp_path) -> None:
    cache = PersonEmbeddingCache(tmp_path, model_name="model-a", model_revision="rev1")
    cache.set("persona text", np.array([1.0, 2.0], dtype=np.float32))

    same_identity_cache = PersonEmbeddingCache(tmp_path, model_name="model-a", model_revision="rev1")
    assert same_identity_cache.get("persona text") is not None

    different_revision_cache = PersonEmbeddingCache(tmp_path, model_name="model-a", model_revision="rev2")
    assert different_revision_cache.get("persona text") is None


def test_person_embedding_cache_passes_revision_to_loader(tmp_path, monkeypatch) -> None:
    calls = []

    class FakeModel:
        def encode(self, text, **_kwargs):
            assert text == "persona text"
            return np.array([1.0, 2.0], dtype=np.float32)

    def fake_loader(
        device=None,
        *,
        model_name,
        model_revision="",
        attention_implementation="",
        torch_dtype="",
        torch_compile=False,
        torch_compile_mode="",
    ):
        calls.append((device, model_name, model_revision, attention_implementation, torch_dtype, torch_compile, torch_compile_mode))
        return FakeModel()

    monkeypatch.setattr(embedding_cache, "_load_kure_model", fake_loader)

    cache = PersonEmbeddingCache(
        tmp_path,
        model_name="dragonkue/snowflake-arctic-embed-l-v2.0-ko",
        model_revision="rev-a",
        device="cpu",
        attention_implementation="sdpa",
        torch_dtype="float16",
    )
    cache.encode("persona text")

    assert calls == [("cpu", "dragonkue/snowflake-arctic-embed-l-v2.0-ko", "rev-a", "sdpa", "float16", False, "reduce-overhead")]


def test_person_embedding_cache_rejects_attention_implementation_mismatch(tmp_path) -> None:
    cache = PersonEmbeddingCache(tmp_path, model_name="model-a", attention_implementation="sdpa")
    cache.set("persona text", np.array([1.0, 2.0], dtype=np.float32))

    same_attention_cache = PersonEmbeddingCache(tmp_path, model_name="model-a", attention_implementation="sdpa")
    assert same_attention_cache.get("persona text") is not None

    eager_cache = PersonEmbeddingCache(tmp_path, model_name="model-a", attention_implementation="")
    assert eager_cache.get("persona text") is None


def test_person_embedding_cache_rejects_torch_dtype_mismatch(tmp_path) -> None:
    cache = PersonEmbeddingCache(tmp_path, model_name="model-a", torch_dtype="float16")
    cache.set("persona text", np.array([1.0, 2.0], dtype=np.float32))

    matching_cache = PersonEmbeddingCache(tmp_path, model_name="model-a", torch_dtype="float16")
    assert matching_cache.get("persona text") is not None

    fp32_cache = PersonEmbeddingCache(tmp_path, model_name="model-a", torch_dtype="float32")
    assert fp32_cache.get("persona text") is None


def test_person_embedding_cache_treats_corrupt_npy_as_miss(tmp_path) -> None:
    cache = PersonEmbeddingCache(tmp_path, model_name="model-a")
    cache.set("persona text", np.array([1.0, 2.0], dtype=np.float32))
    cache_path = cache._cache_path("persona text")
    assert cache_path is not None
    cache_path.write_bytes(b"not-a-valid-npy")

    fresh_cache = PersonEmbeddingCache(tmp_path, model_name="model-a")
    assert fresh_cache.get("persona text") is None


def test_person_embedding_cache_skips_legacy_cache_for_non_default_identity(tmp_path) -> None:
    text = "persona text"
    legacy_key = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    legacy_path = tmp_path / f"person_emb_{legacy_key}.npy"
    np.save(legacy_path, np.array([1.0, 2.0], dtype=np.float32))

    explicit_legacy_cache = PersonEmbeddingCache(tmp_path, attention_implementation="", torch_dtype="float32")
    assert explicit_legacy_cache.get(text) is not None

    sdpa_cache = PersonEmbeddingCache(tmp_path)
    assert sdpa_cache.get(text) is None

    revisioned_cache = PersonEmbeddingCache(tmp_path, model_revision="rev1")
    assert revisioned_cache.get(text) is None


def test_hobby_embedding_cache_rejects_tampered_metadata(tmp_path) -> None:
    cache = HobbyEmbeddingCache(tmp_path, model_name="model-a")
    cache.set("등산", np.array([0.1, 0.2, 0.3], dtype=np.float32))
    cache_path = cache._cache_path("등산")
    assert cache_path is not None
    metadata_path = cache_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["embedding_dim"] = 999
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    fresh_cache = HobbyEmbeddingCache(tmp_path, model_name="model-a")
    assert fresh_cache.get("등산") is None


def test_hobby_embedding_cache_treats_corrupt_npy_as_miss(tmp_path) -> None:
    cache = HobbyEmbeddingCache(tmp_path, model_name="model-a")
    cache.set("hobby", np.array([0.1, 0.2, 0.3], dtype=np.float32))
    cache_path = cache._cache_path("hobby")
    assert cache_path is not None
    cache_path.write_bytes(b"not-a-valid-npy")

    fresh_cache = HobbyEmbeddingCache(tmp_path, model_name="model-a")
    assert fresh_cache.get("hobby") is None


def test_hobby_matrix_cache_rejects_preprocessing_mismatch(tmp_path) -> None:
    hobby_names = ["등산", "요리"]
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    cache = HobbyEmbeddingCache(tmp_path, model_name="model-a", preprocessing_version="raw_v1")
    cache.save_matrix(hobby_names, matrix)

    matching_cache = HobbyEmbeddingCache(tmp_path, model_name="model-a", preprocessing_version="raw_v1")
    loaded, metadata = matching_cache.load_matrix(hobby_names)
    assert loaded is not None
    assert metadata is not None

    changed_preprocessing_cache = HobbyEmbeddingCache(tmp_path, model_name="model-a", preprocessing_version="masked_v1")
    loaded, metadata = changed_preprocessing_cache.load_matrix(hobby_names)
    assert loaded is None
    assert metadata is None


def test_hobby_matrix_cache_rejects_attention_implementation_mismatch(tmp_path) -> None:
    hobby_names = ["등산", "요리"]
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    cache = HobbyEmbeddingCache(tmp_path, model_name="model-a", attention_implementation="sdpa")
    cache.save_matrix(hobby_names, matrix)

    matching_cache = HobbyEmbeddingCache(tmp_path, model_name="model-a", attention_implementation="sdpa")
    loaded, metadata = matching_cache.load_matrix(hobby_names)
    assert loaded is not None
    assert metadata is not None

    eager_cache = HobbyEmbeddingCache(tmp_path, model_name="model-a", attention_implementation="")
    loaded, metadata = eager_cache.load_matrix(hobby_names)
    assert loaded is None
    assert metadata is None


def test_hobby_matrix_cache_rejects_torch_dtype_mismatch(tmp_path) -> None:
    hobby_names = ["등산", "요리"]
    matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    cache = HobbyEmbeddingCache(tmp_path, model_name="model-a", torch_dtype="float16")
    cache.save_matrix(hobby_names, matrix)

    matching_cache = HobbyEmbeddingCache(tmp_path, model_name="model-a", torch_dtype="float16")
    loaded, metadata = matching_cache.load_matrix(hobby_names)
    assert loaded is not None
    assert metadata is not None

    fp32_cache = HobbyEmbeddingCache(tmp_path, model_name="model-a", torch_dtype="float32")
    loaded, metadata = fp32_cache.load_matrix(hobby_names)
    assert loaded is None
    assert metadata is None
