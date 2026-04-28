import numpy as np
import pandas as pd
import pytest

from src.embeddings.kure_model import KureEmbedder, _to_float_vectors
from src.embeddings.vector_index import build_embedding_rows


def test_to_float_vectors_converts_numpy_array_to_python_lists() -> None:
    result = _to_float_vectors(np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64))

    assert result == [[pytest.approx(0.1), pytest.approx(0.2)], [pytest.approx(0.3), pytest.approx(0.4)]]


def test_to_float_vectors_handles_single_vector() -> None:
    result = _to_float_vectors(np.array([0.1, 0.2], dtype=np.float64))

    assert result == [[pytest.approx(0.1), pytest.approx(0.2)]]


def test_build_embedding_rows_pairs_uuid_and_vectors() -> None:
    df = pd.DataFrame([{"uuid": "a"}, {"uuid": "b"}])
    embeddings = [[0.1, 0.2], [0.3, 0.4]]

    result = build_embedding_rows(df, embeddings)

    assert result == [
        {"uuid": "a", "text_embedding": [0.1, 0.2]},
        {"uuid": "b", "text_embedding": [0.3, 0.4]},
    ]


def test_build_embedding_rows_rejects_mismatched_lengths() -> None:
    df = pd.DataFrame([{"uuid": "a"}, {"uuid": "b"}])

    with pytest.raises(ValueError, match="must match"):
        build_embedding_rows(df, [[0.1, 0.2]])


def test_kure_embedder_rejects_cpu_inference() -> None:
    embedder = KureEmbedder(device="cpu")

    with pytest.raises(RuntimeError, match="CPU embedding inference is disabled"):
        embedder._load_model()
