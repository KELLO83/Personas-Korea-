import pandas as pd

import run_pipeline


def _mock_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "uuid": "uuid-1",
                "persona": "원본 페르소나 텍스트",
                "embedding_text": "임베딩 텍스트",
            }
        ]
    )


def _load_dataset_noop(sample_size):
    return _mock_dataset()


def _preprocess_passthrough(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy()


def test_run_pipeline_uses_embedding_text_for_vector_encoding(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeEmbedder:
        def encode(self, texts: list[str]) -> list[list[float]]:
            captured["texts"] = texts
            return [[0.0]]

    class FakeVectorIndex:
        stored_rows: list[object] = []

        def __init__(self) -> None:
            self.closed = False

        def create_index(self) -> None:
            return None

        def get_embedded_uuids(self) -> list[str]:
            return []

        def set_embeddings(self, rows: list[object], batch_size: int = 500) -> int:
            self.stored_rows.extend(rows)
            return len(rows)

        def close(self) -> None:
            self.closed = True

    class FakeGraphLoader:
        def create_schema(self) -> None:
            return None

        def load_personas(self, _df: pd.DataFrame, batch_size: int = 1000) -> int:
            return len(_df)

        def count_nodes_by_label(self) -> dict[str, int]:
            return {"Person": 1}

        def close(self) -> None:
            return None

    class FakeService:
        def __init__(self) -> None:
            return None

        def drop_graph(self) -> None:
            return None

        def project_graph(self) -> dict[str, int]:
            return {"nodeCount": 1, "relationshipCount": 0}

        def write_embeddings(self) -> dict[str, int]:
            return {"nodePropertiesWritten": 1}

        def project_graph_with_fastrp_embeddings(self) -> dict[str, int]:
            return {"nodeCount": 1, "relationshipCount": 0}

        def write_knn_relationships(self) -> dict[str, int]:
            return {"relationshipsWritten": 1}

        def write_communities(self) -> dict[str, int]:
            return {"communityCount": 1}

        def close(self) -> None:
            return None

    def fake_build_embedding_rows(_df: pd.DataFrame, _embeddings: list[list[float]]) -> list[tuple[str, list[float]]]:
        return [(row["uuid"], embedding) for row, embedding in zip(_df.to_dict(orient="records"), _embeddings)]

    monkeypatch.setattr(run_pipeline, "load_dataset", lambda sample_size: _load_dataset_noop(sample_size))
    monkeypatch.setattr(run_pipeline, "preprocess", _preprocess_passthrough)
    monkeypatch.setattr(run_pipeline, "KureEmbedder", FakeEmbedder)
    monkeypatch.setattr(run_pipeline, "Neo4jVectorIndex", FakeVectorIndex)
    monkeypatch.setattr(run_pipeline, "build_embedding_rows", fake_build_embedding_rows)
    monkeypatch.setattr(run_pipeline, "GraphLoader", FakeGraphLoader)
    monkeypatch.setattr(run_pipeline, "FastRPService", FakeService)
    monkeypatch.setattr(run_pipeline, "SimilarityService", FakeService)
    monkeypatch.setattr(run_pipeline, "CommunityService", FakeService)

    run_pipeline.run_pipeline(sample_size=5)

    assert captured["texts"] == ["임베딩 텍스트"]


def test_main_defaults_to_full_load(monkeypatch) -> None:
    captured_sample_sizes: list[int | None] = []

    monkeypatch.setattr(run_pipeline, "run_pipeline", lambda sample_size: captured_sample_sizes.append(sample_size))
    monkeypatch.setattr(run_pipeline.sys, "argv", ["run_pipeline.py"])

    run_pipeline.main()

    assert captured_sample_sizes == [None]
