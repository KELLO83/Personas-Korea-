import polars as pl
import pytest

from ops.graph import build_graph as ops_build_graph
from ops.graph import build_graph


def test_ops_import_points_to_build_graph_module() -> None:
    assert build_graph is ops_build_graph


def test_load_to_neo4j_requires_display_name_column() -> None:
    df = pl.DataFrame({"uuid": ["uuid-1"]})

    with pytest.raises(ValueError, match="display_name column is required"):
        build_graph._load_to_neo4j(df=df, reset=False, batch_size=1000)


def test_load_to_neo4j_requires_at_least_one_display_name() -> None:
    df = pl.DataFrame({"uuid": ["uuid-1"], "display_name": [None]})

    with pytest.raises(ValueError, match="at least one non-null"):
        build_graph._load_to_neo4j(df=df, reset=False, batch_size=1000)


def test_load_to_neo4j_passes_display_name_to_graph_loader(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeGraphLoader:
        def create_schema(self) -> None:
            captured["schema_created"] = True

        def clear_graph(self) -> int:
            captured["cleared"] = True
            return 0

        def load_personas(self, df: pl.DataFrame, batch_size: int = 1000) -> int:
            captured["batch_size"] = batch_size
            captured["display_names"] = df.get_column("display_name").to_list()
            return df.height

        def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr(build_graph, "GraphLoader", FakeGraphLoader)
    df = pl.DataFrame({"uuid": ["uuid-1"], "display_name": ["최은지"]})

    build_graph._load_to_neo4j(df=df, reset=True, batch_size=123)

    assert captured == {
        "schema_created": True,
        "cleared": True,
        "batch_size": 123,
        "display_names": ["최은지"],
        "closed": True,
    }


def test_preprocess_single_chunk_derives_display_name() -> None:
    df = pl.DataFrame(
        {
            "uuid": ["73f75d42a3934626b0d9a4bff062715a"],
            "persona": ["최은지 씨는 서초구의 회계 사무원입니다."],
            "bachelors_field": ["해당없음"],
        }
    )

    result = build_graph._preprocess_single_chunk(df, fast_mode=False)

    assert result.get_column("display_name").to_list() == ["최은지"]
