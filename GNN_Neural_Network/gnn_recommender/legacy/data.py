from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..data import (
    EdgeSplit,
    HobbyEdge,
    IndexedEdges,
    PersonContext,
    PreparedEdges,
    build_domain_tagged_persona_text,
    iter_bpr_batches,
    save_json,
)


class DataProcessor:
    def __init__(self, project_root: str | Path | None = None) -> None:
        self.root = Path(__file__).resolve().parents[2] if project_root is None else Path(project_root)

    def load_edges(self, sample_size: int | None = None) -> pd.DataFrame:
        path = self.root / "data" / "person_hobby_edges.csv"
        if not path.exists():
            return pd.DataFrame(columns=["person_uuid", "hobby_name"])
        frame = pd.read_csv(path)
        if sample_size:
            frame = frame.sample(n=min(sample_size, len(frame)), random_state=42)
        return frame


default_processor = DataProcessor()

__all__ = [
    "DataProcessor",
    "EdgeSplit",
    "HobbyEdge",
    "IndexedEdges",
    "PersonContext",
    "PreparedEdges",
    "build_domain_tagged_persona_text",
    "default_processor",
    "iter_bpr_batches",
    "save_json",
]
