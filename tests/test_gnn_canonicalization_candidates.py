import json
import sys
from pathlib import Path

from GNN_Neural_Network.scripts import build_canonicalization_candidates


def test_build_canonicalization_candidates_writes_clusters(tmp_path: Path, monkeypatch) -> None:
    edge_csv = tmp_path / "edges.csv"
    edge_csv.write_text(
        "person_uuid,hobby_name\n"
        "p1,석촌호수 주변 산책\n"
        "p2,탄천 산책로 걷기\n"
        "p3,코인 노래방 방문\n"
        "p4,유튜브 시사 및 교양 콘텐츠 시청\n",
        encoding="utf-8",
    )
    taxonomy_path = tmp_path / "taxonomy.json"
    taxonomy_path.write_text(
        json.dumps(
            {
                "version": 1,
                "rules": [
                    {
                        "canonical_hobby": "산책",
                        "include_keywords": ["산책", "걷기"],
                        "exclude_keywords": [],
                        "taxonomy": {"category": "야외활동"},
                    }
                ],
                "manual_aliases": {"코인 노래방 방문": "노래방"},
                "taxonomy": {"산책": {"category": "야외활동"}, "노래방": {"category": "문화콘텐츠"}},
                "display_examples": {"산책": ["석촌호수 주변 산책"]},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "canonicalization_candidates.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_canonicalization_candidates.py",
            "--input",
            str(edge_csv),
            "--taxonomy",
            str(taxonomy_path),
            "--output",
            str(output_path),
        ],
    )

    build_canonicalization_candidates.main()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["summary"]["candidate_clusters"] >= 3
    assert any(cluster["canonical_candidate"] == "산책" for cluster in payload["clusters"])
    assert any(group["pattern"] == "시청" for group in payload["ambiguous_groups"])
