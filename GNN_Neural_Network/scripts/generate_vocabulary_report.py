"""
50K 데이터를 사용하여 keep_with_fallback 정책이 적용된 vocabulary_report.json 생성
(Phase 5-Pre 필수 선행 작업)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GNN_Neural_Network.gnn_recommender.data import (
    load_person_hobby_edges,
    load_alias_map,
    load_json,
    prepare_hobby_edges,
    save_json,
)
from GNN_Neural_Network.gnn_recommender.config import load_config

def main():
    # 스크립트 위치를 기준으로 프로젝트 루트 계산
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    config_path = project_root / "configs" / "lightgcn_hobby.yaml"
    config = load_config(config_path)
    
    print("[1/3] 50K Person-Hobby edges 로드 중...")
    edges = load_person_hobby_edges(config.paths.edge_csv)
    print(f"  - 로드된 edges: {len(edges)}개")
    
    alias_map = {}
    if config.paths.hobby_aliases.exists():
        alias_map = load_alias_map(config.paths.hobby_aliases)
    
    hobby_taxonomy = None
    if config.paths.hobby_taxonomy.exists():
        hobby_taxonomy = load_json(config.paths.hobby_taxonomy)
    
    print("[2/3] prepare_hobby_edges() 실행 (rare_item_policy=keep_with_fallback)...")
    prepared = prepare_hobby_edges(
        edges,
        normalize_hobbies=config.data.normalize_hobbies,
        alias_map=alias_map,
        hobby_taxonomy=hobby_taxonomy,
        min_item_degree=config.data.min_item_degree,
        rare_item_policy=config.data.rare_item_policy,
    )
    
    print("[3/3] vocabulary_report.json 저장 중...")
    report_path = config.paths.vocabulary_report
    save_json(report_path, prepared.report)
    
    print(f"\n[SUCCESS] 리포트 생성 완료: {report_path}")
    print(f"  - raw_edges: {prepared.report['raw_edges']}")
    print(f"  - canonical_edges: {prepared.report['canonical_edges']}")
    print(f"  - retained_edges: {prepared.report['retained_edges']}")
    print(f"  - rare_items_count: {prepared.report.get('rare_items_count', 'N/A')}")
    print(f"  - fallback_edges_count: {prepared.report.get('fallback_edges_count', 'N/A')}")
    print(f"  - rare_item_policy: {prepared.report['rare_item_policy']}")

if __name__ == "__main__":
    main()
