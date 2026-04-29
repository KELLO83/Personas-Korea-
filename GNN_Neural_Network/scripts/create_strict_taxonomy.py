"""Transform loose taxonomy review into strict version.

Removes generic/non-hobby canonicals, redirects sub-hobbies to existing
base canonicals, and merges duplicate canonical pairs.

Usage:
    python GNN_Neural_Network/scripts/create_strict_taxonomy.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

REVIEW_PATH = Path("GNN_Neural_Network/configs/hobby_taxonomy_review.json")
LOOSE_BACKUP = Path("GNN_Neural_Network/configs/hobby_taxonomy_review_loose.json")

REMOVE_CANONICAL: set[str] = {
    "탐방", "만들기", "듣기", "분석", "탐색", "기록", "시도", "작성",
    "수강", "마시기", "돌보기", "쉬기",
    "식사", "시켜 먹기", "주문하기", "주문", "맛보기",
    "담소", "잡담", "전화 통화", "안부 전화", "조합 찾기",
    "가벼운 반주", "소주 한잔",
    "tv 시청하기", "정주행",
}

REDIRECT_CANONICAL: dict[str, str] = {
    "음식 탐방": "맛집/카페 탐방",
    "음식점 탐방": "맛집/카페 탐방",
    "먹거리 탐방": "맛집/카페 탐방",
    "식당 탐방": "맛집/카페 탐방",
    "레스토랑 탐방": "맛집/카페 탐방",
    "양식당 탐방": "맛집/카페 탐방",
    "고기집 탐방": "맛집/카페 탐방",
    "메뉴 탐방": "맛집/카페 탐방",
    "이자카야 탐방": "맛집/카페 탐방",
    "주점 탐방": "맛집/카페 탐방",
    "전문점 탐방": "맛집/카페 탐방",
    "음식 관광": "맛집/카페 탐방",
    "음식 즐기기": "맛집/카페 탐방",
    "시음": "맛집/카페 탐방",
    "시식": "맛집/카페 탐방",
    "맥주 시음": "맛집/카페 탐방",
    "역사 탐방": "전시/역사 관람",
    "경관 탐방": "여행/나들이",
    "서점 탐방": "독서",
    "정독": "독서",
    "만화 정주행": "웹툰 감상",
    "따라 부르기": "노래 부르기",
    "멍 때리기": "낮잠/휴식",
    "사이클링": "자전거 라이딩",
    "라이딩": "자전거 라이딩",
    "채소 가꾸기": "원예/식물 가꾸기",
    "재배": "원예/식물 가꾸기",
    "바둑 두기": "바둑 대국",
    "장기 두기": "장기 대국",
    "골프장 라운딩": "골프 라운딩",
    "야간 러닝": "러닝",
    "아침 조깅": "조깅",
    "새벽 조깅": "조깅",
    "야간 조깅": "조깅",
    "직관": "경기 직관",
    "외모 관리": "피부/뷰티 관리",
    "스케치": "그리기",
    "근력 운동": "웨이트 트레이닝",
}


def main() -> None:
    with REVIEW_PATH.open("r", encoding="utf-8") as f:
        loose = json.load(f)

    shutil.copy2(REVIEW_PATH, LOOSE_BACKUP)
    print(f"Backed up loose version to {LOOSE_BACKUP}")

    old_approved = loose.get("approved_clusters", [])
    old_rejected = set(loose.get("rejected_patterns", []))
    old_split = loose.get("split_required", [])

    kept: list[dict] = []
    redirected: list[dict] = []
    removed_names: list[str] = []

    for cluster in old_approved:
        canonical = cluster.get("canonical_hobby", "")
        if canonical in REMOVE_CANONICAL:
            removed_names.append(canonical)
            continue
        if canonical in REDIRECT_CANONICAL:
            new_cluster = dict(cluster)
            new_cluster["canonical_hobby"] = REDIRECT_CANONICAL[canonical]
            redirected.append(new_cluster)
            continue
        kept.append(cluster)

    new_rejected = sorted(old_rejected | set(removed_names))
    new_approved = kept + redirected

    strict = {
        "version": 2,
        "description": "Strict taxonomy review: generic/non-hobby canonicals removed, sub-hobbies redirected to base canonicals, duplicates merged.",
        "approved_clusters": new_approved,
        "manual_aliases": loose.get("manual_aliases", {}),
        "rejected_patterns": new_rejected,
        "split_required": old_split,
    }

    with REVIEW_PATH.open("w", encoding="utf-8") as f:
        json.dump(strict, f, ensure_ascii=False, indent=2)

    print(f"\n=== STRICT TAXONOMY SUMMARY ===")
    print(f"  Original approved clusters: {len(old_approved)}")
    print(f"  Kept as-is:                 {len(kept)}")
    print(f"  Redirected:                 {len(redirected)}")
    print(f"  Removed:                    {len(removed_names)}")
    print(f"  New approved total:         {len(new_approved)}")
    print(f"  Rejected patterns:          {len(old_rejected)} -> {len(new_rejected)}")
    print()
    print("Removed canonicals:")
    for name in sorted(removed_names):
        print(f"  - {name}")
    print()
    print("Redirected canonicals:")
    for cluster in redirected:
        orig = [c for c in old_approved if c.get("source_cluster_id") == cluster.get("source_cluster_id")]
        orig_name = orig[0]["canonical_hobby"] if orig else "?"
        print(f"  {orig_name:25s} -> {cluster['canonical_hobby']}")


if __name__ == "__main__":
    main()
