from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import TypedDict, cast

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GNN_Neural_Network.gnn_recommender.data import save_json


GENERIC_SINGLE_TOKEN_SUFFIXES = {
    "가꾸기",
    "감상",
    "경기",
    "관리",
    "관람",
    "구경",
    "기르기",
    "꾸미기",
    "노력",
    "모임",
    "먹기",
    "방문",
    "시청",
    "연습",
    "즐기기",
    "참가",
    "참여",
    "체험",
    "투어",
    "활동",
    "키우기",
}

GENERIC_MULTI_TOKEN_PATTERNS = {
    "및 분석",
    "한 잔",
}


class CandidateCluster(TypedDict, total=False):
    canonical_candidate: str
    cluster_id: str
    confidence: str
    member_count: int
    proposed_rule: dict[str, object]
    proposed_taxonomy: dict[str, object]
    reasons: list[str]
    source: str
    status: str


class ApprovedCluster(TypedDict):
    canonical_hobby: str
    include_keywords: list[str]
    exclude_keywords: list[str]
    taxonomy: dict[str, object]
    source_cluster_id: str


class SplitRequiredEntry(TypedDict):
    original_suffix: str
    note: str


class ReviewPayload(TypedDict, total=False):
    version: int
    description: str
    approved_clusters: list[ApprovedCluster]
    manual_aliases: dict[str, str]
    rejected_patterns: list[str]
    split_required: list[SplitRequiredEntry]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-approve high-quality mined canonicalization candidates into hobby_taxonomy_review.json."
    )
    _ = parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("GNN_Neural_Network/artifacts/canonicalization_candidates.json"),
    )
    _ = parser.add_argument(
        "--output",
        type=Path,
        default=Path("GNN_Neural_Network/configs/hobby_taxonomy_review.json"),
    )
    _ = parser.add_argument("--min-members", type=int, default=10)
    _ = parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as file:
        payload = cast(object, json.load(file))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast(dict[str, object], payload)


def get_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            items.append(text)
    return items


def get_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return int(stripped)
    return default


def get_candidate_clusters(payload: dict[str, object]) -> list[CandidateCluster]:
    raw_clusters = payload.get("clusters", [])
    if not isinstance(raw_clusters, list):
        raise ValueError("canonicalization_candidates.json must contain a list at 'clusters'")
    clusters: list[CandidateCluster] = []
    for cluster in raw_clusters:
        if isinstance(cluster, dict):
            clusters.append(cast(CandidateCluster, cast(object, cluster)))
    return clusters


def is_generic_single_token_suffix(name: str) -> bool:
    tokens = name.strip().split()
    return len(tokens) == 1 and tokens[0] in GENERIC_SINGLE_TOKEN_SUFFIXES


def is_generic_multi_token_suffix(name: str) -> bool:
    normalized = name.strip()
    if normalized in GENERIC_MULTI_TOKEN_PATTERNS:
        return True
    tokens = normalized.split()
    return bool(tokens) and any(token == "및" for token in tokens)


def build_approved_cluster(cluster: CandidateCluster) -> ApprovedCluster:
    proposed_rule = cluster.get("proposed_rule", {})
    include_keywords = get_string_list(proposed_rule.get("include_keywords", []))
    exclude_keywords = get_string_list(proposed_rule.get("exclude_keywords", []))
    proposed_taxonomy = cluster.get("proposed_taxonomy", {})

    canonical_hobby = str(cluster.get("canonical_candidate", "")).strip()
    source_cluster_id = str(cluster.get("cluster_id", "")).strip()

    approved: ApprovedCluster = {
        "canonical_hobby": canonical_hobby,
        "include_keywords": include_keywords,
        "exclude_keywords": exclude_keywords,
        "taxonomy": cast(dict[str, object], cast(object, proposed_taxonomy)),
        "source_cluster_id": source_cluster_id,
    }
    return approved


def build_split_required_entry(cluster: CandidateCluster) -> SplitRequiredEntry:
    suffix = str(cluster.get("canonical_candidate", "")).strip()
    reasons = cluster.get("reasons", [])
    if isinstance(reasons, list):
        reason_text = "; ".join(str(item).strip() for item in reasons if str(item).strip())
    else:
        reason_text = ""
    note = reason_text or "Auto-carried from canonicalization candidates: split required"
    return {"original_suffix": suffix, "note": note}


def should_auto_approve(cluster: CandidateCluster, min_members: int) -> bool:
    if str(cluster.get("source", "")) != "mined_suffix":
        return False
    if str(cluster.get("status", "")) != "pending_review":
        return False
    if int(cluster.get("member_count", 0)) < min_members:
        return False
    if str(cluster.get("confidence", "")) not in {"high", "medium"}:
        return False
    canonical_candidate = str(cluster.get("canonical_candidate", "")).strip()
    if not canonical_candidate:
        return False
    if is_generic_single_token_suffix(canonical_candidate) or is_generic_multi_token_suffix(canonical_candidate):
        return False
    return True


def build_review_payload(
    existing_review: dict[str, object],
    approved_clusters: list[ApprovedCluster],
    rejected_patterns: list[str],
    split_required: list[SplitRequiredEntry],
) -> ReviewPayload:
    payload: ReviewPayload = cast(ReviewPayload, cast(object, dict(existing_review)))
    payload["version"] = get_int(existing_review.get("version", 1), default=1)
    if "description" not in payload:
        payload["description"] = (
            "Human review decisions for mined canonicalization candidates. Only approved_clusters become canonicalization rules."
        )
    if "manual_aliases" not in payload or not isinstance(payload["manual_aliases"], dict):
        payload["manual_aliases"] = {}
    payload["approved_clusters"] = approved_clusters
    payload["rejected_patterns"] = rejected_patterns
    payload["split_required"] = split_required
    return payload


def main() -> None:
    args = parse_args()
    candidates_path = cast(Path, args.candidates)
    output_path = cast(Path, args.output)
    min_members = cast(int, args.min_members)
    dry_run = cast(bool, args.dry_run)

    candidates_payload = load_json(candidates_path)
    clusters = get_candidate_clusters(candidates_payload)

    existing_review = load_json(output_path) if output_path.exists() else {}

    approved_clusters = [build_approved_cluster(cluster) for cluster in clusters if should_auto_approve(cluster, min_members)]
    rejected_patterns = sorted(
        {
            str(cluster.get("canonical_candidate", "")).strip()
            for cluster in clusters
            if str(cluster.get("source", "")) == "mined_suffix"
            and str(cluster.get("status", "")) == "split_required"
            and str(cluster.get("canonical_candidate", "")).strip()
        }
    )
    split_required = [
        build_split_required_entry(cluster)
        for cluster in clusters
        if str(cluster.get("source", "")) == "mined_suffix"
        and str(cluster.get("status", "")) == "split_required"
    ]

    review_payload = build_review_payload(existing_review, approved_clusters, rejected_patterns, split_required)

    mined_suffix_total = sum(
        1 for cluster in clusters if str(cluster.get("source", "")) == "mined_suffix"
    )

    print(f"Candidates file: {candidates_path}")
    print(f"Output file: {output_path}")
    print(f"Mined suffix clusters: {mined_suffix_total}")
    print(f"Approved clusters: {len(approved_clusters)}")
    print(f"Rejected patterns: {len(rejected_patterns)}")
    print(f"Split required entries: {len(split_required)}")
    print(f"Min members threshold: {min_members}")

    if dry_run:
        print("Dry run enabled; no file was written.")
        return

    save_json(output_path, review_payload)
    print("Review file updated.")


if __name__ == "__main__":
    main()
