from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import TypedDict, cast

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GNN_Neural_Network.gnn_recommender.data import load_hobby_taxonomy, load_person_hobby_edges, normalize_hobby_name, save_json


GENERIC_TOKENS = {"시청", "감상", "모임", "투어", "체험", "활동", "방문", "관람", "참여", "참가"}
AMBIGUOUS_PATTERNS = ["시청", "감상", "모임", "투어", "체험"]


class SingletonInfo(TypedDict):
    normalized_name: str
    members: list[str]
    display_name: str
    support_edges: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build canonicalization candidate clusters from raw hobby names.")
    parser.add_argument("--input", type=Path, default=Path("GNN_Neural_Network/data/person_hobby_edges.csv"))
    parser.add_argument("--taxonomy", type=Path, default=Path("GNN_Neural_Network/configs/hobby_taxonomy.json"))
    parser.add_argument("--output", type=Path, default=Path("GNN_Neural_Network/artifacts/canonicalization_candidates.json"))
    parser.add_argument("--min-support", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = cast(Path, args.input)
    taxonomy_path = cast(Path, args.taxonomy)
    output_path = cast(Path, args.output)
    min_support = cast(int, args.min_support)
    edges = load_person_hobby_edges(input_path)
    taxonomy = load_hobby_taxonomy(taxonomy_path if taxonomy_path.exists() else None)
    raw_counts = Counter(normalize_hobby_name(edge.hobby_name) for edge in edges)
    grouped: dict[str, list[str]] = defaultdict(list)
    raw_support: dict[str, int] = defaultdict(int)
    for edge in edges:
        raw_name = edge.hobby_name.strip()
        normalized = normalize_hobby_name(raw_name)
        canonical = _canonical_candidate(normalized, taxonomy)
        grouped[canonical].append(raw_name)
        raw_support[canonical] += 1

    existing_rule_clusters: list[dict[str, object]] = []
    unmatched_singletons: dict[str, SingletonInfo] = {}
    for canonical, members in grouped.items():
        unique_members = list(dict.fromkeys(members))
        if _is_existing_rule_cluster(canonical, unique_members, taxonomy):
            confidence = _confidence_for_candidate(canonical, taxonomy)
            existing_rule_clusters.append(
                {
                    "canonical_candidate": canonical,
                    "members": unique_members,
                    "member_count": len(unique_members),
                    "support_edges": raw_support[canonical],
                    "confidence": confidence,
                    "reasons": _reasons_for_candidate(canonical, unique_members, taxonomy),
                    "proposed_rule": _proposed_rule(canonical, taxonomy),
                    "proposed_taxonomy": _proposed_taxonomy(canonical, taxonomy),
                    "display_examples": unique_members[:5],
                    "status": "pending_review",
                    "source": "existing_rule",
                }
            )
            continue
        unmatched_singletons[canonical] = {
            "normalized_name": canonical,
            "members": unique_members,
            "display_name": unique_members[0] if unique_members else canonical,
            "support_edges": raw_support[canonical],
        }

    mined_clusters, absorbed_hobbies = _mine_suffix_clusters(unmatched_singletons, min_support)
    singleton_clusters = _build_singleton_clusters(unmatched_singletons, absorbed_hobbies)

    ordered_clusters = sorted(existing_rule_clusters, key=_cluster_sort_key)
    ordered_clusters.extend(sorted(mined_clusters, key=_cluster_sort_key))
    ordered_clusters.extend(sorted(singleton_clusters, key=_cluster_sort_key))

    clusters: list[dict[str, object]] = []
    for index, cluster in enumerate(ordered_clusters, start=1):
        cluster_with_id = dict(cluster)
        cluster_with_id["cluster_id"] = f"cluster_{index:04d}"
        clusters.append(cluster_with_id)

    ambiguous_groups = _build_ambiguous_groups(raw_counts)
    payload = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "raw_hobbies": len(raw_counts),
            "raw_singleton_ratio": sum(1 for count in raw_counts.values() if count == 1) / len(raw_counts) if raw_counts else 0.0,
            "candidate_clusters": len(clusters),
            "high_confidence_clusters": sum(1 for cluster in clusters if cluster["confidence"] == "high"),
            "medium_confidence_clusters": sum(1 for cluster in clusters if cluster["confidence"] == "medium"),
            "low_confidence_clusters": sum(1 for cluster in clusters if cluster["confidence"] == "low"),
            "mined_cluster_count": sum(1 for cluster in clusters if cluster["source"] == "mined_suffix"),
            "mined_hobby_coverage": len(absorbed_hobbies),
        },
        "clusters": clusters,
        "ambiguous_groups": ambiguous_groups,
    }
    save_json(output_path, payload)
    print(f"Built {len(clusters)} canonicalization candidate clusters")


def _canonical_candidate(hobby_name: str, taxonomy: dict[str, object]) -> str:
    manual_aliases = taxonomy.get("manual_aliases", {})
    if isinstance(manual_aliases, dict) and hobby_name in manual_aliases:
        return str(manual_aliases[hobby_name])
    rules = taxonomy.get("rules", [])
    if isinstance(rules, list):
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            include_keywords = rule.get("include_keywords", [])
            exclude_keywords = rule.get("exclude_keywords", [])
            canonical_hobby = rule.get("canonical_hobby", "")
            if not isinstance(include_keywords, list) or not isinstance(exclude_keywords, list) or not isinstance(canonical_hobby, str):
                continue
            if include_keywords and any(keyword in hobby_name for keyword in include_keywords) and not any(keyword in hobby_name for keyword in exclude_keywords):
                return canonical_hobby
    return hobby_name


def _is_existing_rule_cluster(canonical: str, members: list[str], taxonomy: dict[str, object]) -> bool:
    if _matching_rule(canonical, taxonomy) is not None:
        return True
    normalized_members = {normalize_hobby_name(member) for member in members}
    return any(member != canonical for member in normalized_members)


def _confidence_for_candidate(canonical: str, taxonomy: dict[str, object]) -> str:
    rules = taxonomy.get("rules", [])
    if isinstance(rules, list):
        for rule in rules:
            if isinstance(rule, dict) and rule.get("canonical_hobby") == canonical:
                include_keywords = rule.get("include_keywords", [])
                if isinstance(include_keywords, list):
                    if any(keyword in {"산책", "걷기", "둘레길", "노래방", "맛집", "카페", "영화", "드라마", "넷플릭스", "왓챠"} for keyword in include_keywords):
                        return "high"
                    if any(keyword in {"유튜브", "영상 시청", "콘텐츠 시청"} for keyword in include_keywords):
                        return "medium"
    if any(term in canonical for term in GENERIC_TOKENS):
        return "low"
    return "low"


def _reasons_for_candidate(canonical: str, members: list[str], taxonomy: dict[str, object]) -> list[str]:
    reasons: list[str] = []
    rule = _matching_rule(canonical, taxonomy)
    if rule is not None:
        include_keywords = rule.get("include_keywords", [])
        if isinstance(include_keywords, list) and include_keywords:
            reasons.append("contains_keywords: " + ", ".join(str(item) for item in include_keywords))
    if len(members) > 1:
        reasons.append("multiple raw variants grouped")
    return reasons or ["no strong rule evidence"]


def _proposed_rule(canonical: str, taxonomy: dict[str, object]) -> dict[str, object]:
    rule = _matching_rule(canonical, taxonomy)
    if rule is None:
        return {}
    return {
        "include_keywords": rule.get("include_keywords", []),
        "exclude_keywords": rule.get("exclude_keywords", []),
    }


def _proposed_taxonomy(canonical: str, taxonomy: dict[str, object]) -> dict[str, object]:
    taxonomy_map = taxonomy.get("taxonomy", {})
    if isinstance(taxonomy_map, dict):
        value = taxonomy_map.get(canonical, {})
        if isinstance(value, dict):
            return dict(value)
    return {}


def _matching_rule(canonical: str, taxonomy: dict[str, object]) -> dict[str, object] | None:
    rules = taxonomy.get("rules", [])
    if not isinstance(rules, list):
        return None
    for rule in rules:
        if isinstance(rule, dict) and rule.get("canonical_hobby") == canonical:
            return rule
    return None


def _build_ambiguous_groups(raw_counts: Counter[str]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for pattern in AMBIGUOUS_PATTERNS:
        examples = [name for name, _ in raw_counts.most_common() if pattern in name][:10]
        if examples:
            output.append(
                {
                    "pattern": pattern,
                    "examples": examples,
                    "reason": "too_generic_split_required",
                }
            )
    return output


def _cluster_sort_key(cluster: dict[str, object]) -> tuple[int, str]:
    return (-cast(int, cluster["support_edges"]), str(cluster["canonical_candidate"]))


def _mine_suffix_clusters(singletons: dict[str, SingletonInfo], min_support: int) -> tuple[list[dict[str, object]], set[str]]:
    if min_support <= 0:
        min_support = 1
    suffix_groups: dict[str, set[str]] = defaultdict(set)
    for hobby_name in singletons:
        for suffix in _suffix_candidates(hobby_name):
            suffix_groups[suffix].add(hobby_name)

    candidate_suffixes = {
        suffix: members
        for suffix, members in suffix_groups.items()
        if len(members) >= min_support
    }
    if not candidate_suffixes:
        return [], set()

    while True:
        assignments = _assign_hobbies_to_suffixes(singletons, candidate_suffixes)
        pruned_suffixes = {
            suffix: candidate_suffixes[suffix]
            for suffix, members in assignments.items()
            if len(members) >= min_support
        }
        if len(pruned_suffixes) == len(candidate_suffixes):
            candidate_suffixes = pruned_suffixes
            break
        candidate_suffixes = pruned_suffixes
        if not candidate_suffixes:
            return [], set()

    assignments = _assign_hobbies_to_suffixes(singletons, candidate_suffixes)
    absorbed_hobbies: set[str] = set()
    clusters: list[dict[str, object]] = []
    for suffix, hobbies in sorted(assignments.items(), key=lambda item: (-_cluster_support(singletons, item[1]), item[0])):
        if len(hobbies) < min_support:
            continue
        ordered_hobbies = sorted(hobbies, key=lambda hobby_name: (-singletons[hobby_name]["support_edges"], hobby_name))
        display_members = [singletons[hobby_name]["display_name"] for hobby_name in ordered_hobbies]
        support_edges = _cluster_support(singletons, hobbies)
        absorbed_hobbies.update(hobbies)
        suffix_tokens = suffix.split()
        status = "split_required" if any(token in GENERIC_TOKENS for token in suffix_tokens) else "pending_review"
        clusters.append(
            {
                "canonical_candidate": suffix,
                "members": display_members,
                "member_count": len(display_members),
                "support_edges": support_edges,
                "confidence": _confidence_for_mined_suffix(suffix, len(display_members)),
                "reasons": [f"mined_suffix_group: {suffix}", "multiple raw variants grouped"],
                "proposed_rule": {
                    "include_keywords": suffix_tokens,
                    "exclude_keywords": [],
                },
                "proposed_taxonomy": {},
                "display_examples": display_members[:5],
                "status": status,
                "source": "mined_suffix",
            }
        )
    return clusters, absorbed_hobbies


def _build_singleton_clusters(singletons: dict[str, SingletonInfo], absorbed_hobbies: set[str]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for hobby_name, info in singletons.items():
        if hobby_name in absorbed_hobbies:
            continue
        members = list(info["members"])
        output.append(
            {
                "canonical_candidate": hobby_name,
                "members": members,
                "member_count": len(members),
                "support_edges": info["support_edges"],
                "confidence": "low",
                "reasons": ["no strong rule evidence"],
                "proposed_rule": {},
                "proposed_taxonomy": {},
                "display_examples": members[:5],
                "status": "pending_review",
                "source": "singleton",
            }
        )
    return output


def _suffix_candidates(hobby_name: str) -> list[str]:
    tokens = hobby_name.split()
    suffixes: list[str] = []
    for width in (2, 1):
        if len(tokens) < width:
            continue
        suffix = " ".join(tokens[-width:]).strip()
        if len(suffix) <= 1 or suffix == hobby_name:
            continue
        suffixes.append(suffix)
    return suffixes


def _assign_hobbies_to_suffixes(
    singletons: dict[str, SingletonInfo],
    candidate_suffixes: dict[str, set[str]],
) -> dict[str, list[str]]:
    assignments: dict[str, list[str]] = defaultdict(list)
    ranked_suffixes = sorted(candidate_suffixes, key=lambda suffix: (-len(suffix.split()), -len(suffix), suffix))
    for hobby_name in singletons:
        for suffix in ranked_suffixes:
            if hobby_name in candidate_suffixes[suffix]:
                assignments[suffix].append(hobby_name)
                break
    return assignments


def _cluster_support(singletons: dict[str, SingletonInfo], hobbies: list[str] | set[str]) -> int:
    return sum(singletons[hobby_name]["support_edges"] for hobby_name in hobbies)


def _confidence_for_mined_suffix(suffix: str, member_count: int) -> str:
    if member_count >= 20 and _is_specific_activity_suffix(suffix):
        return "high"
    if member_count >= 10:
        return "medium"
    return "low"


def _is_specific_activity_suffix(suffix: str) -> bool:
    tokens = suffix.split()
    if not tokens:
        return False
    if any(token in GENERIC_TOKENS for token in tokens):
        return False
    activity_lexicon = {
        "걷기",
        "공부",
        "구경",
        "노래방",
        "독서",
        "드라이브",
        "등산",
        "러닝",
        "볼링",
        "산책",
        "수영",
        "여행",
        "요가",
        "운동",
        "자전거",
        "조깅",
        "캠핑",
    }
    last_token = tokens[-1]
    if suffix in activity_lexicon or last_token in activity_lexicon:
        return True
    return last_token.endswith(("하기", "타기", "걷기", "뛰기", "읽기", "보기", "듣기", "치기", "배우기", "만들기"))


if __name__ == "__main__":
    main()
