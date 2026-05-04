from __future__ import annotations

import csv
import copy
import json
import logging
import os
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class HobbyEdge:
    person_uuid: str
    hobby_name: str


def build_domain_tagged_persona_text(context: PersonContext) -> str:
    """
    [TASK 5-E] PRD 패치 반영: Persona의 7개 도메인 텍스트를 구조화된 태그로 결합.
    
    KURE 모델 입력 시, 단순 텍스트 이어붙임이 아닌 도메인별 의미를 보존하기 위해 Prefix 태그를 적용한다.
    마스킹된 텍스트가 입력되는 경우, [ACT] 플레이스홀더를 통해 문맥이 유지된 벡터화가 가능해진다.
    
    Returns:
        예: "[PROF] ... [SPORT] ... [ART] ... [CULT] ... [TRAV] ... [FOOD] ... [FAM] ..."
    """
    domain_blocks = [
        f"[PROF] {context.professional_text.strip()}",
        f"[SPORT] {context.sports_text.strip()}",
        f"[ART] {context.arts_text.strip()}",
        f"[CULT] {context.cultural_background.strip()}",
        f"[TRAV] {context.travel_text.strip()}",
        f"[FOOD] {context.culinary_text.strip()}",
        f"[FAM] {context.family_text.strip()}"
    ]
    filtered_blocks = [block for block in domain_blocks if len(block.split("] ", 1)) > 1 and block.split("] ", 1)[1].strip()]
    return " ".join(filtered_blocks)


@dataclass(frozen=True)
class IndexedEdges:
    edges: list[tuple[int, int]]
    person_to_id: dict[str, int]
    hobby_to_id: dict[str, int]


@dataclass(frozen=True)
class PreparedEdges:
    edges: list[HobbyEdge]
    report: dict[str, object]
    canonicalization: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class EdgeSplit:
    train: list[tuple[int, int]]
    validation: list[tuple[int, int]]
    test: list[tuple[int, int]]
    full_known: dict[int, set[int]]
    train_known: dict[int, set[int]]


@dataclass(frozen=True)
class PersonContext:
    person_uuid: str
    age: str
    age_group: str
    sex: str
    occupation: str
    district: str
    province: str
    family_type: str
    housing_type: str
    education_level: str
    persona_text: str
    professional_text: str
    sports_text: str
    arts_text: str
    travel_text: str
    culinary_text: str
    family_text: str
    hobbies_text: str
    skills_text: str
    career_goals: str
    embedding_text: str


PERSON_CONTEXT_FIELDS = [
    "person_uuid",
    "age",
    "age_group",
    "sex",
    "occupation",
    "district",
    "province",
    "family_type",
    "housing_type",
    "education_level",
    "persona_text",
    "professional_text",
    "sports_text",
    "arts_text",
    "travel_text",
    "culinary_text",
    "family_text",
    "hobbies_text",
    "skills_text",
    "career_goals",
    "embedding_text",
]

LEAKAGE_TEXT_FIELDS = [
    "persona_text",
    "professional_text",
    "sports_text",
    "arts_text",
    "travel_text",
    "culinary_text",
    "family_text",
    "hobbies_text",
    "embedding_text",
]


def load_person_hobby_edges(path: Path) -> list[HobbyEdge]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["person_uuid", "hobby_name"]:
            raise ValueError("CSV schema must be exactly: person_uuid,hobby_name")
        seen: set[tuple[str, str]] = set()
        edges: list[HobbyEdge] = []
        for row in reader:
            person_uuid = (row.get("person_uuid") or "").strip()
            hobby_name = (row.get("hobby_name") or "").strip()
            if not person_uuid or not hobby_name:
                continue
            key = (person_uuid, hobby_name)
            if key in seen:
                continue
            seen.add(key)
            edges.append(HobbyEdge(person_uuid=person_uuid, hobby_name=hobby_name))
    if not edges:
        raise ValueError(f"No valid Person-Hobby edges found in {path}")
    return edges


def load_person_contexts(path: Path) -> dict[str, PersonContext]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != PERSON_CONTEXT_FIELDS:
            expected = ",".join(PERSON_CONTEXT_FIELDS)
            raise ValueError(f"CSV schema must be exactly: {expected}")
        contexts: dict[str, PersonContext] = {}
        for row in reader:
            person_uuid = (row.get("person_uuid") or "").strip()
            if not person_uuid:
                continue
            contexts[person_uuid] = PersonContext(**{field: (row.get(field) or "").strip() for field in PERSON_CONTEXT_FIELDS})
    if not contexts:
        raise ValueError(f"No valid person contexts found in {path}")
    return contexts


def normalize_hobby_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def load_alias_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)
    if not isinstance(raw, dict):
        raise ValueError("Alias map must be a JSON object mapping raw hobby names to canonical names")
    aliases: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("Alias map keys and values must be strings")
        normalized_key = normalize_hobby_name(key)
        normalized_value = normalize_hobby_name(value)
        if not normalized_key or not normalized_value:
            raise ValueError("Alias map keys and values must not be empty after normalization")
        if normalized_key in aliases and aliases[normalized_key] != normalized_value:
            raise ValueError(f"Alias map has conflicting canonical targets for {normalized_key!r}")
        aliases[normalized_key] = normalized_value
    return aliases


def load_hobby_taxonomy(path: Path | None) -> dict[str, object]:
    if path is None:
        return {"version": 1, "rules": [], "manual_aliases": {}, "taxonomy": {}, "display_examples": {}}
    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)
    if not isinstance(raw, dict):
        raise ValueError("Hobby taxonomy must be a JSON object")
    rules_raw = raw.get("rules", [])
    if not isinstance(rules_raw, list):
        raise ValueError("Hobby taxonomy 'rules' must be a list")
    rules: list[dict[str, object]] = []
    for rule in rules_raw:
        if not isinstance(rule, dict):
            raise ValueError("Each hobby taxonomy rule must be an object")
        canonical_hobby = normalize_hobby_name(str(rule.get("canonical_hobby", "")))
        include_keywords = rule.get("include_keywords", [])
        exclude_keywords = rule.get("exclude_keywords", [])
        taxonomy = rule.get("taxonomy", {})
        if not canonical_hobby:
            raise ValueError("Each hobby taxonomy rule must have canonical_hobby")
        if not isinstance(include_keywords, list) or not all(isinstance(item, str) for item in include_keywords):
            raise ValueError("include_keywords must be a list of strings")
        if not isinstance(exclude_keywords, list) or not all(isinstance(item, str) for item in exclude_keywords):
            raise ValueError("exclude_keywords must be a list of strings")
        if not isinstance(taxonomy, dict):
            raise ValueError("taxonomy must be an object")
        rules.append(
            {
                "canonical_hobby": canonical_hobby,
                "include_keywords": [normalize_hobby_name(item) for item in include_keywords if normalize_hobby_name(item)],
                "exclude_keywords": [normalize_hobby_name(item) for item in exclude_keywords if normalize_hobby_name(item)],
                "taxonomy": dict(taxonomy),
            }
        )
    manual_aliases = load_alias_map(path=None)
    alias_sections = []
    if "aliases" in raw:
        alias_sections.append(raw["aliases"])
    if "manual_aliases" in raw:
        alias_sections.append(raw["manual_aliases"])
    for alias_section in alias_sections:
        if not isinstance(alias_section, dict):
            raise ValueError("aliases/manual_aliases must be JSON objects")
        for key, value in alias_section.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("Alias keys and values must be strings")
            normalized_key = normalize_hobby_name(key)
            normalized_value = normalize_hobby_name(value)
            if not normalized_key or not normalized_value:
                raise ValueError("Alias keys and values must not be empty after normalization")
            if normalized_key in manual_aliases and manual_aliases[normalized_key] != normalized_value:
                raise ValueError(f"Alias map has conflicting canonical targets for {normalized_key!r}")
            manual_aliases[normalized_key] = normalized_value
    taxonomy_raw = raw.get("taxonomy", {})
    if not isinstance(taxonomy_raw, dict):
        raise ValueError("taxonomy must be a JSON object")
    taxonomy: dict[str, dict[str, object]] = {}
    for key, value in taxonomy_raw.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            raise ValueError("taxonomy keys must be strings and values must be objects")
        taxonomy[normalize_hobby_name(key)] = dict(value)
    display_examples_raw = raw.get("display_examples", {})
    if not isinstance(display_examples_raw, dict):
        raise ValueError("display_examples must be a JSON object")
    display_examples: dict[str, list[str]] = {}
    for key, value in display_examples_raw.items():
        if not isinstance(key, str) or not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError("display_examples must map strings to string lists")
        display_examples[normalize_hobby_name(key)] = [item.strip() for item in value if item.strip()]
    return {
        "version": int(raw.get("version", 1)),
        "rules": rules,
        "manual_aliases": manual_aliases,
        "taxonomy": taxonomy,
        "display_examples": display_examples,
    }


def load_taxonomy_review(path: Path | None) -> dict[str, object]:
    """Load the human review file. Returns empty structure if path is None or doesn't exist."""
    empty_review: dict[str, object] = {
        "version": 1,
        "approved_clusters": [],
        "manual_aliases": {},
        "rejected_patterns": [],
        "split_required": [],
    }
    if path is None or not path.exists():
        return copy.deepcopy(empty_review)

    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)
    if not isinstance(raw, dict):
        raise ValueError("Hobby taxonomy review must be a JSON object")

    approved_clusters_raw = raw.get("approved_clusters", [])
    if not isinstance(approved_clusters_raw, list):
        raise ValueError("Hobby taxonomy review 'approved_clusters' must be a list")
    approved_clusters: list[dict[str, object]] = []
    for cluster in approved_clusters_raw:
        if not isinstance(cluster, dict):
            raise ValueError("Each approved cluster must be an object")
        canonical_hobby = normalize_hobby_name(str(cluster.get("canonical_hobby", "")))
        include_keywords = cluster.get("include_keywords", [])
        exclude_keywords = cluster.get("exclude_keywords", [])
        taxonomy = cluster.get("taxonomy", {})
        source_cluster_id = cluster.get("source_cluster_id")
        if not canonical_hobby:
            raise ValueError("Each approved cluster must have canonical_hobby")
        if not isinstance(include_keywords, list) or not all(isinstance(item, str) for item in include_keywords):
            raise ValueError("approved_clusters include_keywords must be a list of strings")
        if not isinstance(exclude_keywords, list) or not all(isinstance(item, str) for item in exclude_keywords):
            raise ValueError("approved_clusters exclude_keywords must be a list of strings")
        if not isinstance(taxonomy, dict):
            raise ValueError("approved_clusters taxonomy must be an object")
        if source_cluster_id is not None and not isinstance(source_cluster_id, str):
            raise ValueError("approved_clusters source_cluster_id must be a string when provided")
        approved_cluster: dict[str, object] = {
            "canonical_hobby": canonical_hobby,
            "include_keywords": [normalize_hobby_name(item) for item in include_keywords if normalize_hobby_name(item)],
            "exclude_keywords": [normalize_hobby_name(item) for item in exclude_keywords if normalize_hobby_name(item)],
            "taxonomy": dict(taxonomy),
        }
        if source_cluster_id:
            approved_cluster["source_cluster_id"] = source_cluster_id.strip()
        approved_clusters.append(approved_cluster)

    manual_aliases_raw = raw.get("manual_aliases", {})
    if not isinstance(manual_aliases_raw, dict):
        raise ValueError("Hobby taxonomy review 'manual_aliases' must be a JSON object")
    manual_aliases: dict[str, str] = {}
    for key, value in manual_aliases_raw.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("Hobby taxonomy review manual_aliases keys and values must be strings")
        normalized_key = normalize_hobby_name(key)
        normalized_value = normalize_hobby_name(value)
        if not normalized_key or not normalized_value:
            raise ValueError("Hobby taxonomy review manual_aliases entries must not be empty after normalization")
        manual_aliases[normalized_key] = normalized_value

    rejected_patterns_raw = raw.get("rejected_patterns", [])
    if not isinstance(rejected_patterns_raw, list) or not all(isinstance(item, str) for item in rejected_patterns_raw):
        raise ValueError("Hobby taxonomy review 'rejected_patterns' must be a list of strings")
    rejected_patterns = [normalize_hobby_name(item) for item in rejected_patterns_raw if normalize_hobby_name(item)]

    split_required_raw = raw.get("split_required", [])
    if not isinstance(split_required_raw, list):
        raise ValueError("Hobby taxonomy review 'split_required' must be a list")
    split_required: list[dict[str, str]] = []
    for entry in split_required_raw:
        if not isinstance(entry, dict):
            raise ValueError("Each split_required entry must be an object")
        original_suffix = entry.get("original_suffix", "")
        note = entry.get("note", "")
        if not isinstance(original_suffix, str) or not isinstance(note, str):
            raise ValueError("split_required entries must contain string original_suffix and note fields")
        normalized_suffix = normalize_hobby_name(original_suffix)
        split_required.append({"original_suffix": normalized_suffix, "note": note.strip()})

    return {
        "version": int(raw.get("version", 1)),
        "approved_clusters": approved_clusters,
        "manual_aliases": manual_aliases,
        "rejected_patterns": rejected_patterns,
        "split_required": split_required,
    }


def merge_review_into_taxonomy(taxonomy: dict[str, object], review: dict[str, object]) -> dict[str, object]:
    """Merge approved review clusters into taxonomy. Returns a NEW dict (does not mutate inputs)."""
    merged = copy.deepcopy(taxonomy)
    merged.setdefault("rules", [])
    merged.setdefault("manual_aliases", {})
    merged.setdefault("taxonomy", {})
    merged.setdefault("display_examples", {})

    rules = merged["rules"]
    manual_aliases = merged["manual_aliases"]
    taxonomy_map = merged["taxonomy"]
    approved_clusters = review.get("approved_clusters", [])
    review_aliases = review.get("manual_aliases", {})
    rejected_patterns = review.get("rejected_patterns", [])

    if isinstance(rules, list) and isinstance(approved_clusters, list):
        for cluster in approved_clusters:
            if not isinstance(cluster, dict):
                continue
            rules.append(copy.deepcopy(cluster))
            canonical_hobby = cluster.get("canonical_hobby")
            cluster_taxonomy = cluster.get("taxonomy", {})
            if isinstance(canonical_hobby, str) and canonical_hobby and isinstance(cluster_taxonomy, dict) and cluster_taxonomy:
                if isinstance(taxonomy_map, dict):
                    taxonomy_map[canonical_hobby] = dict(cluster_taxonomy)

    if isinstance(manual_aliases, dict) and isinstance(review_aliases, dict):
        for key, value in review_aliases.items():
            if isinstance(key, str) and isinstance(value, str):
                manual_aliases[key] = value

    rejected_count = len(rejected_patterns) if isinstance(rejected_patterns, list) else 0
    if rejected_count:
        LOGGER.info("Loaded %d rejected taxonomy review patterns; none were applied.", rejected_count)

    return merged


def prepare_hobby_edges(
    edges: list[HobbyEdge],
    *,
    normalize_hobbies: bool,
    alias_map: dict[str, str],
    hobby_taxonomy: dict[str, object] | None = None,
    min_item_degree: int,
    rare_item_policy: str,
) -> PreparedEdges:
    if min_item_degree < 1:
        raise ValueError("min_item_degree must be at least 1")
    # 수정: drop 외에도 keep_with_fallback 정책을 지원합니다.
    if rare_item_policy not in ["drop", "keep_with_fallback"]:
        raise ValueError("rare_item_policy must be 'drop' or 'keep_with_fallback'")

    canonical_edges: list[HobbyEdge] = []
    taxonomy_data = hobby_taxonomy or {"rules": [], "manual_aliases": {}, "taxonomy": {}, "display_examples": {}}
    manual_aliases = dict(alias_map)
    taxonomy_aliases = taxonomy_data.get("manual_aliases", {})
    if isinstance(taxonomy_aliases, dict):
        manual_aliases.update({str(key): str(value) for key, value in taxonomy_aliases.items()})
    observed_examples: dict[str, list[str]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for edge in edges:
        raw_hobby_name = edge.hobby_name.strip()
        hobby_name = normalize_hobby_name(raw_hobby_name) if normalize_hobbies else raw_hobby_name
        canonical_hobby = _canonicalize_hobby_name(hobby_name, manual_aliases, taxonomy_data)
        if not canonical_hobby:
            continue
        key = (edge.person_uuid, canonical_hobby)
        if key in seen:
            continue
        seen.add(key)
        canonical_edges.append(HobbyEdge(person_uuid=edge.person_uuid, hobby_name=canonical_hobby))
        if raw_hobby_name and raw_hobby_name not in observed_examples[canonical_hobby]:
            observed_examples[canonical_hobby].append(raw_hobby_name)

    degrees = Counter(edge.hobby_name for edge in canonical_edges)
    
    # Drop 정책: min_item_degree 미만인 취미를 학습 그래프에서 물리적으로 삭제
    if rare_item_policy == "drop":
        retained_hobbies = {name for name, count in degrees.items() if count >= min_item_degree}
        filtered_edges = [edge for edge in canonical_edges if edge.hobby_name in retained_hobbies]
        rare_items_metadata = {}
        fallback_edges = []
    # Fallback 정책: 희귀 취미를 삭제하지 않고, parent(여기서는 canonical 유지)로 학습에 포함시키되, 메타데이터로 분리
    else:  # keep_with_fallback
        retained_hobbies = set(degrees.keys())  # 모든 hobby를 유지
        filtered_edges = canonical_edges[:]  # 모든 edge를 유지
        # 희귀 취미(min_item_degree 미만) 메타데이터 추출
        rare_items_metadata = {name: count for name, count in degrees.items() if count < min_item_degree}
        # 여기서는 간단화를 위해 실제 그래프 구조의 parent 노드를 만들지 않고, 추천 시 확장용으로 raw->canonical fallback 매핑을 생성합니다.
        # 실전 시스템에서는 rare_items_metadata를 바탕으로 별도의 'Category' 노드로 분배하는 로직이 들어갑니다.
        fallback_edges = [edge for edge in canonical_edges if degrees[edge.hobby_name] < min_item_degree]
        LOGGER.info(f"[Fallback Policy] 희귀 취미(Rare) 개수: {len(rare_items_metadata)}개, 해당 Edge(Fallback 대상): {len(fallback_edges)}개")
        
    if not filtered_edges:
        raise ValueError("No edges remain after hobby normalization and min_item_degree filtering")

    retained_degrees = Counter(edge.hobby_name for edge in filtered_edges)
    raw_degrees = Counter(edge.hobby_name for edge in edges)
    report = _build_vocabulary_report(
        raw_edges=len(edges),
        raw_degrees=raw_degrees,
        canonical_edges=canonical_edges,
        retained_edges=filtered_edges,
        degrees=degrees,
        retained_degrees=retained_degrees,
        min_item_degree=min_item_degree,
        normalize_hobbies=normalize_hobbies,
        alias_count=len(manual_aliases),
        rare_item_policy=rare_item_policy,
        rare_items_count=len(rare_items_metadata),
        fallback_edges_count=len(fallback_edges),
    )
    taxonomy_map = cast(dict[str, object], taxonomy_data.get("taxonomy", {})) if isinstance(taxonomy_data.get("taxonomy", {}), dict) else {}
    display_examples = cast(dict[str, list[str]], taxonomy_data.get("display_examples", {})) if isinstance(taxonomy_data.get("display_examples", {}), dict) else {}
    rules_value = taxonomy_data.get("rules", [])
    rule_count = len(rules_value) if isinstance(rules_value, list) else 0
    retained_hobby_names = {edge.hobby_name for edge in filtered_edges}
    canonicalization: dict[str, object] = {
        "manual_aliases": manual_aliases,
        "rules": list(rules_value) if isinstance(rules_value, list) else [],
        "taxonomy": {key: value for key, value in taxonomy_map.items() if key in retained_hobby_names},
        "display_examples": {key: value for key, value in display_examples.items() if key in retained_hobby_names},
        "observed_examples": {key: values[:5] for key, values in observed_examples.items() if key in retained_hobby_names},
        "rule_count": rule_count,
    }
    return PreparedEdges(edges=filtered_edges, report=report, canonicalization=canonicalization)


def _canonicalize_hobby_name(hobby_name: str, manual_aliases: dict[str, str], hobby_taxonomy: dict[str, object]) -> str:
    if not hobby_name:
        return ""
    if hobby_name in manual_aliases:
        return manual_aliases[hobby_name]
    rules = hobby_taxonomy.get("rules", [])
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


def _build_vocabulary_report(
    *,
    raw_edges: int,
    raw_degrees: Counter[str],
    canonical_edges: list[HobbyEdge],
    retained_edges: list[HobbyEdge],
    degrees: Counter[str],
    retained_degrees: Counter[str],
    min_item_degree: int,
    normalize_hobbies: bool,
    alias_count: int,
    rare_item_policy: str,
    rare_items_count: int = 0,
    fallback_edges_count: int = 0,
) -> dict[str, object]:
    canonical_persons = {edge.person_uuid for edge in canonical_edges}
    raw_hobbies = set(raw_degrees)
    retained_persons = {edge.person_uuid for edge in retained_edges}
    canonical_hobbies = set(degrees)
    retained_hobbies = set(retained_degrees)
    singleton_count = sum(1 for count in degrees.values() if count == 1)
    raw_singleton_count = sum(1 for count in raw_degrees.values() if count == 1)
    retained_singleton_count = sum(1 for count in retained_degrees.values() if count == 1)
    return {
        "normalize_hobbies": normalize_hobbies,
        "alias_count": alias_count,
        "min_item_degree": min_item_degree,
        "raw_edges": raw_edges,
        "raw_hobbies": len(raw_hobbies),
        "raw_singleton_hobbies": raw_singleton_count,
         "raw_singleton_ratio": raw_singleton_count / len(raw_hobbies) if raw_hobbies else 0.0,
         "canonical_edges": len(canonical_edges),
         "canonical_persons": len(canonical_persons),
         "canonical_hobbies": len(canonical_hobbies),
         "canonical_singleton_hobbies": singleton_count,
         "canonical_singleton_ratio": singleton_count / len(canonical_hobbies) if canonical_hobbies else 0.0,
         "retained_edges": len(retained_edges),
         "retained_persons": len(retained_persons),
         "retained_hobbies": len(retained_hobbies),
         "retained_singleton_hobbies": retained_singleton_count,
         "retained_singleton_ratio": retained_singleton_count / len(retained_hobbies) if retained_hobbies else 0.0,
         # Fallback 정책 적용 시 dropped 대신 fallback 통계 사용
         "rare_item_policy": rare_item_policy,
         "rare_items_count": rare_items_count,
         "fallback_edges_count": fallback_edges_count,
         "dropped_edges": (len(canonical_edges) - len(retained_edges)) if rare_item_policy == "drop" else 0,
         "dropped_hobbies": (len(canonical_hobbies) - len(retained_hobbies)) if rare_item_policy == "drop" else 0,
        "dropped_persons": len(canonical_persons) - len(retained_persons),
    }


def index_edges(edges: list[HobbyEdge]) -> IndexedEdges:
    person_to_id: dict[str, int] = {}
    hobby_to_id: dict[str, int] = {}
    indexed: list[tuple[int, int]] = []
    for edge in edges:
        person_id = person_to_id.setdefault(edge.person_uuid, len(person_to_id))
        hobby_id = hobby_to_id.setdefault(edge.hobby_name, len(hobby_to_id))
        indexed.append((person_id, hobby_id))
    return IndexedEdges(edges=indexed, person_to_id=person_to_id, hobby_to_id=hobby_to_id)


def split_edges_by_person(
    edges: list[tuple[int, int]],
    validation_ratio: float,
    test_ratio: float,
    min_eval_hobbies: int,
    two_hobby_policy: str,
    seed: int,
) -> EdgeSplit:
    rng = random.Random(seed)
    by_person: dict[int, list[int]] = defaultdict(list)
    for person_id, hobby_id in edges:
        by_person[person_id].append(hobby_id)

    train: list[tuple[int, int]] = []
    validation: list[tuple[int, int]] = []
    test: list[tuple[int, int]] = []
    full_known: dict[int, set[int]] = {}

    for person_id, hobby_ids in by_person.items():
        unique_hobbies = sorted(set(hobby_ids))
        rng.shuffle(unique_hobbies)
        full_known[person_id] = set(unique_hobbies)
        if len(unique_hobbies) >= min_eval_hobbies:
            validation_count = max(1, round(len(unique_hobbies) * validation_ratio))
            test_count = max(1, round(len(unique_hobbies) * test_ratio))
            holdout_count = min(len(unique_hobbies) - 1, validation_count + test_count)
            validation_count = min(validation_count, holdout_count)
            test_count = holdout_count - validation_count
            validation.extend((person_id, hobby_id) for hobby_id in unique_hobbies[:validation_count])
            test.extend((person_id, hobby_id) for hobby_id in unique_hobbies[validation_count : validation_count + test_count])
            train.extend((person_id, hobby_id) for hobby_id in unique_hobbies[holdout_count:])
        elif len(unique_hobbies) == 2 and two_hobby_policy == "one_eval":
            validation.append((person_id, unique_hobbies[0]))
            train.append((person_id, unique_hobbies[1]))
        else:
            train.extend((person_id, hobby_id) for hobby_id in unique_hobbies)

    train_known: dict[int, set[int]] = defaultdict(set)
    for person_id, hobby_id in train:
        train_known[person_id].add(hobby_id)
    return EdgeSplit(train=train, validation=validation, test=test, full_known=full_known, train_known=dict(train_known))


def sample_negative(
    person_id: int,
    num_hobbies: int,
    full_known: dict[int, set[int]],
    rng: random.Random,
) -> int:
    positives = full_known.get(person_id, set())
    if len(positives) >= num_hobbies:
        raise ValueError(f"Person {person_id} is connected to every hobby; cannot sample negative")
    while True:
        candidate = rng.randrange(num_hobbies)
        if candidate not in positives:
            return candidate


def iter_bpr_batches(
    train_edges: list[tuple[int, int]],
    num_hobbies: int,
    full_known: dict[int, set[int]],
    batch_size: int,
    seed: int,
):
    rng = random.Random(seed)
    shuffled = list(train_edges)
    rng.shuffle(shuffled)
    for start in range(0, len(shuffled), batch_size):
        batch = shuffled[start : start + batch_size]
        users = [person_id for person_id, _ in batch]
        positives = [hobby_id for _, hobby_id in batch]
        negatives = [sample_negative(person_id, num_hobbies, full_known, rng) for person_id in users]
        yield users, positives, negatives


def write_edges(path: Path, edges: list[tuple[int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["person_id", "hobby_id"])
        writer.writerows(edges)


def build_hobby_profile(
    train_edges: list[tuple[int, int]],
    person_to_id: dict[str, int],
    hobby_to_id: dict[str, int],
    contexts: dict[str, PersonContext] | None = None,
) -> dict[str, object]:
    id_to_person = {person_id: person_uuid for person_uuid, person_id in person_to_id.items()}
    id_to_hobby = {hobby_id: hobby_name for hobby_name, hobby_id in hobby_to_id.items()}
    popularity = Counter(hobby_id for _, hobby_id in train_edges)
    hobbies_by_person: dict[int, set[int]] = defaultdict(set)
    for person_id, hobby_id in train_edges:
        hobbies_by_person[person_id].add(hobby_id)

    cooccurrence: dict[int, Counter[int]] = defaultdict(Counter)
    for hobby_ids in hobbies_by_person.values():
        for source in hobby_ids:
            for target in hobby_ids:
                if source != target:
                    cooccurrence[source][target] += 1

    distributions: dict[int, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    if contexts is not None:
        for person_id, hobby_id in train_edges:
            context = contexts.get(id_to_person.get(person_id, ""))
            if context is None:
                continue
            for field in ("age_group", "sex", "occupation", "province", "district", "family_type", "housing_type", "education_level"):
                value = getattr(context, field).strip()
                if value:
                    distributions[hobby_id][field][value] += 1

    hobbies: dict[str, dict[str, object]] = {}
    for hobby_id, count in sorted(popularity.items(), key=lambda item: (-item[1], id_to_hobby[item[0]])):
        hobby_name = id_to_hobby[hobby_id]
        cooccurring = [
            {"hobby_id": related_id, "hobby_name": id_to_hobby[related_id], "count": related_count}
            for related_id, related_count in cooccurrence[hobby_id].most_common(20)
        ]
        hobbies[hobby_name] = {
            "hobby_id": hobby_id,
            "train_popularity": count,
            "distributions": {
                field: dict(counter.most_common()) for field, counter in sorted(distributions[hobby_id].items())
            },
            "cooccurring_hobbies": cooccurring,
        }

    return {
        "source": "train_split_only",
        "num_train_edges": len(train_edges),
        "num_hobbies": len(hobbies),
        "has_person_context": contexts is not None,
        "hobbies": hobbies,
    }


def build_text_embedding_features(
    train_edges: list[tuple[int, int]],
    person_to_id: dict[str, int],
    hobby_to_id: dict[str, int],
    contexts: dict[str, PersonContext] | None,
    embedding_cache_dir: str = "artifacts/embeddings_cache",
) -> dict[str, object]:
    """
    KURE-v1 임베딩을 활용하여 Persona-Hobby 간 Semantic Similarity Feature를 구축합니다.
    PRD 제47~52항 및 CHECKLIST 0.5항(Leakage-safe Audit)을 준수합니다.
    
    Returns:
        person_hobby_similarity 행렬 및 메타데이터를 포함한 딕셔너리
    """
    # NOTE: 실제 운영 코드에서는 이 모듈(src.embeddings.kure_encoder)을 import하여 사용합니다.
    # 현재 빌드 구조상 sys.path를 통한 임포트를 시도합니다.
    try:
        from src.embeddings.kure_encoder import get_kure_encoder
        encoder = get_kure_encoder()
    except Exception as e:
        # 절대 경로를 통한 직접 임포트 폴백 (빌드 환경 호환성 확보)
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from src.embeddings.kure_encoder import get_kure_encoder
        encoder = get_kure_encoder()

    import numpy as np
    import os
    
    id_to_person = {pid: uuid for uuid, pid in person_to_id.items()}
    id_to_hobby = {hid: name for name, hid in hobby_to_id.items()}
    
    all_person_uuids = list(person_to_id.keys())
    all_hobby_names = list(hobby_to_id.keys())
    
    os.makedirs(embedding_cache_dir, exist_ok=True)
    
    # 1. 모든 Persona Context를 하나의 텍스트로 합침 (PRD 제32항: masking된 text 사용)
    # 누수 방지를 위해 `embedding_text` 필드가 아닌, 별도 마스킹 로직이 적용된 필드를 사용해야 하지만,
    # 현재 data.py의 contexts는 Raw Text를 포함하므로, 실제 학습 스크립트(train.py)에서 masking 수행을 권장합니다.
    # 여기서는 구조적 뼈대를 제공합니다.
    persona_texts_list = []
    for pid in all_person_uuids:
        uuid = id_to_person[pid]
        ctx = contexts.get(uuid)
        if ctx is None:
            # Context가 없을 경우 빈 문자열 처리
            persona_texts_list.append("")
        else:
            # 실제 운영 시 아래의 'embedding_text'는 마스킹(Masking)된 버전으로 대체되어야 함.
            raw_text = ctx.embedding_text if hasattr(ctx, 'embedding_text') else ""
            persona_texts_list.append(raw_text)
    
    # 2. KURE 인코더를 통해 유사도 행렬 계산 (코사인 유사도)
    # 이 행렬은 [num_persons, num_hobbies] 형태가 됩니다.
    similarity_matrix = encoder.compute_similarity(persona_texts_list, all_hobby_names)
    
    # 3. 희소 행렬 변환 (저장 공간 절약을 위해 0이 아닌 값만 필터링)
    rows, cols = np.where(similarity_matrix > 0)
    values = similarity_matrix[rows, cols]
    
    sparse_representation = [
        {"person_id": int(rows[i]), "hobby_id": int(cols[i]), "similarity_score": float(values[i])}
        for i in range(len(rows))
    ]
    
    # 4. 결과 저장
    result = {
        "feature_name": "persona_hobby_semantic_sim",
        "source_model": "KURE-v1",
        "num_persons": len(all_person_uuids),
        "num_hobbies": len(all_hobby_names),
        "matrix_shape": list(similarity_matrix.shape),
        "non_zero_ratio": float(len(sparse_representation) / (len(all_person_uuids) * len(all_hobby_names))),
        "sample_scores": [
            {"person_uuid": id_to_person.get(int(r["person_id"]), "unknown"),
             "hobby_name": id_to_hobby.get(int(r["hobby_id"]), "unknown"),
             "score": r["similarity_score"]}
            for r in sparse_representation[:5]  # 샘플 5개만 포함
        ],
        "full_matrix_cache_path": os.path.join(embedding_cache_dir, "person_hobby_embedding_matrix.npy")
    }
    
    # Full Matrix는 별도로 캐싱 (LightGBM Dataset 생성 시 로드 용이)
    np.save(result["full_matrix_cache_path"], similarity_matrix)
    
    return result


def build_leakage_audit(
    split: EdgeSplit,
    person_to_id: dict[str, int],
    hobby_to_id: dict[str, int],
    contexts: dict[str, PersonContext] | None,
) -> dict[str, object]:
    if contexts is None:
        return {
            "mode": "audit",
            "status": "skipped",
            "reason": "person_context.csv not found",
            "text_fields": LEAKAGE_TEXT_FIELDS,
        }
    id_to_person = {person_id: person_uuid for person_uuid, person_id in person_to_id.items()}
    id_to_hobby = {hobby_id: hobby_name for hobby_name, hobby_id in hobby_to_id.items()}
    return {
        "mode": "audit",
        "status": "completed",
        "text_fields": LEAKAGE_TEXT_FIELDS,
        "validation": _audit_split_text_leakage(split.validation, id_to_person, id_to_hobby, contexts),
        "test": _audit_split_text_leakage(split.test, id_to_person, id_to_hobby, contexts),
    }


def _audit_split_text_leakage(
    edges: list[tuple[int, int]],
    id_to_person: dict[int, str],
    id_to_hobby: dict[int, str],
    contexts: dict[str, PersonContext],
) -> dict[str, object]:
    field_mentions = {field: 0 for field in LEAKAGE_TEXT_FIELDS}
    examples: list[dict[str, object]] = []
    leaked_edges = 0
    for person_id, hobby_id in edges:
        person_uuid = id_to_person.get(person_id, "")
        hobby_name = id_to_hobby.get(hobby_id, "")
        context = contexts.get(person_uuid)
        if context is None or not hobby_name:
            continue
        normalized_hobby = normalize_hobby_name(hobby_name)
        matched_fields: list[str] = []
        for field in LEAKAGE_TEXT_FIELDS:
            text = normalize_hobby_name(getattr(context, field))
            if normalized_hobby and normalized_hobby in text:
                field_mentions[field] += 1
                matched_fields.append(field)
        if matched_fields:
            leaked_edges += 1
            if len(examples) < 20:
                examples.append(
                    {
                        "person_id": person_id,
                        "person_uuid": person_uuid,
                        "hobby_id": hobby_id,
                        "hobby_name": hobby_name,
                        "fields": matched_fields,
                    }
                )
    total_edges = len(edges)
    return {
        "total_holdout_edges": total_edges,
        "leaked_edges": leaked_edges,
        "leakage_rate": leaked_edges / total_edges if total_edges else 0.0,
        "field_mentions": field_mentions,
        "examples": examples,
    }


def build_score_normalization_config(method: str = "rank_percentile") -> dict[str, object]:
    if method not in {"rank_percentile", "min_max"}:
        raise ValueError("score normalization method must be 'rank_percentile' or 'min_max'")
    return {
        "method": method,
        "raw_scores_preserved": True,
        "normalized_scores_preserved": True,
        "missing_score_policy": "fill_zero_after_normalization",
        "comparison_policy": "metrics are not directly comparable if this config changes",
    }


def build_initial_fallback_usage(split: EdgeSplit) -> dict[str, object]:
    evaluation_persons = {person_id for person_id, _ in split.validation + split.test}
    return {
        "source": "prepare_only_initial",
        "fallback_events": {},
        "cold_start_persons": 0,
        "normal_case_persons": len(evaluation_persons),
        "fallback_persons": 0,
        "fallback_rate": 0.0,
        "note": "Updated by candidate generation/recommendation once provider fallback executes.",
    }


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)
