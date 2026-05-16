from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from experiments.persona_similarity.scripts.common import file_sha256, load_config, mark_cache_hit, should_use_cache, stable_json_hash, write_json
from experiments.persona_similarity.scripts.text_feature_builder import TEXT_DOMAINS, build_domain_text, text_hash


STRUCTURED_TEXT_COLUMNS = [
    "persona",
    "professional_persona",
    "sports_persona",
    "arts_persona",
    "travel_persona",
    "culinary_persona",
    "family_persona",
    "cultural_background",
    "career_goals_and_ambitions",
    "skills_and_expertise",
    "hobbies_and_interests",
]


def missing_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        column: int(frame[column].fillna("").astype(str).str.strip().eq("").sum())
        for column in STRUCTURED_TEXT_COLUMNS
        if column in frame.columns
    }


def duplicate_domain_hashes(frame: pd.DataFrame, domain: str) -> dict[str, Any]:
    hashes: list[str] = []
    examples: dict[str, list[str]] = defaultdict(list)
    for row in frame.to_dict(orient="records"):
        text = build_domain_text(row, domain)
        if not text:
            continue
        digest = text_hash(text)
        hashes.append(digest)
        if len(examples[digest]) < 5:
            examples[digest].append(str(row["uuid"]))
    counts = Counter(hashes)
    duplicate_hashes = {digest: count for digest, count in counts.items() if count > 1}
    top_duplicates = sorted(duplicate_hashes.items(), key=lambda item: item[1], reverse=True)[:20]
    return {
        "non_empty_count": len(hashes),
        "unique_hash_count": len(counts),
        "duplicate_hash_count": len(duplicate_hashes),
        "top_duplicate_examples": [
            {"hash": digest, "count": count, "uuid_sample": examples[digest]} for digest, count in top_duplicates
        ],
    }


def simple_structured_overlap_flags(frame: pd.DataFrame) -> dict[str, Any]:
    checks = {
        "hobbies_text_mentions_hobby_like_terms": ("hobbies_and_interests", ["취미", "운동", "독서", "여행", "요리", "음악"]),
        "skills_text_mentions_skill_like_terms": ("skills_and_expertise", ["기술", "전문", "경험", "역량", "자격", "능력"]),
        "career_text_mentions_career_like_terms": ("career_goals_and_ambitions", ["커리어", "목표", "성장", "직무", "일", "업무"]),
    }
    results: dict[str, Any] = {}
    for name, (column, terms) in checks.items():
        if column not in frame.columns:
            continue
        series = frame[column].fillna("").astype(str)
        mask = series.apply(lambda text: any(term in text for term in terms))
        results[name] = {
            "count": int(mask.sum()),
            "ratio": float(mask.mean()) if len(mask) else 0.0,
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/persona_similarity/configs/lightgbm_reranker.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    cache_metadata = {
        "stage": "audit_text_feature_leakage",
        "input_path": config["paths"]["persona_texts"],
        "input_hash": file_sha256(config["paths"]["persona_texts"]),
        "config_hash": stable_json_hash({"text_domains": list(TEXT_DOMAINS.keys())}),
    }
    use_cache, cache_reason = should_use_cache(config["paths"]["text_leakage_audit"], config["paths"]["text_leakage_audit"], cache_metadata, args.force)
    if use_cache:
        mark_cache_hit(config["paths"]["text_leakage_audit"], cache_metadata, config["paths"]["text_leakage_audit"])
        return

    start_time = time.perf_counter()
    frame = pd.read_parquet(PROJECT_ROOT / config["paths"]["persona_texts"])
    domain_reports = {domain: duplicate_domain_hashes(frame, domain) for domain in TEXT_DOMAINS}
    write_json(
        config["paths"]["text_leakage_audit"],
        {
            **cache_metadata,
            "cache_hit": False,
            "cache_reason": cache_reason,
            "rows": int(len(frame)),
            "missing_text_counts": missing_counts(frame),
            "domain_duplicate_hashes": domain_reports,
            "structured_overlap_flags": simple_structured_overlap_flags(frame),
            "runtime_seconds": time.perf_counter() - start_time,
            "decision_note": "This audit is a warning signal only. Manual review is required before promoting text features.",
        },
    )


if __name__ == "__main__":
    main()
