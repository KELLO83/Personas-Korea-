import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, Query
from neo4j import GraphDatabase

from src.api.schemas import (
    BridgePersonaCandidate,
    CommunityLabelCandidate,
    GraphDataQualityIssue,
    GraphInsightsResponse,
    HobbyNormalizationCandidate,
    HobbyOccupationRegionPath,
    HobbyVariant,
    SkillExtractionCandidate,
)
from src.config import settings

router = APIRouter(prefix="/api/graph-insights", tags=["graph-insights"])

HOBBY_KEYWORDS = [
    "배드민턴",
    "헬스",
    "노래방",
    "UFC",
    "사우나",
    "등산",
    "요가",
    "축구",
    "독서",
    "영화",
    "여행",
    "캠핑",
    "요리",
    "카페",
    "게임",
    "러닝",
    "수영",
    "자전거",
    "골프",
    "테니스",
    "클라이밍",
    "낚시",
    "사진",
]
SKILL_SPLIT_PATTERN = re.compile(r"[,;/·\n]| 및 | 그리고 | 또는 |과 |와 ")
MIN_SKILL_LENGTH = 2
SKILL_SAMPLE_LIMIT = 5000
HOBBY_VARIANT_LIMIT = 6
SKILL_EXAMPLE_LIMIT = 3
SKILL_EXAMPLE_CHAR_LIMIT = 120
COMMUNITY_LABEL_PART_LIMIT = 3
HOBBY_KEYWORD_FALLBACK_LENGTH = 18
QUALITY_RATIO_PRECISION = 4
LOW_VALUE_SKILL_TERMS = {"기타", "없음", "무", "등", "관련", "능력", "경험"}


@dataclass(frozen=True)
class QualityIssueInput:
    name: str
    severity: str
    value: float
    total: float
    impact: str
    recommendation: str
    examples: list[str] | None = None

SUMMARY_QUERY = """
MATCH (p:Person)
WITH count(p) AS personas
MATCH (h:Hobby)
WITH personas, count(h) AS hobbies
MATCH (h:Hobby)
WITH personas, hobbies, count { (h)<-[:ENJOYS_HOBBY]-(:Person) } AS degree
WITH personas, hobbies, count(CASE WHEN degree = 1 THEN 1 END) AS singleton_hobbies
MATCH ()-[r:SIMILAR_TO]->()
WITH personas, hobbies, singleton_hobbies, count(r) AS similar_edges
MATCH (p:Person)
RETURN personas,
       hobbies,
       singleton_hobbies,
       similar_edges,
       count(DISTINCT p.community_id) AS communities
"""

HOBBY_CANDIDATES_QUERY = """
UNWIND $keywords AS keyword
MATCH (h:Hobby)<-[:ENJOYS_HOBBY]-(:Person)
WHERE h.name CONTAINS keyword
WITH keyword, h.name AS name, count(*) AS count
ORDER BY keyword, count DESC
WITH keyword, collect({name: name, count: count}) AS variants, sum(count) AS support_count
RETURN keyword,
       support_count,
       size(variants) AS variant_count,
       variants[..$variant_limit] AS variants
ORDER BY support_count DESC
LIMIT $limit
"""

SKILL_TEXT_QUERY = """
MATCH (p:Person)
WHERE p.skills_and_expertise IS NOT NULL AND trim(p.skills_and_expertise) <> ''
RETURN p.skills_and_expertise AS text
LIMIT $limit
"""

COMMUNITY_LABEL_QUERY = """
MATCH (p:Person)
WHERE p.community_id IS NOT NULL
WITH p.community_id AS community_id, count(*) AS size
ORDER BY size DESC
LIMIT $limit
CALL (community_id) {
  MATCH (p:Person)-[:LIVES_IN]->(:District)-[:IN_PROVINCE]->(prov:Province)
  WHERE p.community_id = community_id
  RETURN prov.name AS top_province, count(*) AS province_count
  ORDER BY province_count DESC
  LIMIT 1
}
CALL (community_id) {
  MATCH (p:Person)-[:WORKS_AS]->(occ:Occupation)
  WHERE p.community_id = community_id
  RETURN occ.name AS top_occupation, count(*) AS occupation_count
  ORDER BY occupation_count DESC
  LIMIT 1
}
CALL (community_id) {
  MATCH (p:Person)-[:ENJOYS_HOBBY]->(h:Hobby)
  WHERE p.community_id = community_id
  RETURN h.name AS top_hobby, count(*) AS hobby_count
  ORDER BY hobby_count DESC
  LIMIT 1
}
RETURN community_id, size, top_province, top_occupation, top_hobby
ORDER BY size DESC
"""

QUALITY_QUERY = """
MATCH (h:Hobby)
WITH count(h) AS total_hobbies
MATCH (h:Hobby)
WITH total_hobbies, h, count { (h)<-[:ENJOYS_HOBBY]-(:Person) } AS degree
WITH total_hobbies,
     count(CASE WHEN degree = 1 THEN 1 END) AS singleton_hobbies,
     collect(CASE
       WHEN size(h.name) < 8
         OR h.name ENDS WITH '한 번'
         OR h.name ENDS WITH '남편'
         OR h.name ENDS WITH '아내'
         OR h.name ENDS WITH '그리고'
       THEN h.name
     END)[..10] AS broken_examples
MATCH (p:Person)-[:ENJOYS_HOBBY]->(h:Hobby)<-[:LIKES]-(p)
WITH total_hobbies, singleton_hobbies, broken_examples, count(*) AS duplicate_hobby_links
MATCH (p:Person)
WITH total_hobbies, singleton_hobbies, broken_examples, duplicate_hobby_links, count(p) AS persons
MATCH (p:Person)
WHERE p.community_id IS NOT NULL
WITH total_hobbies,
     singleton_hobbies,
     broken_examples,
     duplicate_hobby_links,
     persons,
     count(p) AS community_persons
OPTIONAL MATCH (s)
WHERE "Skill" IN labels(s)
WITH total_hobbies,
     singleton_hobbies,
     broken_examples,
     duplicate_hobby_links,
     persons,
     community_persons,
     count(s) AS skill_nodes
OPTIONAL MATCH ()-[r]->()
WHERE type(r) = "HAS_SKILL"
RETURN total_hobbies,
       singleton_hobbies,
       broken_examples,
       duplicate_hobby_links,
       persons,
       community_persons,
       skill_nodes,
       count(r) AS skill_relationships
"""

BRIDGE_PERSONA_QUERY = """
MATCH (p:Person)-[r:SIMILAR_TO]->(neighbor:Person)
WHERE p.community_id IS NOT NULL
  AND neighbor.community_id IS NOT NULL
  AND p.community_id <> neighbor.community_id
WITH p,
     count(DISTINCT neighbor.community_id) AS neighbor_community_count,
     avg(r.score) AS average_similarity,
     collect(DISTINCT neighbor.community_id)[..6] AS neighbor_communities
RETURN p.uuid AS uuid,
       p.display_name AS display_name,
       p.community_id AS community_id,
       neighbor_community_count,
       average_similarity,
       neighbor_communities
ORDER BY neighbor_community_count DESC, average_similarity DESC
LIMIT $limit
"""

HOBBY_OCCUPATION_REGION_QUERY = """
UNWIND $keywords AS keyword
MATCH (p:Person)-[:ENJOYS_HOBBY]->(h:Hobby),
      (p)-[:WORKS_AS]->(occ:Occupation),
      (p)-[:LIVES_IN]->(:District)-[:IN_PROVINCE]->(prov:Province)
WHERE h.name CONTAINS keyword
WITH keyword,
     occ.name AS occupation,
     prov.name AS province,
     count(p) AS support_count,
     collect({uuid: p.uuid, display_name: p.display_name}) AS personas
RETURN keyword AS hobby_keyword,
       occupation,
       province,
       support_count,
       personas[0].uuid AS representative_persona_uuid,
       personas[0].display_name AS representative_persona_name
ORDER BY support_count DESC
LIMIT $limit
"""


@router.get("/dashboard", response_model=GraphInsightsResponse)
def graph_insights_dashboard(limit: int = Query(default=12, ge=3, le=30)) -> GraphInsightsResponse:
    driver = GraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))
    try:
        with driver.session(database=settings.NEO4J_DATABASE) as session:
            summary = _fetch_summary(session)
            hobby_candidates = _fetch_hobby_candidates(session, limit)
            skill_candidates = _fetch_skill_candidates(session, limit)
            community_labels = _fetch_community_labels(session, limit)
            quality_issues = _fetch_quality_issues(session)
            bridge_personas = _fetch_bridge_personas(session, limit)
            graph_paths = _fetch_hobby_occupation_region_paths(session, limit)
    finally:
        driver.close()

    return GraphInsightsResponse(
        summary=summary,
        dashboard_policy="읽기 전용 그래프 인사이트입니다. 정규화/스킬 노드 생성은 후보 검수 후 별도 배치로 적용합니다.",
        hobby_normalization_candidates=hobby_candidates,
        skill_extraction_candidates=skill_candidates,
        community_label_candidates=community_labels,
        data_quality_issues=quality_issues,
        bridge_personas=bridge_personas,
        hobby_occupation_region_paths=graph_paths,
    )


def _fetch_summary(session: Any) -> dict[str, int | float]:
    record = session.run(SUMMARY_QUERY).single()
    if not record:
        return {}
    data = dict(record)
    hobbies = max(int(data.get("hobbies") or 0), 1)
    data["singleton_hobby_ratio"] = round(
        int(data.get("singleton_hobbies") or 0) / hobbies,
        QUALITY_RATIO_PRECISION,
    )
    return data


def _fetch_hobby_candidates(session: Any, limit: int) -> list[HobbyNormalizationCandidate]:
    records = session.run(
        HOBBY_CANDIDATES_QUERY,
        keywords=HOBBY_KEYWORDS,
        variant_limit=HOBBY_VARIANT_LIMIT,
        limit=limit,
    )
    return [
        HobbyNormalizationCandidate(
            keyword=record["keyword"],
            canonical_label=record["keyword"],
            support_count=int(record["support_count"]),
            variant_count=int(record["variant_count"]),
            variants=[HobbyVariant(**variant) for variant in record["variants"]],
        )
        for record in records
    ]


def _fetch_skill_candidates(session: Any, limit: int) -> list[SkillExtractionCandidate]:
    texts = [record["text"] for record in session.run(SKILL_TEXT_QUERY, limit=SKILL_SAMPLE_LIMIT)]
    counts, examples = _extract_skill_terms(texts)
    candidates = []
    for name, count in counts.most_common(limit):
        candidates.append(
            SkillExtractionCandidate(
                name=name,
                count=count,
                examples=examples[name][:SKILL_EXAMPLE_LIMIT],
            )
        )
    return candidates


def _extract_skill_terms(texts: list[str]) -> tuple[Counter[str], dict[str, list[str]]]:
    counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    for text in texts:
        for term in _split_skill_text(text):
            counts[term] += 1
            if len(examples[term]) < SKILL_EXAMPLE_LIMIT:
                examples[term].append(text[:SKILL_EXAMPLE_CHAR_LIMIT])
    return counts, examples


def _split_skill_text(text: str) -> list[str]:
    terms = []
    for raw_term in SKILL_SPLIT_PATTERN.split(text):
        term = raw_term.strip(" .·,;:/()[]{}")
        if len(term) >= MIN_SKILL_LENGTH and not _is_low_value_skill_term(term):
            terms.append(term)
    return terms


def _is_low_value_skill_term(term: str) -> bool:
    return term in LOW_VALUE_SKILL_TERMS or term.startswith("및 ")


def _fetch_community_labels(session: Any, limit: int) -> list[CommunityLabelCandidate]:
    labels = []
    for record in session.run(COMMUNITY_LABEL_QUERY, limit=limit):
        top_hobby_keyword = _best_hobby_keyword(record["top_hobby"])
        label = _community_label(record["top_province"], record["top_occupation"], top_hobby_keyword)
        labels.append(
            CommunityLabelCandidate(
                community_id=int(record["community_id"]),
                label=label,
                size=int(record["size"]),
                top_province=record["top_province"],
                top_occupation=record["top_occupation"],
                top_hobby_keyword=top_hobby_keyword,
                summary=f"{label} 커뮤니티는 {record['size']}명 규모입니다.",
            )
        )
    return labels


def _best_hobby_keyword(text: str | None) -> str | None:
    if not text:
        return None
    for keyword in HOBBY_KEYWORDS:
        if keyword in text:
            return keyword
    return text[:HOBBY_KEYWORD_FALLBACK_LENGTH]


def _community_label(province: str | None, occupation: str | None, hobby: str | None) -> str:
    parts = [part for part in (province, occupation, hobby) if part]
    if not parts:
        return "미분류 커뮤니티"
    return " / ".join(parts[:COMMUNITY_LABEL_PART_LIMIT])


def _fetch_quality_issues(session: Any) -> list[GraphDataQualityIssue]:
    record = session.run(QUALITY_QUERY).single()
    if not record:
        return []
    data = dict(record)
    total_hobbies = float(data["total_hobbies"] or 0)
    persons = float(data["persons"] or 0)
    return [
        _quality_issue(
            QualityIssueInput(
                name="1인 취미 노드",
                severity="warning",
                value=float(data["singleton_hobbies"] or 0),
                total=total_hobbies,
                impact="취미 추천과 검색 필터가 원문 단위로 과도하게 분산됩니다.",
                recommendation="상위 키워드부터 CanonicalHobby 후보를 검수하세요.",
            )
        ),
        _quality_issue(
            QualityIssueInput(
                name="잘린 취미 구문 후보",
                severity="warning",
                value=float(len([item for item in data["broken_examples"] if item])),
                total=10,
                impact="사용자 화면에 어색한 취미명이 노출될 수 있습니다.",
                recommendation="짧거나 접속사로 끝나는 취미 원문을 정제 후보로 분리하세요.",
                examples=[item for item in data["broken_examples"] if item],
            )
        ),
        _quality_issue(
            QualityIssueInput(
                name="중복 취미 관계",
                severity="info",
                value=float(data["duplicate_hobby_links"] or 0),
                total=float(data["duplicate_hobby_links"] or 1),
                impact="LIKES와 ENJOYS_HOBBY가 같은 의미로 중복 저장되어 지표 해석이 헷갈릴 수 있습니다.",
                recommendation="운영 쿼리는 ENJOYS_HOBBY를 기준 관계로 통일하세요.",
            )
        ),
        _quality_issue(
            QualityIssueInput(
                name="스킬 그래프 미구축",
                severity="warning",
                value=float(data["skill_relationships"] or 0),
                total=max(persons, 1),
                impact="스킬 기반 추천/커뮤니티 설명이 제한됩니다.",
                recommendation="skills_and_expertise에서 Skill 후보를 검수한 뒤 HAS_SKILL 배치를 실행하세요.",
            )
        ),
        _quality_issue(
            QualityIssueInput(
                name="커뮤니티 커버리지",
                severity="ok",
                value=float(data["community_persons"] or 0),
                total=max(persons, 1),
                impact="커뮤니티 기반 추천과 라벨링에 사용할 수 있는 인원 비율입니다.",
                recommendation="커버리지가 낮아질 경우 Leiden 배치 결과를 확인하세요.",
            )
        ),
    ]


def _quality_issue(issue: QualityIssueInput) -> GraphDataQualityIssue:
    ratio = round(issue.value / max(issue.total, 1), QUALITY_RATIO_PRECISION)
    return GraphDataQualityIssue(
        name=issue.name,
        severity=issue.severity,
        value=issue.value,
        total=issue.total,
        ratio=ratio,
        impact=issue.impact,
        recommendation=issue.recommendation,
        examples=issue.examples or [],
    )


def _fetch_bridge_personas(session: Any, limit: int) -> list[BridgePersonaCandidate]:
    return [
        BridgePersonaCandidate(
            uuid=record["uuid"],
            display_name=record["display_name"],
            community_id=record["community_id"],
            neighbor_community_count=int(record["neighbor_community_count"]),
            average_similarity=round(float(record["average_similarity"] or 0), QUALITY_RATIO_PRECISION),
            neighbor_communities=[int(community) for community in record["neighbor_communities"]],
        )
        for record in session.run(BRIDGE_PERSONA_QUERY, limit=limit)
    ]


def _fetch_hobby_occupation_region_paths(session: Any, limit: int) -> list[HobbyOccupationRegionPath]:
    return [
        HobbyOccupationRegionPath(
            hobby_keyword=record["hobby_keyword"],
            occupation=record["occupation"],
            province=record["province"],
            support_count=int(record["support_count"]),
            representative_persona_uuid=record["representative_persona_uuid"],
            representative_persona_name=record["representative_persona_name"],
        )
        for record in session.run(HOBBY_OCCUPATION_REGION_QUERY, keywords=HOBBY_KEYWORDS, limit=limit)
    ]
