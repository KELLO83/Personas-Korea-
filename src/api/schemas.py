from typing import Any

from pydantic import BaseModel, Field, field_validator


class InsightRequest(BaseModel):
    question: str = Field(min_length=1)


class InsightResponse(BaseModel):
    answer: str
    sources: list[Any] = Field(default_factory=list)
    query_type: str


class SimilarRequest(BaseModel):
    top_k: int = Field(default=5, ge=1, le=50)
    weights: dict[str, float] = Field(default_factory=lambda: {"graph": 0.6, "text": 0.4})

    @field_validator("weights")
    @classmethod
    def _validate_weights(cls, value: dict[str, float]) -> dict[str, float]:
        allowed_keys = {"graph", "text"}
        extra_keys = set(value.keys()) - allowed_keys
        if extra_keys:
            raise ValueError(f"weights keys must be graph or text, got: {extra_keys}")
        for key, weight in value.items():
            if not isinstance(weight, (int, float)):
                raise ValueError(f"weights['{key}'] must be a number")
            if weight < 0:
                raise ValueError(f"weights['{key}'] must be non-negative")
        total = sum(value.values())
        if total == 0:
            raise ValueError("weights sum must be greater than 0")
        return value


class SimilarPersona(BaseModel):
    uuid: str
    display_name: str | None = None
    persona: str | None = None
    age: int | None = None
    sex: str | None = None
    similarity: float | None = None
    shared_traits: list[dict[str, Any]] = Field(default_factory=list)
    shared_hobbies: list[str] = Field(default_factory=list)
    summary: str = ""


class QueryPersona(BaseModel):
    uuid: str
    name_summary: str


class SimilarResponse(BaseModel):
    query_uuid: str
    query_persona: QueryPersona | None = None
    similar_personas: list[SimilarPersona]


class CommunityResponse(BaseModel):
    communities: list[dict[str, Any]]


class PathResponse(BaseModel):
    path_found: bool
    length: int
    path: list[dict[str, Any]] = Field(default_factory=list)
    shared_nodes: list[dict[str, Any]] = Field(default_factory=list)
    summary: str = ""


class Demographics(BaseModel):
    age: int | None = None
    age_group: str | None = None
    sex: str | None = None
    marital_status: str | None = None
    military_status: str | None = None
    family_type: str | None = None
    housing_type: str | None = None
    education_level: str | None = None
    bachelors_field: str | None = None


class Location(BaseModel):
    country: str | None = None
    province: str | None = None
    district: str | None = None


class Personas(BaseModel):
    summary: str | None = None
    professional: str | None = None
    sports: str | None = None
    arts: str | None = None
    travel: str | None = None
    culinary: str | None = None
    family: str | None = None


class SimilarPreview(BaseModel):
    uuid: str
    display_name: str | None = None
    age: int | None = None
    similarity: float | None = None
    shared_hobbies: list[str] = Field(default_factory=list)


class CommunityInfo(BaseModel):
    community_id: int | None = None
    label: str | None = None


class GraphStats(BaseModel):
    total_connections: int = 0
    hobby_count: int = 0
    skill_count: int = 0


class PersonaProfileResponse(BaseModel):
    uuid: str
    display_name: str | None = None
    demographics: Demographics = Field(default_factory=Demographics)
    location: Location = Field(default_factory=Location)
    occupation: str | None = None
    personas: Personas = Field(default_factory=Personas)
    cultural_background: str | None = None
    career_goals: str | None = None
    skills: list[str] = Field(default_factory=list)
    hobbies: list[str] = Field(default_factory=list)
    community: CommunityInfo = Field(default_factory=CommunityInfo)
    similar_preview: list[SimilarPreview] = Field(default_factory=list)
    graph_stats: GraphStats = Field(default_factory=GraphStats)


class DistributionItem(BaseModel):
    label: str
    count: int
    ratio: float = 0.0


class RankedItem(BaseModel):
    label: str
    count: int


class StatsResponse(BaseModel):
    total_personas: int
    age_distribution: list[DistributionItem] = Field(default_factory=list)
    sex_distribution: list[DistributionItem] = Field(default_factory=list)
    province_distribution: list[DistributionItem] = Field(default_factory=list)
    top_occupations: list[RankedItem] = Field(default_factory=list)
    top_hobbies: list[RankedItem] = Field(default_factory=list)
    top_skills: list[RankedItem] = Field(default_factory=list)
    education_distribution: list[DistributionItem] = Field(default_factory=list)
    marital_distribution: list[DistributionItem] = Field(default_factory=list)


class DimensionStatsResponse(BaseModel):
    dimension: str
    filters_applied: dict[str, str] = Field(default_factory=dict)
    filtered_count: int = 0
    distribution: list[DistributionItem] = Field(default_factory=list)


class SearchResult(BaseModel):
    uuid: str
    display_name: str | None = None
    age: int | None = None
    sex: str | None = None
    province: str | None = None
    district: str | None = None
    occupation: str | None = None
    education_level: str | None = None
    persona: str | None = None


class SearchResponse(BaseModel):
    total_count: int
    page: int
    page_size: int
    total_pages: int
    results: list[SearchResult] = Field(default_factory=list)


class GraphNode(BaseModel):
    id: str
    label: str
    type: str
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    source: str
    target: str
    type: str


class SubgraphResponse(BaseModel):
    center_uuid: str
    center_label: str | None = None
    node_count: int = 0
    edge_count: int = 0
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)


class SegmentFilter(BaseModel):
    province: str | None = None
    district: str | None = None
    age_group: str | None = None
    sex: str | None = None
    education_level: str | None = None
    hobby: str | None = None
    skill: str | None = None


class SegmentDefinition(BaseModel):
    label: str
    filters: SegmentFilter = Field(default_factory=SegmentFilter)


class CompareDistributionItem(BaseModel):
    name: str
    count: int
    ratio: float


class DimensionComparison(BaseModel):
    segment_a: list[CompareDistributionItem] = Field(default_factory=list)
    segment_b: list[CompareDistributionItem] = Field(default_factory=list)
    common: list[str] = Field(default_factory=list)
    only_a: list[str] = Field(default_factory=list)
    only_b: list[str] = Field(default_factory=list)


class SegmentSummary(BaseModel):
    label: str
    count: int


class SegmentCompareRequest(BaseModel):
    segment_a: SegmentDefinition
    segment_b: SegmentDefinition
    dimensions: list[str] = Field(default_factory=lambda: ["hobby", "occupation", "education"])
    top_k: int = Field(default=10, ge=1, le=50)


class SegmentCompareResponse(BaseModel):
    segment_a: SegmentSummary
    segment_b: SegmentSummary
    comparisons: dict[str, DimensionComparison] = Field(default_factory=dict)
    ai_analysis: str = ""


class InfluenceTopItem(BaseModel):
    uuid: str
    display_name: str | None = None
    score: float
    rank: int
    community_id: int | None = None


class InfluenceResponse(BaseModel):
    metric: str
    last_updated_at: str | None = None
    run_id: str | None = None
    stale_warning: bool = False
    results: list[InfluenceTopItem] = Field(default_factory=list)


class RemovalSimulationRequest(BaseModel):
    target_uuids: list[str] = Field(min_length=1, max_length=5)
    max_depth: int = Field(default=3, ge=1, le=3)


class RemovalSimulationResponse(BaseModel):
    path_found: bool = False
    original_connectivity: float = 0.0
    current_connectivity: float = 0.0
    fragmentation_increase: float = 0.0
    affected_communities: list[int] = Field(default_factory=list)


class RecommendItem(BaseModel):
    item_name: str
    reason: str
    reason_score: float
    similar_users_count: int
    supporting_personas: list[SimilarPreview] = Field(default_factory=list)


class RecommendResponse(BaseModel):
    uuid: str
    category: str
    recommendations: list[RecommendItem] = Field(default_factory=list)


class TargetPersonaSample(BaseModel):
    uuid: str
    display_name: str | None = None
    age: int | None = None
    sex: str | None = None
    province: str | None = None
    district: str | None = None
    occupation: str | None = None
    persona: str | None = None


class TargetPersonaResponse(BaseModel):
    filters: dict[str, str] = Field(default_factory=dict)
    matched_count: int = 0
    sample_size: int = 0
    representative_persona: str = ""
    representative_hobbies: list[str] = Field(default_factory=list)
    representative_skills: list[str] = Field(default_factory=list)
    sample_personas: list[TargetPersonaSample] = Field(default_factory=list)
    evidence_uuids: list[str] = Field(default_factory=list)
    generation_method: str = "deterministic"
    synthesis_prompt: str = ""
    guardrails: list[str] = Field(default_factory=list)
    input_policy: str = ""


class LifestyleMapEdge(BaseModel):
    source_field: str
    target_field: str
    source_keyword: str
    target_keyword: str
    overlap_count: int
    target_support_count: int
    conditional_ratio: float


class LifestyleMapResponse(BaseModel):
    filters: dict[str, str] = Field(default_factory=dict)
    source_field: str
    target_field: str
    source_keyword: str
    matched_source_count: int = 0
    available_fields: list[str] = Field(default_factory=list)
    keyword_policy: str = ""
    segment_policy: str = ""
    visualization_policy: str = ""
    edges: list[LifestyleMapEdge] = Field(default_factory=list)


class CareerTransitionItem(BaseModel):
    name: str
    count: int
    ratio: float


class CareerTransitionResponse(BaseModel):
    filters: dict[str, str] = Field(default_factory=dict)
    matched_count: int = 0
    top_goals: list[CareerTransitionItem] = Field(default_factory=list)
    top_skills: list[CareerTransitionItem] = Field(default_factory=list)
    top_neighbor_occupations: list[CareerTransitionItem] = Field(default_factory=list)
    segment_distribution: list[CareerTransitionItem] = Field(default_factory=list)
    mapping_policy: str = ""
    top_k_limit: int = 30
    analysis_scope: str = ""


class GraphMigrationStep(BaseModel):
    name: str
    cypher: str
    validation: str


class GraphQualityDistributionItem(BaseModel):
    label: str
    count: int
    ratio: float


class GraphQualityCheck(BaseModel):
    name: str
    cardinality: int
    total_count: int
    issue: str
    recommendation: str
    action: str
    severity: str
    dominant_ratio: float = 0.0
    distribution: list[GraphQualityDistributionItem] = Field(default_factory=list)


class GraphQualityResponse(BaseModel):
    checks: list[GraphQualityCheck] = Field(default_factory=list)
    migration_plan: list[GraphMigrationStep] = Field(default_factory=list)


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    stream: bool = False


class ChatResponse(BaseModel):
    response: str
    context_filters: dict[str, str] = Field(default_factory=dict)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    turn_count: int = 0
