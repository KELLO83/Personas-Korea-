export type ApiRecord = Record<string, unknown>;

export interface DistributionItem {
  label: string;
  count: number;
  ratio: number;
}

export interface RankedItem {
  label: string;
  count: number;
}

export interface StatsResponse {
  total_personas: number;
  age_distribution: DistributionItem[];
  sex_distribution: DistributionItem[];
  province_distribution: DistributionItem[];
  top_occupations: RankedItem[];
  top_hobbies: RankedItem[];
  top_skills: RankedItem[];
  education_distribution: DistributionItem[];
  marital_distribution: DistributionItem[];
}

export interface SearchResult {
  uuid: string;
  display_name: string | null;
  age: number | null;
  sex: string | null;
  province: string | null;
  district: string | null;
  occupation: string | null;
  education_level: string | null;
  persona: string | null;
}

export interface SearchResponse {
  total_count: number;
  page: number;
  page_size: number;
  total_pages: number;
  results: SearchResult[];
}

export interface Demographics {
  age: number | null;
  age_group: string | null;
  sex: string | null;
  marital_status: string | null;
  military_status: string | null;
  family_type: string | null;
  housing_type: string | null;
  education_level: string | null;
  bachelors_field: string | null;
}

export interface Location {
  country: string | null;
  province: string | null;
  district: string | null;
}

export interface Personas {
  summary: string | null;
  professional: string | null;
  sports: string | null;
  arts: string | null;
  travel: string | null;
  culinary: string | null;
  family: string | null;
}

export interface SimilarPreview {
  uuid: string;
  display_name: string | null;
  age: number | null;
  similarity: number | null;
  shared_hobbies: string[];
}

export interface PersonaProfileResponse {
  uuid: string;
  display_name: string | null;
  demographics: Demographics;
  location: Location;
  occupation: string | null;
  personas: Personas;
  cultural_background: string | null;
  career_goals: string | null;
  skills: string[];
  hobbies: string[];
  community: {
    community_id: number | null;
    label: string | null;
  };
  similar_preview: SimilarPreview[];
  graph_stats: {
    total_connections: number;
    hobby_count: number;
    skill_count: number;
  };
}

export interface SimilarityReason {
  feature: string;
  label: string;
  value: string;
  contribution: number;
  raw_score: number;
}

export interface SimilarityExplanationResponse {
  source_uuid: string;
  target_uuid: string;
  similarity_score: number | null;
  method: string;
  top_reasons: SimilarityReason[];
  shared_hobbies: string[];
  shared_skills: string[];
  note: string;
}

export interface GraphNode {
  id: string;
  label: string;
  type: string;
  properties: ApiRecord;
}

export interface GraphEdge {
  source: string;
  target: string;
  type: string;
}

export interface SubgraphResponse {
  center_uuid: string;
  center_label: string | null;
  node_count: number;
  edge_count: number;
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface ChatResponse {
  response: string;
  context_filters: Record<string, string>;
  sources: Record<string, unknown>[];
  turn_count: number;
}

export interface RecommendationModelInfo {
  target: string;
  status: string;
  score_source: string;
  model_version: string | null;
  fallback_used: boolean;
  fallback_reason: string;
  message: string;
}

export interface RecommendationStatusResponse {
  hobby_recommender: RecommendationModelInfo;
  persona_similarity_recommender: RecommendationModelInfo;
  product_policy: string;
}

export interface RagTraceSpan {
  name: string;
  status: string;
  latency_ms: number;
  metadata: Record<string, unknown>;
  error_type: string | null;
  error_message: string | null;
}

export interface RagTraceRecord {
  trace_id: string;
  route: string;
  session_id: string | null;
  question: string | null;
  status: string;
  created_at: string;
  latency_ms: number;
  spans: RagTraceSpan[];
  response_preview: string | null;
  error_type: string | null;
  error_message: string | null;
}

export interface RagTraceListResponse {
  tracing_enabled: boolean;
  traces: RagTraceRecord[];
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  filters?: Record<string, string>;
  sources?: Record<string, unknown>[];
}

export interface TargetPersonaSample {
  uuid: string;
  display_name: string | null;
  age: number | null;
  sex: string | null;
  province: string | null;
  district: string | null;
  occupation: string | null;
  persona: string | null;
}

export interface TargetPersonaResponse {
  filters: Record<string, string>;
  matched_count: number;
  sample_size: number;
  representative_persona: string;
  representative_hobbies: string[];
  representative_skills: string[];
  sample_personas: TargetPersonaSample[];
  evidence_uuids: string[];
  generation_method: string;
  synthesis_prompt: string;
  guardrails: string[];
  input_policy: string;
}

export interface LifestyleMapEdge {
  source_field: string;
  target_field: string;
  source_keyword: string;
  target_keyword: string;
  overlap_count: number;
  target_support_count: number;
  conditional_ratio: number;
}

export interface LifestyleMapResponse {
  filters: Record<string, string>;
  source_field: string;
  target_field: string;
  source_keyword: string;
  matched_source_count: number;
  available_fields: string[];
  keyword_policy: string;
  segment_policy: string;
  visualization_policy: string;
  edges: LifestyleMapEdge[];
}

export interface CareerTransitionItem {
  name: string;
  count: number;
  ratio: number;
}

export interface CareerTransitionResponse {
  filters: Record<string, string>;
  matched_count: number;
  top_goals: CareerTransitionItem[];
  top_skills: CareerTransitionItem[];
  top_neighbor_occupations: CareerTransitionItem[];
  segment_distribution: CareerTransitionItem[];
  mapping_policy: string;
  top_k_limit: number;
  analysis_scope: string;
}

export interface GraphMigrationStep {
  name: string;
  cypher: string;
  validation: string;
}

export interface GraphQualityDistributionItem {
  label: string;
  count: number;
  ratio: number;
}

export interface GraphQualityCheck {
  name: string;
  cardinality: number;
  total_count: number;
  issue: string;
  recommendation: string;
  action: string;
  severity: string;
  dominant_ratio: number;
  distribution: GraphQualityDistributionItem[];
}

export interface GraphQualityResponse {
  checks: GraphQualityCheck[];
  migration_plan: GraphMigrationStep[];
}

export interface ApiErrorBody {
  error?: string;
  detail?: string | { msg?: string }[];
}
