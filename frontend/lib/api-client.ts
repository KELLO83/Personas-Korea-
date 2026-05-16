import type {
  ApiErrorBody,
  ChatResponse,
  CareerTransitionResponse,
  GraphQualityResponse,
  LifestyleMapResponse,
  PersonaProfileResponse,
  RagTraceListResponse,
  RecommendationStatusResponse,
  SearchResponse,
  SimilarityExplanationResponse,
  StatsResponse,
  SubgraphResponse,
  TargetPersonaResponse,
} from "./api-types";

const DEFAULT_API_BASE_URL = "http://localhost:8000";
const DEFAULT_POST_TIMEOUT_MS = 60_000;
const CHAT_TIMEOUT_MS = 90_000;

interface ApiPostOptions {
  timeoutMs?: number;
}

export function apiBaseUrl(): string {
  return process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") || DEFAULT_API_BASE_URL;
}

function buildUrl(path: string, params?: Record<string, string | number | boolean | null | undefined>): string {
  const url = new URL(path, `${apiBaseUrl()}/`);
  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null && value !== "") {
        url.searchParams.set(key, String(value));
      }
    });
  }
  return url.toString();
}

function extractApiError(body: ApiErrorBody | null, fallback: string): string {
  if (!body) return fallback;
  if (body.error) return body.error;
  if (typeof body.detail === "string") return body.detail;
  if (Array.isArray(body.detail)) {
    const messages = body.detail.map((item) => item.msg).filter((message): message is string => Boolean(message));
    if (messages.length > 0) return messages.join(" / ");
  }
  return fallback;
}

async function parseError(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as ApiErrorBody;
    return extractApiError(body, `API 요청 실패 (${response.status})`);
  } catch {
    return `API 요청 실패 (${response.status})`;
  }
}

export async function apiGet<T>(path: string, params?: Record<string, string | number | boolean | null | undefined>): Promise<T> {
  const response = await fetch(buildUrl(path, params), { cache: "no-store" });
  if (!response.ok) throw new Error(await parseError(response));
  return (await response.json()) as T;
}

export async function apiPost<T>(path: string, payload: Record<string, unknown>, options: ApiPostOptions = {}): Promise<T> {
  const controller = new AbortController();
  const timeoutMs = options.timeoutMs ?? DEFAULT_POST_TIMEOUT_MS;
  const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(buildUrl(path), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      cache: "no-store",
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(await parseError(response));
    return (await response.json()) as T;
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new Error("응답 시간이 길어져 요청을 중단했습니다. 잠시 후 다시 시도해 주세요.");
    }
    throw error;
  } finally {
    window.clearTimeout(timeoutId);
  }
}

export const personaApi = {
  stats: () => apiGet<StatsResponse>("/api/stats"),
  search: (params: Record<string, string | number | boolean | null | undefined>) => apiGet<SearchResponse>("/api/search", params),
  profile: (uuid: string) => apiGet<PersonaProfileResponse>(`/api/persona/${encodeURIComponent(uuid)}`),
  similarityExplanation: (sourceUuid: string, targetUuid: string) =>
    apiGet<SimilarityExplanationResponse>(`/api/persona/${encodeURIComponent(sourceUuid)}/similar/${encodeURIComponent(targetUuid)}/explanation`),
  graph: (uuid: string, params: { depth: number; max_nodes: number; include_similar: boolean }) =>
    apiGet<SubgraphResponse>(`/api/graph/subgraph/${encodeURIComponent(uuid)}`, params),
  chat: (payload: { session_id: string; message: string; stream: boolean }) => apiPost<ChatResponse>("/api/chat", payload, { timeoutMs: CHAT_TIMEOUT_MS }),
  targetPersona: (params: Record<string, string | number | boolean | null | undefined>) => apiGet<TargetPersonaResponse>("/api/target-persona", params),
  lifestyleMap: (params: Record<string, string | number | boolean | null | undefined>) => apiGet<LifestyleMapResponse>("/api/lifestyle-map", params),
  careerTransition: (params: Record<string, string | number | boolean | null | undefined>) => apiGet<CareerTransitionResponse>("/api/career-transition-map", params),
  graphQuality: () => apiGet<GraphQualityResponse>("/api/graph-quality"),
  recommendationStatus: () => apiGet<RecommendationStatusResponse>("/api/recommendation/status"),
  ragTraces: (params: { limit?: number } = {}) => apiGet<RagTraceListResponse>("/api/admin/rag/traces", params),
};
