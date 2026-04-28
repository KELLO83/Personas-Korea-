from typing import Any

from fastapi import APIRouter

from src.api.exceptions import NotFoundException, ServiceUnavailableException
from src.api.schemas import QueryPersona, SimilarPersona, SimilarRequest, SimilarResponse
from src.gds.similarity import SimilarityService

router = APIRouter(prefix="/api", tags=["similarity"])


def get_similarity_service() -> SimilarityService:
    return SimilarityService()


@router.post("/similar/{uuid}", response_model=SimilarResponse)
def similar(uuid: str, request: SimilarRequest) -> SimilarResponse:
    service = get_similarity_service()
    try:
        query_persona_row = service.get_query_persona(uuid)
        if not query_persona_row:
            raise NotFoundException("Persona not found")

        query_persona = _build_query_persona(query_persona_row, uuid)
        graph_weight = request.weights.get("graph", 0.6)
        text_weight = request.weights.get("text", 0.4)

        graph_results = {r["uuid"]: r for r in service.find_similar_personas(uuid, top_k=request.top_k * 2)}
        text_results = {r["uuid"]: r for r in service.find_text_similar_personas(uuid, top_k=request.top_k * 2)}

        if not graph_results and not text_results:
            raise ServiceUnavailableException("Similarity data is not available")

        all_uuids = set(graph_results.keys()) | set(text_results.keys())
        merged: list[dict[str, Any]] = []
        for target_uuid in all_uuids:
            g_score = graph_results.get(target_uuid, {}).get("similarity", 0.0) or 0.0
            t_score = text_results.get(target_uuid, {}).get("similarity", 0.0) or 0.0
            combined = graph_weight * float(g_score) + text_weight * float(t_score)
            base = graph_results.get(target_uuid) or text_results.get(target_uuid) or {}
            merged.append({**base, "uuid": target_uuid, "similarity": round(combined, 4)})

        merged.sort(key=lambda x: x["similarity"], reverse=True)
        top_results = merged[: request.top_k]

        personas = []
        for result in top_results:
            target_uuid = result["uuid"]
            shared_traits = service.find_shared_traits(uuid, target_uuid)
            shared_hobbies = service.find_shared_hobbies(uuid, target_uuid)
            summary = _build_summary(result.get("display_name"), shared_hobbies, shared_traits)
            personas.append(
                SimilarPersona(
                    **result,
                    shared_traits=shared_traits,
                    shared_hobbies=shared_hobbies,
                    summary=summary,
                )
            )
        return SimilarResponse(query_uuid=uuid, query_persona=query_persona, similar_personas=personas)
    finally:
        service.close()


def _build_summary(display_name: str | None, hobbies: list[str], traits: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    if hobbies:
        parts.append(f"공통 취미: {', '.join(hobbies[:3])}")
    if traits:
        trait_names = [str(t.get("name")) for t in traits[:3] if t.get("name")]
        if trait_names:
            parts.append(f"공통 속성: {', '.join(trait_names)}")
    if display_name and parts:
        return f"{display_name}와(과) " + "; ".join(parts)
    return "; ".join(parts) if parts else ""


def _build_query_persona(row: dict[str, Any] | None, fallback_uuid: str) -> QueryPersona:
    if not row:
        return QueryPersona(uuid=fallback_uuid, name_summary=fallback_uuid)

    display_name = str(row.get("display_name") or fallback_uuid)
    age = row.get("age")
    district_name = row.get("district_name")
    occupation_name = row.get("occupation_name")

    parts = [display_name]
    if age is not None:
        parts.append(f"{age}세")
    if district_name:
        parts.append(str(district_name))
    if occupation_name:
        parts.append(str(occupation_name))

    return QueryPersona(uuid=str(row.get("uuid") or fallback_uuid), name_summary=", ".join(parts))
