from __future__ import annotations

from collections import OrderedDict
import json
import logging
import re
from typing import Any, Literal, LiteralString, Protocol, TypedDict, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from neo4j import GraphDatabase, Query

from src.config import settings
from src.graph.search_queries import build_search_query
from src.graph.stats_queries import VALID_DIMENSIONS, build_dimension_query
from src.rag.llm import create_llm
from src.rag.router import get_insight_router

logger = logging.getLogger(__name__)

FilterKey = Literal[
    "province",
    "district",
    "age_group",
    "sex",
    "occupation",
    "education_level",
    "hobby",
    "skill",
    "keyword",
]
Intent = Literal["search", "stats", "profile", "recommend", "influence", "reset", "general"]

MAX_HISTORY_TURNS = 5
MAX_CHAT_SESSIONS = 1_000
DEFAULT_PAGE_SIZE = 5
MAX_SYNTHESIS_RESULTS = 20
RESET_KEYWORDS = ("리셋", "초기화", "처음부터", "새로 시작")
ACCUMULATE_KEYWORDS = ("그중", "그 중", "거기서", "추가로", "그리고", "또", "더")
REPLACE_KEYWORDS = ("대신", "말고", "아니고")
DESCRIPTIVE_SEARCH_MARKERS = ("같은 사람", "비슷한 사람", "성향", "라이프스타일", "누구인지", "누구야")
DESCRIPTIVE_KEYWORD_CANDIDATES = (
    "디저트",
    "소금빵",
    "케이크",
    "식도락",
    "빵집",
    "여행",
    "맛집",
    "카페",
    "음악",
    "필라테스",
    "연극",
    "웹툰",
)
DISTRICT_FALSE_POSITIVES = {"친구", "구경", "여가시", "가구"}

PROVINCES = (
    "서울",
    "부산",
    "대구",
    "인천",
    "광주",
    "대전",
    "울산",
    "세종",
    "경기",
    "강원",
    "충북",
    "충청북도",
    "충남",
    "충청남도",
    "전북",
    "전라북도",
    "전남",
    "전라남도",
    "경북",
    "경상북도",
    "경남",
    "경상남도",
    "제주",
)

FILTER_KEY_LABELS: dict[FilterKey, str] = {
    "province": "지역",
    "district": "시군구",
    "age_group": "연령대",
    "sex": "성별",
    "occupation": "직업",
    "education_level": "학력",
    "hobby": "취미",
    "skill": "기술",
    "keyword": "키워드",
}


class FilterState(TypedDict, total=False):
    province: str
    district: str
    age_group: str
    sex: str
    occupation: str
    education_level: str
    hobby: str
    skill: str
    keyword: str


class ChatMessage(TypedDict):
    role: str
    content: str


class ChatState(TypedDict, total=False):
    session_id: str
    history: list[ChatMessage]
    current_filters: FilterState
    last_intent: Intent | None
    turn_count: int
    pending_message: str
    intent: Intent
    stats_dimension: str | None
    response: str
    sources: list[dict[str, Any]]
    raw_results: list[dict[str, Any]]


class InsightAnswerRouter(Protocol):
    def ask(self, question: str) -> dict[str, Any]: ...


class ChatGraph:
    def __init__(self, max_sessions: int = MAX_CHAT_SESSIONS, insight_router: InsightAnswerRouter | None = None) -> None:
        self.max_sessions = max(1, max_sessions)
        self.insight_router = insight_router
        self.sessions: OrderedDict[str, ChatState] = OrderedDict()
        self.graph = self._build_graph()

    def invoke(self, session_id: str, message: str) -> dict[str, Any]:
        normalized_message = message.strip()
        state = self._get_or_create_state(session_id)
        result = self.graph.invoke({**state, "pending_message": normalized_message})
        state.update(
            {
                "history": result.get("history", []),
                "current_filters": result.get("current_filters", {}),
                "last_intent": result.get("last_intent"),
                "turn_count": int(result.get("turn_count", 0)),
            }
        )

        return {
            "response": result.get("response", ""),
            "context_filters": dict(state.get("current_filters", {})),
            "sources": result.get("sources", []),
            "turn_count": int(state.get("turn_count", 0)),
        }

    def clear(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)

    def _get_or_create_state(self, session_id: str) -> ChatState:
        if session_id in self.sessions:
            self.sessions.move_to_end(session_id)
        else:
            self._evict_oldest_session_if_needed()
            self.sessions[session_id] = {
                "session_id": session_id,
                "history": [],
                "current_filters": {},
                "last_intent": None,
                "turn_count": 0,
            }
        return self.sessions[session_id]

    def _evict_oldest_session_if_needed(self) -> None:
        while len(self.sessions) >= self.max_sessions:
            self.sessions.popitem(last=False)

    def _build_graph(self):  # noqa: ANN202
        graph = StateGraph(ChatState)
        graph.add_node("classify", self._classify_node)
        graph.add_node("merge_filters", self._merge_filters_node)
        graph.add_node("respond", self._respond_node)
        graph.add_node("synthesize", self._synthesize_node)
        graph.add_node("commit_history", self._commit_history_node)
        graph.set_entry_point("classify")
        graph.add_edge("classify", "merge_filters")
        graph.add_edge("merge_filters", "respond")
        graph.add_edge("respond", "synthesize")
        graph.add_edge("synthesize", "commit_history")
        graph.add_edge("commit_history", END)
        return graph.compile()

    def _classify_node(self, state: ChatState) -> ChatState:
        message = state.get("pending_message", "")
        intent = classify_intent(message, state)
        result: ChatState = {"intent": intent}
        if intent == "stats":
            result["stats_dimension"] = _infer_requested_stats_dimension(message)
        return result

    def _merge_filters_node(self, state: ChatState) -> ChatState:
        message = state.get("pending_message", "")
        intent = state.get("intent", "general")
        if intent == "reset":
            return {"current_filters": {}}
        current_filters = state.get("current_filters", {})
        extracted = extract_filters(message)
        return {"current_filters": merge_filters(current_filters, extracted, message)}

    def _respond_node(self, state: ChatState) -> ChatState:
        intent = state.get("intent", "general")
        if intent == "reset":
            return {
                "response": "필터를 초기화했습니다. 새 조건으로 다시 탐색할 수 있습니다.",
                "sources": [{"type": "reset"}],
                "raw_results": [],
            }
        response, sources, raw_results = self._respond(intent, state)
        return {"response": response, "sources": sources, "raw_results": raw_results}

    def _synthesize_node(self, state: ChatState) -> ChatState:
        intent = state.get("intent", "general")
        if intent not in ("search", "stats"):
            return {}
        raw_results = state.get("raw_results", [])
        if not raw_results:
            return {}
        try:
            return {
                "response": self._synthesize_response(
                    intent=intent,
                    message=state.get("pending_message", ""),
                    filters=state.get("current_filters", {}),
                    sources=state.get("sources", []),
                    raw_results=raw_results,
                )
            }
        except Exception:
            logger.exception("Failed to synthesize chat response; falling back to template response")
            return {}

    def _commit_history_node(self, state: ChatState) -> ChatState:
        history = state.get("history", [])
        message = state.get("pending_message", "")
        response = state.get("response", "")
        return {
            "history": trim_history(
                [
                    *history,
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": response},
                ]
            ),
            "last_intent": state.get("intent", "general"),
            "turn_count": int(state.get("turn_count", 0)) + 1,
        }

    def _respond(self, intent: Intent, state: ChatState) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
        current_filters = state.get("current_filters", {})
        if intent == "stats":
            return self._run_stats(current_filters, state.get("stats_dimension"))
        if intent == "search":
            return self._run_search(current_filters)
        if intent == "general":
            return self._run_general(state.get("pending_message", ""), current_filters)
        return _general_response(), [{"type": "general"}], []

    def _run_general(self, message: str, filters: FilterState) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
        try:
            router = self.insight_router or get_insight_router()
            result = router.ask(_format_insight_question(message, filters))
        except Exception:
            logger.exception("Failed to route general chat question through InsightRouter")
            return _general_response(), [{"type": "general"}], []

        answer = str(result.get("answer") or "").strip()
        if not answer:
            return _general_response(), [{"type": "general"}], []
        return (
            answer,
            [
                {
                    "type": "insight",
                    "query_type": result.get("query_type", "unknown"),
                    "sources": result.get("sources", []),
                }
            ],
            [],
        )

    def _run_search(self, filters: FilterState) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
        data_query, count_query, params = build_search_query(
            province=_as_list(filters.get("province")),
            district=_as_list(filters.get("district")),
            age_group=_as_list(filters.get("age_group")),
            sex=filters.get("sex"),
            occupation=_as_list(filters.get("occupation")),
            education_level=_as_list(filters.get("education_level")),
            hobby=_as_list(filters.get("hobby")),
            skill=_as_list(filters.get("skill")),
            keyword=filters.get("keyword"),
            sort_by="age",
            sort_order="asc",
            page=1,
            page_size=DEFAULT_PAGE_SIZE,
        )
        driver = GraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))
        try:
            with driver.session(database=settings.NEO4J_DATABASE) as session:
                count_record = session.run(Query(cast(LiteralString, count_query)), **cast(dict[str, Any], params)).single()
                total_count = int(count_record["total_count"]) if count_record else 0
                records = [
                    dict(record)
                    for record in session.run(Query(cast(LiteralString, data_query)), **cast(dict[str, Any], params))
                ] if total_count else []
        finally:
            driver.close()

        if not records:
            return (
                f"{_format_filter_summary(filters)} 조건에 맞는 페르소나를 찾지 못했습니다.",
                [{"type": "search", "filters": dict(filters), "total_count": total_count}],
                [],
            )

        lines = [f"{_format_filter_summary(filters)} 조건에서 {total_count:,}명을 찾았습니다. 상위 {len(records)}명입니다."]
        for index, record in enumerate(records, start=1):
            name = record.get("display_name") or "이름 없음"
            age = record.get("age") or "-"
            sex = record.get("sex") or "-"
            occupation = record.get("occupation") or "직업 정보 없음"
            uuid = record.get("uuid") or ""
            lines.append(f"{index}. {name} ({age}세/{sex}, {occupation}) - {uuid}")
        return "\n".join(lines), [{"type": "search", "filters": dict(filters), "total_count": total_count}], records

    def _run_stats(
        self, filters: FilterState, requested_dimension: str | None = None
    ) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
        dimension = requested_dimension or _infer_stats_dimension(filters)
        query, params = build_dimension_query(
            dimension,
            province=filters.get("province"),
            age_group=filters.get("age_group"),
            sex=filters.get("sex"),
            occupation=filters.get("occupation"),
            keyword=filters.get("keyword"),
        )
        params["limit"] = 5
        driver = GraphDatabase.driver(settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD))
        try:
            with driver.session(database=settings.NEO4J_DATABASE) as session:
                records = [
                    dict(record)
                    for record in session.run(Query(cast(LiteralString, query)), **cast(dict[str, Any], params))
                ]
        finally:
            driver.close()

        if not records:
            return (
                f"제가 이해한 조건은 {_format_filter_summary(filters)}이고, 요청하신 항목은 {_dimension_label(dimension)}입니다.\n\n"
                "이 조건으로는 통계 결과를 찾지 못했습니다. "
                "조건을 조금 넓히거나 표현을 바꿔 다시 질문해 주세요.",
                [{"type": "stats", "dimension": dimension, "filters": dict(filters)}],
                [],
            )

        title = _dimension_label(dimension)
        lines = [
            f"제가 이해한 조건은 {_format_filter_summary(filters)}이고, 요청하신 항목은 {title}입니다.",
            "",
            f"{title} 상위 분포는 다음과 같습니다.",
        ]
        for index, record in enumerate(records, start=1):
            lines.append(f"{index}. {record.get('label')} ({int(record.get('count') or 0):,}명)")
        return "\n".join(lines), [{"type": "stats", "dimension": dimension, "filters": dict(filters)}], records

    def _synthesize_response(
        self,
        *,
        intent: Intent,
        message: str,
        filters: FilterState,
        sources: list[dict[str, Any]],
        raw_results: list[dict[str, Any]],
    ) -> str:
        source = sources[0] if sources else {}
        result_lines = _format_results_jsonl(raw_results)
        system_prompt = (
            "당신은 한국인 페르소나 데이터를 분석하는 어시스턴트입니다. "
            "아래 제공된 데이터만 근거로 답변하세요. "
            "데이터에 없는 내용은 추측하지 마세요. "
            "수치를 인용할 때는 정확한 값을 사용하세요. "
            "한국어로 자연스럽고 간결하게 답변하세요."
        )
        user_prompt = (
            f"사용자 질문: {message}\n"
            f"요청 유형: {intent}\n"
            f"현재 필터: {_format_filter_summary(filters)}\n"
            f"메타데이터: {json.dumps(source, ensure_ascii=False, default=str)}\n"
            "조회 결과(JSON Lines, 최대 20건):\n"
            f"{result_lines}"
        )
        llm = create_llm()
        response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
        content = getattr(response, "content", "")
        if isinstance(content, str):
            synthesized = content.strip()
        elif isinstance(content, list):
            synthesized = "".join(str(part) for part in content).strip()
        else:
            synthesized = str(content).strip()
        if not synthesized:
            raise ValueError("LLM returned an empty chat synthesis")
        return synthesized


def classify_intent(message: str, state: ChatState | None = None) -> Intent:
    if _has_any(message, RESET_KEYWORDS):
        return "reset"
    if state and state.get("last_intent") in ("search", "stats") and (
        _has_any(message, ACCUMULATE_KEYWORDS) or _has_any(message, REPLACE_KEYWORDS)
    ):
        return state.get("last_intent") or "search"
    if _is_work_description_question(message):
        return "general"
    if any(keyword in message for keyword in ("분포", "통계", "많", "순위", "비율", "어떤 취미", "어떤 직업", "취미는", "취미가", "직업은", "직업이", "학력", "교육", "혼인", "결혼", "병역", "주거", "가구", "여가", "여가시간", "무엇", "뭐해", "뭐 하", "하는편", "하는 편", "즐겨")):
        return "stats"
    if any(keyword in message for keyword in ("보여", "찾", "검색", "좁혀", "사람", "페르소나")):
        return "search"
    return "general"


def extract_filters(message: str) -> FilterState:
    filters: FilterState = {}
    province = _extract_province(message)
    if province:
        filters["province"] = province

    district = None if _is_descriptive_persona_search(message) else _extract_district(message)
    if district:
        filters["district"] = district

    age_group = _extract_age_group(message)
    if age_group:
        filters["age_group"] = age_group

    sex = _extract_sex(message)
    if sex:
        filters["sex"] = sex

    occupation = _extract_after_keywords(message, ("직업", "개발자", "디자이너", "교사", "학생", "의사", "뷰티", "미용"))
    if occupation:
        filters["occupation"] = occupation

    hobby = _extract_named_value(message, ("취미가", "취미는"))
    if hobby:
        filters["hobby"] = hobby

    skill = _extract_named_value(message, ("기술이", "기술은", "스킬이", "스킬은"))
    if skill:
        filters["skill"] = skill

    keyword = _extract_keyword(message)
    if keyword:
        filters["keyword"] = keyword

    return filters


def merge_filters(current: FilterState, extracted: FilterState, message: str) -> FilterState:
    if _has_any(message, RESET_KEYWORDS):
        return {}
    merged = _copy_filters(current)
    for key, value in extracted.items():
        if value:
            merged[key] = value
    return merged


def trim_history(history: list[ChatMessage]) -> list[ChatMessage]:
    max_messages = MAX_HISTORY_TURNS * 2
    return history[-max_messages:]


def _copy_filters(filters: FilterState) -> FilterState:
    copied: FilterState = {}
    if "province" in filters:
        copied["province"] = filters["province"]
    if "district" in filters:
        copied["district"] = filters["district"]
    if "age_group" in filters:
        copied["age_group"] = filters["age_group"]
    if "sex" in filters:
        copied["sex"] = filters["sex"]
    if "occupation" in filters:
        copied["occupation"] = filters["occupation"]
    if "education_level" in filters:
        copied["education_level"] = filters["education_level"]
    if "hobby" in filters:
        copied["hobby"] = filters["hobby"]
    if "skill" in filters:
        copied["skill"] = filters["skill"]
    if "keyword" in filters:
        copied["keyword"] = filters["keyword"]
    return copied


def _extract_province(message: str) -> str | None:
    for province in PROVINCES:
        if province in message:
            return province
    return None


def _extract_district(message: str) -> str | None:
    for match in re.finditer(r"([가-힣]{1,8}(?:구|군|시))", message):
        district = match.group(1)
        next_char = message[match.end() : match.end() + 1]
        if district in PROVINCES or district in DISTRICT_FALSE_POSITIVES or next_char in ("간", "에", "들", "와", "과", "를", "을"):
            continue
        return district
    return None


def _extract_age_group(message: str) -> str | None:
    match = re.search(r"(\d{1,2})\s*대", message)
    if match:
        age_decade = int(match.group(1))
        if age_decade < 10:
            return None
        return f"{age_decade}대"
    return None


def _extract_sex(message: str) -> str | None:
    if any(token in message for token in ("남성", "남자")):
        return "남자"
    if any(token in message for token in ("여성", "여자")):
        return "여자"
    return None


def _extract_after_keywords(message: str, keywords: tuple[str, ...]) -> str | None:
    for keyword in keywords:
        if keyword in ("개발자", "디자이너", "교사", "학생", "의사") and keyword in message:
            return keyword
    match = re.search(
        r"(?:들\s*중에|들중에|중에서|중에)\s*([가-힣A-Za-z0-9+#.\s]{1,24}?)(?:\s*종사자|\s*업계|\s*직군)",
        message,
    )
    if match:
        return _clean_occupation_phrase(match.group(1))
    match = re.search(r"([가-힣A-Za-z0-9+#.\s]{2,24}?)(?:\s*종사자|\s*업계|\s*직군)", message)
    if match:
        return _clean_occupation_phrase(match.group(1))
    match = re.search(r"직업(?:이|은|은\s*)?\s*([가-힣A-Za-z0-9+#.]{2,20})", message)
    return _clean_extracted(match.group(1)) if match else None


def _extract_named_value(message: str, keywords: tuple[str, ...]) -> str | None:
    for keyword in keywords:
        match = re.search(rf"{keyword}\s*([가-힣A-Za-z0-9+#.]+)", message)
        if match:
            return _clean_extracted(match.group(1))
    return None


def _extract_keyword(message: str) -> str | None:
    match = re.search(r"키워드\s*([가-힣A-Za-z0-9+#.]{2,20})", message)
    if match:
        return _clean_extracted(match.group(1))
    if _is_descriptive_persona_search(message):
        for keyword in DESCRIPTIVE_KEYWORD_CANDIDATES:
            if keyword in message:
                return keyword
    return None


def _is_descriptive_persona_search(message: str) -> bool:
    return len(message) >= 40 and any(marker in message for marker in DESCRIPTIVE_SEARCH_MARKERS)


def _is_work_description_question(message: str) -> bool:
    compact_message = re.sub(r"\s+", "", message)
    return any(token in compact_message for token in ("어떤일", "무슨일", "하는일"))


def _clean_extracted(value: str) -> str:
    return re.sub(r"(인|인 사람|만|으로|로|보여줘|찾아줘|알려줘)$", "", value.strip())


def _clean_occupation_phrase(value: str) -> str:
    cleaned = _clean_extracted(value)
    cleaned = re.sub(r"(?:들\s*중에|들중에|중에서|중에|중)$", " ", cleaned)
    cleaned = re.sub(r"^(?:들\s*중에|들중에|중에서|중에|중)\s*", " ", cleaned)
    for province in PROVINCES:
        cleaned = cleaned.replace(province, " ")
    cleaned = re.sub(r"\d{2}\s*대", " ", cleaned)
    cleaned = re.sub(r"(?:남성|남자|여성|여자)", " ", cleaned)
    cleaned = re.sub(r"관련$", "", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _has_any(message: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in message for keyword in keywords)


def _as_list(value: str | None) -> list[str] | None:
    return [value] if value else None


def _format_results_jsonl(raw_results: list[dict[str, Any]]) -> str:
    limited_results = raw_results[:MAX_SYNTHESIS_RESULTS]
    if not limited_results:
        return "{}"
    return "\n".join(json.dumps(result, ensure_ascii=False, default=str) for result in limited_results)


def _format_insight_question(message: str, filters: FilterState) -> str:
    if not filters:
        return message
    return (
        f"현재 대화에서 유지 중인 필터는 {_format_filter_summary(filters)}입니다. "
        "이 조건을 우선 반영해 다음 질문에 답하세요.\n\n"
        f"질문: {message}"
    )


def _infer_stats_dimension(filters: FilterState) -> str:
    if filters.get("hobby"):
        return "hobby"
    if filters.get("skill"):
        return "skill"
    if filters.get("occupation"):
        return "occupation"
    return "hobby" if "hobby" in VALID_DIMENSIONS else "age"


def _infer_requested_stats_dimension(message: str) -> str | None:
    if any(token in message for token in ("기술", "스킬")):
        return "skill"
    if "직업" in message:
        return "occupation"
    if re.search(r"지역(?!내|에서)", message):
        return "province"
    if "성별" in message:
        return "sex"
    if any(token in message for token in ("나이", "연령")):
        return "age"
    if any(token in message for token in ("학력", "교육")):
        return "education"
    if any(token in message for token in ("혼인", "결혼")):
        return "marital"
    if "병역" in message:
        return "military"
    if "주거" in message:
        return "housing"
    if "가구" in message:
        return "family_type"
    if any(token in message for token in ("취미", "여가", "여가시간", "뭐해", "뭐 하", "무엇", "하는편", "하는 편", "즐겨")):
        return "hobby"
    return None


def _dimension_label(dimension: str) -> str:
    return {
        "hobby": "취미",
        "skill": "기술",
        "occupation": "직업",
        "age": "연령대",
        "sex": "성별",
        "province": "지역",
        "education": "학력",
        "marital": "혼인 상태",
        "military": "병역 상태",
        "housing": "주거 형태",
        "family_type": "가구 형태",
    }.get(dimension, dimension)


def _format_filter_summary(filters: FilterState) -> str:
    if not filters:
        return "전체"
    parts = [f"{FILTER_KEY_LABELS[cast(FilterKey, key)]}={value}" for key, value in filters.items()]
    return ", ".join(parts)


def _general_response() -> str:
    return (
        "검색 조건을 대화로 입력해 주세요. 예: '서울 20대 남성 보여줘', "
        "'그중에서 개발자만', '이 조건에서 취미 분포 알려줘', '리셋'."
    )
