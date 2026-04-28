from src.rag.chat_graph import ChatGraph, FilterState, classify_intent, extract_filters, merge_filters, trim_history


class FakeInsightRouter:
    def __init__(self, result: dict[str, object] | None = None, *, should_fail: bool = False) -> None:
        self.result = result or {"answer": "인사이트 응답", "sources": [{"uuid": "u-1"}], "query_type": "vector"}
        self.should_fail = should_fail
        self.questions: list[str] = []

    def ask(self, question: str) -> dict[str, object]:
        self.questions.append(question)
        if self.should_fail:
            raise RuntimeError("InsightRouter down")
        return self.result


def test_extract_filters_from_korean_message() -> None:
    filters = extract_filters("서울 20대 남성 개발자 보여줘")

    assert filters.get("province") == "서울"
    assert filters.get("age_group") == "20대"
    assert filters.get("sex") == "남자"
    assert filters.get("occupation") == "개발자"


def test_extract_filters_does_not_infer_sex_from_common_verbs() -> None:
    filters = extract_filters("서울 사람 보여줘")

    assert filters == {"province": "서울"}


def test_extract_filters_does_not_treat_stats_dimension_as_filter() -> None:
    filters = extract_filters("이 조건에서 취미 분포 알려줘")

    assert "hobby" not in filters


def test_extract_filters_ignores_invalid_single_digit_age_group() -> None:
    filters = extract_filters("0대 서울여성의 취미는?")

    assert filters == {"province": "서울", "sex": "여자"}


def test_extract_filters_does_not_treat_common_words_as_district() -> None:
    filters = extract_filters("30대 부산남성은 여가시간에 무엇을하는편?")

    assert filters == {"province": "부산", "age_group": "30대", "sex": "남자"}


def test_extract_filters_for_descriptive_persona_search() -> None:
    filters = extract_filters(
        "20대 여성중에 해운대나 전포동의 유명한 디저트 가게를 섭렵하기 위해 "
        "친구들과 함께 정교한 식도락 지도를 그려 여행합니다. "
        "단순히 풍경을 구경하는 일보다 그 지역에서만 맛볼 수 있는 특별한 소금빵이나 "
        "케이크를 찾아내어 맛보는 것에 모든 집중력을 쏟습니다 와 같은사람이 누구인지 알려줘"
    )

    assert filters.get("age_group") == "20대"
    assert filters.get("sex") == "여자"
    assert filters.get("keyword") == "디저트"
    assert "district" not in filters


def test_reported_hobby_question_is_stats_intent() -> None:
    assert classify_intent("서울 20대여성 뷰티 미용 종사자들의 취미는?") == "stats"


def test_extract_filters_from_reported_hobby_question() -> None:
    filters = extract_filters("서울 20대여성 뷰티 미용 종사자들의 취미는?")

    assert filters.get("province") == "서울"
    assert filters.get("age_group") == "20대"
    assert filters.get("sex") == "여자"
    assert filters.get("occupation") == "뷰티 미용"
    assert "hobby" not in filters


def test_extract_filters_cleans_connector_before_beauty_occupation() -> None:
    filters = extract_filters("20대여성들중에 뷰티관련종사자들은 어떤일을하는편?")

    assert filters.get("age_group") == "20대"
    assert filters.get("sex") == "여자"
    assert filters.get("occupation") == "뷰티"


def test_work_description_question_is_general_intent() -> None:
    assert classify_intent("20대여성들중에 뷰티관련종사자들은 어떤일을하는편?") == "general"


def test_personality_question_is_general_intent_with_filters() -> None:
    message = "20대여성들의 성격은 어떤편?"

    assert classify_intent(message) == "general"
    assert extract_filters(message) == {"age_group": "20대", "sex": "여자"}


def test_reported_hobby_question_runs_stats(monkeypatch) -> None:
    graph = ChatGraph()
    monkeypatch.setattr(
        graph,
        "_run_stats",
        lambda filters, requested_dimension=None: (
            "stats ok",
            [{"type": "stats", "dimension": requested_dimension, "filters": filters}],
            [],
        ),
    )

    result = graph.invoke("session-a", "서울 20대여성 뷰티 미용 종사자들의 취미는?")

    assert result["response"] == "stats ok"
    assert result["sources"][0]["type"] == "stats"
    assert result["sources"][0]["dimension"] == "hobby"
    assert result["context_filters"]["occupation"] == "뷰티 미용"


def test_education_question_is_stats_intent() -> None:
    assert classify_intent("20대 부산여성들의 학력은?") == "stats"


def test_education_question_runs_education_stats(monkeypatch) -> None:
    graph = ChatGraph()
    monkeypatch.setattr(
        graph,
        "_run_stats",
        lambda filters, requested_dimension=None: (
            "education stats ok",
            [{"type": "stats", "dimension": requested_dimension, "filters": filters}],
            [],
        ),
    )

    result = graph.invoke("session-a", "20대 부산여성들의 학력은?")

    assert result["response"] == "education stats ok"
    assert result["sources"][0]["dimension"] == "education"
    assert result["context_filters"] == {"province": "부산", "age_group": "20대", "sex": "여자"}


def test_education_distribution_after_hobby_question_changes_dimension(monkeypatch) -> None:
    graph = ChatGraph()

    def fake_run_stats(filters, requested_dimension=None):
        return (
            f"{requested_dimension} stats ok",
            [{"type": "stats", "dimension": requested_dimension, "filters": filters}],
            [],
        )

    monkeypatch.setattr(graph, "_run_stats", fake_run_stats)

    first_result = graph.invoke("session-a", "20대 부산 여성의 취미는?")
    second_result = graph.invoke("session-a", "20대 부산여성의 학력분포는?")

    assert first_result["sources"][0]["dimension"] == "hobby"
    assert second_result["sources"][0]["dimension"] == "education"
    assert second_result["context_filters"] == {"province": "부산", "age_group": "20대", "sex": "여자"}


def test_explicit_education_keyword_wins_over_generic_question_words(monkeypatch) -> None:
    graph = ChatGraph()
    monkeypatch.setattr(
        graph,
        "_run_stats",
        lambda filters, requested_dimension=None: (
            "education stats ok",
            [{"type": "stats", "dimension": requested_dimension, "filters": filters}],
            [],
        ),
    )

    result = graph.invoke("session-a", "20대 부산여성의 학력은 무엇인가요?")

    assert result["sources"][0]["dimension"] == "education"


def test_leisure_question_runs_hobby_stats(monkeypatch) -> None:
    graph = ChatGraph()
    monkeypatch.setattr(
        graph,
        "_run_stats",
        lambda filters, requested_dimension=None: (
            "hobby stats ok",
            [{"type": "stats", "dimension": requested_dimension, "filters": filters}],
            [],
        ),
    )

    result = graph.invoke("session-a", "30대 부산남성은 여가시간에 무엇을하는편?")

    assert result["response"] == "hobby stats ok"
    assert result["sources"][0]["dimension"] == "hobby"
    assert result["context_filters"] == {"province": "부산", "age_group": "30대", "sex": "남자"}


def test_contextual_region_reference_does_not_override_hobby(monkeypatch) -> None:
    graph = ChatGraph()
    monkeypatch.setattr(
        graph,
        "_run_stats",
        lambda filters, requested_dimension=None: (
            "hobby stats ok",
            [{"type": "stats", "dimension": requested_dimension, "filters": filters}],
            [],
        ),
    )

    result = graph.invoke("session-a", "그 지역내에서 여성들이 퇴근후 무엇을하는편?")

    assert result["sources"][0]["dimension"] == "hobby"


def test_stats_response_explains_interpreted_conditions(monkeypatch) -> None:
    graph = ChatGraph()
    monkeypatch.setattr(graph, "_synthesize_response", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("LLM down")))
    monkeypatch.setattr(
        "src.rag.chat_graph.build_dimension_query",
        lambda *args, **kwargs: ("RETURN '대학교' AS label, 3 AS count", {"limit": 5}),
    )

    class FakeSession:
        def run(self, query, **params):
            return [{"label": "대학교", "count": 3}]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

    class FakeDriver:
        def session(self, database=None):
            return FakeSession()

        def close(self):
            return None

    monkeypatch.setattr("src.rag.chat_graph.GraphDatabase.driver", lambda *args, **kwargs: FakeDriver())

    result = graph.invoke("session-a", "20대 부산여성들의 학력은?")

    assert "제가 이해한 조건" in result["response"]
    assert "요청하신 항목은 학력" in result["response"]


def test_stats_response_is_synthesized_from_raw_results(monkeypatch) -> None:
    graph = ChatGraph()
    captured = {}

    monkeypatch.setattr(
        graph,
        "_run_stats",
        lambda filters, requested_dimension=None: (
            "template stats",
            [{"type": "stats", "dimension": requested_dimension, "filters": filters}],
            [{"label": "독서", "count": 7}],
        ),
    )

    def fake_synthesize_response(**kwargs):
        captured.update(kwargs)
        return "LLM이 합성한 통계 응답"

    monkeypatch.setattr(graph, "_synthesize_response", fake_synthesize_response)

    result = graph.invoke("session-a", "서울 20대 여성의 취미는?")

    assert result["response"] == "LLM이 합성한 통계 응답"
    assert captured["intent"] == "stats"
    assert captured["filters"] == {"province": "서울", "age_group": "20대", "sex": "여자"}
    assert captured["raw_results"] == [{"label": "독서", "count": 7}]


def test_search_response_is_synthesized_from_raw_results(monkeypatch) -> None:
    graph = ChatGraph()
    captured = {}

    monkeypatch.setattr(
        graph,
        "_run_search",
        lambda filters: (
            "template search",
            [{"type": "search", "filters": filters, "total_count": 1}],
            [{"display_name": "김서울", "age": 25, "sex": "여자", "occupation": "개발자", "uuid": "u-1"}],
        ),
    )

    def fake_synthesize_response(**kwargs):
        captured.update(kwargs)
        return "LLM이 합성한 검색 응답"

    monkeypatch.setattr(graph, "_synthesize_response", fake_synthesize_response)

    result = graph.invoke("session-a", "서울 20대 여성 보여줘")

    assert result["response"] == "LLM이 합성한 검색 응답"
    assert captured["intent"] == "search"
    assert captured["raw_results"][0]["uuid"] == "u-1"


def test_llm_synthesis_failure_falls_back_to_template_response(monkeypatch) -> None:
    graph = ChatGraph()
    monkeypatch.setattr(
        graph,
        "_run_stats",
        lambda filters, requested_dimension=None: (
            "template fallback",
            [{"type": "stats", "dimension": requested_dimension, "filters": filters}],
            [{"label": "등산", "count": 3}],
        ),
    )
    monkeypatch.setattr(graph, "_synthesize_response", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("LLM down")))

    result = graph.invoke("session-a", "부산 30대 남성 취미는?")

    assert result["response"] == "template fallback"


def test_synthesis_prompt_limits_raw_results_to_twenty(monkeypatch) -> None:
    graph = ChatGraph()
    captured = {}

    class FakeResponse:
        content = "상위 결과만 요약했습니다."

    class FakeLlm:
        def invoke(self, messages):
            captured["prompt"] = messages[1].content
            return FakeResponse()

    monkeypatch.setattr("src.rag.chat_graph.create_llm", lambda: FakeLlm())

    response = graph._synthesize_response(
        intent="stats",
        message="취미 분포 알려줘",
        filters={},
        sources=[{"type": "stats", "dimension": "hobby"}],
        raw_results=[{"i": index} for index in range(25)],
    )

    assert response == "상위 결과만 요약했습니다."
    assert captured["prompt"].count('"i"') == 20
    assert '"i": 19' in captured["prompt"]
    assert '"i": 20' not in captured["prompt"]


def test_neo4j_driver_is_closed_before_synthesis(monkeypatch) -> None:
    graph = ChatGraph()
    driver_state = {"closed": False}
    monkeypatch.setattr(
        "src.rag.chat_graph.build_dimension_query",
        lambda *args, **kwargs: ("RETURN '독서' AS label, 3 AS count", {"limit": 5}),
    )

    class FakeSession:
        def run(self, query, **params):
            return [{"label": "독서", "count": 3}]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

    class FakeDriver:
        def session(self, database=None):
            return FakeSession()

        def close(self):
            driver_state["closed"] = True

    def fake_synthesize_response(**kwargs):
        assert driver_state["closed"] is True
        return "드라이버 종료 후 합성"

    monkeypatch.setattr("src.rag.chat_graph.GraphDatabase.driver", lambda *args, **kwargs: FakeDriver())
    monkeypatch.setattr(graph, "_synthesize_response", fake_synthesize_response)

    result = graph.invoke("session-a", "서울 20대 여성 취미는?")

    assert result["response"] == "드라이버 종료 후 합성"


def test_general_intent_uses_insight_router() -> None:
    router = FakeInsightRouter({"answer": "고급 분석 응답", "sources": [{"type": "vector"}], "query_type": "vector"})
    graph = ChatGraph(insight_router=router)

    result = graph.invoke("session-a", "이 데이터셋의 특징을 설명해줘")

    assert result["response"] == "고급 분석 응답"
    assert result["sources"][0]["type"] == "insight"
    assert result["sources"][0]["query_type"] == "vector"
    assert router.questions == ["이 데이터셋의 특징을 설명해줘"]


def test_general_intent_passes_current_filters_as_context_prefix(monkeypatch) -> None:
    router = FakeInsightRouter()
    graph = ChatGraph(insight_router=router)
    monkeypatch.setattr(graph, "_run_search", lambda filters: ("search ok", [{"type": "search", "filters": filters}], []))

    graph.invoke("session-a", "서울 20대 여성 보여줘")
    result = graph.invoke("session-a", "이 조건의 라이프스타일을 분석해줘")

    assert result["response"] == "인사이트 응답"
    assert "현재 대화에서 유지 중인 필터는 지역=서울, 연령대=20대, 성별=여자입니다" in router.questions[-1]
    assert "질문: 이 조건의 라이프스타일을 분석해줘" in router.questions[-1]


def test_work_description_query_uses_insight_router_with_corrected_filters(monkeypatch) -> None:
    router = FakeInsightRouter({"answer": "뷰티 직무 분석", "sources": [], "query_type": "vector"})
    graph = ChatGraph(insight_router=router)
    monkeypatch.setattr(graph, "_run_search", lambda filters: ("search ok", [{"type": "search", "filters": filters}], []))

    graph.invoke("session-a", "부산 20대 여성 보여줘")
    result = graph.invoke("session-a", "20대여성들중에 뷰티관련종사자들은 어떤일을하는편?")

    assert result["response"] == "뷰티 직무 분석"
    assert result["context_filters"] == {"province": "부산", "age_group": "20대", "sex": "여자", "occupation": "뷰티"}
    assert "지역=부산, 연령대=20대, 성별=여자, 직업=뷰티" in router.questions[-1]
    assert "질문: 20대여성들중에 뷰티관련종사자들은 어떤일을하는편?" in router.questions[-1]


def test_personality_query_uses_insight_router_with_filters() -> None:
    router = FakeInsightRouter({"answer": "20대 여성 성격 분석", "sources": [], "query_type": "vector"})
    graph = ChatGraph(insight_router=router)

    result = graph.invoke("session-a", "20대여성들의 성격은 어떤편?")

    assert result["response"] == "20대 여성 성격 분석"
    assert result["context_filters"] == {"age_group": "20대", "sex": "여자"}
    assert "연령대=20대, 성별=여자" in router.questions[-1]
    assert "질문: 20대여성들의 성격은 어떤편?" in router.questions[-1]


def test_general_intent_insight_response_is_saved_to_history() -> None:
    router = FakeInsightRouter({"answer": "히스토리에 저장될 응답", "sources": [], "query_type": "vector"})
    graph = ChatGraph(insight_router=router)

    graph.invoke("session-a", "데이터셋 설명해줘")

    history = graph.sessions["session-a"].get("history", [])
    assert history[-1] == {"role": "assistant", "content": "히스토리에 저장될 응답"}


def test_general_intent_falls_back_when_insight_router_fails() -> None:
    graph = ChatGraph(insight_router=FakeInsightRouter(should_fail=True))

    result = graph.invoke("session-a", "데이터셋 설명해줘")

    assert result["response"] == (
        "검색 조건을 대화로 입력해 주세요. 예: '서울 20대 남성 보여줘', "
        "'그중에서 개발자만', '이 조건에서 취미 분포 알려줘', '리셋'."
    )
    assert result["sources"] == [{"type": "general"}]


def test_filter_retention_across_turns(monkeypatch) -> None:
    graph = ChatGraph()
    monkeypatch.setattr(graph, "_run_search", lambda filters: ("ok", [{"type": "search"}], []))

    graph.invoke("session-a", "서울 사람 보여줘")
    result = graph.invoke("session-a", "그중에서 20대만")

    assert result["context_filters"]["province"] == "서울"
    assert result["context_filters"]["age_group"] == "20대"


def test_filter_replacement_logic() -> None:
    current: FilterState = {"province": "서울", "age_group": "20대"}
    merged = merge_filters(current, {"province": "부산"}, "대신 부산")

    assert merged == {"province": "부산", "age_group": "20대"}


def test_replace_phrase_keeps_previous_search_intent(monkeypatch) -> None:
    graph = ChatGraph()
    monkeypatch.setattr(graph, "_run_search", lambda filters: ("search ok", [{"type": "search", "filters": filters}], []))

    graph.invoke("session-a", "서울 사람 보여줘")
    result = graph.invoke("session-a", "대신 부산")

    assert result["response"] == "search ok"
    assert result["sources"][0]["type"] == "search"
    assert result["context_filters"]["province"] == "부산"


def test_extract_filters_does_not_treat_household_as_district() -> None:
    filters = extract_filters("1인 가구 비율 알려줘")

    assert "district" not in filters


def test_filter_reset_command(monkeypatch) -> None:
    graph = ChatGraph()
    monkeypatch.setattr(graph, "_run_search", lambda filters: ("ok", [{"type": "search"}], []))

    graph.invoke("session-a", "서울 20대 보여줘")
    result = graph.invoke("session-a", "처음부터")

    assert result["context_filters"] == {}


def test_max_history_retention(monkeypatch) -> None:
    graph = ChatGraph()
    monkeypatch.setattr(graph, "_run_search", lambda filters: ("ok", [{"type": "search"}], []))

    for index in range(6):
        graph.invoke("session-a", f"서울 {20 + index}대 보여줘")

    assert len(graph.sessions["session-a"].get("history", [])) == 10


def test_session_isolation(monkeypatch) -> None:
    graph = ChatGraph()
    monkeypatch.setattr(graph, "_run_search", lambda filters: ("ok", [{"type": "search"}], []))

    graph.invoke("session-a", "서울 사람 보여줘")
    graph.invoke("session-b", "부산 사람 보여줘")

    assert graph.sessions["session-a"].get("current_filters", {}).get("province") == "서울"
    assert graph.sessions["session-b"].get("current_filters", {}).get("province") == "부산"


def test_session_store_evicts_oldest_session(monkeypatch) -> None:
    graph = ChatGraph(max_sessions=2)
    monkeypatch.setattr(graph, "_run_search", lambda filters: ("ok", [{"type": "search"}], []))

    graph.invoke("session-a", "서울 사람 보여줘")
    graph.invoke("session-b", "부산 사람 보여줘")
    graph.invoke("session-c", "대구 사람 보여줘")

    assert list(graph.sessions) == ["session-b", "session-c"]


def test_session_store_refreshes_recently_used_session(monkeypatch) -> None:
    graph = ChatGraph(max_sessions=2)
    monkeypatch.setattr(graph, "_run_search", lambda filters: ("ok", [{"type": "search"}], []))

    graph.invoke("session-a", "서울 사람 보여줘")
    graph.invoke("session-b", "부산 사람 보여줘")
    graph.invoke("session-a", "그중에서 20대만")
    graph.invoke("session-c", "대구 사람 보여줘")

    assert list(graph.sessions) == ["session-a", "session-c"]


def test_trim_history_keeps_last_five_turns() -> None:
    history = []
    for index in range(12):
        history.append({"role": "user", "content": str(index)})

    trimmed = trim_history(history)

    assert len(trimmed) == 10
    assert trimmed[0]["content"] == "2"
