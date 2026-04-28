import os
import json
import re
import uuid
from html import escape as html_escape
from typing import Any, cast
from urllib.parse import quote, urlparse

import requests
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
DEFAULT_PERSONA_UUID = "a5ad493e75e74e5cb4a81ac934a1db8f"
UUID_PATTERN = re.compile(r"^[0-9a-fA-F]{32}$")

RELATION_LABELS = {
    "ENJOYS_HOBBY": "취미",
    "HAS_SKILL": "보유 스킬",
    "WORKS_AS": "직업",
    "LIVES_IN": "거주 지역",
    "LIVES_IN_HOUSING": "주거 형태",
    "LIVES_WITH": "가구 형태",
    "EDUCATED_AT": "학력",
    "MARITAL_STATUS": "혼인 상태",
    "MILITARY_STATUS": "병역 상태",
    "SIMILAR_TO": "유사 페르소나",
}

NODE_TYPE_LABELS = {
    "Person": "페르소나",
    "Hobby": "취미",
    "Skill": "스킬",
    "Occupation": "직업",
    "District": "지역",
    "HousingType": "주거",
    "FamilyType": "가구",
    "EducationLevel": "학력",
    "MaritalStatus": "혼인",
    "MilitaryStatus": "병역",
    "Province": "시도",
    "Country": "국가",
    "Field": "전공",
}

NODE_STYLES = {
    "Person": {"color": "#4A90D9", "shape": "dot", "size": 24},
    "Hobby": {"color": "#7ED321", "shape": "box", "size": 18},
    "Skill": {"color": "#F5A623", "shape": "diamond", "size": 18},
    "District": {"color": "#D0021B", "shape": "triangle", "size": 18},
    "Province": {"color": "#BD10E0", "shape": "triangleDown", "size": 18},
    "Occupation": {"color": "#9013FE", "shape": "hexagon", "size": 18},
    "EducationLevel": {"color": "#50E3C2", "shape": "square", "size": 18},
    "HousingType": {"color": "#B8E986", "shape": "box", "size": 16},
    "FamilyType": {"color": "#F8E71C", "shape": "box", "size": 16},
    "MaritalStatus": {"color": "#FFB6C1", "shape": "ellipse", "size": 16},
    "MilitaryStatus": {"color": "#9B9B9B", "shape": "ellipse", "size": 16},
}

CENTRALITY_METRIC_LABELS = {
    "pagerank": "PageRank",
    "betweenness": "Betweenness",
    "degree": "Degree",
}

RECOMMENDATION_CATEGORY_LABELS = {
    "hobby": "취미",
    "skill": "기술",
    "occupation": "직업",
    "district": "지역",
}

def main() -> None:
    st.set_page_config(page_title="Persona KG Dashboard", layout="wide")
    validate_api_base_url(API_BASE_URL)
    init_session_state()
    _apply_pending_uuid_sync()
    st.title("Korean Persona Knowledge Graph 📊")
    st.caption("100만 한국인 페르소나 지식 그래프 대시보드 및 AI 분석 시스템")
    selected_label = st.session_state.get("selected_persona_label") or "아직 선택되지 않음"
    st.info(f"현재 선택: **{selected_label}**")

    tabs = st.tabs([
        "📊 대시보드", "🔍 검색/필터", "👤 프로필 뷰", "⚖️ 세그먼트 비교", "🕸️ 지식 그래프",
        "🌐 핵심 인물", "🤖 대화형 탐색", "💡 인사이트 질의", "🤝 유사 페르소나", "🏘️ 커뮤니티", "🗺️ 관계 경로"
    ])

    with tabs[0]: render_dashboard_tab()
    with tabs[1]: render_search_tab()
    with tabs[2]: render_profile_tab()
    with tabs[3]: render_compare_tab()
    with tabs[4]: render_graph_tab()
    
    with tabs[5]: render_influence_tab()
    with tabs[6]: render_chat_tab()
    with tabs[7]: render_insight_tab()
    with tabs[8]: render_similar_tab()
    with tabs[9]: render_communities_tab()
    with tabs[10]: render_path_tab()

def render_dashboard_tab() -> None:
    st.header("📊 전체 인구통계 대시보드")
    st.caption("첫 화면에서 전체 데이터 규모와 주요 분포를 바로 확인할 수 있습니다.")
    if st.button("통계 새로고침", key="refresh_stats"):
        get_stats_cached.clear()

    with st.spinner("대시보드 통계를 불러오는 중..."):
        try:
            stats = get_stats_cached()
            st.metric("총 페르소나 수", f"{stats.get('total_personas', 0):,}명")
            
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("연령대 분포")
                df_age = pd.DataFrame(stats['age_distribution']).set_index("label")
                st.bar_chart(df_age['count'])
            with c2:
                st.subheader("성별 분포")
                df_sex = pd.DataFrame(stats['sex_distribution']).set_index("label")
                st.bar_chart(df_sex['count'])
            
            c3, c4 = st.columns(2)
            with c3:
                st.subheader("거주 지역 분포")
                df_prov = pd.DataFrame(stats['province_distribution']).set_index("label")
                st.bar_chart(df_prov['count'])
            with c4:
                st.subheader("가장 인기 있는 취미 (Top 20)")
                df_hobby = pd.DataFrame(stats['top_hobbies']).set_index("label")
                st.bar_chart(df_hobby['count'])
        except Exception as e:
            st.error(format_error_message(e))

def render_search_tab() -> None:
    st.header("🔍 페르소나 상세 검색")
    if "search_filters" not in st.session_state:
        st.session_state.search_filters = {
            "sort_by": "age",
            "sort_order": "asc",
            "page_size": 20,
            "page": 1,
        }
    if "search_results" not in st.session_state:
        st.session_state.search_results = None

    with st.container(border=True):
        with st.expander("검색 조건", expanded=True):
            c1, c2, c3 = st.columns(3)
            province = c1.text_input("지역 (예: 서울)", key="search_province")
            age_group = c2.text_input("연령대 (예: 20대)", key="search_age_group")
            sex = c3.selectbox("성별", ["전체", "남자", "여자"], key="search_sex")

            c4, c5, c6 = st.columns(3)
            hobby = c4.text_input("취미 (예: 등산)", key="search_hobby")
            occupation = c5.text_input("직업명 키워드", key="search_occupation")
            keyword = c6.text_input("성향 키워드 (예: 직장인)", key="search_keyword")

            SORT_BY_OPTIONS = {"나이": "age", "이름": "display_name"}
            SORT_ORDER_OPTIONS = {"오름차순": "asc", "내림차순": "desc"}
            c7, c8, c9 = st.columns(3)
            sort_by_label = c7.selectbox("정렬 기준", list(SORT_BY_OPTIONS.keys()), index=0, key="search_sort_by")
            sort_order_label = c8.selectbox("정렬 방향", list(SORT_ORDER_OPTIONS.keys()), index=0, key="search_sort_order")
            sort_by = SORT_BY_OPTIONS[sort_by_label]
            sort_order = SORT_ORDER_OPTIONS[sort_order_label]
            page_size = c9.select_slider("한 번에 보기", options=[10, 20, 30, 50, 100], value=20, key="search_page_size")

    if st.button("검색하기", key="search_btn", type="primary", use_container_width=True):
        st.session_state.search_detail_uuid = ""
        effective_sex = sex if sex != "전체" else ""
        st.session_state.search_filters = {
            **{key: value for key, value in {
                "province": province,
                "age_group": age_group,
                "sex": effective_sex,
                "hobby": hobby,
                "occupation": occupation,
                "keyword": keyword,
                "sort_by": sort_by,
                "sort_order": sort_order,
                "page_size": page_size,
                "page": 1,
            }.items() if value},
            "sort_by": sort_by,
            "sort_order": sort_order,
            "page_size": page_size,
            "page": 1,
        }

        with st.spinner("검색 중..."):
            _run_search(cast(dict[str, object], st.session_state.search_filters))

    def _go_to_page(target_page: int) -> None:
        if st.session_state.search_results is None:
            return
        active_filters = dict(st.session_state.search_filters)
        active_filters["page"] = max(1, target_page)
        st.session_state.search_filters = active_filters
        with st.spinner("페이지 이동 중..."):
            _run_search(cast(dict[str, object], st.session_state.search_filters))

    if st.session_state.search_results is not None:
        res = st.session_state.search_results
        total_count = int(res.get("total_count", 0))
        page = int(res.get("page", 1))
        page_size = int(res.get("page_size", 20))
        total_pages = int(res.get("total_pages", 0))

        st.success(f"검색 완료: 총 {total_count:,}명 발견")
        st.caption(f"페이지 {page:,} / {total_pages or 1} · 페이지 크기 {page_size:,}명")

        results = res.get("results", [])
        if results:
            st.markdown("#### 검색 결과")
            for idx, persona in enumerate(results):
                _render_search_result_card(persona, idx)

            if total_pages > 1:
                p_prev, p_mid, p_next = st.columns(3)
                with p_prev:
                    if st.button("이전 페이지", key="search_prev", disabled=page <= 1):
                        _go_to_page(page - 1)
                with p_mid:
                    st.write(f"현재 {page} / {total_pages}")
                with p_next:
                    if st.button("다음 페이지", key="search_next", disabled=page >= total_pages):
                        _go_to_page(page + 1)
        else:
            st.warning("조건에 맞는 페르소나가 없습니다.")

def _run_search(filters: dict[str, object]) -> None:
    payload = {key: value for key, value in filters.items() if value is not None}
    if not payload:
        st.session_state.search_results = {"total_count": 0, "page": 1, "page_size": 20, "total_pages": 0, "results": []}
        return

    try:
        st.session_state.search_results = get_json("/api/search", params=payload)
    except Exception as e:
        st.session_state.search_results = None
        st.error(format_error_message(e))


def _render_search_detail_dialog(detail_uuid: str) -> None:
    target_uuid = str(detail_uuid or "").strip()
    if not target_uuid:
        return

    @st.dialog("선택한 사람 상세 보기", width="large")
    def _search_detail_modal() -> None:
        _render_persona_profile_content(
            target_uuid,
            show_close_button=True,
            close_target_key="search_detail_uuid",
        )

    _search_detail_modal()


def _render_search_result_card(persona: dict[str, object], index: int) -> None:
    name = str(persona.get("display_name", "이름 없음"))
    uuid = str(persona.get("uuid", ""))
    age = persona.get("age")
    sex = str(persona.get("sex", ""))
    province = str(persona.get("province", ""))
    district = str(persona.get("district", ""))
    occupation = str(persona.get("occupation", ""))
    education_level = str(persona.get("education_level", ""))
    persona_text = str(persona.get("persona", "")).strip()

    with st.container(border=True):
        left, right = st.columns([4, 1], vertical_alignment="center")
        with left:
            age_text = f"{age}세" if age is not None else "연령 미확인"
            st.markdown(f"**{name}** · {age_text}{' · ' + sex if sex else ''}")

            location = " ".join([part for part in (province, district) if part])
            details = [part for part in [location, occupation, education_level] if part]
            if details:
                st.markdown(" • ".join(details))
            st.caption(f"UUID: {short_uuid(uuid)}")

            if persona_text:
                render_collapsible_text(persona_text, max_chars=180, max_lines=2, expander_label="성향 전문 보기")

        with right:
            if st.button("이 사람 선택", key=f"select_persona_{uuid}_{index}", use_container_width=True):
                if uuid and uuid != st.session_state.selected_uuid:
                    set_selected_persona(uuid, name)
                    st.rerun()
            if st.button("상세 보기", key=f"view_persona_{uuid}_{index}", use_container_width=True):
                _render_search_detail_dialog(uuid)
            st.write("")


def _truncate_text(value: str, max_length: int = 180) -> str:
    text = value.strip()
    if len(text) <= max_length:
        return text
    return text[:max_length] + "…"


def render_collapsible_text(
    text: str,
    *,
    max_chars: int = 180,
    max_lines: int = 3,
    expander_label: str = "자세히 보기",
) -> None:
    """Render long text with compact preview plus optional expansion."""
    raw = str(text or "").strip()
    if not raw:
        return

    preview = _truncate_text(raw, max_chars)
    preview_html = f"<div style='display:-webkit-box; -webkit-line-clamp:{max_lines}; "
    preview_html += "-webkit-box-orient:vertical; overflow:hidden; line-height:1.35em; margin-bottom:0.35rem'>"
    preview_html += f"{html_escape(preview)}</div>"
    st.markdown(preview_html, unsafe_allow_html=True)

    if len(raw) > max_chars:
        with st.expander(expander_label, expanded=False):
            st.write(raw)


def _render_persona_profile_content(
    uuid: str,
    *,
    show_close_button: bool = False,
    close_target_key: str | None = None,
) -> None:
    target_uuid = str(uuid or "").strip()
    if not target_uuid:
        st.info("표시할 대상이 없습니다.")
        return

    if not UUID_PATTERN.fullmatch(target_uuid):
        st.error("UUID 형식이 잘못되었습니다.")
        return

    try:
        res = get_json(api_path("/api/persona", target_uuid))
    except Exception as e:
        st.error(format_error_message(e))
        return

    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader(res.get("display_name", "이름 없음"))
        demo = res.get("demographics", {})
        st.write(f"**나이/성별**: {demo.get('age')}세 ({demo.get('sex')})")
        loc = res.get("location", {})
        st.write(f"**거주지**: {loc.get('province')} {loc.get('district')}")
        st.write(f"**직업**: {res.get('occupation')}")
        _render_education_line(demo)
        render_list_as_bullets(res.get("hobbies", []), "취미")
        render_list_as_bullets(res.get("skills", []), "스킬")
    with c2:
        st.subheader("성향 및 배경 (Persona)")
        summary = res.get("personas", {}).get("summary", "설명 없음")
        st.write(summary)

        persona_details = res.get("personas", {})
        if persona_details:
            for key, value in persona_details.items():
                if key == "summary":
                    continue
                if not value:
                    continue
                if isinstance(value, list):
                    render_list_as_bullets(value, persona_key_label(key))
                else:
                    st.markdown(f"**{persona_key_label(key)}**")
                    st.write(str(value))
        else:
            st.write("추가 설명이 없습니다.")

    st.divider()
    render_recommendation_section(target_uuid, key_suffix=f"detail_{target_uuid}")

    if st.button("프로필 탭에서 보기", key=f"open_profile_{target_uuid}"):
        set_selected_persona(target_uuid, str(res.get("display_name", "")))
        if show_close_button and close_target_key:
            st.session_state[close_target_key] = ""
        st.rerun()


def render_recommendation_section(target_uuid: str, key_suffix: str) -> None:
    st.subheader("🎯 페르소나 추천")
    st.caption("유사 페르소나가 자주 가진 속성 중, 현재 페르소나에게 없는 항목을 추천합니다.")
    category_label = st.radio(
        "추천 카테고리",
        list(RECOMMENDATION_CATEGORY_LABELS.values()),
        horizontal=True,
        key=f"recommend_category_{key_suffix}",
    )
    category = _category_value_from_label(category_label)
    top_n = st.slider("추천 개수", min_value=1, max_value=10, value=5, key=f"recommend_top_n_{key_suffix}")

    if st.button("추천 불러오기", key=f"load_recommend_{key_suffix}"):
        st.session_state[f"recommend_loaded_{key_suffix}"] = True

    if not st.session_state.get(f"recommend_loaded_{key_suffix}", False):
        st.info("추천을 보려면 [추천 불러오기]를 눌러주세요.")
        return

    with st.spinner("유사 페르소나 기반 추천을 계산 중입니다..."):
        try:
            result = get_json(api_path("/api/recommend", target_uuid), params={"category": category, "top_n": top_n})
        except Exception as e:
            st.warning(format_error_message(e))
            return

    recommendations = result.get("recommendations", [])
    if not recommendations:
        st.info("추천할 새 항목이 없습니다.")
        return

    columns = st.columns(2)
    for index, item in enumerate(recommendations):
        with columns[index % 2]:
            with st.container(border=True):
                st.markdown(f"#### {item.get('item_name', '이름 없음')}")
                st.write(item.get("reason", "추천 사유가 없습니다."))
                c1, c2 = st.columns(2)
                c1.metric("추천 점수", f"{float(item.get('reason_score', 0.0)):.0%}")
                c2.metric("유사 페르소나", f"{int(item.get('similar_users_count', 0)):,}명")
                supporters = item.get("supporting_personas", [])
                if supporters:
                    with st.expander("이 추천의 근거가 된 유사 페르소나"):
                        for supporter in supporters:
                            supporter_uuid = str(supporter.get("uuid", ""))
                            name = supporter.get("display_name") or short_uuid(supporter_uuid)
                            similarity = supporter.get("similarity")
                            st.write(f"- {name} · 유사도 {float(similarity or 0.0):.3f} · `{supporter_uuid}`")
                else:
                    st.caption("근거 유사 페르소나 상세 목록이 없습니다.")


def _category_value_from_label(label: str) -> str:
    for value, display in RECOMMENDATION_CATEGORY_LABELS.items():
        if display == label:
            return value
    return "hobby"

def render_list_as_bullets(values: object, title: str) -> None:
    items = list(values or []) if isinstance(values, list) else []
    st.markdown(f"**{title}:**")
    if not items:
        st.caption("- 정보 없음")
        return
    st.markdown("\n".join(f"- {item}" for item in items))


def _render_education_line(demo: dict[str, Any]) -> None:
    education = demo.get("education_level", "")
    field = demo.get("bachelors_field")
    if field and str(field).lower() not in ("none", ""):
        st.write(f"**학력**: {education} ({field})")
    else:
        st.write(f"**학력**: {education}")


def _open_similar_persona(df_similar: pd.DataFrame, row_index: int) -> None:
    if row_index < 0 or row_index >= len(df_similar):
        return

    row = df_similar.iloc[row_index]
    selected_uuid = str(row.get("uuid", "")).strip()
    if not selected_uuid:
        return

    set_selected_persona(selected_uuid, str(row.get("display_name", "")))
    st.rerun()


def _render_similar_persona_selector(df_similar: pd.DataFrame, target_uuid: str) -> None:
    header = st.columns([3, 2, 1, 1, 1])
    header[0].markdown("**UUID**")
    header[1].markdown("**이름**")
    header[2].markdown("**나이**")
    header[3].markdown("**유사도**")
    header[4].markdown("**이동**")

    visible_df = df_similar.head(10).reset_index(drop=True)
    for row_index, row in enumerate(visible_df.to_dict("records")):
        cols = st.columns([3, 2, 1, 1, 1], vertical_alignment="center")
        uuid = str(row.get("uuid", ""))
        name = str(row.get("display_name", "이름 없음"))
        age = row.get("age", "-")
        score = row.get("score", row.get("similarity", "-"))
        try:
            score_text = f"{float(score):.3f}"
        except (TypeError, ValueError):
            score_text = str(score)

        cols[0].code(uuid, language=None)
        cols[1].write(name)
        cols[2].write(age)
        cols[3].write(score_text)
        if cols[4].button("선택", key=f"select_similar_{target_uuid}_{row_index}"):
            _open_similar_persona(visible_df, row_index)


def _render_profile_details_for_uuid(target_uuid: str) -> None:
    try:
        res = get_json(api_path("/api/persona", target_uuid))
        c1, c2 = st.columns([1, 2])
        with c1:
            st.subheader(res.get("display_name", "이름 없음"))
            demo = res.get("demographics", {})
            st.write(f"**나이/성별**: {demo.get('age')}세 ({demo.get('sex')})")
            loc = res.get("location", {})
            st.write(f"**거주지**: {loc.get('province')} {loc.get('district')}")
            st.write(f"**직업**: {res.get('occupation')}")
            _render_education_line(demo)
            render_list_as_bullets(res.get("hobbies", []), "취미")
            render_list_as_bullets(res.get("skills", []), "스킬")
        with c2:
            st.subheader("성향 및 배경 (Persona)")
            summary = res.get("personas", {}).get("summary", "설명 없음")
            st.write(summary)

            persona_details = res.get("personas", {})
            if persona_details:
                for key, value in persona_details.items():
                    if key == "summary":
                        continue
                    if not value:
                        continue
                    if isinstance(value, list):
                        render_list_as_bullets(value, persona_key_label(key))
                    else:
                        st.markdown(f"**{persona_key_label(key)}**")
                        st.write(str(value))
            else:
                st.write("추가 설명이 없습니다.")
                
        st.divider()
        st.subheader("이 사람과 비슷한 페르소나 (KNN Graph)")
        st.caption("오른쪽 [선택] 버튼을 누르면 해당 페르소나의 프로필로 즉시 이동합니다.")
        similar = res.get("similar_preview", [])
        if similar:
            _render_similar_persona_selector(pd.DataFrame(similar), target_uuid)
        else:
            st.write("유사 인물 정보가 없습니다.")

        st.divider()
        render_recommendation_section(target_uuid, key_suffix=f"profile_{target_uuid}")
            
        st.divider()
        st.subheader("🕸️ 연관 네트워크 그래프")
        st.caption("이 페르소나와 취미/스킬/지역을 공유하는 사람들을 보여줍니다.")
        profile_graph_depth = st.radio(
            "탐색 깊이",
            [1, 2, 3],
            index=1,
            horizontal=True,
            key=f"profile_graph_depth_{target_uuid}",
            help="1: 직접 연결, 2: 같은 객체를 공유하는 다른 사람, 3: 그 사람의 추가 연결까지 확장",
        )
        try:
            graph_res = get_json(
                api_path("/api/graph/subgraph", target_uuid),
                params={"depth": profile_graph_depth, "include_similar": False, "max_nodes": 60},
            )
            render_graph_explorer(res, graph_res, target_uuid, show_summary=False, key_suffix="profile")
        except Exception as e:
            st.warning("그래프 데이터를 불러오는 중 문제가 발생했습니다.")
    except Exception as e:
        st.error(format_error_message(e))

def render_profile_tab() -> None:
    st.header("👤 페르소나 프로필 뷰")
    uuid = st.session_state.selected_uuid
    pending_profile_uuid = st.session_state.pop("_pending_profile_navigation_uuid", None)
    with st.expander("고급: UUID 직접 변경", expanded=False):
        uuid = st.text_input("페르소나 UUID", value=st.session_state.selected_uuid, key="profile_uuid")

    target_uuid = ""
    auto_load_profile = False
    if isinstance(pending_profile_uuid, str):
        pending_profile_uuid = pending_profile_uuid.strip()
    if pending_profile_uuid:
        target_uuid = pending_profile_uuid
        auto_load_profile = True

    if st.button("프로필 조회") and uuid:
        if not validate_uuid_input(uuid):
            return
        set_selected_persona(uuid)
        target_uuid = uuid
        auto_load_profile = True

    if not auto_load_profile and not target_uuid:
        current_uuid = st.session_state.selected_uuid
        if current_uuid and UUID_PATTERN.fullmatch(current_uuid.strip()):
            target_uuid = current_uuid.strip()
            auto_load_profile = True

    if auto_load_profile:
        if not target_uuid:
            return
        normalized_target = target_uuid.strip()
        if not normalized_target:
            return
        with st.spinner("프로필 구성 중..."):
            if validate_uuid_input(normalized_target):
                _render_profile_details_for_uuid(normalized_target)


def render_compare_tab() -> None:
    st.header("⚖️ 그룹 세그먼트 비교 분석")
    st.write("두 집단의 차이를 통계적으로 비교하고, AI 모델이 한국어로 특징을 분석합니다.")
    
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 그룹 A 설정")
            a_label = st.text_input("그룹 A 이름", "서울 30대")
            a_prov = st.text_input("그룹 A 지역", "서울")
            a_age = st.text_input("그룹 A 연령대", "30대")
        with c2:
            st.markdown("#### 그룹 B 설정")
            b_label = st.text_input("그룹 B 이름", "부산 30대")
            b_prov = st.text_input("그룹 B 지역", "부산")
            b_age = st.text_input("그룹 B 연령대", "30대")
        
    if st.button("비교 분석 실행", type="primary"):
        payload = {
            "segment_a": {"label": a_label, "filters": {"province": a_prov if a_prov else None, "age_group": a_age if a_age else None}},
            "segment_b": {"label": b_label, "filters": {"province": b_prov if b_prov else None, "age_group": b_age if b_age else None}},
            "dimensions": ["hobby", "occupation"]
        }
        with st.spinner("통계 집계 및 AI 텍스트 분석 중... (최대 10~20초 소요)"):
            try:
                res = post_json("/api/compare/segments", payload)
                
                st.markdown("### 🤖 AI 차이점 분석")
                st.success(res.get("ai_analysis", "분석 결과가 없습니다."))
                
                st.markdown("### 📊 취미(Hobby) 비교 데이터")
                comps = res.get("comparisons", {}).get("hobby", {})
                st.write(f"**공통 관심사**: {', '.join(comps.get('common', []))}")
                st.write(f"**{a_label}만의 관심사**: {', '.join(comps.get('only_a', []))}")
                st.write(f"**{b_label}만의 관심사**: {', '.join(comps.get('only_b', []))}")
            except Exception as e:
                st.error(format_error_message(e))

def render_graph_tab() -> None:
    st.header("🕸️ 객체 관계 그래프 탐색")
    st.caption("Neo4j 관계를 사람-공통객체-사람 네트워크로 보여줍니다. 노드는 드래그/확대/축소할 수 있습니다.")
    render_relationship_mapping_primer()
    with st.expander("고급: 중심 페르소나 직접 지정", expanded=False):
        uuid = st.text_input("중심 페르소나 UUID", value=st.session_state.selected_uuid, key="graph_uuid")
        st.caption("처음 사용하는 경우 UUID를 직접 입력하지 말고 [검색/필터]에서 행을 클릭해 선택하세요.")
    c1, c2, c3 = st.columns(3)
    depth = c1.radio(
        "탐색 깊이",
        [1, 2, 3],
        index=1,
        horizontal=True,
        help="1: 직접 연결, 2: 같은 객체를 공유하는 다른 사람, 3: 그 사람의 추가 연결까지 확장",
    )
    max_nodes = c2.slider("최대 노드 수", min_value=10, max_value=120, value=60, step=10)
    include_similar = c3.checkbox("유사도 관계 포함", value=False, help="GDS KNN으로 계산된 수학적 유사도 관계입니다. 먼저 공통 취미/스킬/지역 관계를 본 뒤 켜는 것을 권장합니다.")

    if st.session_state.graph_auto_loaded_uuid != st.session_state.selected_uuid:
        load_graph_data(st.session_state.selected_uuid, depth, max_nodes, include_similar)
        st.session_state.graph_auto_loaded_uuid = st.session_state.selected_uuid
    
    if st.button("그래프 그리기") and uuid:
        if not validate_uuid_input(uuid):
            return
        set_selected_persona(uuid)
        load_graph_data(uuid, depth, max_nodes, include_similar)
        st.session_state.graph_auto_loaded_uuid = uuid

    if "graph_data" in st.session_state and "graph_profile" in st.session_state:
        render_graph_explorer(
            st.session_state.graph_profile,
            st.session_state.graph_data,
            st.session_state.get("graph_center_uuid", st.session_state.selected_uuid),
            show_summary=True,
            key_suffix="main_tab",
        )

def load_graph_data(uuid: str, depth: int, max_nodes: int, include_similar: bool) -> None:
    with st.spinner("관계 그래프를 구성 중..."):
        try:
            profile = get_json(api_path("/api/persona", uuid))
            res = get_json(
                api_path("/api/graph/subgraph", uuid),
                params={"depth": depth, "include_similar": include_similar, "max_nodes": max_nodes},
            )
            st.session_state.graph_profile = profile
            st.session_state.graph_data = res
            st.session_state.graph_center_uuid = uuid
            if profile.get("display_name"):
                st.session_state.selected_persona_label = profile.get("display_name")
        except Exception as e:
            st.error(format_error_message(e))


def render_influence_tab() -> None:
    st.header("🌐 네트워크 핵심 인물 분석")
    st.caption("사전 계산된 PageRank, Betweenness, Degree 점수로 커뮤니티 안팎의 핵심 페르소나를 확인합니다.")
    metric_label = st.radio(
        "중심성 지표",
        list(CENTRALITY_METRIC_LABELS.values()),
        horizontal=True,
        key="influence_metric_label",
    )
    metric = _centrality_metric_from_label(metric_label)
    c1, c2 = st.columns([1, 1])
    limit = c1.slider("조회 인원", min_value=5, max_value=100, value=20, step=5, key="influence_limit")
    community_input = str(c2.text_input("커뮤니티 ID (선택)", value=st.session_state.get("influence_community_id", "")) or "")
    st.session_state.influence_community_id = community_input

    if st.button("핵심 인물 조회", type="primary", key="load_influence"):
        st.session_state.influence_loaded = True

    if not st.session_state.get("influence_loaded", False):
        st.info("핵심 인물을 보려면 [핵심 인물 조회]를 눌러주세요.")
        return

    params: dict[str, Any] = {"metric": metric, "limit": limit}
    if community_input.strip():
        try:
            params["community_id"] = int(community_input.strip())
        except ValueError:
            st.error("커뮤니티 ID는 숫자로 입력해주세요.")
            return

    with st.spinner("중심성 점수를 불러오는 중입니다..."):
        try:
            result = get_json("/api/influence/top", params=params)
        except Exception as e:
            st.error(format_error_message(e))
            return

    last_updated_at = result.get("last_updated_at")
    if last_updated_at:
        st.success(f"마지막 갱신: {last_updated_at}")
    else:
        st.warning("마지막 갱신 시각이 없습니다. 중심성 배치 상태를 확인해주세요.")

    rows = result.get("results", [])
    if not rows:
        st.info("표시할 핵심 인물 결과가 없습니다.")
        return

    influence_df = pd.DataFrame(rows)
    display_df = influence_df.rename(
        columns={
            "rank": "순위",
            "uuid": "UUID",
            "display_name": "이름",
            "score": "점수",
            "community_id": "커뮤니티",
        }
    )
    st.subheader("상위 핵심 인물")
    st.dataframe(display_df[["순위", "이름", "UUID", "점수", "커뮤니티"]], width="stretch", hide_index=True)

    chart_df = influence_df.head(10).copy()
    chart_df["label"] = chart_df.apply(
        lambda row: row.get("display_name") or short_uuid(str(row.get("uuid", ""))), axis=1
    )
    st.subheader("Top 10 점수 분포")
    st.bar_chart(chart_df.set_index("label")["score"])

    st.subheader("노드 제거 시뮬레이션")
    st.caption("최대 5명을 선택해 해당 페르소나 제거 시 서브그래프 연결성이 얼마나 낮아지는지 확인합니다.")
    options = {
        f"#{row.get('rank')} {row.get('display_name') or short_uuid(str(row.get('uuid', '')))} ({short_uuid(str(row.get('uuid', '')))})": str(row.get("uuid"))
        for row in rows
    }
    selected_labels = st.multiselect(
        "시뮬레이션 대상",
        list(options.keys()),
        max_selections=5,
        key="influence_simulation_labels",
    )
    sim_col1, sim_col2 = st.columns([1, 1])
    if sim_col1.button("제거 영향 계산", key="simulate_removal"):
        selected_uuids = [options[label] for label in selected_labels]
        if not selected_uuids:
            st.warning("선택된 노드가 없습니다.")
            return
        with st.spinner("서브그래프 영향을 계산 중입니다..."):
            try:
                sim = post_json("/api/influence/simulate-removal", {"target_uuids": selected_uuids, "max_depth": 3})
                st.session_state.influence_simulation_result = sim
            except Exception as e:
                st.error(format_error_message(e))
                return
    if sim_col2.button("선택/결과 초기화", key="clear_simulation"):
        st.session_state.influence_simulation_labels = []
        st.session_state.influence_simulation_result = None
        st.rerun()

    sim_result = st.session_state.get("influence_simulation_result")
    if sim_result:
        m1, m2, m3 = st.columns(3)
        m1.metric("기존 연결성", f"{float(sim_result.get('original_connectivity', 0.0)):.2f}")
        m2.metric("제거 후 연결성", f"{float(sim_result.get('current_connectivity', 0.0)):.2f}")
        m3.metric("분절 증가", f"{float(sim_result.get('fragmentation_increase', 0.0)):.2f}")
        communities = sim_result.get("affected_communities", [])
        st.caption(f"영향 커뮤니티: {', '.join(map(str, communities)) if communities else '없음'}")


def _centrality_metric_from_label(label: str) -> str:
    for value, display in CENTRALITY_METRIC_LABELS.items():
        if display == label:
            return value
    return "pagerank"


def render_chat_tab() -> None:
    st.markdown(
        """
        <style>
        .chat-page-title {
            font-size: 1.55rem;
            font-weight: 800;
            color: var(--text-color);
            margin-bottom: 0.15rem;
        }
        .chat-page-subtitle {
            color: var(--text-color);
            opacity: 0.65;
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="chat-page-title">대화형 탐색</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="chat-page-subtitle">질문하면 그래프 검색/통계로 바로 답변합니다.</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    messages = st.session_state.get("chat_messages", [])
    if not messages:
        with st.chat_message("assistant"):
            st.markdown("궁금한 조건을 자연스럽게 입력해 주세요. 제가 필터와 통계를 정리해서 답변합니다.")

    for message in messages[-10:]:
        role = str(message.get("role", "assistant"))
        content = str(message.get("content", ""))
        with st.chat_message(role):
            _render_chat_bubble(
                role,
                content,
                sources=message.get("sources") if isinstance(message.get("sources"), list) else None,
                filters=message.get("filters") if isinstance(message.get("filters"), dict) else None,
            )

    prompt = st.chat_input("질문을 입력하세요")
    if prompt:
        user_message = prompt.strip()
        if user_message:
            st.session_state.chat_messages.append({"role": "user", "content": user_message})
            with st.chat_message("user"):
                st.markdown(user_message)
            with st.chat_message("assistant"):
                assistant_message = _fetch_chat_response(user_message)
                st.markdown(assistant_message["content"])
                if assistant_message.get("filters") or assistant_message.get("sources"):
                    _render_chat_bubble(
                        "assistant",
                        "",
                        sources=assistant_message.get("sources") if isinstance(assistant_message.get("sources"), list) else None,
                        filters=assistant_message.get("filters") if isinstance(assistant_message.get("filters"), dict) else None,
                        show_content=False,
                    )
            st.session_state.chat_messages.append(assistant_message)
            st.session_state.chat_messages = st.session_state.chat_messages[-10:]


def _render_chat_bubble(
    role: str,
    content: str,
    *,
    sources: list[dict[str, Any]] | None = None,
    filters: dict[str, Any] | None = None,
    show_content: bool = True,
) -> None:
    is_user = role == "user"
    if show_content:
        st.markdown(content)
    if not is_user and (sources or filters):
        with st.expander("검색 근거", expanded=False):
            if filters:
                st.write("적용 필터")
                st.json(filters)
            if sources:
                st.write("참조 소스")
                st.json(sources)


def _send_chat_message(message: str) -> None:
    user_message = message.strip()
    if not user_message:
        return

    st.session_state.chat_messages.append({"role": "user", "content": user_message})
    assistant_message = _fetch_chat_response(user_message)
    st.session_state.chat_messages.append(assistant_message)
    st.session_state.chat_messages = st.session_state.chat_messages[-10:]


def _fetch_chat_response(user_message: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "session_id": st.session_state.chat_session_id,
        "message": user_message,
        "stream": False,
    }
    try:
        result = post_json("/api/chat", payload)
    except Exception as e:
        return {"role": "assistant", "content": format_error_message(e)}

    response_text = str(result.get("response", "응답이 없습니다."))
    st.session_state.chat_context_filters = result.get("context_filters", {}) or {}
    st.session_state.chat_sources = result.get("sources", []) or []
    return {
        "role": "assistant",
        "content": response_text,
        "filters": st.session_state.chat_context_filters,
        "sources": st.session_state.chat_sources,
    }

def render_insight_tab() -> None:
    st.header("💡 자연어 인사이트 질의")
    st.caption("예시: \"서울 20대 여성들이 가장 많이 하는 취미는?\", \"부산 40대 남성의 주요 직업은?\", \"강남구 거주자들의 공통 관심사는?\"")

    messages: list[dict[str, Any]] = st.session_state.get("insight_messages", [])
    if not messages:
        with st.chat_message("assistant"):
            st.markdown("궁금한 내용을 자연어로 질문하면, 지식 그래프와 AI가 분석하여 한국어로 답변합니다.")

    for msg in messages[-10:]:
        role = str(msg.get("role", "assistant"))
        content = str(msg.get("content", ""))
        with st.chat_message(role):
            st.markdown(content)
            sources = msg.get("sources")
            if role == "assistant" and sources:
                with st.expander("참조 소스", expanded=False):
                    st.json(sources)

    prompt = st.chat_input("질문을 입력하세요", key="insight_chat_input")
    if prompt:
        user_message = prompt.strip()
        if user_message:
            st.session_state.insight_messages.append({"role": "user", "content": user_message})
            with st.chat_message("user"):
                st.markdown(user_message)
            with st.chat_message("assistant"):
                try:
                    result = post_json("/api/insight", {"question": user_message})
                    answer = str(result.get("answer", "응답이 없습니다."))
                    sources = result.get("sources", [])
                    st.markdown(answer)
                    if sources:
                        with st.expander("참조 소스", expanded=False):
                            st.json(sources)
                    st.session_state.insight_messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })
                except Exception as e:
                    error_msg = format_error_message(e)
                    st.error(error_msg)
                    st.session_state.insight_messages.append({
                        "role": "assistant",
                        "content": f"⚠️ {error_msg}",
                    })
            st.session_state.insight_messages = st.session_state.insight_messages[-10:]

def render_similar_tab() -> None:
    st.header("🤝 유사 페르소나 매칭")
    st.write("현재 선택된 페르소나와 라이프스타일이 비슷한 사람들을 AI가 찾아줍니다.")
    st.caption("GDS KNN 알고리즘 기반으로 취미·스킬·지역·직업 등을 종합 비교합니다.")
    uuid = str(st.session_state.selected_uuid or "")
    with st.expander("고급: 기준 UUID 직접 변경", expanded=False):
        uuid = str(st.text_input("기준 페르소나 UUID", value=st.session_state.selected_uuid, key="similar_uuid") or "")
    top_k = st.slider("Top K", min_value=1, max_value=20, value=5)
    if st.button("비슷한 사람 찾기", key="similar"):
        if not validate_uuid_input(uuid):
            return
        set_selected_persona(uuid)
        try:
            result = post_json(api_path("/api/similar", uuid), {"top_k": top_k})
            st.subheader("유사 페르소나")
            similar_people = result.get("similar_personas", [])
            if similar_people:
                first_uuid = similar_people[0].get("uuid")
                if first_uuid:
                    st.session_state.last_similar_uuid = str(first_uuid)
            for index, persona in enumerate(similar_people):
                persona_uuid = str(persona.get("uuid", ""))
                with st.container(border=True):
                    left, right = st.columns([4, 1], vertical_alignment="center")
                    with left:
                        st.write(f"{persona.get('display_name')} · ID {short_uuid(persona_uuid)}")
                    with right:
                        if st.button("상세 보기", key=f"view_similar_{persona_uuid}_{index}", use_container_width=True):
                            _render_search_detail_dialog(persona_uuid)
                    render_collapsible_text(
                        persona.get("persona", ""),
                        max_chars=220,
                        max_lines=3,
                        expander_label="성향 전문 보기",
                    )
                    st.metric("similarity", persona.get("similarity"))
        except Exception as e:
            st.error(format_error_message(e))

def render_communities_tab() -> None:
    st.header("🏘️ 라이프스타일 커뮤니티 탐지")
    st.write("Neo4j 그래프 알고리즘(Leiden)이 사람들의 공통 취미와 스킬을 분석해 **숨겨진 라이프스타일 그룹**을 자동으로 찾아냅니다.")
    
    min_size = st.number_input("최소 커뮤니티 크기 (몇 명 이상 모인 그룹을 볼까요?)", min_value=1, value=10)
    if st.button("커뮤니티 조회", key="communities"):
        with st.spinner("커뮤니티 그룹을 분석 중입니다..."):
            try:
                result = get_json("/api/communities", params={"min_size": min_size})
                communities = result.get("communities", [])
                
                if not communities:
                    st.warning("조건에 맞는 커뮤니티가 없습니다.")
                    return
                    
                st.success(f"총 {len(communities)}개의 주요 커뮤니티 그룹이 발견되었습니다.")
                
                for idx, comm in enumerate(communities):
                    with st.container(border=True):
                        c1, c2 = st.columns([1, 3])
                        with c1:
                            st.metric(f"그룹 #{idx+1}", f"{comm.get('size', 0):,}명")
                            rep_uuid = comm.get("representative_persona_uuid")
                            if rep_uuid:
                                st.caption(f"대표: {short_uuid(str(rep_uuid))}")
                        with c2:
                            st.subheader(comm.get("label", "이름 없는 그룹"))
                            traits = comm.get("top_traits", {})
                            if isinstance(traits, dict):
                                has_traits = False
                                for key, val in traits.items():
                                    if isinstance(val, list):
                                        st.markdown(f"**{persona_key_label(key)}**")
                                        if val:
                                            for item in val:
                                                st.markdown(f"- {item}")
                                        else:
                                            st.caption("- 없음")
                                        has_traits = True
                                    else:
                                        st.markdown(f"**{persona_key_label(key)}**")
                                        st.markdown(f"- {val}")
                                        has_traits = True
                                if not has_traits:
                                    st.caption("주요 특징 요약 없음")
                            else:
                                st.write("주요 특징 요약 없음")
            except Exception as e:
                st.error(format_error_message(e))

def render_path_tab() -> None:
    st.header("🗺️ 두 사람 사이의 관계 경로 탐색")
    st.write("두 페르소나가 어떤 취미·스킬·지역·직업을 통해 연결되는지 최단 경로를 찾아줍니다.")
    st.caption("팁: [유사 페르소나] 탭에서 먼저 비슷한 사람을 찾으면 두 번째 UUID가 자동으로 채워집니다.")
    uuid1 = str(st.session_state.selected_uuid or "")
    uuid2 = str(st.session_state.get("last_similar_uuid", "") or "")
    with st.expander("고급: 경로 분석 UUID 직접 변경", expanded=False):
        uuid1 = str(st.text_input("UUID 1", value=st.session_state.selected_uuid, key="path_uuid1") or "")
        uuid2 = str(st.text_input("UUID 2", value=st.session_state.get("last_similar_uuid", ""), key="path_uuid2") or "")
        st.caption("팁: [유사 페르소나]를 먼저 조회하면 두 번째 UUID 예시가 자동으로 채워집니다.")
    max_depth = st.slider("최대 깊이", min_value=1, max_value=6, value=4)
    if not uuid2:
        st.warning("두 번째 페르소나가 아직 선택되지 않았습니다. [유사 페르소나] 탭에서 먼저 비슷한 사람을 찾아보세요.")
    if st.button("경로 찾기", key="path"):
        if not validate_uuid_input(uuid1, "UUID 1") or not validate_uuid_input(uuid2, "UUID 2"):
            return
        try:
            result = get_json(api_path("/api/path", uuid1, uuid2), params={"max_depth": max_depth})
            st.write(result.get("summary", ""))
            st.json(result)
        except Exception as e:
            st.error(format_error_message(e))

def post_json(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=60)
    response.raise_for_status()
    return response.json()

def get_json(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    response = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=60)
    response.raise_for_status()
    return response.json()

@st.cache_data(ttl=600)
def get_stats_cached() -> dict[str, Any]:
    return get_json("/api/stats")

def api_path(prefix: str, *parts: str) -> str:
    encoded_parts = [quote(part.strip(), safe="") for part in parts]
    return "/".join([prefix.rstrip("/"), *encoded_parts])

def validate_uuid_input(uuid: str, label: str = "UUID") -> bool:
    if not UUID_PATTERN.fullmatch(uuid.strip()):
        st.error(f"{label}는 32자리 hexadecimal UUID여야 합니다. 검색 결과에서 행을 선택해 자동 입력하는 방식을 권장합니다.")
        return False
    return True

def validate_api_base_url(base_url: str) -> None:
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("API_BASE_URL은 http(s) URL이어야 합니다.")

def format_error_message(error: Exception) -> str:
    if isinstance(error, requests.exceptions.HTTPError):
        status = error.response.status_code if error.response is not None else "unknown"
        return f"요청을 처리하지 못했습니다. 입력값과 서버 상태를 확인해주세요. (HTTP {status})"
    if isinstance(error, requests.exceptions.RequestException):
        return "API 서버에 연결하지 못했습니다. FastAPI 서버가 실행 중인지 확인해주세요."
    return "요청 처리 중 오류가 발생했습니다. 입력값을 확인한 뒤 다시 시도해주세요."

def escape_dot_value(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

def init_session_state() -> None:
    if "selected_uuid" not in st.session_state:
        st.session_state.selected_uuid = DEFAULT_PERSONA_UUID
    if "selected_persona_label" not in st.session_state:
        st.session_state.selected_persona_label = ""
    if "graph_uuid" not in st.session_state:
        st.session_state.graph_uuid = st.session_state.selected_uuid
    if "profile_uuid" not in st.session_state:
        st.session_state.profile_uuid = st.session_state.selected_uuid
    if "graph_auto_loaded_uuid" not in st.session_state:
        st.session_state.graph_auto_loaded_uuid = ""
    if "last_similar_uuid" not in st.session_state:
        st.session_state.last_similar_uuid = ""
    if "similar_uuid" not in st.session_state:
        st.session_state.similar_uuid = st.session_state.selected_uuid
    if "path_uuid1" not in st.session_state:
        st.session_state.path_uuid1 = st.session_state.selected_uuid
    if "search_detail_uuid" not in st.session_state:
        st.session_state.search_detail_uuid = ""
    if "search_province" not in st.session_state:
        st.session_state.search_province = "서울"
    if "search_age_group" not in st.session_state:
        st.session_state.search_age_group = "20대"
    if "search_sex" not in st.session_state:
        st.session_state.search_sex = "여자"
    if "influence_loaded" not in st.session_state:
        st.session_state.influence_loaded = False
    if "influence_community_id" not in st.session_state:
        st.session_state.influence_community_id = ""
    if "influence_simulation_result" not in st.session_state:
        st.session_state.influence_simulation_result = None
    if "chat_session_id" not in st.session_state:
        st.session_state.chat_session_id = str(uuid.uuid4())
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "chat_context_filters" not in st.session_state:
        st.session_state.chat_context_filters = {}
    if "chat_sources" not in st.session_state:
        st.session_state.chat_sources = []
    if "insight_messages" not in st.session_state:
        st.session_state.insight_messages = []


def _apply_pending_uuid_sync() -> None:
    pending_uuid = st.session_state.get("_pending_selected_uuid")
    if pending_uuid is None:
        return

    st.session_state.graph_uuid = pending_uuid
    st.session_state.profile_uuid = pending_uuid
    st.session_state.similar_uuid = pending_uuid
    st.session_state.path_uuid1 = pending_uuid
    del st.session_state["_pending_selected_uuid"]

def set_selected_persona(uuid: str, label: str | None = None) -> None:
    normalized_uuid = uuid.strip()
    if not normalized_uuid:
        return

    st.session_state.selected_uuid = normalized_uuid
    st.session_state._pending_selected_uuid = normalized_uuid
    st.session_state._pending_profile_navigation_uuid = normalized_uuid
    if label:
        st.session_state.selected_persona_label = label

def short_uuid(uuid: str) -> str:
    return f"{uuid[:8]}…{uuid[-6:]}" if len(uuid) > 16 else uuid

def persona_key_label(key: str) -> str:
    labels = {
        "summary": "요약",
        "professional": "직업/업무",
        "sports": "운동/건강",
        "arts": "문화/예술",
        "travel": "여행/이동",
        "culinary": "식문화",
        "family": "가족/관계",
        "hobbies": "주요 취미",
        "skills": "주요 스킬",
        "province": "주요 거주지",
        "occupation": "주요 직업",
    }
    return labels.get(key, key)

def render_graph_profile_summary(profile: dict[str, Any], graph: dict[str, Any]) -> None:
    demo = profile.get("demographics", {})
    location = profile.get("location", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("중심 페르소나", profile.get("display_name", "이름 없음"))
    c2.metric("나이/성별", f"{demo.get('age', '-')}/{demo.get('sex', '-')}")
    c3.metric("연결 노드", graph.get("node_count", 0))
    c4.metric("관계 수", graph.get("edge_count", 0))
    st.caption(f"거주지: {location.get('province', '-')} {location.get('district', '')} · 직업: {profile.get('occupation', '-')}")

def render_relationship_mapping_primer() -> None:
    with st.expander("이 화면은 무엇을 보여주나요?", expanded=False):
        st.write(
            "커머스에서 `고객 → 구매 → 상품`, 소셜에서 `사용자 → 좋아요 → 게시물`을 추적하듯이, "
            "여기서는 `페르소나 → 공통 객체 → 다른 페르소나` 연결을 추적합니다."
        )
        c1, c2, c3 = st.columns(3)
        c1.info("**취미 기반**\n\n이하윤 → 취미 → 보드게임 모임\n\n다른 사람도 같은 취미로 연결")
        c2.info("**스킬 기반**\n\n이하윤 → 스킬 → 일정 조율\n\n같은 스킬을 가진 페르소나 발견")
        c3.info("**지역 기반**\n\n이하윤 → 지역 → 강서구\n\n같은 지역 페르소나 네트워크 확인")
        st.caption("읽는 법: 선을 따라가면 ‘왜 두 객체가 연결됐는지’가 보입니다. depth=2는 같은 취미·스킬·지역을 공유하는 다른 페르소나까지 보여줍니다.")

def render_graph_explorer(profile: dict[str, Any], graph: dict[str, Any], center_uuid: str, show_summary: bool = True, key_suffix: str = "main") -> None:
    if show_summary:
        render_graph_profile_summary(profile, graph)
        render_graph_reading_guide()

    available_types = sorted({str(node.get("type", "Unknown")) for node in graph.get("nodes", [])})
    default_types = [node_type for node_type in available_types if node_type in {"Person", "Hobby", "Skill", "Occupation"}]
    if not default_types:
        default_types = available_types

    st.subheader("객체 유형 필터")
    selected_types = st.multiselect(
        "표시할 객체 유형",
        options=available_types,
        default=default_types,
        format_func=node_type_label,
        key=f"graph_node_type_filter_{key_suffix}",
    )
    st.caption("처음에는 취미·스킬·직업 중심으로 보여줍니다. 지역/학력까지 보고 싶으면 필터에서 추가하세요.")
    st.caption("탐색 깊이는 그래프를 다시 조회하는 옵션입니다. Depth 3은 관계가 많아질 수 있으니 필요한 경우에만 사용하세요.")

    filtered_graph = filter_graph_by_types(graph, set(selected_types), center_uuid)
    render_graph_legend(filtered_graph)
    render_relationship_summary(filtered_graph)
    render_commonality_cards(filtered_graph)

    st.subheader("인터랙티브 관계 그래프")
    st.caption("마우스 휠로 확대/축소하고, 노드를 드래그해 배치를 조정할 수 있습니다.")
    components.html(build_interactive_graph_html(filtered_graph, center_uuid), height=720, scrolling=False)
    render_relationship_table(filtered_graph, center_uuid)
    render_relationship_sentences(filtered_graph, center_uuid)

def render_graph_reading_guide() -> None:
    with st.expander("그래프 읽는 법", expanded=False):
        st.markdown(
            "- **주황색 별**: 현재 선택한 중심 페르소나\n"
            "- **파란 원**: 다른 페르소나\n"
            "- **초록/노랑/빨강 객체**: 취미, 스킬, 지역 같은 연결 이유\n"
            "- **사람 → 객체 → 사람**: 두 사람이 같은 객체를 공유한다는 뜻\n"
            "- **유사도 관계**: 공통 객체가 아니라 GDS/KNN 계산으로 비슷하다고 판단된 관계"
        )

def filter_graph_by_types(graph: dict[str, Any], selected_types: set[str], center_uuid: str) -> dict[str, Any]:
    center_id = f"person_{center_uuid}"
    kept_nodes = [
        node for node in graph.get("nodes", [])
        if node.get("id") == center_id or node.get("type") in selected_types
    ]
    kept_ids = {node.get("id") for node in kept_nodes}
    kept_edges = dedupe_edges([
        edge for edge in graph.get("edges", [])
        if edge.get("source") in kept_ids and edge.get("target") in kept_ids
    ])
    return {
        **graph,
        "nodes": kept_nodes,
        "edges": kept_edges,
        "node_count": len(kept_nodes),
        "edge_count": len(kept_edges),
    }

def render_graph_legend(graph: dict[str, Any]) -> None:
    used_types = sorted({str(node.get("type", "Unknown")) for node in graph.get("nodes", [])})
    cols = st.columns(min(max(len(used_types), 1), 4))
    for idx, node_type in enumerate(used_types):
        style = node_style(node_type)
        cols[idx % len(cols)].markdown(
            f"<span style='display:inline-block;width:12px;height:12px;border-radius:50%;background:{style['color']};margin-right:6px;'></span>"
            f"{node_type_label(node_type)}",
            unsafe_allow_html=True,
        )

def build_interactive_graph_html(graph: dict[str, Any], center_uuid: str) -> str:
    center_id = f"person_{center_uuid}"
    nodes = []
    for node in graph.get("nodes", []):
        node_type = str(node.get("type", "Unknown"))
        style = node_style(node_type)
        is_center = node.get("id") == center_id
        label = str(node.get("label", ""))
        font_color = "#ffffff" if is_center else "#f8fafc"
        nodes.append({
            "id": node.get("id"),
            "label": label,
            "group": node_type_label(node_type),
            "color": {"background": "#FF6B35" if is_center else style["color"], "border": "#ffffff" if is_center else "#222222"},
            "shape": "star" if is_center else style["shape"],
            "size": 40 if is_center else style["size"] + 6,
            "font": {"size": 22 if is_center else 16, "face": "Malgun Gothic, Apple SD Gothic Neo, sans-serif", "color": font_color, "strokeWidth": 3, "strokeColor": "#0f172a"},
            "borderWidth": 4 if is_center else 1,
        })

    edges = []
    for edge in graph.get("edges", []):
        rel = relation_label(str(edge.get("type", "")))
        edges.append({
            "from": edge.get("source"),
            "to": edge.get("target"),
            "label": rel,
            "arrows": "to",
            "color": {"color": relation_color(str(edge.get("type", "")))},
            "font": {"align": "middle", "size": 13, "face": "Malgun Gothic, Apple SD Gothic Neo, sans-serif", "color": "#e2e8f0", "strokeWidth": 4, "strokeColor": "#0f172a"},
            "smooth": {"type": "dynamic"},
            "width": 2,
        })

    return f"""
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
  <style>
    body {{ margin: 0; font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif; background: #0f172a; }}
    #graph {{ width: 100%; height: 650px; border: 1px solid #334155; border-radius: 16px; background: radial-gradient(circle at top left, #1e293b 0, #020617 58%); }}
    .hint {{ color: #94a3b8; }}
  </style>
</head>
<body>
  <div id="graph"></div>
  <script>
    const nodes = new vis.DataSet({json.dumps(nodes, ensure_ascii=False)});
    const edges = new vis.DataSet({json.dumps(edges, ensure_ascii=False)});
    const container = document.getElementById('graph');
    const options = {{
      interaction: {{ hover: false, navigationButtons: true, keyboard: true, multiselect: true }},
      physics: {{
        enabled: true,
        solver: 'forceAtlas2Based',
        forceAtlas2Based: {{ gravitationalConstant: -90, centralGravity: 0.015, springLength: 220, springConstant: 0.05 }},
        stabilization: {{ iterations: 150 }}
      }},
      nodes: {{ shadow: true }},
      edges: {{ shadow: false, selectionWidth: 4 }}
    }};
    const network = new vis.Network(container, {{ nodes, edges }}, options);
  </script>
</body>
</html>
"""

def node_style(node_type: str) -> dict[str, Any]:
    return NODE_STYLES.get(node_type, {"color": "#CBD5E1", "shape": "ellipse", "size": 16})

def relation_color(relation_type: str) -> str:
    if relation_type == "SIMILAR_TO":
        return "#FF6B35"
    if relation_type in {"ENJOYS_HOBBY", "HAS_SKILL"}:
        return "#22C55E"
    if relation_type in {"LIVES_IN", "IN_PROVINCE"}:
        return "#EF4444"
    return "#94A3B8"

def render_relationship_summary(graph: dict[str, Any]) -> None:
    edges = graph.get("edges", [])
    if not edges:
        st.warning("표시할 관계가 없습니다.")
        return

    counts: dict[str, int] = {}
    for edge in edges:
        label = relation_label(edge.get("type", ""))
        counts[label] = counts.get(label, 0) + 1
    summary = pd.DataFrame(
        [{"관계": label, "개수": count} for label, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)]
    )

    st.subheader("관계 요약")
    st.dataframe(summary, width="stretch", hide_index=True)

def render_relationship_table(graph: dict[str, Any], center_uuid: str) -> None:
    nodes_by_id = {node.get("id"): node for node in graph.get("nodes", [])}
    rows = []
    center_marker = center_uuid.replace("-", "")

    for edge in dedupe_edges(graph.get("edges", [])):
        source = nodes_by_id.get(edge.get("source"), {})
        target = nodes_by_id.get(edge.get("target"), {})
        source_label = source.get("label", edge.get("source", ""))
        target_label = target.get("label", edge.get("target", ""))
        source_type = node_type_label(source.get("type", ""))
        target_type = node_type_label(target.get("type", ""))
        source_id = str(edge.get("source", ""))
        target_id = str(edge.get("target", ""))
        rel_type = str(edge.get("type", ""))

        if center_marker in source_id:
            subject = "중심 페르소나"
            target_display = target_label
            target_display_type = target_type
        elif center_marker in target_id:
            subject = source_label
            target_display = "중심 페르소나"
            target_display_type = source_type
        elif target.get("type") == "Person" and source.get("type") != "Person":
            subject = target_label
            target_display = source_label
            target_display_type = source_type
        else:
            subject = source_label
            target_display = target_label
            target_display_type = target_type

        rows.append({
            "출발": subject,
            "관계": relation_context_label(rel_type, source.get("type"), target.get("type")),
            "대상 유형": target_display_type,
            "대상": target_display,
        })

    st.subheader("관계 목록")
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.info("관계 목록이 비어 있습니다.")

def render_relationship_sentences(graph: dict[str, Any], center_uuid: str) -> None:
    rows = relationship_sentence_rows(graph, center_uuid)
    st.subheader("관계 문장")
    if rows:
        for row in rows[:30]:
            st.write(f"- {row}")
    else:
        st.info("문장으로 표시할 관계가 없습니다.")

def render_commonality_cards(graph: dict[str, Any]) -> None:
    cards = commonality_cards(graph)
    st.subheader("공통점 카드")
    if not cards:
        st.info("현재 필터에서는 공통 객체로 연결된 다른 페르소나가 없습니다. depth=2 또는 객체 필터를 확인해보세요.")
        return

    cols = st.columns(min(len(cards), 3))
    for idx, card in enumerate(cards[:6]):
        with cols[idx % len(cols)]:
            st.info(
                f"**{card['object_label']}**\n\n"
                f"유형: {card['object_type']}\n\n"
                f"연결된 페르소나: {card['person_count']}명\n\n"
                f"예: {', '.join(card['examples'])}"
            )

def commonality_cards(graph: dict[str, Any]) -> list[dict[str, Any]]:
    nodes_by_id = {node.get("id"): node for node in graph.get("nodes", [])}
    grouped: dict[str, set[str]] = {}
    for edge in dedupe_edges(graph.get("edges", [])):
        source = nodes_by_id.get(edge.get("source"), {})
        target = nodes_by_id.get(edge.get("target"), {})
        if source.get("type") != "Person" and target.get("type") == "Person":
            grouped.setdefault(str(source.get("id")), set()).add(str(target.get("label")))

    cards = []
    for node_id, people in grouped.items():
        if not people:
            continue
        node = nodes_by_id.get(node_id, {})
        cards.append({
            "object_label": node.get("label", node_id),
            "object_type": node_type_label(str(node.get("type", ""))),
            "person_count": len(people),
            "examples": sorted(people)[:4],
        })
    return sorted(cards, key=lambda item: item["person_count"], reverse=True)

def relationship_sentence_rows(graph: dict[str, Any], center_uuid: str) -> list[str]:
    nodes_by_id = {node.get("id"): node for node in graph.get("nodes", [])}
    center_id = f"person_{center_uuid}"
    sentences = []
    for edge in dedupe_edges(graph.get("edges", [])):
        source_id = edge.get("source")
        target_id = edge.get("target")
        source = nodes_by_id.get(source_id, {})
        target = nodes_by_id.get(target_id, {})
        rel_type = str(edge.get("type", ""))
        source_label = "중심 페르소나" if source_id == center_id else source.get("label", source_id)
        target_label = "중심 페르소나" if target_id == center_id else target.get("label", target_id)
        if target.get("type") == "Person" and source.get("type") != "Person":
            sentences.append(f"{target_label} → {relation_context_label(rel_type, source.get('type'), target.get('type'))} → {source_label}")
        else:
            sentences.append(f"{source_label} → {relation_context_label(rel_type, source.get('type'), target.get('type'))} → {target_label}")
    return sentences

def dedupe_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    unique = []
    for edge in edges:
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        rel_type = str(edge.get("type", ""))
        key = tuple(sorted([source, target]) + [rel_type]) if rel_type == "SIMILAR_TO" else (source, target, rel_type)
        if key not in seen:
            seen.add(key)
            unique.append(edge)
    return unique

def relation_context_label(relation_type: str, source_type: object, target_type: object) -> str:
    if target_type == "Person" and source_type != "Person":
        if relation_type == "ENJOYS_HOBBY":
            return "공통 취미"
        if relation_type == "HAS_SKILL":
            return "공통 스킬"
        if relation_type == "LIVES_IN":
            return "같은 지역"
    return relation_label(relation_type)

def relation_label(value: str) -> str:
    return RELATION_LABELS.get(value, value)

def node_type_label(value: str) -> str:
    return NODE_TYPE_LABELS.get(value, value)

if __name__ == "__main__":
    main()
