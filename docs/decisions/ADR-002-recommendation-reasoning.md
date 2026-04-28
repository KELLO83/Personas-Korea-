# ADR-002: 추천 Reasoning은 템플릿 기반, LLM 동기 호출 금지

**상태**: Accepted  
**날짜**: 2026-04-28  
**결정자**: Oracle (Architecture Review) + 개발팀  
**관련 PRD**: Feature 11 (페르소나 추천 엔진)

---

## 문제 (Context)

Feature 11의 추천 응답에는 "왜 이것을 추천하는가"에 대한 사유(reasoning)가 포함되어야 합니다.

초기 PRD P2에는 LLM을 사용하여 "당신과 유사한 80%의 개발자가 TypeScript를 보유하고 있습니다"와 같은 문구를 동기 생성하는 방식이 제안되었습니다.

## 고려사항 (Considered Options)

### Option A: LLM 동기 호출 (초기 PRD P2 방식)
```python
# API 엔드포인트 내에서
reason = llm.invoke(f"왜 {item_name}을 추천하는가? {stats}를 근거로 설명하라")
```
- ✅ 자연스러운 한국어 문장 생성
- ✅ 복잡한 통계 해석 가능
- ❌ NVIDIA API LLM 지연: 500ms ~ 3000ms (변동 심함)
- ❌ `/api/recommend/{uuid}` 응답 시간: 2초 SLA 위반
- ❌ LLM 장애 시 추천 API 전체 다운

### Option B: 템플릿 기반 (채택)
```python
def generate_reason(item_name, similar_count, ratio):
    return f"당신과 유사한 {similar_count}명 중 {ratio:.0%}가 '{item_name}'을 가지고 있습니다."
```
- ✅ 응답 시간: < 1ms
- ✅ SLA 확실히 준수 (< 500ms)
- ✅ LLM 장애와 무관
- ❌ 문장 패턴이 고정됨 (다양성 부족)
- ❌ 복잡한 통계 해석 불가

### Option C: 비동기 LLM + 캐싱
- 추천 결과는 템플릿으로 즉시 반환
- reasoning은 백그라운드에서 LLM으로 생성 후 캐싱
- 두 번째 요청부터는 캐싱된 자연어 reasoning 반환
- ❌ 구현 복잡도 높음, Phase 3 MVP에는 과함

## 결정 (Decision)

**Option B (템플릿 기반)을 채택합니다. LLM 동기 호출은 금지합니다.**

### 근거
1. **API SLA**: `/api/recommend/{uuid}`는 < 500ms가 목표
2. **NVIDIA API LLM 지연**: 평균 1초+, 최악 5초+ (불확실성)
3. **추천은 "빠른 제안"**: 사용자가 즉시 다음 행동(클릭/무시)을 결정
4. **템플릿도 충분**: "유사한 X명 중 Y%가..."는 설득력 있는 사유

### 예외
- Phase 3 P2 이후 **비동기 LLM 사유 생성**을 검토할 수 있음
- 이 경우에도 동기 경로에는 영향 없음

## 구현 상세 (Implementation)

### 템플릿 정의
```python
REASON_TEMPLATES = {
    "hobby": "당신과 유사한 {similar_count}명 중 {ratio:.0%}가 '{item_name}'을(를) 취미로 가지고 있습니다.",
    "skill": "당신과 유사한 {similar_count}명 중 {ratio:.0%}가 '{item_name}' 스킬을 보유하고 있습니다.",
    "occupation": "당신과 유사한 {similar_count}명 중 {ratio:.0%}가 '{item_name}' 직업을 가지고 있습니다.",
    "district": "당신과 유사한 {similar_count}명 중 {ratio:.0%}가 '{item_name}'에 거주하고 있습니다.",
}
```

### 응답 예시
```json
{
  "item_name": "클라이밍",
  "reason": "당신과 유사한 128명 중 73%가 '클라이밍'을(를) 취미로 가지고 있습니다.",
  "reason_score": 0.73,
  "similar_users_count": 128
}
```

## 영향 (Consequences)

- `/api/recommend/{uuid}` 응답 시간: 2초 → 200ms (10x 개선)
- LLM 장애 영향: 추천 API는 LLM과 완전히 분리
- 사용자 경험: 사유 문장의 다양성은 감소하지만 신뢰도는 유지
- 향후 개선: 비동기 LLM 사유 생성은 Phase 3 이후 로드맵에 포함

## 관련 문서
- PRD v2.0 §4 (Feature 11)
- `src/graph/recommendation.py` (구현 예정)
