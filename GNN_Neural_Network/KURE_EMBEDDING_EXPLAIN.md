# KURE-v1 임베딩 모델 기술 스펙 및 전처리 규칙

## 1. 모델 개요
- **모델명:** `nlpai-lab/KURE-v1`
- **라이브러리:** `sentence-transformers`
- **출력 차원:** 768 (Dense Vector)
- **역할:** Persona(페르소나) 텍스트와 후보 취미(Hobby) 간의 의미론적 유사도(Cosine Similarity) 계산.

## 2. 입력 데이터 구조화 규칙 (Domain Tagging)
단순 텍스트 이어붙임이 아닌, 문맥(Context) 유지를 위해 **7개 도메인별 Prefix 태그**를 적용하여 결합합니다.

### 태그 규칙
| 태그 | 칼럼명 | 의미 |
|------|--------|------|
| `[PROF]` | `professional_persona` | 직업 및 업무 맥락 |
| `[SPORT]` | `sports_persona` | 운동/활동 선호도 |
| `[ART]` | `arts_persona` | 문화/예술 취향 |
| `[TRAV]` | `travel_persona` | 여행 스타일 |
| `[FOOD]` | `culinary_persona` | 식습관 및 요리 스타일 |
| `[FAM]` | `family_persona` | 가족 관계 및 주거 형태 |
| `[CULT]` | `cultural_background` | 가치관 및 성장 배경 |

### 결합 예시
```text
[PROF] IT 개발자로 일하며 주로 야근을 한다. [SPORT] 격렬한 운동보다는 퇴근 후 가벼운 산책을 즐긴다. [FAM] 현재는 미혼이며 개인의 여가를 중시한다.
```

## 3. 누수 방지 (Leakage-Safe) 규칙
정답인 '취미(Hobby)'가 텍스트에 그대로 노출되는 것을 막기 위해 **`[ACT]` 플레이스홀더**로 대체합니다.

- **적용 전:** "나는 주말에 **등산**을 즐긴다."
- **적용 후:** "나는 주말에 **[ACT]**를 즐긴다."

**목적:** KURE 모델이 특정 취미 단어를 외워서 속이는(Overfitting) 것을 방지하고, '활동(Action)에 대한 라이프스타일 맥락' 자체를 벡터화하도록 유도합니다.

## 4. 데이터 플로우 (Pipeline)
1. **추출:** Persona의 7개 도메인 텍스트 추출.
2. **마스킹:** 후보 취미 단어를 모두 `[ACT]`로 치환.
3. **태깅:** 각 텍스트 블록 앞에 `[PROF]`, `[SPORT]` 등의 Prefix 붙여 하나의 문장으로 결합.
4. **임베딩:** KURE-v1 모델에 입력하여 768차원 Persona 벡터 생성.
5. **유사도 계산:** 모든 후보 취미 벡터와 코사인 유사도(Cosine Similarity) 산출.
