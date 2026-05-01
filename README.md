# Korean Persona Knowledge Graph Insight Platform

NVIDIA `Nemotron-Personas-Korea` 데이터셋을 Neo4j 지식 그래프로 구성하고, 한국인 가상 페르소나 데이터를 검색·비교·분석할 수 있도록 만든 프로젝트입니다.

성별, 연령, 지역, 직업, 취미, 기술, 가족 정보 등 여러 속성을 연결해 다음과 같은 질문을 확인할 수 있습니다.

- 광주에 사는 30대 남성들이 많이 가진 취미는 무엇인가?
- 서울 서초구 여성들의 직업 목표나 관심사는 어떤 경향이 있는가?
- 특정 페르소나와 생활 방식이 비슷한 사람은 누구인가?
- 두 페르소나는 어떤 지역, 직업, 취미, 기술로 연결되는가?

## 문서 역할 및 우선순위

- `PRD.md`: 요구사항, 범위, 아키텍처 결정의 기준 문서입니다.
- `TASKS.md`: PRD 실행 상태와 진행 조건을 추적하는 실행 기준 문서입니다.
- `README.md`: 설치/실행 방법, 현재 상태, 참고 자료를 정리하는 운영 안내서입니다.
- 충돌이 있을 경우 우선순위는 **`PRD.md` → `TASKS.md` → `README.md`** 입니다.

이 규칙은 하위 실험 문서(`docs/` 및 GNN 보조 문서)에도 동일하게 적용됩니다.

---

## 프로젝트 목적

이 프로젝트의 목적은 큰 규모의 페르소나 데이터를 단순 표 형태로만 보는 것이 아니라, 사람과 속성 사이의 관계를 그래프로 구성해 탐색하기 쉽게 만드는 것입니다.

일반 사용자는 화면에서 조건을 선택하거나 자연어로 질문해 데이터를 확인할 수 있고, 개발자는 FastAPI 엔드포인트를 통해 검색, 통계, 추천, 그래프 탐색 기능을 사용할 수 있습니다.

---

## 주요 기능

### 1. 페르소나 검색
지역, 연령대, 성별, 직업, 취미, 기술 등의 조건으로 페르소나를 검색합니다.

예시:
- 서울에 사는 20대 개발자
- 등산을 취미로 가진 40대 남성
- 특정 기술을 가진 페르소나 목록

### 2. 프로필 상세 보기
특정 페르소나의 기본 정보와 연결된 속성을 한 번에 확인합니다.

포함 정보:
- 나이, 성별, 거주 지역, 직업
- 취미, 관심사, 보유 기술
- 유사 페르소나 미리보기
- 커뮤니티 정보

### 3. 통계 대시보드
전체 데이터 또는 필터링된 그룹의 분포를 확인합니다.

확인 가능한 항목:
- 연령대 분포
- 성별 분포
- 지역 분포
- 직업, 취미, 기술 순위

### 4. 자연어 인사이트 질의
Cypher 쿼리를 직접 작성하지 않아도 문장 형태의 질문으로 데이터를 조회할 수 있습니다.

예시:
- “부산 30대 여성들이 많이 가진 취미는?”
- “서울 거주자 중 개발자 비율은?”
- “특정 UUID와 비슷한 사람을 찾아줘”

### 5. 유사 페르소나 및 추천
특정 페르소나를 기준으로 유사한 사람을 찾고, 유사 그룹에서 자주 나타나는 취미나 기술을 추천합니다.

추천은 실시간 LLM 호출이 아니라 그래프에 저장된 유사도 관계와 속성 빈도를 기반으로 계산합니다.

### 6. 커뮤니티 및 관계 경로 분석
비슷한 속성을 공유하는 페르소나 그룹을 확인하고, 두 페르소나가 어떤 속성으로 연결되는지 경로를 조회합니다.

---

## 데이터셋

| 항목 | 내용 |
|---|---|
| Dataset ID | `nvidia/Nemotron-Personas-Korea` |
| 제공처 | Hugging Face |
| 데이터 수 | 1,000,000 rows |
| 주요 컬럼 | 성별, 나이, 직업, 지역, 취미, 기술, 페르소나 텍스트 |
| 언어 | 한국어 |
| 라이선스 | CC-BY-4.0 |

기본 설정에서는 전체 데이터를 사용할 수 있습니다. 개발 또는 테스트 환경에서는 `.env`의 `DATA_SAMPLE_SIZE` 값을 지정해 일부 데이터만 로드할 수 있습니다.

# Personas-Korea-

---

## 현재 구현 상태

| 구분 | 상태 | 내용 |
|---|---|---|
| Phase 1 | 완료 | 데이터 적재, 지식 그래프 구축, 유사도 매칭, 커뮤니티 탐지 |
| Phase 2 | 완료 | 검색/필터, 통계 대시보드, 프로필 상세, 세그먼트 비교, 서브그래프 조회 |
| Phase 3 | 핵심 기능 구현 완료 및 검증 진행 중 | 네트워크 영향력 분석, 추천 엔진, 대화형 탐색 기능 |

자세한 구현 항목은 `TASKS.md`에서 확인할 수 있습니다. 기능 요구사항과 설계 배경은 `PRD.md`에 정리되어 있습니다.

---

## 시스템 구성

```text
[Next.js React 화면]
      |
      v
[FastAPI 서버]
      |
      +-- 검색 / 통계 / 프로필 / 추천 API
      +-- 자연어 질의 처리
      +-- 그래프 경로 및 커뮤니티 조회
      |
      v
[Neo4j 지식 그래프]
      |
      +-- Person, Region, Occupation, Hobby, Skill 노드
      +-- LIVES_IN, WORKS_AS, ENJOYS_HOBBY, HAS_SKILL 관계
      +-- GDS 기반 유사도, 커뮤니티, 중심성 결과
```

---

## 기술 스택

| 영역 | 사용 기술 |
|---|---|
| 백엔드 | Python 3.11, FastAPI |
| 화면 | Next.js React |
| 그래프 DB | Neo4j Community Edition 5.x |
| 그래프 분석 | Neo4j GDS, FastRP, KNN, Leiden, PageRank |
| 자연어 질의 | LangChain, LangGraph |
| 임베딩 | KURE-v1 |
| 테스트 | PyTest |

---

## 로컬 실행 방법

이 프로젝트는 Python 3.11 가상환경 기준으로 실행합니다.

### 1. 가상환경 생성 및 패키지 설치

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. 환경 변수 설정

`.env.example`을 복사해 `.env` 파일을 만든 뒤 Neo4j 등 실행에 필요한 환경 변수 값을 입력합니다.

```powershell
Copy-Item .env.example .env
```

주요 설정값:

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here

NVIDIA_API_KEY=your_nvidia_api_key_here
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
LLM_MODEL=deepseek-ai/deepseek-v4-pro

HF_DATASET_ID=nvidia/Nemotron-Personas-Korea
HF_DATASET_SPLIT=train
DATA_SAMPLE_SIZE=

EMBEDDING_MODEL_NAME=nlpai-lab/KURE-v1
EMBEDDING_DEVICE=cuda
```

### 3. 백엔드 실행

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### 4. 화면 실행

### Next.js React 화면

React 프론트는 `frontend/`에서 실행합니다. FastAPI 서버는 먼저 `:8000`에서 실행되어 있어야 합니다.

```powershell
cd frontend
npm install
npm run dev
```

기본 접속 주소는 `http://localhost:3000`입니다. API 주소를 바꾸려면 `frontend/.env.local`에 아래 값을 설정합니다.

```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

---

## 테스트

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

---

## 관련 문서

- `PRD.md`: 제품 요구사항과 기능 설계
- `TASKS.md`: 구현 진행 상황
- `docs/embedding-storage-workflow.md`: 임베딩 저장 및 적재 절차
- `docs/centrality-batch-operations.md`: F10 중심성 배치 실행 및 외부 스케줄러 운영 절차
- `docs/f12-chatbot-implementation-plan.md`: F12 대화형 탐색 챗봇 구현 계획
- `docs/decisions/`: 주요 아키텍처 결정 기록

---

## 참고

이 저장소에는 실제 원본 데이터와 개인 설정 파일을 포함하지 않습니다. 데이터 파일, `.env`, 로컬 DB 파일은 `.gitignore`에 의해 제외됩니다.
