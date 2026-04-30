# 실행 가이드 (프론트 + 백엔드 한 번에 실행)

아래 스크립트는 **백엔드(FastAPI)**와 **Next.js React 프론트엔드**를 한 번에 실행합니다.

> 참고: 기존 Streamlit 프론트는 폐기되어 더 이상 저장소에 포함되지 않습니다. 현재 실행 대상 프론트는 Next.js(`frontend/`)뿐입니다.

## 사전 준비

1. Python 3.11 가상환경(`.venv`)이 이미 만들어져 있어야 합니다.
2. `.env` 파일이 있어야 합니다.
   - 없으면 먼저 `.env.example`을 복사해서 사용합니다.
3. Neo4j이 실행 중이어야 합니다.
   - 기본은 `bolt://localhost:7687`를 사용합니다.

## 실행 순서 (권장)

1. 프로젝트 루트에서 다음 파일만 실행하세요.

```powershell
run.bat
```

2. 실행이 되면 콘솔 창이 2개 열립니다.
   - FastAPI: `http://localhost:8000`
   - Next.js: `http://localhost:3000`

3. 동작 확인

- 백엔드 Swagger: `http://localhost:8000/docs`
- Next.js: `http://localhost:3000`

## `run.bat`의 동작

- `.env`가 없으면 `.env.example`에서 자동 복사합니다.
- `.venv`가 있으면 이를 사용해 FastAPI를 실행합니다.
- `docker`가 있고 Neo4j 컨테이너가 없으면 `neo4j:5`로 컨테이너를 시작합니다.
  - 이미 `neo4j-personas` 컨테이너가 존재하면 재실행 대신 `start`만 수행합니다.
  - 컨테이너명: `neo4j-personas`
  - `.neo4j-docker.env`를 함께 사용합니다.
- `frontend/node_modules`가 없으면 `npm install`을 먼저 실행한 뒤 Next.js를 시작합니다.

## 수동 실행(참고)

백엔드만 먼저 실행

```powershell
.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Next.js

```powershell
cd frontend
npm install
npm run dev
```

### 실행 실패 시 확인

- `'<command>' is not recognized` 류의 명령 오류가 뜬다면 `run.bat` 인코딩 또는 문자열 처리 이슈가 있었던 가능성이 있습니다. 이 경우 다음 순서로 점검하세요:
  - `run.bat`가 UTF-8로 저장되어 있는지 확인
  - `run.bat` 경로에 공백이 없는지 확인
  - 콘솔에서 `where docker`, `node --version`, `npm --version`이 정상인지 확인
  - 이미 실행 중인 `neo4j-personas`가 있으면 `docker ps -a --filter name=neo4j-personas`로 충돌 상태를 점검

## 참고

- Next.js API 주소는 `frontend/.env.local`에서 필요 시 변경하세요.
  - `NEXT_PUBLIC_API_BASE_URL=http://localhost:8000`
