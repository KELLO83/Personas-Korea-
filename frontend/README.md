# Nemotron Personas React Frontend

Next.js 기반 신규 프론트엔드입니다. 기존 Streamlit 앱(`app/streamlit_app.py`)은 삭제하지 않고 reference UI로 유지합니다.

## 실행

```powershell
cd frontend
npm install
npm run dev
```

기본 API 주소는 `http://localhost:8000`입니다. 변경이 필요하면 `.env.local`에 설정합니다.

```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

## 검증

```powershell
npm run typecheck
npm run lint
npm run build
```
