# Nemotron Personas React Frontend

Next.js 기반 현재 프론트엔드입니다. 기존 Streamlit 프론트는 폐기되었으며 더 이상 저장소에 유지되지 않습니다.

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
