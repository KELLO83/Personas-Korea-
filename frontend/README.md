# Nemotron Personas React Frontend

This is the maintained frontend for the project. It is a Next.js application using React 19 and TypeScript.

The previous Streamlit frontend has been retired and is no longer maintained as the production UI. `streamlit` may still exist in the Python requirements for legacy compatibility, but new frontend work should happen in this directory.

## Run

```powershell
cd frontend
npm install
npm run dev
```

The default API base URL is `http://localhost:8000`. Override it with `.env.local` when needed.

```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

## Validate

```powershell
npm run typecheck
npm run lint
npm run build
```

## Runtime Stack

- Next.js 16
- React 19
- TypeScript 5
- D3 / d3-force
- ECharts
