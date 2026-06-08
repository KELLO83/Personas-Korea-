@echo off
setlocal EnableExtensions
chcp 65001 >nul

set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
cd /d "%ROOT%"

set "PYTHON=%ROOT%\.venv314\Scripts\python.exe"

if not exist "%PYTHON%" (
  echo [ERROR] Python runtime not found at .venv314\Scripts\python.exe
  echo Fix:
  echo   py -3.14 -m venv .venv314
  echo   .\.venv314\Scripts\python.exe -m pip install -r requirements.txt
  goto :END
)

if not exist "%ROOT%\.env" (
  if exist "%ROOT%\.env.example" (
    echo [INFO] .env not found. Copying from .env.example.
    copy "%ROOT%\.env.example" "%ROOT%\.env" >nul
  ) else (
    echo [WARN] .env not found. Create environment variables manually.
  )
)

where node >nul 2>nul
if %errorlevel%==0 (
  set "HAS_NODE=1"
) else (
  set "HAS_NODE="
)

where npm >nul 2>nul
if %errorlevel%==0 (
  set "HAS_NPM=1"
) else (
  set "HAS_NPM="
)

where docker >nul 2>nul
if %errorlevel%==0 (
  if exist "%ROOT%\.neo4j-docker.env" (
    set "NEO4J_EXISTS="
    set "NEO4J_RUNNING="

    for /f "delims=" %%C in ('docker ps -a --format "{{.Names}}"') do if /I "%%C"=="neo4j-personas" set "NEO4J_EXISTS=%%C"

    if defined NEO4J_EXISTS (
      for /f "delims=" %%R in ('docker inspect -f "{{.State.Running}}" neo4j-personas 2^>nul') do set "NEO4J_RUNNING=%%R"

      if /I "%NEO4J_RUNNING%"=="true" (
        echo [INFO] Existing Neo4j container is already running.
      ) else (
        echo [INFO] Existing Neo4j container found. Starting it.
        docker start neo4j-personas >nul 2>nul
      )
    ) else (
      echo [INFO] Neo4j container not found. Creating neo4j:5 container.
      docker run -d --name neo4j-personas -p 7474:7474 -p 7687:7687 --env-file "%ROOT%\.neo4j-docker.env" neo4j:5 >nul
    )
  ) else (
    echo [WARN] .neo4j-docker.env not found. Skip auto-start Neo4j.
  )
) else (
  echo [WARN] docker not found. Skip auto-start Neo4j.
)

echo [INFO] PostgreSQL/pgvector is expected on local PostgreSQL: localhost:5432/persona_vector.
"%PYTHON%" -c "from dotenv import dotenv_values; import psycopg; uri = dotenv_values('.env').get('PGVECTOR_DATABASE_URI', 'postgresql://postgres:1234@localhost:5432/persona_vector'); conn = psycopg.connect(uri, connect_timeout=5); row = conn.execute('select current_database(), current_user').fetchone(); ext = conn.execute('select extversion from pg_extension where extname = chr(118) || chr(101) || chr(99) || chr(116) || chr(111) || chr(114)').fetchone(); conn.close(); pgvector = ext[0] if ext else 'missing'; print(f'[INFO] Local PostgreSQL ready: database={row[0]} user={row[1]} pgvector={pgvector}')"
if errorlevel 1 (
  echo [WARN] Local PostgreSQL/pgvector is not ready. Expected PGVECTOR_DATABASE_URI from .env.
)

if defined HAS_NODE (
  if defined HAS_NPM (
    if not exist "%ROOT%\frontend\node_modules" (
      start "Next.js (install + dev)" /D "%ROOT%\frontend" cmd /k "npm install && npm run dev"
    ) else (
      start "Next.js" /D "%ROOT%\frontend" cmd /k "npm run dev"
    )
  ) else (
    echo [WARN] npm not found. Next.js frontend start skipped.
  )
) else (
  echo [WARN] node not found. Next.js frontend start skipped.
)

start "FastAPI" /D "%ROOT%" cmd /k ""%PYTHON%" -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000"

echo [DONE] Started FastAPI(8000) and Next.js(4000).
echo Open: http://localhost:8000/docs and http://localhost:4000

:END
pause
