#!/bin/bash
# =============================================================
# Universal Run Script (run.bat - Unix/Linux/WSL Conversion)
# Description: 프로젝트 빌드 및 실행 파이프라인
# - Neo4j Docker 인스턴스 자동 관리
# - Python 가상환경(.venv) 확인
# - FastAPI 백그라운드 실행 (uvicorn)
# - Next.js 프론트엔드 실행 (npm run dev)
#
# [주의] 원본 run.bat 파일은 그대로 보존됩니다.
# =============================================================

set -e  # 오류 발생 시 즉시 종료

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PYTHON="./.venv/bin/python"

# (Windows나 특수 환경일 경우 python.exe 경로 조정)
if [ ! -f "$PYTHON" ]; then
    if [ -f "./.venv/Scripts/python.exe" ]; then
        PYTHON="./.venv/Scripts/python.exe"
    fi
fi

echo "============================================"
echo "  Nemotron-Personas-Korea - Runner"
echo "============================================"

# --- [1] Python 가상환경 확인 ---
if [ ! -f "$PYTHON" ]; then
    echo "[ERROR] Python runtime not found at .venv"
    echo "[Fix]   py -3.11 -m venv .venv"
    echo "[Fix]   .\.venv\Scripts\python.exe -m pip install -r requirements.txt"
    exit 1
fi
echo "[INFO] Python Runtime: OK"

# --- [2] .env 파일 확인 ---
if [ ! -f "$ROOT/.env" ]; then
    if [ -f "$ROOT/.env.example" ]; then
        echo "[INFO] .env not found. Copying from .env.example."
        cp "$ROOT/.env.example" "$ROOT/.env"
    else
        echo "[WARN] .env not found. Create environment variables manually."
    fi
fi

# --- [3] Docker 및 Neo4j 컨테이너 자동 관리 (run.bat 로직 반영) ---
if command -v docker &> /dev/null; then
    if [ -f "$ROOT/.neo4j-docker.env" ]; then
        NEO4J_EXISTS=$(docker ps -a --filter "name=neo4j-personas" --format "{{.Names}}")
        
        if [ "$NEO4J_EXISTS" == "neo4j-personas" ]; then
            NEO4J_RUNNING=$(docker inspect -f '{{.State.Running}}' neo4j-personas)
            if [ "$NEO4J_RUNNING" == "true" ]; then
                echo "[INFO] Existing Neo4j container is already running."
            else
                echo "[INFO] Existing Neo4j container found. Starting it."
                docker start neo4j-personas > /dev/null 2>&1
            fi
        else
            echo "[INFO] Neo4j container not found. Creating neo4j:5 container."
            docker run -d --name neo4j-personas -p 7474:7474 -p 7687:7687 --env-file "$ROOT/.neo4j-docker.env" neo4j:5 > /dev/null 2>&1
        fi
    else
        echo "[WARN] .neo4j-docker.env not found. Skip auto-start Neo4j."
    fi
else
    echo "[WARN] docker not found. Skip auto-start Neo4j."
fi

# --- [4] Frontend 실행 (Next.js) ---
if command -v node &> /dev/null && command -v npm &> /dev/null; then
    if [ ! -d "$ROOT/frontend/node_modules" ]; then
        echo "[INFO] Next.js: Installing dependencies..."
        echo "[INFO] Starting Next.js Frontend (New Terminal context)..."
        (cd "$ROOT/frontend" && npm install && npm run dev) &
    else
        echo "[INFO] Starting Next.js Frontend..."
        (cd "$ROOT/frontend" && npm run dev) &
    fi
    FRONTEND_PID=$!
else
    echo "[WARN] node or npm not found. Next.js frontend start skipped."
fi

# --- [5] 백엔드 실행 (FastAPI - run.bat Line 84) ---
echo "[INFO] Starting FastAPI Backend..."
"$PYTHON" -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# --- [6] 완료 안내 ---
echo "============================================"
echo "[DONE] Started FastAPI(8000) and Next.js(3000)."
echo "Open: http://localhost:8000/docs and http://localhost:3000"
echo "============================================"

# 백그라운드 프로세스 유지 및 안전한 종료 대기
trap "kill $FRONTEND_PID $BACKEND_PID 2>/dev/null; exit" SIGINT SIGTERM
wait