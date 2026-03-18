@echo off
REM =====================================================
REM CUBICZAN Agent Swarm Intelligence Platform
REM Windows Setup Script
REM =====================================================

echo [CUBICZAN] Checking prerequisites...

where docker >nul 2>&1 || (echo [ERROR] Docker not found. Install Docker Desktop. && exit /b 1)
where git >nul 2>&1 || (echo [ERROR] Git not found. && exit /b 1)

echo [OK] Prerequisites found

REM ============= ENVIRONMENT SETUP =============
echo [CUBICZAN] Setting up environment...

if not exist .env (
    copy .env.example .env
    echo [WARN] .env file created. Edit with your API keys before launching.
)

REM ============= CLONE MIROFISH =============
echo [CUBICZAN] Setting up MiroFish core...

if not exist mirofish (
    git clone https://github.com/666ghj/MiroFish.git mirofish
    echo [OK] MiroFish cloned
) else (
    echo [OK] MiroFish directory exists
)

REM ============= CREATE DIRECTORIES =============
echo [CUBICZAN] Creating directory structure...

if not exist monitoring\grafana\dashboards mkdir monitoring\grafana\dashboards
if not exist monitoring\prometheus mkdir monitoring\prometheus
if not exist mirofish\backend\uploads mkdir mirofish\backend\uploads
if not exist agents\profiles mkdir agents\profiles

echo [OK] Directories created

REM ============= DOCKER BUILD =============
echo [CUBICZAN] Building Docker containers...

docker compose build 2>nul || echo [WARN] Docker build needs .env configured first.

echo.
echo ==============================================
echo   CUBICZAN Setup Complete!
echo ==============================================
echo.
echo Next steps:
echo   1. Edit .env with your API keys
echo   2. Run: docker compose up -d
echo   3. Pull Ollama models:
echo      docker exec cubiczan-ollama ollama pull qwen2.5:32b
echo      docker exec cubiczan-ollama ollama pull deepseek-r1:32b
echo      docker exec cubiczan-ollama ollama pull llama3.3:latest
echo.
echo Access: http://localhost:3000 (Frontend)
echo ==============================================

pause
