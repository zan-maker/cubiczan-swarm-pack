#!/usr/bin/env bash
set -euo pipefail

# =====================================================
# CUBICZAN Agent Swarm Intelligence Platform
# One-Command Setup Script
# =====================================================

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() { echo -e "${CYAN}[CUBICZAN]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# ============= PREREQUISITES CHECK =============
log "Checking prerequisites..."

command -v docker >/dev/null 2>&1 || error "Docker not found. Install: https://docs.docker.com/get-docker/"
command -v docker compose >/dev/null 2>&1 || error "Docker Compose not found. Install with Docker Desktop."
command -v git >/dev/null 2>&1 || error "Git not found."

# Check for Node.js (optional if using Docker only)
if command -v node >/dev/null 2>&1; then
    NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
    if [ "$NODE_VERSION" -lt 18 ]; then
        warn "Node.js 18+ recommended. Found: $(node -v)"
    else
        success "Node.js $(node -v) found"
    fi
else
    warn "Node.js not found. Docker deployment will handle this."
fi

# Check for Python (optional if using Docker only)
if command -v python3 >/dev/null 2>&1; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
    success "Python $PYTHON_VERSION found"
else
    warn "Python 3.11+ not found. Docker deployment will handle this."
fi

# ============= ENVIRONMENT SETUP =============
log "Setting up environment..."

if [ ! -f .env ]; then
    cp .env.example .env
    warn ".env file created from template. Please edit with your API keys before launching."
    warn "At minimum, configure: LLM_API_KEY, ZEP_API_KEY"
fi

# ============= CLONE MIROFISH =============
log "Setting up MiroFish core..."

if [ ! -d "mirofish" ]; then
    git clone https://github.com/666ghj/MiroFish.git mirofish
    success "MiroFish cloned successfully"
else
    log "MiroFish directory exists, pulling latest..."
    cd mirofish && git pull origin main 2>/dev/null || true && cd ..
    success "MiroFish updated"
fi

# ============= CREATE DIRECTORIES =============
log "Creating directory structure..."

mkdir -p monitoring/grafana/dashboards
mkdir -p monitoring/prometheus
mkdir -p mirofish/backend/uploads
mkdir -p agents/profiles

success "Directory structure created"

# ============= MONITORING CONFIG =============
log "Setting up monitoring..."

if [ ! -f monitoring/prometheus/prometheus.yml ]; then
    cat > monitoring/prometheus/prometheus.yml << 'PROMEOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: "cubiczan-orchestrator"
    static_configs:
      - targets: ["orchestrator:5002"]

  - job_name: "cubiczan-mirofish"
    static_configs:
      - targets: ["mirofish:5001"]

  - job_name: "ollama"
    static_configs:
      - targets: ["ollama:11434"]
    metrics_path: /api/metrics
PROMEOF
    success "Prometheus config created"
fi

# ============= PULL OLLAMA MODELS =============
log "Checking GPU availability for Ollama..."

if command -v nvidia-smi >/dev/null 2>&1; then
    success "NVIDIA GPU detected"
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "Unknown GPU")
    log "GPU: $GPU_INFO"
else
    warn "No NVIDIA GPU detected. Ollama will use CPU (slower)."
    warn "For production, GPU is strongly recommended."
fi

# ============= DOCKER BUILD =============
log "Building Docker containers..."

docker compose build --no-cache 2>/dev/null || warn "Docker build failed. You may need to configure .env first."

# ============= SUMMARY =============
echo ""
echo "=============================================="
echo -e "${GREEN}  CUBICZAN Setup Complete!${NC}"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Edit .env with your API keys"
echo "  2. Run: docker compose up -d"
echo "  3. Pull Ollama models (after containers start):"
echo "     docker exec cubiczan-ollama ollama pull qwen2.5:32b"
echo "     docker exec cubiczan-ollama ollama pull deepseek-r1:32b"
echo "     docker exec cubiczan-ollama ollama pull llama3.3:latest"
echo ""
echo "Access points:"
echo "  Frontend:   http://localhost:3000"
echo "  Backend:    http://localhost:5001"
echo "  Orchestrator: http://localhost:5002"
echo "  Neo4j:      http://localhost:7474"
echo "  Grafana:    http://localhost:3001"
echo "  Qdrant:     http://localhost:6333"
echo "  Ollama:     http://localhost:11434"
echo ""
echo "Active domains: $(grep ACTIVE_DOMAINS .env 2>/dev/null | cut -d= -f2 || echo 'financial,cybersecurity,business-intel')"
echo "=============================================="
