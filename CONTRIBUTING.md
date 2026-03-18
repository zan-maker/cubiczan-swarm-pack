# Contributing to Cubiczan

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/cubiczan-swarm-pack.git`
3. Create a branch: `git checkout -b feature/your-feature`
4. Run setup: `./setup.sh` (Linux/Mac) or `setup.bat` (Windows)

## Development Setup

### Prerequisites
- Docker Desktop
- Python 3.11+
- Node.js 18+
- At least one LLM API key (or Ollama installed locally)

### Running locally (without Docker)
```bash
# Backend (orchestrator)
cd orchestrator
pip install -r requirements.txt
python server.py

# MiroFish
cd mirofish
npm run setup:all
npm run dev
```

## Architecture Overview

The codebase has three coordination layers:

1. **Stigmergy** (`orchestrator/stigmergy.py`, `orchestrator/task_dag.py`) — Zero-token coordination via scent signals
2. **PARL** (`orchestrator/parl.py`, `orchestrator/hybrid_coordinator.py`) — Dynamic task decomposition
3. **Consensus** (`orchestrator/consensus.py`, `agents/contrarian.py`) — Anti-sycophancy adversarial debate

## Adding a New Domain

1. Create `domains/your-domain/swarm_config.json` following existing examples
2. Define `swarm_composition` with 4-5 specialist roles (last role should be Contrarian)
3. Add domain key to `.env.example` under `ACTIVE_DOMAINS`
4. Add data sources to `integrations/`
5. Document in `docs/DOMAIN_PLAYBOOKS.md`

## Pull Request Process

1. Ensure code runs with `python -c "from orchestrator.hybrid_coordinator import HybridCoordinator"`
2. Add tests for new stigmergy signals or scoring changes
3. Update README if adding new services to docker-compose
4. PR description must include benchmark numbers if changing coordination logic

## Code Style

- Python: Follow PEP 8, use type hints
- Use dataclasses over plain dicts
- Docstrings on all public functions
- Comments explaining "why", not "what"
