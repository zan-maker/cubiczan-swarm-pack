# CUBICZAN Agent Swarm Intelligence Platform — Installation Pack

> Zero-token coordination. Heterogeneous agents. Enterprise-grade swarm intelligence.

Built on [MiroFish](https://github.com/666ghj/MiroFish) (AGPL-3.0) swarm simulation + [TEMM1E](https://github.com/nagisanzenin/temm1e) (MIT) stigmergic coordination + [Kimi K2.5 PARL](https://arxiv.org/abs/2602.02276) parallel agent architecture.

---

## Why This Exists

Every major multi-agent framework (AutoGen, CrewAI, LangGraph) coordinates agents by making them **talk to each other**. Every coordination message is an LLM call. Every LLM call costs tokens. In complex workflows, the coordination overhead can **exceed the actual work**.

**This is an architecture bug, not a feature.**

Cubiczan replaces inter-agent LLM chat with **stigmergy** — indirect communication via environmental signals (scent pheromones), the same mechanism ant colonies use to solve NP-hard routing problems without centralized control.

### The Math That Matters

| Metric | Traditional (AutoGen/CrewAI) | Cubiczan Hybrid |
|--------|------------------------------|-----------------|
| 12-subtask coordination tokens | ~78 LLM calls | **0 coordination calls** |
| Context growth per subtask | **28x** (quadratic: h̄·m(m+1)/2) | **~190 bytes flat** (linear) |
| Speed (12 independent tasks) | 103s | **18s (5.86x faster)** |
| Cost (12 independent tasks) | 7,379 tokens | **2,149 tokens (3.4x cheaper)** |
| Simple task overhead | Framework boot cost | **Zero. Invisible.** |

---

## Architecture: 3-Layer Hybrid

```
  REQUEST
     │
     ▼
┌──────────────────────────────────────────────────────────────┐
│  Layer 1: MoE ROUTER (1 nano LLM call)                      │
│  Classifies domain + complexity. Simple tasks → single agent │
│  Complex tasks (3+ deliverables, speedup ≥1.3x) → swarm     │
└────────────────────────┬─────────────────────────────────────┘
                         │ (complex only)
                         ▼
┌──────────────────────────────────────────────────────────────┐
│  Layer 2: ALPHA DECOMPOSITION (1 LLM call → DAG)            │
│  Kahn's topological sort, critical path optimization         │
│  Produces dependency graph with tagged subtasks               │
└────────────────────────┬─────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   Worker 1   │ │   Worker 2   │ │   Worker 3   │   Heterogeneous
│  Qwen-2.5    │ │  DeepSeek-R1 │ │  Llama-3.3   │   models (frozen)
│  ┌────────┐  │ │  ┌────────┐  │ │  ┌────────┐  │
│  │STIGMERGY│ │ │  │STIGMERGY│ │ │  │STIGMERGY│ │   Zero-token
│  │ Claim   │  │ │  │ Claim   │  │ │  │ Claim   │  │   coordination
│  │ Emit    │  │ │  │ Emit    │  │ │  │ Emit    │  │
│  │ Read    │  │ │  │ Read    │  │ │  │ Read    │  │
│  └────────┘  │ │  └────────┘  │ │  └────────┘  │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       └────────────────┼────────────────┘
                        ▼
              ┌──────────────────┐
              │  SCENT FIELD     │  SQLite-backed pheromone environment
              │  6 signal types  │  Exponential decay, GC every 10s
              │  Task selection: │  S = A^2.0 · U^1.5 · (1-D)^1.0
              │  Pure arithmetic │     · (1-F)^0.8 · R^1.2
              │  Zero LLM calls │
              └──────────────────┘
                        │
                        ▼ (if high-stakes)
              ┌──────────────────┐
              │ CONSENSUS ENGINE │  Adversarial debate + LMSR scoring
              │ Anti-sycophancy  │  Contrarian agent stress-testing
              │ Heterogeneous    │  Min 2 distinct model families
              └──────────────────┘
                        │
                        ▼
              ┌──────────────────┐
              │  PARL REWARD     │  rPARL = λ1·r_parallel + λ2·r_finish + r_perf
              │  (Kimi K2.5)     │  Prevents serial collapse + over-decomposition
              └──────────────────┘
```

### Stigmergy: How Workers Coordinate Without Talking

```
Worker 1 completes Task A → emits COMPLETION scent (5 min half-life)
Worker 2 reads scent field → sees Task B now has all deps met
Worker 2 claims Task B via atomic SQLite UPDATE...WHERE status='ready'
Worker 3 struggles on Task C → emits DIFFICULTY scent (2 min half-life)
Worker 4 reads field → avoids Task C, picks Task D instead (higher score)

Zero LLM calls. Zero coordination tokens. Pure arithmetic.
```

**6 Scent Signal Types:**

| Signal | Half-Life | Purpose |
|--------|-----------|---------|
| `COMPLETION` | 5 min | Task finished |
| `FAILURE` | 6 min | Attempt failed |
| `DIFFICULTY` | 2 min | Worker struggling |
| `URGENCY` | Grows (cap 5.0) | Prevents starvation |
| `PROGRESS` | 20 sec | Worker heartbeat |
| `HELP_WANTED` | 2 min | Specialist needed |

**Task Selection Formula** (40 lines of arithmetic, not an LLM call):
```
S(worker, task) = Affinity^2.0 × Urgency^1.5 × (1-Difficulty)^1.0
                  × (1-Failure)^0.8 × Reward^1.2
```

---

## Supported Domains (9)

| # | Domain | Feasibility | Key Use Cases |
|---|--------|------------|---------------|
| 1 | **Financial Markets & Trading** | HIGH | Kalshi/Polymarket, commodities, crypto, M&A screening |
| 2 | **Business Intelligence** | HIGH | M&A targets, competitive scanning, multi-source synthesis |
| 3 | **Cybersecurity & Threat Intel** | HIGH | SOC triage (88-97% noise reduction), threat hunting |
| 4 | **Predictive Simulation** | HIGH | Social dynamics forecasting via MiroFish OASIS engine |
| 5 | **Content & Marketing** | MED-HIGH | 3-5x production speed, viral prediction, SEO |
| 6 | **Healthcare & Drug Discovery** | MEDIUM | Drug repurposing, clinical trial design, regulatory nav |
| 7 | **Political & Social Forecasting** | MEDIUM | Election prediction (Brier 0.101 → closing gap) |
| 8 | **Real Estate & Location Intel** | MEDIUM | Gentrification prediction, site selection, climate risk |
| 9 | **Talent & HR Intelligence** | MEDIUM | Hiring timing, retention signals, comp benchmarking |

---

## Open-Source Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Simulation Engine** | MiroFish + OASIS (CAMEL-AI) | Multi-agent swarm simulation |
| **Coordination** | Stigmergy (TEMM1E-inspired) | Zero-token pheromone signals |
| **Parallelism** | PARL (Kimi K2.5-inspired) | Dynamic decomposition + reward shaping |
| **Consensus** | LMSR + Contrarian Agents | Anti-sycophancy adversarial debate |
| **LLM Backend** | Ollama / vLLM / llama.cpp | Self-hosted inference ($0 API cost) |
| **Models** | Qwen-2.5, DeepSeek-R1, Llama-3.3 | Heterogeneous pool (anti-groupthink) |
| **Knowledge Graph** | Neo4j + GraphRAG | Entity relationships & memory |
| **Vector Store** | Qdrant | Semantic search & retrieval |
| **Agent Memory** | Zep Cloud / self-hosted | Long-term memory with graph |
| **Task Store** | SQLite (atomic claiming) | Zero-lock task DAG management |
| **Frontend** | Vue 3 + D3.js | Dashboard & visualization |
| **Backend** | Flask | API layer |
| **Monitoring** | Grafana + Prometheus | Swarm health, cost, sycophancy alerts |
| **Containerization** | Docker Compose | Single-command deployment |

---

## Quick Start

```bash
# 1. Clone this pack
git clone <this-repo>
cd cubiczan-swarm-pack

# 2. Copy and configure environment
cp .env.example .env
# Edit .env with your API keys and preferences

# 3. One-command setup (installs everything + clones MiroFish)
./setup.sh          # Linux/Mac
setup.bat           # Windows

# 4. Launch the full stack
docker compose up -d

# 5. Pull self-hosted models (after Ollama container starts)
docker exec cubiczan-ollama ollama pull qwen2.5:32b
docker exec cubiczan-ollama ollama pull deepseek-r1:32b
docker exec cubiczan-ollama ollama pull llama3.3:latest

# 6. Open dashboard
# Frontend:     http://localhost:3000
# Backend API:  http://localhost:5001
# Orchestrator: http://localhost:5002
# Neo4j:        http://localhost:7474
# Grafana:      http://localhost:3001
# Qdrant:       http://localhost:6333
# Ollama:       http://localhost:11434
```

---

## Project Structure

```
cubiczan-swarm-pack/
├── .env.example                    # Full environment configuration
├── docker-compose.yml              # 8-service orchestration
├── setup.sh / setup.bat            # One-command installers
│
├── mirofish/                       # MiroFish core (cloned at setup)
│   ├── backend/                    # Flask API + OASIS simulation
│   └── frontend/                   # Vue 3 dashboard
│
├── orchestrator/                   # Hybrid coordination engine
│   ├── hybrid_coordinator.py       # MAIN: PARL + Stigmergy + Consensus
│   ├── stigmergy.py                # Scent field (TEMM1E-inspired)
│   ├── task_dag.py                 # DAG + atomic SQLite claiming
│   ├── parl.py                     # PARL reward shaping (Kimi K2.5)
│   ├── router.py                   # MoE nano-model router
│   ├── consensus.py                # Adversarial LMSR consensus
│   ├── anti_sycophancy.py          # Diversity metrics + enforcement
│   ├── swarm.py                    # Legacy swarm coordinator
│   ├── requirements.txt            # Python dependencies
│   └── Dockerfile                  # Container definition
│
├── agents/                         # Agent definitions
│   ├── base_agent.py               # Heterogeneous base agent
│   ├── contrarian.py               # Adversarial stress-tester
│   └── profiles/                   # Persona templates
│
├── domains/                        # 9 domain configurations
│   ├── financial-markets/swarm_config.json
│   ├── business-intelligence/swarm_config.json
│   ├── cybersecurity/swarm_config.json
│   ├── predictive-simulation/swarm_config.json
│   ├── content-marketing/swarm_config.json
│   ├── healthcare-pharma/swarm_config.json
│   ├── political-forecasting/swarm_config.json
│   ├── real-estate/swarm_config.json
│   └── talent-hr/swarm_config.json
│
├── integrations/                   # Data source connectors
│   └── market_data.py              # Financial, cyber, BI feeds
│
├── monitoring/                     # Observability stack
│   ├── grafana/dashboards/         # Pre-built dashboards
│   └── prometheus/prometheus.yml   # Metrics collection
│
└── docs/
    └── DOMAIN_PLAYBOOKS.md         # Per-domain deployment guides
```

---

## How It Compares

### vs. AutoGen / CrewAI / LangGraph

| Feature | AutoGen | CrewAI | LangGraph | **Cubiczan** |
|---------|---------|--------|-----------|-------------|
| Coordination | LLM-to-LLM chat | LLM delegation | Graph routing | **Stigmergy (0 tokens)** |
| Context growth | Quadratic | Quadratic | Linear (nodes) | **Flat (~190 bytes/worker)** |
| Anti-sycophancy | None | None | None | **LMSR + contrarian + model diversity** |
| Parallel execution | Sequential | Sequential default | Node-level | **Task-level (atomic SQLite)** |
| Simple task overhead | Framework boot | Framework boot | Framework boot | **Zero. Invisible.** |
| Domain specialization | Manual | Role-based | Manual | **9 pre-built enterprise domains** |
| Simulation engine | None | None | None | **MiroFish OASIS (33K+ stars)** |

### vs. TEMM1E (direct)

| Feature | TEMM1E | **Cubiczan** |
|---------|--------|-------------|
| Language | Rust (17 crates) | Python (enterprise-friendly) |
| Coordination | Stigmergy | **Stigmergy + PARL + Consensus** |
| Domain configs | Generic | **9 enterprise domain packs** |
| Simulation | None | **MiroFish parallel worlds** |
| Anti-sycophancy | N/A (single-model) | **Heterogeneous models + contrarian** |
| Knowledge graph | None | **Neo4j + Zep GraphRAG** |

### vs. Kimi K2.5 Agent Swarm (direct)

| Feature | Kimi K2.5 PARL | **Cubiczan** |
|---------|---------------|-------------|
| Coordination tokens | Reduced | **Zero (stigmergy)** |
| Open source models | Kimi K2.5 only | **Any OpenAI-compatible** |
| Task selection | LLM-based | **Arithmetic (40 LOC formula)** |
| Deployment | Research | **Docker Compose production** |
| Enterprise domains | None | **9 pre-configured domains** |

---

## Cost Estimates

| Configuration | Cloud API (Tiered) | DeepSeek-only | Self-hosted (Ollama) |
|---------------|--------------------|---------------|---------------------|
| 20 agents, hourly | ~$55/mo | ~$15/mo | Compute only |
| 20 agents, 5-min | ~$660/mo | ~$180/mo | Compute only |
| 50 agents, 5-min | ~$5,000/mo | ~$1,166/mo | Compute only |
| **Stigmergy savings** | **~3.4x cheaper** | **~3.4x cheaper** | **~3.4x fewer GPU-hours** |

---

## Key Research References

1. **MiroFish** — Swarm intelligence engine (33K+ GitHub stars). Parallel digital world simulation with OASIS.
2. **TEMM1E v3.0.0** — Stigmergic coordination. 5.86x faster, 3.4x cheaper than LLM-to-LLM chat. MIT licensed.
3. **Kimi K2.5 Agent Swarm** (arXiv:2602.02276) — PARL: trainable orchestrator + frozen subagents. 3-4.5x latency reduction.
4. **CONSENSAGENT** (ACL 2025) — Same-model agents converge sycophantically in 1-2 rounds. Heterogeneous models required.
5. **Google Research** — Centralized hub-and-spoke contains errors to 4.4x vs free-form multi-agent chat.

---

## License

- MiroFish core: AGPL-3.0
- TEMM1E stigmergy concepts: MIT
- Cubiczan extensions: AGPL-3.0
- Domain configurations: AGPL-3.0
