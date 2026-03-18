"""
Swarm Coordinator — Orchestrates the full pipeline:
MoE Router → Specialist Cluster → Adversarial Consensus → Result

Integrates MiroFish simulation engine with LangGraph-style stateful workflows.
"""

import os
import json
import logging
from dataclasses import dataclass, field
from router import MoERouter, RoutingDecision, Domain
from consensus import ConsensusEngine, ConsensusResult

logger = logging.getLogger("cubiczan.swarm")


@dataclass
class SwarmTask:
    task_id: str
    description: str
    routing: RoutingDecision | None = None
    consensus: ConsensusResult | None = None
    status: str = "pending"  # pending, routing, analyzing, consensus, complete, escalated
    metadata: dict = field(default_factory=dict)


@dataclass
class SwarmConfig:
    agents_per_cluster: int = 4
    max_rounds: int = 20
    consensus_threshold: float = 0.75
    min_model_diversity: int = 2

    @classmethod
    def from_env(cls) -> "SwarmConfig":
        return cls(
            agents_per_cluster=int(os.getenv("AGENTS_PER_CLUSTER", "4")),
            max_rounds=int(os.getenv("MAX_SIMULATION_ROUNDS", "20")),
            consensus_threshold=float(os.getenv("CONSENSUS_THRESHOLD", "0.75")),
            min_model_diversity=int(os.getenv("MIN_MODEL_DIVERSITY", "2")),
        )


# Domain-specific system prompts loaded from config files
DOMAIN_PROMPTS: dict[str, str] = {}


def _load_domain_prompts():
    """Load domain-specific prompts from domains/ directory."""
    domains_dir = os.path.join(os.path.dirname(__file__), "..", "domains")
    if not os.path.isdir(domains_dir):
        return
    for domain_dir in os.listdir(domains_dir):
        config_path = os.path.join(domains_dir, domain_dir, "swarm_config.json")
        if os.path.isfile(config_path):
            with open(config_path) as f:
                config = json.load(f)
                DOMAIN_PROMPTS[domain_dir] = config.get("system_prompt", "")


class SwarmCoordinator:
    def __init__(self):
        self.router = MoERouter()
        self.consensus_engine = ConsensusEngine()
        self.config = SwarmConfig.from_env()
        _load_domain_prompts()

    def execute(self, task_id: str, description: str) -> SwarmTask:
        """Full swarm execution pipeline."""
        task = SwarmTask(task_id=task_id, description=description)

        # Step 1: Route
        task.status = "routing"
        logger.info(f"[{task_id}] Routing task...")
        try:
            task.routing = self.router.route(description)
        except ValueError as e:
            task.status = "error"
            task.metadata["error"] = str(e)
            return task

        logger.info(
            f"[{task_id}] Routed to {task.routing.domain.value} "
            f"(confidence={task.routing.confidence:.2f})"
        )

        # Step 2: Get domain context
        domain_prompt = DOMAIN_PROMPTS.get(task.routing.domain.value, "")
        enriched_description = description
        if domain_prompt:
            enriched_description = f"{domain_prompt}\n\nTASK: {description}"

        # Step 3: Run consensus
        task.status = "consensus"
        logger.info(f"[{task_id}] Running adversarial consensus...")
        task.consensus = self.consensus_engine.run_consensus(
            task=enriched_description,
            cluster_size=self.config.agents_per_cluster,
        )

        # Step 4: Determine outcome
        if task.consensus.escalate_to_human:
            task.status = "escalated"
            logger.warning(
                f"[{task_id}] No consensus reached after "
                f"{task.consensus.debate_rounds} rounds. Escalating."
            )
        else:
            task.status = "complete"
            logger.info(
                f"[{task_id}] Consensus reached: "
                f"score={task.consensus.consensus_score:.3f}, "
                f"rounds={task.consensus.debate_rounds}"
            )

        return task

    def execute_multi_domain(self, task_id: str, description: str) -> list[SwarmTask]:
        """Execute across multiple domains for cross-cutting tasks."""
        routing_decisions = self.router.route_multi(description)
        tasks = []
        for i, routing in enumerate(routing_decisions):
            sub_id = f"{task_id}-{routing.domain.value}"
            sub_task = SwarmTask(
                task_id=sub_id,
                description=description,
                routing=routing,
            )
            sub_task.status = "consensus"
            domain_prompt = DOMAIN_PROMPTS.get(routing.domain.value, "")
            enriched = f"{domain_prompt}\n\nTASK: {description}" if domain_prompt else description
            sub_task.consensus = self.consensus_engine.run_consensus(
                task=enriched,
                cluster_size=self.config.agents_per_cluster,
            )
            sub_task.status = "escalated" if sub_task.consensus.escalate_to_human else "complete"
            tasks.append(sub_task)

        return tasks
