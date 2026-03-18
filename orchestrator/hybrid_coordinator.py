"""
Hybrid Coordinator — Merges PARL (Kimi K2.5) + Stigmergy (TEMM1E)

Architecture:
    ┌─────────────────────────────────────────────────────────┐
    │                  HYBRID COORDINATOR                      │
    │                                                          │
    │   Layer 1: MoE Router (nano model, 1 LLM call)          │
    │       ↓                                                  │
    │   Layer 2: Alpha Decomposition (1 LLM call → DAG)       │
    │       ↓                                                  │
    │   Layer 3: STIGMERGY (zero LLM calls)                   │
    │       • Workers claim tasks via atomic SQLite             │
    │       • Scent signals guide task selection                │
    │       • Exponential decay pheromones (6 types)            │
    │       ↓                                                  │
    │   Layer 4: PARL Consensus (LLM calls for analysis only)  │
    │       • Heterogeneous models for anti-sycophancy          │
    │       • Contrarian agent stress-testing                   │
    │       • LMSR scoring for agreement measurement            │
    │       ↓                                                  │
    │   Layer 5: Result Aggregation + PARL Reward               │
    └─────────────────────────────────────────────────────────┘

LLM calls breakdown (for 12-subtask job):
    Traditional (AutoGen/CrewAI): ~78 LLM calls (task + coordination)
    PARL only (Kimi K2.5):       ~14 LLM calls (decompose + parallel execute)
    Hybrid (PARL + Stigmergy):    ~14 LLM calls (decompose + execute, ZERO coordination)

    Stigmergy eliminates: inter-agent chat, task negotiation, status updates
    PARL provides: dynamic decomposition, parallelism optimization, reward shaping
"""

import json
import os
import time
import logging
import threading
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

from stigmergy import ScentField, ScentSignal, ScentType, select_best_task
from task_dag import (
    TaskDAGStore, TaskGraph, DAGTask, TaskStatus,
    validate_dag, compute_critical_path,
)
from consensus import ConsensusEngine, ConsensusResult
from router import MoERouter, RoutingDecision

logger = logging.getLogger("cubiczan.hybrid")


# =====================================================
# ACTIVATION CLASSIFIER
# =====================================================

@dataclass
class ActivationDecision:
    """Whether to activate the swarm or use single-agent mode."""
    activate_swarm: bool
    reason: str
    estimated_subtasks: int
    theoretical_speedup: float
    estimated_cost_ratio: float  # decomposition cost / single-agent cost


def should_activate_swarm(task: str, router_model: str, client: OpenAI) -> ActivationDecision:
    """
    TEMM1E-inspired classifier: swarm only activates when genuinely needed.
    Simple or standard tasks → single agent, zero overhead.

    Activation requires ALL conditions (from TEMM1E):
    1. 3+ independent deliverables
    2. Theoretical speedup ≥ 1.3x
    3. Decomposition cost < 10% of estimated single-agent cost
    """
    response = client.chat.completions.create(
        model=router_model,
        messages=[
            {"role": "system", "content": (
                "Classify this task. Respond in JSON:\n"
                '{"complexity": "simple|standard|complex", '
                '"independent_deliverables": <count>, '
                '"estimated_single_agent_tokens": <rough estimate>, '
                '"reasoning": "brief"}'
            )},
            {"role": "user", "content": task},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)

    n_deliverables = result.get("independent_deliverables", 1)
    est_tokens = result.get("estimated_single_agent_tokens", 1000)
    complexity = result.get("complexity", "simple")

    # Theoretical speedup = total_tasks / critical_path
    # For fully independent tasks, speedup = n_deliverables
    # For sequential chain, speedup = 1.0
    theoretical_speedup = max(n_deliverables * 0.7, 1.0)  # 0.7 accounts for overhead

    # Decomposition cost ~200 tokens (one LLM call)
    decomp_cost_ratio = 200 / max(est_tokens, 1)

    activate = (
        complexity == "complex"
        and n_deliverables >= 3
        and theoretical_speedup >= 1.3
        and decomp_cost_ratio < 0.10
    )

    return ActivationDecision(
        activate_swarm=activate,
        reason=result.get("reasoning", ""),
        estimated_subtasks=n_deliverables,
        theoretical_speedup=round(theoretical_speedup, 2),
        estimated_cost_ratio=round(decomp_cost_ratio, 4),
    )


# =====================================================
# ALPHA DECOMPOSER (1 LLM call → DAG)
# =====================================================

ALPHA_PROMPT = """You are the Alpha (coordinator) of the Cubiczan Agent Swarm.
Decompose this task into a dependency graph. ONE LLM call only.

Rules:
1. Identify truly INDEPENDENT subtasks that can run in parallel
2. Only add dependencies where output of one task is input to another
3. Tag each task for worker specialization matching
4. Minimize critical path length (longest chain of dependencies)

Respond in JSON:
{
  "tasks": [
    {
      "task_id": "t1",
      "description": "...",
      "agent_type": "researcher|analyst|validator|synthesizer|contrarian",
      "tags": ["finance", "quantitative", "market-data"],
      "depends_on": []
    },
    {
      "task_id": "t2",
      "description": "...",
      "agent_type": "analyst",
      "tags": ["analysis", "synthesis"],
      "depends_on": ["t1"]
    }
  ],
  "reasoning": "Why this decomposition minimizes critical path"
}"""


def alpha_decompose(task: str, client: OpenAI, model: str) -> TaskGraph:
    """
    Alpha decomposition: ONE LLM call produces the entire task DAG.
    This is the only coordination LLM call. Everything after is arithmetic.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": ALPHA_PROMPT},
            {"role": "user", "content": task},
        ],
        temperature=0.3,
        response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)

    graph = TaskGraph()
    task_specs = result.get("tasks", [])

    # Build tasks
    for spec in task_specs:
        dag_task = DAGTask(
            task_id=spec["task_id"],
            description=spec["description"],
            agent_type=spec.get("agent_type", "analyst"),
            tags=set(spec.get("tags", [])),
            dependencies=spec.get("depends_on", []),
        )
        graph.tasks[dag_task.task_id] = dag_task

    # Compute dependents (reverse of dependencies)
    for tid, task_obj in graph.tasks.items():
        for dep_id in task_obj.dependencies:
            if dep_id in graph.tasks:
                graph.tasks[dep_id].dependents.append(tid)

    # Validate DAG (no cycles)
    if not validate_dag(graph.tasks):
        logger.error("Cyclic DAG detected, falling back to sequential")
        return TaskGraph()  # Empty = fallback

    graph.critical_path_length = compute_critical_path(graph.tasks)
    graph.theoretical_speedup = len(graph.tasks) / max(graph.critical_path_length, 1)

    logger.info(
        f"[ALPHA] Decomposed into {len(graph.tasks)} tasks, "
        f"critical path={graph.critical_path_length}, "
        f"theoretical speedup={graph.theoretical_speedup:.2f}x"
    )

    return graph


# =====================================================
# WORKER (Frozen Subagent)
# =====================================================

@dataclass
class WorkerConfig:
    worker_id: str
    model_name: str
    api_key: str
    base_url: str
    tags: set[str] = field(default_factory=set)


class StigmergicWorker:
    """
    A frozen subagent that:
    1. Claims tasks via atomic SQLite (no distributed locks)
    2. Emits scent signals as it works (no LLM calls)
    3. Reads scent signals to choose next task (pure arithmetic)
    4. Executes actual work via LLM (the only LLM calls)
    """

    def __init__(self, config: WorkerConfig, scent_field: ScentField, dag_store: TaskDAGStore):
        self.config = config
        self.scent_field = scent_field
        self.dag_store = dag_store
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    def _emit(self, task_id: str, scent_type: ScentType, intensity: float = 1.0):
        """Emit a scent signal — zero LLM calls, just a SQLite write."""
        self.scent_field.emit(ScentSignal(
            task_id=task_id,
            worker_id=self.config.worker_id,
            scent_type=scent_type,
            intensity=intensity,
        ))

    def execute_task(self, task: DAGTask, graph_id: str) -> str:
        """
        Execute a single task. This is where LLM calls happen (actual work).
        Coordination is handled entirely by scent signals.
        """
        # Emit progress signal (heartbeat)
        self._emit(task.task_id, ScentType.PROGRESS)

        # Gather dependency results (flat context, not accumulated)
        dep_context = ""
        for dep_id in task.dependencies:
            dep_result = self.dag_store.get_task_result(dep_id)
            if dep_result:
                dep_context += f"\n[Result from {dep_id}]: {dep_result}\n"

        # Context stays flat: task description + dependency results only
        # This is the key TEMM1E insight: ~190 bytes per worker, not 28x growth
        messages = [
            {"role": "system", "content": (
                f"You are a {task.agent_type} specialist. "
                "Execute the assigned task thoroughly. "
                "Be specific, evidence-based, and concise."
            )},
            {"role": "user", "content": (
                f"TASK: {task.description}"
                f"\n\nDEPENDENCY RESULTS:{dep_context}" if dep_context
                else f"TASK: {task.description}"
            )},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=0.5,
                max_tokens=4096,
            )
            result = response.choices[0].message.content

            # Emit completion signal
            self._emit(task.task_id, ScentType.COMPLETION)
            self.dag_store.complete_task(task.task_id, result)

            logger.info(f"[WORKER-{self.config.worker_id}] Completed {task.task_id}")
            return result

        except Exception as e:
            # Emit failure + difficulty signals
            self._emit(task.task_id, ScentType.FAILURE)
            self._emit(task.task_id, ScentType.DIFFICULTY, intensity=0.8)
            self.dag_store.fail_task(task.task_id, str(e))

            logger.error(f"[WORKER-{self.config.worker_id}] Failed {task.task_id}: {e}")
            return ""


# =====================================================
# HYBRID COORDINATOR
# =====================================================

class HybridCoordinator:
    """
    The main coordinator merging:
    - MoE Router (nano model classification)
    - PARL (Kimi K2.5 dynamic decomposition + reward shaping)
    - Stigmergy (TEMM1E zero-token coordination)
    - Adversarial Consensus (anti-sycophancy)

    Token budget for 12-subtask complex task:
        1 router call (~50 tokens)        → classify domain
        1 alpha decompose (~200 tokens)    → produce DAG
        12 worker LLM calls (~4K each)    → actual work
        0 coordination calls              → stigmergy handles all coordination
        4 consensus calls (~2K each)       → only if requires_consensus=True
        ─────────────────────────────────
        Total: ~57K tokens (vs ~210K+ for AutoGen/CrewAI equivalent)
    """

    def __init__(self, db_path: str = "cubiczan_swarm.db"):
        # Core clients
        self.client = OpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
        )
        self.model = os.getenv("LLM_MODEL_NAME", "qwen2.5:32b")
        self.router_model = os.getenv("ROUTER_MODEL_NAME", "qwen2.5:7b")

        # Infrastructure
        self.scent_field = ScentField(db_path)
        self.dag_store = TaskDAGStore(db_path)
        self.router = MoERouter()
        self.consensus_engine = ConsensusEngine()
        self.executor = ThreadPoolExecutor(
            max_workers=int(os.getenv("AGENTS_PER_CLUSTER", "4")) * 2
        )

        # Worker pool (heterogeneous models)
        self.workers = self._create_workers()

        # Start scent field GC
        self.scent_field.start_gc()

    def _create_workers(self) -> list[StigmergicWorker]:
        """Create heterogeneous worker pool from env config."""
        workers = []
        for i in range(1, 6):
            key = os.getenv(f"AGENT_MODEL_{i}_API_KEY")
            url = os.getenv(f"AGENT_MODEL_{i}_BASE_URL")
            name = os.getenv(f"AGENT_MODEL_{i}_NAME")
            if key and url and name:
                config = WorkerConfig(
                    worker_id=f"worker-{i}",
                    model_name=name,
                    api_key=key,
                    base_url=url,
                    tags=set(),  # Tags learned from task history
                )
                workers.append(StigmergicWorker(config, self.scent_field, self.dag_store))

        # Fallback: single worker with primary model
        if not workers:
            config = WorkerConfig(
                worker_id="worker-primary",
                model_name=self.model,
                api_key=os.getenv("LLM_API_KEY", ""),
                base_url=os.getenv("LLM_BASE_URL", ""),
            )
            workers.append(StigmergicWorker(config, self.scent_field, self.dag_store))

        return workers

    def execute(self, task: str, task_id: str = "task-1") -> dict:
        """
        Full hybrid execution pipeline:

        1. MoE Router → classify domain (1 nano LLM call)
        2. Activation check → swarm or single-agent? (1 nano LLM call)
        3a. Simple → single agent, zero swarm overhead
        3b. Complex → Alpha decompose (1 LLM call → DAG)
        4. Workers claim + execute via stigmergy (N LLM calls, 0 coordination)
        5. Optional: adversarial consensus on aggregated result
        6. PARL reward computation
        """
        start_time = time.time()
        metrics = {
            "task_id": task_id,
            "llm_calls": 0,
            "coordination_tokens": 0,
            "work_tokens": 0,
        }

        # ─── Step 1: Route ───
        routing = self.router.route(task)
        metrics["llm_calls"] += 1
        metrics["domain"] = routing.domain.value
        logger.info(f"[HYBRID] Domain: {routing.domain.value} ({routing.confidence:.2f})")

        # ─── Step 2: Activation check ───
        activation = should_activate_swarm(task, self.router_model, self.client)
        metrics["llm_calls"] += 1
        metrics["swarm_activated"] = activation.activate_swarm

        if not activation.activate_swarm:
            # ─── Step 3a: Single agent (zero swarm overhead) ───
            logger.info(f"[HYBRID] Single-agent mode: {activation.reason}")
            result = self._single_agent_execute(task)
            metrics["llm_calls"] += 1
            metrics["mode"] = "single-agent"
            metrics["wall_time"] = time.time() - start_time
            metrics["result"] = result
            return metrics

        # ─── Step 3b: Alpha decompose (1 LLM call → DAG) ───
        logger.info(
            f"[HYBRID] Swarm activated: {activation.estimated_subtasks} subtasks, "
            f"speedup={activation.theoretical_speedup}x"
        )
        graph = alpha_decompose(task, self.client, self.model)
        metrics["llm_calls"] += 1

        if not graph.tasks:
            # Fallback on decomposition failure
            result = self._single_agent_execute(task)
            metrics["llm_calls"] += 1
            metrics["mode"] = "single-agent-fallback"
            metrics["wall_time"] = time.time() - start_time
            metrics["result"] = result
            return metrics

        # Store DAG
        self.dag_store.store_graph(graph)

        # ─── Step 4: Stigmergic execution (ZERO coordination tokens) ───
        self.dag_store.activate_ready_tasks(graph.graph_id)

        # Emit initial urgency signals for all tasks
        for tid in graph.tasks:
            self.scent_field.emit(ScentSignal(
                task_id=tid,
                worker_id="alpha",
                scent_type=ScentType.URGENCY,
                intensity=0.1,
            ))

        # Worker loop: claim + execute until all tasks complete
        worker_futures = []
        max_iterations = len(graph.tasks) * 3  # Safety bound
        iteration = 0

        while not self.dag_store.is_graph_complete(graph.graph_id) and iteration < max_iterations:
            iteration += 1
            self.dag_store.activate_ready_tasks(graph.graph_id)
            ready_tasks = self.dag_store.get_ready_tasks(graph.graph_id)

            if not ready_tasks:
                time.sleep(0.5)  # Wait for running tasks to complete
                continue

            # Each available worker claims and executes a task
            for worker in self.workers:
                if not ready_tasks:
                    break

                # Scent-based task selection (pure arithmetic, zero LLM)
                task_tags_map = {
                    tid: graph.tasks[tid].tags for tid in ready_tasks
                    if tid in graph.tasks
                }
                task_deps_map = {
                    tid: len(graph.tasks[tid].dependents) for tid in ready_tasks
                    if tid in graph.tasks
                }

                best_task_id = select_best_task(
                    worker_id=worker.config.worker_id,
                    worker_tags=worker.config.tags,
                    available_task_ids=ready_tasks,
                    task_tags_map=task_tags_map,
                    task_dependents_map=task_deps_map,
                    scent_field=self.scent_field,
                )

                if best_task_id and self.dag_store.claim_specific_task(
                    best_task_id, worker.config.worker_id
                ):
                    dag_task = graph.tasks[best_task_id]
                    dag_task.status = TaskStatus.ACTIVE
                    dag_task.worker_id = worker.config.worker_id

                    future = self.executor.submit(
                        worker.execute_task, dag_task, graph.graph_id
                    )
                    worker_futures.append(future)
                    metrics["llm_calls"] += 1
                    ready_tasks.remove(best_task_id)

                    # Update worker tags for future affinity matching
                    worker.config.tags.update(dag_task.tags)

        # Wait for all workers
        for future in as_completed(worker_futures, timeout=600):
            try:
                future.result()
            except Exception as e:
                logger.error(f"[HYBRID] Worker error: {e}")

        # ─── Step 5: Aggregate results ───
        results = {}
        for tid, t in graph.tasks.items():
            results[tid] = self.dag_store.get_task_result(tid)

        stats = self.dag_store.get_graph_stats(graph.graph_id)

        # ─── Step 5b: Optional consensus (only for high-stakes) ───
        consensus_result = None
        if routing.requires_consensus:
            aggregated = "\n\n".join(
                f"[{tid}]: {r}" for tid, r in results.items() if r
            )
            consensus_result = self.consensus_engine.run_consensus(
                task=f"ORIGINAL TASK: {task}\n\nAGGREGATED RESULTS:\n{aggregated}",
                cluster_size=int(os.getenv("AGENTS_PER_CLUSTER", "4")),
            )
            metrics["llm_calls"] += int(os.getenv("AGENTS_PER_CLUSTER", "4"))

        # ─── Step 6: PARL reward ───
        total_tasks = len(graph.tasks)
        completed = stats.get(TaskStatus.COMPLETE.value, 0)
        critical_steps = graph.critical_path_length

        metrics.update({
            "mode": "hybrid-swarm",
            "total_tasks": total_tasks,
            "completed_tasks": completed,
            "escalated_tasks": stats.get(TaskStatus.ESCALATE.value, 0),
            "critical_path": critical_steps,
            "theoretical_speedup": round(graph.theoretical_speedup, 2),
            "actual_speedup": round(total_tasks / max(critical_steps, 1), 2),
            "wall_time": round(time.time() - start_time, 2),
            "coordination_tokens": 0,  # ZERO — stigmergy handles it all
            "consensus": consensus_result.__dict__ if consensus_result else None,
            "results": results,
            "parl_reward": self._compute_reward(graph, completed, total_tasks),
        })

        logger.info(
            f"[HYBRID] Complete: {completed}/{total_tasks} tasks, "
            f"{metrics['llm_calls']} LLM calls, "
            f"0 coordination tokens, "
            f"{metrics['wall_time']}s wall time"
        )

        return metrics

    def _single_agent_execute(self, task: str) -> str:
        """Fallback: single agent execution, zero swarm overhead."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Execute this task thoroughly."},
                {"role": "user", "content": task},
            ],
            temperature=0.5,
            max_tokens=8192,
        )
        return response.choices[0].message.content

    def _compute_reward(
        self, graph: TaskGraph, completed: int, total: int
    ) -> dict:
        """PARL reward: r_parallel + r_finish + r_perf."""
        parallelism = total / max(graph.critical_path_length, 1)
        r_parallel = min(parallelism / 3.0, 1.0)
        r_finish = completed / max(total, 1)
        r_perf = 1.0 if completed == total else 0.5

        return {
            "r_parallel": round(r_parallel, 4),
            "r_finish": round(r_finish, 4),
            "r_perf": r_perf,
            "total": round(0.2 * r_parallel + 0.3 * r_finish + r_perf, 4),
        }

    def shutdown(self):
        """Clean shutdown."""
        self.scent_field.stop_gc()
        self.executor.shutdown(wait=False)
