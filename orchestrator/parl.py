"""
PARL — Parallel Agent Reinforcement Learning
Implements the Kimi K2.5 Agent Swarm architecture (arXiv:2602.02276).

Key principles from the paper:
1. Trainable orchestrator + frozen subagents (decoupled architecture)
2. Dynamic subagent instantiation — not pre-defined, learned via feedback
3. Critical Steps metric — measures wall-clock via critical path, not total steps
4. rPARL = λ1·r_parallel + λ2·r_finish + r_perf
5. 3-4.5x latency reduction via parallelism on wide/deep search tasks
"""

import json
import os
import time
import asyncio
import logging
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

logger = logging.getLogger("cubiczan.parl")


@dataclass
class SubAgentTask:
    """A subtask assigned to a frozen subagent."""
    task_id: str
    description: str
    agent_type: str  # e.g., "researcher", "analyst", "validator"
    status: str = "pending"  # pending, running, completed, failed
    result: str = ""
    steps_taken: int = 0
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration(self) -> float:
        if self.end_time > 0:
            return self.end_time - self.start_time
        return 0.0


@dataclass
class SwarmExecution:
    """Tracks a full agent swarm execution with critical path metrics."""
    execution_id: str
    stages: list[list[SubAgentTask]] = field(default_factory=list)
    total_critical_steps: int = 0
    total_wall_time: float = 0.0
    total_agent_steps: int = 0  # Sum of all agent steps (for comparison)

    @property
    def parallelism_ratio(self) -> float:
        """How much parallelism was achieved. >1.0 means speedup."""
        if self.total_critical_steps == 0:
            return 1.0
        return self.total_agent_steps / self.total_critical_steps


ORCHESTRATOR_PROMPT = """You are the ORCHESTRATOR of an Agent Swarm system.
Your job is to decompose complex tasks into parallelizable subtasks and
assign them to specialized subagents.

RULES:
1. Analyze the task and identify independent subtasks that can run in PARALLEL
2. Only create subtasks that are truly independent — no data dependencies between parallel tasks
3. Group dependent tasks into sequential stages
4. For each subtask, specify the agent type and a clear instruction
5. Minimize CRITICAL STEPS (wall-clock time), not total work

AGENT TYPES AVAILABLE:
- researcher: Searches for information, gathers data
- analyst: Analyzes data, creates models, generates insights
- validator: Cross-checks facts, verifies claims, identifies errors
- synthesizer: Combines results from multiple sources into coherent output
- contrarian: Challenges conclusions, finds counter-arguments

Respond in JSON:
{
  "stages": [
    {
      "stage_id": 1,
      "parallel_tasks": [
        {"task_id": "1a", "agent_type": "researcher", "instruction": "..."},
        {"task_id": "1b", "agent_type": "analyst", "instruction": "..."}
      ]
    },
    {
      "stage_id": 2,
      "parallel_tasks": [
        {"task_id": "2a", "agent_type": "synthesizer", "instruction": "Use results from stage 1 to..."}
      ]
    }
  ],
  "reasoning": "Why this decomposition is optimal"
}"""


class PARLOrchestrator:
    """
    Parallel Agent Reinforcement Learning orchestrator.

    Implements the decoupled architecture from Kimi K2.5:
    - Orchestrator is the "trainable" component (makes decomposition decisions)
    - Subagents are "frozen" (execute subtasks independently)
    - Critical Steps metric drives optimization
    """

    def __init__(self):
        self.orchestrator_client = OpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
        )
        self.orchestrator_model = os.getenv("LLM_MODEL_NAME", "qwen2.5:32b")

        # Subagent pool — heterogeneous models (frozen)
        self.subagent_models = self._load_subagent_models()
        self.executor = ThreadPoolExecutor(
            max_workers=int(os.getenv("AGENTS_PER_CLUSTER", "4")) * 2
        )

    def _load_subagent_models(self) -> list[dict]:
        models = []
        for i in range(1, 6):
            key = os.getenv(f"AGENT_MODEL_{i}_API_KEY")
            url = os.getenv(f"AGENT_MODEL_{i}_BASE_URL")
            name = os.getenv(f"AGENT_MODEL_{i}_NAME")
            if key and url and name:
                models.append({"api_key": key, "base_url": url, "model": name})
        if not models:
            models.append({
                "api_key": os.getenv("LLM_API_KEY"),
                "base_url": os.getenv("LLM_BASE_URL"),
                "model": os.getenv("LLM_MODEL_NAME", "qwen2.5:32b"),
            })
        return models

    def decompose(self, task: str) -> dict:
        """Use orchestrator to decompose task into parallel stages."""
        response = self.orchestrator_client.chat.completions.create(
            model=self.orchestrator_model,
            messages=[
                {"role": "system", "content": ORCHESTRATOR_PROMPT},
                {"role": "user", "content": task},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)

    def _execute_subtask(self, subtask: SubAgentTask, context: str = "") -> SubAgentTask:
        """Execute a single subtask using a frozen subagent."""
        model_cfg = self.subagent_models[
            hash(subtask.task_id) % len(self.subagent_models)
        ]
        client = OpenAI(api_key=model_cfg["api_key"], base_url=model_cfg["base_url"])

        subtask.status = "running"
        subtask.start_time = time.time()

        try:
            messages = [
                {"role": "system", "content": (
                    f"You are a {subtask.agent_type} subagent. "
                    "Execute the assigned task thoroughly and return your findings. "
                    "Be specific and evidence-based."
                )},
                {"role": "user", "content": (
                    f"TASK: {subtask.description}\n\n"
                    f"CONTEXT: {context}" if context else f"TASK: {subtask.description}"
                )},
            ]
            response = client.chat.completions.create(
                model=model_cfg["model"],
                messages=messages,
                temperature=0.5,
                max_tokens=4096,
            )
            subtask.result = response.choices[0].message.content
            subtask.status = "completed"
            subtask.steps_taken = 1
        except Exception as e:
            subtask.status = "failed"
            subtask.result = f"Error: {str(e)}"
            logger.error(f"Subtask {subtask.task_id} failed: {e}")

        subtask.end_time = time.time()
        return subtask

    def execute_swarm(self, task: str, execution_id: str = "swarm-1") -> SwarmExecution:
        """
        Full PARL execution pipeline:
        1. Orchestrator decomposes task into stages
        2. Each stage runs subtasks in parallel
        3. Results flow to next stage as context
        4. Critical Steps = sum of max(subtask_steps) per stage
        """
        execution = SwarmExecution(execution_id=execution_id)
        start_time = time.time()

        # Step 1: Decompose
        decomposition = self.decompose(task)
        logger.info(
            f"[{execution_id}] Decomposed into "
            f"{len(decomposition.get('stages', []))} stages"
        )

        accumulated_context = ""

        # Step 2: Execute stages sequentially, tasks within each stage in parallel
        for stage in decomposition.get("stages", []):
            stage_tasks = []
            futures = []

            for task_spec in stage.get("parallel_tasks", []):
                subtask = SubAgentTask(
                    task_id=task_spec["task_id"],
                    description=task_spec["instruction"],
                    agent_type=task_spec["agent_type"],
                )
                future = self.executor.submit(
                    self._execute_subtask, subtask, accumulated_context
                )
                futures.append((subtask, future))

            # Wait for all parallel tasks in this stage
            for subtask, future in futures:
                completed = future.result(timeout=300)
                stage_tasks.append(completed)

            execution.stages.append(stage_tasks)

            # Calculate critical steps for this stage (max steps among parallel tasks)
            if stage_tasks:
                max_steps = max(t.steps_taken for t in stage_tasks)
                execution.total_critical_steps += 1 + max_steps  # 1 for orchestrator
                execution.total_agent_steps += sum(t.steps_taken for t in stage_tasks)

            # Build context for next stage
            accumulated_context = "\n\n".join(
                f"[{t.task_id} ({t.agent_type})] {t.result}"
                for t in stage_tasks if t.status == "completed"
            )

        execution.total_wall_time = time.time() - start_time

        logger.info(
            f"[{execution_id}] Complete. "
            f"Critical steps: {execution.total_critical_steps}, "
            f"Total agent steps: {execution.total_agent_steps}, "
            f"Parallelism ratio: {execution.parallelism_ratio:.2f}x, "
            f"Wall time: {execution.total_wall_time:.1f}s"
        )

        return execution

    def compute_parl_reward(
        self,
        execution: SwarmExecution,
        task_success: bool,
        lambda1: float = 0.2,
        lambda2: float = 0.3,
    ) -> dict:
        """
        Compute PARL reward as defined in Kimi K2.5 paper:
        rPARL = λ1·r_parallel + λ2·r_finish + r_perf

        r_parallel: Rewards actual parallelism (prevents serial collapse)
        r_finish: Sub-agent completion rate (prevents over-decomposition)
        r_perf: Task-level outcome (primary objective)
        """
        # r_parallel: ratio of parallelism achieved
        r_parallel = min(execution.parallelism_ratio / 3.0, 1.0)  # Normalize to ~3x target

        # r_finish: completion rate of all subtasks
        all_tasks = [t for stage in execution.stages for t in stage]
        completed = sum(1 for t in all_tasks if t.status == "completed")
        r_finish = completed / len(all_tasks) if all_tasks else 0.0

        # r_perf: binary task success (in production, use verifiable reward)
        r_perf = 1.0 if task_success else 0.0

        total_reward = lambda1 * r_parallel + lambda2 * r_finish + r_perf

        return {
            "r_parallel": round(r_parallel, 4),
            "r_finish": round(r_finish, 4),
            "r_perf": r_perf,
            "total_reward": round(total_reward, 4),
            "lambda1": lambda1,
            "lambda2": lambda2,
            "parallelism_ratio": round(execution.parallelism_ratio, 2),
            "critical_steps": execution.total_critical_steps,
            "total_steps": execution.total_agent_steps,
        }
