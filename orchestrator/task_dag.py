"""
Task DAG — Dependency graph with atomic SQLite task claiming.

Adapted from TEMM1E v3.0.0 (MIT License).

State machine:
    PENDING → READY → ACTIVE → COMPLETE
      ├→ BLOCKED → RETRY → READY
      └→ ESCALATE (max retries exceeded)

Atomicity: SQLite's write serialization provides mutual exclusion.
Task claiming is a single UPDATE...WHERE status='ready' — succeeds only
if no other worker claimed it first. Zero distributed locks.
"""

import json
import sqlite3
import time
import uuid
import logging
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

logger = logging.getLogger("cubiczan.task_dag")


class TaskStatus(str, Enum):
    PENDING = "pending"     # Created but dependencies not met
    READY = "ready"         # All dependencies complete, available for claiming
    ACTIVE = "active"       # Claimed by a worker, in progress
    COMPLETE = "complete"   # Successfully finished
    BLOCKED = "blocked"     # Hit a retryable error
    RETRY = "retry"         # Scheduled for retry → transitions to READY
    ESCALATE = "escalate"   # Max retries exceeded, needs human/orchestrator


MAX_RETRIES = 3


@dataclass
class DAGTask:
    """A single task node in the dependency graph."""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str = ""
    agent_type: str = "analyst"  # researcher, analyst, validator, synthesizer, contrarian
    tags: set[str] = field(default_factory=set)
    dependencies: list[str] = field(default_factory=list)  # task_ids this depends on
    dependents: list[str] = field(default_factory=list)     # task_ids that depend on this
    status: TaskStatus = TaskStatus.PENDING
    worker_id: str | None = None
    result: str = ""
    retries: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    completed_at: float = 0.0

    @property
    def duration(self) -> float:
        if self.completed_at and self.started_at:
            return self.completed_at - self.started_at
        return 0.0


@dataclass
class TaskGraph:
    """Complete dependency graph for a decomposed task."""
    graph_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    tasks: dict[str, DAGTask] = field(default_factory=dict)
    critical_path_length: int = 0
    theoretical_speedup: float = 1.0


class TaskDAGStore:
    """
    SQLite-backed task DAG with atomic claiming.

    Key design: Workers claim tasks via atomic UPDATE...WHERE status='ready'.
    SQLite write serialization = mutual exclusion without distributed locks.
    """

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS dag_tasks (
                task_id TEXT PRIMARY KEY,
                graph_id TEXT NOT NULL,
                description TEXT NOT NULL,
                agent_type TEXT DEFAULT 'analyst',
                tags TEXT DEFAULT '[]',
                dependencies TEXT DEFAULT '[]',
                dependents TEXT DEFAULT '[]',
                status TEXT DEFAULT 'pending',
                worker_id TEXT,
                result TEXT DEFAULT '',
                retries INTEGER DEFAULT 0,
                created_at REAL,
                started_at REAL DEFAULT 0,
                completed_at REAL DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_graph_status
            ON dag_tasks(graph_id, status)
        """)
        conn.commit()
        conn.close()

    def store_graph(self, graph: TaskGraph):
        """Store a complete task graph."""
        conn = sqlite3.connect(self.db_path)
        for task in graph.tasks.values():
            conn.execute(
                """INSERT OR REPLACE INTO dag_tasks
                   (task_id, graph_id, description, agent_type, tags,
                    dependencies, dependents, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    task.task_id,
                    graph.graph_id,
                    task.description,
                    task.agent_type,
                    json.dumps(list(task.tags)),
                    json.dumps(task.dependencies),
                    json.dumps(task.dependents),
                    task.status.value,
                    task.created_at,
                ),
            )
        conn.commit()
        conn.close()

    def activate_ready_tasks(self, graph_id: str):
        """
        Transition PENDING tasks to READY when all dependencies are COMPLETE.
        Also transitions RETRY tasks back to READY.
        """
        conn = sqlite3.connect(self.db_path)

        # Get all tasks in this graph
        rows = conn.execute(
            "SELECT task_id, dependencies, status FROM dag_tasks WHERE graph_id = ?",
            (graph_id,),
        ).fetchall()

        # Get completed task IDs
        completed = {
            r[0] for r in rows if r[2] == TaskStatus.COMPLETE.value
        }

        for task_id, deps_json, status in rows:
            deps = json.loads(deps_json)

            # PENDING → READY if all dependencies complete
            if status == TaskStatus.PENDING.value:
                if not deps or all(d in completed for d in deps):
                    conn.execute(
                        "UPDATE dag_tasks SET status = ? WHERE task_id = ?",
                        (TaskStatus.READY.value, task_id),
                    )

            # RETRY → READY
            elif status == TaskStatus.RETRY.value:
                conn.execute(
                    "UPDATE dag_tasks SET status = ? WHERE task_id = ?",
                    (TaskStatus.READY.value, task_id),
                )

        conn.commit()
        conn.close()

    def claim_task(self, graph_id: str, worker_id: str) -> DAGTask | None:
        """
        Atomically claim a READY task for a worker.

        This is the core TEMM1E mechanism:
        UPDATE ... WHERE status = 'ready' — succeeds only if unclaimed.
        SQLite write serialization = mutual exclusion without distributed locks.
        """
        conn = sqlite3.connect(self.db_path)
        conn.execute("BEGIN EXCLUSIVE")

        try:
            # Find first READY task
            row = conn.execute(
                "SELECT task_id, description, agent_type, tags, dependencies, "
                "dependents, retries, created_at "
                "FROM dag_tasks WHERE graph_id = ? AND status = ? LIMIT 1",
                (graph_id, TaskStatus.READY.value),
            ).fetchone()

            if not row:
                conn.rollback()
                return None

            task_id = row[0]
            now = time.time()

            # Atomic claim: READY → ACTIVE
            conn.execute(
                "UPDATE dag_tasks SET status = ?, worker_id = ?, started_at = ? "
                "WHERE task_id = ? AND status = ?",
                (TaskStatus.ACTIVE.value, worker_id, now, task_id, TaskStatus.READY.value),
            )
            conn.commit()

            task = DAGTask(
                task_id=task_id,
                description=row[1],
                agent_type=row[2],
                tags=set(json.loads(row[3])),
                dependencies=json.loads(row[4]),
                dependents=json.loads(row[5]),
                status=TaskStatus.ACTIVE,
                worker_id=worker_id,
                retries=row[6],
                created_at=row[7],
                started_at=now,
            )
            logger.info(f"[DAG] Worker {worker_id} claimed task {task_id}")
            return task

        except Exception as e:
            conn.rollback()
            logger.error(f"[DAG] Claim failed: {e}")
            return None
        finally:
            conn.close()

    def claim_specific_task(
        self, task_id: str, worker_id: str
    ) -> bool:
        """Atomically claim a specific task (used with scent-based selection)."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "UPDATE dag_tasks SET status = ?, worker_id = ?, started_at = ? "
                "WHERE task_id = ? AND status = ?",
                (TaskStatus.ACTIVE.value, worker_id, time.time(),
                 task_id, TaskStatus.READY.value),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def complete_task(self, task_id: str, result: str):
        """Mark a task as COMPLETE with its result."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "UPDATE dag_tasks SET status = ?, result = ?, completed_at = ? "
            "WHERE task_id = ?",
            (TaskStatus.COMPLETE.value, result, time.time(), task_id),
        )
        conn.commit()
        conn.close()
        logger.info(f"[DAG] Task {task_id} completed")

    def fail_task(self, task_id: str, error: str):
        """Handle task failure with retry logic."""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT retries FROM dag_tasks WHERE task_id = ?", (task_id,)
        ).fetchone()

        if row and row[0] < MAX_RETRIES:
            conn.execute(
                "UPDATE dag_tasks SET status = ?, retries = retries + 1, "
                "result = ?, worker_id = NULL "
                "WHERE task_id = ?",
                (TaskStatus.RETRY.value, f"RETRY: {error}", task_id),
            )
            logger.warning(f"[DAG] Task {task_id} failed, retry {row[0]+1}/{MAX_RETRIES}")
        else:
            conn.execute(
                "UPDATE dag_tasks SET status = ?, result = ? WHERE task_id = ?",
                (TaskStatus.ESCALATE.value, f"ESCALATED: {error}", task_id),
            )
            logger.error(f"[DAG] Task {task_id} escalated after {MAX_RETRIES} retries")

        conn.commit()
        conn.close()

    def get_ready_tasks(self, graph_id: str) -> list[str]:
        """Get all READY task IDs."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT task_id FROM dag_tasks WHERE graph_id = ? AND status = ?",
            (graph_id, TaskStatus.READY.value),
        ).fetchall()
        conn.close()
        return [r[0] for r in rows]

    def get_task_result(self, task_id: str) -> str:
        """Get a completed task's result (for dependency resolution)."""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT result FROM dag_tasks WHERE task_id = ? AND status = ?",
            (task_id, TaskStatus.COMPLETE.value),
        ).fetchone()
        conn.close()
        return row[0] if row else ""

    def is_graph_complete(self, graph_id: str) -> bool:
        """Check if all tasks in a graph are COMPLETE or ESCALATED."""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT COUNT(*) FROM dag_tasks WHERE graph_id = ? "
            "AND status NOT IN (?, ?)",
            (graph_id, TaskStatus.COMPLETE.value, TaskStatus.ESCALATE.value),
        ).fetchone()
        conn.close()
        return row[0] == 0

    def get_graph_stats(self, graph_id: str) -> dict:
        """Get completion statistics for a graph."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT status, COUNT(*) FROM dag_tasks WHERE graph_id = ? "
            "GROUP BY status",
            (graph_id,),
        ).fetchall()
        conn.close()
        return {r[0]: r[1] for r in rows}


def validate_dag(tasks: dict[str, DAGTask]) -> bool:
    """
    Validate DAG has no cycles using Kahn's topological sort.
    Rejects cyclic graphs (falls back to single-agent mode per TEMM1E).
    """
    in_degree: dict[str, int] = {tid: 0 for tid in tasks}
    adj: dict[str, list[str]] = {tid: [] for tid in tasks}

    for tid, task in tasks.items():
        for dep_id in task.dependents:
            if dep_id in adj:
                adj[tid].append(dep_id)
                in_degree[dep_id] += 1

    queue = deque(tid for tid, deg in in_degree.items() if deg == 0)
    visited = 0

    while queue:
        node = queue.popleft()
        visited += 1
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if visited != len(tasks):
        logger.error("[DAG] Cycle detected! Falling back to single-agent mode.")
        return False
    return True


def compute_critical_path(tasks: dict[str, DAGTask]) -> int:
    """Compute critical path length for speedup estimation."""
    # Topological sort + longest path
    in_degree: dict[str, int] = {tid: len(t.dependencies) for tid, t in tasks.items()}
    dist: dict[str, int] = {tid: 1 for tid in tasks}
    queue = deque(tid for tid, deg in in_degree.items() if deg == 0)

    while queue:
        node = queue.popleft()
        task = tasks[node]
        for dep_id in task.dependents:
            if dep_id in dist:
                dist[dep_id] = max(dist[dep_id], dist[node] + 1)
                in_degree[dep_id] -= 1
                if in_degree[dep_id] == 0:
                    queue.append(dep_id)

    return max(dist.values()) if dist else 1
