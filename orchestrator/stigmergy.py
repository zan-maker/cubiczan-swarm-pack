"""
Stigmergy Module — Zero-Token Coordination via Scent Signals

Adapted from TEMM1E v3.0.0 (MIT License, https://github.com/nagisanzenin/temm1e)
Ported from Rust (temm1e-hive) to Python for Cubiczan integration.

Key insight: Multi-agent coordination via LLM chat is an architecture bug.
Ant colony optimization solves NP-hard routing with ZERO centralized control.
Scent signals (exponential-decay pheromones) replace all inter-agent LLM calls.

Result: 5.86x faster, 3.4x cheaper, identical quality. Zero coordination tokens.

Traditional multi-agent cost (quadratic):
    C_single = m·(S+T) + h̄·m(m+1)/2

Stigmergic cost (linear):
    C_pack = C_alpha + m·(S + R̄)

Where h̄·m(m+1)/2 is eliminated entirely — no inter-agent conversations.
"""

import math
import time
import threading
import sqlite3
import json
import uuid
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger("cubiczan.stigmergy")


# =====================================================
# SCENT SIGNAL TYPES
# =====================================================

class ScentType(str, Enum):
    """Six pheromone signal types with distinct decay rates."""
    COMPLETION = "completion"       # Task finished (~5 min half-life)
    FAILURE = "failure"             # Attempt failed (~6 min half-life)
    DIFFICULTY = "difficulty"       # Worker struggling (~2 min half-life)
    URGENCY = "urgency"             # Grows over time, capped at 5.0
    PROGRESS = "progress"           # Worker heartbeat (~20 sec half-life)
    HELP_WANTED = "help_wanted"     # Specialist needed (~2 min half-life)


# Half-lives in seconds for exponential decay
SCENT_HALF_LIVES: dict[ScentType, float] = {
    ScentType.COMPLETION: 300.0,    # 5 minutes
    ScentType.FAILURE: 360.0,       # 6 minutes
    ScentType.DIFFICULTY: 120.0,    # 2 minutes
    ScentType.URGENCY: -1.0,        # Special: grows, doesn't decay
    ScentType.PROGRESS: 20.0,       # 20 seconds
    ScentType.HELP_WANTED: 120.0,   # 2 minutes
}

# Decay constant: λ = ln(2) / half_life
DECAY_CONSTANTS: dict[ScentType, float] = {
    st: (math.log(2) / hl if hl > 0 else 0.0)
    for st, hl in SCENT_HALF_LIVES.items()
}

# Garbage collection threshold — signals below this are purged
GC_THRESHOLD = 0.01
GC_INTERVAL = 10.0  # seconds


# =====================================================
# SCENT SIGNAL
# =====================================================

@dataclass
class ScentSignal:
    """A single pheromone emission from a worker."""
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    task_id: str = ""
    worker_id: str = ""
    scent_type: ScentType = ScentType.PROGRESS
    intensity: float = 1.0
    emitted_at: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    @property
    def current_intensity(self) -> float:
        """Calculate decayed intensity at current time."""
        if self.scent_type == ScentType.URGENCY:
            # Urgency GROWS over time, capped at 5.0
            elapsed = time.time() - self.emitted_at
            return min(self.intensity + (elapsed / 60.0), 5.0)

        decay_constant = DECAY_CONSTANTS.get(self.scent_type, 0.01)
        elapsed = time.time() - self.emitted_at
        return self.intensity * math.exp(-decay_constant * elapsed)

    @property
    def is_expired(self) -> bool:
        """Check if signal has decayed below GC threshold."""
        if self.scent_type == ScentType.URGENCY:
            return False  # Urgency never expires
        return self.current_intensity < GC_THRESHOLD


# =====================================================
# SCENT FIELD (Pheromone Environment)
# =====================================================

class ScentField:
    """
    The shared pheromone environment that all workers read/write.

    This replaces ALL inter-agent LLM communication.
    Workers emit scent signals as they work; other workers read these
    signals to choose their next task. Pure arithmetic, zero LLM calls.

    Uses SQLite for atomic signal storage (same as TEMM1E).
    """

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()
        self._gc_thread: threading.Thread | None = None
        self._running = False

    def _init_db(self):
        """Initialize SQLite storage for scent signals."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scent_signals (
                signal_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                worker_id TEXT NOT NULL,
                scent_type TEXT NOT NULL,
                intensity REAL NOT NULL,
                emitted_at REAL NOT NULL,
                metadata TEXT DEFAULT '{}'
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_task_id ON scent_signals(task_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_scent_type ON scent_signals(scent_type)
        """)
        conn.commit()
        conn.close()

    def emit(self, signal: ScentSignal):
        """Emit a scent signal into the field."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """INSERT OR REPLACE INTO scent_signals
               (signal_id, task_id, worker_id, scent_type, intensity, emitted_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                signal.signal_id,
                signal.task_id,
                signal.worker_id,
                signal.scent_type.value,
                signal.intensity,
                signal.emitted_at,
                json.dumps(signal.metadata),
            ),
        )
        conn.commit()
        conn.close()
        logger.debug(
            f"[SCENT] {signal.worker_id} emitted {signal.scent_type.value} "
            f"on task {signal.task_id} (intensity={signal.intensity:.2f})"
        )

    def read(self, task_id: str, scent_type: ScentType | None = None) -> float:
        """
        Read aggregated scent intensity for a task.
        Returns the sum of all current (decayed) intensities for matching signals.
        This is pure arithmetic — ZERO LLM calls.
        """
        conn = sqlite3.connect(self.db_path)
        if scent_type:
            rows = conn.execute(
                "SELECT intensity, emitted_at, scent_type FROM scent_signals "
                "WHERE task_id = ? AND scent_type = ?",
                (task_id, scent_type.value),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT intensity, emitted_at, scent_type FROM scent_signals "
                "WHERE task_id = ?",
                (task_id,),
            ).fetchall()
        conn.close()

        total = 0.0
        now = time.time()
        for intensity, emitted_at, st in rows:
            st_enum = ScentType(st)
            if st_enum == ScentType.URGENCY:
                elapsed = now - emitted_at
                total += min(intensity + (elapsed / 60.0), 5.0)
            else:
                decay = DECAY_CONSTANTS.get(st_enum, 0.01)
                total += intensity * math.exp(-decay * (now - emitted_at))
        return total

    def read_all_for_task(self, task_id: str) -> dict[ScentType, float]:
        """Read all scent types for a task as a dictionary."""
        result = {}
        for st in ScentType:
            val = self.read(task_id, st)
            if val > GC_THRESHOLD or st == ScentType.URGENCY:
                result[st] = val
        return result

    def garbage_collect(self):
        """Purge expired signals (below GC_THRESHOLD). Called every 10 seconds."""
        conn = sqlite3.connect(self.db_path)
        now = time.time()
        # Fetch all non-urgency signals and check decay
        rows = conn.execute(
            "SELECT signal_id, intensity, emitted_at, scent_type "
            "FROM scent_signals WHERE scent_type != ?",
            (ScentType.URGENCY.value,),
        ).fetchall()

        expired_ids = []
        for signal_id, intensity, emitted_at, st in rows:
            decay = DECAY_CONSTANTS.get(ScentType(st), 0.01)
            current = intensity * math.exp(-decay * (now - emitted_at))
            if current < GC_THRESHOLD:
                expired_ids.append(signal_id)

        if expired_ids:
            placeholders = ",".join("?" * len(expired_ids))
            conn.execute(
                f"DELETE FROM scent_signals WHERE signal_id IN ({placeholders})",
                expired_ids,
            )
            conn.commit()
            logger.debug(f"[SCENT-GC] Purged {len(expired_ids)} expired signals")
        conn.close()

    def start_gc(self):
        """Start background garbage collection thread."""
        self._running = True

        def _gc_loop():
            while self._running:
                time.sleep(GC_INTERVAL)
                try:
                    self.garbage_collect()
                except Exception as e:
                    logger.error(f"[SCENT-GC] Error: {e}")

        self._gc_thread = threading.Thread(target=_gc_loop, daemon=True)
        self._gc_thread.start()

    def stop_gc(self):
        """Stop background garbage collection."""
        self._running = False

    def clear(self):
        """Clear all signals (for testing)."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM scent_signals")
        conn.commit()
        conn.close()


# =====================================================
# TASK SELECTION SCORING (40 lines of arithmetic)
# =====================================================

def jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    """Jaccard similarity between two tag sets."""
    if not set_a and not set_b:
        return 0.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


def compute_task_score(
    worker_tags: set[str],
    task_tags: set[str],
    scent_field: ScentField,
    task_id: str,
    total_tasks: int,
    num_dependents: int,
) -> float:
    """
    TEMM1E task selection scoring — pure arithmetic, zero LLM calls.

    S(worker, task) = A^α · U^β · (1-D)^γ · (1-F)^δ · R^ζ

    Exponents (from TEMM1E research paper):
        α = 2.0  (affinity weight — strongest, encourages specialization)
        β = 1.5  (urgency weight — prevents starvation)
        γ = 1.0  (difficulty penalty)
        δ = 0.8  (failure penalty)
        ζ = 1.2  (downstream reward — prioritizes high-impact work)
    """
    # A: Affinity — Jaccard similarity between worker skills and task tags
    affinity = jaccard_similarity(worker_tags, task_tags)
    affinity = max(affinity, 0.1)  # Floor to prevent zero scores

    # U: Urgency — read from scent field
    urgency = scent_field.read(task_id, ScentType.URGENCY)
    urgency = max(urgency, 0.1)

    # D: Difficulty — read from scent field, clamp to [0, 1)
    difficulty = min(scent_field.read(task_id, ScentType.DIFFICULTY), 0.99)

    # F: Failure — read from scent field, clamp to [0, 1)
    failure = min(scent_field.read(task_id, ScentType.FAILURE), 0.99)

    # R: Downstream reward — tasks with more dependents are higher value
    reward = 1.0 + (num_dependents / max(total_tasks, 1))

    # Apply TEMM1E scoring formula with tuned exponents
    score = (
        (affinity ** 2.0)
        * (urgency ** 1.5)
        * ((1.0 - difficulty) ** 1.0)
        * ((1.0 - failure) ** 0.8)
        * (reward ** 1.2)
    )

    return score


def select_best_task(
    worker_id: str,
    worker_tags: set[str],
    available_task_ids: list[str],
    task_tags_map: dict[str, set[str]],
    task_dependents_map: dict[str, int],
    scent_field: ScentField,
    tie_threshold: float = 0.05,
) -> str | None:
    """
    Select the best task for a worker using scent-based scoring.
    Tie-breaking: scores within 5% of max trigger random selection
    to prevent pathological herding.
    """
    import random

    if not available_task_ids:
        return None

    total_tasks = len(available_task_ids)
    scored = []

    for task_id in available_task_ids:
        tags = task_tags_map.get(task_id, set())
        deps = task_dependents_map.get(task_id, 0)
        score = compute_task_score(
            worker_tags, tags, scent_field, task_id, total_tasks, deps
        )
        scored.append((task_id, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    if not scored:
        return None

    max_score = scored[0][1]
    if max_score == 0:
        return scored[0][0]

    # Tie-breaking: tasks within 5% of max
    threshold = max_score * (1.0 - tie_threshold)
    tied = [tid for tid, s in scored if s >= threshold]

    return random.choice(tied)
