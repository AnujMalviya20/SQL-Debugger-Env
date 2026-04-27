"""
SQL Debugger & Optimizer Environment — Main Environment Class
=============================================================
Implements the OpenEnv interface: reset(), step(), state property.

Design
------
* Pure SQLite in-memory execution — no external DB required.
* Fully deterministic: same task + same queries → same rewards.
* Concurrent-session safe: each reset() creates isolated episode state.
* Rich reward shaping: partial credit throughout trajectory.
* Penalises destructive SQL and step-limit overruns.
"""

from __future__ import annotations

import os
import uuid
import sqlite3
from typing import Any, Dict, Optional

from openenv.core import Environment

from models import SQLAction, SQLObservation, SQLState
from tasks import Task, get_task, list_tasks, _has_destructive, _run_query


# ---------------------------------------------------------------------------
# Reward constants  (must sum to ≤ 1.0 when all positive)
# ---------------------------------------------------------------------------
R_NO_DESTRUCTIVE   = 0.10   # no destructive operations
R_RUNS             = 0.15   # query executes without error
R_NONEMPTY         = 0.10   # result is non-empty
R_COLUMNS          = 0.15   # all expected columns present
R_ROW_COUNT        = 0.15   # correct number of rows
R_CONTENT_MAX      = 0.30   # full Jaccard similarity bonus
R_EFFICIENCY       = 0.05   # JOIN-based approach (medium/hard)

STEP_PENALTY       = 0.02   
DESTRUCTIVE_PENALTY= 0.10   # applied when destructive SQL detected
MAX_EPISODE_STEPS  = 12     # absolute ceiling across all tasks


class SQLDebuggerEnv(Environment[SQLAction, SQLObservation, SQLState]):
    """
    RL environment simulating the real-world task of SQL debugging and optimisation.

    An agent is presented with a broken or inefficient SQL query plus the
    database schema.  It iteratively submits revised queries and receives
    structured feedback until it produces the correct result set or exhausts
    its step budget.

    Supports concurrent sessions (each WebSocket connection gets its own
    isolated episode state).
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._task: Optional[Task] = None
        self._state: Optional[SQLState] = None

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SQLObservation:
        """
        Initialise a new episode.

        Args:
            seed      : Ignored (tasks are deterministic).
            episode_id: Optional custom episode identifier.
            task_id   : One of "easy", "medium", "hard".
                        Defaults to "easy" if not specified.

        Returns:
            Initial SQLObservation with task description, schema, and
            the broken starting query as the current_query.
        """
        # Resolve task
        if task_id is None:
            task_id = kwargs.get("task", "easy")
        if task_id not in list_tasks():
            task_id = "easy"

        self._task = get_task(task_id)
        ep_id = episode_id or str(uuid.uuid4())

    
        episodes_dir = os.path.join(os.path.dirname(__file__), "server", "episodes")
        os.makedirs(episodes_dir, exist_ok=True)
        db_path = os.path.join(episodes_dir, f"{ep_id}.db")

        # initialise persistent DB for this episode
        conn = sqlite3.connect(db_path)
        conn.executescript(self._task.db_schema)
        conn.close()

        self._state = SQLState(
            episode_id=ep_id,
            step_count=0,
            task_id=task_id,
            task_description=self._task.description,
            db_schema=self._task.db_schema,
            expected_columns=self._task.expected_columns,
            expected_rows=self._task.expected_rows,
            max_steps=self._task.max_steps,
            is_correct=False,
            best_partial_score=0.0,
            destructive_penalty_applied=False,
        )

        # store DB path on the instance for step() to use
        self._db_path = db_path

        return SQLObservation(
            done=False,
            reward=0.0,
            task_description=self._task.description,
            db_schema=self._task.db_schema,
            current_query=self._task.starting_query.strip(),
            error_message=None,
            query_result=None,
            expected_columns=self._task.expected_columns,
            feedback=(
                "Episode started. Read the task description and database schema, "
                "then submit a corrected SQL query."
            ),
            step_count=0,
            max_steps=self._task.max_steps,
            is_correct=False,
            partial_score=0.0,
        )

    def step(
        self,
        action: SQLAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> SQLObservation:
        """
        Execute one agent step: evaluate the submitted SQL query.

        Args:
            action: SQLAction containing the agent's sql_query.

        Returns:
            SQLObservation with feedback, reward, and done flag.
        """
        if self._state is None or self._task is None:
            raise RuntimeError("Call reset() before step().")

        self._state.step_count += 1
        sql = action.sql_query.strip()

        # --- Destructive query check ---
        destructive = _has_destructive(sql)
        destructive_penalty = 0.0
        if destructive and not self._state.destructive_penalty_applied:
            destructive_penalty = -DESTRUCTIVE_PENALTY
            self._state.destructive_penalty_applied = True

        # --- Run the query against the persistent episode DB ---
        rows, error = _run_query(sql, db_path=getattr(self, "_db_path", None))

        # --- Grade via task's deterministic grader ---
        raw_score, feedback = self._task.grade(sql)

        # Apply step penalty (encourages efficiency)
        step_pen = STEP_PENALTY * (self._state.step_count - 1)
        shaped_reward = max(0.0, round(raw_score - step_pen + destructive_penalty, 4))

        # Track best score this episode
        if raw_score > self._state.best_partial_score:
            self._state.best_partial_score = raw_score

        # Episode complete?
        is_correct = (raw_score >= 0.95)
        self._state.is_correct = is_correct
        done = is_correct or (self._state.step_count >= self._task.max_steps)

        # Add concise completion messages
        if done and not is_correct:
            feedback += (
                f" | Episode ended after {self._state.step_count} steps. "
                f"Best score achieved: {self._state.best_partial_score:.2f}."
            )
        elif is_correct:
            feedback += (
                f" | Task solved in {self._state.step_count} step(s). "
                f"Final score: {shaped_reward:.4f}."
            )

        return SQLObservation(
            done=done,
            reward=shaped_reward,
            task_description=self._task.description,
            db_schema=self._task.db_schema,
            current_query=sql,
            error_message=error,
            query_result=rows[:20] if rows else None,  # cap at 20 rows for payload size
            expected_columns=self._task.expected_columns,
            feedback=feedback,
            step_count=self._state.step_count,
            max_steps=self._task.max_steps,
            is_correct=is_correct,
            partial_score=raw_score,
        )

    @property
    def state(self) -> SQLState:
        """Return current episode state (server-side metadata)."""
        if self._state is None:
            raise RuntimeError("Call reset() before accessing state.")
        return self._state
