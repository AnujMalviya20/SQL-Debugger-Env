"""
SQL Debugger & Optimizer Environment — Client
=============================================
Typed EnvClient subclass. Users install this alongside the running
server (on HF Spaces or locally) and interact with it from their
training code.

Usage (async):
    from sql_debugger_env import SQLAction, SQLDebuggerClient

    async with SQLDebuggerClient(base_url="https://your-space.hf.space") as env:
        result = await env.reset(task_id="medium")
        while not result.done:
            action = SQLAction(sql_query="SELECT ...")
            result = await env.step(action)
            print(result.observation.feedback)

Usage (sync):
    from sql_debugger_env import SQLAction, SQLDebuggerClient

    with SQLDebuggerClient(base_url="https://your-space.hf.space").sync() as env:
        result = env.reset(task_id="easy")
        result = env.step(SQLAction(sql_query="SELECT full_name, salary FROM employees WHERE dept='Engineering' ORDER BY salary DESC;"))
        print(result.observation.is_correct)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from openenv.core import EnvClient
from openenv.core.env_client import StepResult

from models import SQLAction, SQLObservation, SQLState


class SQLDebuggerClient(EnvClient[SQLAction, SQLObservation, SQLState]):
    """
    Typed async client for the SQL Debugger & Optimizer environment.

    Connects to the running FastAPI server via WebSocket and exposes
    the standard reset() / step() / state interface with full type safety.
    """

    def _step_payload(self, action: SQLAction) -> Dict[str, Any]:
        """Serialise SQLAction → dict for the wire format."""
        return {
            "sql_query": action.sql_query,
            "metadata":  action.metadata,
        }

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[SQLObservation]:
        """Deserialise server response → StepResult[SQLObservation]."""
        obs = SQLObservation(
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            task_description=payload.get("task_description", ""),
            db_schema=payload.get("db_schema", ""),
            current_query=payload.get("current_query"),
            error_message=payload.get("error_message"),
            query_result=payload.get("query_result"),
            expected_columns=payload.get("expected_columns", []),
            feedback=payload.get("feedback", ""),
            step_count=payload.get("step_count", 0),
            max_steps=payload.get("max_steps", 10),
            is_correct=payload.get("is_correct", False),
            partial_score=payload.get("partial_score", 0.0),
        )
        return StepResult(
            observation=obs,
            reward=obs.reward,
            done=obs.done,
        )

    def _parse_state(self, payload: Dict[str, Any]) -> SQLState:
        """Deserialise server state response → SQLState."""
        return SQLState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            task_description=payload.get("task_description", ""),
            db_schema=payload.get("db_schema", ""),
            expected_columns=payload.get("expected_columns", []),
            expected_rows=payload.get("expected_rows", []),
            max_steps=payload.get("max_steps", 10),
            is_correct=payload.get("is_correct", False),
            best_partial_score=payload.get("best_partial_score", 0.0),
            destructive_penalty_applied=payload.get("destructive_penalty_applied", False),
        )
