"""
SQL Debugger & Optimizer Environment - Pydantic Models
======================================================
Defines the typed contracts: Action, Observation, and State.
These are the 'language' spoken between agent and environment.
"""

from typing import Any, Dict, List, Optional
from pydantic import Field
from openenv.core import Action, Observation, State


# ---------------------------------------------------------------------------
# ACTION
# ---------------------------------------------------------------------------

class SQLAction(Action):
    """
    The agent's action: submit a SQL query string.

    The agent reads the task description + schema + feedback, then
    crafts a SQL query it believes solves the problem. Each call to
    step() consumes one action and advances the episode.
    """

    sql_query: str = Field(
        ...,
        description="The SQL query submitted by the agent. Must be a valid SQLite SQL statement.",
        min_length=1,
        max_length=4096,
    )


# ---------------------------------------------------------------------------
# OBSERVATION
# ---------------------------------------------------------------------------

class SQLObservation(Observation):
    """
    What the agent sees after each reset() or step().

    Fields:
        task_description : Natural-language description of what must be fixed/optimized.
        db_schema        : The full DDL (CREATE TABLE ...) statements for context.
        current_query    : The SQL query the agent just submitted (echoed back for logging).
        error_message    : If the query failed, the SQLite error text; else None.
        query_result     : Rows returned by a successful query (list of row dicts); else None.
        expected_columns : Column names that must appear in the final result.
        feedback         : Human-readable feedback explaining what was right/wrong.
        step_count       : How many steps have elapsed in this episode.
        max_steps        : Maximum steps allowed before episode forced-terminates.
        is_correct       : True if the agent has fully solved the task.
        partial_score    : Running partial-credit score in [0.0, 1.0].

    Inherited from Observation (openenv.core):
        done   : bool   – episode finished (success or max-steps reached)
        reward : float  – reward for this step
    """

    task_description: str = Field(
        ...,
        description="Natural-language description of the SQL problem to solve.",
    )
    episode_id: Optional[str] = Field(
        default=None,
        description="Identifier for the active episode (if provided by the server).",
    )
    db_schema: str = Field(
        ...,
        description="DDL statements that define all tables available to the agent.",
    )
    current_query: Optional[str] = Field(
        default=None,
        description="The SQL query the agent just submitted.",
    )
    error_message: Optional[str] = Field(
        default=None,
        description="SQLite error message if the query raised an exception.",
    )
    query_result: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Rows returned by the query, each row as a dict {column: value}.",
    )
    expected_columns: List[str] = Field(
        default_factory=list,
        description="Column names the correct result set must contain.",
    )
    feedback: str = Field(
        default="",
        description="Step-level feedback explaining incremental progress or errors.",
    )
    step_count: int = Field(
        default=0,
        description="Number of steps taken so far in this episode.",
        ge=0,
    )
    max_steps: int = Field(
        default=10,
        description="Maximum number of steps allowed before forced termination.",
        ge=1,
    )
    is_correct: bool = Field(
        default=False,
        description="True when the agent's query fully satisfies the task requirements.",
    )
    partial_score: float = Field(
        default=0.0,
        description="Running partial-credit score in [0.0, 1.0].",
        ge=0.0,
        le=1.0,
    )


# ---------------------------------------------------------------------------
# STATE  (server-side, richer than observation)
# ---------------------------------------------------------------------------

class SQLState(State):
    """
    Full server-side episode state.

    Includes everything in the observation PLUS internal bookkeeping
    that should not be exposed to the agent (e.g. ground-truth answer).

    Inherited from State (openenv.core):
        episode_id : str  – unique identifier for this episode
        step_count : int  – steps taken
    """

    task_id: str = Field(
        default="",
        description="Identifier of the active task (easy / medium / hard).",
    )
    task_description: str = Field(default="")
    db_schema: str = Field(default="")
    expected_columns: List[str] = Field(default_factory=list)
    expected_rows: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Ground-truth result rows the agent must reproduce.",
    )
    max_steps: int = Field(default=10)
    is_correct: bool = Field(default=False)
    best_partial_score: float = Field(
        default=0.0,
        description="Best partial score achieved so far in this episode.",
    )
    destructive_penalty_applied: bool = Field(
        default=False,
        description="Whether a destructive-query penalty was applied this episode.",
    )
