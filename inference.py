"""
SQL Debugger & Optimizer — Baseline Inference Script
=====================================================
Evaluates an LLM agent against all three tasks using the OpenAI
Python client (compatible with any OpenAI-API-spec endpoint).

Environment variables (required):
  API_BASE_URL   — base URL of the LLM API  (e.g. https://api.openai.com/v1)
  MODEL_NAME     — model identifier          (e.g. gpt-4o-mini)
  HF_TOKEN       — Hugging Face token        (used to authenticate with the
                   environment server on HF Spaces; can be the same as
                   OPENAI_API_KEY for OpenAI endpoints)
  OPENAI_API_KEY — fallback API key if HF_TOKEN is not set
  ENV_BASE_URL   — base URL of the running environment server
                   (default: http://localhost:8000)

Usage:
  export API_BASE_URL=https://api.openai.com/v1
  export MODEL_NAME=gpt-4o-mini
  export OPENAI_API_KEY=sk-...
  export ENV_BASE_URL=http://localhost:8000
  python inference.py

The script runs each task for up to MAX_STEPS steps, prints per-step
results, and outputs a final baseline score table.  Total runtime is
well under 20 minutes even on slow endpoints.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Try to import openai; graceful error if missing
# ---------------------------------------------------------------------------
try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not found. Run: pip install openai")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Try to import requests for HTTP calls to the env server
# ---------------------------------------------------------------------------
try:
    import requests
except ImportError:
    print("ERROR: requests package not found. Run: pip install requests")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
API_KEY: str      = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY", "")
ENV_BASE_URL: str = os.environ.get("ENV_BASE_URL", "http://localhost:8000")

MAX_STEPS: int    = 10   # hard cap per task (keeps total runtime < 20 min)
TASKS: List[str]  = ["easy", "medium", "hard"]


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------
def build_llm_client() -> OpenAI:
    if not API_KEY:
        print(
            "WARNING: No API key found in HF_TOKEN or OPENAI_API_KEY. "
            "Requests will likely be rejected."
        )
    return OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "MISSING")


# ---------------------------------------------------------------------------
# Environment HTTP helpers  (synchronous; no WebSocket needed for baseline)
# ---------------------------------------------------------------------------
def env_reset(task_id: str) -> Dict[str, Any]:
    """POST /reset with task_id, return observation dict."""
    resp = requests.post(
        f"{ENV_BASE_URL}/reset",
        json={"task_id": task_id},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_step(sql_query: str) -> Dict[str, Any]:
    """POST /step with sql_query, return observation dict."""
    resp = requests.post(
        f"{ENV_BASE_URL}/step",
        json={"sql_query": sql_query, "metadata": {}},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert SQL engineer. You will be given:
1. A task description explaining what is wrong with a SQL query.
2. The database schema (DDL statements).
3. Step-by-step feedback on your previous attempts.

Your job is to write a corrected, efficient SQL query that satisfies the task.

RULES:
- Output ONLY the raw SQL query — no markdown fences, no explanation, no preamble.
- Do NOT use DROP, DELETE, UPDATE, INSERT, TRUNCATE, or ALTER statements.
- Use explicit JOINs (not comma-separated FROM clauses).
- Alias aggregated columns exactly as specified in the task.
- Terminate your SQL with a semicolon.
"""


# ---------------------------------------------------------------------------
# Agent: builds a prompt from observation and calls the LLM
# ---------------------------------------------------------------------------
def build_user_prompt(obs: Dict[str, Any], history: List[Dict]) -> str:
    lines: List[str] = []

    lines.append("=== TASK ===")
    lines.append(obs.get("task_description", ""))
    lines.append("")
    lines.append("=== DATABASE SCHEMA ===")
    lines.append(obs.get("db_schema", ""))
    lines.append("")

    if history:
        lines.append("=== ATTEMPT HISTORY ===")
        for i, h in enumerate(history, 1):
            lines.append(f"Attempt {i}:")
            lines.append(f"  Query   : {h['query']}")
            lines.append(f"  Feedback: {h['feedback']}")
            lines.append(f"  Score   : {h['score']:.4f}")
            if h.get("error"):
                lines.append(f"  Error   : {h['error']}")
        lines.append("")

    lines.append("=== YOUR TASK ===")
    lines.append(
        "Write the corrected SQL query. Output ONLY the raw SQL, nothing else."
    )

    return "\n".join(lines)


def call_llm(client: OpenAI, messages: List[Dict]) -> str:
    """Call the LLM and return the text of the first choice."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.0,   # deterministic for reproducible baseline
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


def clean_sql(raw: str) -> str:
    """Strip markdown fences if the model accidentally added them."""
    raw = raw.strip()
    for fence in ("```sql", "```SQL", "```"):
        if raw.startswith(fence):
            raw = raw[len(fence):]
    if raw.endswith("```"):
        raw = raw[:-3]
    return raw.strip()


# ---------------------------------------------------------------------------
# Run one task episode
# ---------------------------------------------------------------------------
def run_task_episode(client: OpenAI, task_id: str) -> Dict[str, Any]:
    """
    Run a full episode for one task.
    Returns a summary dict with task_id, steps, final_score, solved.
    """
    print(f"\n{'='*60}")
    print(f"  TASK: {task_id.upper()}")
    print(f"{'='*60}")

    obs = env_reset(task_id)
    ep_id = obs.get("episode_id")
    print(f"  Description preview: {obs.get('task_description','')[:120]}")

    history: List[Dict] = []
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    final_score = 0.0
    solved = False

    for step in range(1, MAX_STEPS + 1):
        user_prompt = build_user_prompt(obs, history)
        messages.append({"role": "user", "content": user_prompt})

        print(f"\n  Step {step}/{MAX_STEPS} — calling LLM…")
        raw_sql = call_llm(client, messages)
        sql = clean_sql(raw_sql)
        print(f"  Submitted SQL (truncated): {sql[:200]}…" if len(sql) > 200 else f"  Submitted SQL: {sql}")

        # send the action and include episode_id as a query param
        step_url = f"{ENV_BASE_URL}/step?episode_id={ep_id}"
        payload = {"sql_query": sql}
        resp = requests.post(step_url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # server responses may wrap the observation under an outer object
        if isinstance(data, dict) and "observation" in data:
            obs = data["observation"]
            reward = data.get("reward", 0.0)
            done = data.get("done", False)
        else:
            obs = data
            reward = obs.get("reward", 0.0)
            done = obs.get("done", False)

        score = obs.get("partial_score", 0.0)
        feedback = obs.get("feedback", "")
        error = obs.get("error_message")

        print(f"  Reward: {reward:.4f} | Score: {score:.4f} | Done: {done}")
        print(f"  Feedback: {feedback[:200]}")

        history.append({
            "query": sql,
            "feedback": feedback,
            "score": score,
            "error": error,
        })

        messages.append({"role": "assistant", "content": sql})

        final_score = score
        if obs.get("is_correct"):
            solved = True

        if done:
            break

    return {
        "task_id": task_id,
        "steps": len(history),
        "final_score": final_score,
        "solved": solved,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("\n" + "="*60)
    print("  SQL Debugger — Baseline Inference")
    print(f"  Model      : {MODEL_NAME}")
    print(f"  API Base   : {API_BASE_URL}")
    print(f"  Env Server : {ENV_BASE_URL}")
    print("="*60)

    # Verify env server is alive
    try:
        health = requests.get(f"{ENV_BASE_URL}/health", timeout=10)
        health.raise_for_status()
        print(f"\n   Environment server healthy: {health.json()}")
    except Exception as exc:
        print(f"\n   Cannot reach environment server at {ENV_BASE_URL}: {exc}")
        print("     Make sure the server is running (python -m uvicorn server.app:app)")
        sys.exit(1)

    client = build_llm_client()
    results: List[Dict] = []

    start_time = time.time()
    for task_id in TASKS:
        result = run_task_episode(client, task_id)
        results.append(result)

    elapsed = time.time() - start_time

    # --- Summary table ---
    print("\n\n" + "="*60)
    print("  BASELINE RESULTS SUMMARY")
    print("="*60)
    print(f"  {'Task':<12} {'Steps':>6} {'Score':>8} {'Solved':>8}")
    print(f"  {'-'*12} {'-'*6} {'-'*8} {'-'*8}")
    total_score = 0.0
    for r in results:
        print(
            f"  {r['task_id']:<12} {r['steps']:>6} "
            f"{r['final_score']:>8.4f} {'' if r['solved'] else '':>8}"
        )
        total_score += r["final_score"]
    avg = total_score / len(results)
    print(f"  {'-'*12} {'-'*6} {'-'*8} {'-'*8}")
    print(f"  {'AVERAGE':<12} {'':>6} {avg:>8.4f}")
    print(f"\n  Total elapsed: {elapsed:.1f}s")
    print("="*60 + "\n")

    # Save results to JSON
    output = {
        "model": MODEL_NAME,
        "api_base": API_BASE_URL,
        "results": results,
        "average_score": avg,
        "elapsed_seconds": elapsed,
    }
    with open("baseline_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("  Results saved to baseline_results.json")


if __name__ == "__main__":
    main()
