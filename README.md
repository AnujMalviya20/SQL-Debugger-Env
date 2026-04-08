
---
title: SQL Debugger Env
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_file: server/app.py
pinned: false
------
title: SQL Debugger Env
emoji: 🛠️
colorFrom: blue
#
---
sdk: docker
app_file: server/app.py
---

# SQL Debugger & Optimizer

Problem
-------
This repository provides an environment where an agent iteratively corrects
or optimises SQL queries against a small SQLite dataset. The environment is
deterministic and returns structured observations and partial-credit rewards.

Approach
--------
- Tasks include full DDL and expected result rows.
- A deterministic grader executes submitted SQL and computes a partial score
    using result similarity and structural checks.
- Each episode uses a persistent per-episode SQLite file created at reset().

API endpoints
-------------
- POST /reset
    - Request: {"task_id": "easy"}
    - Response: initial observation (includes "episode_id")
- POST /step
    - Request: {"sql_query": "SELECT ..."} or include as JSON body
    - Query param: ?episode_id=<id> (required when multiple sessions active)
    - Response: observation with reward, partial_score, feedback, done
- GET /health
    - Liveness probe

How to run locally
-------------------
1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start the server:

```bash
uvicorn server.app:app --host 127.0.0.1 --port 8000
```

Docker run
----------
Build and run the container:

```bash
docker build -t sql-debugger-env .
docker run -p 8000:8000 sql-debugger-env
```

The health endpoint is available at http://localhost:8000/health
