"""
SQL Debugger & Optimizer Environment - Tasks & Graders
======================================================
Defines three tasks of increasing difficulty, each with
    - A natural-language description
    - A DB schema (DDL)
    - A broken/inefficient SQL query to start from
    - Expected output rows (ground truth)
    - A deterministic grader returning a score in [0.0, 1.0]

Tasks and deterministic graders for the SQL Debugger environment.

This module defines three tasks (easy, medium, hard) and a shared
deterministic grader that awards partial credit. The grader runs
queries against either an ephemeral in-memory DB (used by the grader)
or a provided persistent SQLite file (used by the running environment).
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------

# ---------------------------------------------------------------------------

DESTRUCTIVE_PATTERNS = re.compile(
    r"\b(DROP|DELETE|TRUNCATE|ALTER|UPDATE|INSERT|REPLACE|ATTACH|DETACH)\b",
    re.IGNORECASE,
)


def _normalize_row(row: Dict[str, Any]) -> frozenset:
    """Normalize a result row for set-based comparison."""
    return frozenset((k.lower(), str(v).strip().lower()) for k, v in row.items())


def _jaccard(a: List[Dict], b: List[Dict]) -> float:
    """Jaccard similarity between two collections of rows."""
    if not a and not b:
        return 1.0
    set_a = set(_normalize_row(r) for r in a)
    set_b = set(_normalize_row(r) for r in b)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _has_destructive(sql: str) -> bool:
    return bool(DESTRUCTIVE_PATTERNS.search(sql))


def _run_query(
    sql: str,
    setup_sql: Optional[str] = None,
    db_path: Optional[str] = None,
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """
    Execute `sql` and return (rows, error_message).

    Behavior:
    - If `db_path` is provided, connect to that SQLite file and execute `sql`.
      The function does not re-run any setup SQL in that case (the caller is
      responsible for initialising the DB file for the episode).
    - Otherwise, create an in-memory DB, apply `setup_sql` (required), then
      execute `sql` inside that transient DB (used by graders).
    """
    try:
        if db_path:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.execute(sql)
            rows = [dict(r) for r in cur.fetchall()]
            conn.close()
            return rows, None

        # Fallback: ephemeral in-memory DB populated by setup_sql
        if not setup_sql:
            raise ValueError("setup_sql must be provided when db_path is not set")
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.executescript(setup_sql)
        cur = conn.execute(sql)
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows, None
    except Exception as exc:
        return None, str(exc)


def _score_query(
    sql: str,
    setup_sql: str,
    expected_rows: List[Dict[str, Any]],
    expected_columns: List[str],
    check_efficiency: bool = False,
) -> Tuple[float, str]:
    """Shared deterministic grader. Returns (score, feedback)."""
    feedback_parts: List[str] = []
    score = 0.0

    # Destructive operation check (penalise once)
    if _has_destructive(sql):
        feedback_parts.append("Query contains destructive SQL (DROP/DELETE/UPDATE/etc.).")
        score -= 0.10
    else:
        score += 0.10
        feedback_parts.append("No destructive operations detected (+0.10).")

    # Execute on ephemeral DB for grading (isolated)
    rows, error = _run_query(sql, setup_sql=setup_sql)

    if error:
        feedback_parts.append(f"Query raised an error: {error}")
        # Simple hints
        el = error.lower()
        if "no such column" in el:
            feedback_parts.append("Hint: check column names against the schema.")
        elif "syntax error" in el:
            feedback_parts.append("Hint: check SQL syntax (keywords, commas, quotes).")
        score = max(0.0, score)
        return round(score, 4), " | ".join(feedback_parts)

    # Runs without error
    score += 0.15
    feedback_parts.append("Query executes without error (+0.15).")

    # Non-empty result
    if rows:
        score += 0.10
        feedback_parts.append(f"Query returned {len(rows)} row(s) (+0.10).")
    else:
        feedback_parts.append("Query returned 0 rows. Check WHERE clauses or JOINs.")

    # Column coverage
    if rows:
        returned_cols = set(c.lower() for c in rows[0].keys())
        expected_cols_lower = set(c.lower() for c in expected_columns)
        missing = expected_cols_lower - returned_cols
        if not missing:
            score += 0.15
            feedback_parts.append("All expected columns present (+0.15).")
        else:
            feedback_parts.append(f"Missing columns: {missing}. Expected: {expected_columns}.")

    # Row count
    if len(rows) == len(expected_rows):
        score += 0.15
        feedback_parts.append(f"Row count matches expected ({len(expected_rows)}) (+0.15).")
    else:
        feedback_parts.append(f"Row count mismatch: got {len(rows)}, expected {len(expected_rows)}.")

  
    jaccard = _jaccard(rows, expected_rows)
    content_score = round(0.30 * jaccard, 4)
    score += content_score
    if jaccard >= 1.0:
        feedback_parts.append("Perfect match: all rows correct (+0.30).")
    elif jaccard > 0:
        feedback_parts.append(f"Partial row match ({jaccard:.0%} Jaccard) (+{content_score:.2f}).")
    else:
        feedback_parts.append("No rows match expected output.")

    # Efficiency bonus for medium/hard
    if check_efficiency:
        sql_upper = sql.upper()
        uses_join = "JOIN" in sql_upper
        avoids_subquery_in_select = sql_upper.count("SELECT") <= 2
        if uses_join and avoids_subquery_in_select:
            score += 0.05
            feedback_parts.append("Efficiency bonus: JOIN-based approach detected (+0.05).")
        else:
            feedback_parts.append("Efficiency tip: prefer explicit JOINs over correlated subqueries.")

    # Clamp and return
    score = round(max(0.0, min(1.0, score)), 4)
    return score, " | ".join(feedback_parts)


# ---------------------------------------------------------------------------
# Task class and concrete tasks
# ---------------------------------------------------------------------------

@dataclass
class Task:
    task_id: str
    difficulty: str          # "easy" | "medium" | "hard"
    title: str
    description: str
    db_schema: str           # DDL to create and populate tables
    starting_query: str      # The broken/inefficient query shown to agent
    expected_columns: List[str]
    expected_rows: List[Dict[str, Any]]
    max_steps: int = 10

    def grade(self, sql: str) -> Tuple[float, str]:
        raise NotImplementedError


# EASY
EASY_SCHEMA = """
CREATE TABLE employees (
    id        INTEGER PRIMARY KEY,
    full_name TEXT    NOT NULL,
    dept      TEXT    NOT NULL,
    salary    REAL    NOT NULL,
    hired_on  TEXT    NOT NULL
);

INSERT INTO employees VALUES
  (1, 'Alice Sharma',   'Engineering', 95000, '2019-03-15'),
  (2, 'Bob Mehta',      'Marketing',   72000, '2020-07-01'),
  (3, 'Carol Patel',    'Engineering', 102000,'2018-11-22'),
  (4, 'David Singh',    'HR',          68000, '2021-01-10'),
  (5, 'Eva Krishnan',   'Engineering', 88000, '2022-05-30'),
  (6, 'Frank Iyer',     'Marketing',   75000, '2019-09-14');
"""

EASY_EXPECTED_ROWS = [
    {"full_name": "Alice Sharma",  "salary": 95000.0},
    {"full_name": "Carol Patel",   "salary": 102000.0},
    {"full_name": "Eva Krishnan",  "salary": 88000.0},
]

EASY_STARTING_QUERY = """
SELCT full_name, pay
FROM employees
WEHRE dept = 'Engineering'
ORDER BY salary DESC;
"""


@dataclass
class EasyTask(Task):
    task_id: str = "easy"
    difficulty: str = "easy"
    title: str = "Fix the Broken Query"
    description: str = (
        "The query below was written hastily and contains two bugs:\n"
        "1. A keyword typo (SELCT instead of SELECT, WEHRE instead of WHERE).\n"
        "2. An incorrect column alias — the column is called salary, not pay.\n\n"
        "Fix the query so it returns the full_name and salary of all Engineering "
        "department employees, ordered by salary descending."
    )
    db_schema: str = EASY_SCHEMA
    starting_query: str = EASY_STARTING_QUERY
    expected_columns: List[str] = field(
        default_factory=lambda: ["full_name", "salary"]
    )
    expected_rows: List[Dict[str, Any]] = field(
        default_factory=lambda: EASY_EXPECTED_ROWS
    )
    max_steps: int = 8

    def grade(self, sql: str) -> Tuple[float, str]:
        return _score_query(
            sql,
            EASY_SCHEMA,
            EASY_EXPECTED_ROWS,
            ["full_name", "salary"],
            check_efficiency=False,
        )


# MEDIUM
MEDIUM_SCHEMA = """
CREATE TABLE orders (
    order_id    INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    amount      REAL    NOT NULL,
    order_date  TEXT    NOT NULL
);

CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    city        TEXT NOT NULL
);

INSERT INTO customers VALUES
  (1, 'Priya Nair',    'Mumbai'),
  (2, 'Rahul Gupta',   'Delhi'),
  (3, 'Sneha Reddy',   'Hyderabad'),
  (4, 'Arjun Verma',   'Bangalore');

INSERT INTO orders VALUES
  (101, 1, 4500.00, '2024-01-15'),
  (102, 2, 1200.00, '2024-01-18'),
  (103, 1, 7800.00, '2024-02-03'),
  (104, 3, 3300.00, '2024-02-10'),
  (105, 2, 2100.00, '2024-02-22'),
  (106, 4, 9500.00, '2024-03-05'),
  (107, 3, 450.00,  '2024-03-12'),
  (108, 1, 620.00,  '2024-03-20');
"""

MEDIUM_EXPECTED_ROWS = [
    {"name": "Priya Nair",   "city": "Mumbai",    "total_spent": 12920.0},
    {"name": "Arjun Verma",  "city": "Bangalore", "total_spent": 9500.0},
    {"name": "Sneha Reddy",  "city": "Hyderabad", "total_spent": 3750.0},
    {"name": "Rahul Gupta",  "city": "Delhi",     "total_spent": 3300.0},
]

MEDIUM_STARTING_QUERY = """
SELECT name, city,
  (SELECT SUM(amount) FROM orders WHERE orders.customer_id = customers.customer_id) AS total_spent
FROM customers
ORDER BY total_spent DESC;
"""


@dataclass
class MediumTask(Task):
    task_id: str = "medium"
    difficulty: str = "medium"
    title: str = "Optimize: Correlated Subquery → JOIN + GROUP BY"
    description: str = (
        "The query below works but is extremely slow on large datasets because it "
        "uses a correlated subquery — it runs a separate SELECT for every customer row.\n\n"
        "Rewrite it using an explicit JOIN and GROUP BY so it only scans the orders "
        "table once. The result must show each customer's name, city, and total amount "
        "spent (aliased as `total_spent`), ordered by total_spent descending."
    )
    db_schema: str = MEDIUM_SCHEMA
    starting_query: str = MEDIUM_STARTING_QUERY
    expected_columns: List[str] = field(
        default_factory=lambda: ["name", "city", "total_spent"]
    )
    expected_rows: List[Dict[str, Any]] = field(
        default_factory=lambda: MEDIUM_EXPECTED_ROWS
    )
    max_steps: int = 10

    def grade(self, sql: str) -> Tuple[float, str]:
        return _score_query(
            sql,
            MEDIUM_SCHEMA,
            MEDIUM_EXPECTED_ROWS,
            ["name", "city", "total_spent"],
            check_efficiency=True,
        )


# HARD
HARD_SCHEMA = """
CREATE TABLE products (
    product_id   INTEGER PRIMARY KEY,
    product_name TEXT    NOT NULL,
    category     TEXT    NOT NULL,
    unit_price   REAL    NOT NULL
);

CREATE TABLE sales (
    sale_id     INTEGER PRIMARY KEY,
    product_id  INTEGER REFERENCES products(product_id),
    quantity    INTEGER NOT NULL,
    discount    REAL    NOT NULL DEFAULT 0.0,   -- fraction 0.0-1.0
    sale_date   TEXT    NOT NULL,
    salesperson TEXT
);

INSERT INTO products VALUES
  (1, 'Laptop Pro',    'Electronics', 85000),
  (2, 'Wireless Mouse','Electronics', 1500),
  (3, 'Office Desk',   'Furniture',   12000),
  (4, 'Standing Lamp', 'Furniture',   3500),
  (5, 'Notebook Pack', 'Stationery',  200),
  (6, 'Pen Set',       'Stationery',  150);

INSERT INTO sales VALUES
  (1,  1, 2, 0.05, '2024-01-10', 'Amit'),
  (2,  2, 5, 0.00, '2024-01-12', 'Bhavna'),
  (3,  1, 1, 0.10, '2024-01-20', 'Amit'),
  (4,  3, 3, 0.00, '2024-02-05', 'Chetan'),
  (5,  4, 2, 0.15, '2024-02-18', 'Bhavna'),
  (6,  2, 8, 0.05, '2024-03-01', 'Amit'),
  (7,  5,20, 0.00, '2024-03-10', NULL),
  (8,  6,15, 0.00, '2024-03-15', NULL),
  (9,  3, 1, 0.20, '2024-03-22', 'Chetan'),
  (10, 1, 3, 0.00, '2024-04-01', 'Deepa');
"""

# Expected: category revenue > 10000, computed as SUM(quantity * unit_price * (1-discount))
HARD_EXPECTED_ROWS = [
    {"category": "Electronics", "total_revenue": 511900.0, "total_units_sold": 19},
    {"category": "Furniture",   "total_revenue": 51550.0,  "total_units_sold": 6},
]

HARD_STARTING_QUERY = """
SELECT category, SUM(quantity * unit_price) AS total_revenue, SUM(quantity) AS total_units_sold
FROM products, sales
WHERE products.product_id = sales.product_id
GROUP BY category;
"""


@dataclass
class HardTask(Task):
    task_id: str = "hard"
    difficulty: str = "hard"
    title: str = "Multi-Table Revenue Report with Discount, NULL Handling & HAVING Filter"
    description: str = (
        "The query below produces wrong revenue figures because it ignores discounts, "
        "and it includes categories with very low sales that should be filtered out.\n\n"
        "Fix and extend the query so that:\n"
        "1. Revenue is computed as `SUM(quantity * unit_price * (1 - discount))` "
        "   (applying each sale's discount correctly).\n"
        "2. Only categories with `total_revenue > 10000` are included (use HAVING).\n"
        "3. Use an explicit JOIN instead of the implicit comma-join.\n"
        "4. Results ordered by total_revenue descending.\n"
        "5. Columns must be named exactly: `category`, `total_revenue`, `total_units_sold`."
    )
    db_schema: str = HARD_SCHEMA
    starting_query: str = HARD_STARTING_QUERY
    expected_columns: List[str] = field(
        default_factory=lambda: ["category", "total_revenue", "total_units_sold"]
    )
    expected_rows: List[Dict[str, Any]] = field(
        default_factory=lambda: HARD_EXPECTED_ROWS
    )
    max_steps: int = 12

    def grade(self, sql: str) -> Tuple[float, str]:
        return _score_query(
            sql,
            HARD_SCHEMA,
            HARD_EXPECTED_ROWS,
            ["category", "total_revenue", "total_units_sold"],
            check_efficiency=True,
        )


# Registry
TASKS: Dict[str, Task] = {
    "easy":   EasyTask(),
    "medium": MediumTask(),
    "hard":   HardTask(),
}


def get_task(task_id: str) -> Task:
    """Return a task by id. Raises KeyError for unknown ids."""
    if task_id not in TASKS:
        raise KeyError(
            f"Unknown task_id '{task_id}'. Available: {list(TASKS.keys())}"
        )
    return TASKS[task_id]


def list_tasks() -> List[str]:
    return list(TASKS.keys())
        
