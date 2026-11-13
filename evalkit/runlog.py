"""SQLite run log: one row per run, one row per answer, with cost and latency.

SQLite keeps the harness self-contained. The schema is plain enough to move to Postgres or to send
the same rows to Langfuse later.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    label TEXT NOT NULL,
    model TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT
);
CREATE TABLE IF NOT EXISTS answers (
    run_id INTEGER NOT NULL REFERENCES runs(id),
    case_id TEXT NOT NULL,
    output TEXT NOT NULL,
    passed INTEGER NOT NULL,
    scores TEXT NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    latency_ms REAL NOT NULL,
    simulated INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    PRIMARY KEY (run_id, case_id)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class RunLog:
    def __init__(self, path: Union[str, Path] = ":memory:"):
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def start_run(self, label: str, model: str) -> int:
        cur = self.db.execute("INSERT INTO runs (label, model, started_at) VALUES (?, ?, ?)",
                              (label, model, _now()))
        self.db.commit()
        return int(cur.lastrowid)

    def log_answer(self, run_id: int, case_id: str, output: str, passed: bool, scores: Dict[str, dict],
                   input_tokens: int, output_tokens: int, cost_usd: float, latency_ms: float,
                   simulated: bool = False, error: Optional[str] = None) -> None:
        self.db.execute(
            "INSERT INTO answers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, case_id, output, int(passed), json.dumps(scores), input_tokens, output_tokens,
             cost_usd, latency_ms, int(simulated), error))
        self.db.commit()

    def finish_run(self, run_id: int) -> None:
        self.db.execute("UPDATE runs SET finished_at = ? WHERE id = ?", (_now(), run_id))
        self.db.commit()

    def answers(self, run_id: int) -> List[sqlite3.Row]:
        return self.db.execute("SELECT * FROM answers WHERE run_id = ? ORDER BY case_id", (run_id,)).fetchall()

    def runs(self) -> List[sqlite3.Row]:
        return self.db.execute("SELECT * FROM runs ORDER BY id").fetchall()

    def close(self) -> None:
        self.db.close()
