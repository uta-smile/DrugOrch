"""Cross-run value cache: ``(subsystem_key, smiles) -> value``, backed by SQLite.

This cache exists to save *compute*, never budget. A docking call that hits the cache still
charges the run, because the run did not know the answer when it asked -- and a cost curve
whose x-axis depends on what some earlier run happened to evaluate is not comparable to
anything. :mod:`drugorch.budget` holds the thing that is actually spent; keeping them in
separate modules keeps the distinction hard to blur.

``subsystem_key`` embeds the recipe version and the parameters that change the number (the
receptor, the exhaustiveness), so two configurations never share a row.

A stored failure is a real hit. Re-running a deterministic subsystem that already failed on a
molecule just burns the same wall time to reach the same ``None``.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

_MISS = object()


class ValueCache:
    """SQLite-backed store shared across runs, seeds and processes."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS value_cache (
                   subsystem_key TEXT NOT NULL,
                   smiles        TEXT NOT NULL,
                   value_json    TEXT,
                   ok            INTEGER NOT NULL,
                   wall_s        REAL,
                   created_at    REAL NOT NULL,
                   PRIMARY KEY (subsystem_key, smiles))"""
        )
        self._conn.commit()
        self.hits = 0
        self.misses = 0

    def get(self, subsystem_key: str, smiles: str) -> tuple[Any, bool] | None:
        """``(value, ok)`` if this pair was evaluated before, else ``None`` for a miss."""
        with self._lock:
            row = self._conn.execute(
                "SELECT value_json, ok FROM value_cache WHERE subsystem_key=? AND smiles=?",
                (subsystem_key, smiles),
            ).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        value = json.loads(row[0]) if row[0] is not None else None
        return (value, bool(row[1]))

    def get_many(self, subsystem_key: str, smiles: list[str]) -> dict[str, tuple[Any, bool]]:
        """Batch lookup. Absent keys are simply missing from the returned mapping."""
        if not smiles:
            return {}
        out: dict[str, tuple[Any, bool]] = {}
        with self._lock:
            # Chunked to stay under SQLite's variable limit on large batches.
            for i in range(0, len(smiles), 500):
                chunk = smiles[i : i + 500]
                q = ",".join("?" * len(chunk))
                rows = self._conn.execute(
                    f"SELECT smiles, value_json, ok FROM value_cache "  # noqa: S608
                    f"WHERE subsystem_key=? AND smiles IN ({q})",
                    (subsystem_key, *chunk),
                ).fetchall()
                for s, vj, ok in rows:
                    out[s] = (json.loads(vj) if vj is not None else None, bool(ok))
        self.hits += len(out)
        self.misses += len(smiles) - len(out)
        return out

    def put(
        self,
        subsystem_key: str,
        smiles: str,
        value: Any,
        ok: bool = True,
        wall_s: float = 0.0,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO value_cache VALUES (?,?,?,?,?,?)",
                (
                    subsystem_key,
                    smiles,
                    json.dumps(value) if value is not None else None,
                    int(ok),
                    wall_s,
                    time.time(),
                ),
            )
            self._conn.commit()

    def put_many(self, subsystem_key: str, items: dict[str, tuple[Any, bool, float]]) -> None:
        """Write a batch in one transaction."""
        if not items:
            return
        now = time.time()
        rows = [
            (
                subsystem_key,
                s,
                json.dumps(v) if v is not None else None,
                int(ok),
                wall,
                now,
            )
            for s, (v, ok, wall) in items.items()
        ]
        with self._lock:
            self._conn.executemany("INSERT OR REPLACE INTO value_cache VALUES (?,?,?,?,?,?)", rows)
            self._conn.commit()

    def count(self, subsystem_key: str | None = None) -> int:
        with self._lock:
            if subsystem_key is None:
                return self._conn.execute("SELECT COUNT(*) FROM value_cache").fetchone()[0]
            return self._conn.execute(
                "SELECT COUNT(*) FROM value_cache WHERE subsystem_key=?", (subsystem_key,)
            ).fetchone()[0]

    def describe(self) -> dict:
        return {"path": str(self.path), "rows": self.count(), "hits": self.hits, "misses": self.misses}

    def close(self) -> None:
        with self._lock:
            self._conn.close()
