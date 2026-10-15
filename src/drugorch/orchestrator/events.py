"""The event stream: one envelope shape, a monotonic sequence, and every consumer agreed.

The browser, the on-disk ``events.jsonl``, and the replay path all read the same records, so
there is one definition of what happened in a run rather than three that can disagree.

Two properties the UI depends on:

* **``seq`` is monotonic per run and never reused.** It is emitted as the SSE ``id:`` field,
  so a browser that reconnects sends ``Last-Event-ID`` and gets exactly the events it missed.
  Replay is a slice, not a re-derivation.
* **every event carries the full budget block.** The routing trace draws its x-axis from
  cumulative spend, so an event that only said *what* happened without saying *where in the
  budget* would leave a gap the renderer could not fill.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

EventType = Literal[
    "run_started",
    "policy_thinking",
    "decision",
    "call_started",
    "call_progress",
    "call_finished",
    "molecules_added",
    "best_updated",
    # Facts used to change silently: the target panel read a snapshot fetched once at mount
    # and then showed it for the whole run, so a structure chosen at step 3 never appeared.
    "facts_updated",
    # Emitted by the LLM policy hook since it was added; it was never declared here, which
    # made it invisible to anything type-checking the stream.
    "llm_call",
    "budget_block",
    "policy_fallback",
    "subsystem_error",
    "warning",
    "run_finished",
    "run_cancelled",
]

# Events safe to drop when a subscriber falls behind. Progress is a continuous signal whose
# latest value supersedes the rest; a decision is a discrete fact that must never be lost.
DROPPABLE: frozenset[str] = frozenset({"call_progress", "policy_thinking"})


@dataclass(slots=True)
class Event:
    """One thing that happened, in the shape every consumer reads."""

    run_id: str
    seq: int
    type: EventType
    ts: float
    step: int
    status: str
    budget: dict
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "seq": self.seq,
            "ts": _iso(self.ts),
            "step": self.step,
            "type": self.type,
            "status": self.status,
            "budget": self.budget,
            "data": self.data,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    def to_sse(self) -> str:
        """SSE framing: named event, ``id:`` carrying the sequence number."""
        return f"id: {self.seq}\nevent: {self.type}\ndata: {self.to_json()}\n\n"


def _iso(ts: float) -> str:
    base = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(ts))
    return f"{base}.{int((ts % 1) * 1000):03d}Z"


class EventLog:
    """Append-only, in-memory event log for one run, with optional mirroring to disk.

    Subscribers are registered callbacks rather than queues so that the same log serves the
    CLI (which prints), the API (which fans out to SSE queues), and the store (which
    persists) without any of them knowing about the others.
    """

    def __init__(self, run_id: str, path: Path | None = None):
        self.run_id = run_id
        self.path = path
        self.events: list[Event] = []
        self._seq = 0
        self._lock = threading.Lock()
        self._subscribers: list[Any] = []
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def subscribe(self, fn) -> None:
        """Register a callback invoked for every event, in order."""
        self._subscribers.append(fn)

    def unsubscribe(self, fn) -> None:
        if fn in self._subscribers:
            self._subscribers.remove(fn)

    def emit(
        self,
        type_: EventType,
        *,
        step: int,
        status: str,
        budget: dict,
        data: dict | None = None,
    ) -> Event:
        with self._lock:
            self._seq += 1
            ev = Event(
                run_id=self.run_id,
                seq=self._seq,
                type=type_,
                ts=time.time(),
                step=step,
                status=status,
                budget=budget,
                data=data or {},
            )
            self.events.append(ev)
            if self.path is not None:
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(ev.to_json() + "\n")

        # Subscriber failures must not take down the run -- a browser disconnecting
        # mid-write is routine, not an error in the chemistry.
        for fn in list(self._subscribers):
            with contextlib.suppress(Exception):
                fn(ev)
        return ev

    def since(self, seq: int) -> list[Event]:
        """Every event after ``seq``. This is what reconnection replay serves."""
        with self._lock:
            return [e for e in self.events if e.seq > seq]

    @property
    def last_seq(self) -> int:
        return self._seq

    def __len__(self) -> int:
        return len(self.events)

    @staticmethod
    def read_jsonl(path: Path) -> list[dict]:
        """Load a recorded run's events -- used by replay and by the fake-run emitter."""
        out = []
        with Path(path).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out
