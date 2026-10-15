"""Owns live runs for the web server: one asyncio task each, fanning events out to browsers.

The orchestrator's ``step()`` is synchronous and does real work -- docking blocks for tens of
seconds -- so it runs on a worker thread and the event loop stays free. Without that, one
docking round would freeze every open browser including the one watching it.

Fan-out is per-subscriber bounded queues. When a browser falls behind, *that* subscriber
drops events and nobody else is affected; and what it drops is restricted to the
continuously-valued ones (progress, state) whose latest value supersedes the rest. A decision
is never dropped, because the routing trace is a record of discrete facts and a hole in it is
indistinguishable from a decision that was never made.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import anyio

from ..config import Settings, get_settings
from ..policy import build_policy
from ..store.runs import summarize, write_run_dir
from ..subsystems.registry import build_registry
from ..types import TargetSpec
from .events import DROPPABLE, Event, EventLog
from .run import Orchestrator
from .state import RunState

QUEUE_LIMIT = 500


@dataclass
class Subscriber:
    """One browser watching one run."""

    queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=QUEUE_LIMIT))
    dropped: int = 0


@dataclass
class LiveRun:
    """A run plus everything needed to observe and stop it."""

    run_id: str
    state: RunState
    orchestrator: Orchestrator
    log: EventLog
    cancel: threading.Event
    policy_name: str
    created_at: float = field(default_factory=time.time)
    task: asyncio.Task | None = None
    subscribers: list[Subscriber] = field(default_factory=list)
    out_dir: str | None = None

    @property
    def status(self) -> str:
        return self.state.status


class RunManager:
    """Creates, drives and serves runs."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.settings.ensure_dirs()
        self.runs: dict[str, LiveRun] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    # --- creation --------------------------------------------------------------
    def create(
        self,
        *,
        goal: str,
        target: str,
        policy: str = "scripted",
        budget_units: float | None = None,
        seed: int = 42,
        max_steps: int | None = None,
        run_id: str | None = None,
        tools: tuple[str, ...] | list[str] | None = None,
        prompt_extra: str = "",
        prompt_override: str = "",
    ) -> LiveRun:
        """Start a run.

        ``tools`` names the subsystems this run gets; ``None`` means every one installed.
        It is applied when the registry is built, so a tool left out is not in the policy's
        schema and cannot be chosen, rather than being chosen and then refused.

        ``prompt_extra`` and ``prompt_override`` are per-run text for the LLM arm and are
        never written anywhere -- see :func:`drugorch.policy.llm.compose_system`.
        """
        from ..budget import BudgetLedger

        rid = run_id or f"r_{time.strftime('%Y%m%d_%H%M%S')}_{policy}_{seed}"
        if rid in self.runs:
            rid = f"{rid}_{int(time.time() * 1000) % 100000}"

        state = RunState(
            run_id=rid,
            goal=goal,
            target=TargetSpec(query=target),
            budget=BudgetLedger(total=budget_units or self.settings.default_budget_units),
            seed=seed,
        )
        registry = build_registry(self.settings, enabled=tools)
        log = EventLog(rid, path=self.settings.runs_dir / rid / "events.jsonl")
        # Only the LLM arm has a prompt; passing these to a rule-based policy would be a
        # TypeError, and silently accepting them there would suggest they did something.
        extra: dict[str, Any] = {}
        if policy == "llm" and (prompt_extra or prompt_override):
            extra = {"prompt_extra": prompt_extra, "prompt_override": prompt_override}
        pol = build_policy(policy, seed=seed, settings=self.settings, **extra)
        cancel = threading.Event()
        orch = Orchestrator(state, registry, pol, log, settings=self.settings, cancel=cancel)
        if max_steps is not None:
            orch.termination.max_steps = max_steps

        live = LiveRun(
            run_id=rid,
            state=state,
            orchestrator=orch,
            log=log,
            cancel=cancel,
            policy_name=policy,
        )
        log.subscribe(lambda ev: self._fanout(live, ev))
        self.runs[rid] = live
        return live

    # --- fan-out ---------------------------------------------------------------
    def _fanout(self, live: LiveRun, ev: Event) -> None:
        """Push an event to every subscriber. Called from the worker thread."""
        loop = self._loop
        if loop is None:
            return
        for sub in list(live.subscribers):
            loop.call_soon_threadsafe(self._offer, sub, ev)

    @staticmethod
    def _offer(sub: Subscriber, ev: Event) -> None:
        try:
            sub.queue.put_nowait(ev)
        except asyncio.QueueFull:
            # Make room by discarding the oldest droppable event rather than the newest
            # arrival -- a stale progress tick is worth less than a fresh one.
            if ev.type in DROPPABLE:
                sub.dropped += 1
                return
            with contextlib.suppress(asyncio.QueueEmpty):
                sub.queue.get_nowait()
                sub.dropped += 1
            with contextlib.suppress(asyncio.QueueFull):
                sub.queue.put_nowait(ev)

    def subscribe(self, run_id: str) -> Subscriber | None:
        live = self.runs.get(run_id)
        if live is None:
            return None
        sub = Subscriber()
        live.subscribers.append(sub)
        return sub

    def unsubscribe(self, run_id: str, sub: Subscriber) -> None:
        live = self.runs.get(run_id)
        if live is not None and sub in live.subscribers:
            live.subscribers.remove(sub)

    # --- driving ---------------------------------------------------------------
    async def start(self, live: LiveRun) -> None:
        """Launch the run loop as a background task."""
        self._loop = asyncio.get_running_loop()
        if live.task is not None:
            return
        live.task = asyncio.create_task(self._drive(live), name=f"run:{live.run_id}")

    async def _drive(self, live: LiveRun) -> None:
        orch = live.orchestrator
        try:
            await anyio.to_thread.run_sync(orch.start)
            while not orch.finished:
                # One step per hop so cancellation lands between steps rather than
                # mid-docking, and so the loop yields to the event loop regularly.
                await anyio.to_thread.run_sync(orch.step)
        except asyncio.CancelledError:
            live.cancel.set()
            with contextlib.suppress(Exception):
                await anyio.to_thread.run_sync(
                    lambda: orch.finish(_cancelled_termination())
                )
            raise
        except Exception as exc:  # noqa: BLE001
            live.state.status = "failed"
            live.state.error = f"{type(exc).__name__}: {exc}"
            live.log.emit(
                "run_finished",
                step=live.state.step,
                status="failed",
                budget=live.state.budget.snapshot(),
                data={"code": "error", "reason": live.state.error},
            )
        finally:
            with contextlib.suppress(Exception):
                live.out_dir = str(self._persist(live))

    def _persist(self, live: LiveRun):
        orch = live.orchestrator
        return write_run_dir(
            self.settings.runs_dir,
            live.state,
            policy_describe=orch.policy.describe(),
            registry_describe=orch.registry.describe(),
            settings_describe=self.settings.describe(),
            termination={
                "code": orch.stop_reason.code if orch.stop_reason else "",
                "reason": orch.stop_reason.reason if orch.stop_reason else "",
            },
            llm_log=orch.llm_log(),
        )

    async def cancel(self, run_id: str) -> bool:
        live = self.runs.get(run_id)
        if live is None:
            return False
        live.cancel.set()
        if live.task is not None and not live.task.done():
            live.task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await live.task
        return True

    async def shutdown(self) -> None:
        for rid in list(self.runs):
            await self.cancel(rid)

    # --- reading ---------------------------------------------------------------
    def get(self, run_id: str) -> LiveRun | None:
        return self.runs.get(run_id)

    def snapshot(self, run_id: str) -> dict | None:
        live = self.runs.get(run_id)
        if live is None:
            return None
        snap = live.state.snapshot()
        snap["policy"] = live.policy_name
        snap["last_seq"] = live.log.last_seq
        snap["out_dir"] = live.out_dir
        return snap

    def listing(self, limit: int = 50) -> list[dict]:
        """Live runs, newest first. Finished runs on disk are listed by the store."""
        out = []
        for live in sorted(self.runs.values(), key=lambda r: -r.created_at)[:limit]:
            s = summarize(live.state, live.policy_name)
            s["created_at"] = live.created_at
            out.append(s)
        return out


def _cancelled_termination():
    from .termination import Termination

    return Termination(True, "cancelled", "cancelled by user")


_manager: RunManager | None = None


def get_manager(settings: Settings | None = None) -> RunManager:
    global _manager  # noqa: PLW0603 - one manager per process, by design
    if _manager is None:
        _manager = RunManager(settings)
    return _manager


__all__ = ["LiveRun", "RunManager", "Subscriber", "get_manager"]


def _unused(x: Any) -> Any:  # pragma: no cover
    return x
