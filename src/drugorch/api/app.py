"""The HTTP API and the static web mount.

The contract the browser depends on is in two places: the SSE stream at
``/api/runs/{id}/events``, and the event envelope defined in
:mod:`drugorch.orchestrator.events`. Everything else is convenience.

Reconnection is the part worth getting right. The server assigns a monotonic ``seq`` and
emits it as the SSE ``id:`` field, so a browser that drops the connection sends
``Last-Event-ID`` automatically and the stream resumes exactly where it stopped. A client can
also pass ``from_seq`` explicitly, which is what a page reload does -- it has the events it
already rendered and only wants the rest. Replay is therefore a slice of the log, not a
re-derivation, and a reloaded page shows the identical trace.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..config import (
    ENV_PATH,
    env_file_has_key,
    get_settings,
    mask_key,
    set_runtime_api_key,
    write_env_key,
)
from ..orchestrator.manager import get_manager
from ..policy import DESCRIPTIONS, POLICY_NAMES
from ..prep import poses as posestore
from ..subsystems.registry import DEFAULT_TOOLS, build_registry

HEARTBEAT_S = 15.0

# The key endpoints accept a secret over plain HTTP, so they are served only to a client on
# this machine. DrugOrch is a local research tool and binds loopback by default, but a
# `--host 0.0.0.0` typed to reach it from a laptop would otherwise expose a key-setting POST
# to the network, and that failure is silent until it is expensive.
LOCAL_CLIENTS = {"127.0.0.1", "::1", "localhost", "testclient"}


@lru_cache(maxsize=64)
def _selection_curve_cached(noise: float, spread: float) -> dict:
    """A selection sweep, memoised on its two parameters.

    About a second of arithmetic for a result that is a pure function of its inputs, behind
    a page whose slider will ask for the same handful of values repeatedly. Bounded rather
    than a plain dict because the parameters are floats off a query string: an unbounded
    cache here is a slow memory leak driven by whatever a client chooses to ask for.
    """
    from ..harness.selection import selection_curve

    return selection_curve(noise=noise, spread=spread).as_dict()


class SetKey(BaseModel):
    key: str = Field(..., min_length=8, max_length=400)
    persist: bool = Field(
        default=False,
        description="Also write the key to .env, so it survives a server restart.",
    )
    verify: bool = Field(default=True, description="Check the key against the API first.")


class SetProvider(BaseModel):
    provider: Literal["anthropic", "openai", "chatgpt"] | None = None
    model: str | None = Field(default=None, max_length=200)
    effort: str | None = Field(default=None, max_length=20)


class CreateRun(BaseModel):
    goal: str = Field(..., min_length=1, max_length=2000)
    target: str = Field(..., min_length=1, max_length=200)
    policy: str = "scripted"
    budget_units: float = Field(default=120.0, gt=0, le=100_000)
    seed: int = 42
    max_steps: int = Field(default=40, ge=1, le=500)
    # Which subsystems this run is given. `None` means every installed one, which is what a
    # client that predates the tool picker sends -- so an old client keeps its old behaviour
    # rather than silently starting runs with no tools.
    tools: list[str] | None = None
    # Per-run prompt text for the llm arm. Deliberately not a setting: it is recorded on the
    # run and nowhere else, so two runs with different instructions can never be mistaken
    # for a repeat of each other.
    prompt_extra: str = Field(default="", max_length=20_000)
    prompt_override: str = Field(default="", max_length=20_000)


class EmbedRequest(BaseModel):
    smiles: list[str] = Field(..., min_length=1, max_length=64)
    seed: int = 42


class ElaborateRequest(BaseModel):
    seeds: list[str] = Field(..., min_length=1, max_length=16)
    n: int = Field(default=20, ge=1, le=64)
    seed: int = 42
    mutation_rate: float = Field(default=0.5, ge=0.0, le=1.0)


class PocketRequest(BaseModel):
    pdb_id: str = Field(..., min_length=4, max_length=10)
    ligand_code: str = Field(default="", max_length=10)


class DockRequest(PocketRequest):
    smiles: list[str] = Field(..., min_length=1, max_length=64)
    seed: int = 42
    exhaustiveness: int = Field(default=8, ge=1, le=32)


class PromptPreview(BaseModel):
    goal: str = Field(default="", max_length=2000)
    target: str = Field(default="", max_length=200)
    budget_units: float = Field(default=120.0, gt=0, le=100_000)
    tools: list[str] | None = None
    prompt_extra: str = Field(default="", max_length=20_000)
    prompt_override: str = Field(default="", max_length=20_000)


def create_app() -> FastAPI:
    settings = get_settings()
    settings.ensure_dirs()
    app = FastAPI(title="DrugOrch", version="0.1.0")
    manager = get_manager(settings)

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        await manager.shutdown()

    # --- manifest ----------------------------------------------------------
    @app.get("/api/manifest")
    def manifest() -> dict:
        """What exists and what it costs. The UI builds its forms from this."""
        reg = build_registry(settings)
        return {
            "subsystems": reg.to_dict(),
            "policies": [
                {"name": n, "description": _policy_description(n, settings)}
                for n in POLICY_NAMES
            ],
            "defaults": {
                "budget_units": settings.default_budget_units,
                "max_steps": settings.max_steps,
                "seed": 42,
                # Which tools start ticked. Everything discovered is offered; this is the
                # set a run gets if nobody touches the picker, and it is the four that every
                # previously recorded run had -- so an untouched form is the same experiment
                # it was before a fifth subsystem existed.
                "tools": [k for k in DEFAULT_TOOLS if k in reg],
            },
            "has_api_key": settings.api_key() is not None,
            "api_key_source": settings.api_key_source(),
            "llm": {"provider": settings.llm_provider, **_llm_readiness()},
            "versions": settings.describe()["versions"],
        }

    # --- the API key -------------------------------------------------------
    # Supplying the key here rather than in .env is the normal path for a local demo: no
    # file editing, no restart, and the UI can confirm the key works before a run is
    # started instead of after a decision silently falls back to the scripted policy.
    def _require_local(request: Request) -> None:
        host = request.client.host if request.client else ""
        if host not in LOCAL_CLIENTS:
            raise HTTPException(
                status_code=403,
                detail="the API-key endpoints are served only to localhost; "
                "set ANTHROPIC_API_KEY in .env on the server instead",
            )

    def _key_status() -> dict:
        return {
            "has_key": settings.api_key() is not None,
            "source": settings.api_key_source(),
            "masked": mask_key(settings.api_key()),
            "env_path": str(ENV_PATH),
            "env_has_key": env_file_has_key(),
            "model": settings.llm_model,
        }

    @app.get("/api/key")
    def key_status(request: Request) -> dict:
        """Whether a key is in effect and where it came from. Never the key itself."""
        _require_local(request)
        return _key_status()

    @app.put("/api/key")
    async def set_key(spec: SetKey, request: Request) -> dict:
        _require_local(request)
        key = spec.key.strip()
        if spec.verify:
            from ..llm.backend import verify_api_key

            ok, message, reachable = await asyncio.to_thread(verify_api_key, key)
            if not ok:
                # `reachable` lets the UI distinguish a wrong key, which the user must fix,
                # from an unreachable API, where saving unverified is the sensible offer.
                raise HTTPException(
                    status_code=400, detail={"message": message, "reachable": reachable}
                )
        else:
            message = "stored without verifying"

        set_runtime_api_key(key)
        persisted = None
        if spec.persist:
            try:
                persisted = str(write_env_key(key))
            except OSError as exc:
                # The key is live for this session either way; say so rather than failing.
                message += f" (but .env could not be written: {exc})"
        return {**_key_status(), "message": message, "persisted_to": persisted}

    @app.delete("/api/key")
    def clear_key(request: Request, forget: bool = Query(False)) -> dict:
        """Drop the session key. ``forget=true`` also blanks the one in ``.env``."""
        _require_local(request)
        set_runtime_api_key(None)
        if forget:
            with contextlib.suppress(OSError):
                write_env_key("")
        return _key_status()

    # --- ChatGPT sign-in ----------------------------------------------------
    # The alternative to an API key: authorize DrugOrch against the user's own ChatGPT plan.
    # Sign-in blocks on a human clicking Approve in a browser, so it runs on a worker thread
    # and the request simply waits -- there is no useful partial state to report, and a
    # polling handshake would be more moving parts for the same five seconds.
    @app.get("/api/chatgpt")
    def chatgpt_status(request: Request) -> dict:
        _require_local(request)
        return _chatgpt_status()

    @app.post("/api/chatgpt/signin")
    async def chatgpt_signin(request: Request) -> dict:
        _require_local(request)
        from ..llm.chatgpt_auth import SessionStore, sign_in
        from ..llm.response_cache import LLMUnavailable

        store = SessionStore(settings.chatgpt_session_path)
        try:
            await asyncio.to_thread(sign_in, store, open_browser=True, timeout_s=300.0)
        except LLMUnavailable as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
        return _chatgpt_status()

    @app.delete("/api/chatgpt")
    def chatgpt_signout(request: Request) -> dict:
        _require_local(request)
        from ..llm.chatgpt_auth import SessionStore

        SessionStore(settings.chatgpt_session_path).clear()
        return _chatgpt_status()

    @app.get("/api/models")
    async def model_catalog(request: Request, provider: str | None = Query(None)) -> dict:
        """Models and reasoning efforts for a provider, live from the account where possible.

        Listing hits the provider's API, so it runs off the event loop. A signed-out install
        still gets a usable catalogue; ``source`` says which it is, so the UI can avoid
        implying it checked the account when it did not.
        """
        _require_local(request)
        from ..llm.catalog import catalog

        prov = provider or settings.llm_provider
        data = await asyncio.to_thread(catalog, prov, settings)
        return {
            **data,
            "current": {
                "model": settings.llm_model if prov == "anthropic" else settings.openai_model,
                "effort": settings.llm_effort if prov == "anthropic" else settings.openai_effort,
            },
        }

    @app.put("/api/provider")
    def set_provider(spec: SetProvider, request: Request) -> dict:
        """Choose the LLM, the model and how hard it thinks.

        All three go into run provenance and into the response-cache key, because all three
        change the arm under test: Opus at ``max`` and GPT-5.1 at ``low`` are two different
        experiments, not two runs of one.
        """
        _require_local(request)
        from ..llm.catalog import (
            efforts_for,
            model_default_effort,
            model_efforts,
            valid_effort,
        )

        note = ""
        if spec.provider:
            settings.set_provider(spec.provider)
        if spec.model:
            settings.set_model(spec.model)
            # Models do not share an effort vocabulary, so a model change can strand the
            # stored effort on a value the new model rejects. Left alone that is invisible
            # until the first decision 400s and the run falls back to the scripted arm, so
            # it is corrected here and said out loud.
            if spec.effort is None:
                allowed = model_efforts(settings.llm_provider, spec.model, settings)
                current = settings.effort_for_provider()
                if allowed and current not in allowed:
                    landed = model_default_effort(settings.llm_provider, spec.model, settings)
                    settings.set_effort(landed)
                    note = (
                        f"{spec.model} does not offer effort {current!r}; "
                        f"switched to {landed!r}."
                    )
        if spec.effort:
            # Checked against the chosen model's own list when the provider publishes one,
            # because that list is authoritative. Rejected rather than silently coerced: a
            # run that recorded an effort nobody chose is worse than an error on screen.
            allowed = model_efforts(
                settings.llm_provider, settings.model_for_provider(), settings
            )
            if not valid_effort(settings.llm_provider, spec.effort, allowed):
                options = allowed or efforts_for(settings.llm_provider)
                raise HTTPException(
                    status_code=400,
                    detail=f"{settings.model_for_provider()} does not accept effort "
                    f"{spec.effort!r}; choose one of {', '.join(options)}",
                )
            settings.set_effort(spec.effort)

        return {
            "provider": settings.llm_provider,
            "model": settings.model_for_provider(),
            "effort": settings.effort_for_provider(),
            "note": note,
            **_llm_readiness(),
        }

    # --- benchmark ---------------------------------------------------------
    @app.get("/api/bench")
    def bench() -> dict:
        """The leaderboard, from the most recent grid on disk.

        Served rather than recomputed: a grid takes minutes to hours to produce and the page
        should open instantly. Re-running it is a CLI action, deliberately -- it spends real
        compute and should not be one click away.
        """
        from ..harness.grid import load
        from ..harness.report import PRIMARY, RANKED_BY, leaderboard, per_target

        path = settings.data_dir / "bench" / "grid.json"
        if not path.is_file():
            return {"available": False, "hint": "run: drugorch bench"}

        try:
            result = load(path)
        except Exception as exc:  # noqa: BLE001
            return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

        board = leaderboard(result.rows, metric=PRIMARY)
        return {
            "available": True,
            "metric": PRIMARY,
            "metric_doc": RANKED_BY.get(PRIMARY, ""),
            "leaderboard": [a.to_dict() for a in board],
            "per_target": per_target(result.rows, metric=PRIMARY),
            "rows": [r.to_dict() for r in result.rows],
            "splits": result.splits,
            "failures": result.failures,
            "n_cells": len(result.rows),
            "elapsed_s": result.to_dict().get("elapsed_s"),
        }

    @app.get("/api/bench/external")
    def bench_external() -> dict:
        """The public-benchmark board: DrugOrch beside the published state of the art.

        Served from disk for the same reason the routing board is, and with the published
        rows attached even when no run has happened. A reader who opens this page before
        anything has been measured should still see what the benchmark is and what the
        published numbers are, rather than an empty table.
        """
        from ..harness.external import baselines
        from ..harness.external.run import load

        path = settings.data_dir / "bench" / "external.json"
        published = {
            "published": baselines.published(),
            "published_source": baselines.SOURCE,
        }
        if not path.is_file():
            return {
                "available": False,
                "hint": "run: drugorch extbench --n 200",
                **published,
            }
        try:
            payload = load(path)
        except Exception as exc:  # noqa: BLE001
            return {"available": False, "error": f"{type(exc).__name__}: {exc}", **published}

        # The attempt transcripts are the bulk of the file and the board does not read them;
        # a 200-instance run carries several megabytes of them.
        payload.pop("attempts", None)
        payload["available"] = True
        return {**published, **payload}

    @app.get("/api/bench/tools")
    def bench_tools() -> dict:
        """What each side of the public benchmark is allowed to do.

        A score is only interpretable beside the toolset that produced it, so the two
        rosters are served together. The benchmark's half is parsed from its own sources on
        each request: it is a few milliseconds of `ast` over seventeen files, and reading it
        live means the page cannot drift from the vendored code the way a copy would.
        """
        from ..harness.external import baselines
        from ..harness.external.toolsheet import sheet

        return {**sheet(), "source": baselines.SOURCE}

    # --- policy cards ------------------------------------------------------
    @app.get("/api/policies")
    def policies() -> dict:
        """The auditable description of every routing arm, with worked examples.

        Served rather than hard-coded in the frontend because the rule-based examples are
        produced by running the policies: a card and the code it describes cannot drift
        apart if the card is generated from the code.
        """
        from ..policy.cards import all_cards

        return {"policies": all_cards(), "chosen_by": "human"}

    @app.get("/api/glossary")
    def glossary() -> dict:
        """Definitions for every metric the interface displays.

        Served rather than written into the frontend so that a metric cannot be described
        one way in a tooltip and another in the goals article. The short text is imported
        from the modules that compute the values, so it cannot drift from them either.
        """
        from ..glossary import as_dict

        return as_dict()

    @app.get("/api/tools/authoring")
    def tool_authoring() -> dict:
        """The contract, a working template, where to save it, and what failed to load.

        The plugin mechanism always accepted third-party subsystems; nothing in the
        interface said so, named the factory, or showed the directory being scanned. This
        exports all of it, including the load failures the registry already collects: a tool
        with a syntax error is otherwise simply absent from the manifest, the prompt and the
        picker, with nothing anywhere saying why.
        """
        from ..subsystems.authoring import authoring_info

        return authoring_info(settings, build_registry(settings))

    @app.get("/api/selection-curve")
    def selection_curve_endpoint(
        noise: float = Query(2.0, ge=0.0, le=6.0),
        spread: float = Query(1.0, gt=0.0, le=5.0),
    ) -> dict:
        """What best-by-measured-score actually selects, as the pool grows.

        This backs the figure on the Goals page arguing that a budget is a statistical
        constraint rather than a billing one. It is served from Python rather than computed
        in the browser for the same reason the policy cards are: the claim is checked by
        :mod:`tests.test_selection`, and a second implementation in JavaScript would be an
        unchecked one that could drift from the tested arithmetic without anyone noticing.

        ``noise`` is the control that matters -- it is the oracle's error, the only
        assumption the whole argument rests on, and at ``noise=0`` the argument correctly
        disappears. Cached because a sweep is about a second of arithmetic and the answer
        is a pure function of its two parameters.
        """
        return _selection_curve_cached(round(noise, 2), round(spread, 2))

    @app.post("/api/prompt")
    def prompt_preview(spec: PromptPreview) -> dict:
        """Exactly what the routing model will be sent, before a run is started.

        Rendered from the same functions the policy calls, against a registry built from
        the same tool selection the run will get -- so a tool switched off in the form is
        absent from the manifest here because it will be absent there. The alternative,
        documenting the prompt somewhere, produces a document that is wrong the first time
        a capability's summary is edited.

        A POST because the body carries the operator's draft prompt. Nothing is stored: the
        text is rendered and returned, and the next request starts from whatever the client
        sends again.
        """
        tools = _resolve_tools(spec.tools, settings)
        reg = build_registry(settings, enabled=tools)
        from ..policy.preview import render

        return render(
            manifest=reg.capabilities(),
            goal=spec.goal or "find potent, selective inhibitors",
            target=spec.target or "<target not chosen yet>",
            budget_units=spec.budget_units,
            prompt_extra=spec.prompt_extra,
            prompt_override=spec.prompt_override,
            max_steps=settings.max_steps,
        )

    # --- explainability ----------------------------------------------------
    @app.get("/api/runs/{run_id}/decisions")
    def run_decisions(run_id: str, step: int | None = Query(None, ge=1)) -> dict:
        """Every decision with the state the policy saw when it made it.

        Separate from the snapshot because `state_seen` is a few KB per step: a scientist
        reconsidering one decision wants all of it, and a page rendering a trace wants none.
        """
        live = manager.get(run_id)
        if live is not None:
            rows = [r.to_dict_full() for r in live.state.history]
        else:
            rows = _recorded_decisions(run_id)
            if rows is None:
                raise HTTPException(status_code=404, detail="no such run")
        if step is not None:
            rows = [r for r in rows if r.get("step") == step]
            if not rows:
                raise HTTPException(status_code=404, detail=f"no decision at step {step}")
        return {"run_id": run_id, "decisions": rows}

    @app.get("/api/runs/{run_id}/llm")
    def run_llm_log(run_id: str, full: bool = Query(False)) -> dict:
        """The LLM exchanges behind a run. Empty for every rule-based arm, by definition.

        `full=true` includes the prompts and responses verbatim. The default omits them,
        because the common use is reading the token and fallback accounting, and the
        prompts are large enough to make that slow for no reason.
        """
        live = manager.get(run_id)
        if live is not None:
            calls = live.orchestrator.llm_log()
            totals = live.orchestrator.llm_totals()
            policy_name = live.policy_name
        else:
            calls = _recorded_llm(run_id)
            if calls is None:
                raise HTTPException(status_code=404, detail="no such run")
            totals = _llm_totals_from(calls)
            policy_name = _recorded_policy_name(run_id)

        # A fallback is the failure mode that looks like success: the run completes, the
        # trace looks normal, and the arm under test did not make the call. Counted here so
        # it is one number on the panel rather than something you have to notice.
        if live is not None:
            sources = [r.decision.source for r in live.state.history]
        else:
            sources = [r.get("source") for r in (_recorded_decisions(run_id) or [])]
        n_fallback = sum(1 for s in sources if s == "fallback")

        if not full:
            calls = [
                {k: v for k, v in c.items() if k not in ("system", "prompt", "response", "usage")}
                for c in calls
            ]
        return {
            "run_id": run_id,
            "policy": policy_name,
            "uses_llm": policy_name == "llm",
            "calls": calls,
            "totals": totals,
            "n_fallback_decisions": n_fallback,
            # Against the number of decisions, not the number of logged calls: when the API
            # is unreachable the policy raises before it can log anything, so the call list
            # is empty exactly when the fallback count matters most.
            "n_decisions": len(sources),
        }

    # --- target search -----------------------------------------------------
    @app.get("/api/targets/search")
    async def target_search(
        q: str = Query(..., min_length=1, max_length=120),
        limit: int = Query(8, ge=1, le=25),
    ) -> dict:
        """Autocomplete across the data banks, resolved server-side.

        Deliberately cheap: one UniProt request in the common case, because this runs while
        the user is still typing. The expensive question -- can this target actually be
        docked and does anyone have bioactivity for it -- is `/api/targets/readiness`, asked
        once after a pick.
        """
        rows = await asyncio.to_thread(_search_targets, q, limit)
        return {"query": q, "kind": _classify(q), "results": rows}

    @app.get("/api/targets/suggested")
    def target_suggestions() -> dict:
        """Starter targets, shown before anything is typed.

        An empty autocomplete is a dead end for anyone without a gene name already in mind,
        which is most people opening this for the first time.
        """
        from ..clients.targets import SUGGESTED

        return {"suggested": list(SUGGESTED)}

    @app.get("/api/targets/readiness")
    async def target_readiness(
        q: str = Query(..., min_length=1, max_length=120),
    ) -> dict:
        """Can a run on this target actually do anything?

        Fans out to RCSB and ChEMBL in parallel and answers the two questions that decide
        whether a budget is worth spending: is there a structure with a bound ligand to dock
        into, and are there measured actives to seed the generator from. Asking here costs
        seconds; finding out from a dead run costs the budget.
        """
        return await asyncio.to_thread(_target_readiness, q)

    # --- the generation workbench ------------------------------------------
    #
    # These do not create runs. A run is a budgeted, routed experiment whose purpose is to
    # be compared with another run; a workbench call is one operation on some molecules.
    # Routing a conformer request through a policy would record scratch work in the same
    # place as measured campaigns, which is how a results directory stops being trustworthy.

    @app.post("/api/gen/embed")
    async def gen_embed(spec: EmbedRequest) -> dict:
        """SMILES to 3D. The exact direction: geometry from distance geometry, not a model."""
        from .. import workbench

        def work() -> dict:
            structures, failed = workbench.embed_many(spec.smiles, seed=spec.seed)
            if not structures and failed:
                raise HTTPException(status_code=400, detail=failed[0]["error"])
            return {"structures": structures, "failed": failed}

        return await asyncio.to_thread(work)

    @app.post("/api/gen/elaborate")
    async def gen_elaborate(spec: ElaborateRequest) -> dict:
        """Seed molecules to analogues, by the same operators the generator subsystem uses."""
        from .. import workbench

        def work() -> dict:
            try:
                return workbench.elaborate(
                    spec.seeds,
                    n=spec.n,
                    seed=spec.seed,
                    mutation_rate=spec.mutation_rate,
                )
            except workbench.WorkbenchError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        return await asyncio.to_thread(work)

    @app.post("/api/gen/pocket")
    async def gen_pocket(spec: PocketRequest) -> dict:
        """Prepare a pocket from a PDB entry, so molecules can be scored against it.

        Separate from docking because it is fast, cached, and worth confirming before a
        batch is committed: which structure, which bound ligand marks the site, and how big
        the search box is are all things to look at before spending minutes of Vina.
        """
        from .. import workbench

        def work() -> dict:
            try:
                pocket = workbench.prepare_pocket(spec.pdb_id, spec.ligand_code, settings)
            except workbench.WorkbenchError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return {**pocket.to_dict(), "vina": settings.resolved_vina() is not None}

        return await asyncio.to_thread(work)

    @app.post("/api/gen/dock", status_code=202)
    def gen_dock(spec: DockRequest) -> dict:
        """Score molecules against a pocket. Returns a job; poll it.

        Tens of seconds per molecule means this cannot be an ordinary request. The job holds
        progress while it runs so the page can show which molecule is being docked rather
        than a spinner of unknown duration.
        """
        from .. import workbench

        try:
            pocket = workbench.prepare_pocket(spec.pdb_id, spec.ligand_code, settings)
        except workbench.WorkbenchError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        def work(job: workbench.Job) -> dict:
            job.total = len(spec.smiles)

            def progress(done: int, total: int, smi: str) -> None:
                job.done, job.total = done, total
                job.message = f"docked {done}/{total}"

            rows = workbench.dock_batch(
                spec.smiles,
                pocket,
                seed=spec.seed,
                exhaustiveness=spec.exhaustiveness,
                settings=settings,
                cancel=job.cancel,
                on_progress=progress,
            )
            # The poses stay on the job and are served one at a time. A nine-model pose is
            # several KB, so returning sixty-four of them inline would make the polling
            # response far larger than the page has any use for -- only the selected one is
            # ever rendered.
            job.poses = [r.pop("pose", "") for r in rows]
            return {"pocket": pocket.to_dict(), "rows": rows}

        job = workbench.board().submit("dock", work)
        return {"job_id": job.job_id, "poll": f"/api/gen/jobs/{job.job_id}"}

    @app.get("/api/gen/receptor")
    def gen_receptor(
        pdb_id: str = Query(..., min_length=4, max_length=10),
        ligand_code: str = Query("", max_length=10),
    ) -> Response:
        """The prepared receptor as PDB, so a workbench pose can be seen in its pocket.

        A pose without the structure it was docked into is a cloud of atoms in space, and
        the workbench has no run directory to serve one from.
        """
        from .. import workbench

        try:
            pocket = workbench.prepare_pocket(pdb_id, ligand_code, settings)
        except workbench.WorkbenchError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not pocket.receptor_pdb.is_file():
            raise HTTPException(status_code=404, detail="prepared receptor PDB is missing")
        return Response(
            content=pocket.receptor_pdb.read_text(encoding="utf-8"),
            media_type="chemical/x-pdb",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    @app.get("/api/gen/jobs/{job_id}")
    def gen_job(job_id: str) -> dict:
        from .. import workbench

        job = workbench.board().get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="no such job")
        return job.snapshot()

    @app.get("/api/gen/jobs/{job_id}/pose/{index}")
    def gen_job_pose(job_id: str, index: int, format: str = Query("sdf")) -> Response:
        """One docked pose from a finished job.

        SDF by default and PDBQT only on request, for the same reason the run viewer
        converts: 3Dmol.js parses PDBQT without raising and renders a different molecule
        than the file holds. The browser must never be handed the raw format.
        """
        from .. import workbench
        from ..prep import poses as posemod

        job = workbench.board().get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="no such job")
        if not 0 <= index < len(job.poses):
            raise HTTPException(status_code=404, detail="no pose at that index")
        text = job.poses[index]
        if not text:
            raise HTTPException(status_code=404, detail="this molecule produced no pose")
        if format == "pdbqt":
            return Response(content=text, media_type="chemical/x-pdbqt")
        try:
            return Response(content=posemod.to_sdf(text), media_type="chemical/x-mdl-sdfile")
        except posemod.PoseConvertError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.delete("/api/gen/jobs/{job_id}", status_code=202)
    def gen_job_cancel(job_id: str) -> dict:
        from .. import workbench

        job = workbench.board().get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="no such job")
        job.cancel.set()
        return {"job_id": job_id, "cancelling": True}

    # --- depiction ---------------------------------------------------------
    @app.get("/api/depict")
    def depict(
        smiles: str = Query(..., max_length=600),
        w: int = Query(260, ge=60, le=900),
        h: int = Query(180, ge=60, le=900),
    ) -> Response:
        """Render a molecule as SVG.

        Drawn server-side with the same RDKit that computes the property panel, so the
        picture and the numbers always describe the same perceived molecule -- a browser
        depiction library would occasionally disagree about aromaticity or tautomers.
        """
        svg = _depict_svg(smiles, w, h)
        if svg is None:
            raise HTTPException(status_code=400, detail="unparseable SMILES")
        return Response(
            content=svg,
            media_type="image/svg+xml",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    # --- 3D poses ----------------------------------------------------------
    @app.get("/api/runs/{run_id}/poses")
    def run_poses(run_id: str) -> dict:
        """Which docked poses this run produced, best score first.

        An entry whose ``file`` is null was scored before the pose store existed, or was
        a cache hit from such a run. It is listed rather than dropped so the count here
        matches the number of docked molecules in the trace -- a viewer that silently
        showed fewer poses than the run docked would look like it had lost some.
        """
        # An existing run that has not docked yet answers with an empty list; a run_id that
        # was never a run is a 404. Same distinction the LLM log draws, and for the same
        # reason -- "nothing to show" and "no such thing" are different answers.
        if manager.get(run_id) is None and not (settings.runs_dir / run_id).is_dir():
            raise HTTPException(status_code=404, detail="no such run")

        index = posestore.read_index(settings.runs_dir / run_id / "poses")
        return {
            "run_id": run_id,
            "bucket": index["bucket"],
            "receptor": index["receptor"],
            "poses": index["poses"],
            "n_with_file": sum(1 for p in index["poses"] if p.get("file")),
        }

    @app.get("/api/runs/{run_id}/poses/{name}")
    def run_pose_file(
        run_id: str,
        name: str,
        format: Literal["sdf", "pdbqt"] = Query("sdf"),
    ) -> Response:
        """One pose ensemble, as SDF by default or as the raw PDBQT Vina wrote.

        SDF is the default because it is the format a browser viewer can be trusted with.
        3Dmol.js accepts PDBQT without error and returns a different molecule than the file
        contains, so serving PDBQT to the page by default would mean rendering a structure
        nobody could tell was wrong. ``format=pdbqt`` returns the original bytes, which is
        what provenance and any external tool should use.

        ``name`` is resolved against this run's own index rather than used as a path, so a
        traversal attempt cannot reach outside the store: a name that the run did not
        record is a 404 regardless of what it contains.
        """
        poses_dir = settings.runs_dir / run_id / "poses"
        index = posestore.read_index(poses_dir)
        known = {p.get("file") for p in index["poses"] if p.get("file")}
        if name not in known or not index["bucket"]:
            raise HTTPException(status_code=404, detail="no such pose in this run")

        path = settings.pose_cache_dir / index["bucket"] / name
        if not path.is_file():
            raise HTTPException(status_code=404, detail="pose file is missing from the store")

        pdbqt = path.read_text(encoding="utf-8")
        if format == "pdbqt":
            return Response(
                content=pdbqt,
                media_type="chemical/x-pdbqt",
                headers={"Cache-Control": "public, max-age=86400"},
            )
        try:
            sdf = posestore.to_sdf(pdbqt)
        except posestore.PoseConvertError as exc:
            # 422, not 500: the file is intact and readable, it just cannot be expressed as
            # SDF -- most likely a pose not written by this project's Meeko path.
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return Response(
            content=sdf,
            media_type="chemical/x-mdl-sdfile",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    @app.get("/api/runs/{run_id}/receptor")
    def run_receptor(run_id: str) -> Response:
        """The prepared receptor this run docked into, as PDB.

        PDB rather than the PDBQT that Vina consumed: every browser viewer parses PDB, and
        the partial charges and atom types PDBQT adds are meaningless to a renderer.

        Read from the pose index rather than from run state, so a finished run answers the
        same way a live one does -- the viewer should not need to know which it is looking
        at.
        """
        poses_dir = settings.runs_dir / run_id / "poses"
        rec = posestore.read_index(poses_dir)["receptor"]
        if not rec:
            live = manager.get(run_id)
            rec = live.state.facts.get("receptor") if live else None
        if not rec:
            raise HTTPException(status_code=404, detail="this run has no prepared receptor")

        pdbqt = Path(rec.get("pdbqt") or "")
        pdb = pdbqt.with_name("receptor.pdb") if pdbqt.name else None
        if not pdb or not pdb.is_file():
            raise HTTPException(status_code=404, detail="receptor file is no longer on disk")
        return Response(
            content=pdb.read_text(encoding="utf-8"),
            media_type="chemical/x-pdb",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    # --- runs --------------------------------------------------------------
    @app.post("/api/runs", status_code=201)
    async def create_run(spec: CreateRun) -> dict:
        if spec.policy not in POLICY_NAMES:
            raise HTTPException(status_code=400, detail=f"unknown policy {spec.policy!r}")
        if spec.policy == "llm":
            # Gate on the configured provider, not on Anthropic specifically: a ChatGPT
            # install has no API key and must still be allowed to run.
            llm = _llm_readiness()
            if not llm["ready"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"the llm policy is not ready: {llm['detail']}. "
                    "Sign in or set a key in the web interface, or choose a rule-based "
                    "policy.",
                )
        tools = _resolve_tools(spec.tools, settings)
        if not tools:
            raise HTTPException(
                status_code=400,
                detail="a run needs at least one subsystem; none of the tools requested "
                "are installed",
            )
        live = manager.create(
            goal=spec.goal,
            target=spec.target,
            policy=spec.policy,
            budget_units=spec.budget_units,
            seed=spec.seed,
            max_steps=spec.max_steps,
            tools=tools,
            prompt_extra=spec.prompt_extra,
            prompt_override=spec.prompt_override,
        )
        await manager.start(live)
        return {
            "run_id": live.run_id,
            "state_url": f"/api/runs/{live.run_id}",
            "events_url": f"/api/runs/{live.run_id}/events",
        }

    @app.get("/api/runs")
    def list_runs(limit: int = Query(50, ge=1, le=200)) -> dict:
        return {"live": manager.listing(limit), "finished": _finished_runs(limit)}

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        snap = manager.snapshot(run_id)
        if snap is None:
            snap = _load_finished(run_id)
        if snap is None:
            raise HTTPException(status_code=404, detail="no such run")
        return snap

    @app.get("/api/runs/{run_id}/events.json")
    def events_json(
        run_id: str,
        from_seq: int = Query(0, ge=0),
        limit: int = Query(2000, ge=1, le=20000),
    ) -> dict:
        """Polling fallback for clients that cannot hold an SSE connection."""
        live = manager.get(run_id)
        if live is not None:
            evs = [e.to_dict() for e in live.log.since(from_seq)][:limit]
            return {"events": evs, "last_seq": live.log.last_seq}
        recorded = _recorded_events(run_id)
        if recorded is None:
            raise HTTPException(status_code=404, detail="no such run")
        evs = [e for e in recorded if e.get("seq", 0) > from_seq][:limit]
        return {"events": evs, "last_seq": recorded[-1]["seq"] if recorded else 0}

    @app.post("/api/runs/{run_id}/cancel", status_code=202)
    async def cancel_run(run_id: str) -> dict:
        ok = await manager.cancel(run_id)
        if not ok:
            raise HTTPException(status_code=404, detail="no such live run")
        return {"run_id": run_id, "cancelled": True}

    @app.get("/api/runs/{run_id}/events")
    async def stream_events(
        run_id: str,
        request: Request,
        from_seq: int = Query(0, ge=0),
    ) -> StreamingResponse:
        live = manager.get(run_id)
        if live is None:
            if _recorded_events(run_id) is None:
                raise HTTPException(status_code=404, detail="no such run")
            return StreamingResponse(
                _replay_recorded(run_id, from_seq), media_type="text/event-stream"
            )

        # The browser's own reconnect sends Last-Event-ID; an explicit from_seq wins when
        # both are present, because a reloading page knows better than the stale header.
        header = request.headers.get("last-event-id")
        start = from_seq
        if start == 0 and header:
            with contextlib.suppress(ValueError):
                start = int(header)

        sub = manager.subscribe(run_id)
        if sub is None:
            raise HTTPException(status_code=404, detail="no such run")

        async def gen():
            try:
                # Backlog first, so a late joiner sees the whole trace, then live events.
                for ev in live.log.since(start):
                    yield ev.to_sse()
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        ev = await asyncio.wait_for(sub.queue.get(), timeout=HEARTBEAT_S)
                    except TimeoutError:
                        yield ": ping\n\n"
                        if live.state.status in ("done", "cancelled", "failed"):
                            break
                        continue
                    if ev.seq > start:
                        yield ev.to_sse()
                    if ev.type in ("run_finished", "run_cancelled"):
                        break
            finally:
                manager.unsubscribe(run_id, sub)

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                # Without this an nginx or similar in front would buffer the stream and the
                # live trace would arrive all at once at the end.
                "X-Accel-Buffering": "no",
            },
        )

    # --- static ------------------------------------------------------------
    web_dir = settings.web_dir
    if web_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(str(web_dir / "index.html"))

    return app


# --- helpers ---------------------------------------------------------------
_PROVIDER_LABEL = {
    "anthropic": "Claude",
    "openai": "GPT",
    "chatgpt": "GPT (your ChatGPT plan)",
}


def _resolve_tools(requested: list[str] | None, settings) -> tuple[str, ...]:
    """Which subsystems a request actually gets, intersected with what is installed.

    ``None`` -- a client that does not know about the tool picker -- means everything, so
    the behaviour is unchanged for anything that predates it. An explicit list is filtered
    against the installation rather than trusted: a name that is not installed would
    otherwise produce a run missing a tool its record claims it had.
    """
    installed = [c.key for c in build_registry(settings).capabilities()]
    if requested is None:
        return tuple(installed)
    want = {str(t) for t in requested}
    return tuple(k for k in installed if k in want)


def _policy_description(name: str, settings) -> str:
    """The llm arm's blurb names whichever model is actually configured.

    Leaving it as "Claude" under an OpenAI provider would misreport the arm under test in
    the one place a user looks before choosing it.
    """
    base = DESCRIPTIONS.get(name, "")
    if name != "llm":
        return base
    who = _PROVIDER_LABEL.get(settings.llm_provider, "The model")
    return base.replace("Claude", who, 1)


def _chatgpt_status() -> dict:
    """Who is signed in, if anyone. Never the tokens."""
    from ..llm.chatgpt_auth import SessionStore

    settings = get_settings()
    session = SessionStore(settings.chatgpt_session_path).load()
    if session is None:
        return {"signed_in": False, "provider": settings.llm_provider}
    return {**session.public(), "provider": settings.llm_provider}


def _llm_readiness() -> dict:
    """Whether the `llm` policy can actually run right now, and on what credential."""
    settings = get_settings()
    provider = settings.llm_provider
    if provider == "anthropic":
        return {
            "ready": settings.api_key() is not None,
            "detail": "Anthropic API key" if settings.api_key() else "no Anthropic API key",
            "model": settings.llm_model,
            "effort": settings.effort_for_provider(),
        }
    if provider == "openai":
        return {
            "ready": settings.openai_key() is not None,
            "detail": "OpenAI API key" if settings.openai_key() else "no OPENAI_API_KEY",
            "model": settings.openai_model,
            "effort": settings.effort_for_provider(),
        }
    status = _chatgpt_status()
    if not (status.get("signed_in") and status.get("authorized_for_plan")):
        return {
            "ready": False,
            "detail": (
                "signed in, but plan usage was not granted"
                if status.get("signed_in")
                else "not signed in to ChatGPT"
            ),
            "model": settings.openai_model,
            "effort": settings.effort_for_provider(),
        }

    # Signed in is not the same as runnable. The plan serves a different model set from the
    # public API, so a model configured for one route can be absent on the other -- and a
    # model the account cannot use fails on the first decision as a silent fallback.
    from ..llm.catalog import live_models

    available = live_models("chatgpt", settings)
    if available is not None and not any(m.id == settings.openai_model for m in available):
        return {
            "ready": False,
            "detail": f"{settings.openai_model} is not available on this plan. Pick another model",
            "model": settings.openai_model,
            "effort": settings.effort_for_provider(),
            "choices": [m.id for m in available][:12],
        }
    return {
        "ready": True,
        "detail": f"ChatGPT plan ({status.get('account_email')})",
        "model": settings.openai_model,
        "effort": settings.effort_for_provider(),
    }


def _bank_client():
    from ..clients.http import BankClient

    settings = get_settings()
    return BankClient(settings.http_cache_dir, offline=settings.offline)


def _classify(q: str) -> str:
    from ..clients.targets import classify

    return classify(q)


def _search_targets(q: str, limit: int = 8) -> list[dict]:
    """Autocomplete rows. Never raises: a dead bank should empty the list, not 500 the page."""
    from ..clients.targets import suggest

    try:
        return [h.to_dict() for h in suggest(_bank_client(), q, limit=limit)]
    except Exception:  # noqa: BLE001
        return []


def _target_readiness(q: str) -> dict:
    from ..clients.targets import readiness

    try:
        return readiness(_bank_client(), q).to_dict()
    except Exception as exc:  # noqa: BLE001
        return {
            "uniprot": "",
            "errors": {"search": f"{type(exc).__name__}: {exc}"},
            "verdict": {"level": "error", "text": f"Could not reach the data banks: {exc}"},
        }


def _depict_svg(smiles: str, w: int, h: int) -> str | None:
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D

    from ..chem import to_mol

    mol = to_mol(smiles)
    if mol is None:
        return None
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(w, h)
    opts = drawer.drawOptions()
    opts.clearBackground = False  # let the page's theme show through
    opts.bondLineWidth = 1
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def _run_dir(run_id: str) -> Path:
    return get_settings().runs_dir / run_id


def _recorded_events(run_id: str) -> list[dict] | None:
    p = _run_dir(run_id) / "events.jsonl"
    if not p.is_file():
        return None
    from ..orchestrator.events import EventLog

    try:
        return EventLog.read_jsonl(p)
    except Exception:
        return None


def _recorded_decisions(run_id: str) -> list[dict] | None:
    p = _run_dir(run_id) / "decisions.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _recorded_llm(run_id: str) -> list[dict] | None:
    p = _run_dir(run_id) / "llm_calls.jsonl"
    if not p.is_file():
        # Absent means the run predates the format, not that it made no calls -- so an
        # existing run directory still answers, with an empty list.
        return [] if _run_dir(run_id).is_dir() else None
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            with contextlib.suppress(json.JSONDecodeError):
                out.append(json.loads(line))
    return out


def _llm_totals_from(calls: list[dict]) -> dict:
    live = [c for c in calls if not c.get("cached")]
    lat = [c["latency_ms"] for c in live if c.get("latency_ms")]

    def _sum(field: str) -> int:
        return sum(int(c.get(field) or 0) for c in live)

    return {
        "input_tokens": _sum("input_tokens"),
        "output_tokens": _sum("output_tokens"),
        "cache_read_input_tokens": _sum("cache_read_input_tokens"),
        "cache_creation_input_tokens": _sum("cache_creation_input_tokens"),
        "n_live_calls": len(live),
        "n_cached_calls": len(calls) - len(live),
        "total_latency_ms": round(sum(lat), 1),
        "mean_latency_ms": round(sum(lat) / len(lat), 1) if lat else None,
    }


def _recorded_policy_name(run_id: str) -> str:
    """Which arm a finished run used.

    From ``run.json``, because ``summary.json`` does not carry it -- and getting this wrong
    is not cosmetic: the LLM panel keys off it, so a missing name made a finished LLM run
    report itself as rule-based, which is the opposite of the truth about which arm ran.
    """
    p = _run_dir(run_id) / "run.json"
    if p.is_file():
        try:
            return str(json.loads(p.read_text(encoding="utf-8")).get("policy", {}).get("name", ""))
        except Exception:
            pass
    return str((_load_finished(run_id) or {}).get("policy", "") or "")


def _load_finished(run_id: str) -> dict | None:
    p = _run_dir(run_id) / "summary.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _finished_runs(limit: int) -> list[dict]:
    root = get_settings().runs_dir
    if not root.is_dir():
        return []
    rows: list[dict] = []
    for d in sorted(root.iterdir(), key=lambda p: -p.stat().st_mtime)[: limit * 2]:
        s = _load_finished(d.name)
        if s:
            rows.append(s)
        if len(rows) >= limit:
            break
    return rows


async def _replay_recorded(run_id: str, from_seq: int):
    """Serve a finished run's recorded events as a stream, so one client handles both."""
    for ev in _recorded_events(run_id) or []:
        if ev.get("seq", 0) <= from_seq:
            continue
        yield f"id: {ev['seq']}\nevent: {ev['type']}\ndata: {json.dumps(ev, default=str)}\n\n"


def _json(obj: Any) -> JSONResponse:  # pragma: no cover
    return JSONResponse(obj)


app = create_app()
