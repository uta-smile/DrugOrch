"""Command line entry points.

Every subsystem is reachable from here without the web server, so a failure can be isolated
to one layer: if ``drugorch call properties`` works and the browser does not, the problem is
in the API, not the chemistry.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from .config import get_settings
from .versions import all_versions

app = typer.Typer(
    name="drugorch",
    help="LLM-orchestrated multi-subsystem drug discovery.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

OK = "[green]ok[/green]"
WARN = "[yellow]warn[/yellow]"
FAIL = "[red]fail[/red]"


def _check(name: str, fn) -> tuple[str, str, str]:
    """Run one probe, converting any failure into a row rather than a traceback."""
    try:
        status, detail = fn()
        return name, status, detail
    except Exception as exc:  # noqa: BLE001 - doctor reports, it does not crash
        return name, FAIL, f"{type(exc).__name__}: {exc}"


@app.command()
def doctor() -> None:
    """Check that this machine can actually run DrugOrch, and say what is missing."""
    st = get_settings()
    st.ensure_dirs()
    rows: list[tuple[str, str, str]] = []

    rows.append(("python", OK, f"{sys.version.split()[0]} at {sys.executable}"))

    def _rdkit():
        import rdkit

        from .chem import canonical

        probe = canonical("CC(=O)Oc1ccccc1C(=O)[O-].[Na+]")
        expected = "CC(=O)Oc1ccccc1C(=O)O"
        if probe != expected:
            return FAIL, f"standardizer returned {probe!r}, expected {expected!r}"
        return OK, f"{rdkit.__version__}; salt stripped and neutralized correctly"

    rows.append(_check("rdkit", _rdkit))

    def _sa():
        from .subsystems.properties import _sascorer

        score = _sascorer().calculateScore(__import__("rdkit.Chem", fromlist=["Chem"]).MolFromSmiles("CCO"))
        return OK, f"contrib sascorer importable (ethanol SA={score:.2f})"

    rows.append(_check("sa_score", _sa))

    def _panel():
        from .subsystems.properties import compute_panel

        p = compute_panel("CC(=O)Oc1ccccc1C(=O)O")
        if p is None:
            return FAIL, "panel returned None for aspirin"
        return OK, f"aspirin QED={p['scores']['qed']:.3f} SA={p['scores']['sa_score']:.2f}"

    rows.append(_check("properties", _panel))

    def _gemmi():
        import gemmi

        return OK, f"{gemmi.__version__}"

    rows.append(_check("gemmi", _gemmi))

    def _meeko():
        import meeko

        return OK, f"{getattr(meeko, '__version__', 'unknown')}"

    rows.append(_check("meeko", _meeko))

    def _vina():
        exe = st.resolved_vina()
        if exe is None:
            return (
                FAIL,
                f"not found at {st.vina_exe} or on PATH -- docking will be unavailable",
            )
        out = subprocess.run(
            [str(exe), "--version"], capture_output=True, text=True, timeout=30
        )
        first = (out.stdout or out.stderr).strip().splitlines()
        return OK, f"{first[0] if first else 'ran'} ({exe})"

    rows.append(_check("autodock vina", _vina))

    def _anthropic():
        import anthropic

        if st.api_key() is None:
            return (
                WARN,
                f"SDK {anthropic.__version__} installed, but no API key is set -- "
                "enter one in the web interface, or put ANTHROPIC_API_KEY in .env. "
                "Rule-based policies still work; the LLM policy will not",
            )
        return OK, f"SDK {anthropic.__version__}, API key present ({st.api_key_source()})"

    rows.append(_check("anthropic", _anthropic))

    def _dirs():
        missing = [str(p) for p in (st.data_dir, st.cache_dir, st.runs_dir) if not p.is_dir()]
        if missing:
            return FAIL, f"missing: {', '.join(missing)}"
        return OK, f"data dir {st.data_dir}"

    rows.append(_check("directories", _dirs))

    def _registry():
        from .subsystems.registry import build_registry

        reg = build_registry(st)
        return OK, f"{len(reg)} subsystem(s): {', '.join(reg.keys)}"

    rows.append(_check("subsystems", _registry))

    table = Table(title="drugorch doctor", show_lines=False)
    table.add_column("check", style="bold")
    table.add_column("status")
    table.add_column("detail", overflow="fold")
    for name, status, detail in rows:
        table.add_row(name, status, detail)
    console.print(table)

    console.print("\n[bold]recipe versions[/bold]")
    console.print(json.dumps(all_versions(), indent=2))

    failed = [r[0] for r in rows if r[1] == FAIL]
    warned = [r[0] for r in rows if r[1] == WARN]
    if failed:
        console.print(f"\n[red]{len(failed)} check(s) failed:[/red] {', '.join(failed)}")
        raise typer.Exit(code=1)
    if warned:
        console.print(f"\n[yellow]{len(warned)} warning(s):[/yellow] {', '.join(warned)}")
    console.print("\n[green]all required checks passed[/green]")


@app.command("manifest")
def manifest() -> None:
    """Print the capability manifest exactly as the routing policy reads it."""
    from .subsystems.registry import build_registry

    reg = build_registry()
    console.print(reg.manifest())


@app.command("run")
def run_cmd(
    goal: str = typer.Option(..., "--goal", "-g", help="What the run is trying to achieve."),
    target: str = typer.Option(..., "--target", "-t", help="Gene name or UniProt accession."),
    policy: str = typer.Option("scripted", "--policy", "-p", help="Routing policy."),
    budget: float = typer.Option(120.0, "--budget", "-b", help="Budget units."),
    seed: int = typer.Option(42, "--seed", help="Seed; arms are paired on it."),
    max_steps: int = typer.Option(40, "--max-steps"),
    run_id: str = typer.Option(None, "--run-id", help="Defaults to a timestamped id."),
    quiet: bool = typer.Option(False, "--quiet", help="Only print the final summary."),
) -> None:
    """Run the orchestrator headlessly and write a full run directory."""
    import time as _time

    from .budget import BudgetLedger
    from .orchestrator.events import EventLog
    from .orchestrator.run import Orchestrator, drive
    from .orchestrator.state import RunState
    from .policy import POLICY_NAMES, build_policy
    from .store.runs import summarize, write_run_dir
    from .subsystems.registry import build_registry
    from .types import TargetSpec

    if policy not in POLICY_NAMES:
        console.print(f"[red]unknown policy {policy!r}[/red]; choose from {', '.join(POLICY_NAMES)}")
        raise typer.Exit(code=2)

    st = get_settings()
    st.ensure_dirs()
    rid = run_id or f"r_{_time.strftime('%Y%m%d_%H%M%S')}_{policy}_{seed}"

    state = RunState(
        run_id=rid,
        goal=goal,
        target=TargetSpec(query=target),
        budget=BudgetLedger(total=budget),
        seed=seed,
    )
    registry = build_registry(st)
    log = EventLog(rid, path=st.runs_dir / rid / "events.jsonl")
    pol = build_policy(policy, seed=seed, settings=st)

    orch = Orchestrator(state, registry, pol, log, settings=st)
    orch.termination.max_steps = max_steps

    colors = {
        "retrieval": "cyan",
        "properties": "green",
        "docking": "magenta",
        "generation": "yellow",
    }

    def on_event(ev) -> None:
        if quiet:
            return
        d = ev.data
        if ev.type == "decision":
            sub = d.get("subsystem") or "(stop)"
            c = colors.get(sub, "white")
            spent = ev.budget["spent"]
            console.print(
                f"\n[bold]step {ev.step}[/bold] [dim]{spent:6.1f}/{ev.budget['total']:.0f}u[/dim]"
                f"  ->  [{c}]{sub}[/{c}]"
            )
            console.print(f"  [dim]why:[/dim] {escape(d.get('rationale', ''))}")
            for alt in d.get("considered", [])[:3]:
                console.print(
                    f"  [dim]  x {alt['subsystem']:<12s} {alt['score']:.2f}  "
                    f"{escape(alt['reason'][:62])}[/dim]"
                )
        elif ev.type == "call_finished":
            mark = "[green]ok[/green]" if d.get("ok") else "[red]failed[/red]"
            console.print(
                f"  {mark} [dim]{d.get('wall_s', 0):.1f}s, "
                f"{d.get('cost', {}).get('budget_units', 0):.2f}u[/dim]"
                f"  {escape(str(d.get('note') or d.get('error') or ''))[:100]}"
            )
        elif ev.type == "best_updated":
            console.print(f"  [bold green]new best[/bold green] {d['objective']} = {d['best']}")
        elif ev.type in ("budget_block", "subsystem_error", "warning", "policy_fallback"):
            console.print(f"  [yellow]{ev.type}:[/yellow] {escape(str(d.get('reason') or d.get('message') or ''))[:120]}")
        elif ev.type in ("run_finished", "run_cancelled"):
            console.print(f"\n[bold]{ev.type}[/bold]: {escape(d.get('reason', ''))}")

    log.subscribe(on_event)
    drive(orch)

    out = write_run_dir(
        st.runs_dir,
        state,
        policy_describe=pol.describe(),
        registry_describe=registry.describe(),
        settings_describe=st.describe(),
        termination={
            "code": orch.stop_reason.code if orch.stop_reason else "",
            "reason": orch.stop_reason.reason if orch.stop_reason else "",
        },
    )

    s = summarize(state)
    t = Table(title=f"run {rid}")
    t.add_column("metric", style="bold")
    t.add_column("value")
    t.add_row("policy", policy)
    t.add_row("status", s["status"])
    t.add_row("steps", str(s["steps"]))
    t.add_row("budget", f"{s['budget']['spent']:.1f} / {s['budget']['total']:.0f} units")
    t.add_row("molecules", str(s["n_molecules"]))
    t.add_row(f"best {s['objective']}", str(s["best"]))
    t.add_row("fallbacks", str(s["n_fallbacks"]))
    t.add_row("elapsed", f"{s['elapsed_s']:.1f}s")
    console.print(t)

    if s["by_subsystem"]:
        t2 = Table(title="spend by subsystem")
        t2.add_column("subsystem")
        t2.add_column("calls", justify="right")
        t2.add_column("failed", justify="right")
        t2.add_column("units", justify="right")
        t2.add_column("wall s", justify="right")
        for k, v in sorted(s["by_subsystem"].items(), key=lambda kv: -kv[1]["units"]):
            t2.add_row(k, str(v["calls"]), str(v["failed"]), f"{v['units']:.2f}", f"{v['wall_s']:.1f}")
        console.print(t2)

    console.print(f"\n[dim]wrote {out}[/dim]")


@app.command("target")
def target_cmd(
    query: str = typer.Argument(..., help="Gene name, protein name, or UniProt accession."),
    actives: int = typer.Option(10, "--actives", help="How many known actives to show."),
) -> None:
    """Resolve a target across all four data banks and show what docking would use."""
    import threading

    from .budget import BudgetLedger
    from .orchestrator.state import RunState
    from .subsystems.registry import build_registry
    from .types import TargetSpec

    reg = build_registry()
    sub = reg.get("retrieval")
    if sub is None:
        console.print("[red]retrieval subsystem is not available[/red]")
        raise typer.Exit(code=2)

    state = RunState(
        run_id="cli",
        goal=f"resolve {query}",
        target=TargetSpec(query=query),
        budget=BudgetLedger(total=100.0),
    )
    params = sub.capability.coerce_params({"fetch": "all", "n_actives": max(actives, 10)})

    def progress(msg: str, frac: float) -> None:
        console.print(f"[dim]  {frac * 100:5.1f}%  {msg}[/dim]")

    result = sub.run(params, state, cancel=threading.Event(), progress=progress)
    if not result.ok:
        console.print(f"[red]failed:[/red] {result.error}")
        raise typer.Exit(code=1)

    ident = result.facts.get("target_identity", {})
    console.print(f"\n[bold]{ident.get('gene') or query}[/bold]  {ident.get('name', '')}")
    console.print(
        f"  UniProt [cyan]{ident.get('uniprot')}[/cyan] · {ident.get('organism')} · "
        f"{ident.get('length')} aa · ChEMBL [cyan]{ident.get('chembl_id')}[/cyan]"
    )

    shortlist = result.facts.get("pdb_shortlist") or []
    if shortlist:
        t = Table(title="dockable structures (ranked)")
        t.add_column("pdb")
        t.add_column("score", justify="right")
        t.add_column("why", overflow="fold")
        for c in shortlist:
            t.add_row(c["pdb_id"], f"{c['score']:.2f}", c["reason"])
        console.print(t)
    else:
        console.print("[yellow]no dockable structures found[/yellow]")

    if result.records:
        ka = result.facts.get("known_actives", {})
        t = Table(title=f"known actives ({ka.get('n', 0)} total, strongest first)")
        t.add_column("pChEMBL", justify="right")
        t.add_column("chembl id")
        t.add_column("smiles", overflow="fold", max_width=58)
        for r in result.records[:actives]:
            t.add_row(
                f"{r['scores']['pchembl']:.2f}", r["props"].get("chembl_id", ""), escape(r["smiles"])
            )
        console.print(t)

    console.print(f"\n[dim]{result.note}[/dim]")
    console.print(
        f"[dim]wall {result.cost.wall_s:.1f}s · {result.cost.api_calls} requests · "
        f"{result.cost.cache_hits} cache hits[/dim]"
    )


@app.command("call")
def call_subsystem(
    subsystem: str = typer.Argument(..., help="Subsystem key, e.g. 'properties'."),
    smiles: list[str] = typer.Option(None, "--smiles", "-s", help="SMILES; repeatable."),
    smiles_file: Path = typer.Option(None, "--smiles-file", help="One SMILES per line."),
    params: str = typer.Option("{}", "--params", help="Extra params as a JSON object."),
    budget: float = typer.Option(1000.0, "--budget", help="Budget units for this call."),
    out: Path = typer.Option(None, "--out", help="Write the full result here as JSON."),
) -> None:
    """Invoke one subsystem directly, with no orchestrator and no policy."""
    import threading

    from .budget import BudgetLedger
    from .chem import dedupe
    from .orchestrator.state import RunState
    from .subsystems.registry import build_registry
    from .types import TargetSpec

    reg = build_registry()
    sub = reg.get(subsystem)
    if sub is None:
        console.print(f"[red]unknown subsystem {subsystem!r}[/red]; have: {', '.join(reg.keys)}")
        raise typer.Exit(code=2)

    smi: list[str] = list(smiles or [])
    if smiles_file:
        smi += [ln.strip() for ln in smiles_file.read_text(encoding="utf-8").splitlines() if ln.strip()]

    extra = json.loads(params)
    merged = sub.capability.coerce_params({"smiles": dedupe(smi), **extra})

    state = RunState(
        run_id="cli",
        goal="direct subsystem call",
        target=TargetSpec(query="n/a"),
        budget=BudgetLedger(total=budget),
    )
    state.add_records([{"smiles": s} for s in dedupe(smi)], origin="cli")

    usable, why = sub.preflight(merged, state)
    if not usable:
        console.print(f"[yellow]preflight refused:[/yellow] {why}")
        raise typer.Exit(code=1)

    est = sub.estimate_cost(merged, state)
    console.print(f"[dim]estimated cost: {est.budget_units:.2f} units, ~{est.wall_s:.1f}s[/dim]")

    def progress(msg: str, frac: float) -> None:
        console.print(f"[dim]  {frac * 100:5.1f}%  {msg}[/dim]")

    result = sub.run(merged, state, cancel=threading.Event(), progress=progress)

    status = "[green]ok[/green]" if result.ok else "[red]failed[/red]"
    console.print(f"\n{status}  {result.note or ''}")
    if result.error:
        console.print(f"[red]error:[/red] {result.error}")

    if result.records:
        table = Table(show_lines=False)
        table.add_column("smiles", overflow="fold", max_width=44)
        keys: list[str] = []
        for r in result.records[:1]:
            keys = list(r.get("scores", {})) + ["mw", "logp", "tpsa", "lipinski_violations"]
        for k in keys:
            table.add_column(k, justify="right")
        for r in result.records[:25]:
            vals = []
            for k in keys:
                v = r.get("scores", {}).get(k, r.get("props", {}).get(k))
                vals.append("" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v)))
            table.add_row(escape(r["smiles"]), *vals)
        console.print(table)
        if len(result.records) > 25:
            console.print(f"[dim]... {len(result.records) - 25} more[/dim]")

    if result.facts:
        console.print("\n[bold]facts[/bold]: " + ", ".join(sorted(result.facts)))

    console.print(
        f"\n[dim]wall {result.cost.wall_s:.2f}s · {result.cost.n_unique_smiles} molecules · "
        f"{result.cost.cache_hits} cache hits[/dim]"
    )

    if out:
        payload = {
            "subsystem": result.subsystem,
            "ok": result.ok,
            "note": result.note,
            "error": result.error,
            "cost": result.cost.to_dict(),
            "records": result.records,
            "facts": result.facts,
            "describe": sub.describe(),
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[dim]wrote {out}[/dim]")


@app.command("bench")
def bench_cmd(
    targets: str = typer.Option(
        "O60674,P24941", "--targets", help="Comma-separated accessions or gene names."
    ),
    policies: str = typer.Option(
        "fixed,scripted", "--policies", help="Comma-separated routing arms to compare."
    ),
    seeds: str = typer.Option("1,2", "--seeds", help="Comma-separated seeds; arms are paired on these."),
    budget: float = typer.Option(120.0, "--budget", "-b"),
    max_steps: int = typer.Option(40, "--max-steps"),
    holdout: float = typer.Option(0.3, "--holdout", help="Fraction of scaffolds withheld."),
    out: Path = typer.Option(None, "--out", help="Where to write the grid; defaults under data/."),
    metric: str = typer.Option("discovery_auc", "--metric"),
    rerun: bool = typer.Option(False, "--rerun", help="Ignore finished cells and run them again."),
) -> None:
    """Run the benchmark grid and print the leaderboard.

    Every arm on a (target, seed) faces the same withheld scaffolds, so the comparison is
    paired. Finished cells are rescored rather than re-run unless --rerun is given.
    """
    from .harness.grid import run_grid, save
    from .harness.report import console_table, leaderboard, per_target

    st = get_settings()
    st.ensure_dirs()
    tl = [t.strip() for t in targets.split(",") if t.strip()]
    pl = [p.strip() for p in policies.split(",") if p.strip()]
    sl = [int(x) for x in seeds.split(",") if x.strip()]

    total = len(tl) * len(pl) * len(sl)
    console.print(
        f"[bold]{total} cells[/bold]: {len(tl)} target(s) x {len(pl)} arm(s) x {len(sl)} seed(s), "
        f"{budget:g} units each, {holdout:.0%} of scaffolds withheld"
    )

    done = {"n": 0}

    def on_cell(cell, row, status):
        done["n"] += 1
        mark = "[green]ok[/green]" if row else "[red]fail[/red]"
        extra = ""
        if row:
            extra = f"auc={row.discovery_auc:.3f} redisc={row.rediscovery:.2f}"
            if row.best_score is not None:
                extra += f" best={row.best_score:.2f}"
        console.print(
            f"  [{done['n']}/{total}] {mark} {cell.target} {cell.policy} seed={cell.seed} "
            f"[dim]{status}[/dim] {extra}"
        )

    result = run_grid(
        tl, pl, sl, budget=budget, max_steps=max_steps, holdout=holdout,
        settings=st, resume=not rerun, on_cell=on_cell,
    )

    path = Path(out) if out else st.data_dir / "bench" / "grid.json"
    save(result, path)

    board = leaderboard(result.rows, metric=metric)
    console.print()
    console.print(console_table(board, metric=metric))
    console.print()
    console.print("[bold]per target[/bold]")
    for tgt, by_policy in per_target(result.rows, metric=metric).items():
        best = max(by_policy, key=lambda k: by_policy[k]) if by_policy else "-"
        cells = "  ".join(f"{p}={v:.3f}" for p, v in sorted(by_policy.items()))
        console.print(f"  {tgt:10} {cells}   [dim]best: {best}[/dim]")

    if result.failures:
        console.print(f"\n[yellow]{len(result.failures)} failure(s)[/yellow]")
        for f in result.failures[:6]:
            console.print(f"  {f.get('target','?')} {f.get('policy','')} "
                          f"[dim]{f.get('stage')}: {str(f.get('error'))[:90]}[/dim]")
    console.print(f"\nwrote {path} and {path.with_suffix('.csv')}")



@app.command("extbench")
def extbench_cmd(
    n: int = typer.Option(200, "--n", help="Instances to sample, split evenly across tasks."),
    arms: str = typer.Option(
        "drugorch,notool", "--arms", help="Comma-separated arms: drugorch, notool."
    ),
    seed: int = typer.Option(0, "--seed", help="Picks the sample and seeds the generator."),
    workers: int = typer.Option(4, "--workers", help="Instances answered in parallel."),
    rerun: bool = typer.Option(False, "--rerun", help="Ignore finished attempts on disk."),
    score_only: bool = typer.Option(
        False, "--score-only", help="Re-score what is on disk without answering anything."
    ),
    out: Path = typer.Option(None, "--out", help="Where to write the board."),
) -> None:
    """Run the public benchmark (MolDesignBench) and print the comparison.

    Answers are scored by the benchmark's own evaluator, in the environment its answer key
    was built in. Published rows come from the paper and are labelled as read rather than
    measured. Finished attempts are reused unless --rerun is given.
    """
    from .harness.external import baselines
    from .harness.external.run import run_external, save

    st = get_settings()
    st.ensure_dirs()
    arm_list = [a.strip() for a in arms.split(",") if a.strip()]

    console.print(
        f"[bold]{n} instances[/bold] x {len(arm_list)} arm(s), "
        f"sampled at seed {seed}; answers scored by MolDesignBench's own evaluator"
    )

    done = {"n": 0}
    total = n * len(arm_list)

    def on_attempt(kind, inst, att):
        done["n"] += 1
        mark = "[dim]cached[/dim]" if kind == "cached" else (
            "[red]fallback[/red]" if att.n_fallbacks else "[green]ok[/green]"
        )
        pred = "None" if att.prediction == "None" else escape(att.prediction[:34])
        console.print(
            f"  [{done['n']}/{total}] {mark} {att.arm:9} {inst.id:20} "
            f"[dim]{att.best_satisfied}/{att.best_total} constraints, "
            f"{att.n_tool_calls} calls[/dim] {pred}"
        )

    result = run_external(
        n=n,
        arms=arm_list,
        seed=seed,
        workers=workers,
        resume=not rerun,
        offline=score_only,
        settings=st,
        on_attempt=None if score_only else on_attempt,
    )

    path = Path(out) if out else st.data_dir / "bench" / "external.json"
    save(result, path)

    for scope in ("generation", "optimization"):
        table = Table(title=f"MolDesignBench, {scope}", title_style="bold")
        for col in ("arm", "tools", "n", "Success", "CSR", "MCD", "InfAcc", "MCC", "Validity", "calls"):
            table.add_column(col, justify="right" if col not in ("arm", "tools") else "left")

        rows = [r for r in result.rows if r["scope"] == scope]
        rows += [r for r in baselines.published(scope)]
        rows.sort(key=lambda r: -r["success"])
        for r in rows:
            mine = r.get("measured_here")
            name = f"[bold]{r['arm']}[/bold]" if mine else r["arm"]
            calls = r.get("tool_calls")
            table.add_row(
                name,
                "yes" if r.get("tools") else "no",
                str(r.get("n", "")),
                f"{r['success']:.3f}",
                f"{r['csr']:.3f}",
                f"{r['mcd']:.2f}",
                f"{r['inf_acc']:.3f}",
                f"{r['mcc']:.3f}",
                f"{r['validity']:.3f}",
                "-" if calls is None else f"{calls:.2f}",
            )
        console.print(table)

    console.print(
        f"\n[dim]Bold rows measured here on {result.sample.get('n')} instances "
        f"(fingerprint {result.sample.get('fingerprint')}); the rest are "
        f"{baselines.SOURCE['venue_short']}, {baselines.SOURCE['table']}, "
        f"{baselines.SOURCE['n_per_task']} instances per task.[/dim]"
    )
    for arm_name, info in result.arms.items():
        if info.get("n_fallbacks"):
            console.print(
                f"[yellow]{arm_name}: {info['n_fallbacks']} fallback decision(s); "
                "those instances are not results for this arm[/yellow]"
            )
    console.print(f"\nwrote {path}")


if __name__ == "__main__":  # pragma: no cover
    app()
