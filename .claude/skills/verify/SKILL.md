---
name: verify
description: Build, run and drive DrugOrch to observe a change at its real surface.
---

# Verifying DrugOrch

Two surfaces: the FastAPI server (`/api/*`) and the no-build web UI it serves.

## Handle

```bash
uv run ruff check src/ tests/          # fast syntax gate; not evidence
uv run uvicorn drugorch.api.app:app --port 8099 --log-level warning &
```

Windows: `pkill -f uvicorn` does NOT work from Git Bash. Stale servers silently hold
port 8099 and you will test an old build while reading a new one. Always:

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -like '*uvicorn*' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

Then check `/tmp/srv.log` for `error while attempting to bind` before trusting any result.

## Driving the UI

Browser clicks at screenshot coordinates do not land in this environment (the screenshot
frame and the viewport differ). Drive the page with `javascript_tool` instead:

```js
const d = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value');
d.set.call(input, 'O60674');
input.dispatchEvent(new Event('input', { bubbles: true }));
document.querySelector('.setup form')
  .dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
```

`.click()` on the submit button does NOT reliably fire the form handler here; dispatching
`submit` on the form does. Never bust the module cache with a reload alone — navigate to
`http://127.0.0.1:8099/?v=N#/...` with a fresh `N` after editing anything under `web/`.

**The tab is usually hidden.** `document.hidden === true` and `requestAnimationFrame` never
fires. Anything rAF-driven will appear broken when it is not — check `document.hidden`
before concluding the UI is frozen. Navigating the tab makes it visible again.

## Flows worth driving

- A run end to end: setup → `#/run/<id>` → trace blocks, inspector, "what the policy saw".
- `scripted` on `O60674` with budget 120 finishes in ~25 s and 18 steps, no network stubs.
- The silent-failure probe: set a bogus key with `{"verify": false}`, start an `llm` run,
  and confirm the LLM panel reports N-of-N fallbacks. This is the highest-value probe in
  the repo — a fallback run looks exactly like a successful one everywhere else.
- Restart the server and re-open a finished run: anything read from `summary.json` vs
  `run.json` diverges here, and only here.

## Target picker

`/api/targets/search` is the fast path (one UniProt call, runs per keystroke);
`/api/targets/readiness` is the slow fan-out (RCSB + ChEMBL in parallel, several seconds
cold, sub-second warm from the disk cache). Drive both — a change that only looks right on
a cached target is not verified.

Queries worth driving: `JAK2` (text), `O60674` (accession), `7LL4` (PDB), `CHEMBL2971`
(ChEMBL), `zzzznotathing` (no match). Gene symbols like `ABL1` must classify as text, not
as a PDB ID — only the leading digit separates them.

The dropdown closes on blur, and running `javascript_tool` drops focus. Re-focus the input
(`inp.focus()` plus a dispatched `focus` event) in the same call that reads `.ac-item`, or
the list will look empty when it is not.

Cross-check the readiness card against the run: the `pdb_id` it names must equal
`facts.receptor.pdb_id` in the finished run, or the preview is lying.

## LLM providers

Three routes behind one backend interface: `anthropic` (API key), `openai` (API key),
`chatgpt` (OAuth against the user's ChatGPT plan). `GET /api/manifest` -> `llm` says which
is active and whether it is ready; `PUT /api/provider` switches.

**You cannot complete a ChatGPT sign-in** — it needs the user's credentials and their
consent to authorize the app. Verify up to the account-chooser page and stop:

```bash
uv run python -c "from drugorch.llm.chatgpt_auth import authorize_url, Pkce, SessionStore; ..."
```

then open that URL in the browser. `Use ChatGPT to sign in to DrugOrch` plus an account
chooser means the request shape is accepted. A JSON `invalid_authorize_request` body names
the bad parameter in `param`. `curl` gets a Cloudflare 403 here — only a real browser works.

`ext_agent_host_id` must be a genuine UUID URN; `urn:uuid:anything-else` is rejected.

The request body for the plan route is pinned by `tests/test_openai_backend.py`: `store:
false`, `stream: true`, no `temperature`/`top_p`/`max_output_tokens`, and system text in
`instructions` rather than a `{role: "system"}` item. Sending a forbidden parameter 400s,
which the policy turns into a silent scripted fallback — check the LLM panel's fallback
count after any change here.

## Model and effort

`GET /api/models?provider=X` returns the catalogue; `source` is `live` when a credential let
us list the account and `fallback` otherwise. The two routes return **different shapes**: the
public API gives `{"data":[{"id":...}]}`, the ChatGPT plan gives
`{"models":[{"slug":...,"supported_reasoning_levels":[...]}]}` — parse both.

Efforts are per model, not per provider. Probes worth running against a signed-in plan:

```bash
curl -X PUT .../api/provider -d '{"model":"gpt-6.1-sol","effort":"ultra"}'   # accepted
curl -X PUT .../api/provider -d '{"model":"gpt-6-luna"}'                      # effort auto-corrects
curl -X PUT .../api/provider -d '{"effort":"ultra"}'                          # 400, luna lacks it
```

Changing model must revalidate the stored effort — otherwise the UI's effort `<select>`
renders with an empty value and the next run 400s on its first decision.

## Benchmark

`drugorch bench` runs a grid of (target x policy x seed). Cells are **resumable** — a run dir
with `summary.json` is rescored, not re-run — so the second invocation is fast and the first
is not. A 12-cell grid at budget 80 takes roughly 10-15 minutes cold; docking dominates.

Run it in the background and wait on `data/bench/grid.json` appearing, not on a fixed sleep.

The test that matters more than the leaderboard:

```bash
# No withheld molecule may appear in a run with origin=retrieval. A leak invalidates
# every rediscovery number in the project.
uv run pytest tests/test_harness.py::test_a_held_out_molecule_cannot_reach_the_run
```

Cross-check a finished grid by hand: every cell for one (target, seed) must carry the same
`split.json` fingerprint. Arms facing different holdouts are not comparable and nothing in
the summary statistics would reveal it.

`#/bench` renders the leaderboard from `data/bench/grid.json`. With no grid it shows the
command rather than an empty table.

## Checking artifacts

```bash
ls data/runs/<id>/          # run.json decisions.json decisions.csv events.jsonl
                            # llm_calls.jsonl molecules.csv summary.json
```

`decisions.json` must carry `state_seen` on every record; `llm_calls.jsonl` is empty (not
absent) for rule-based arms.

## The public benchmark

`drugorch extbench` answers MolDesignBench instances and scores them with the benchmark's
own evaluator. Three environments are involved and each fails differently:

```bash
# measurement: the benchmark's analyzer in its own venv (~20 s cold, ~0.5 s per batch after)
uv run python -c "from drugorch.harness.external.panel import get_panel; \
  print(get_panel().measure(['CC(=O)Oc1ccccc1C(=O)O']))"

# scoring: their evaluator, run unmodified in .venv-mdb
uv run drugorch extbench --n 4 --score-only
```

`.venv-mdb` needs `openai`, `anthropic`, `requests` and `pubchempy` on top of the benchmark's
own requirements: `moldesign/__init__.py` imports its LLM clients at package import, and
`get_global_tools()` loads the retrieval tools, so the evaluator fails on an import long
before it reads a prediction. The failure surfaces as
`the benchmark evaluator failed: ModuleNotFoundError`.

The evaluator is run with `PYTHONPATH` set to the repository and `cwd` inside it. Both are
load-bearing: a script run by path puts `evaluate/` on `sys.path` rather than the repository,
and the tools resolve their data paths relative to the working directory.

The test that matters more than the board:

```bash
# No prompt, on either arm, may contain the answer key. A leak still produces a score.
uv run pytest tests/test_external.py::test_no_prompt_can_carry_the_answer_key
```

Checks worth making on a finished run:

- `data/bench/external.json` → every row carries `n` and `measured_here`; published rows
  must never carry `measured_here: true`.
- `arms.<name>.n_fallbacks` is 0. A fallback decision never reached the model, so that
  instance is not a result for the arm, and `load_attempt` refuses to reuse it on resume.
- `mean_tool_calls` on the `drugorch` arm is well above zero. An orchestrated arm that never
  called a subsystem is the no-tool arm wearing its label.
- The sample fingerprint in the board matches the one `--n`/`--seed` reproduce: two success
  rates over different subsets are not comparable and nothing else in the file says so.

Strict JSON schemas: the ChatGPT-plan route rejects any schema whose `required` omits a key,
and the run loop turns that 400 into a fallback. Every object in `spec.py` and `agent.py`
lists all of its keys and uses nullable types for optional values. A schema change that
forgets this produces an all-`None` board with no visible error.
