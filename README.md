# DrugOrch

LLM-orchestrated multi-subsystem drug discovery with a visible routing trace.

State a goal in natural language, pick a protein target, and an orchestrator decides which
discovery subsystem to invoke, when, and under what budget. The web interface shows the routing
decisions live — which subsystem was called, in what order, and why.

## Subsystems

| Key | What it does |
|---|---|
| `retrieval` | RCSB PDB, UniProt, ChEMBL, PubChem lookups |
| `properties` | RDKit property/ADMET panel (QED, SA, Lipinski, Veber, PAINS) |
| `docking` | AutoDock Vina binding-affinity estimation |
| `generation` | GraphGA molecule generation and lead optimization |

## Setup

```powershell
uv sync
uv run drugorch doctor
```

Only the `llm` routing policy needs an Anthropic API key; every other policy runs without one.
See **The API key** below for the two ways to supply it.

## Running the web interface

```powershell
uv run uvicorn drugorch.api.app:app --port 8099
```

Then open <http://127.0.0.1:8099/>. Add `--reload` while editing Python; the `web/` files are
served from disk, so a browser refresh is enough to pick up frontend changes.

The server reads `.env` once at startup, so a key *edited in the file* needs a restart — a key
entered in the web interface does not.

## Which model routes, and how it is paid for

The `llm` policy can run on any of three credentials. Every other policy runs on none.

**ChatGPT plan — no API key.** OpenAI's token-sharing preview lets a locally hosted,
open-source app run inference against your own ChatGPT Plus or Pro plan. Pick *ChatGPT plan*
in the setup screen and click **Sign in with ChatGPT**: a browser tab opens OpenAI's own
sign-in page, you approve DrugOrch, and decisions bill to the plan you already pay for.
DrugOrch never sees your password, OpenAI shares only your name and email, and you can cap
or revoke the app from ChatGPT settings. The token is stored at
`data/cache/chatgpt_auth.json` on this machine only.

> **Research caveat.** This route cannot pin sampling (`temperature` is rejected) and cannot
> cache the prompt prefix, so a ChatGPT-plan arm is less reproducible and costs more per
> decision than the Anthropic one. Recorded runs still replay exactly from the response
> cache. The provider is written into every run's provenance — **do not pool results across
> providers**; switching provider changes the arm under test.

**Anthropic API key.** Claude, with prompt caching. See below.

**OpenAI API key.** Set `OPENAI_API_KEY` in `.env` and pick *OpenAI API key*.

### Model and reasoning effort

Both are chosen in the setup screen, under the provider picker, and both are recorded in run
provenance and in the response-cache key — changing either starts a fresh set of decisions
rather than replaying someone else's.

The model list is fetched **from your account** whenever a credential exists, because a
hardcoded catalogue is wrong the moment a model ships. A signed-out install falls back to a
short list and says so.

Reasoning effort follows the selected model, not the provider. The vocabularies genuinely
differ — on a ChatGPT plan some models accept `ultra` and others stop at `max`, while the
public GPT-5.1 docs list a `none` level these reject. Picking a level a model does not accept
is a 400, which the run loop turns into a scripted fallback, so DrugOrch validates it up
front and auto-corrects the effort when you change model (and tells you it did).

## The Anthropic API key

Two ways, and the first is the easy one for local use:

**In the browser.** Start the server, and the setup screen has an *Anthropic API key* field
right under the policy picker. Paste a key and it is checked against the API before being
accepted — a typo is caught there rather than halfway through a run. The key is held in the
server process, so it lasts until the server stops. Tick *also write it to `.env`* to keep it
across restarts.

**In `.env`.** Copy `.env.example` to `.env`, set `ANTHROPIC_API_KEY=…`, restart the server.

A key entered in the browser takes precedence over one in `.env`. Either way the key stays
server-side: it is never echoed back to the page (only a mask like `sk-ant-api0…7788`), and
run records store `api_key_source` — `"session"` or `"env"` — never the key itself.

The key endpoints are served only to a client on this machine, so running with
`--host 0.0.0.0` to reach the UI from another device leaves them closed; use `.env` in that
case.

## Command line

```powershell
uv run drugorch doctor                      # environment + dependency check
uv run drugorch target JAK2                 # resolve a target across the data banks
uv run drugorch call properties --smiles "CCO"
uv run drugorch run --goal "selective JAK2 inhibitors" --target O60674 `
                    --policy scripted --budget 120 --seed 42
uv run drugorch extbench --n 200             # the public benchmark, scored by its own evaluator
```

## Finding a target

The target box searches UniProt, RCSB PDB and ChEMBL from inside the app — no account, no
API key, no second tab. It accepts whatever you already have:

| You type | It resolves to |
|---|---|
| `JAK2`, `cyclin-dependent kinase 2` | ranked UniProt matches, reviewed and human first |
| `O60674` | that accession directly |
| `7LL4` | the protein in that structure |
| `CHEMBL2971` | the protein behind that ChEMBL target |

Focus the empty box for a list of well-characterised starter targets. Arrow keys move,
Enter picks, Escape closes.

Picking a target runs a **readiness check** — a parallel fan-out to RCSB and ChEMBL that
answers the two questions deciding whether a run can do anything:

```
Ready. Docking into 7LL4, 1.31 A, ligand Y5D; 17,653 measured actives
to seed the generator. Every subsystem can run.

  identity     JAK2 - Homo sapiens
  docking      7LL4 - 1.31 A
  generation   17,653 actives in ChEMBL
```

No dockable structure means the run has no objective to optimise; too few actives means the
generator has nothing to breed from. Both are worth knowing before you spend a budget rather
than after, so the check runs before the run does — and the structure it names is the one
the run will actually dock into.

## The public benchmark

The leaderboard opens on **MolDesignBench** (COLM 2026): 2,000 scenario-grounded molecular
design instances, each stating some constraints outright and implying others through the
scenario, with about one in ten contradictory by construction. It is somebody else's
benchmark, and that is the point — a benchmark whose author also wrote the system under test
proves very little.

```powershell
uv run drugorch extbench --n 200
```

Then open <http://127.0.0.1:8099/#/bench>.

**Two arms run, and the gap between them is the result.** `drugorch` reads the question,
writes down the constraint set it believes was asked for, and then routes among the
subsystems — pool retrieval, GraphGA generation, measurement — until a molecule measures up.
`notool` is the same model answering in one shot with no tools. The paper reports its models
the same way, with and without tools, so the comparison isolates the orchestration from the
model underneath it.

**Scoring is theirs, not ours.** Answers are written into the layout
`evaluate/evaluate_benchmark.py` reads and that script is run unmodified, inside the
environment the benchmark's answer key was built in. Nothing in this repository computes
Success, CSR, MCD, InfAcc, MCC or Validity. Molecules are measured by the benchmark's own
analyzer for the same reason: `CalcNumHBA` returns 4 under RDKit 2025.03.6 and 3 under
2026.09.1, and a constraint whose window opens at 4 would otherwise turn on which
interpreter measured it.

**Published rows are labelled as read rather than measured.** The GPT-5.4, Claude-Opus-4.6,
Qwen3, GPT-OSS, InternLM and Ether0 rows come from Table 3 of the paper. They were not re-run
here, every row carries the number of instances behind it, and rows measured here are marked.

**No arm can see the answer key.** An arm receives the question, the task type and (for
optimization) the seed molecule. It never receives `answer` or `ref_smiles`, and
`tests/test_external.py::test_no_prompt_can_carry_the_answer_key` asserts that no prompt on
either arm contains them. Half this benchmark's difficulty is inferring constraints that are
only implied, so a leak would look like unusually good reading rather than like cheating.

Attempts are cached per instance, so an interrupted run resumes and `--score-only` re-scores
what is on disk without spending model calls.

**The benchmark, cited.** Jeong, Ko, Kim, Lee, Hormazabal, Lee, Han, Lim and Kim.
*MolDesignBench: Evaluating LLM-based Agent for Scenario-grounded Molecular Design.*
Conference on Language Modeling (COLM), 2026. Published figures quoted here are its Table 3,
read from `third_party/MolDesignBench/docs/leaderboard.md` and checked against that file by
`tests/test_external.py`. Benchmark code is MIT, (c) 2026 Korea University and LG AI
Research; the benchmark data is CC BY-NC 4.0.

- Code: <https://github.com/KU-AGI/MolDesignBench>
- Data: <https://huggingface.co/datasets/LG-AI-Research/MolDesignBench>

The same citation is on the leaderboard page, with a copyable BibTeX entry.

## The routing benchmark

A single run tells you what happened once. The benchmark turns that into evidence.

```powershell
uv run drugorch bench --targets O60674,P24941 --policies fixed,scripted,llm --seeds 1,2
```

Then open <http://127.0.0.1:8099/#/bench> and switch to **Routing arms**.

**What is held back.** For each (target, seed) a fraction of the known actives is withheld —
by **Bemis-Murcko scaffold**, not at random. A random split leaves close analogues of every
withheld compound in the visible half, so "rediscovery" degenerates into proposing something
one methyl away from a molecule the run was handed. Holding out whole chemotypes means a hit
requires reaching something genuinely unseen. Retrieval enforces this at the one point
actives enter a run, so a withheld molecule can arrive exactly one way: the run proposed it.

**What it ranks on.** `discovery_auc` — the area under *held-out scaffolds found vs budget
spent*. An area rather than an endpoint, because a run that finds everything on its last unit
and one that finds it immediately are not equally good and an endpoint cannot tell them apart.

**What keeps it honest**, reported beside every row:

| | |
|---|---|
| `rediscovery` | fraction of the held-out set reached at all |
| `best_score`, `score_auc` | the docking endpoint — reported, never the ranking metric |
| `novelty`, `diversity`, `n_scaffolds` | an arm that wins by re-proposing its seeds has not discovered anything |
| `sa_mean`, `pains_rate` | a winner made of unsynthesisable molecules is not a winner |
| `n_fallbacks` | a run whose decisions fell back is not a run of the arm it claims to be |

**How arms are compared.** Paired by (target, seed) against the `fixed` baseline, not averaged
independently. Targets differ from each other far more than policies differ on any one of
them, so an unpaired mean mostly measures which targets happened to be easy. Win rate and
mean delta are shown together because either alone misleads — an arm can win narrowly
everywhere, or lose everywhere except one blowout, and those are different findings. The
confidence interval is a bootstrap; a delta whose interval straddles zero is left uncoloured
rather than presented as a win.

Grids are resumable: a finished cell is rescored, not re-run.

## Routing policies

| Name | Role |
|---|---|
| `fixed` | Runs the pipeline in a hard-coded order — the no-routing baseline |
| `random` / `roundrobin` | Controls: routing with no information |
| `scripted` | Hand-written rules over the same state view the LLM sees |
| `oracle` | Reads the ChEMBL answer key — a ceiling, not a method |
| `llm` | `claude-opus-5-5` planner; requires `ANTHROPIC_API_KEY` |

Every policy observes an identical `state.summary()` and emits an identical `Decision`, so the
arms are comparable. All of them run without a key except `llm`.
