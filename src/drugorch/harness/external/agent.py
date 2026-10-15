"""The arms that answer a MolDesignBench instance.

Two of them, and the pair is the experiment.

``notool`` is one call to the model: read the question, return a SMILES or ``None``. It is
the control, and it is the same control the paper reports in its w/o tool column.

``drugorch`` is the product: the model reads the question and writes down the constraint set
it believes was asked for, then routes among DrugOrch's subsystems until it has a molecule
that measurement says satisfies that set. It may search the molecule pool, run the generator,
propose molecules itself, or revise its reading of the question, and it decides which, in
what order, from what the measurements came back as. The routing is the contribution; the
subsystems underneath are the ones a target-driven run uses.

Three rules keep the comparison honest.

**The answer key is not in this process.** An arm receives :func:`instances.prompt_view` and
nothing else, so no prompt can carry a constraint the arm was supposed to infer.

**Measurement is the benchmark's, not ours.** Candidates are measured by MolDesignBench's own
analyzer, which is also one of the seventeen tools it gives the models it evaluates.

**A tool call is counted the way the paper counts one.** Every subsystem invocation is a tool
call, the budget is the paper's twenty, and the count is reported on the board beside the
score rather than left out of it.

A failure to reach the model is recorded as a fallback and the instance answers ``None``.
That costs the arm the instance, which is correct: a run whose decisions silently degraded to
a scripted path is not a run of the arm it claims to be, and the board flags any row carrying
fallbacks.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from ...config import Settings, get_settings
from ...llm import build_backend
from ...llm.response_cache import LLMUnavailable, ResponseCache, request_key
from .ga import ANCHOR_FLOOR, optimize
from .instances import Instance, prompt_view
from .panel import PanelUnavailable, PanelWorker, get_panel
from .pool import MoleculePool, PoolUnavailable, get_pool
from .spec import SPEC_SCHEMA, DesignSpec, merge

EXT_V = "v3"
"""The arm recipe: prompts, schemas, tool set and budgets. In every response-cache key, so a
change here starts fresh decisions rather than replaying answers to a different question."""

MAX_TOOL_CALLS = 20
"""The paper's allowance in its w/ tool setting. Matched rather than chosen."""

MAX_DECISIONS = 6
"""How many times the model may route before it must answer. Each decision is a request, so
this is the arm's cost per instance as much as its depth."""

MAX_VERIFICATIONS = 2
"""How many times an arm may be made to re-read the question before answering.

This exists because of what the first run of this benchmark measured. On 174 feasible
instances the arm believed it had satisfied its own specification 103 times and was right
21 of those: **82 instances were lost to finishing against an incomplete reading of the
question**, not to failing at the chemistry. It wrote down 8.01 constraints where the key
held 8.71, and under-extracted on 92 of 174.

It is also not a budget problem. The arm used fewer subsystem calls when it succeeded (3.48)
than when it failed (4.42), against a cap of 20 it never approached. It stops early because
it believes it is finished, so the cheapest available improvement is not another tool, it is
making the belief answer for itself once before it becomes an answer.

Two passes rather than one because a verification can add a constraint that the next round
of optimisation then misses; more than two spends requests on diminishing returns, since a
third pass can only find what two readings of the same text both missed."""

ANSWER_NONE = "None"


# --- prompts ----------------------------------------------------------------------
#
# Split into a stable part and a volatile part because the stable part is marked cacheable:
# it is identical across all 200 instances, and paying for it once per instance rather than
# once per run would be most of the bill.

SYSTEM = (
    "You are a medicinal chemist designing a molecule to a specification. "
    "Answer only with the JSON object the schema describes."
)

TOOL_MANUAL = """\
# Task

Each instance states a design goal in prose. Some constraints are written out as numbers.
Others are implied by the scenario: a compound that must cross the blood-brain barrier is
being constrained on molecular weight, lipophilicity and polar surface area whether or not
those words appear. The scoring key contains both kinds, so a reading that takes only the
explicit numbers will miss constraints that were present.

An instance is infeasible only when its constraints contradict each other, for example a
minimum above a maximum on one property. Roughly one instance in ten is infeasible. A hard
instance is not an infeasible one, and answering None to a solvable instance scores zero.

# Properties that can be measured

MW, logP, logD, logS, HBD, HBA, TPSA, rotB, rings_total, rings_aromatic, QED, BBBP, Mutag,
MR, heavy_atoms, formal_charge, Fsp3, hetero, topo_diameter, HIA.

logD, logS, BBBP, Mutag and HIA are model predictions. They cannot be reasoned to from
structure with any accuracy, so a molecule constrained on them has to be measured.

# Functional groups that can be required

benzene ring, pyridine ring, furan ring, thiophene ring, thiazole ring, oxazole ring,
imidazole ring, piperidine ring, piperzine ring, morpholine ring, benzodiazepine,
beta lactam, amide, ester, ketone, aldehyde, carboxylic acid, carbonyl o, phenol,
aliphatic hydroxyl, methoxy, ether oxygens (including phenoxy), primary amine,
secondary amine, tertiary amine, aniline, aromatic nitrogen, amidine, guanidine, urea,
hydrazine, hydrazone, oxime, hydroxylamine, nitrile, nitro, nitroso, azo, diazo, halogen,
sulfonamide, sulfone, thioether, thiol, thiocarbonyl, thiocyanate, isothiocyanate,
isocyanate, alkyl carbamate, barbiturate, aryl methyl sites for hydroxylation.

# Subsystems

**search_pool** queries 1.9 million MOSES molecules tagged with the same property
predictions used for scoring, and returns ones inside the windows of the current
specification. A molecule that already satisfies five of seven constraints is a better
starting point than a blank page. Windows the pool cannot satisfy are relaxed one at a time
and the response says which.

**generate** runs a graph genetic algorithm from the molecules found so far, scoring every
child by how much of the specification it satisfies. It repairs a constraint or two; it does
not invent a chemotype.

**propose** puts molecules written in this reply into the pool of candidates and measures
them. Use it to seed a scaffold the pool does not hold.

**answer** ends the instance with one SMILES, or with None for a contradictory
specification.

A specification can be revised at any step with `spec_update`. Measurements often show that
a constraint was read too tightly or missed entirely; revising is cheaper than optimising
toward the wrong target.

Budget: at most 20 subsystem calls and 6 routing decisions per instance.
"""

DECISION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["search_pool", "generate", "propose", "answer"],
        },
        "reason": {"type": "string", "description": "One sentence."},
        "propose_smiles": {
            "type": "array",
            "items": {"type": "string"},
            "description": "SMILES to measure, for the propose action.",
        },
        "generate_rounds": {
            "type": ["integer", "null"],
            "description": "Genetic algorithm rounds, 1 to 8, for the generate action.",
        },
        "answer_smiles": {
            "type": ["string", "null"],
            "description": "The final molecule, or null to answer None.",
        },
        "spec_update": {
            "type": ["object", "null"],
            "properties": {
                "properties": SPEC_SCHEMA["properties"]["properties"],
                "fragments": SPEC_SCHEMA["properties"]["fragments"],
            },
            "required": ["properties", "fragments"],
            "additionalProperties": False,
        },
    },
    # Every key is required, with unused ones answered as null. The strict schema route
    # rejects a partial `required` outright, and a rejected schema is a fallback rather than
    # a decision.
    "required": [
        "action",
        "reason",
        "propose_smiles",
        "generate_rounds",
        "answer_smiles",
        "spec_update",
    ],
    "additionalProperties": False,
}

VERIFY_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "complete": {
            "type": "boolean",
            "description": "True only if the specification already covers every stated and "
            "implied requirement of the scenario.",
        },
        "reason": {"type": "string"},
        "properties": SPEC_SCHEMA["properties"]["properties"],
        "fragments": SPEC_SCHEMA["properties"]["fragments"],
    },
    "required": ["complete", "reason", "properties", "fragments"],
    "additionalProperties": False,
}

VERIFY_PROMPT = """\
A molecule is about to be submitted as the answer to this instance. Before it is, read the
scenario once more against the specification that was extracted from it.

The scenario states some requirements as numbers and implies others through its narrative. A
compound described as crossing the blood-brain barrier is being constrained on size,
lipophilicity and polar surface area whether or not those words appear. A compound for
systemic distribution, for environmental delivery, for a shallow binding surface, for
conjugation: each implies a region of property space.

List every requirement the scenario carries that the specification below does not already
contain. Repeat nothing it already has. If the specification is complete, say so and list
nothing: adding a constraint the scenario does not carry costs the instance exactly as much
as missing one.
"""

NOTOOL_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "smiles": {
            "type": ["string", "null"],
            "description": "The designed molecule, or null if the constraints contradict.",
        },
        "reason": {"type": "string"},
    },
    "required": ["smiles", "reason"],
    "additionalProperties": False,
}

NOTOOL_MANUAL = """\
# Task

Each instance states a design goal in prose. Some constraints are written out as numbers;
others are implied by the scenario and are scored all the same.

An instance is infeasible only when its constraints contradict each other. About one in ten
is. Answer null only in that case; a hard instance is not an infeasible one.

Return one SMILES string that satisfies every stated and implied constraint, or null.
"""


# --- the result of one instance ----------------------------------------------------


@dataclass
class ToolCall:
    """One subsystem invocation, as it will be counted and shown."""

    tool: str
    detail: str = ""
    n_molecules: int = 0
    best_satisfied: int = 0
    wall_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "tool": self.tool,
            "detail": self.detail,
            "n_molecules": self.n_molecules,
            "best_satisfied": self.best_satisfied,
            "wall_s": round(self.wall_s, 2),
        }


@dataclass
class Attempt:
    """One arm's answer to one instance, with everything needed to audit it."""

    instance_id: str
    arm: str
    prediction: str = ANSWER_NONE
    spec: dict = field(default_factory=dict)
    tool_calls: list[ToolCall] = field(default_factory=list)
    decisions: list[dict] = field(default_factory=list)
    n_fallbacks: int = 0
    n_verify_failures: int = 0
    """Verification requests that never reached the model. Not a fallback: the arm ran."""

    error: str = ""
    wall_s: float = 0.0
    best_satisfied: int = 0
    best_total: int = 0

    @property
    def n_tool_calls(self) -> int:
        return len(self.tool_calls)

    def to_dict(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "arm": self.arm,
            "prediction": self.prediction,
            "spec": self.spec,
            "tool_calls": [t.to_dict() for t in self.tool_calls],
            "decisions": self.decisions,
            "n_tool_calls": self.n_tool_calls,
            "n_fallbacks": self.n_fallbacks,
            "n_verify_failures": self.n_verify_failures,
            "error": self.error,
            "wall_s": round(self.wall_s, 2),
            "best_satisfied": self.best_satisfied,
            "best_total": self.best_total,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Attempt:
        a = cls(
            instance_id=d.get("instance_id", ""),
            arm=d.get("arm", ""),
            prediction=d.get("prediction", ANSWER_NONE),
            spec=d.get("spec") or {},
            decisions=d.get("decisions") or [],
            n_fallbacks=int(d.get("n_fallbacks", 0)),
            n_verify_failures=int(d.get("n_verify_failures", 0)),
            error=d.get("error", ""),
            wall_s=float(d.get("wall_s", 0.0)),
            best_satisfied=int(d.get("best_satisfied", 0)),
            best_total=int(d.get("best_total", 0)),
        )
        a.tool_calls = [ToolCall(**{k: v for k, v in t.items()}) for t in (d.get("tool_calls") or [])]
        return a


# --- the arms ----------------------------------------------------------------------


@dataclass
class Arm:
    """Shared machinery: one backend, one response cache, one of each subsystem."""

    name: str
    settings: Settings = field(default_factory=get_settings)
    backend: object | None = None
    cache: ResponseCache | None = None
    panel: PanelWorker | None = None
    pool: MoleculePool | None = None
    offline: bool = False

    def __post_init__(self) -> None:
        self.settings.ensure_dirs()
        if self.backend is None:
            self.backend = build_backend(self.settings)
        if self.cache is None:
            self.cache = ResponseCache(self.settings.llm_cache_dir)
        if self.panel is None:
            self.panel = get_panel()
        if self.pool is None:
            self.pool = get_pool()

    def _ask(self, prompt: str, schema: dict, manual: str) -> dict:
        """One request, replayed from the cache when the same question was asked before.

        The cache is keyed on backend, prompts and schema, so a resumed run costs nothing
        and reproduces its earlier decisions exactly. This is the same cache the routing
        policy uses, for the same reason.
        """
        assert self.cache is not None and self.backend is not None
        key = request_key(
            self.backend.key(), SYSTEM, manual + "\n\n" + prompt, schema, EXT_V
        )
        hit = self.cache.get(key)
        if hit is not None and isinstance(hit.get("payload"), dict):
            return hit["payload"]
        if self.offline:
            raise LLMUnavailable("offline, and this decision is not in the response cache")

        payload, meta = self.backend.complete(SYSTEM, manual, prompt, schema)
        self.cache.put(key, {"payload": payload, "meta": meta, "version": EXT_V})
        return payload

    # -- the control -----------------------------------------------------------
    def answer_notool(self, inst: Instance) -> Attempt:
        """One call, no tools. The paper's w/o tool setting."""
        started = time.time()
        att = Attempt(instance_id=inst.uid, arm=self.name)
        view = prompt_view(inst)
        prompt = json.dumps(view, indent=1)
        try:
            payload = self._ask(prompt, NOTOOL_SCHEMA, NOTOOL_MANUAL)
        except LLMUnavailable as exc:
            att.n_fallbacks += 1
            att.error = str(exc)
            att.wall_s = time.time() - started
            return att

        smiles = payload.get("smiles")
        att.prediction = str(smiles).strip() if isinstance(smiles, str) and smiles.strip() else ANSWER_NONE
        att.decisions.append({"action": "answer", "reason": str(payload.get("reason", ""))[:300]})
        att.wall_s = time.time() - started
        return att

    # -- the product -----------------------------------------------------------
    def answer_orchestrated(self, inst: Instance, *, seed: int = 0) -> Attempt:
        """Plan, then route among the subsystems until a molecule measures up."""
        started = time.time()
        att = Attempt(instance_id=inst.uid, arm=self.name)
        view = prompt_view(inst)

        # --- the plan: what does this question actually ask for? ---
        try:
            payload = self._ask(
                json.dumps(view, indent=1)
                + "\n\nWrite down the constraint set this question specifies, including the "
                "constraints implied by the scenario, and opening candidate molecules.",
                SPEC_SCHEMA,
                TOOL_MANUAL,
            )
        except LLMUnavailable as exc:
            att.n_fallbacks += 1
            att.error = str(exc)
            att.wall_s = time.time() - started
            return att

        spec = DesignSpec.from_llm(payload, at_least=inst.task_type == "optimization")
        att.spec = spec.to_dict()
        att.decisions.append(
            {"action": "plan", "reason": spec.reason[:300], "constraints": spec.total}
        )

        # A structurally contradictory specification is the benchmark's definition of an
        # infeasible instance, so it is answered immediately rather than optimised against.
        if spec.infeasible and spec.contradictions():
            att.prediction = ANSWER_NONE
            att.best_total = spec.total
            att.wall_s = time.time() - started
            return att

        candidates: list[str] = [
            s for s in (payload.get("candidates") or []) if isinstance(s, str) and s.strip()
        ][:8]
        if inst.seed_smiles:
            candidates.insert(0, inst.seed_smiles)

        best_smiles: str | None = None
        best = None
        seen: dict[str, object] = {}
        verifications = 0
        # On an optimization instance the question supplies a lead and asks for it to be
        # improved. The search is anchored to it, because the first run of this benchmark
        # lost that task by answering a different question well.
        anchor = inst.seed_smiles if inst.task_type == "optimization" else None

        def remember(cands: list[str], origin: str) -> ToolCall:
            """Measure a batch, score it against the current spec, keep the best."""
            nonlocal best_smiles, best
            t0 = time.time()
            call = ToolCall(tool=origin, n_molecules=len(cands))
            try:
                measured = self.panel.measure(cands)
            except PanelUnavailable as exc:
                call.detail = f"measurement unavailable: {exc}"
                call.wall_s = time.time() - t0
                return call
            for smi, m in measured.items():
                if not m.valid:
                    continue
                sc = spec.score(m)
                seen[smi] = sc
                if best is None or sc.fitness > best.fitness:
                    best, best_smiles = sc, smi
            call.best_satisfied = best.satisfied if best else 0
            call.wall_s = time.time() - t0
            return call

        def verify() -> bool:
            """Re-read the question against the specification. True when the spec grew.

            This is the one request the arm makes about its own reading rather than about
            the chemistry. Anything it returns is merged rather than replacing, because a
            second reading that forgets a constraint the first caught would make the arm
            oscillate between two incomplete specifications.
            """
            nonlocal spec, best, best_smiles, verifications
            verifications += 1
            state = {
                "instance": view,
                "specification_so_far": spec.describe(),
                "candidate": best_smiles,
                "candidate_satisfies": (
                    f"{best.satisfied}/{best.total}" if best else "nothing measured yet"
                ),
                "candidate_still_missing": best.misses[:6] if best else [],
            }
            try:
                payload = self._ask(
                    VERIFY_PROMPT + "\n\n" + json.dumps(state, indent=1),
                    VERIFY_SCHEMA,
                    TOOL_MANUAL,
                )
            except LLMUnavailable as exc:
                # Counted apart from the plan and routing fallbacks. Those mean the arm
                # never ran, so the instance is excluded from scoring. This one means an
                # optional extra reading did not happen: the arm has its molecule and
                # answers with it. Folding the two together dropped solved instances off
                # the board, and preferentially so, since verification only runs once the
                # arm already believes it has succeeded.
                att.n_verify_failures += 1
                att.error = str(exc)
                return False

            added = DesignSpec.from_llm(
                {**payload, "infeasible": False},
                at_least=inst.task_type == "optimization",
            )
            att.decisions.append(
                {
                    "action": "verify",
                    "reason": str(payload.get("reason", ""))[:300],
                    "added": added.total,
                }
            )
            if payload.get("complete") or not added.total:
                return False

            grown = merge(spec, added)
            if grown.total <= spec.total:
                return False
            spec = grown
            att.spec = spec.to_dict()

            # Everything measured so far was ranked against the old reading, so it is
            # rescored before the loop continues.
            previous = list(seen)
            seen.clear()
            best, best_smiles = None, None
            if previous:
                att.tool_calls.append(remember(previous[:40], "measure"))
            return True

        if candidates:
            att.tool_calls.append(remember(candidates, "measure"))

        # --- the loop: route until solved, out of budget, or answered ---
        for step in range(MAX_DECISIONS):
            if best is not None and best.solved:
                # The arm believes it is finished. That belief was right one time in five
                # on the first run of this benchmark, so it is checked before it becomes an
                # answer rather than after.
                if verifications < MAX_VERIFICATIONS and verify():
                    continue
                break
            if att.n_tool_calls >= MAX_TOOL_CALLS:
                break

            state = {
                "instance": view,
                "specification": spec.describe(),
                "best_molecule": best_smiles,
                "best_satisfies": f"{best.satisfied}/{best.total}" if best else "nothing measured yet",
                "still_missing": best.misses[:6] if best else [],
                "subsystem_calls_used": att.n_tool_calls,
                "subsystem_calls_left": MAX_TOOL_CALLS - att.n_tool_calls,
                "decisions_left": MAX_DECISIONS - step,
                "history": [t.to_dict() for t in att.tool_calls[-4:]],
            }
            try:
                decision = self._ask(json.dumps(state, indent=1), DECISION_SCHEMA, TOOL_MANUAL)
            except LLMUnavailable as exc:
                att.n_fallbacks += 1
                att.error = str(exc)
                break

            action = str(decision.get("action") or "answer")
            att.decisions.append(
                {"action": action, "reason": str(decision.get("reason", ""))[:300]}
            )

            update = decision.get("spec_update")
            if isinstance(update, dict) and (update.get("properties") or update.get("fragments")):
                # Merged, not substituted. The tool manual invites the model to revise a
                # constraint it "missed entirely", and a model acting on that sends the one
                # it now wants added; replacing would drop the rest of its own reading.
                revised = merge(
                    spec,
                    DesignSpec.from_llm(
                        {**update, "infeasible": False},
                        at_least=inst.task_type == "optimization",
                    ),
                )
                if revised.total:
                    spec = revised
                    att.spec = spec.to_dict()
                    # Scores against the old reading are not comparable with scores against
                    # the new one: a molecule satisfying six of the old constraints may
                    # satisfy two of the new. Everything measured so far is rescored, which
                    # costs one call and keeps one ordering over the whole candidate set.
                    previous = list(seen)
                    seen.clear()
                    best, carry = None, best_smiles
                    best_smiles = None
                    if previous:
                        att.tool_calls.append(remember(previous[:40], "measure"))
                    elif carry:
                        att.tool_calls.append(remember([carry], "measure"))

            if action == "answer":
                # The same check as on a self-declared success: the arm's reading of the
                # question answers for itself once before its molecule does.
                if verifications < MAX_VERIFICATIONS and verify():
                    continue
                final = decision.get("answer_smiles")
                if isinstance(final, str) and final.strip():
                    chosen = final.strip()
                    att.tool_calls.append(remember([chosen], "measure"))
                    # The model's last word is taken when measurement does not contradict
                    # it. Otherwise the best measured molecule stands, because the benchmark
                    # scores the molecule rather than the reasoning that produced it.
                    cand = seen.get(chosen)
                    if cand is not None and (best is None or cand.fitness >= best.fitness):
                        best_smiles, best = chosen, cand
                else:
                    best_smiles = None
                break

            if action == "search_pool":
                t0 = time.time()
                call = ToolCall(tool="search_pool")
                try:
                    found = self.pool.search(spec, limit=24, seed=seed)
                    hits = found.smiles
                    call.detail = found.summary()
                    if anchor and hits:
                        # An optimization instance asks for this lead improved, not for a
                        # different molecule that happens to fit the windows. A pool hit
                        # unrelated to the lead is a wrong answer that measures well, and
                        # it is where this arm lost the task on the first run.
                        from ...fingerprints import max_similarity

                        sims = max_similarity(hits, [anchor])
                        kept = [s for s in hits if sims.get(s, 0.0) >= ANCHOR_FLOOR]
                        call.detail += (
                            f". {len(hits) - len(kept)} of {len(hits)} discarded as "
                            f"unrelated to the lead"
                        )
                        hits = kept
                    call.n_molecules = len(hits)
                    att.tool_calls.append(call)
                    if hits:
                        att.tool_calls.append(remember(hits, "measure"))
                except PoolUnavailable as exc:
                    call.detail = f"pool unavailable: {exc}"
                    att.tool_calls.append(call)
                call.wall_s = time.time() - t0

            elif action == "generate":
                rounds = decision.get("generate_rounds")
                rounds = max(1, min(8, int(rounds))) if isinstance(rounds, int) else 4
                seeds = [s for s in [best_smiles] if s] or candidates
                if not seeds:
                    att.tool_calls.append(
                        ToolCall(tool="generate", detail="no seed molecule to evolve from")
                    )
                    continue
                t0 = time.time()
                try:
                    res = optimize(
                        seeds, spec, self.panel, rounds=rounds, seed=seed, anchor=anchor
                    )
                except PanelUnavailable as exc:
                    att.tool_calls.append(ToolCall(tool="generate", detail=str(exc)))
                    continue
                call = ToolCall(
                    tool="generate",
                    detail=res.summary(),
                    n_molecules=res.evaluated,
                    best_satisfied=res.best.score.satisfied if res.best else 0,
                    wall_s=time.time() - t0,
                )
                att.tool_calls.append(call)
                if res.best is not None and (best is None or res.best.fitness > best.fitness):
                    best, best_smiles = res.best.score, res.best.smiles

            elif action == "propose":
                proposed = [
                    s
                    for s in (decision.get("propose_smiles") or [])
                    if isinstance(s, str) and s.strip()
                ][:12]
                if proposed:
                    att.tool_calls.append(remember(proposed, "propose"))
                else:
                    att.tool_calls.append(
                        ToolCall(tool="propose", detail="no molecules in the reply")
                    )

        att.prediction = best_smiles.strip() if best_smiles else ANSWER_NONE
        if best is not None:
            att.best_satisfied, att.best_total = best.satisfied, best.total
        else:
            att.best_total = spec.total
        att.wall_s = time.time() - started
        return att

    def answer(self, inst: Instance, *, seed: int = 0) -> Attempt:
        if self.name == "notool":
            return self.answer_notool(inst)
        return self.answer_orchestrated(inst, seed=seed)

    def describe(self) -> dict:
        assert self.backend is not None
        return {
            "arm": self.name,
            "backend": self.backend.describe(),
            "version": EXT_V,
            "max_tool_calls": MAX_TOOL_CALLS if self.name != "notool" else 0,
            "max_decisions": MAX_DECISIONS if self.name != "notool" else 1,
        }


__all__ = [
    "ANSWER_NONE",
    "EXT_V",
    "MAX_DECISIONS",
    "MAX_TOOL_CALLS",
    "Arm",
    "Attempt",
    "ToolCall",
]
