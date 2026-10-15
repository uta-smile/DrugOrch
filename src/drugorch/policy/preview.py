"""The exact request the routing model will receive, assembled before a run starts.

A prompt you cannot read is a prompt you cannot argue with. The LLM log shows what was sent
*after* a run -- too late to change anything, and only for runs that happened at all. This
renders the same three pieces beforehand, from the same functions the policy itself calls,
so what the page shows and what the model gets cannot drift:

* the **system text** -- the controller instructions, plus whatever this run appends to or
  replaces them with;
* the **capability manifest** -- generated from the registry, so it reflects exactly the
  tools this run was given and nothing else. Switch a tool off in the run form and it
  disappears from here, because it will not be in the registry the policy is handed;
* the **state block** -- ``RunState.summary()`` rendered as it will be at the first step.

The state is necessarily a projection: at step one nothing has been retrieved, so the
preview shows a run that has just started rather than inventing numbers. That is stated
rather than papered over, and the rest is literal.

Nothing here is written anywhere. The preview is computed per request from the arguments
given, which is why editing the prompt in the browser cannot leak into another run.
"""

from __future__ import annotations

import json

from ..budget import BudgetLedger
from ..orchestrator.state import RunState
from ..subsystems.base import Capability
from ..types import TargetSpec
from ..versions import PROMPT_V, SCHEMA_V
from .llm import PROMPT_NOTE, build_prompt, build_schema, compose_system, prompt_kind


def sample_state(
    goal: str, target: str, budget_units: float, seed: int = 42, max_steps: int = 40
) -> RunState:
    """A run as it stands at step one: resolved nothing, spent nothing, knows nothing.

    Constructed with the same :class:`~drugorch.orchestrator.state.RunState` the orchestrator
    builds, not a hand-written dictionary, so a field added to ``summary()`` shows up in the
    preview without anyone remembering to add it here.
    """
    return RunState(
        run_id="preview",
        goal=goal,
        target=TargetSpec(query=target),
        budget=BudgetLedger(total=float(budget_units)),
        seed=seed,
    )


def render(
    *,
    manifest: list[Capability],
    goal: str,
    target: str,
    budget_units: float,
    prompt_extra: str = "",
    prompt_override: str = "",
    max_steps: int = 40,
) -> dict:
    """Everything the model will be sent, in the three parts it is sent as."""
    state = sample_state(goal, target, budget_units, max_steps=max_steps)
    for cap in manifest:
        state.available.setdefault(cap.key, True)

    system = compose_system(prompt_extra, prompt_override)
    manifest_text = "Available subsystems:\n\n" + "\n".join(c.manifest_line() for c in manifest)
    state_text = build_prompt(state, manifest)
    schema = build_schema(manifest)

    return {
        "note": PROMPT_NOTE,
        "system": system,
        "manifest": manifest_text,
        "state": state_text,
        "schema": schema,
        "tools": [c.key for c in manifest],
        "kind": prompt_kind(prompt_extra, prompt_override),
        "versions": {"prompt": PROMPT_V, "schema": SCHEMA_V},
        # Roughly what the stable half costs to send. The manifest is the cached prefix, so
        # this is the number that stops being paid on every step after the first.
        "approx_chars": {
            "system": len(system),
            "manifest": len(manifest_text),
            "state": len(state_text),
            "schema": len(json.dumps(schema)),
        },
    }


__all__ = ["render", "sample_state"]
