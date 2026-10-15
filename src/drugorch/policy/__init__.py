"""Routing policies: the ladder of baselines plus the LLM planner.

The ladder exists so that a result can be attributed. Each rung removes one explanation for
why the arm under test might be winning:

===============  ==========================================================
``fixed``        the null -- a hard-coded pipeline that reads no state.
                 This is what most published systems do, and the thing the
                 paper argues against.
``random``       control: does *any* variation in subsystem order help?
``roundrobin``   control: does varying it *unpredictably* help, or would
                 any rotation do?
``scripted``     the control that matters -- competent hand-written rules
                 reading exactly the state the LLM reads.
``oracle``       ceiling, not a method. Reads the answer key.
``llm``          the arm under test.
===============  ==========================================================

Every arm emits the same ``Decision`` type, sees the same ``RunState.summary()``, and is
wrapped in the same :class:`~drugorch.policy.base.SafePolicy`. Anything that broke that
symmetry would make a difference between arms uninterpretable.
"""

from __future__ import annotations

from .base import Policy, SafePolicy

POLICY_NAMES = ("fixed", "random", "roundrobin", "scripted", "oracle", "llm")

DESCRIPTIONS = {
    "fixed": "Hard-coded pipeline order; reads no run state. The baseline.",
    "random": "Uniform random among available subsystems. Control.",
    "roundrobin": "Fixed rotation through available subsystems. Control.",
    "scripted": "Hand-written rules over the run state. The control that matters.",
    "oracle": "Reads the known-actives answer key. A ceiling, not a method.",
    "llm": "Claude chooses the next subsystem from the state and capability manifest.",
}


def build_policy(name: str, *, seed: int = 42, settings=None, **kwargs) -> SafePolicy:
    """Construct a policy by name, wrapped so it can never raise into the run loop.

    Every arm falls back to ``scripted`` rather than to ``fixed``: when a policy fails we
    want the run to continue sensibly, and a fallback that ignores the state would make the
    remainder of the run less informative than it needs to be. Fallbacks are recorded as
    ``source="fallback"`` so they are visible in the trace and countable in the ablation.
    """
    key = name.strip().lower()
    from .scripted import ScriptedPolicy

    fallback = ScriptedPolicy()

    if key == "fixed":
        from .fixed import FixedPolicy

        return SafePolicy(FixedPolicy(), fallback)
    if key == "random":
        from .controls import RandomPolicy

        return SafePolicy(RandomPolicy(seed=seed), fallback)
    if key == "roundrobin":
        from .controls import RoundRobinPolicy

        return SafePolicy(RoundRobinPolicy(), fallback)
    if key == "scripted":
        return SafePolicy(ScriptedPolicy(), None)
    if key == "oracle":
        from .oracle import OraclePolicy

        return SafePolicy(OraclePolicy(), fallback)
    if key == "llm":
        from .llm import LLMPolicy

        return SafePolicy(LLMPolicy(settings=settings, **kwargs), fallback)

    raise ValueError(f"unknown policy {name!r}; choose from {', '.join(POLICY_NAMES)}")


__all__ = ["DESCRIPTIONS", "POLICY_NAMES", "Policy", "SafePolicy", "build_policy"]
