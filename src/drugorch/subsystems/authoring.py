"""Everything needed to write a subsystem, exported so it can be read from the interface.

The plugin mechanism in :mod:`drugorch.subsystems.plugins` has always accepted third-party
tools. Nothing said so. The contract lived in the base-class docstrings, the factory had to
be named correctly by guesswork, and the directory the loader scans was a settings field
nobody surfaced. A mechanism that works and cannot be found is not usable.

This module exports the contract instead of describing it:

* :func:`contract` gives the machine-readable shape of a ``Capability`` and a ``ParamSpec``,
  generated from the dataclasses rather than written out, so it cannot fall behind them.
* :data:`TEMPLATE` is a complete working subsystem. It is not a sketch with holes: it is
  valid Python that loads, declares two parameters, returns records and can be routed to on
  the first run after it is saved.
* :func:`authoring_info` adds the two facts an author needs from the running server, which
  are where to put the file and what went wrong with the last attempt.

**Load failures are the important half.** A tool with a syntax error is discovered, fails to
import, and is then absent from the manifest, the prompt and the picker. Everything looks
normal and the subsystem is simply not there. :func:`authoring_info` surfaces the failure
list that the registry already collects, so a broken file reports itself.

**No sandbox.** A plugin runs in this process with the permissions of the server. Putting a
file in a plugin directory is the same trust decision as installing a package, and the
exported text says so where an author will read it.
"""

from __future__ import annotations

import inspect
from dataclasses import fields
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

from .base import Capability, ParamSpec, Subsystem, SubsystemResult
from .plugins import ENTRY_POINT_GROUP, FACTORY_NAME

TEMPLATE = '''"""An example DrugOrch subsystem.

Save this file into one of the plugin directories the Tools panel lists and restart the
server. It will appear in the tool picker and in the routing policy's prompt. Nothing needs
to be pip-installed.

A subsystem runs in the server process with no sandbox. Adding a plugin directory is the
same trust decision as installing a package.
"""

from __future__ import annotations

import threading

from drugorch.subsystems.base import (
    BaseSubsystem,
    Capability,
    Cost,
    ParamSpec,
    ProgressFn,
    SubsystemResult,
)

KEY = "example"

# The declaration. This single object produces the JSON schema the LLM is constrained to,
# the clamping applied to whatever it proposes, the prompt manifest, and the web form.
CAPABILITY = Capability(
    key=KEY,
    title="Example tool",
    # The LLM reads this. Say what the tool is for and when it is worth the budget, not how
    # it is implemented.
    summary=(
        "Assigns an example score to molecules already in the run. "
        "Call it after any tool that produces molecules."
    ),
    requires=("molecules",),      # preconditions, reported in the trace when unmet
    produces=("example_score",),
    params=(
        ParamSpec(
            name="limit",
            type="int",
            default=10,
            lo=1,
            hi=100,
            doc="How many molecules to process.",
        ),
        ParamSpec(
            name="mode",
            type="enum",
            default="fast",
            choices=("fast", "thorough"),
            doc="Which variant to run.",
        ),
    ),
    unit_cost=0.5,
    per="call",                   # or "molecule", which prices per novel SMILES
    typical_wall_s=1.0,
    version="v1",
)


class ExampleSubsystem(BaseSubsystem):
    """BaseSubsystem supplies preflight, estimate_cost, describe and urgency.

    Override any of them when the defaults are wrong. `urgency` in particular decides how
    much a rule-based policy wants to call this; the default is a neutral 0.15.
    """

    capability = CAPABILITY

    def __init__(self, ctx):
        # ctx.settings is the server configuration, ctx.cache a shared value cache, and
        # ctx.config whatever was put under this plugin's name in plugin_config.
        self.ctx = ctx

    def preflight(self, params: dict, state) -> tuple[bool, str]:
        """(usable, reason). False is not an error; it is a precondition not yet met."""
        if not state.molecules:
            return False, "no molecules in the run yet"
        return True, ""

    def run(
        self,
        params: dict,
        state,
        *,
        cancel: threading.Event,
        progress: ProgressFn,
    ) -> SubsystemResult:
        """Do the work. Must never raise, and must check `cancel` between units."""
        return self._safe(lambda: self._run(params, state, cancel, progress))

    def _run(self, params: dict, state, cancel, progress) -> SubsystemResult:
        p = self.capability.coerce_params(params)
        molecules = list(state.molecules.values())[: p["limit"]]

        records = []
        for i, mol in enumerate(molecules):
            if cancel is not None and cancel.is_set():
                break
            if progress is not None:
                progress(f"scoring {i + 1}/{len(molecules)}", (i + 1) / max(len(molecules), 1))
            records.append(
                {
                    "smiles": mol.smiles,
                    "props": {"example_mode": p["mode"]},
                    "scores": {"example_score": float(len(mol.smiles))},
                }
            )

        return SubsystemResult(
            subsystem=KEY,
            ok=True,
            records=records,
            note=f"scored {len(records)} molecules in {p['mode']} mode",
            cost=Cost(budget_units=CAPABILITY.unit_cost),
            # Rendered in the trace. Say how this call reached its result: which molecules,
            # chosen how, and what was discarded.
            explain={
                "selection": f"the first {p['limit']} molecules in the run, in insertion order",
                "mode": p["mode"],
                "n_considered": len(state.molecules),
            },
        )


def build(ctx):
    """The factory the loader looks for. The name must be exactly this."""
    return ExampleSubsystem(ctx)
'''

def _type_name(annotation: Any) -> str:
    """A readable name for a dataclass field's annotation.

    The annotations here include `tuple[str, ...]`, `float | None` and a `Literal`, none of
    which have a usable `__name__`. Rendering them by hand would be a second description of
    the same types, so this reads them off the objects.
    """
    if annotation is Ellipsis:
        return "..."
    origin = get_origin(annotation)
    if origin is not None:
        args = get_args(annotation)
        if str(origin).endswith("Literal"):
            return " | ".join(repr(a) for a in args)
        # `float | None` arrives as a UnionType whose __name__ is "UnionType". Rendering it
        # as a union is what the author will type, so spell it that way.
        if origin in (Union, UnionType):
            return " | ".join(_type_name(a) for a in args)
        inner = ", ".join(_type_name(a) for a in args)
        name = getattr(origin, "__name__", str(origin))
        return f"{name}[{inner}]"
    if annotation is type(None):
        return "None"
    return getattr(annotation, "__name__", str(annotation))


def _members() -> dict:
    """The protocol's members, read off the protocol.

    This was a hand-written dict, and it was wrong in all three entries: it named
    `capability()` as a method when it is a data attribute, named an `available()` hook that
    does not exist, and gave `run`'s parameters in the wrong order. Following it produced a
    subsystem that raises `AttributeError` inside `plugins.instantiate`, which escapes
    `build_registry` and turns `/api/manifest` into a 500, so the entire interface fails with
    "Could not load the manifest".

    The module docstring already promised the contract was generated rather than described.
    The one key that was not generated is the one that drifted, so it is generated now.
    """
    out: dict[str, str] = {}
    hints = get_type_hints(Subsystem)
    for name in sorted(getattr(Subsystem, "__protocol_attrs__", ())):
        member = getattr(Subsystem, name, None)
        if name in hints and not callable(member):
            out[name] = f"attribute, {_type_name(hints[name])}; read by the policy and the UI"
            continue
        sig = str(inspect.signature(member)).replace("(self, ", "(").replace("(self)", "()")
        doc = (inspect.getdoc(member) or "").splitlines()[:1]
        doc = doc[0] if doc else ""
        out[f"{name}{sig}"] = doc
    return out


def contract() -> dict:
    """The declaration shape of a subsystem, generated from the dataclasses and the protocol.

    Generated rather than written out: a field added to `Capability` appears here without
    anybody remembering to update a document, and a field removed cannot linger.

    `base.py` uses `from __future__ import annotations`, so every `field.type` is the source
    string rather than the object. Those strings render fine, but a `Literal` has to be
    resolved before its members can be listed, which is what `get_type_hints` is for. Reading
    `field.type` directly returned an empty list of parameter types and did it silently.
    """
    hints = {cls: get_type_hints(cls) for cls in (Capability, ParamSpec)}

    def describe(cls) -> list[dict]:
        out = []
        for f in fields(cls):
            resolved = hints[cls].get(f.name, f.type)
            out.append({"name": f.name, "type": _type_name(resolved)})
        return out

    return {
        "factory_name": FACTORY_NAME,
        "entry_point_group": ENTRY_POINT_GROUP,
        "capability_fields": describe(Capability),
        "param_spec_fields": describe(ParamSpec),
        "param_types": [str(a) for a in get_args(hints[ParamSpec]["type"])],
        # Derived, not listed: a hand-written copy had already lost `artifacts`.
        "result_keys": [f.name for f in fields(SubsystemResult)],
        "record_shape": {"smiles": "str", "props": "dict", "scores": "dict"},
        "methods": _members(),
    }


def authoring_info(settings, registry=None) -> dict:
    """The contract, the template, where files go, and what last failed to load."""
    paths = []
    for p in getattr(settings, "plugin_paths", ()) or ():
        path = Path(p)
        paths.append(
            {
                "path": str(path),
                "exists": path.is_dir(),
                # An author who points at a missing directory gets silence otherwise: the
                # loader records "not a directory" and the tool simply never appears.
                "writable": path.is_dir() and _writable(path),
            }
        )

    info: dict[str, Any] = {
        "plugin_paths": paths,
        "allow_list": list(getattr(settings, "plugins_allow", ()) or ()),
        "factory_name": FACTORY_NAME,
        "entry_point_group": ENTRY_POINT_GROUP,
        "contract": contract(),
        "template": TEMPLATE,
        "template_filename": "example_tool.py",
        "sandbox": (
            "A subsystem runs in the server process with the permissions of the server. "
            "Adding a plugin directory is the same trust decision as installing a package."
        ),
        "reload": (
            "A new file is picked up when the server restarts. Editing a file already "
            "loaded also needs a restart, because the module stays in memory."
        ),
    }

    if registry is not None:
        loaded = {
            k: {"key": k, "source": o.source, "distribution": o.distribution}
            for k, o in getattr(registry, "origins", {}).items()
        }
        info["loaded"] = list(loaded.values())

        # A failure for something that is loaded is not a failure to report.
        #
        # With an allow-list set, `build_registry` records a LoadFailure for each built-in
        # it filters out and then re-adds them from its own list. Passing the raw failures
        # through put a red "5 tools failed to load" banner directly above those same five
        # tools listed as loaded. The panel exists to make a genuinely missing tool visible,
        # and a banner that cries wolf on every allow-listed install defeats that.
        info["load_failures"] = [
            f.to_dict()
            for f in getattr(registry, "load_failures", [])
            if f.name not in loaded
        ]
    return info


def _writable(path: Path) -> bool:
    import os

    return os.access(path, os.W_OK)


__all__ = ["TEMPLATE", "authoring_info", "contract"]
