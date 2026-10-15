"""Subsystem discovery: how DrugOrch finds the tools it can route between.

The subsystem layer was already most of a plugin system before this module existed. A
``Capability`` plus a ``ParamSpec`` tuple already generates the LLM's action enum, its JSON
schema, the prompt manifest, the API manifest and the web form, with no per-subsystem code
anywhere downstream; the trace view already gives a subsystem it has never heard of its own
lane. The only thing missing was a way to find a subsystem that does not live in this
package.

So this module adds discovery and nothing else. Three sources, searched in order and each
recorded on what it produced:

1. **Entry points** in the ``drugorch.subsystems`` group -- anything pip-installable.
2. **Plugin directories** from ``Settings.plugin_paths`` -- single ``.py`` files or packages,
   for a tool you are still writing.
3. The **built-ins**, declared as entry points in this project's own ``pyproject.toml`` so
   they load through exactly the same path. There is no privileged tier: if the mechanism is
   broken for plugins it is broken for docking too, which is the only way to be sure it works.

Two design positions worth stating.

**A plugin that fails to load must be visible.** The previous code caught ``ImportError`` and
passed, so a subsystem that could not import simply was not there -- no log line, nothing in
the run record, and a prompt that never mentioned it. The run would proceed and look normal.
Every failure here is captured instead and travels into ``run.json``, because "the tool was
not available" and "the tool was never installed" are different facts about a result.

**A plugin runs in this process with no sandbox.** Dropping a file in a plugin directory is
the same trust decision as `pip install`. This module does not pretend otherwise; it only
makes sure a misbehaving plugin degrades the run rather than ending it.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ENTRY_POINT_GROUP = "drugorch.subsystems"

FACTORY_NAME = "build"
"""The function a plugin module must expose. One name, so discovery never guesses."""


@dataclass(frozen=True, slots=True)
class SubsystemContext:
    """Everything a subsystem may be handed at construction.

    Built-in subsystems used to take whatever their author felt like -- ``cache=``,
    ``settings=``, both, or neither -- which meant ``build_registry`` had private knowledge of
    four different constructors and a loader had no way to call a fifth. One context object
    makes the construction contract declarable, and the two dependencies below are the entire
    surface anything has ever needed.
    """

    settings: Any
    cache: Any
    config: dict = field(default_factory=dict)
    """This plugin's slice of ``Settings.plugin_config``, by its entry-point name."""


@dataclass(frozen=True, slots=True)
class LoadFailure:
    """A subsystem that was found and could not be used.

    Recorded rather than swallowed: a run whose docking plugin failed to import is not the
    same experiment as a run configured without docking, and only this distinguishes them.
    """

    name: str
    source: str
    error: str

    def to_dict(self) -> dict:
        return {"name": self.name, "source": self.source, "error": self.error}


@dataclass(frozen=True, slots=True)
class Discovered:
    """One factory found somewhere, with enough provenance to put in the run record."""

    name: str
    factory: Any
    source: str
    distribution: str = ""
    distribution_version: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "source": self.source,
            "distribution": self.distribution,
            "distribution_version": self.distribution_version,
        }


# --- entry points ---------------------------------------------------------------
def discover_entry_points(
    group: str = ENTRY_POINT_GROUP,
) -> tuple[list[Discovered], list[LoadFailure]]:
    """Factories from installed distributions.

    Distribution name and version are read from the entry point's own metadata rather than
    asked of the plugin, so a plugin cannot report a version it is not actually running.
    """
    found: list[Discovered] = []
    failed: list[LoadFailure] = []

    try:
        eps = importlib.metadata.entry_points(group=group)
    except Exception as exc:  # noqa: BLE001 - a broken environment is not a crash
        return [], [LoadFailure("<entry_points>", "entry_point", f"{type(exc).__name__}: {exc}")]

    for ep in eps:
        try:
            factory = ep.load()
        except Exception as exc:  # noqa: BLE001 - one bad plugin, not a dead process
            failed.append(
                LoadFailure(ep.name, f"entry_point:{ep.value}", f"{type(exc).__name__}: {exc}")
            )
            continue
        dist = getattr(ep, "dist", None)
        found.append(
            Discovered(
                name=ep.name,
                factory=factory,
                source=f"entry_point:{ep.value}",
                distribution=getattr(dist, "name", "") or "",
                distribution_version=getattr(dist, "version", "") or "",
            )
        )
    return found, failed


# --- plugin directories ----------------------------------------------------------
def discover_paths(paths: Iterable[Path]) -> tuple[list[Discovered], list[LoadFailure]]:
    """Factories from local files, for a plugin that is not packaged yet.

    A directory entry is either ``<name>.py`` or ``<name>/__init__.py``; either must expose
    ``build``. Loaded under a ``drugorch_plugin_`` prefix so a plugin called ``docking.py``
    cannot shadow a real module on ``sys.path``.
    """
    found: list[Discovered] = []
    failed: list[LoadFailure] = []

    for raw in paths:
        root = Path(raw).expanduser()
        if not root.is_dir():
            failed.append(LoadFailure(str(root), "path", "not a directory"))
            continue

        for entry in sorted(root.iterdir()):
            if entry.name.startswith((".", "_")):
                continue
            if entry.is_dir():
                target = entry / "__init__.py"
                name = entry.name
            elif entry.suffix == ".py":
                target = entry
                name = entry.stem
            else:
                continue
            if not target.is_file():
                continue

            try:
                factory = _load_factory(name, target)
            except Exception as exc:  # noqa: BLE001
                failed.append(LoadFailure(name, f"path:{target}", f"{type(exc).__name__}: {exc}"))
                continue
            found.append(Discovered(name=name, factory=factory, source=f"path:{target}"))

    return found, failed


def _load_factory(name: str, target: Path):
    mod_name = f"drugorch_plugin_{name}"
    spec = importlib.util.spec_from_file_location(mod_name, target)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {target}")
    module = importlib.util.module_from_spec(spec)
    # Registered before exec so a plugin split across several files can import its siblings.
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    factory = getattr(module, FACTORY_NAME, None)
    if factory is None:
        raise AttributeError(f"{target} defines no {FACTORY_NAME}() factory")
    return factory


# --- construction -----------------------------------------------------------------
def instantiate(
    discovered: Iterable[Discovered], ctx: SubsystemContext
) -> tuple[list[Any], dict[str, Discovered], list[LoadFailure]]:
    """Call each factory and collect what it produced.

    A factory may return one subsystem or several -- a plugin that wraps a family of related
    tools should not have to ship one module per tool.
    """
    from .base import Subsystem

    built: list[Any] = []
    origins: dict[str, Discovered] = {}
    failed: list[LoadFailure] = []

    for d in discovered:
        plugin_ctx = SubsystemContext(
            settings=ctx.settings,
            cache=ctx.cache,
            config=dict((ctx.settings.plugin_config or {}).get(d.name, {}))
            if hasattr(ctx.settings, "plugin_config")
            else {},
        )
        try:
            produced = d.factory(plugin_ctx)
        except Exception as exc:  # noqa: BLE001
            failed.append(LoadFailure(d.name, d.source, f"{type(exc).__name__}: {exc}"))
            continue

        for sub in produced if isinstance(produced, list | tuple) else [produced]:
            # Structural only -- it checks the methods exist, not that they behave. Enough to
            # catch a factory that returned a config dict or None by mistake.
            if not isinstance(sub, Subsystem):
                failed.append(
                    LoadFailure(
                        d.name, d.source, f"{type(sub).__name__} does not satisfy Subsystem"
                    )
                )
                continue
            built.append(sub)
            origins.setdefault(sub.capability.key, d)

    return built, origins, failed


__all__ = [
    "ENTRY_POINT_GROUP",
    "FACTORY_NAME",
    "Discovered",
    "LoadFailure",
    "SubsystemContext",
    "discover_entry_points",
    "discover_paths",
    "instantiate",
]
