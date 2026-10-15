"""The subsystem registry: what exists, and what the policy is told about it.

Built once per run. The registry is also what generates the LLM's enum of legal subsystem
names and the capability manifest it reads, so there is exactly one list of subsystems in the
project and no way for the prompt to advertise something that is not wired up.

Subsystems are *discovered* rather than listed -- see :mod:`drugorch.subsystems.plugins`.
The four built-ins are declared as entry points in this project's own ``pyproject.toml`` and
load through exactly the same path as a third-party plugin, so there is no privileged tier to
rot: if discovery breaks, docking stops working too.
"""

from __future__ import annotations

from ..cache import ValueCache
from ..config import Settings, get_settings
from .base import Capability, Subsystem
from .plugins import (
    Discovered,
    LoadFailure,
    SubsystemContext,
    discover_entry_points,
    discover_paths,
    instantiate,
)


class Registry:
    """An ordered collection of subsystems, keyed by capability key."""

    def __init__(
        self,
        subsystems: list[Subsystem],
        *,
        load_failures: list[LoadFailure] | None = None,
        origins: dict[str, Discovered] | None = None,
        disabled: list[str] | None = None,
    ):
        self._items: dict[str, Subsystem] = {}
        self.load_failures: list[LoadFailure] = list(load_failures or [])
        self.origins: dict[str, Discovered] = dict(origins or {})
        self.disabled: list[str] = list(disabled or [])
        """Subsystems this installation has but this run was not given.

        Kept apart from ``load_failures`` on purpose: "the tool was switched off" and "the
        tool was broken" are different facts about a result, and a reader comparing two runs
        needs to be able to tell which one they are looking at.
        """

        for s in subsystems:
            key = s.capability.key
            if key in self._items:
                # First registration wins, and the loser is recorded. Silently replacing a
                # subsystem would let a plugin substitute itself for docking and leave no
                # trace in the run record, which is how a result becomes unreadable.
                origin = self.origins.get(key)
                self.load_failures.append(
                    LoadFailure(
                        name=key,
                        source=origin.source if origin else "?",
                        error=f"duplicate capability key {key!r}; first registration kept",
                    )
                )
                continue
            self._items[key] = s

        self._check_param_collisions()

    def _check_param_collisions(self) -> None:
        """Catch two subsystems declaring one parameter name with different types.

        The LLM's params object is a union across every subsystem and the first declaration
        of a name wins. A plugin declaring ``smiles`` as an int would therefore corrupt
        docking's schema with no error anywhere -- the model would be handed a contract that
        does not match the subsystem it is calling.
        """
        seen: dict[str, tuple[str, str]] = {}
        for key, sub in self._items.items():
            for p in sub.capability.params:
                prior = seen.get(p.name)
                if prior is None:
                    seen[p.name] = (key, p.type)
                elif prior[1] != p.type:
                    self.load_failures.append(
                        LoadFailure(
                            name=key,
                            source="param_collision",
                            error=(
                                f"parameter {p.name!r} is {p.type!r} here but {prior[1]!r} in "
                                f"{prior[0]!r}; the schema the model sees will not match"
                            ),
                        )
                    )

    def __contains__(self, key: object) -> bool:
        return key in self._items

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items.values())

    def get(self, key: str) -> Subsystem | None:
        return self._items.get(key)

    @property
    def keys(self) -> list[str]:
        return list(self._items)

    def capabilities(self) -> list[Capability]:
        return [s.capability for s in self._items.values()]

    def manifest(self, available: dict[str, bool] | None = None) -> str:
        """The capability manifest as the policy prompt renders it.

        Unavailable subsystems are listed but struck through with their reason, rather than
        hidden. A policy that cannot see that docking has failed will keep proposing it; one
        that can see *why* will route around it -- and the difference is worth observing.
        """
        avail = available or {}
        blocks = []
        for s in self._items.values():
            cap = s.capability
            if avail.get(cap.key, True):
                blocks.append(cap.manifest_line())
            else:
                blocks.append(f"- {cap.key}: UNAVAILABLE for the rest of this run")
        return "\n".join(blocks)

    def describe(self) -> dict:
        """Per-subsystem provenance, plus whatever failed to load.

        Distribution name and version are injected from the entry point's own metadata rather
        than asked of the subsystem, so a plugin cannot report a version it is not running.
        Load failures travel with it because "the tool was unavailable" and "the tool was
        never installed" are different facts about a result.
        """
        out: dict = {}
        for k, s in self._items.items():
            blob = dict(s.describe())
            origin = self.origins.get(k)
            if origin is not None:
                blob["source"] = origin.source
                if origin.distribution:
                    blob["distribution"] = origin.distribution
                    blob["distribution_version"] = origin.distribution_version
            out[k] = blob
        if self.load_failures:
            out["_load_failures"] = [f.to_dict() for f in self.load_failures]
        if self.disabled:
            out["_disabled"] = list(self.disabled)
        return out

    def to_dict(self) -> list[dict]:
        """The payload ``GET /api/manifest`` serves; the UI builds its forms from this."""
        return [s.capability.to_dict() for s in self._items.values()]


# --- the built-ins ----------------------------------------------------------------
#
# Each is a factory with the one signature every plugin uses. They are referenced from
# `pyproject.toml`'s entry points; `_build_builtins` calls them directly only when entry-point
# metadata is missing, which happens with a stale editable install.


def build_properties(ctx: SubsystemContext):
    from .properties import PropertiesSubsystem

    return PropertiesSubsystem(cache=ctx.cache)


def build_retrieval(ctx: SubsystemContext):
    from .retrieval import RetrievalSubsystem

    return RetrievalSubsystem(settings=ctx.settings)


def build_generation(ctx: SubsystemContext):
    from .generation import GenerationSubsystem

    return GenerationSubsystem()


def build_docking(ctx: SubsystemContext):
    from .docking import DockingSubsystem

    return DockingSubsystem(settings=ctx.settings, cache=ctx.cache)


def build_similarity(ctx: SubsystemContext):
    from .similarity import SimilaritySubsystem

    return SimilaritySubsystem(settings=ctx.settings)


# Registry order is load-bearing: `RoundRobinPolicy` cycles it, so changing it changes what
# that arm does and silently breaks comparability with every run recorded before. Entry
# points come back alphabetically, so the historical order is pinned here instead.
#
# `similarity` is appended rather than inserted, so the first four positions of the cycle are
# the ones every previously recorded run used. That is as much comparability as can be
# preserved -- a fifth subsystem does change what round-robin does from step five on, which
# is why it is also off by default in the run form.
BUILTIN_ORDER = ("properties", "retrieval", "generation", "docking", "similarity")


def _ordered(discovered: list[Discovered]) -> list[Discovered]:
    """Built-ins in their original order, then plugins alphabetically."""
    rank = {name: i for i, name in enumerate(BUILTIN_ORDER)}
    return sorted(discovered, key=lambda d: (rank.get(d.name, len(rank)), d.name))


_BUILTINS = {
    "properties": build_properties,
    "retrieval": build_retrieval,
    "generation": build_generation,
    "docking": build_docking,
    "similarity": build_similarity,
}


def _builtin_discovered() -> list[Discovered]:
    """The built-ins as if discovery had found them, for a stale editable install."""
    return [
        Discovered(name=k, factory=f, source=f"builtin:{k}") for k, f in _BUILTINS.items()
    ]


DEFAULT_TOOLS = ("properties", "retrieval", "generation", "docking")
"""Which subsystems a run gets when nobody says otherwise.

Everything discoverable is *offered* in the run form; this is what is ticked. The list is
the four that every recorded run so far has had, so a run started without touching the tool
picker is the same experiment it was before `similarity` existed. Turning a tool off is a
statement about the run, not a repair -- which is why it is a per-run choice and not a
setting in a file.
"""


def build_registry(
    settings: Settings | None = None,
    *,
    cache: ValueCache | None = None,
    enabled: tuple[str, ...] | list[str] | None = None,
) -> Registry:
    """Discover and construct every subsystem available to this installation.

    ``enabled`` restricts the result to the named subsystems. It is applied here, at
    construction, rather than by marking them unavailable later, and the difference is not
    cosmetic: an unavailable subsystem is still in the policy's JSON schema and can still be
    named by a model, which then fails validation and degrades the step to the fallback
    policy. A tool the operator switched off should be a tool the model was never offered.

    An unknown name is ignored rather than raised on -- the caller is a web form, and a
    stale tick should not be able to stop a run from starting.
    """
    st = settings or get_settings()
    st.ensure_dirs()
    vc = cache if cache is not None else ValueCache(st.value_cache_path)
    ctx = SubsystemContext(settings=st, cache=vc)

    discovered, failures = discover_entry_points()

    if getattr(st, "plugins_enabled", True) and getattr(st, "plugin_paths", ()):
        found, failed = discover_paths(st.plugin_paths)
        discovered.extend(found)
        failures.extend(failed)

    allow = tuple(getattr(st, "plugins_allow", ()) or ())
    if allow:
        kept, dropped = [], []
        for d in discovered:
            (kept if d.name in allow else dropped).append(d)
        for d in dropped:
            failures.append(LoadFailure(d.name, d.source, "not in DRUGORCH_PLUGINS_ALLOW"))
        discovered = kept

    # A built-in that discovery did not find is added directly. The usual cause is an
    # editable install whose `entry_points.txt` was written before the built-in existed, and
    # the old behaviour -- fall back only when *nothing* was discovered -- did not cover it:
    # four of five tools would load and the fifth would be quietly absent from the manifest,
    # from the model's schema and from the run record, with nothing anywhere saying so.
    # Entry points stay the discovery path, so a plugin still loads the way a built-in does;
    # this only guarantees that the built-ins are never the thing that goes missing, and
    # `describe()` reports the source either way. Done before the `enabled` filter, or a
    # stale install would turn a tool selection into an empty registry.
    have = {d.name for d in discovered}
    discovered.extend(d for d in _builtin_discovered() if d.name not in have)

    disabled: list[str] = []
    if enabled is not None:
        want = set(enabled)
        disabled = sorted(d.name for d in discovered if d.name not in want)
        discovered = [d for d in discovered if d.name in want]

    subsystems, origins, build_failures = instantiate(_ordered(discovered), ctx)
    failures.extend(build_failures)

    return Registry(subsystems, load_failures=failures, origins=origins, disabled=disabled)
