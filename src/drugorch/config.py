"""Process-wide settings, resolved once from the environment and a ``.env`` file.

Everything that varies between machines lives here, so no module below reaches for an
environment variable of its own. ``Settings.describe()`` is what the run provenance blob
records -- it deliberately omits the API key.

The Anthropic key has one extra wrinkle: it can also be supplied at runtime from the web UI,
which is the common case for a local demo where editing ``.env`` and restarting the server is
friction with no payoff. That key lives in a module-level slot rather than on ``Settings``
because ``Settings`` is an lru_cached singleton -- rebuilding it to change one field would
hand out a different object to anything still holding a reference. ``api_key()`` reads the
slot on every call, so a policy constructed after the key is entered picks it up without a
restart.
"""

from __future__ import annotations

import os
import re
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .versions import all_versions

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"

# Set from the web UI (or a CLI flag) at runtime; takes precedence over `.env` and the
# environment, because a key entered deliberately now should beat one configured earlier.
_runtime_api_key: str | None = None


def set_runtime_api_key(key: str | None) -> None:
    """Install -- or clear, with ``None`` -- the process-wide runtime Anthropic key."""
    global _runtime_api_key  # noqa: PLW0603 - one process-wide slot, by design
    cleaned = (key or "").strip()
    _runtime_api_key = cleaned or None


def get_runtime_api_key() -> str | None:
    return _runtime_api_key


def mask_key(key: str | None) -> str:
    """``sk-ant-api0...a1b2`` -- enough to tell two keys apart, not enough to use one.

    Worth doing even on a localhost-only server, because this UI gets screen-recorded.
    """
    if not key:
        return ""
    if len(key) <= 12:
        return key[:3] + "\u2026"
    return f"{key[:11]}\u2026{key[-4:]}"


def env_file_has_key(path: Path | None = None) -> bool:
    """Whether ``.env`` currently carries a non-empty ``ANTHROPIC_API_KEY``.

    Read from the file rather than from ``Settings``, which snapshots ``.env`` at import.
    A key persisted during this session would otherwise be reported as absent, and the UI
    would offer no way to remove it.
    """
    p = path or ENV_PATH
    if not p.is_file():
        return False
    for line in p.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*ANTHROPIC_API_KEY\s*=\s*(.*)", line)
        if m and m.group(1).strip().strip("\"'"):
            return True
    return False


def write_env_value(name: str, value: str, path: Path | None = None) -> Path:
    """Persist one ``NAME=value`` into ``.env``, rewriting the line if it already exists.

    Settings changed in the web interface mutate the running process, which is invisible to
    the next one: a provider chosen in the UI reverted to the default on restart, and the
    CLI never saw it at all. Writing them here makes the choice outlive the process that
    made it. Everything else in the file is preserved verbatim -- it is the user's file,
    and it may carry other overrides and comments worth keeping.
    """
    p = path or ENV_PATH
    value = (value or "").strip()
    lines = p.read_text(encoding="utf-8").splitlines() if p.is_file() else []
    out: list[str] = []
    replaced = False
    for line in lines:
        if re.match(rf"\s*{re.escape(name)}\s*=", line):
            if not replaced:
                out.append(f"{name}={value}")
                replaced = True
            continue  # drop duplicate assignments; the last one would have won silently
        out.append(line)
    if not replaced:
        out.append(f"{name}={value}")
    p.write_text("\n".join(out).rstrip("\n") + "\n", encoding="utf-8")
    return p


def write_env_key(key: str | None, path: Path | None = None) -> Path:
    """Persist ``ANTHROPIC_API_KEY`` into ``.env``."""
    p = path or ENV_PATH
    value = (key or "").strip()
    lines = p.read_text(encoding="utf-8").splitlines() if p.is_file() else []
    out: list[str] = []
    replaced = False
    for line in lines:
        if re.match(r"\s*ANTHROPIC_API_KEY\s*=", line):
            if not replaced:
                out.append(f"ANTHROPIC_API_KEY={value}")
                replaced = True
            continue  # drop duplicate assignments; the last one would have won silently
        out.append(line)
    if not replaced:
        out.append(f"ANTHROPIC_API_KEY={value}")
    p.write_text("\n".join(out).rstrip("\n") + "\n", encoding="utf-8")
    return p


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DRUGORCH_",
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- paths -----------------------------------------------------------------
    data_dir: Path = Field(default=REPO_ROOT / "data")
    web_dir: Path = Field(default=REPO_ROOT / "web")
    vina_exe: Path = Field(default=REPO_ROOT / "tools" / "vina.exe")

    # --- behaviour -------------------------------------------------------------
    offline: bool = Field(
        default=False,
        description="Refuse network and LLM calls; a cache miss raises instead of fetching. "
        "This is what makes a recorded run replayable.",
    )

    # --- LLM -------------------------------------------------------------------
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    llm_provider: Literal["anthropic", "openai", "chatgpt"] = "anthropic"
    """Which backend the ``llm`` policy uses.

    ``chatgpt`` runs decisions against the user's own ChatGPT plan over OAuth and needs no
    API key at all; ``openai`` uses a platform API key; ``anthropic`` is the default.
    """

    llm_model: str = "claude-opus-5-5"
    openai_model: str = "gpt-5.5"

    # Effort is stored per provider rather than shared, because the vocabularies differ:
    # `max` is Anthropic-only and `none` is OpenAI-only. One field would silently carry an
    # invalid value across a provider switch, and the rejection lands mid-run as a fallback.
    llm_effort: str = "medium"
    openai_effort: str = "medium"
    llm_timeout_s: float = 60.0
    llm_max_tokens: int = 8000

    # --- plugins ---------------------------------------------------------------
    plugins_enabled: bool = True
    plugin_paths: tuple[Path, ...] = ()
    """Directories scanned for local plugin modules exposing a ``build(ctx)`` factory.

    A plugin runs in this process with no sandbox, so adding a directory here is the same
    trust decision as installing a package. Stated plainly rather than implied.
    """

    plugins_allow: tuple[str, ...] = ()
    """When non-empty, only these plugin names load. Everything else is recorded as refused
    rather than quietly skipped."""

    plugin_config: dict = Field(default_factory=dict)
    """Per-plugin settings, keyed by plugin name, handed to its factory. ``Settings`` is a
    closed model, so a plugin cannot add its own fields -- this is its slot."""

    # --- docking ---------------------------------------------------------------
    vina_exhaustiveness: int = 8
    vina_cpu_per_worker: int = 4
    vina_workers: int = 4
    vina_timeout_s: float = 300.0

    # --- orchestration defaults ------------------------------------------------
    default_budget_units: float = 120.0
    max_steps: int = 40
    no_progress_steps: int = 6
    wall_clock_cap_s: float = 3600.0

    # --- derived paths ---------------------------------------------------------
    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def http_cache_dir(self) -> Path:
        return self.cache_dir / "http"

    @property
    def llm_cache_dir(self) -> Path:
        return self.cache_dir / "llm"

    @property
    def receptor_cache_dir(self) -> Path:
        return self.cache_dir / "receptors"

    @property
    def pose_cache_dir(self) -> Path:
        """Docked poses, keyed by the same string as the score cache.

        Beside the value cache rather than under ``runs/`` on purpose: scores are shared
        across runs, so a pose store that was not would leave a cache-hit run with scores
        and no geometry to show for them.
        """
        return self.cache_dir / "poses"

    @property
    def value_cache_path(self) -> Path:
        return self.cache_dir / "values.sqlite"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "drugorch.sqlite"

    @property
    def runs_dir(self) -> Path:
        return self.data_dir / "runs"

    @property
    def registry_dir(self) -> Path:
        return self.data_dir / "registry"

    def ensure_dirs(self) -> None:
        """Create every directory the app writes to. Idempotent."""
        for p in (
            self.data_dir,
            self.cache_dir,
            self.http_cache_dir,
            self.llm_cache_dir,
            self.receptor_cache_dir,
            self.pose_cache_dir,
            self.runs_dir,
            self.registry_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)

    # --- resolution ------------------------------------------------------------
    def resolved_vina(self) -> Path | None:
        """The Vina executable, or ``None`` if it cannot be found.

        Checks the configured path first, then ``PATH`` -- so a system-wide install works
        without configuration, and a vendored ``tools/vina.exe`` wins when both exist.
        """
        if self.vina_exe.is_file():
            return self.vina_exe
        found = shutil.which("vina") or shutil.which("vina.exe")
        return Path(found) if found else None

    def api_key(self) -> str | None:
        """The effective Anthropic key: runtime slot, then ``.env``, then the environment."""
        return (
            _runtime_api_key
            or self.anthropic_api_key
            or os.environ.get("ANTHROPIC_API_KEY")
            or None
        )

    def openai_key(self) -> str | None:
        return self.openai_api_key or os.environ.get("OPENAI_API_KEY") or None

    @property
    def chatgpt_session_path(self) -> Path:
        return self.cache_dir / "chatgpt_auth.json"

    def model_for_provider(self) -> str:
        return self.llm_model if self.llm_provider == "anthropic" else self.openai_model

    def effort_for_provider(self) -> str:
        """The effort for the active provider, corrected if it is not one that provider takes."""
        from .llm.catalog import coerce_effort

        raw = self.llm_effort if self.llm_provider == "anthropic" else self.openai_effort
        return coerce_effort(self.llm_provider, raw)

    def set_model(self, model: str) -> None:
        if self.llm_provider == "anthropic":
            self.llm_model = model
            self.persist("DRUGORCH_LLM_MODEL", model)
        else:
            self.openai_model = model
            self.persist("DRUGORCH_OPENAI_MODEL", model)

    def set_effort(self, effort: str) -> None:
        from .llm.catalog import coerce_effort

        value = coerce_effort(self.llm_provider, effort)
        if self.llm_provider == "anthropic":
            self.llm_effort = value
            self.persist("DRUGORCH_LLM_EFFORT", value)
        else:
            self.openai_effort = value
            self.persist("DRUGORCH_OPENAI_EFFORT", value)

    def set_provider(self, provider: str) -> None:
        self.llm_provider = provider  # type: ignore[assignment]
        self.persist("DRUGORCH_LLM_PROVIDER", provider)

    def persist(self, name: str, value: str) -> None:
        """Write a setting to ``.env`` so it survives this process. Best effort.

        A read-only checkout should not stop someone changing a model for this session, so
        a failure here is silent -- the in-memory change still took.
        """
        import contextlib

        with contextlib.suppress(OSError):
            write_env_value(name, value)

    def api_key_source(self) -> str | None:
        """Where the effective key came from: ``"session"``, ``"env"`` or ``None``.

        Reported to the UI and into run provenance, so a run records *how* it was keyed
        without recording the key itself.
        """
        if _runtime_api_key:
            return "session"
        if self.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY"):
            return "env"
        return None

    def describe(self) -> dict:
        """JSON-safe provenance blob. Never includes the API key."""
        return {
            "data_dir": str(self.data_dir),
            "vina_exe": str(self.vina_exe),
            "vina_resolved": str(self.resolved_vina() or ""),
            "offline": self.offline,
            "llm_provider": self.llm_provider,
            "llm_model": self.model_for_provider(),
            "llm_effort": self.effort_for_provider(),
            "plugins_enabled": self.plugins_enabled,
            "plugin_paths": [str(p) for p in self.plugin_paths],
            "plugins_allow": list(self.plugins_allow),
            "has_api_key": self.api_key() is not None,
            "api_key_source": self.api_key_source(),
            "vina_exhaustiveness": self.vina_exhaustiveness,
            "vina_workers": self.vina_workers,
            "default_budget_units": self.default_budget_units,
            "max_steps": self.max_steps,
            "versions": all_versions(),
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """The process-wide settings singleton."""
    return Settings()
