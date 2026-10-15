"""The runtime API key: precedence, masking, .env rewriting, and what never leaks.

The key can now arrive three ways -- the web UI, ``.env``, the environment -- so the rule
about which one wins needs pinning, as does the guarantee that none of them reach a run
record. A key written into provenance would be published the moment a run directory is
shared alongside a paper.
"""

from __future__ import annotations

import json

import pytest

from drugorch.config import (
    Settings,
    env_file_has_key,
    get_runtime_api_key,
    mask_key,
    set_runtime_api_key,
    write_env_key,
)

FAKE = "sk-ant-api03-" + "x" * 40 + "WXYZ"


@pytest.fixture(autouse=True)
def _clear_runtime_key():
    """The runtime key is process-wide, so a test that sets it must not leak into others."""
    set_runtime_api_key(None)
    yield
    set_runtime_api_key(None)


# --- precedence ------------------------------------------------------------
def test_runtime_key_overrides_env():
    """A key typed in the UI must beat a stale one in .env, or entering it does nothing."""
    s = Settings(anthropic_api_key="sk-ant-from-dotenv")
    assert s.api_key() == "sk-ant-from-dotenv"
    assert s.api_key_source() == "env"

    set_runtime_api_key(FAKE)
    assert s.api_key() == FAKE
    assert s.api_key_source() == "session"


def test_clearing_the_runtime_key_falls_back_to_env():
    s = Settings(anthropic_api_key="sk-ant-from-dotenv")
    set_runtime_api_key(FAKE)
    set_runtime_api_key(None)
    assert s.api_key() == "sk-ant-from-dotenv"
    assert s.api_key_source() == "env"


def test_blank_input_clears_rather_than_storing_an_empty_key():
    set_runtime_api_key(FAKE)
    set_runtime_api_key("   ")
    assert get_runtime_api_key() is None


def test_source_is_none_when_nothing_is_configured(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    s = Settings(anthropic_api_key=None)
    assert s.api_key() is None
    assert s.api_key_source() is None


# --- masking ---------------------------------------------------------------
def test_mask_shows_enough_to_identify_and_not_enough_to_use():
    masked = mask_key(FAKE)
    assert FAKE not in masked
    assert masked.startswith("sk-ant-api")
    assert masked.endswith("WXYZ")
    assert len(masked) < len(FAKE)


def test_mask_of_no_key_is_empty():
    assert mask_key(None) == ""


# --- .env round trip -------------------------------------------------------
def test_write_env_key_preserves_other_settings(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "# a comment\nDRUGORCH_OFFLINE=true\nANTHROPIC_API_KEY=old\nDRUGORCH_LLM_EFFORT=high\n",
        encoding="utf-8",
    )
    write_env_key(FAKE, env)
    text = env.read_text(encoding="utf-8")
    assert f"ANTHROPIC_API_KEY={FAKE}" in text
    assert "old" not in text
    assert "DRUGORCH_OFFLINE=true" in text
    assert "DRUGORCH_LLM_EFFORT=high" in text
    assert "# a comment" in text


def test_write_env_key_appends_when_absent(tmp_path):
    env = tmp_path / ".env"
    env.write_text("DRUGORCH_OFFLINE=true\n", encoding="utf-8")
    write_env_key(FAKE, env)
    assert f"ANTHROPIC_API_KEY={FAKE}" in env.read_text(encoding="utf-8")


def test_write_env_key_collapses_duplicate_assignments(tmp_path):
    """Two assignments would leave the loser invisible but still on disk."""
    env = tmp_path / ".env"
    env.write_text("ANTHROPIC_API_KEY=one\nANTHROPIC_API_KEY=two\n", encoding="utf-8")
    write_env_key(FAKE, env)
    text = env.read_text(encoding="utf-8")
    assert text.count("ANTHROPIC_API_KEY=") == 1
    assert "one" not in text and "two" not in text


def test_write_env_key_creates_the_file_when_missing(tmp_path):
    env = tmp_path / ".env"
    write_env_key(FAKE, env)
    assert env.is_file()
    assert f"ANTHROPIC_API_KEY={FAKE}" in env.read_text(encoding="utf-8")


def test_env_file_has_key_reads_the_file_not_the_cached_settings(tmp_path):
    """A key persisted mid-session must be visible, or the UI cannot offer to remove it.

    ``Settings`` snapshots ``.env`` at import, so asking it would report a key written
    minutes ago as absent.
    """
    env = tmp_path / ".env"
    assert env_file_has_key(env) is False  # missing file

    env.write_text("ANTHROPIC_API_KEY=\n", encoding="utf-8")
    assert env_file_has_key(env) is False  # present but blank

    env.write_text(f"# comment\nANTHROPIC_API_KEY={FAKE}\n", encoding="utf-8")
    assert env_file_has_key(env) is True

    write_env_key("", env)
    assert env_file_has_key(env) is False  # blanked again


def test_env_file_has_key_ignores_quotes_and_spacing(tmp_path):
    env = tmp_path / ".env"
    env.write_text(f'  ANTHROPIC_API_KEY = "{FAKE}"\n', encoding="utf-8")
    assert env_file_has_key(env) is True
    env.write_text('ANTHROPIC_API_KEY=""\n', encoding="utf-8")
    assert env_file_has_key(env) is False


# --- the guarantee ---------------------------------------------------------
def test_a_runtime_key_never_reaches_provenance():
    """describe() feeds run.json; a key there would be published with the run directory."""
    set_runtime_api_key(FAKE)
    s = Settings(anthropic_api_key=None)
    blob = json.dumps(s.describe())
    assert FAKE not in blob
    assert "sk-ant" not in blob
    # But *that* a key was in use, and where it came from, is recorded.
    assert s.describe()["has_api_key"] is True
    assert s.describe()["api_key_source"] == "session"
