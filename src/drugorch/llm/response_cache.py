"""On-disk cache of model responses, keyed by the exact request that produced them.

This is what makes a run with an LLM in the loop reproducible. The model is not
deterministic, so a recorded run cannot be re-derived -- it can only be replayed, and replay
needs the original answers. One JSON file per request hash holds the system prompt, the user
prompt, the response and the usage, which is also enough to audit afterwards what the model
was actually shown.

In ``offline`` mode a miss raises instead of calling the API. That is deliberate: if a replay
asks something the recording never asked, the honest outcome is a loud failure, not a
quietly different run that still produces plausible numbers.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class LLMUnavailable(RuntimeError):
    """The model could not be reached, or refused, or replay hit an uncached request."""


def request_key(backend_key: str, system: str, prompt: str, schema: dict, version: str) -> str:
    """Stable 32-hex key for one request.

    The schema is part of the key because changing what the model is allowed to return
    changes what it returns; serving an old answer against a new schema would be a silent
    recipe mix of exactly the kind the version constants exist to prevent.
    """
    blob = json.dumps(
        {
            "backend": backend_key,
            "system": system,
            "prompt": prompt,
            "schema": schema,
            "v": version,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


class ResponseCache:
    """One pretty-printed JSON file per request, flat in one directory."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    def path(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str) -> dict | None:
        p = self.path(key)
        if not p.is_file():
            self.misses += 1
            return None
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # A truncated write is a miss, not a crash.
            self.misses += 1
            return None
        self.hits += 1
        return payload

    def put(self, key: str, payload: dict[str, Any]) -> None:
        p = self.path(key)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        tmp.replace(p)

    def __len__(self) -> int:
        return len(list(self.root.glob("*.json")))

    def describe(self) -> dict:
        return {"root": str(self.root), "hits": self.hits, "misses": self.misses}
