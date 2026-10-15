"""Shared HTTP plumbing for the data-bank clients: rate limiting, retries, disk cache.

The four banks are public services with published courtesy limits, and a routing policy that
decides to re-query one is a normal event -- so the caching here is not an optimization, it
is what keeps a run polite and what makes ``offline`` replay possible at all.

Three layers, in order:

1. **Disk cache** keyed on the full request. A hit never touches the network.
2. **Token bucket** per host, so a burst of parallel lookups still respects the published
   rate.
3. **Retry** on 429 and 5xx with exponential backoff; 4xx other than 429 is not retried
   because asking the same wrong question again will not make it right.

In ``offline`` mode a cache miss raises :class:`OfflineCacheMiss` rather than fetching. That
is what makes a recorded run reproducible: if replay needs something the recording never
fetched, we want a loud failure, not a silently different run.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..versions import HTTP_V

# Published courtesy limits. PubChem documents 5 requests/second and 400/minute; the others
# are not hard limits but these keep us well inside polite usage.
RATE_LIMITS: dict[str, float] = {
    "rest.uniprot.org": 5.0,
    "search.rcsb.org": 8.0,
    "data.rcsb.org": 8.0,
    "files.rcsb.org": 8.0,
    "www.ebi.ac.uk": 5.0,
    "pubchem.ncbi.nlm.nih.gov": 4.0,
}
DEFAULT_RATE = 4.0

USER_AGENT = "DrugOrch/0.1 (research prototype; https://github.com/)"


class OfflineCacheMiss(RuntimeError):
    """Offline mode was asked for something that is not in the cache."""


class TokenBucket:
    """A simple per-host rate limiter, safe across threads."""

    def __init__(self, rate_per_s: float, burst: int = 4):
        self.rate = rate_per_s
        self.burst = max(1, burst)
        self._tokens = float(self.burst)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def take(self) -> None:
        """Block until a token is available, then consume it."""
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(self.burst, self._tokens + (now - self._last) * self.rate)
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self.rate
            time.sleep(min(wait, 1.0))


_buckets: dict[str, TokenBucket] = {}
_buckets_lock = threading.Lock()


def _bucket(host: str) -> TokenBucket:
    with _buckets_lock:
        if host not in _buckets:
            _buckets[host] = TokenBucket(RATE_LIMITS.get(host, DEFAULT_RATE))
        return _buckets[host]


class RetryableStatus(RuntimeError):
    """A response worth trying again (429 or 5xx)."""


class BankClient:
    """One HTTP client for all four data banks, with caching and rate limiting."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        offline: bool = False,
        timeout: float = 30.0,
        ttl_s: float = 30 * 24 * 3600,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.offline = offline
        self.ttl_s = ttl_s
        self._client = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        )
        self.n_requests = 0
        self.n_cache_hits = 0

    # --- cache -----------------------------------------------------------------
    def _key(self, method: str, url: str, params: dict | None, body: Any) -> str:
        blob = json.dumps(
            {
                "m": method.upper(),
                "u": url,
                "p": params or {},
                "b": body,
                "v": HTTP_V,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(blob.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        # Two-level fan-out: a flat directory with tens of thousands of entries is slow to
        # list and unpleasant to inspect by hand.
        return self.cache_dir / key[:2] / f"{key}.json"

    def _read_cache(self, key: str) -> dict | None:
        p = self._cache_path(key)
        if not p.is_file():
            return None
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if self.ttl_s and (time.time() - payload.get("fetched_at", 0)) > self.ttl_s:
            return None
        return payload

    def _write_cache(self, key: str, status: int, content: Any, is_json: bool) -> None:
        p = self._cache_path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "fetched_at": time.time(),
            "status": status,
            "is_json": is_json,
            "content": content,
        }
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(p)

    # --- requests --------------------------------------------------------------
    @retry(
        retry=retry_if_exception_type((RetryableStatus, httpx.TransportError)),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=0.8, min=0.8, max=12),
        reraise=True,
    )
    def _fetch(self, method: str, url: str, params: dict | None, body: Any) -> httpx.Response:
        _bucket(urlparse(url).netloc).take()
        self.n_requests += 1
        resp = self._client.request(method, url, params=params, json=body)
        if resp.status_code == 429 or resp.status_code >= 500:
            raise RetryableStatus(f"{resp.status_code} from {url}")
        return resp

    def request(
        self,
        method: str,
        url: str,
        *,
        params: dict | None = None,
        body: Any = None,
        as_json: bool = True,
        allow_404: bool = True,
    ) -> Any:
        """Fetch, with cache. Returns parsed JSON, text, or ``None`` on an allowed 404."""
        key = self._key(method, url, params, body)
        hit = self._read_cache(key)
        if hit is not None:
            self.n_cache_hits += 1
            if hit["status"] == 404:
                return None
            return hit["content"]

        if self.offline:
            raise OfflineCacheMiss(f"offline: no cached response for {method} {url} {params or ''}")

        resp = self._fetch(method, url, params, body)

        if resp.status_code == 404:
            self._write_cache(key, 404, None, as_json)
            if allow_404:
                return None
            resp.raise_for_status()

        resp.raise_for_status()
        if as_json:
            try:
                content = resp.json()
            except ValueError:
                content = resp.text
                as_json = False
        else:
            content = resp.text
        self._write_cache(key, resp.status_code, content, as_json)
        return content

    def get(self, url: str, **kw) -> Any:
        return self.request("GET", url, **kw)

    def post(self, url: str, **kw) -> Any:
        return self.request("POST", url, **kw)

    def get_text(self, url: str, **kw) -> str | None:
        return self.request("GET", url, as_json=False, **kw)

    def describe(self) -> dict:
        return {
            "cache_dir": str(self.cache_dir),
            "offline": self.offline,
            "requests": self.n_requests,
            "cache_hits": self.n_cache_hits,
            "version": HTTP_V,
        }

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> BankClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
