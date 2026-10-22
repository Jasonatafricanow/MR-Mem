"""Volcengine Ark multimodal embedding client with a persistent per-text cache.

Ark's ``/embeddings/multimodal`` fuses every element of ``input`` into ONE vector,
so texts cannot be batched: each text is its own request. Requests run on a thread
pool, and every vector is cached in SQLite keyed by (model, dimensions, text), so a
rerun of the same data costs nothing and several processes can share the cache.

Key: env ``ARK_API_KEY``, else the first line of the file named by ``ARK_API_KEY_FILE``.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

DEFAULT_BASE = "https://ark.cn-beijing.volces.com/api/plan/v3"


def _load_key() -> str:
    key = os.environ.get("ARK_API_KEY", "").strip()
    if not key and os.environ.get("ARK_API_KEY_FILE"):
        text = Path(os.environ["ARK_API_KEY_FILE"]).read_text(encoding="utf-8-sig")
        key = text.strip().splitlines()[0].strip() if text.strip() else ""
    if not key:
        raise RuntimeError("Ark embedding needs ARK_API_KEY or ARK_API_KEY_FILE")
    return key


class ArkEmbedder:
    def __init__(
        self,
        model: str,
        *,
        dimensions: int = 2048,
        concurrency: int = 32,
        cache_path: Path,
        base_url: str | None = None,
        attempts: int = 8,
        requests_per_second: float = 4.0,
    ) -> None:
        self.model = model
        self.dimensions = dimensions
        self.concurrency = concurrency
        self.base_url = (base_url or os.environ.get("ARK_BASE") or DEFAULT_BASE).rstrip("/")
        self.attempts = attempts
        self._key = _load_key()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(cache_path), check_same_thread=False, timeout=60)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("CREATE TABLE IF NOT EXISTS vec (k TEXT PRIMARY KEY, v BLOB)")
        self._db.commit()
        self._db_lock = threading.Lock()
        # The account has a request-rate limit (HTTP 429 AccountRateLimitExceeded):
        # space request starts evenly instead of bursting the whole pool at once.
        self._interval = 1.0 / requests_per_second if requests_per_second > 0 else 0.0
        self._next_slot = time.monotonic()
        self._slot_lock = threading.Lock()

    def _wait_slot(self) -> None:
        if not self._interval:
            return
        with self._slot_lock:
            now = time.monotonic()
            slot = max(now, self._next_slot)
            self._next_slot = slot + self._interval
        if slot > now:
            time.sleep(slot - now)

    def _cache_key(self, text: str) -> str:
        raw = f"{self.model}\x00{self.dimensions}\x00{text}".encode()
        return hashlib.sha256(raw).hexdigest()

    def _request(self, text: str) -> np.ndarray:
        body = json.dumps({
            "model": self.model,
            "input": [{"type": "text", "text": text}],
            "dimensions": self.dimensions,
        }).encode()
        last = ""
        for attempt in range(self.attempts):
            self._wait_slot()
            req = urllib.request.Request(
                self.base_url + "/embeddings/multimodal", data=body,
                headers={"Authorization": "Bearer " + self._key,
                         "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read())["data"]
                    item = data[0] if isinstance(data, list) else data
                    return np.asarray(item["embedding"], dtype=np.float32)
            except urllib.error.HTTPError as exc:
                last = f"HTTP {exc.code}: {exc.read().decode(errors='replace')[:300]}"
                if exc.code not in (429, 500, 502, 503, 504):
                    break
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last = repr(exc)
            time.sleep(min(60, 2 ** attempt))
        raise RuntimeError(f"Ark embedding failed: {last}")

    def embed(self, texts: list[str]) -> np.ndarray:
        keys = [self._cache_key(t) for t in texts]
        found: dict[str, np.ndarray] = {}
        with self._db_lock:
            for start in range(0, len(keys), 500):
                chunk = keys[start:start + 500]
                marks = ",".join("?" * len(chunk))
                for k, blob in self._db.execute(
                        f"SELECT k, v FROM vec WHERE k IN ({marks})", chunk):
                    found[k] = np.frombuffer(blob, dtype=np.float32)
        missing = {k: t for k, t in zip(keys, texts, strict=True) if k not in found}
        if missing:
            items = list(missing.items())

            def one(kt: tuple[str, str]) -> np.ndarray:
                vector = self._request(kt[1])
                with self._db_lock:  # store at once: a later failure keeps finished work
                    self._db.execute("INSERT OR REPLACE INTO vec VALUES (?, ?)",
                                     (kt[0], vector.tobytes()))
                    self._db.commit()
                return vector

            with ThreadPoolExecutor(min(self.concurrency, len(items))) as pool:
                vectors = list(pool.map(one, items))
            for (k, _), v in zip(items, vectors, strict=True):
                found[k] = v
        return np.stack([found[k] for k in keys])
