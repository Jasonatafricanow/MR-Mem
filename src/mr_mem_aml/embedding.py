"""Embedding providers for the dense retrieval arm.

All implement ``mr_mem.memory.embedding.EmbeddingProvider``; vectors stay a
rebuildable derived index under MR-Mem's existing Qdrant projection.

- ``open_fastembed``: MR-Mem's own local ONNX provider (e.g. BAAI/bge-small-en-v1.5),
  the same class the MR-Mem baseline uses, so a baseline run is reproduced exactly.
- ``OpenAICompatibleEmbedding``: any ``/embeddings`` endpoint (OpenAI, SiliconFlow,
  Hugging Face TEI, Ollama, vLLM, ...), key optional for local servers.
- ``GeminiEmbedding``: Gemini ``embedContent``.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from mr_mem.memory.embedding import EmbeddingIdentity


def _post(url: str, body: dict, headers: dict[str, str], timeout: float, attempts: int) -> dict:
    data = json.dumps(body).encode("utf-8")
    last = ""
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(
            url, data=data, method="POST", headers={"Content-Type": "application/json", **headers}
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last = f"HTTP {exc.code}: {exc.read()[:300].decode('utf-8', 'replace')}"
            if exc.code in (400, 401, 403, 404, 422):
                break
        except Exception as exc:  # noqa: BLE001 - transport retry boundary
            last = f"{type(exc).__name__}: {exc}"
        if attempt < attempts:
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"embedding endpoint failed: {last}")


def _key(env: str) -> str:
    key = os.environ.get(env)
    if not key:
        raise RuntimeError(f"environment variable {env} is not set")
    return key


class GeminiEmbedding:
    def __init__(
        self,
        *,
        model: str,
        dimension: int,
        api_key_env: str = "GEMINI_API_KEY",
        timeout: float = 60,
        attempts: int = 4,
    ) -> None:
        self._identity = EmbeddingIdentity("gemini-http", model, dimension, "api")
        self.model, self.api_key_env = model, api_key_env
        self.timeout, self.attempts = timeout, attempts

    @property
    def identity(self) -> EmbeddingIdentity:
        return self._identity

    def embed(self, text: str) -> tuple[float, ...]:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:embedContent"
        data = _post(
            url,
            {
                "model": f"models/{self.model}",
                "content": {"parts": [{"text": text}]},
                "outputDimensionality": self._identity.dimension,
            },
            {"x-goog-api-key": _key(self.api_key_env)},
            self.timeout,
            self.attempts,
        )
        return tuple(float(v) for v in data["embedding"]["values"])


class _VectorCache:
    """Persistent SQLite cache keyed by (model, dimension, text). Hosted endpoints do not return
    the same vector twice (2026-10-31: qwen3.7-text-embedding-flash, cosine 0.998 between two
    calls), so without it a restart or a rerun re-pays every text and reorders the dense arm."""

    def __init__(self, path: str | Path, model: str, dimension: int) -> None:
        import sqlite3
        import threading

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS vec (k TEXT PRIMARY KEY, v BLOB)")
        self._lock = threading.Lock()
        self._prefix = f"{model}|{dimension}|"

    def _k(self, text: str) -> str:
        import hashlib

        return hashlib.sha256((self._prefix + text).encode("utf-8")).hexdigest()

    def get(self, text: str) -> tuple[float, ...] | None:
        import array

        with self._lock:
            row = self._db.execute("SELECT v FROM vec WHERE k = ?", (self._k(text),)).fetchone()
        return tuple(array.array("f", row[0])) if row else None

    def put(self, text: str, vector: tuple[float, ...]) -> None:
        import array

        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO vec VALUES (?, ?)",
                (self._k(text), array.array("f", vector).tobytes()),
            )
            self._db.commit()


class OpenAICompatibleEmbedding:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        dimension: int,
        api_key_env: str | None = None,
        send_dimensions: bool = False,
        headers: dict[str, str] | None = None,
        timeout: float = 60,
        attempts: int = 4,
        cache_path: str | Path | None = None,
        batch: int = 10,
        workers: int = 4,
    ) -> None:
        self._identity = EmbeddingIdentity("openai-compatible-http", model, dimension, "api")
        self.url = base_url.rstrip("/") + "/embeddings"
        self.model, self.api_key_env = model, api_key_env
        self.send_dimensions, self.headers = send_dimensions, dict(headers or {})
        self.timeout, self.attempts = timeout, attempts
        self._cache = _VectorCache(cache_path, model, dimension) if cache_path else None
        # texts per request and requests in flight: one request per message made an Add of a
        # few dozen messages take 10+ seconds (2026-10-31, behind a 20-second gateway)
        self.batch, self.workers = max(1, int(batch)), max(1, int(workers))

    def embed_many(self, texts: list[str]) -> list[tuple[float, ...]]:
        """Vectors of ``texts`` in input order: cached ones from the cache, the rest in batches
        of ``batch`` texts, ``workers`` requests at a time."""
        out: dict[str, tuple[float, ...]] = {}
        missing = []
        for text in dict.fromkeys(texts):
            known = self._cache.get(text) if self._cache is not None else None
            if known is not None:
                out[text] = known
            else:
                missing.append(text)
        chunks = [missing[i : i + self.batch] for i in range(0, len(missing), self.batch)]
        if chunks:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(min(self.workers, len(chunks))) as pool:
                for chunk, vectors in zip(
                    chunks, pool.map(self._request_many, chunks), strict=True
                ):
                    for text, vector in zip(chunk, vectors, strict=True):
                        out[text] = vector
                        if self._cache is not None:
                            self._cache.put(text, vector)
        return [out[t] for t in texts]

    def _request_many(self, texts: list[str]) -> list[tuple[float, ...]]:
        if len(texts) == 1:
            return [self._request(texts[0])]
        headers = dict(self.headers)
        if self.api_key_env:
            headers["Authorization"] = f"Bearer {_key(self.api_key_env)}"
        body: dict = {"model": self.model, "input": texts}
        if self.send_dimensions:
            body["dimensions"] = self._identity.dimension
        data = _post(self.url, body, headers, self.timeout, self.attempts)["data"]
        if len(data) != len(texts):
            raise RuntimeError(f"embedding endpoint returned {len(data)} rows for {len(texts)}")
        rows = sorted(data, key=lambda d: d.get("index", 0))
        vectors = [tuple(float(v) for v in d["embedding"]) for d in rows]
        if any(len(v) != self._identity.dimension for v in vectors):
            raise RuntimeError("embedding endpoint returned vectors of the wrong dimension")
        return vectors

    @property
    def identity(self) -> EmbeddingIdentity:
        return self._identity

    def embed(self, text: str) -> tuple[float, ...]:
        if self._cache is not None and (known := self._cache.get(text)) is not None:
            return known
        vector = self._request(text)
        if self._cache is not None:
            self._cache.put(text, vector)
        return vector

    def _request(self, text: str) -> tuple[float, ...]:
        headers = dict(self.headers)
        if self.api_key_env:
            headers["Authorization"] = f"Bearer {_key(self.api_key_env)}"
        body: dict = {"model": self.model, "input": text}
        if self.send_dimensions:  # only models that support shortening (e.g. text-embedding-3)
            body["dimensions"] = self._identity.dimension
        data = _post(self.url, body, headers, self.timeout, self.attempts)
        vector = tuple(float(v) for v in data["data"][0]["embedding"])
        if len(vector) != self._identity.dimension:
            raise RuntimeError(
                f"embedding endpoint returned {len(vector)} dims, configured "
                f"{self._identity.dimension}"
            )
        return vector


def open_fastembed(*, model: str, dimension: int, model_dir: str | Path):
    """MR-Mem's local FastEmbed provider over verified model files in ``model_dir``."""
    from mr_mem.memory.providers.fastembed import FastEmbedEmbedding, model_revision

    path = Path(model_dir).expanduser().resolve()
    identity = EmbeddingIdentity("fastembed", model, dimension, model_revision(path))
    return FastEmbedEmbedding(identity, model_dir=path)


def fetch_fastembed_model(model: str, target_dir: str | Path) -> Path:
    """Download a FastEmbed model once and flatten its files into ``target_dir``."""
    import shutil
    import tempfile

    from fastembed import TextEmbedding

    target = Path(target_dir).expanduser().resolve()
    if target.exists() and any(target.rglob("*.onnx")):
        return target
    with tempfile.TemporaryDirectory() as tmp:
        TextEmbedding(model_name=model, cache_dir=tmp)
        onnx = sorted(Path(tmp).rglob("*.onnx"))
        if not onnx:
            raise RuntimeError(f"no ONNX artifacts downloaded for {model}")
        source = onnx[0].parent
        target.mkdir(parents=True, exist_ok=True)
        for item in source.iterdir():
            if item.name.startswith("."):
                continue
            destination = target / item.name
            if item.is_dir():
                shutil.copytree(item.resolve(), destination, dirs_exist_ok=True)
            else:
                shutil.copyfile(item.resolve(), destination)
    return target


class ArkEmbedding:
    """Volcengine Ark multimodal embedding (e.g. doubao-embedding-vision), text only.

    One request per text (the endpoint fuses every input into one vector), rate
    limited, with a persistent SQLite cache keyed by (model, dimensions, text) so
    reruns and other processes sharing the cache pay nothing (see ark_embed.py).
    """

    def __init__(
        self,
        *,
        model: str,
        dimension: int,
        cache_path: str | Path,
        base_url: str | None = None,
        requests_per_second: float = 4.0,
        concurrency: int = 16,
    ) -> None:
        from mr_mem_aml.ark_embed import ArkEmbedder

        self._client = ArkEmbedder(
            model,
            dimensions=dimension,
            concurrency=concurrency,
            cache_path=Path(cache_path),
            base_url=base_url,
            requests_per_second=requests_per_second,
        )
        self._identity = EmbeddingIdentity("ark", model, dimension, "v1")

    @property
    def identity(self) -> EmbeddingIdentity:
        return self._identity

    def embed(self, text: str) -> tuple[float, ...]:
        return tuple(float(v) for v in self._client.embed([text])[0])

    def embed_many(self, texts: list[str]):
        """Batch (parallel) embedding; numpy array, rows in input order."""
        return self._client.embed(texts)
