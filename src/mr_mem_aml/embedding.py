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
        self, *, model: str, dimension: int, api_key_env: str = "GEMINI_API_KEY",
        timeout: float = 60, attempts: int = 4,
    ) -> None:
        self._identity = EmbeddingIdentity("gemini-http", model, dimension, "api")
        self.model, self.api_key_env = model, api_key_env
        self.timeout, self.attempts = timeout, attempts

    @property
    def identity(self) -> EmbeddingIdentity:
        return self._identity

    def embed(self, text: str) -> tuple[float, ...]:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:embedContent"
        )
        data = _post(
            url,
            {
                "model": f"models/{self.model}",
                "content": {"parts": [{"text": text}]},
                "outputDimensionality": self._identity.dimension,
            },
            {"x-goog-api-key": _key(self.api_key_env)},
            self.timeout, self.attempts,
        )
        return tuple(float(v) for v in data["embedding"]["values"])


class OpenAICompatibleEmbedding:
    def __init__(
        self, *, base_url: str, model: str, dimension: int, api_key_env: str | None = None,
        send_dimensions: bool = False, headers: dict[str, str] | None = None,
        timeout: float = 60, attempts: int = 4,
    ) -> None:
        self._identity = EmbeddingIdentity("openai-compatible-http", model, dimension, "api")
        self.url = base_url.rstrip("/") + "/embeddings"
        self.model, self.api_key_env = model, api_key_env
        self.send_dimensions, self.headers = send_dimensions, dict(headers or {})
        self.timeout, self.attempts = timeout, attempts

    @property
    def identity(self) -> EmbeddingIdentity:
        return self._identity

    def embed(self, text: str) -> tuple[float, ...]:
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

    def __init__(self, *, model: str, dimension: int, cache_path: str | Path,
                 base_url: str | None = None, requests_per_second: float = 4.0,
                 concurrency: int = 16) -> None:
        from mr_mem_aml.ark_embed import ArkEmbedder

        self._client = ArkEmbedder(
            model, dimensions=dimension, concurrency=concurrency, cache_path=Path(cache_path),
            base_url=base_url, requests_per_second=requests_per_second,
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
