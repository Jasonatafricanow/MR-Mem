"""HTTP embedding providers for the dense retrieval arm (no new dependencies).

They implement ``mr_mem.memory.embedding.EmbeddingProvider``; vectors stay a
rebuildable derived index under MR-Mem's existing Qdrant projection.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request

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
        except Exception as exc:  # noqa: BLE001 - transport retry boundary
            last = f"{type(exc).__name__}: {exc}"
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
        self, *, base_url: str, model: str, dimension: int, api_key_env: str,
        timeout: float = 60, attempts: int = 4,
    ) -> None:
        self._identity = EmbeddingIdentity("openai-compatible-http", model, dimension, "api")
        self.url = base_url.rstrip("/") + "/embeddings"
        self.model, self.api_key_env = model, api_key_env
        self.timeout, self.attempts = timeout, attempts

    @property
    def identity(self) -> EmbeddingIdentity:
        return self._identity

    def embed(self, text: str) -> tuple[float, ...]:
        data = _post(
            self.url, {"model": self.model, "input": text},
            {"Authorization": f"Bearer {_key(self.api_key_env)}"},
            self.timeout, self.attempts,
        )
        return tuple(float(v) for v in data["data"][0]["embedding"])
