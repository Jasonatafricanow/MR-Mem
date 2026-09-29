"""Derived embedding identities; never fields of canonical Memory."""

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class EmbeddingIdentity:
    provider: str
    model_id: str
    dimension: int
    revision: str

    def __post_init__(self) -> None:
        for value in (self.provider, self.model_id, self.revision):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("embedding provider/model/revision must be nonempty")
        if type(self.dimension) is not int or self.dimension <= 0:
            raise ValueError("embedding dimension must be a positive integer")


class EmbeddingProvider(Protocol):
    @property
    def identity(self) -> EmbeddingIdentity: ...

    def embed(self, text: str) -> tuple[float, ...]: ...


def projection_target(identity: EmbeddingIdentity) -> str:
    payload = json.dumps(["qdrant-cosine-v1", asdict(identity)], sort_keys=True)
    return "qdrant-" + hashlib.sha256(payload.encode()).hexdigest()


def validate_vector(vector: Sequence[float], identity: EmbeddingIdentity) -> tuple[float, ...]:
    if len(vector) != identity.dimension or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
        for v in vector
    ):
        raise ValueError("invalid embedding vector dimension or coordinates")
    values = tuple(float(v) for v in vector)
    norm = math.hypot(*values)
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("invalid embedding vector norm")
    return values
