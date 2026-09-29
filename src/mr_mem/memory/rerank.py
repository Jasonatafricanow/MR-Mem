"""Optional provider-neutral semantic reranking seam for Memory retrieval."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class MemoryRerankCandidate:
    memory_id: str
    content: str

    def __post_init__(self) -> None:
        if not isinstance(self.memory_id, str) or not self.memory_id.strip():
            raise ValueError("memory_id must be nonempty")
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("content must be nonempty")


@runtime_checkable
class MemoryReranker(Protocol):
    """Optional semantic decision seam; MR-Mem does not own the model."""

    @property
    def available(self) -> bool: ...

    def rerank(
        self,
        *,
        query: str,
        candidates: tuple[MemoryRerankCandidate, ...],
    ) -> tuple[str, ...]: ...
