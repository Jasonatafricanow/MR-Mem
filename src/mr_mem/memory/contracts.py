"""Provider-independent canonical values; construction grants no write authority."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from mr_mem.contracts.common import (
    SyncFields,
    freeze_refs,
    require_aware_utc,
    require_non_empty,
    validate_sync_fields,
)
from mr_mem.contracts.scope import Scope


class MemoryLifecycle(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"


@dataclass(frozen=True, slots=True)
class MemoryProvenance:
    evidence_refs: tuple[str, ...]
    observation_id: str
    extractor_version: str
    interaction_id: str | None = None

    def __post_init__(self) -> None:
        refs = freeze_refs(self.evidence_refs, "evidence_refs")
        if not refs or len(set(refs)) != len(refs):
            raise ValueError("evidence_refs must be nonempty and unique")
        object.__setattr__(self, "evidence_refs", tuple(sorted(refs)))
        require_non_empty(self.observation_id, "observation_id")
        require_non_empty(self.extractor_version, "extractor_version")
        if self.interaction_id is not None:
            require_non_empty(self.interaction_id, "interaction_id")


@dataclass(frozen=True, slots=True)
class MemoryCandidate:
    candidate_id: str
    scope: Scope
    content: str
    provenance: MemoryProvenance

    def __post_init__(self) -> None:
        require_non_empty(self.candidate_id, "candidate_id")
        _validate_content(self.scope, self.content, self.provenance)


def _validate_content(scope: Scope, content: str, provenance: MemoryProvenance) -> None:
    if not isinstance(scope, Scope) or not isinstance(provenance, MemoryProvenance):
        raise ValueError("structured Scope and MemoryProvenance required")
    require_non_empty(content, "content")
    if len(content.encode("utf-8")) > 16384:
        raise ValueError("Memory content exceeds 16384 bytes")


@dataclass(frozen=True, slots=True)
class CommittedMemory:
    memory_id: str
    scope: Scope
    content: str
    provenance: MemoryProvenance
    origin_runtime_id: str
    committed_at: datetime
    sync: SyncFields
    lifecycle: MemoryLifecycle = MemoryLifecycle.ACTIVE
    supersedes_memory_id: str | None = None

    def __post_init__(self) -> None:
        require_non_empty(self.memory_id, "memory_id")
        require_non_empty(self.origin_runtime_id, "origin_runtime_id")
        _validate_content(self.scope, self.content, self.provenance)
        require_aware_utc(self.committed_at, "committed_at")
        if not isinstance(self.lifecycle, MemoryLifecycle):
            raise ValueError("lifecycle must be MemoryLifecycle")
        if self.supersedes_memory_id is not None:
            require_non_empty(self.supersedes_memory_id, "supersedes_memory_id")
            if self.supersedes_memory_id == self.memory_id:
                raise ValueError("Memory cannot supersede itself")
        validate_sync_fields(
            self.sync,
            scope=self.scope,
            origin_runtime_id=self.origin_runtime_id,
            object_id=self.memory_id,
        )

    def sync_fields(self) -> SyncFields:
        return self.sync
