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
from mr_mem.memory.source import SourceRef


class MemoryLifecycle(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"
    INVALIDATED = "invalidated"


@dataclass(frozen=True, slots=True)
class MemoryProvenance:
    evidence_refs: tuple[str, ...] = ()
    observation_id: str | None = None
    extractor_version: str = ""
    interaction_id: str | None = None
    source_refs: tuple[SourceRef, ...] = ()

    def __post_init__(self) -> None:
        refs = freeze_refs(self.evidence_refs, "evidence_refs")
        if len(set(refs)) != len(refs):
            raise ValueError("evidence_refs must be unique")
        object.__setattr__(self, "evidence_refs", tuple(sorted(refs)))
        sources = self.source_refs
        if not isinstance(sources, tuple) or any(not isinstance(r, SourceRef) for r in sources):
            raise ValueError("source_refs must be a tuple of SourceRef")
        if len({r.source_key for r in sources}) != len(sources):
            raise ValueError("source_refs must identify unique source events")
        if bool(refs) == bool(sources):
            raise ValueError("use native source_refs or legacy evidence_refs exclusively")
        object.__setattr__(self, "source_refs", tuple(sorted(sources, key=lambda r: r.source_key)))
        if refs:
            require_non_empty(self.observation_id, "observation_id")
        elif self.observation_id is not None or self.interaction_id is not None:
            raise ValueError("native provenance must not fabricate factual pair identities")
        require_non_empty(self.extractor_version, "extractor_version")
        if self.interaction_id is not None:
            require_non_empty(self.interaction_id, "interaction_id")

    @property
    def support_event_ids(self) -> tuple[str, ...]:
        if self.source_refs:
            return tuple(ref.source_key for ref in self.source_refs)
        return () if self.interaction_id is None else (self.interaction_id,)


@dataclass(frozen=True, slots=True)
class SemanticMemoryCandidate:
    """Already accepted host semantics; construction alone grants no admission."""

    semantic_id: str
    scope: Scope
    content: str
    source_refs: tuple[SourceRef, ...]
    compiler_version: str
    attributes: tuple[tuple[str, str], ...] = ()
    supports_memory_ids: tuple[str, ...] = ()
    contradicts_memory_ids: tuple[str, ...] = ()
    supersedes_memory_id: str | None = None

    def __post_init__(self) -> None:
        require_non_empty(self.semantic_id, "semantic_id")
        provenance = MemoryProvenance(
            extractor_version=self.compiler_version, source_refs=self.source_refs
        )
        _validate_content(self.scope, self.content, provenance)
        object.__setattr__(self, "source_refs", provenance.source_refs)
        _validate_semantic_fields(self)


def _validate_semantic_fields(value: "SemanticMemoryCandidate | CommittedMemory") -> None:
    attrs = value.attributes
    if not isinstance(attrs, tuple) or len(attrs) > 64:
        raise ValueError("attributes must be a bounded tuple")
    for pair in attrs:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError("attributes must contain key/value pairs")
        require_non_empty(pair[0], "attribute key")
        require_non_empty(pair[1], "attribute value")
    if sum(len(key.encode()) + len(text.encode()) for key, text in attrs) > 16384:
        raise ValueError("semantic attributes exceed 16384 bytes")
    if len({key for key, _ in attrs}) != len(attrs):
        raise ValueError("duplicate semantic attributes")
    for name in ("supports_memory_ids", "contradicts_memory_ids"):
        refs = freeze_refs(getattr(value, name), name)
        if len(refs) > 64 or len(set(refs)) != len(refs):
            raise ValueError("semantic relation refs must be bounded and unique")
        object.__setattr__(value, name, refs)
    if value.supersedes_memory_id is not None:
        require_non_empty(value.supersedes_memory_id, "supersedes_memory_id")


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
    attributes: tuple[tuple[str, str], ...] = ()
    supports_memory_ids: tuple[str, ...] = ()
    contradicts_memory_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_non_empty(self.memory_id, "memory_id")
        require_non_empty(self.origin_runtime_id, "origin_runtime_id")
        _validate_content(self.scope, self.content, self.provenance)
        _validate_semantic_fields(self)
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

    @property
    def known_at(self) -> datetime:
        return self.committed_at

    @property
    def occurred_at(self) -> datetime | None:
        refs = self.provenance.source_refs
        return min(ref.occurred_at for ref in refs) if refs else None

    @property
    def evidence_refs(self) -> tuple[str, ...]:
        """Thread compatibility surface over exact canonical semantic identity."""
        return (self.memory_id,)
