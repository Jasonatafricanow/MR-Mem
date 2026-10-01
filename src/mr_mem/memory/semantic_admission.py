"""Admission of host-accepted semantics without an intermediate raw/fact mirror."""

import hashlib
import json

from mr_mem.contracts import SyncFields
from mr_mem.contracts.common import require_non_empty
from mr_mem.memory.contracts import (
    CommittedMemory,
    MemoryLifecycle,
    MemoryProvenance,
    SemanticMemoryCandidate,
)
from mr_mem.memory.source import Clock, SourceRefReader
from mr_mem.memory.store import CanonicalMemoryStore, MemoryConflict, scope_json


class SemanticAdmissionService:
    """Validate current native provenance, then freeze semantic identity atomically."""

    def __init__(
        self, *, store: CanonicalMemoryStore, sources: SourceRefReader,
        clock: Clock, origin_runtime_id: str,
    ) -> None:
        require_non_empty(origin_runtime_id, "origin_runtime_id")
        self._store, self._sources, self._clock = store, sources, clock
        self._origin = origin_runtime_id

    def admit(self, candidate: SemanticMemoryCandidate) -> CommittedMemory:
        if not isinstance(candidate, SemanticMemoryCandidate):
            raise ValueError("accepted SemanticMemoryCandidate required")
        for ref in candidate.source_refs:
            if self._sources.current_ref(candidate.scope, ref) != ref:
                raise ValueError("semantic source is unavailable, out of scope, or stale")
        identity = [
            "mr-semantic-v1", self._origin, scope_json(candidate.scope),
            candidate.semantic_id, candidate.compiler_version,
            [ref.version_key for ref in candidate.source_refs],
        ]
        memory_id = "semantic-" + hashlib.sha256(json.dumps(identity).encode()).hexdigest()
        existing = self._store.get(memory_id)
        memory = CommittedMemory(
            memory_id=memory_id,
            scope=candidate.scope,
            content=candidate.content,
            provenance=MemoryProvenance(
                extractor_version=candidate.compiler_version, source_refs=candidate.source_refs
            ),
            origin_runtime_id=self._origin,
            committed_at=existing.known_at if existing is not None else self._clock.now(),
            sync=SyncFields(candidate.scope, self._origin, memory_id, 1, memory_id),
            attributes=candidate.attributes,
            supports_memory_ids=candidate.supports_memory_ids,
            contradicts_memory_ids=candidate.contradicts_memory_ids,
            supersedes_memory_id=candidate.supersedes_memory_id,
        )
        if existing is not None:
            if existing != memory or existing.lifecycle is not MemoryLifecycle.ACTIVE:
                raise MemoryConflict("immutable semantic identity conflict")
            return existing
        self._store._commit_semantic(memory)
        return memory
