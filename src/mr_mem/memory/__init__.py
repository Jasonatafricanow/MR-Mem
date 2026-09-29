"""Canonical Memory core, admission governance, and local product projections."""

from mr_mem.memory.admission import MemoryAdmissionService
from mr_mem.memory.contracts import (
    CommittedMemory,
    MemoryCandidate,
    MemoryLifecycle,
    MemoryProvenance,
)
from mr_mem.memory.core import MemoryCore, MemoryCoreSelectionError
from mr_mem.memory.extraction import (
    DeterministicExtractor,
    MemoryExtractor,
    memory_identity,
)
from mr_mem.memory.product import (
    MemoryAttention,
    MemoryProductStore,
    MemoryThread,
    ThreadStatus,
)
from mr_mem.memory.source import (
    AdmissionDisposition,
    Clock,
    DurableFactReader,
    MemoryAdmissionResult,
    SourceEvidence,
    SourceObservation,
)
from mr_mem.memory.store import CanonicalMemoryStore, MemoryConflict

__all__ = [
    "AdmissionDisposition",
    "CanonicalMemoryStore",
    "Clock",
    "CommittedMemory",
    "DeterministicExtractor",
    "DurableFactReader",
    "MemoryAdmissionResult",
    "MemoryAdmissionService",
    "MemoryAttention",
    "MemoryCandidate",
    "MemoryConflict",
    "MemoryCore",
    "MemoryCoreSelectionError",
    "MemoryExtractor",
    "MemoryLifecycle",
    "MemoryProductStore",
    "MemoryProvenance",
    "MemoryThread",
    "SourceEvidence",
    "SourceObservation",
    "ThreadStatus",
    "memory_identity",
]
