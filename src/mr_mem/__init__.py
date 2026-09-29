"""MR-Mem: standalone canonical Memory runtime extracted from Mind Runtime."""

from mr_mem.contracts import Scope, ScopeDomain, SyncFields
from mr_mem.memory import (
    CanonicalMemoryStore,
    CommittedMemory,
    MemoryAttention,
    MemoryCandidate,
    MemoryConflict,
    MemoryCore,
    MemoryCoreSelectionError,
    MemoryLifecycle,
    MemoryProductStore,
    MemoryProvenance,
    MemoryThread,
    ThreadStatus,
)

__version__ = "0.1.0"

__all__ = [
    "CanonicalMemoryStore",
    "CommittedMemory",
    "MemoryAttention",
    "MemoryCandidate",
    "MemoryConflict",
    "MemoryCore",
    "MemoryCoreSelectionError",
    "MemoryLifecycle",
    "MemoryProductStore",
    "MemoryProvenance",
    "MemoryThread",
    "Scope",
    "ScopeDomain",
    "SyncFields",
    "ThreadStatus",
]
