from __future__ import annotations

from datetime import UTC, datetime

from mr_mem import (
    CommittedMemory,
    MemoryProvenance,
    Scope,
    ScopeDomain,
    SyncFields,
)


def memory(
    memory_id: str = "memory-1",
    *,
    content: str = "The user prefers quiet coastal towns.",
    user_id: str = "user-1",
    interaction_id: str = "interaction-1",
) -> CommittedMemory:
    scope = Scope(ScopeDomain.USER, user_id=user_id)
    return CommittedMemory(
        memory_id=memory_id,
        scope=scope,
        content=content,
        provenance=MemoryProvenance(
            evidence_refs=(f"evidence-{memory_id}",),
            observation_id=f"observation-{memory_id}",
            extractor_version="test-v1",
            interaction_id=interaction_id,
        ),
        origin_runtime_id="test-runtime",
        committed_at=datetime(2026, 9, 29, tzinfo=UTC),
        sync=SyncFields(
            scope=scope,
            origin_runtime_id="test-runtime",
            object_id=memory_id,
            version=1,
            idempotency_key=f"idem-{memory_id}",
        ),
    )
