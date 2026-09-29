"""Bounded Memory extraction from provider-neutral admitted source records."""

import hashlib
import json
from collections.abc import Mapping
from typing import Protocol

from mr_mem.memory.contracts import MemoryCandidate, MemoryProvenance
from mr_mem.memory.source import SourceEvidence, SourceObservation
from mr_mem.memory.store import scope_json


class MemoryExtractor(Protocol):
    def extract(
        self,
        evidence: SourceEvidence,
        observation: SourceObservation,
    ) -> tuple[MemoryCandidate, ...]: ...


class DeterministicExtractor:
    """Minimal text extractor preserving the original MR admission baseline."""

    def extract(
        self,
        evidence: SourceEvidence,
        observation: SourceObservation,
    ) -> tuple[MemoryCandidate, ...]:
        text = evidence.payload.get("text") if isinstance(evidence.payload, Mapping) else None
        if not isinstance(text, str) or not text.strip():
            return ()
        return (
            MemoryCandidate(
                "text-v1:0",
                evidence.scope,
                text,
                MemoryProvenance((evidence.id,), observation.id, "text-v1"),
            ),
        )


def memory_identity(candidate: MemoryCandidate, origin_runtime_id: str) -> str:
    """Return the stable historical Memory identity.

    The ``mr-memory-v1`` prefix is intentionally retained so extracting the
    subsystem into MR-Mem does not rewrite existing canonical identities.
    """
    identity = [
        "mr-memory-v1",
        origin_runtime_id,
        json.loads(scope_json(candidate.scope)),
        candidate.provenance.observation_id,
        candidate.provenance.evidence_refs,
        candidate.provenance.extractor_version,
        candidate.candidate_id,
    ]
    return "memory-" + hashlib.sha256(
        json.dumps(identity, ensure_ascii=False).encode()
    ).hexdigest()
