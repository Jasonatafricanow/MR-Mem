"""Memory-owned admission governance over durably admitted source records."""

import json
from dataclasses import replace

from mr_mem.contracts import SyncFields
from mr_mem.contracts.common import require_non_empty
from mr_mem.memory.contracts import CommittedMemory, MemoryCandidate
from mr_mem.memory.extraction import DeterministicExtractor, MemoryExtractor, memory_identity
from mr_mem.memory.source import (
    AdmissionDisposition,
    Clock,
    DurableFactReader,
    MemoryAdmissionResult,
    SourceEvidence,
)
from mr_mem.memory.store import CanonicalMemoryStore, scope_json


class MemoryAdmissionService:
    """Legacy factual-pair compatibility adapter into canonical Memory.

    The service owns Memory eligibility and provenance constraints, but does
    not own the upstream factual store. Native hosts use SemanticAdmissionService.
    """

    def __init__(
        self,
        *,
        store: CanonicalMemoryStore,
        facts: DurableFactReader,
        clock: Clock,
        origin_runtime_id: str,
        enabled: bool = False,
        extractor: MemoryExtractor | None = None,
    ) -> None:
        require_non_empty(origin_runtime_id, "origin_runtime_id")
        if type(enabled) is not bool:
            raise ValueError("enabled must be bool")
        self._store, self._facts, self._clock = store, facts, clock
        self._origin, self._enabled = origin_runtime_id, enabled
        self._extractor = extractor if extractor is not None else DeterministicExtractor()

    def after_admission(
        self,
        evidence: SourceEvidence,
        result: MemoryAdmissionResult,
    ) -> None:
        """Process one upstream durable-admission outcome.

        NEW may register Memory eligibility. REPAIRED and REPLAY never create
        historical backfill eligibility; they may only finish an already
        registered job created by the original NEW event.
        """
        if not self._enabled:
            return
        observation = result.observation
        disposition = AdmissionDisposition(result.disposition)
        pair = self._facts.find_evidence(evidence.scope, evidence.id)
        admitted = self._facts.find_observation(
            evidence.scope, f"observation-{evidence.id}"
        )
        if (
            pair is None
            or pair[0] != evidence
            or admitted != observation
            or admitted is None
            or observation.scope != evidence.scope
            or observation.evidence_refs != (evidence.id,)
            or pair[1] != observation.interaction_id
        ):
            raise ValueError(
                "Memory requires the exact durable admitted Evidence/Observation pair"
            )
        key = json.dumps(
            [self._origin, scope_json(evidence.scope), evidence.id, observation.id]
        )
        if disposition is AdmissionDisposition.NEW:
            self._store._register_job(key, self._clock.now())
        job = self._store._job(key)
        if job is None or job[2]:
            return
        if job[1] is None:
            candidates = self._extractor.extract(evidence, observation)
            if not isinstance(candidates, tuple) or len(candidates) > 32:
                raise ValueError("extractor must return a tuple of at most 32 candidates")
            memories: list[CommittedMemory] = []
            seen: set[str] = set()
            for candidate in candidates:
                if (
                    not isinstance(candidate, MemoryCandidate)
                    or candidate.scope != evidence.scope
                    or candidate.provenance.observation_id != observation.id
                    or not set(candidate.provenance.evidence_refs).issubset({evidence.id})
                ):
                    raise ValueError("candidate exceeds its admitted source authority")
                memory_id = memory_identity(candidate, self._origin)
                if memory_id in seen:
                    raise ValueError("duplicate candidate identity")
                seen.add(memory_id)
                memories.append(
                    CommittedMemory(
                        memory_id=memory_id,
                        scope=evidence.scope,
                        content=candidate.content,
                        provenance=replace(
                            candidate.provenance,
                            interaction_id=observation.interaction_id,
                        ),
                        origin_runtime_id=self._origin,
                        committed_at=job[0],
                        sync=SyncFields(
                            evidence.scope, self._origin, memory_id, 1, memory_id
                        ),
                    )
                )
            self._store._freeze_job(key, tuple(memories))
        self._store._complete_job(key)
