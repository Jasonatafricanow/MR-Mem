"""Admission of host-accepted semantics without an intermediate raw/fact mirror."""

import hashlib
import json
from dataclasses import asdict

from mr_mem.contracts import SyncFields
from mr_mem.contracts.common import require_non_empty
from mr_mem.memory.contracts import (
    CommittedMemory,
    MemoryLifecycle,
    MemoryProvenance,
    SemanticMemoryCandidate,
)
from mr_mem.memory.semantic_closure import compile_semantic_delta
from mr_mem.memory.semantic_contracts import (
    COMPILER_VERSION,
    SCHEMA_VERSION,
    BoundaryPolicy,
    DependencyTargetKind,
    SemanticDeltaError,
    SemanticDeltaErrorCode,
)
from mr_mem.memory.semantic_store import (
    SemanticDeltaReceipt,
    SemanticSourceBinding,
    binding_payload,
)
from mr_mem.memory.semantic_validator import validate_semantic_delta
from mr_mem.memory.source import Clock, SourceRefReader
from mr_mem.memory.store import CanonicalMemoryStore, MemoryConflict, scope_json


class SemanticAdmissionService:
    """Validate current native provenance, then freeze semantic identity atomically."""

    def __init__(
        self,
        *,
        store: CanonicalMemoryStore,
        sources: SourceRefReader,
        clock: Clock,
        origin_runtime_id: str,
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
        memory = self._prepare_memory(candidate, self._clock.now())
        existing = self._store.get(memory.memory_id)
        if existing is not None:
            if existing != memory or existing.lifecycle is not MemoryLifecycle.ACTIVE:
                raise MemoryConflict("immutable semantic identity conflict")
            return existing
        self._store._commit_semantic(memory)
        return memory

    def _prepare_memory(self, candidate: SemanticMemoryCandidate, at) -> CommittedMemory:
        # The approved native identity algorithm is reused unchanged by Delta V1.
        identity = [
            "mr-semantic-v1",
            self._origin,
            scope_json(candidate.scope),
            candidate.semantic_id,
            candidate.compiler_version,
            [ref.version_key for ref in candidate.source_refs],
        ]
        memory_id = "semantic-" + hashlib.sha256(json.dumps(identity).encode()).hexdigest()
        existing = self._store.get(memory_id)
        return CommittedMemory(
            memory_id=memory_id,
            scope=candidate.scope,
            content=candidate.content,
            provenance=MemoryProvenance(
                extractor_version=candidate.compiler_version, source_refs=candidate.source_refs
            ),
            origin_runtime_id=self._origin,
            committed_at=existing.known_at if existing is not None else at,
            sync=SyncFields(candidate.scope, self._origin, memory_id, 1, memory_id),
            attributes=candidate.attributes,
            supports_memory_ids=candidate.supports_memory_ids,
            contradicts_memory_ids=candidate.contradicts_memory_ids,
            supersedes_memory_id=candidate.supersedes_memory_id,
        )

    def _validate_binding(self, binding: SemanticSourceBinding) -> None:
        if not isinstance(binding, SemanticSourceBinding):
            raise SemanticDeltaError(
                SemanticDeltaErrorCode.INVALID_SCHEMA, "system SemanticSourceBinding required"
            )
        if self._sources.current_ref(binding.scope, binding.source_ref) != binding.source_ref:
            raise SemanticDeltaError(
                SemanticDeltaErrorCode.SCOPE_MISMATCH,
                "native source is unavailable, out of scope, or stale",
            )
        associate = getattr(self._sources, "current_user_source", None)
        if (
            associate is None
            or associate(binding.scope, binding.interaction_id) != binding.source_ref
        ):
            raise SemanticDeltaError(
                SemanticDeltaErrorCode.SCOPE_MISMATCH,
                "exact durable user-source/interaction association required",
            )

    def _delta_job_key(self, binding: SemanticSourceBinding) -> str:
        return json.dumps(
            [
                self._origin,
                scope_json(binding.scope),
                binding.interaction_id,
                binding.source_ref.version_key,
                SCHEMA_VERSION,
            ]
        )

    def admit_semantic_delta(
        self,
        payload,
        *,
        binding: SemanticSourceBinding,
        activated_memory_ids: tuple[str, ...],
    ) -> SemanticDeltaReceipt:
        self._validate_binding(binding)
        delta = validate_semantic_delta(payload, activated_memory_ids=activated_memory_ids)
        key = self._delta_job_key(binding)
        prior = self._store._semantic_compilation(key)
        if prior is not None:
            accepted = prior[0]
            # Once accepted, retries cannot replace interpretation under the same source key.
            if accepted["delta"] != json.loads(json.dumps(asdict(delta))) or accepted[
                "activated_memory_ids"
            ] != list(activated_memory_ids):
                raise MemoryConflict("conflicting accepted semantic compilation")
            self._complete_delta_job(key, binding)
            return self._delta_receipt(key)
        for mid in activated_memory_ids:
            target = self._store.get(mid)
            if target is None:
                raise SemanticDeltaError(
                    SemanticDeltaErrorCode.UNKNOWN_MEMORY, "missing activated ID"
                )
            if target.scope != binding.scope:
                raise SemanticDeltaError(SemanticDeltaErrorCode.SCOPE_MISMATCH, "activated scope")
            if target.lifecycle is not MemoryLifecycle.ACTIVE:
                raise SemanticDeltaError(
                    SemanticDeltaErrorCode.INVALID_DEPENDENCY, "activated memory is not active"
                )
        blocks = compile_semantic_delta(delta, activated_memory_ids=activated_memory_ids)
        self._store._register_job(key, self._clock.now())
        at = self._store._job(key)[0]
        memories = tuple(
            self._prepare_memory(
                SemanticMemoryCandidate(
                    semantic_id=json.dumps(
                        [SCHEMA_VERSION, binding.interaction_id, block.member_point_ids]
                    ),
                    scope=binding.scope,
                    content=block.analysis_text,
                    source_refs=(binding.source_ref,),
                    compiler_version=COMPILER_VERSION,
                ),
                at,
            )
            for block in blocks
        )
        point_memory = {
            pid: memory.memory_id
            for block, memory in zip(blocks, memories, strict=True)
            for pid in block.member_point_ids
        }
        metadata, relations = [], set()
        for block, memory in zip(blocks, memories, strict=True):
            context_ids = set(block.context_memory_ids)
            for edge in block.dependencies:
                target = (
                    edge.target_id
                    if edge.target_kind is DependencyTargetKind.MEMORY
                    else point_memory[edge.target_id]
                )
                if edge.boundary_policy is BoundaryPolicy.CONTEXT and target != memory.memory_id:
                    context_ids.add(target)
                    relations.add(
                        (
                            memory.memory_id,
                            target,
                            edge.relation,
                            edge.boundary_policy.value,
                            edge.lifecycle_effect.value,
                            binding.interaction_id,
                        )
                    )
            metadata.append(
                dict(
                    memory_id=memory.memory_id,
                    schema_version=SCHEMA_VERSION,
                    compiler_version=COMPILER_VERSION,
                    member_point_ids=list(block.member_point_ids),
                    context_memory_ids=sorted(context_ids),
                    source_interaction_id=binding.interaction_id,
                    source_refs=[binding_payload(binding)["source_ref"]],
                )
            )
        accepted = dict(
            binding=binding_payload(binding),
            delta=asdict(delta),
            activated_memory_ids=list(activated_memory_ids),
            metadata=metadata,
            relations=sorted(relations),
        )
        # JSON roundtrip provides the same representation as durable audit decoding.
        accepted = json.loads(json.dumps(accepted))
        self._store._freeze_job(key, memories, semantic_compilation=accepted)
        self._complete_delta_job(key, binding)
        return self._delta_receipt(key)

    def _complete_delta_job(self, key: str, binding: SemanticSourceBinding) -> None:
        self._validate_binding(binding)
        try:
            self._store._complete_job(key)
        except Exception as exc:
            raise SemanticDeltaError(
                SemanticDeltaErrorCode.COMMIT_FAILED, "canonical semantic transaction rolled back"
            ) from exc

    def _delta_receipt(self, key: str) -> SemanticDeltaReceipt:
        accepted, status = self._store._semantic_compilation(key)
        if status != "complete":
            raise MemoryConflict("semantic compilation is not complete")
        from mr_mem.memory.semantic_store import read_binding

        binding = read_binding(accepted["binding"])
        return SemanticDeltaReceipt(
            binding.interaction_id,
            binding.source_ref,
            tuple(meta["memory_id"] for meta in accepted["metadata"]),
            "deferred" if not accepted["metadata"] and accepted["delta"]["points"] else "committed",
        )

    def semantic_delta_receipt(self, binding: SemanticSourceBinding) -> SemanticDeltaReceipt | None:
        self._validate_binding(binding)
        key = self._delta_job_key(binding)
        prior = self._store._semantic_compilation(key)
        return self._delta_receipt(key) if prior is not None and prior[1] == "complete" else None

    def resume_semantic_delta(self, binding: SemanticSourceBinding) -> SemanticDeltaReceipt | None:
        """Finish the original accepted job, never parse/propose/recompile a new delta."""
        self._validate_binding(binding)
        key = self._delta_job_key(binding)
        if self._store._semantic_compilation(key) is None:
            return None
        self._complete_delta_job(key, binding)
        return self._delta_receipt(key)

    def recover_semantic_deltas(self, *, limit: int = 32) -> tuple[SemanticDeltaReceipt, ...]:
        from mr_mem.memory.semantic_store import read_binding

        receipts = []
        for key, accepted in self._store._pending_semantic_compilations(limit):
            binding = read_binding(accepted["binding"])
            if self._delta_job_key(binding) == key:
                receipts.append(self.resume_semantic_delta(binding))
        return tuple(receipts)
